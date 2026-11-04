# -*- coding: utf-8 -*-
"""Definición común de modelos (referencias, mixtos y aprendizaje automático).

Todos los modelos comparten la misma interfaz: .fit(df) y .predict(df), donde df es la base
analítica con la columna 'nota'. Así 05, 06, 10 y 12 usan exactamente las mismas definiciones.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from common import GroupMeanBaseline, fit_mixed, predict_mixed

CAT_HI = ["universidad", "programa", "univ_prog", "departamento"]   # alta cardinalidad
def _xgboost_ok() -> bool:
    """En Mac, XGBoost necesita OpenMP (brew install libomp). Si no carga, se omite con aviso."""
    try:
        import xgboost  # noqa: F401
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"AVISO: XGBoost no disponible ({type(exc).__name__}); se omite. En Mac: brew install libomp")
        return False


ML_MODELS = ["Random Forest"] + (["XGBoost"] if _xgboost_ok() else []) + ["CatBoost", "SVR"]
BASELINES = ["Media global", "Media universidad", "Media universidad-programa", "Ridge",
             "Mixto institucional", "Mixto + covariables"]
TUNED = ["Ridge"] + ML_MODELS
MIXED_CATS = ["anio", "gestion", "licenciado", "constitucion", "area", "sexo", "macroregion"]

SPACES = {
    "Ridge": {"m__alpha": [0.1, 0.3, 1, 3, 10, 30, 100]},
    "Random Forest": {"m__n_estimators": [200, 400, 600], "m__max_depth": [None, 10, 20, 30],
                      "m__min_samples_leaf": [1, 5, 10, 20], "m__max_features": [0.3, 0.5, 0.8, 1.0]},
    "XGBoost": {"m__n_estimators": [300, 600, 1000], "m__learning_rate": [0.03, 0.05, 0.1],
                "m__max_depth": [3, 4, 6, 8], "m__subsample": [0.7, 0.85, 1.0],
                "m__colsample_bytree": [0.6, 0.8, 1.0], "m__min_child_weight": [1, 5, 10],
                "m__reg_lambda": [0.5, 1, 5]},
    "SVR": {"m__C": [0.3, 1, 3, 10, 30], "m__gamma": ["scale", 0.03, 0.1, 0.3],
            "m__epsilon": [0.05, 0.1, 0.2, 0.5]},
    "CatBoost": {"depth": [4, 6, 8], "learning_rate": [0.03, 0.05, 0.1],
                 "l2_leaf_reg": [1, 3, 5, 10], "iterations": [500, 1000, 1500]},
}


def mixed_formulas(use_year: bool, df: pd.DataFrame) -> dict:
    base = "C(anio)" if use_year and df["anio"].nunique() > 1 else "1"
    # constitución (pública / societaria / asociativa) contiene a gestión: se usa una u otra, no ambas
    usa_const = "constitucion" in df and df["constitucion"].nunique() > 1
    cov = base + (" + C(constitucion)" if usa_const else " + C(gestion)")
    for c in ("licenciado", "area"):
        if c in df and df[c].nunique() > 1:
            cov += f" + C({c})"
    cov += " + edad + I((edad - 25) ** 2) + C(sexo) + creditos + C(macroregion)"
    if "nota_minima" in df and df["nota_minima"].notna().all():
        cov += " + nota_minima"
    return {"Mixto institucional": base, "Mixto + covariables": cov}


def _prep(df: pd.DataFrame, cat: list, num: list) -> pd.DataFrame:
    X = df[cat + num].copy()
    for c in cat:
        X[c] = X[c].astype(str)
    return X


def sk_pipeline(name: str, cat: list, num: list, seed: int, n_jobs: int) -> Pipeline:
    hi = [c for c in cat if c in CAT_HI]
    lo = [c for c in cat if c not in CAT_HI]
    if name == "Ridge":  # modelo lineal aditivo con efectos fijos (one-hot)
        pre = ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore"), cat),
                                 ("num", StandardScaler(), num)])
        return Pipeline([("pre", pre), ("m", Ridge())])
    pre = ColumnTransformer([("te", TargetEncoder(target_type="continuous", random_state=seed), hi),
                             ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False), lo),
                             ("num", "passthrough", num)])
    if name == "Random Forest":
        from sklearn.ensemble import RandomForestRegressor
        m = RandomForestRegressor(random_state=seed, n_jobs=n_jobs)
    elif name == "XGBoost":
        from xgboost import XGBRegressor
        m = XGBRegressor(random_state=seed, n_jobs=n_jobs, tree_method="hist", objective="reg:squarederror")
    elif name == "SVR":
        from sklearn.svm import SVR
        m = SVR(kernel="rbf", cache_size=2000)
    else:
        raise ValueError(name)
    return Pipeline([("pre", pre), ("sc", StandardScaler()), ("m", m)])


def catboost_model(cat: list, seed: int, params: dict | None = None):
    from catboost import CatBoostRegressor
    p = {"loss_function": "RMSE", "random_seed": seed, "verbose": 0, "thread_count": -1,
         "cat_features": cat, "allow_writing_files": False}
    p.update(params or {})
    return CatBoostRegressor(**p)


class Wrapped:
    """Interfaz común .fit(df) / .predict(df)."""

    def __init__(self, name, cfg, cat, num, params=None, use_year=True):
        self.name, self.cfg, self.cat, self.num = name, cfg, cat, num
        self.params = params or {}
        self.use_year = use_year
        self.seed = cfg["split"]["seed"]
        self.info = {}

    def fit(self, df: pd.DataFrame):
        y = df["nota"].to_numpy()
        n = self.name
        if n == "Media global":
            self.mu = y.mean()
        elif n == "Media universidad":
            self.m = GroupMeanBaseline(["universidad"]).fit(df, y)
        elif n == "Media universidad-programa":
            self.m = GroupMeanBaseline(["univ_prog", "universidad"]).fit(df, y)
        elif n in ("Mixto institucional", "Mixto + covariables"):
            self.formula = mixed_formulas(self.use_year, df)[n]
            # niveles vistos en entrenamiento (para no fallar con categorías nuevas al predecir)
            self.levels = {c: (set(df[c].astype(str)), df[c].astype(str).mode().iat[0])
                           for c in MIXED_CATS if c in df and f"C({c})" in self.formula}
            self.m = fit_mixed(df, self.formula)
        elif n == "CatBoost":
            self.m = catboost_model(self.cat, self.seed, self.params)
            self.m.fit(_prep(df, self.cat, self.num), y)
        else:
            d = df
            if n == "SVR":
                cap = self.cfg["models"]["svr_fit_max_n"]
                if len(d) > cap:
                    d = d.sample(cap, random_state=self.seed)
                    self.info["svr_submuestra"] = cap
            self.m = sk_pipeline(n, self.cat, self.num, self.seed, self.cfg["models"]["n_jobs"])
            self.m.set_params(**self.params)
            self.m.fit(_prep(d, self.cat, self.num), d["nota"].to_numpy())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        n = self.name
        if n == "Media global":
            return np.full(len(df), self.mu)
        if n in ("Media universidad", "Media universidad-programa"):
            return self.m.predict(df)
        if n in ("Mixto institucional", "Mixto + covariables"):
            d = df.copy()
            for c, (seen, moda) in self.levels.items():
                d[c] = d[c].astype(str).where(d[c].astype(str).isin(seen), moda)
                if c == "anio":
                    d[c] = d[c].astype(int)
            return predict_mixed(self.m, d)
        return np.asarray(self.m.predict(_prep(df, self.cat, self.num)), float)


def tune(name: str, train: pd.DataFrame, cfg: dict, cat: list, num: list, log) -> dict:
    """Búsqueda aleatoria con validación cruzada agrupada por persona (criterio: MAE)."""
    seed, mc = cfg["split"]["seed"], cfg["models"]
    d = train
    if name == "SVR" and len(d) > mc["svr_tune_n"]:
        d = d.sample(mc["svr_tune_n"], random_state=seed)
    X, y = _prep(d, cat, num), d["nota"].to_numpy()
    cv = GroupKFold(n_splits=cfg["split"]["n_folds_cv"])
    space = SPACES[name]
    n_comb = int(np.prod([len(v) for v in space.values()]))
    groups = d["grupo_persona"].to_numpy()
    if name == "CatBoost":  # búsqueda manual (CatBoost no se clona bien con scikit-learn reciente)
        from sklearn.model_selection import ParameterSampler
        rows = []
        for params in ParameterSampler(space, n_iter=min(mc["n_iter_search"], n_comb), random_state=seed):
            maes = []
            for tr_i, va_i in cv.split(X, y, groups):
                m = catboost_model(cat, seed, params)
                m.fit(X.iloc[tr_i], y[tr_i])
                maes.append(np.abs(y[va_i] - m.predict(X.iloc[va_i])).mean())
            rows.append({"params": params, "mean_test_MAE": float(np.mean(maes)), "std_test_score": float(np.std(maes))})
        res = pd.DataFrame(rows).sort_values("mean_test_MAE")
        best = res.iloc[0]["params"]
        log.info("%s · mejor MAE de VC = %.4f · %s", name, res.iloc[0]["mean_test_MAE"], best)
        return {"best_params": best, "cv_mae": float(res.iloc[0]["mean_test_MAE"]), "cv_results": res}
    est, n_jobs = sk_pipeline(name, cat, num, seed, 1), mc["n_jobs"]  # paralelo entre combinaciones
    search = RandomizedSearchCV(est, space, n_iter=min(mc["n_iter_search"], n_comb),
                                scoring="neg_mean_absolute_error", cv=cv, random_state=seed,
                                n_jobs=n_jobs, refit=False)
    search.fit(X, y, groups=groups)
    best = search.best_params_
    log.info("%s · mejor MAE de VC = %.4f · %s", name, -search.best_score_, best)
    res = pd.DataFrame(search.cv_results_)[["params", "mean_test_score", "std_test_score"]]
    res["mean_test_MAE"] = -res.pop("mean_test_score")
    return {"best_params": best, "cv_mae": float(-search.best_score_), "cv_results": res}


def save_params(path, params: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False, default=str)


def load_params(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
