# -*- coding: utf-8 -*-
"""Utilidades compartidas: configuración, rutas, registro, métricas, bootstrap y modelos mixtos."""
from __future__ import annotations

import json
import logging
import os
import platform
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
QUICK = os.environ.get("NPE_QUICK", "0") == "1"   # modo prueba rápida (datos sintéticos)


# ------------------------------------------------------------------ configuración
def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if QUICK:  # parámetros mínimos para la prueba de humo
        cfg["models"]["n_iter_search"] = 2
        cfg["models"]["svr_tune_n"] = 800
        cfg["models"]["svr_fit_max_n"] = 1500
        cfg["bootstrap"]["n_boot"] = 50
        cfg["bootstrap"]["n_boot_icc"] = 3
        cfg["shap"]["max_n"] = 300
        cfg["shap"]["n_repeats_perm"] = 2
        cfg["split"]["n_folds_cv"] = 3
        cfg["split"]["n_folds_univ"] = 3
    return cfg


def path(cfg: dict, key: str) -> Path:
    p = ROOT / cfg["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


def out_dir(cfg: dict, sub: str) -> Path:
    p = path(cfg, "outputs") / sub
    p.mkdir(parents=True, exist_ok=True)
    return p


# ------------------------------------------------------------------ registro
def get_logger(name: str, cfg: dict) -> logging.Logger:
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(out_dir(cfg, "logs") / f"{name}.log", mode="w", encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)
    for h in (fh, sh):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.info("Python %s | %s | modo rápido=%s", platform.python_version(), platform.platform(), QUICK)
    return log


class Timer:
    def __init__(self, log, label):
        self.log, self.label = log, label

    def __enter__(self):
        self.t = time.time()
        self.log.info("Inicio: %s", self.label)
        return self

    def __exit__(self, *a):
        self.log.info("Fin: %s (%.1f s)", self.label, time.time() - self.t)


# ------------------------------------------------------------------ cifras clave
def save_numbers(cfg: dict, values: dict) -> None:
    """Acumula cifras clave en outputs/numbers.json (luego 11 las convierte en macros LaTeX)."""
    f = path(cfg, "outputs") / "numbers.json"
    data = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    for k, v in values.items():
        data[k] = float(v) if isinstance(v, (np.floating, np.integer)) else v
    f.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_table(cfg: dict, df: pd.DataFrame, name: str, index: bool = False) -> Path:
    p = out_dir(cfg, "tables") / f"{name}.csv"
    df.to_csv(p, index=index, encoding="utf-8")
    return p


# ------------------------------------------------------------------ texto
def norm_text(s):
    """Mayúsculas, sin tildes y sin espacios extra (para comparar nombres)."""
    if pd.isna(s):
        return s
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", s).strip().upper()


# ------------------------------------------------------------------ lectura de cortes TUNI
def read_raw(cfg: dict, year) -> pd.DataFrame:
    """Lee un corte y renombra columnas a los nombres internos de config.yaml."""
    f = ROOT / cfg["paths"]["raw_files"][year]
    r = cfg["read"]
    df = pd.read_csv(f, sep=r["sep"], encoding=r["encoding"], decimal=r["decimal"],
                     dtype=str, keep_default_na=True, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    ren = {v: k for k, v in cfg["columns"].items() if v}
    missing = [v for v in ren if v not in df.columns]
    if missing:
        raise KeyError(f"Corte {year}: faltan columnas {missing}. Revise 'columns' en config.yaml. "
                       f"Columnas del archivo: {list(df.columns)}")
    df = df[list(ren)].rename(columns=ren)
    for c in ("nota", "creditos", "edad", "anio_nacimiento"):
        if c in df.columns:
            # TUNI mezcla punto y coma decimal ("13.38" y "13,38"): se aceptan ambos
            df[c] = pd.to_numeric(df[c].astype(str).str.strip().str.replace(",", ".", regex=False),
                                  errors="coerce")
    for c in ("universidad", "programa", "gestion", "sexo", "departamento", "tipo_entidad", "nivel",
              "licenciado", "constitucion", "area", "discapacidad"):
        if c in df.columns:
            df[c] = df[c].map(norm_text)
    df["anio"] = int(year)
    return df


def parse_fecha(s: pd.Series) -> pd.Series:
    """Fechas de TUNI ('2022-12-31 05:00:00.0000000'); si no, intenta día/mes/año."""
    s = s.astype(str)
    f = pd.to_datetime(s.str[:10], format="%Y-%m-%d", errors="coerce")
    miss = f.isna()
    if miss.any():
        f[miss] = pd.to_datetime(s[miss], errors="coerce", dayfirst=True)
    return f


def harmonize_universities(df: pd.DataFrame) -> pd.DataFrame:
    """Una sola etiqueta por universidad entre años, usando el código (CODIGO_INEI) si existe:
    el nombre más frecuente de cada código. Evita contar dos veces una universidad que cambió de nombre."""
    if "codigo_universidad" not in df.columns or df["codigo_universidad"].isna().all():
        return df
    d = df.copy()
    d["codigo_universidad"] = d["codigo_universidad"].astype(str).str.strip()
    last = d.sort_values("anio").groupby("codigo_universidad")["universidad"].agg(lambda x: x.mode().iat[-1] if len(x.mode()) else x.iat[-1])
    dup = last[last.duplicated(keep=False)]
    last.loc[dup.index] = last.loc[dup.index] + " [" + dup.index + "]"
    d["universidad"] = d["codigo_universidad"].map(last)
    return d


# ------------------------------------------------------------------ filtros de población
def flujo(df, cfg):
    """Cuenta registros tras cada filtro, en el mismo orden que 02_build_dataset.py."""
    f = cfg["filters"]
    pasos = [("Registros iniciales", len(df))]
    if "tipo_entidad" in df and f["tipo_entidad_valores"]:
        df = df[df["tipo_entidad"].isin([v.upper() for v in f["tipo_entidad_valores"]])]
        pasos.append(("Solo universidades", len(df)))
    if "nivel" in df and f["nivel_valores"]:
        df = df[df["nivel"].isin([v.upper() for v in f["nivel_valores"]])]
        pasos.append(("Solo pregrado", len(df)))
    df = df[df["nota"].notna()]
    pasos.append(("Con nota registrada", len(df)))
    if f["excluir_nota_cero"]:
        df = df[df["nota"] != 0]
        pasos.append(("Nota distinta de 0", len(df)))
    df = df[df["nota"].between(f["nota_min_valida"], f["nota_max_valida"])]
    pasos.append((f"Nota entre {f['nota_min_valida']} y {f['nota_max_valida']}", len(df)))
    return pd.DataFrame(pasos, columns=["paso", "n"]), df


# ------------------------------------------------------------------ datos procesados
def load_analytic(cfg: dict) -> pd.DataFrame:
    f = path(cfg, "processed") / "base_analitica.parquet"
    if not f.exists():
        sys.exit("Falta data/processed/base_analitica.parquet: ejecute antes 02_build_dataset.py")
    return pd.read_parquet(f)


FEATURES_CAT = ["universidad", "programa", "univ_prog", "gestion", "licenciado", "constitucion", "area",
                "sexo", "departamento", "anio"]
FEATURES_NUM = ["creditos", "edad"]


def feature_lists(df: pd.DataFrame, use_year: bool = True):
    cat = [c for c in FEATURES_CAT if c in df.columns and (use_year or c != "anio")]
    num = [c for c in FEATURES_NUM if c in df.columns]
    if "nota_minima" in df.columns and df["nota_minima"].notna().mean() > 0.95:
        num.append("nota_minima")
    return cat, num


# ------------------------------------------------------------------ métricas
def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = y - p
    sst = ((y - y.mean()) ** 2).sum()
    return {"MAE": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean()), "R2": 1 - (e ** 2).sum() / sst}


def cluster_bootstrap(y, preds: dict, cluster, n_boot: int, seed: int, ref: str | None = None) -> pd.DataFrame:
    """IC del 95 % por bootstrap de conglomerados (universidades) para MAE, RMSE y R²,
    y para la diferencia de MAE de cada modelo frente a `ref` (ref − modelo; positivo = el modelo mejora)."""
    y = np.asarray(y, float)
    codes, uniq = pd.factorize(pd.Series(cluster).astype(str), sort=True)  # [R1] códigos independientes del orden de filas
    G = len(uniq)
    n_g = np.bincount(codes, minlength=G).astype(float)
    sy = np.bincount(codes, y, G)
    sy2 = np.bincount(codes, y ** 2, G)
    sums = {}
    for m, p in preds.items():
        e = y - np.asarray(p, float)
        sums[m] = (np.bincount(codes, np.abs(e), G), np.bincount(codes, e ** 2, G))
    rng = np.random.default_rng(seed)
    W = rng.multinomial(G, np.full(G, 1 / G), size=n_boot).astype(float)  # pesos por conglomerado
    N = W @ n_g
    SST = W @ sy2 - (W @ sy) ** 2 / N
    rows, boot_mae = [], {}
    for m, (sa, se) in sums.items():
        mae, sse = (W @ sa) / N, W @ se
        rmse, r2 = np.sqrt(sse / N), 1 - sse / SST
        boot_mae[m] = mae
        pt = metrics(y, preds[m])
        rows.append({"modelo": m, **{k: pt[k] for k in pt},
                     "MAE_li": np.percentile(mae, 2.5), "MAE_ls": np.percentile(mae, 97.5),
                     "RMSE_li": np.percentile(rmse, 2.5), "RMSE_ls": np.percentile(rmse, 97.5),
                     "R2_li": np.percentile(r2, 2.5), "R2_ls": np.percentile(r2, 97.5)})
    out = pd.DataFrame(rows)
    if ref is not None and ref in preds:
        d_pt = {m: out.loc[out.modelo == ref, "MAE"].iloc[0] - out.loc[out.modelo == m, "MAE"].iloc[0] for m in preds}
        out["dMAE_vs_ref"] = out["modelo"].map(d_pt)
        out["dMAE_li"] = out["modelo"].map({m: np.percentile(boot_mae[ref] - boot_mae[m], 2.5) for m in preds})
        out["dMAE_ls"] = out["modelo"].map({m: np.percentile(boot_mae[ref] - boot_mae[m], 97.5) for m in preds})
        out["referencia"] = ref
    return out.sort_values("MAE").reset_index(drop=True)


# ------------------------------------------------------------------ modelos de referencia simples
class GroupMeanBaseline:
    """Media por grupo con respaldo jerárquico: univ_prog -> universidad -> global."""

    def __init__(self, levels=("univ_prog", "universidad")):
        self.levels = list(levels)

    def fit(self, df, y):
        y = pd.Series(np.asarray(y, float), index=df.index)
        self.global_ = y.mean()
        self.maps_ = {lv: y.groupby(df[lv]).mean() for lv in self.levels}
        return self

    def predict(self, df):
        p = pd.Series(np.nan, index=df.index)
        for lv in self.levels:
            p = p.fillna(df[lv].map(self.maps_[lv]))
        return p.fillna(self.global_).to_numpy()


# ------------------------------------------------------------------ modelo mixto (statsmodels)
def fit_mixed(df: pd.DataFrame, fixed: str = "1", reml: bool = True, log=None):
    """Interceptos aleatorios de universidad y de programa anidado en universidad.
    `fixed` es la parte fija en sintaxis de fórmulas (p. ej. "C(anio) + edad")."""
    import statsmodels.formula.api as smf
    import warnings
    d = df.reset_index(drop=True).copy()  # índice único: evita desalinear grupos
    d["prog_key"] = d["univ_prog"].astype(str)
    md = smf.mixedlm(f"nota ~ {fixed}", d, groups=d["universidad"].astype(str),
                     re_formula="1", vc_formula={"prog": "0 + C(prog_key)"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = None
        for method in ("lbfgs", "powell", "nm"):
            try:
                res = md.fit(reml=reml, method=method)
                if np.isfinite(res.llf):
                    if log:  # [R2] deja constancia del método y de la convergencia
                        log.info("Mixto [%s] método=%s convergió=%s llf=%.3f", fixed, method,
                                 getattr(res, "converged", "?"), res.llf)
                    break
            except Exception as exc:  # noqa: BLE001
                if log:
                    log.warning("Ajuste mixto falló con %s: %s", method, exc)
    if res is None:
        raise RuntimeError(f"No convergió el modelo mixto con parte fija: {fixed}. "
                           "Suele deberse a variables fijas redundantes (colinealidad).")
    return res


def mixed_components(res) -> dict:
    v_u = float(res.cov_re.iloc[0, 0])
    v_p = float(np.asarray(res.vcomp)[0]) if len(np.asarray(res.vcomp)) else 0.0
    v_e = float(res.scale)
    tot = v_u + v_p + v_e
    return {"var_universidad": v_u, "var_programa": v_p, "var_residual": v_e, "var_total": tot,
            "icc_universidad": v_u / tot, "icc_univ_prog": (v_u + v_p) / tot}


def _blups(res):
    """Diccionarios de efectos aleatorios: universidad -> efecto; univ_prog -> efecto."""
    u_eff, p_eff = {}, {}
    for g, s in res.random_effects.items():
        for name, val in s.items():
            if name in ("Group", "Group Var") or name.lower().startswith("group"):
                u_eff[str(g)] = float(val)
            else:
                m = re.search(r"\[([^\[\]]*)\]\]?$", name)
                key = m.group(1) if m else name
                p_eff[key] = float(val)
    return u_eff, p_eff


def predict_mixed(res, df_new: pd.DataFrame) -> np.ndarray:
    """Parte fija + BLUP de universidad + BLUP de programa (0 si el grupo no se vio)."""
    d = df_new.copy()
    d["prog_key"] = d["univ_prog"].astype(str)
    fe = np.asarray(res.predict(d), float)  # solo la parte fija
    u_eff, p_eff = _blups(res)
    re_u = d["universidad"].astype(str).map(u_eff).fillna(0.0).to_numpy()
    re_p = d["prog_key"].map(p_eff).fillna(0.0).to_numpy()
    return fe + re_u + re_p


def university_effects(res) -> pd.Series:
    u, _ = _blups(res)
    return pd.Series(u, name="efecto_universidad")


# ------------------------------------------------------------------ figuras
def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    return plt


def save_fig(cfg, fig, name):
    d = out_dir(cfg, "figures")
    fig.savefig(d / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(d / f"{name}.png", bbox_inches="tight")
