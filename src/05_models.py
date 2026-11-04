# -*- coding: utf-8 -*-
"""05 · Comparación de modelos (RQ2) con partición entrenamiento / validación / prueba.

1. Afina Ridge, RF, XGBoost, CatBoost y SVR con validación cruzada agrupada por persona (solo entrenamiento).
2. Ajusta todos los modelos (referencias, mixtos y AA) en entrenamiento.
3. Selecciona con el conjunto de validación (MAE).
4. Evalúa UNA vez en prueba, con IC del 95 % por bootstrap agrupado por universidad.
Salidas: Tabla 4, Fig. 4, hiperparámetros, modelos guardados y predicciones de prueba (para 08 y 09).
"""
import joblib
import numpy as np
import pandas as pd

from common import (Timer, cluster_bootstrap, feature_lists, get_logger, load_analytic, load_config,
                    metrics, out_dir, path, save_fig, save_numbers, save_table, setup_matplotlib)
from models import BASELINES, ML_MODELS, TUNED, Wrapped, save_params, tune


def main():
    cfg = load_config()
    log = get_logger("05_models", cfg)
    plt = setup_matplotlib()
    df = load_analytic(cfg)
    cat, num = feature_lists(df, use_year=True)
    log.info("Predictores categóricos: %s · numéricos: %s", cat, num)
    tr, va, te = (df[df.particion == p].copy() for p in ("train", "val", "test"))
    log.info("Entrenamiento=%d · validación=%d · prueba=%d", len(tr), len(va), len(te))

    # 1. Afinamiento
    mdir = out_dir(cfg, "models")
    best, cv_tabs = {}, []
    for name in TUNED:
        with Timer(log, f"Afinamiento {name}"):
            r = tune(name, tr, cfg, cat, num, log)
        best[name] = r["best_params"]
        t = r["cv_results"]
        t.insert(0, "modelo", name)
        cv_tabs.append(t)
    save_params(mdir / "hiperparametros.json", best)
    cvt = pd.concat(cv_tabs)
    cvt["params"] = cvt["params"].astype(str)
    save_table(cfg, cvt, "05_busqueda_hiperparametros")

    # 2. Ajuste final en entrenamiento
    fitted = {}
    for name in BASELINES + ML_MODELS:
        with Timer(log, f"Ajuste {name}"):
            fitted[name] = Wrapped(name, cfg, cat, num, best.get(name)).fit(tr)
        if fitted[name].info:
            log.info("%s · %s", name, fitted[name].info)

    # 3. Selección en validación
    val_rows = [{"modelo": n, **metrics(va["nota"], m.predict(va))} for n, m in fitted.items()]
    val = pd.DataFrame(val_rows).sort_values("MAE")
    save_table(cfg, val, "05_validacion")
    log.info("Validación:\n%s", val.round(4).to_string(index=False))
    seleccion = {
        "mejor_global": val.iloc[0]["modelo"],
        "mejor_aa": val[val.modelo.isin(ML_MODELS)].iloc[0]["modelo"],
        "mejor_sklearn": val[val.modelo.isin(["Random Forest", "XGBoost", "SVR"])].iloc[0]["modelo"],
        "mejor_referencia": val[val.modelo.isin(BASELINES)].iloc[0]["modelo"],
        "predictores_cat": cat, "predictores_num": num,
    }
    save_params(mdir / "seleccion.json", seleccion)
    log.info("Selección (antes de ver la prueba): %s", seleccion)

    # 4. Prueba (una sola vez)
    preds = {n: m.predict(te) for n, m in fitted.items()}
    nb, seed = cfg["bootstrap"]["n_boot"], cfg["split"]["seed"]
    tab = cluster_bootstrap(te["nota"], preds, te["universidad"], nb, seed, ref="Media universidad-programa")
    tab2 = cluster_bootstrap(te["nota"], preds, te["universidad"], nb, seed + 1, ref="Mixto + covariables")
    tab = tab.merge(tab2[["modelo", "dMAE_vs_ref", "dMAE_li", "dMAE_ls"]].rename(
        columns={"dMAE_vs_ref": "dMAE_vs_mixto_cov", "dMAE_li": "dMAE_mixto_li", "dMAE_ls": "dMAE_mixto_ls"}), on="modelo")
    tab["tipo"] = np.where(tab.modelo.isin(ML_MODELS), "Aprendizaje automático", "Referencia")
    save_table(cfg, tab, "05_prueba_metricas_ic")
    log.info("Prueba:\n%s", tab.round(4).to_string(index=False))

    # predicciones de prueba para auditoría (08) y explicabilidad (09)
    keep = ["nota", "universidad", "univ_prog", "gestion", "licenciado", "constitucion", "area", "sexo",
            "departamento", "macroregion", "discapacidad", "edad", "creditos", "anio", "grupo_persona"]
    pt = te[[c for c in keep if c in te]].copy()
    for n, p in preds.items():
        pt[f"pred_{n}"] = p
    pt.to_parquet(path(cfg, "processed") / "predicciones_prueba.parquet", index=False)

    # guardar modelos de AA (para 09)
    for n in ML_MODELS:
        if n == "CatBoost":
            fitted[n].m.save_model(str(mdir / "catboost.cbm"))
        else:
            joblib.dump(fitted[n].m, mdir / f"{n.replace(' ', '_').lower()}.joblib")

    # Fig. 4
    t = tab.sort_values("MAE", ascending=False)
    fig, ax = plt.subplots(figsize=(5.2, 0.35 * len(t) + 0.8))
    colors = np.where(t.tipo == "Aprendizaje automático", "#3b6ea8", "#9a9a9a")
    ax.barh(t.modelo, t.MAE, color=colors)
    ax.errorbar(t.MAE, t.modelo, xerr=[t.MAE - t.MAE_li, t.MAE_ls - t.MAE], fmt="none", ecolor="black", lw=0.8, capsize=2)
    ax.set_xlabel("MAE en prueba (puntos; IC 95 % bootstrap por universidad)")
    ax.set_xlim(0, t.MAE_ls.max() * 1.08)
    save_fig(cfg, fig, "fig04_comparacion_modelos")

    g = tab.set_index("modelo")
    best_ml = seleccion["mejor_aa"]
    save_numbers(cfg, {
        "mae_mejor_aa": g.loc[best_ml, "MAE"], "mae_univprog": g.loc["Media universidad-programa", "MAE"],
        "mae_mixto_inst": g.loc["Mixto institucional", "MAE"], "mae_mixto_cov": g.loc["Mixto + covariables", "MAE"],
        "dmae_aa_vs_univprog": g.loc[best_ml, "dMAE_vs_ref"], "dmae_aa_vs_univprog_li": g.loc[best_ml, "dMAE_li"],
        "dmae_aa_vs_univprog_ls": g.loc[best_ml, "dMAE_ls"],
        "dmae_aa_vs_mixtocov": g.loc[best_ml, "dMAE_vs_mixto_cov"], "dmae_aa_vs_mixtocov_li": g.loc[best_ml, "dMAE_mixto_li"],
        "dmae_aa_vs_mixtocov_ls": g.loc[best_ml, "dMAE_mixto_ls"], "r2_mejor_aa": g.loc[best_ml, "R2"],
        "mejor_aa": best_ml,
    })
    log.info("Listo: Tabla 4 y Fig. 4. ¿El AA supera al mixto con covariables? dMAE=%.4f [%.4f; %.4f]",
             g.loc[best_ml, "dMAE_vs_mixto_cov"], g.loc[best_ml, "dMAE_mixto_li"], g.loc[best_ml, "dMAE_mixto_ls"])


if __name__ == "__main__":
    main()
