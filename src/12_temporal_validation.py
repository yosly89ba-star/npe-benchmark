# -*- coding: utf-8 -*-
"""12 · Validación temporal: entrenar con los cortes anteriores y evaluar en el último.

El error del último corte se reporta por separado para universidades ya vistas en entrenamiento
y para universidades nuevas en ese año (prueba de arranque en frío).
"""
import numpy as np
import pandas as pd

from common import (Timer, cluster_bootstrap, feature_lists, get_logger, load_analytic, load_config,
                    out_dir, save_fig, save_numbers, save_table, setup_matplotlib)
from models import ML_MODELS, Wrapped, load_params

MODELOS = ["Media global", "Media universidad-programa", "Mixto institucional", "Mixto + covariables",
           "Ridge"] + ML_MODELS


def main():
    cfg = load_config()
    log = get_logger("12_temporal_validation", cfg)
    plt = setup_matplotlib()
    df = load_analytic(cfg)
    years = sorted(df["anio"].unique())
    if len(years) < 2:
        log.warning("Solo hay un corte: no se puede hacer validación temporal")
        return
    test_year = years[-1]
    tr, te = df[df.anio < test_year], df[df.anio == test_year].copy()
    cat, num = feature_lists(df, use_year=False)
    best = load_params(out_dir(cfg, "models") / "hiperparametros.json")
    vistas = set(tr["universidad"])
    te["segmento"] = te["universidad"].isin(vistas).map({True: "Universidades vistas", False: "Universidades nuevas"})
    log.info("Entrenamiento: cortes %s (%d) · prueba: %s (%d) · universidades nuevas en %s: %d",
             years[:-1], len(tr), test_year, len(te), test_year, te.loc[te.segmento == "Universidades nuevas", "universidad"].nunique())

    preds = {}
    for m in MODELOS:
        with Timer(log, m):
            preds[m] = Wrapped(m, cfg, cat, num, best.get(m), use_year=False).fit(tr).predict(te)

    rows = []
    for seg in ["Todas", "Universidades vistas", "Universidades nuevas"]:
        mask = pd.Series(True, index=te.index) if seg == "Todas" else te.segmento == seg
        if seg != "Todas" and mask.all():  # [R5] si todas las universidades ya se vieron, el segmento repite "Todas"
            log.info("Segmento %s idéntico a 'Todas' (no hay universidades nuevas); se omite", seg)
            continue
        if mask.sum() < 30 or te.loc[mask, "universidad"].nunique() < 2:
            log.warning("Segmento %s con muy pocos datos; se omite", seg)
            continue
        p = {k: v[mask.to_numpy()] for k, v in preds.items()}
        t = cluster_bootstrap(te.loc[mask, "nota"], p, te.loc[mask, "universidad"], cfg["bootstrap"]["n_boot"],
                              cfg["split"]["seed"], ref="Media universidad-programa")
        t.insert(0, "segmento", seg)
        t.insert(1, "n", int(mask.sum()))
        t.insert(2, "universidades", te.loc[mask, "universidad"].nunique())
        rows.append(t)
    tab = pd.concat(rows)
    save_table(cfg, tab.round(4), "12_validacion_temporal")
    log.info("\n%s", tab[["segmento", "modelo", "MAE", "MAE_li", "MAE_ls", "R2"]].round(4).to_string(index=False))

    segs = tab.segmento.unique()
    fig, axes = plt.subplots(1, len(segs), figsize=(3.4 * len(segs) + 1.6, 3.2), sharey=len(segs) > 1)
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, segs):
        t = tab[tab.segmento == s].set_index("modelo").sort_values("MAE")  # [R5] barras ordenadas por MAE
        ax.barh(t.index, t.MAE, xerr=[t.MAE - t.MAE_li, t.MAE_ls - t.MAE], color="#3b6ea8", ecolor="black")
        ax.set_title(s, fontsize=9)
        ax.set_xlabel(f"MAE en {test_year}")
    axes[0].invert_yaxis()
    save_fig(cfg, fig, "fig09_validacion_temporal")

    g = tab[tab.segmento == "Todas"].set_index("modelo")
    save_numbers(cfg, {"temp_anio_prueba": int(test_year), "temp_mejor": g.index[0], "temp_mae_mejor": g.iloc[0]["MAE"],
                       "temp_mae_univprog": g.loc["Media universidad-programa", "MAE"]})
    log.info("Listo: validación temporal")


if __name__ == "__main__":
    main()
