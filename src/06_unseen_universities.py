# -*- coding: utf-8 -*-
"""06 · Validación con universidades no vistas (RQ2, generalización institucional).

Cada pliegue deja fuera un grupo de universidades completas: el modelo se entrena con las demás
y estima la NPE de universidades que nunca vio. Usa los hiperparámetros elegidos en 05.
"""
import numpy as np
import pandas as pd

from common import (GroupMeanBaseline, Timer, cluster_bootstrap, get_logger, load_analytic, load_config,
                    out_dir, save_fig, save_numbers, save_table, setup_matplotlib)
from models import ML_MODELS, Wrapped, load_params

MODELOS = ["Media global", "Mixto + covariables", "Ridge"] + ML_MODELS


def main():
    cfg = load_config()
    log = get_logger("06_unseen_universities", cfg)
    plt = setup_matplotlib()
    df = load_analytic(cfg)
    mdir = out_dir(cfg, "models")
    best, sel = load_params(mdir / "hiperparametros.json"), load_params(mdir / "seleccion.json")
    cat, num = sel["predictores_cat"], sel["predictores_num"]

    oof = pd.DataFrame(index=df.index)
    for k in sorted(df["fold_univ"].unique()):
        tr, te = df[df.fold_univ != k], df[df.fold_univ == k]
        log.info("Pliegue %s: %d universidades fuera (%d registros)", k, te.universidad.nunique(), len(te))
        # [R4] etiqueta: la media del programa se calcula solo con las universidades de entrenamiento
        oof.loc[te.index, "Media del programa (universidades de entrenamiento)"] = GroupMeanBaseline(["programa"]).fit(tr, tr.nota).predict(te)
        for n in MODELOS:
            with Timer(log, f"pliegue {k} · {n}"):
                oof.loc[te.index, n] = Wrapped(n, cfg, cat, num, best.get(n)).fit(tr).predict(te)

    preds = {c: oof[c].to_numpy() for c in oof.columns}
    tab = cluster_bootstrap(df["nota"], preds, df["universidad"], cfg["bootstrap"]["n_boot"],
                            cfg["split"]["seed"], ref="Media global")
    save_table(cfg, tab, "06_universidades_no_vistas")
    log.info("\n%s", tab.round(4).to_string(index=False))

    # MAE por universidad (distribución)
    per_u = []
    for c in oof.columns:
        e = (df["nota"] - oof[c]).abs()
        per_u.append(e.groupby(df["universidad"]).mean().rename(c))
    per_u = pd.concat(per_u, axis=1)
    save_table(cfg, per_u.reset_index(), "06_mae_por_universidad")

    order = tab["modelo"].tolist()
    fig, ax = plt.subplots(figsize=(5.4, 0.4 * len(order) + 0.8))
    ax.boxplot([per_u[m].dropna() for m in order], vert=False, tick_labels=order, showfliers=True,
               flierprops={"markersize": 2})
    ax.invert_yaxis()
    ax.set_xlabel("MAE por universidad no vista (puntos)")
    save_fig(cfg, fig, "fig05_universidades_no_vistas")

    g = tab.set_index("modelo")
    best_name = tab.iloc[0]["modelo"]
    save_numbers(cfg, {"nv_mejor_modelo": best_name, "nv_mae_mejor": g.loc[best_name, "MAE"],
                       "nv_mae_global": g.loc["Media global", "MAE"],
                       "nv_mae_mejor_li": g.loc[best_name, "MAE_li"], "nv_mae_mejor_ls": g.loc[best_name, "MAE_ls"]})
    log.info("Listo: Tabla 5 y Fig. 5")


if __name__ == "__main__":
    main()
