# -*- coding: utf-8 -*-
"""10 · Análisis de sensibilidad: ¿los resultados de 05 se sostienen si cambian los datos?

Escenarios (entrenamiento y prueba se filtran igual; hiperparámetros de 05):
  Base · Sin créditos inválidos · Sin universidades con redondeo · Solo escala 11 · Solo escala 13
"""
import numpy as np
import pandas as pd

from common import (Timer, cluster_bootstrap, get_logger, load_analytic, load_config, out_dir,
                    save_numbers, save_table)
from models import Wrapped, load_params


def main():
    cfg = load_config()
    log = get_logger("10_sensitivity", cfg)
    df = load_analytic(cfg)
    mdir = out_dir(cfg, "models")
    best, sel = load_params(mdir / "hiperparametros.json"), load_params(mdir / "seleccion.json")
    cat, num = sel["predictores_cat"], sel["predictores_num"]
    modelos = list(dict.fromkeys(["Media universidad-programa", "Mixto institucional", "Mixto + covariables",
                                  "CatBoost", sel["mejor_aa"]]))

    tr_all = df[df.particion == "train"]
    enteras = pd.Series(np.isclose(tr_all["nota"], tr_all["nota"].round()), index=tr_all.index).groupby(tr_all["universidad"]).mean()
    con_redondeo = set(enteras[enteras > cfg["sensitivity"]["umbral_redondeo"]].index)
    escenarios = {
        "Base": pd.Series(True, index=df.index),
        "Sin créditos inválidos": ~df["flag_creditos_invalidos"],
        "Sin universidades con redondeo": ~df["universidad"].isin(con_redondeo),
    }
    if "nota_minima" in df and df["nota_minima"].notna().any():
        escenarios["Solo escala 11"] = df["nota_minima"] == 11
        escenarios["Solo escala 13"] = df["nota_minima"] == 13
    else:
        log.warning("Sin nota mínima: se omiten los escenarios por escala aprobatoria")

    rows = []
    for esc, mask in escenarios.items():
        d = df[mask]
        tr, te = d[d.particion == "train"], d[d.particion == "test"]
        if te["universidad"].nunique() < 5 or len(te) < 200:
            log.warning("Escenario %s con muy pocos datos; se omite", esc)
            continue
        preds = {}
        for m in modelos:
            with Timer(log, f"{esc} · {m}"):
                preds[m] = Wrapped(m, cfg, cat, num, best.get(m)).fit(tr).predict(te)
        t = cluster_bootstrap(te["nota"], preds, te["universidad"], cfg["bootstrap"]["n_boot"],
                              cfg["split"]["seed"], ref="Media universidad-programa")
        t.insert(0, "escenario", esc)
        t.insert(1, "n_prueba", len(te))
        t.insert(2, "universidades_prueba", te["universidad"].nunique())
        rows.append(t)
    tab = pd.concat(rows)
    save_table(cfg, tab.round(4), "10_sensibilidad")
    log.info("\n%s", tab[["escenario", "modelo", "MAE", "MAE_li", "MAE_ls", "dMAE_vs_ref", "dMAE_li", "dMAE_ls"]]
             .round(4).to_string(index=False))
    save_numbers(cfg, {"n_univ_con_redondeo": len(con_redondeo)})
    log.info("Listo: tablas suplementarias de sensibilidad")


if __name__ == "__main__":
    main()
