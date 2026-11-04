# -*- coding: utf-8 -*-
"""07 · NPE contextualizada (RQ3): la aplicación del artículo.

Referencia institucional = NPE esperada según universidad y programa (modelo mixto con año fijo y
BLUP de universidad y de programa; los programas pequeños se encogen hacia la media de su
universidad). Se estima por predicción cruzada: cada registro se predice con un modelo que no lo vio.

REGLA ÉTICA: la referencia usa SOLO universidad, programa y año. Sexo, edad y origen no ajustan notas.

NPE contextualizada = NPE observada − NPE esperada (puntos sobre o bajo lo habitual en su programa).
Luego se compara el ranking nacional por nota bruta con el ranking por nota contextualizada.
"""
import numpy as np
import pandas as pd

from common import (Timer, fit_mixed, get_logger, load_analytic, load_config, path, predict_mixed,
                    save_fig, save_numbers, save_table, setup_matplotlib)


def main():
    cfg = load_config()
    log = get_logger("07_contextualized_npe", cfg)
    plt = setup_matplotlib()
    df = load_analytic(cfg)
    fixed = "C(anio)" if df["anio"].nunique() > 1 else "1"

    df["npe_esperada"] = np.nan
    for k in sorted(df["fold_cv"].unique()):
        tr, te = df[df.fold_cv != k], df[df.fold_cv == k]
        with Timer(log, f"Predicción cruzada pliegue {k}"):
            res = fit_mixed(tr, fixed, log=log)
            df.loc[te.index, "npe_esperada"] = predict_mixed(res, te)
    df["npe_contextualizada"] = df["nota"] - df["npe_esperada"]
    df["pct_en_programa"] = df.groupby(["anio", "univ_prog"])["nota"].rank(pct=True)
    df["n_programa"] = df.groupby(["anio", "univ_prog"])["nota"].transform("size")

    # percentiles nacionales por año (bruto y contextualizado)
    df["pct_bruto"] = df.groupby("anio")["nota"].rank(pct=True)
    df["pct_contexto"] = df.groupby("anio")["npe_contextualizada"].rank(pct=True)
    edges = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labs = ["Q1 (inferior)", "Q2", "Q3", "Q4", "Q5 (superior)"]
    df["quintil_bruto"] = pd.cut(df["pct_bruto"], edges, labels=labs, include_lowest=True)
    df["quintil_contexto"] = pd.cut(df["pct_contexto"], edges, labels=labs, include_lowest=True)

    trans = pd.crosstab(df["quintil_bruto"], df["quintil_contexto"], normalize="index") * 100
    save_table(cfg, trans.round(1), "07_matriz_transicion_quintiles", index=True)

    res_rows = []
    for top in (0.10, 0.20):
        tb, tc = df["pct_bruto"] > 1 - top, df["pct_contexto"] > 1 - top
        sale = 100 * (tb & ~tc).sum() / tb.sum()
        res_rows.append({"grupo": f"{int(top * 100)} % superior", "n_grupo_bruto": int(tb.sum()),
                         "pct_que_sale_al_contextualizar": sale,
                         "pct_publica_bruto": 100 * df.loc[tb, "gestion"].str.contains("PUBLIC").mean(),
                         "pct_publica_contexto": 100 * df.loc[tc, "gestion"].str.contains("PUBLIC").mean(),
                         "universidades_bruto": df.loc[tb, "universidad"].nunique(),
                         "universidades_contexto": df.loc[tc, "universidad"].nunique()})
    rr = pd.DataFrame(res_rows)
    save_table(cfg, rr.round(2), "07_cambios_ranking")
    log.info("\n%s", rr.round(2).to_string(index=False))

    comp = []
    for col in ("macroregion", "gestion"):
        for lab, mask in (("bruto", df["pct_bruto"] > 0.8), ("contexto", df["pct_contexto"] > 0.8)):
            s = df.loc[mask, col].value_counts(normalize=True) * 100
            comp.append(pd.DataFrame({"variable": col, "ranking": lab, "categoria": s.index, "pct_en_20_superior": s.values}))
    save_table(cfg, pd.concat(comp).round(2), "07_composicion_20_superior")

    # validez de la referencia: correlación entre residuo y percentil dentro del programa
    rho = df[df.n_programa >= 30][["npe_contextualizada", "pct_en_programa"]].corr(method="spearman").iloc[0, 1]
    log.info("Spearman residuo vs percentil en el programa (programas con n>=30): %.3f", rho)

    keep = ["anio", "universidad", "univ_prog", "gestion", "macroregion", "nota", "npe_esperada",
            "npe_contextualizada", "pct_en_programa", "pct_bruto", "pct_contexto"]
    df[keep].to_parquet(path(cfg, "processed") / "npe_contextualizada.parquet", index=False)

    # Fig. 6: matriz de transición + composición pública en el 20 % superior
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.1), gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.55})
    im = a1.imshow(trans.values, cmap="Blues", vmin=0, vmax=100)
    a1.set_xticks(range(5), [l.split(" ")[0] for l in labs])
    a1.set_yticks(range(5), [l.split(" ")[0] for l in labs])
    a1.set_xlabel("Quintil por NPE contextualizada")
    a1.set_ylabel("Quintil por NPE bruta")
    for i in range(5):
        for j in range(5):
            v = trans.values[i, j]
            a1.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7, color="white" if v > 55 else "black")
    cb = fig.colorbar(im, ax=a1, fraction=0.046)
    cb.set_label("% de la fila", fontsize=7)
    r20 = rr[rr.grupo == "20 % superior"].iloc[0]
    a2.bar(["NPE bruta", "NPE\ncontextualizada"], [r20.pct_publica_bruto, r20.pct_publica_contexto], color=["#9a9a9a", "#3b6ea8"])
    a2.axhline(100 * df["gestion"].str.contains("PUBLIC").mean(), ls="--", color="black", lw=0.8)
    a2.set_ylabel("% de egresados de universidades\npúblicas en el 20 % superior", fontsize=8)
    save_fig(cfg, fig, "fig06_npe_contextualizada")

    save_numbers(cfg, {"pct_sale_top20": r20.pct_que_sale_al_contextualizar,
                       "pct_sale_top10": rr.iloc[0].pct_que_sale_al_contextualizar,
                       "pct_publica_top20_bruto": r20.pct_publica_bruto,
                       "pct_publica_top20_contexto": r20.pct_publica_contexto,
                       "rho_residuo_percentil": rho})
    log.info("Listo: Tabla 6 y Fig. 6")


if __name__ == "__main__":
    main()
