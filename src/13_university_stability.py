# -*- coding: utf-8 -*-
"""13 · Estabilidad del efecto universidad entre años y tendencia de la NPE.

a) Por corte: modelo mixto nulo -> ICC del año y efecto (BLUP) de cada universidad.
   Correlación de los efectos entre años para las universidades comunes: si una universidad
   «califica alto» todos los años, la referencia institucional es estable y confiable.
b) Tendencia: NPE ~ año con interceptos de universidad y programa, en el PANEL BALANCEADO
   (universidades presentes en todos los cortes), y como sensibilidad con todas las universidades.
"""
import numpy as np
import pandas as pd

from common import (Timer, fit_mixed, get_logger, load_analytic, load_config, mixed_components,
                    save_fig, save_numbers, save_table, setup_matplotlib, university_effects)


def corr_ci(x, y, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(x)
    rs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        rs.append(np.corrcoef(x[i], y[i])[0, 1])
    return np.nanpercentile(rs, [2.5, 97.5])


def main():
    cfg = load_config()
    log = get_logger("13_university_stability", cfg)
    plt = setup_matplotlib()
    df = load_analytic(cfg)
    years = sorted(df["anio"].unique())
    if len(years) < 2:
        log.warning("Solo hay un corte: análisis no aplicable")
        return

    effs, comps = {}, []
    for y in years:
        with Timer(log, f"Modelo nulo {y}"):
            res = fit_mixed(df[df.anio == y], "1", log=log)
        effs[y] = university_effects(res)
        comps.append({"anio": y, **mixed_components(res), "universidades": df.loc[df.anio == y, "universidad"].nunique()})
    comps = pd.DataFrame(comps)
    save_table(cfg, comps.round(4), "13_icc_por_anio")
    E = pd.DataFrame(effs)
    save_table(cfg, E.round(4), "13_efecto_universidad_por_anio", index=True)

    rows = []
    for i, a in enumerate(years):
        for b in years[i + 1:]:
            d = E[[a, b]].dropna()
            if len(d) < 5:
                continue
            li, ls = corr_ci(d[a].to_numpy(), d[b].to_numpy(), cfg["bootstrap"]["n_boot"], cfg["split"]["seed"])
            rows.append({"anio_a": a, "anio_b": b, "universidades_comunes": len(d),
                         "pearson": d[a].corr(d[b]), "pearson_li": li, "pearson_ls": ls,
                         "spearman": d[a].corr(d[b], method="spearman")})
    est = pd.DataFrame(rows)
    save_table(cfg, est.round(3), "13_estabilidad_efecto_universidad")
    log.info("ICC por año:\n%s\nEstabilidad:\n%s", comps.round(3).to_string(index=False), est.round(3).to_string(index=False))

    # tendencia: panel balanceado vs todas
    pres = df.groupby("universidad")["anio"].nunique()
    panel = set(pres[pres == len(years)].index)
    trend = []
    for lab, d in (("Panel balanceado", df[df.universidad.isin(panel)]), ("Todas las universidades", df)):
        with Timer(log, f"Tendencia · {lab}"):
            r = fit_mixed(d, "C(anio)", log=log)
        ci = r.conf_int()
        for t in [t for t in r.fe_params.index if t.startswith("C(anio)")]:
            trend.append({"muestra": lab, "termino": t, "cambio_vs_" + str(years[0]): r.fe_params[t],
                          "li": ci.loc[t, 0], "ls": ci.loc[t, 1], "p": r.pvalues[t],
                          "universidades": d.universidad.nunique(), "n": len(d)})
    trend = pd.DataFrame(trend)
    save_table(cfg, trend.round(4), "13_tendencia_npe")
    log.info("Tendencia:\n%s", trend.round(3).to_string(index=False))

    # figura: efectos en los dos últimos años + medias brutas por año
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    a, b = years[-2], years[-1]
    d = E[[a, b]].dropna()
    a1.scatter(d[a], d[b], s=12, color="#3b6ea8")
    lim = [min(d.min()), max(d.max())]
    a1.plot(lim, lim, color="gray", lw=0.8, ls="--")
    a1.set_xlabel(f"Efecto universidad {a}")
    a1.set_ylabel(f"Efecto universidad {b}")
    for lab, dd, c in (("Panel balanceado", df[df.universidad.isin(panel)], "#3b6ea8"), ("Todas", df, "#9a9a9a")):
        m = dd.groupby("anio")["nota"].mean()
        a2.plot(m.index, m.values, marker="o", color=c, label=lab)
    a2.set_xticks(years)
    a2.set_ylabel("NPE media observada")  # [R4] medias brutas, no ajustadas
    a2.legend(frameon=False, fontsize=8)
    save_fig(cfg, fig, "fig10_estabilidad_tendencia")

    last = est[(est.anio_a == a) & (est.anio_b == b)]
    save_numbers(cfg, {"univ_panel": len(panel),
                       "estabilidad_r_ultimos": float(last.pearson.iloc[0]) if len(last) else float("nan")})
    log.info("Listo: estabilidad y tendencia")


if __name__ == "__main__":
    main()
