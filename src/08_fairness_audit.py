# -*- coding: utf-8 -*-
"""08 · Auditoría por subgrupos (RQ4): ¿la referencia y el mejor modelo de AA se equivocan igual en todos?

Para sexo, gestión, macrorregión y grupo de edad: n, MAE y sesgo medio (predicha − observada) con
IC del 95 % por bootstrap agrupado por universidad, en el conjunto de prueba.
Nota de interpretación: en la referencia institucional (que no usa sexo ni edad), un sesgo por
subgrupo indica diferencias reales de desempeño dentro de los programas, no un modelo injusto;
en el modelo de AA (que sí usa esas variables) el sesgo debe ser cercano a cero.
"""
import numpy as np
import pandas as pd

from common import (get_logger, load_config, out_dir, path, save_fig, save_numbers, save_table,
                    setup_matplotlib)
from models import load_params


ETQ_VAR = {"sexo": "Sexo", "gestion": "Gestión", "macroregion": "Macrorregión", "grupo_edad": "Edad",
           "discapacidad": "Discapacidad"}
ETQ_CAT = {"F": "Mujeres", "M": "Hombres", "PUBLICO": "Pública", "PRIVADO": "Privada", "SI": "Sí", "NO": "No",
           "LIMA Y CALLAO": "Lima y Callao", "NO REGISTRADO": "No registrada", "≤23": "≤ 23 años",
           "24–27": "24–27 años", "28–34": "28–34 años", "≥35": "≥ 35 años"}


def boot_group(err, cluster, n_boot, seed):
    codes, uniq = pd.factorize(pd.Series(cluster).astype(str), sort=True)  # [R1]
    G = len(uniq)
    n_g = np.bincount(codes, minlength=G).astype(float)
    s_e = np.bincount(codes, err, G)
    s_a = np.bincount(codes, np.abs(err), G)
    W = np.random.default_rng(seed).multinomial(G, np.full(G, 1 / G), size=n_boot).astype(float)
    N = np.maximum(W @ n_g, 1)
    bias, mae = (W @ s_e) / N, (W @ s_a) / N
    return np.percentile(bias, [2.5, 97.5]), np.percentile(mae, [2.5, 97.5])


def main():
    cfg = load_config()
    log = get_logger("08_fairness_audit", cfg)
    plt = setup_matplotlib()
    pt = pd.read_parquet(path(cfg, "processed") / "predicciones_prueba.parquet")
    sel = load_params(out_dir(cfg, "models") / "seleccion.json")
    modelos = ["Mixto institucional", sel["mejor_aa"]]
    pt["grupo_edad"] = pd.cut(pt["edad"], [0, 23, 27, 34, 200], labels=["≤23", "24–27", "28–34", "≥35"])

    rows = []
    for m in modelos:
        err = pt[f"pred_{m}"] - pt["nota"]
        for var in [v for v in ("sexo", "gestion", "macroregion", "grupo_edad", "discapacidad") if v in pt]:
            for cat, idx in pt.groupby(var, observed=True).groups.items():
                e = err.loc[idx].to_numpy()
                if len(e) < 30:
                    continue
                (b_li, b_ls), (m_li, m_ls) = boot_group(e, pt.loc[idx, "universidad"], cfg["bootstrap"]["n_boot"],
                                                        cfg["split"]["seed"])
                rows.append({"modelo": m, "variable": var, "categoria": str(cat), "n": len(e),
                             "MAE": np.abs(e).mean(), "MAE_li": m_li, "MAE_ls": m_ls,
                             "sesgo": e.mean(), "sesgo_li": b_li, "sesgo_ls": b_ls})
    tab = pd.DataFrame(rows)
    save_table(cfg, tab.round(4), "08_auditoria_subgrupos")
    log.info("\n%s", tab.round(3).to_string(index=False))

    # Fig. 7
    fig, axes = plt.subplots(1, len(modelos), figsize=(3.6 * len(modelos), 0.22 * len(tab) / len(modelos) + 1.2),
                             sharey=True, sharex=True)  # [R4] mismo eje X para comparar modelos
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, modelos):
        t = tab[tab.modelo == m].reset_index(drop=True)
        y = np.arange(len(t))
        ax.errorbar(t.sesgo, y, xerr=[t.sesgo - t.sesgo_li, t.sesgo_ls - t.sesgo], fmt="o", ms=3, color="#3b6ea8",
                    ecolor="gray", lw=0.8)
        ax.axvline(0, color="black", lw=0.6)
        ax.set_yticks(y, [f"{ETQ_VAR.get(v, v)}: {ETQ_CAT.get(c, c.capitalize())}" for v, c in zip(t.variable, t.categoria)],
                      fontsize=7)  # [R4] etiquetas legibles
        ax.invert_yaxis()
        ax.set_title(m, fontsize=9)
        ax.set_xlabel("Sesgo medio (predicha − observada)")
    save_fig(cfg, fig, "fig07_auditoria_subgrupos")

    aa = tab[tab.modelo == sel["mejor_aa"]]
    save_numbers(cfg, {"sesgo_max_abs_aa": aa["sesgo"].abs().max()})
    log.info("Listo: Tabla 7 y Fig. 7")


if __name__ == "__main__":
    main()
