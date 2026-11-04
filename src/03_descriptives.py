# -*- coding: utf-8 -*-
"""03 · Descriptivos: características de la muestra (Tabla 2) y distribución de la NPE (Fig. 2)."""
import numpy as np
import pandas as pd

from common import get_logger, load_analytic, load_config, save_fig, save_numbers, save_table, setup_matplotlib


def resumen(d: pd.DataFrame) -> dict:
    q1, q3 = d["nota"].quantile([0.25, 0.75])
    return {
        "n": len(d), "universidades": d["universidad"].nunique(), "programas": d["univ_prog"].nunique(),
        "nota_media": d["nota"].mean(), "nota_de": d["nota"].std(), "nota_mediana": d["nota"].median(),
        "nota_q1": q1, "nota_q3": q3,
        "pct_mujeres": 100 * d["sexo"].str.startswith("F").mean(),
        "edad_media": d["edad"].mean(), "edad_de": d["edad"].std(),
        "creditos_mediana": d["creditos"].median(),
        "pct_publica": 100 * d["gestion"].str.contains("PUBLIC").mean(),
        "pct_notas_enteras": 100 * np.isclose(d["nota"], d["nota"].round()).mean(),
    }


def main():
    cfg = load_config()
    log = get_logger("03_descriptives", cfg)
    plt = setup_matplotlib()
    df = load_analytic(cfg)

    rows = [{"grupo": "Total", **resumen(df)}]
    for y, d in df.groupby("anio"):
        rows.append({"grupo": f"Corte {y}", **resumen(d)})
    for g, d in df.groupby("gestion"):
        rows.append({"grupo": f"Gestión {g.title()}", **resumen(d)})
    for s, d in df.groupby("sexo"):
        rows.append({"grupo": f"Sexo {s.title()}", **resumen(d)})
    tab = pd.DataFrame(rows)
    save_table(cfg, tab.round(2), "03_caracteristicas_muestra")
    log.info("\n%s", tab.round(2).to_string(index=False))

    # Fig. 2: distribución de la NPE por gestión (y por corte)
    years = sorted(df["anio"].unique())
    fig, axes = plt.subplots(1, len(years), figsize=(3.2 * len(years), 2.8), sharey=True)
    axes = np.atleast_1d(axes)
    bins = np.arange(7, 20.25, 0.25)
    for ax, y in zip(axes, years):
        d = df[df.anio == y]
        for g, c in zip(sorted(d["gestion"].unique()), ["#3b6ea8", "#d08c2a", "#888888"]):
            ax.hist(d.loc[d.gestion == g, "nota"], bins=bins, alpha=0.6, color=c, label={"PUBLICO": "Pública", "PRIVADO": "Privada"}.get(g, g.title()))  # [R4]
            ax.axvline(d.loc[d.gestion == g, "nota"].mean(), color=c, ls="--", lw=1)
        ax.set_title(f"Corte {y}")
        ax.set_xlabel("NPE (escala 0–20)")
    axes[0].set_ylabel("Egresados")
    axes[0].legend(frameon=False)
    save_fig(cfg, fig, "fig02_distribucion_npe")

    t = tab.set_index("grupo")
    save_numbers(cfg, {"npe_media": t.loc["Total", "nota_media"], "npe_de": t.loc["Total", "nota_de"],
                       "pct_publica": t.loc["Total", "pct_publica"]})
    log.info("Listo: Tabla 2 y Fig. 2")


if __name__ == "__main__":
    main()
