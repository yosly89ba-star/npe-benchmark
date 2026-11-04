# -*- coding: utf-8 -*-
"""04 · Modelos multinivel (RQ1): ¿qué parte de la varianza de la NPE es institucional?

M0  nulo: año fijo; interceptos aleatorios de universidad y de programa anidado en universidad
M1  M0 + variables de universidad y programa (gestión, licenciamiento, tipo de constitución, área del
    programa y, si existe, nota mínima aprobatoria)
M2  M1 + variables individuales (edad, edad², sexo, créditos, macrorregión de nacimiento)
Se usa toda la base analítica (todos los cortes): es un análisis descriptivo, no predictivo.
"""
import numpy as np
import pandas as pd

from common import (Timer, fit_mixed, get_logger, load_analytic, load_config, mixed_components,
                    save_fig, save_numbers, save_table, setup_matplotlib)


M0 = "M0: año"
M1 = "M1: + constitución, licenciamiento y área"
M2 = "M2: + edad, sexo, créditos y macrorregión"


def boot_icc(df, fixed, n_boot, seed, log):
    rng = np.random.default_rng(seed)
    univs = np.sort(df["universidad"].unique())  # [R1] orden fijo antes del remuestreo
    by_u = {u: d for u, d in df.groupby("universidad")}
    out = []
    for b in range(n_boot):
        pick = rng.choice(univs, size=len(univs), replace=True)
        parts = []
        for k, u in enumerate(pick):
            d = by_u[u].copy()
            d["universidad"] = f"{u}#{k}"
            d["univ_prog"] = d["univ_prog"] + f"#{k}"
            parts.append(d)
        try:
            comp = mixed_components(fit_mixed(pd.concat(parts), fixed))
            out.append((comp["icc_universidad"], comp["icc_univ_prog"]))
        except Exception as exc:  # noqa: BLE001
            log.warning("Réplica %d sin convergencia: %s", b, exc)
        if (b + 1) % 10 == 0:
            log.info("Bootstrap ICC: %d/%d", b + 1, n_boot)
    return np.array(out)


def main():
    cfg = load_config()
    log = get_logger("04_multilevel", cfg)
    plt = setup_matplotlib()
    df = load_analytic(cfg)
    df["edad2"] = (df["edad"] - df["edad"].mean()) ** 2

    multi_year = df["anio"].nunique() > 1
    f0 = "C(anio)" if multi_year else "1"
    usa_const = "constitucion" in df and df["constitucion"].nunique() > 1  # contiene a gestión
    f1 = f0 + (" + C(constitucion)" if usa_const else " + C(gestion)")
    for c in ("licenciado", "area"):
        if c in df and df[c].nunique() > 1:
            f1 += f" + C({c})"
    if "nota_minima" in df and df["nota_minima"].notna().all():
        f1 += " + nota_minima"
    f2 = f1 + " + edad + edad2 + C(sexo) + creditos + C(macroregion)"

    rows, fits = [], {}
    for name, fx in ((M0, f0), (M1, f1), (M2, f2)):  # [R4] mismos nombres que la Tabla 3
        with Timer(log, name):
            res = fit_mixed(df, fx, log=log)
        fits[name] = res
        comp = mixed_components(res)
        rows.append({"modelo": name, "parte_fija": fx, **comp, "loglik": res.llf, "n": int(res.nobs)})
        log.info("%s: %s", name, {k: round(v, 4) for k, v in comp.items()})
    tab = pd.DataFrame(rows)
    v0 = tab.loc[0, "var_universidad"]
    tab["reduccion_var_universidad_pct"] = 100 * (1 - tab["var_universidad"] / v0)
    save_table(cfg, tab, "04_modelos_multinivel")

    # coeficientes fijos de M1 y M2
    coefs = []
    for name in (M1, M2):
        r = fits[name]
        ci = r.conf_int()
        for term in r.fe_params.index:
            coefs.append({"modelo": name, "termino": term, "coef": r.fe_params[term],
                          "li": ci.loc[term, 0], "ls": ci.loc[term, 1], "p": r.pvalues[term]})
    save_table(cfg, pd.DataFrame(coefs), "04_coeficientes_fijos")

    # IC bootstrap del ICC (M0)
    nb = cfg["bootstrap"]["n_boot_icc"]
    with Timer(log, f"Bootstrap ICC ({nb} réplicas)"):
        bs = boot_icc(df, f0, nb, cfg["split"]["seed"], log)
    icc = tab.loc[0]
    ic = {"icc_univ": icc["icc_universidad"], "icc_univ_li": np.percentile(bs[:, 0], 2.5) if len(bs) else np.nan,
          "icc_univ_ls": np.percentile(bs[:, 0], 97.5) if len(bs) else np.nan,
          "icc_univprog": icc["icc_univ_prog"], "icc_univprog_li": np.percentile(bs[:, 1], 2.5) if len(bs) else np.nan,
          "icc_univprog_ls": np.percentile(bs[:, 1], 97.5) if len(bs) else np.nan}
    save_table(cfg, pd.DataFrame([ic]), "04_icc_bootstrap")
    save_numbers(cfg, {**ic, "red_var_univ_M1": tab.loc[1, "reduccion_var_universidad_pct"]})
    log.info("ICC: %s", {k: round(v, 3) for k, v in ic.items()})

    # Fig. 3: componentes de varianza
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    labels = ["Universidad", "Programa (en universidad)", "Residual (egresados)"]
    cols = ["#3b6ea8", "#7aa6d6", "#cfcfcf"]
    for i, r in tab.iterrows():
        left = 0
        for lab, key, c in zip(labels, ["var_universidad", "var_programa", "var_residual"], cols):
            w = 100 * r[key] / r["var_total"]
            ax.barh(i, w, left=left, color=c, label=lab if i == 0 else None)
            if w > 6:
                ax.text(left + w / 2, i, f"{w:.0f} %", ha="center", va="center", fontsize=8)
            left += w
    ax.set_yticks(range(len(tab)), tab["modelo"])
    ax.invert_yaxis()
    ax.set_xlabel("% de la varianza estimada")
    ax.legend(frameon=False, ncol=3, bbox_to_anchor=(0.5, -0.35), loc="upper center", fontsize=8)
    save_fig(cfg, fig, "fig03_componentes_varianza")
    log.info("Listo: Tabla 3 y Fig. 3")


if __name__ == "__main__":
    main()
