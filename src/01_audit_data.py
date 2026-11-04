# -*- coding: utf-8 -*-
"""01 · Auditoría de los cortes de TUNI (no modifica datos).

Responde, por corte: cuántos registros y universidades hay; qué valores toman los campos de
filtro; el flujo de exclusiones; qué universidades no están en todos los años; si hay personas
repetidas entre cortes (¿cortes acumulados?); y el porcentaje de notas enteras por universidad.
"""
import numpy as np
import pandas as pd

from common import flujo, get_logger, harmonize_universities, load_config, parse_fecha, read_raw, save_numbers, save_table


def main():
    cfg = load_config()
    log = get_logger("01_audit_data", cfg)
    years = sorted(cfg["paths"]["raw_files"])
    raw = {y: read_raw(cfg, y) for y in years}

    # 1. Valores de los campos de filtro y categóricos
    for y, df in raw.items():
        for c in ("tipo_entidad", "nivel", "gestion", "sexo", "licenciado", "constitucion", "area"):
            if c in df:
                log.info("Corte %s · valores de %s: %s", y, c, df[c].value_counts(dropna=False).to_dict())

    # 2. Flujo de exclusiones por corte
    flujos, filt = [], {}
    for y, df in raw.items():
        fl, filt[y] = flujo(df, cfg)
        fl.insert(0, "anio", y)
        flujos.append(fl)
        log.info("Corte %s · flujo:\n%s", y, fl.to_string(index=False))
    save_table(cfg, pd.concat(flujos), "01_flujo_por_corte")
    allf = harmonize_universities(pd.concat(filt.values(), ignore_index=True))
    filt = {y: allf[allf.anio == int(y)] for y in years}
    if "codigo_universidad" in allf:
        log.info("Universidades identificadas por código: %d", allf["codigo_universidad"].nunique())

    # edad: ¿EDAD es actual o al egreso?
    if {"anio_nacimiento", "fecha_egreso"} <= set(allf.columns):
        e_eg = parse_fecha(allf["fecha_egreso"]).dt.year - allf["anio_nacimiento"]
        log.info("EDAD del archivo: media=%.1f · edad al egreso calculada: media=%.1f · diferencia media=%.1f años",
                 allf["edad"].mean(), e_eg.mean(), (allf["edad"] - e_eg).mean())

    # 3. Universidades por corte: unión, intersección y presencia
    sets = {y: set(d["universidad"].dropna()) for y, d in filt.items()}
    union = set.union(*sets.values())
    inter = set.intersection(*sets.values())
    pres = pd.DataFrame({"universidad": sorted(union)})
    for y in years:
        pres[f"en_{y}"] = pres["universidad"].isin(sets[y])
    pres["n_cortes"] = pres[[f"en_{y}" for y in years]].sum(axis=1)
    for col in ("gestion", "licenciado", "constitucion"):
        if col in allf:
            moda = allf.groupby("universidad")[col].agg(lambda s: s.mode().iat[0] if len(s.mode()) else np.nan)
            pres[col] = pres["universidad"].map(moda)
    for y in years:
        pres[f"n_{y}"] = pres["universidad"].map(filt[y]["universidad"].value_counts()).fillna(0).astype(int)
    save_table(cfg, pres, "01_presencia_universidades")
    resumen = pd.DataFrame({
        "anio": years,
        "registros": [len(filt[y]) for y in years],
        "universidades": [len(sets[y]) for y in years],
        "programas": [filt[y]["programa"].nunique() for y in years],
    })
    save_table(cfg, resumen, "01_resumen_por_corte")
    log.info("Resumen por corte:\n%s", resumen.to_string(index=False))
    log.info("Universidades: unión=%d, en todos los cortes=%d, en un solo corte=%d",
             len(union), len(inter), int((pres.n_cortes == 1).sum()))
    log.info("Universidades que NO están en todos los cortes:\n%s",
             pres[pres.n_cortes < len(years)].to_string(index=False))

    # 4. Personas repetidas entre cortes (¿cortes acumulados?)
    acumulado = None
    if "id_persona" in raw[years[0]]:
        ids = {y: set(d["id_persona"].dropna()) for y, d in filt.items()}
        for i, a in enumerate(years):
            for b in years[i + 1:]:
                com = len(ids[a] & ids[b])
                log.info("Personas en %s y %s: %d (%.1f %% de %s)", a, b, com, 100 * com / max(len(ids[b]), 1), b)
        claves = allf[["id_persona", "universidad", "programa", "anio"]]
        rep = claves.duplicated(["id_persona", "universidad", "programa"], keep=False)
        acumulado = rep.mean() > 0.05
        log.info("Registros persona-universidad-programa repetidos entre cortes: %.1f %% -> cortes %s",
                 100 * rep.mean(), "ACUMULADOS (dedupe necesario)" if acumulado else "por año")
        multi = filt[years[-1]].groupby("id_persona").size()
        log.info("Corte %s: personas con más de un egreso: %d", years[-1], int((multi > 1).sum()))
    else:
        log.warning("Sin id_persona: no se puede verificar si los cortes son acumulados")

    # 5. Fecha de egreso frente al año del corte
    if "fecha_egreso" in raw[years[0]]:
        for y, d in filt.items():
            fe = parse_fecha(d["fecha_egreso"])
            log.info("Corte %s · años de egreso: %s", y, fe.dt.year.value_counts().sort_index().to_dict())

    # 6. Redondeo: % de notas enteras por universidad
    allf = allf.copy()
    allf["entera"] = np.isclose(allf["nota"], np.round(allf["nota"]))
    red = allf.groupby("universidad").agg(n=("nota", "size"), pct_enteras=("entera", "mean")).reset_index()
    red = red.sort_values("pct_enteras", ascending=False)
    save_table(cfg, red, "01_redondeo_por_universidad")
    umbral = cfg["sensitivity"]["umbral_redondeo"]
    log.info("Universidades con más de %.0f %% de notas enteras: %d", 100 * umbral, int((red.pct_enteras > umbral).sum()))

    save_numbers(cfg, {
        "univ_union": len(union), "univ_todos_cortes": len(inter),
        **{f"registros_{y}": len(filt[y]) for y in years},
        **{f"universidades_{y}": len(sets[y]) for y in years},
        "cortes_acumulados_detectado": bool(acumulado) if acumulado is not None else "sin_id",
    })
    log.info("Listo. Revise outputs/tables/01_*.csv y ajuste 'filters' en config.yaml si hace falta.")


if __name__ == "__main__":
    main()
