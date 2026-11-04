# -*- coding: utf-8 -*-
"""02 · Construye la base analítica (todos los cortes) y las particiones.

Salidas: data/processed/base_analitica.parquet (sin identificador personal), la tabla de flujo
y el manifiesto de la partición (data/processed/manifiesto_particion.csv + hashes en el log).
Columnas de partición:
  particion   train / val / test (80/10/10), agrupada por persona si hay identificador
  fold_cv     pliegue para predicción cruzada (07), agrupado por persona
  fold_univ   pliegue por universidad (06: universidades no vistas)

Versión C (subsanación de la crítica 3). Cambios respecto de la versión B:
  [C1] Huella estable de cada registro (hash del contenido) para desempates y como clave de respaldo.
  [C2] Registros sin id_persona: reciben un identificador propio (antes todos los NaN eran "nan" = una persona).
  [C3] Deduplicación con orden total (fecha, año, universidad, programa, huella); fechas faltantes = más antiguas.
  [C4] Edad de respaldo con un año de referencia fijo (config), ya no con el año del sistema.
  [C5] Imputación de edad y créditos con la mediana del conjunto de ENTRENAMIENTO, después de la partición.
  [C6] Orden final determinista de las filas (afecta submuestras por posición en scripts posteriores).
  [C7] grupo_persona y fallback sin ID independientes del orden de las filas.
  [C8] Manifiesto de la partición con SHA-256 por conjunto, para comparar corridas.
"""
import hashlib

import numpy as np
import pandas as pd

from common import (ROOT, flujo, get_logger, harmonize_universities, load_config, norm_text, parse_fecha,
                    path, read_raw, save_numbers, save_table)


def assign_group_folds(groups: pd.Series, k: int, seed: int) -> np.ndarray:
    uniq = pd.Series(np.sort(groups.astype(str).unique()))  # orden fijo: no depende del orden de filas
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(uniq, rng.permutation(np.arange(len(uniq)) % k)))
    return groups.astype(str).map(fold_of).to_numpy()


def row_fingerprint(df: pd.DataFrame) -> pd.Series:  # [C1]
    """Huella estable del contenido de cada fila: no depende del orden de filas ni de columnas."""
    cols = sorted(df.columns)
    return pd.util.hash_pandas_object(df[cols].astype(str), index=False).astype("uint64").astype(str)


def sha256_of(values) -> str:  # [C8]
    return hashlib.sha256("\n".join(sorted(map(str, values))).encode("utf-8")).hexdigest()


def main():
    cfg = load_config()
    log = get_logger("02_build_dataset", cfg)
    f, sp = cfg["filters"], cfg["split"]
    years = sorted(cfg["paths"]["raw_files"])

    parts, flows = [], []
    for y in years:
        fl, d = flujo(read_raw(cfg, y), cfg)
        fl.insert(0, "anio", y)
        flows.append(fl)
        parts.append(d)
    df = harmonize_universities(pd.concat(parts, ignore_index=True))
    df["_huella"] = row_fingerprint(df)  # [C1] se calcula sobre los datos originales armonizados
    has_id = "id_persona" in df.columns and df["id_persona"].notna().mean() > 0.9

    # [C2] identificador estable: los registros sin id_persona no se agrupan entre sí
    if has_id:
        sin_id = df["id_persona"].isna()
        df["_id"] = df["id_persona"].astype(str).where(~sin_id, "SINID_" + df["_huella"])
        log.info("Registros sin id_persona (identificador propio por huella): %d", int(sin_id.sum()))

    # [C3] orden total y determinista antes de deduplicar
    df["_fe"] = parse_fecha(df["fecha_egreso"]) if "fecha_egreso" in df else pd.NaT
    df = df.sort_values(["_fe", "anio", "universidad", "programa", "_huella"],
                        kind="mergesort", na_position="first")  # fecha faltante = más antigua

    # --- duplicados dentro de cada corte (misma persona, universidad y programa): se conserva el egreso más reciente
    extra = []
    if has_id:
        n0 = len(df)
        df = df.drop_duplicates(["anio", "_id", "universidad", "programa"], keep="last")
        extra.append(("Duplicados dentro del corte", n0 - len(df)))

        # --- cortes acumulados: una persona-programa solo una vez (primer corte en que aparece)
        acum = cfg["cortes_acumulados"]
        if acum == "auto":
            acum = df.duplicated(["_id", "universidad", "programa"], keep=False).mean() > 0.05
        if acum:
            n0 = len(df)
            df = (df.sort_values(["anio", "_fe", "_huella"], kind="mergesort", na_position="first")
                    .drop_duplicates(["_id", "universidad", "programa"], keep="first"))
            extra.append(("Repetidos entre cortes (acumulados)", n0 - len(df)))
        log.info("Cortes tratados como %s", "acumulados" if acum else "por año")
    else:
        log.warning("Sin identificador de persona: no se deduplica por persona ni se agrupa la partición")

    # --- edad al egreso (en TUNI, EDAD es la edad a la fecha de extracción del archivo)
    if {"anio_nacimiento", "fecha_egreso"} <= set(df.columns):
        anio_eg = df["_fe"].dt.year.fillna(df["anio"])
        edad_eg = anio_eg - df["anio_nacimiento"]
        # [C4] año de referencia fijo para la EDAD del archivo: config["anio_referencia_edad"] puede ser
        # un entero (mismo año para todos los cortes) o un diccionario {corte: año de extracción}.
        # Verificado en los cuatro cortes: ANIO_NACIMIENTO + EDAD = 2026 (año de descarga de TUNI).
        ref = cfg.get("anio_referencia_edad", 2026)
        log.info("Año de referencia de la EDAD de TUNI: %s", ref)
        if isinstance(ref, dict):
            anio_ref = df["anio"].map({int(k): v for k, v in ref.items()})
        else:
            anio_ref = pd.Series(int(ref), index=df.index)
        edad_respaldo = df["edad"] - (anio_ref - anio_eg)
        log.info("Edad al egreso calculada para %.1f %% (media %.1f); respaldo con EDAD del archivo para %d registros",
                 100 * edad_eg.notna().mean(), edad_eg.mean(), int((edad_eg.isna() & edad_respaldo.notna()).sum()))
        df["edad"] = edad_eg.fillna(edad_respaldo)

    # --- variables derivadas
    df["univ_prog"] = df["universidad"].astype(str) + " || " + df["programa"].astype(str)
    for c in ("departamento", "sexo", "gestion", "licenciado", "constitucion", "area", "discapacidad"):
        if c in df:
            df[c] = df[c].fillna("NO REGISTRADO")
    if "constitucion" in df:  # en TUNI la constitución está vacía justo en las públicas
        pub = df["gestion"].str.contains("PUBLIC") & (df["constitucion"] == "NO REGISTRADO")
        df.loc[pub, "constitucion"] = "PUBLICA"
    mapa = {norm_text(d): reg for reg, lst in cfg["macroregiones"].items() for d in lst}
    df["macroregion"] = df["departamento"].map(mapa).fillna("NO REGISTRADO")

    # indicadores de revisión (no excluyen); se calculan antes de imputar
    df["flag_creditos_invalidos"] = (df["creditos"] < f["creditos_min_validos"]) | df["creditos"].isna()
    df["flag_edad_extrema"] = ~df["edad"].between(f["edad_min"], f["edad_max"])
    df["flag_nota_baja"] = df["nota"] < 11
    df.loc[df["edad"] > f["edad_max"], "edad"] = np.nan  # regla fija: edad no plausible -> faltante
    for c in ("creditos", "edad"):
        df[f"{c}_faltante"] = df[c].isna()
    # [C5] la imputación se hace DESPUÉS de la partición, con medianas del entrenamiento

    # nota mínima aprobatoria (opcional)
    nm_f = ROOT / cfg["paths"]["nota_minima"]
    if nm_f.exists():
        nm = pd.read_csv(nm_f, encoding="utf-8")
        nm["universidad"] = nm["universidad"].map(norm_text)
        df["nota_minima"] = df["universidad"].map(nm.set_index("universidad")["nota_minima"])
        log.info("Nota mínima aprobatoria disponible para %.1f %% de los registros", 100 * df["nota_minima"].notna().mean())
    else:
        log.warning("Sin archivo de nota mínima: los análisis que la usan se omiten")

    # [C6] orden final determinista de las filas
    df = df.sort_values(["anio", "universidad", "programa", "_huella"], kind="mergesort").reset_index(drop=True)

    # --- particiones
    rng = np.random.default_rng(sp["seed"])
    grp = df["_id"] if has_id else df["_huella"]  # [C7] sin ID: huella del registro, no su posición
    g_uniq = np.sort(grp.unique())  # orden fijo antes de permutar
    g_perm = rng.permutation(len(g_uniq))
    n_test, n_val = int(len(g_uniq) * sp["test_size"]), int(len(g_uniq) * sp["val_size"])
    lab = np.array(["train"] * len(g_uniq), dtype=object)
    lab[g_perm[:n_test]] = "test"
    lab[g_perm[n_test:n_test + n_val]] = "val"
    df["particion"] = grp.map(dict(zip(g_uniq, lab))).to_numpy()
    df["fold_cv"] = assign_group_folds(grp, sp["n_folds_cv"], sp["seed"] + 1)
    df["fold_univ"] = assign_group_folds(df["universidad"], sp["n_folds_univ"], sp["seed"] + 2)
    df["grupo_persona"] = pd.Categorical(grp, categories=g_uniq).codes  # [C7] código interno, no reversible

    # [C5] imputación con medianas calculadas solo en entrenamiento
    medianas = df.loc[df["particion"] == "train", ["creditos", "edad"]].median()
    for c in ("creditos", "edad"):
        log.info("Imputación de %s: %d faltantes con la mediana de entrenamiento (%.2f)",
                 c, int(df[f"{c}_faltante"].sum()), medianas[c])
        df[c] = df[c].fillna(medianas[c])

    # [C8] manifiesto de la partición y hashes por conjunto (sin identificador personal)
    man = df[["_huella", "anio", "particion", "fold_cv", "fold_univ"]].rename(columns={"_huella": "id_registro"})
    man.to_csv(path(cfg, "processed") / "manifiesto_particion.csv", index=False)
    hashes = {f"sha256_{p}": sha256_of(man.loc[man["particion"] == p, "id_registro"]) for p in ("train", "val", "test")}
    hashes["sha256_fold_univ"] = sha256_of(df["universidad"].astype(str) + "|" + df["fold_univ"].astype(str))
    for k, v in hashes.items():
        log.info("%s = %s", k, v)

    df = df.drop(columns=[c for c in ("id_persona", "_id", "fecha_egreso", "_fe", "_huella", "tipo_entidad", "nivel",
                                      "anio_nacimiento") if c in df])
    out = path(cfg, "processed") / "base_analitica.parquet"
    df.to_parquet(out, index=False)

    flow = pd.concat(flows)
    for paso, n in extra:
        flow = pd.concat([flow, pd.DataFrame([{"anio": "todos", "paso": paso, "n": -n}])])
    flow = pd.concat([flow, pd.DataFrame([{"anio": "todos", "paso": "Base analítica final", "n": len(df)}])])
    save_table(cfg, flow, "02_flujo_final")

    log.info("Base analítica: %d registros, %d universidades, %d combinaciones universidad-programa",
             len(df), df["universidad"].nunique(), df["univ_prog"].nunique())
    log.info("Partición:\n%s", df["particion"].value_counts().to_string())
    log.info("Indicadores: créditos inválidos=%d, edad extrema=%d, nota < 11=%d",
             df.flag_creditos_invalidos.sum(), df.flag_edad_extrema.sum(), df.flag_nota_baja.sum())
    save_numbers(cfg, {"n_total": len(df), "n_universidades": df["universidad"].nunique(),
                       "n_programas": df["programa"].nunique(), "n_univ_prog": df["univ_prog"].nunique(),
                       **{f"n_{k}": int(v) for k, v in df["particion"].value_counts().items()},
                       "particion_por_persona": bool(has_id),
                       "n_creditos_imputados": int(df["creditos_faltante"].sum()),
                       "n_edad_imputada": int(df["edad_faltante"].sum()),
                       "mediana_train_creditos": float(medianas["creditos"]),
                       "mediana_train_edad": float(medianas["edad"]),
                       **hashes})


if __name__ == "__main__":
    main()
