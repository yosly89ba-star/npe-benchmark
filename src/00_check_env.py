# -*- coding: utf-8 -*-
"""00 · Verifica librerías, config.yaml y que los cortes de TUNI tengan las columnas configuradas."""
import importlib

import pandas as pd

from common import ROOT, get_logger, load_config

REQ = ["pandas", "numpy", "scipy", "sklearn", "statsmodels", "catboost",
       "matplotlib", "yaml", "pyarrow", "joblib"]
OPCIONALES = {"xgboost": "en Mac instale OpenMP con: brew install libomp (si no, XGBoost se omite)"}


def main():
    cfg = load_config()
    log = get_logger("00_check_env", cfg)
    ok = True
    for m in REQ:
        try:
            mod = importlib.import_module(m)
            log.info("%-12s %s", m, getattr(mod, "__version__", "?"))
        except ImportError:
            ok = False
            log.error("Falta la librería %s: pip install -r requirements.txt", m)
    for m, ayuda in OPCIONALES.items():
        try:
            mod = importlib.import_module(m)
            log.info("%-12s %s", m, getattr(mod, "__version__", "?"))
        except Exception as exc:  # noqa: BLE001
            log.warning("%s no carga (%s): %s", m, type(exc).__name__, ayuda)

    for year, rel in cfg["paths"]["raw_files"].items():
        f = ROOT / rel
        if not f.exists():
            ok = False
            log.error("Corte %s: no existe %s", year, f)
            continue
        head = pd.read_csv(f, sep=cfg["read"]["sep"], encoding=cfg["read"]["encoding"], nrows=5, dtype=str)
        cols = [c.strip() for c in head.columns]
        log.info("Corte %s: %d columnas -> %s", year, len(cols), cols)
        for k, v in cfg["columns"].items():
            if v and v not in cols:
                ok = False
                log.error("Corte %s: la columna '%s' (%s) no está en el archivo", year, v, k)
            elif not v:
                log.warning("Corte %s: '%s' está en null (análisis que la usan se omiten)", year, k)

    nm = ROOT / cfg["paths"]["nota_minima"]
    log.info("Archivo de nota mínima: %s", "presente" if nm.exists() else "AUSENTE (opcional)")
    log.info("RESULTADO: %s", "todo listo" if ok else "corrija los errores de arriba antes de seguir")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
