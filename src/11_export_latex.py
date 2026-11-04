# -*- coding: utf-8 -*-
"""11 · Exporta resultados al manuscrito LaTeX (Overleaf).

- outputs/tables/*.csv  -> manuscript/tables/*.tex (booktabs; incluir con \\input{tables/xx})
- outputs/figures/*.pdf -> manuscript/figures/
- outputs/numbers.json  -> manuscript/numbers.tex (macros: \\input{numbers} en el preámbulo y luego
  usar, p. ej., \\npeMaeMejorAa en el texto; ninguna cifra se escribe a mano)
"""
import json
import re
import shutil

import pandas as pd

from common import ROOT, get_logger, load_config, out_dir, path

DIG = dict(zip("0123456789", ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]))


def esc(s) -> str:
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"),
                 ("$", r"\$"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\^{}")):
        s = s.replace(a, b)
    return s


def fmt(v) -> str:
    if isinstance(v, float):
        return "" if pd.isna(v) else f"{v:.3f}"
    return esc(v)


def to_booktabs(df: pd.DataFrame) -> str:
    cols = "l" * 1 + "r" * (df.shape[1] - 1)
    lines = [r"\begin{tabular}{" + cols + "}", r"\toprule",
             " & ".join(esc(c) for c in df.columns) + r" \\", r"\midrule"]
    for _, r in df.iterrows():
        lines.append(" & ".join(fmt(v) for v in r.tolist()) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def macro_name(key: str) -> str:
    parts = re.split(r"[^0-9A-Za-z]+", key)
    name = "npe" + "".join(p[:1].upper() + p[1:] for p in parts if p)
    return "".join(DIG.get(ch, ch) for ch in name)


def main():
    cfg = load_config()
    log = get_logger("11_export_latex", cfg)
    man = ROOT / "manuscript"
    (man / "tables").mkdir(parents=True, exist_ok=True)
    (man / "figures").mkdir(parents=True, exist_ok=True)

    for f in sorted(out_dir(cfg, "tables").glob("*.csv")):
        df = pd.read_csv(f)
        (man / "tables" / f"{f.stem}.tex").write_text(to_booktabs(df), encoding="utf-8")
    for f in sorted(out_dir(cfg, "figures").glob("*.pdf")):
        shutil.copy(f, man / "figures" / f.name)

    nf = path(cfg, "outputs") / "numbers.json"
    nums = json.loads(nf.read_text(encoding="utf-8")) if nf.exists() else {}
    lines = ["% Generado por 11_export_latex.py — no editar a mano"]
    for k, v in sorted(nums.items()):
        if isinstance(v, float):
            v = f"{v:.3f}" if abs(v) < 10 else f"{v:,.2f}"
        lines.append(rf"\newcommand{{\{macro_name(k)}}}{{{esc(v)}}}")
    (man / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Exportadas %d tablas, %d figuras y %d cifras clave a manuscript/",
             len(list((man / 'tables').glob('*.tex'))), len(list((man / 'figures').glob('*.pdf'))), len(nums))


if __name__ == "__main__":
    main()
