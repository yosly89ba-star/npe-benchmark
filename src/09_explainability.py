# -*- coding: utf-8 -*-
"""09 · Explicabilidad (RQ4): importancia por bloques de variables.

a) SHAP exacto (TreeSHAP) sobre CatBoost en prueba: importancia por variable y por bloque
   (institucional, académico, demográfico, temporal). La contribución de un bloque es la suma de
   los SHAP de sus variables; así se evita repartir el crédito entre variables redundantes
   (p. ej., gestión está determinada por la universidad).
b) Importancia por permutación de bloques (aumento del MAE) para CatBoost y para el mejor modelo de
   scikit-learn: verifica que la lectura no depende del modelo explicado.
"""
import joblib
import numpy as np
import pandas as pd

from common import (get_logger, load_analytic, load_config, out_dir, save_fig, save_numbers, save_table,
                    setup_matplotlib)
from models import _prep, load_params


def block_permutation(model, X, y, blocks, n_rep, seed):
    rng = np.random.default_rng(seed)
    base = np.abs(y - model.predict(X)).mean()
    rows = []
    for b, cols in blocks.items():
        cols = [c for c in cols if c in X.columns]
        if not cols:
            continue
        d = []
        for _ in range(n_rep):
            Xp = X.copy()
            idx = rng.permutation(len(X))
            Xp[cols] = X[cols].iloc[idx].to_numpy()
            d.append(np.abs(y - model.predict(Xp)).mean() - base)
        rows.append({"bloque": b, "dMAE": np.mean(d), "dMAE_de": np.std(d)})
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    log = get_logger("09_explainability", cfg)
    plt = setup_matplotlib()
    from catboost import CatBoostRegressor, Pool
    df = load_analytic(cfg)
    mdir = out_dir(cfg, "models")
    sel = load_params(mdir / "seleccion.json")
    cat, num = sel["predictores_cat"], sel["predictores_num"]
    te = df[df.particion == "test"]
    if len(te) > cfg["shap"]["max_n"]:
        te = te.sample(cfg["shap"]["max_n"], random_state=cfg["split"]["seed"])
    X, y = _prep(te, cat, num), te["nota"].to_numpy()
    blocks = {b: [c for c in cols if c in X.columns] for b, cols in cfg["blocks"].items()}
    blocks = {b: c for b, c in blocks.items() if c}

    # a) SHAP CatBoost
    cb = CatBoostRegressor()
    cb.load_model(str(mdir / "catboost.cbm"))
    sv = cb.get_feature_importance(Pool(X, cat_features=cat), type="ShapValues")[:, :-1]  # última col = valor esperado
    S = pd.DataFrame(sv, columns=X.columns, index=X.index)
    feat = S.abs().mean().sort_values(ascending=False).rename("shap_abs_medio").reset_index().rename(columns={"index": "variable"})
    blk = pd.DataFrame([{"bloque": b, "shap_abs_medio": S[c].sum(axis=1).abs().mean()} for b, c in blocks.items()])
    blk = blk.sort_values("shap_abs_medio", ascending=False)
    save_table(cfg, feat.round(4), "09_shap_por_variable")
    save_table(cfg, blk.round(4), "09_shap_por_bloque")
    log.info("SHAP por variable:\n%s\nSHAP por bloque:\n%s", feat.round(4).to_string(index=False), blk.round(4).to_string(index=False))

    # b) permutación por bloques
    perms = []
    n_rep, seed = cfg["shap"]["n_repeats_perm"], cfg["split"]["seed"]
    p = block_permutation(cb, X, y, blocks, n_rep, seed)
    p.insert(0, "modelo", "CatBoost")
    perms.append(p)
    sk = sel["mejor_sklearn"]
    skm = joblib.load(mdir / f"{sk.replace(' ', '_').lower()}.joblib")
    p = block_permutation(skm, X, y, blocks, n_rep, seed)
    p.insert(0, "modelo", sk)
    perms.append(p)
    perm = pd.concat(perms)
    save_table(cfg, perm.round(4), "09_permutacion_por_bloque")
    log.info("Permutación por bloques:\n%s", perm.round(4).to_string(index=False))

    # Fig. 8: bloques (SHAP y permutación) + dependencia de edad y créditos
    fig, ax = plt.subplots(1, 4, figsize=(13, 2.9), gridspec_kw={"wspace": 0.75})
    ETQ = {"institucional": "Institucional", "academico": "Académico", "demografico": "Demográfico",
           "temporal": "Temporal"}  # [R4] etiquetas con tildes
    ax[0].barh([ETQ.get(b, b) for b in blk.bloque], blk.shap_abs_medio, color="#3b6ea8")
    ax[0].invert_yaxis()
    ax[0].set_xlabel("|SHAP| medio por bloque (CatBoost)")
    w = 0.38
    bl = blk.bloque.tolist()
    for i, (m, c) in enumerate(zip(perm.modelo.unique(), ["#3b6ea8", "#d08c2a"])):
        t = perm[perm.modelo == m].set_index("bloque").reindex(bl)
        ax[1].barh(np.arange(len(bl)) + i * w, t.dMAE, height=w, xerr=t.dMAE_de, color=c, label=m)
    ax[1].set_yticks(np.arange(len(bl)) + w / 2, [ETQ.get(b, b) for b in bl])
    ax[1].invert_yaxis()
    ax[1].set_xlabel("Aumento del MAE al permutar")
    ax[1].legend(frameon=False, fontsize=7)
    for a, (var, lab) in zip(ax[2:], (("edad", "Edad al egreso (años)"), ("creditos", "Créditos aprobados"))):
        if var in S:
            a.scatter(X[var], S[var], s=3, alpha=0.3, color="#3b6ea8")
            a.axhline(0, color="black", lw=0.6)
            a.set_xlabel(lab)
            a.set_ylabel(f"SHAP de {var}")
    save_fig(cfg, fig, "fig08_explicabilidad")

    tot = blk.shap_abs_medio.sum()
    save_numbers(cfg, {f"shap_pct_{b}": 100 * v / tot for b, v in zip(blk.bloque, blk.shap_abs_medio)})
    log.info("Listo: Fig. 8")


if __name__ == "__main__":
    main()
