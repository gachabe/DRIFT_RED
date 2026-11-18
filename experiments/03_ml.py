"""ML: LR y RF sobre crudo y PCA-r; evalúa fase 'developed', 'transition' y 'pre' en test."""
from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import encode, normalized_entropy
from src.data import SCENARIOS, load_config
from src.evaluation import (
    classification_metrics,
    confusion,
    confusion_counts_csv,
    fig_components_vs_performance,
    fig_confusions,
    save_tables,
)
from src.features import prepare, rows
from src.models import make_model

EVAL_PHASES = ("developed", "transition", "pre")


def rep_label(rep) -> str:
    return "Raw" if rep.pca is None else f"PCA-{rep.n_components}"


def train_arrays(ds, split, cfg, seed, extra_phases=None):
    """Filas de entrenamiento (fase train_phase, con tope reproducible de filas)."""
    X_tr, y_tr = rows(ds, split.train, [cfg.get("ml", {}).get("train_phase", "developed")])
    cap = cfg.get("ml", {}).get("max_train_rows")
    if cap and len(X_tr) > cap:
        keep = np.random.default_rng(seed).choice(len(X_tr), cap, replace=False)
        X_tr, y_tr = X_tr[keep], y_tr[keep]
    return X_tr, y_tr


def bl_stats(clf, rep, X_bl, dates_bl, X_pre) -> list[dict]:
    """Control negativo: probabilidades de clase de los días del baseline (no se usa para entrenar) vs días 'pre' de test."""
    out = []
    classes = list(clf.classes_)
    for name, X, dates in (("BL", X_bl, dates_bl), ("pre_test", X_pre, None)):
        if X is None or len(X) == 0:
            continue
        P = clf.predict_proba(rep.transform(X))
        pred = np.array(classes, dtype=object)[P.argmax(1)]
        row = dict(source=name, n_days=len(X), mean_entropy=float(normalized_entropy(P).mean()), mean_max_prob=float(P.max(1).mean()))
        for c, p in zip(classes, P.mean(0)):
            row[f"mean_prob_{c}"] = float(p)
        for c in classes:
            row[f"frac_pred_{c}"] = float((pred == c).mean())
        out.append(row)
        if dates is not None:  # por año
            for y in sorted(set(pd.DatetimeIndex(dates).year)):
                sel = pd.DatetimeIndex(dates).year == y
                r2 = dict(source=f"BL_{y}", n_days=int(sel.sum()), mean_entropy=float(normalized_entropy(P[sel]).mean()),
                          mean_max_prob=float(P[sel].max(1).mean()))
                for c, p in zip(classes, P[sel].mean(0)):
                    r2[f"mean_prob_{c}"] = float(p)
                for c in classes:
                    r2[f"frac_pred_{c}"] = float((pred[sel] == c).mean())
                out.append(r2)
    return out


def run(ds, split, reps, cfg, outdir: Path, save: bool = True, collect: dict | None = None, bl=None) -> pd.DataFrame:
    """Entrena y evalúa. `collect` (dict) recibe las predicciones sobre TODAS las filas de test y, si se da `bl`
    (X_bl, fechas), las estadísticas del control negativo. El baseline nunca entra en el entrenamiento."""
    seed = cfg["seed"]
    assert set(split.train).isdisjoint(split.test)  # antes de entrenar
    X_tr, y_tr = train_arrays(ds, split, cfg, seed)
    te_mask = ds.meta["realization"].isin(split.test).to_numpy()
    X_te_all, y_te_all = ds.X[te_mask], ds.meta.loc[te_mask, "scenario"].to_numpy()
    ph_te = ds.meta.loc[te_mask, "phase"].to_numpy()
    if collect is not None:
        collect["test_idx"] = np.flatnonzero(te_mask).astype(np.int32)
        collect["bl"] = []

    out, cms = [], {}
    for model_name in cfg["models"]:
        for rep in reps:
            t0 = time.time()
            clf = make_model(model_name, seed).fit(rep.transform(X_tr), y_tr)
            fit_s = time.time() - t0
            Zte = rep.transform(X_te_all)
            y_pred_all = clf.predict(Zte)
            rmse, ev = rep.rmse(X_te_all), rep.explained_variance
            label = rep_label(rep)
            for phase in EVAL_PHASES:
                s = ph_te == phase
                y_te, y_pred = y_te_all[s], y_pred_all[s]
                out.append(
                    dict(
                        model=model_name, phase=phase, representation=rep.name, components=rep.n_components,
                        **classification_metrics(y_te, y_pred), recon_rmse=rmse, explained_variance=ev,
                        n_test_rows=len(y_te), fit_seconds=fit_s,
                    )
                )
                cms[(model_name, phase, label)] = confusion(y_te, y_pred)
            if collect is not None:
                collect[(model_name, label)] = encode(y_pred_all)
                if bl is not None:
                    X_bl, d_bl = bl
                    for r in bl_stats(clf, rep, X_bl, d_bl, X_te_all[ph_te == "pre"]):
                        collect["bl"].append(dict(model=model_name, representation=label, **r))
    results = pd.DataFrame(out)
    if save:
        save_tables(results, outdir)
        keep = (cfg.get("figures") or {}).get("confusion_components")
        cms_fig = {k: v for k, v in cms.items() if keep is None or k[2] == "Raw" or int(k[2].split("-")[1]) in keep}
        fig_confusions(cms_fig, outdir)
        confusion_counts_csv(cms, outdir)
        (outdir / "confusion_matrices.json").write_text(
            json.dumps({"|".join(k): v.tolist() for k, v in cms.items()}), encoding="utf-8"
        )
        fig_components_vs_performance(results, outdir)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    out = Path(cfg.get("results_dir", f"results/{cfg['mode']}"))
    out.mkdir(parents=True, exist_ok=True)
    ds, split = prepare(cfg, out)
    reps = importlib.import_module("experiments.02_rom").run(ds, split, cfg, out)
    run(ds, split, reps, cfg, out)
