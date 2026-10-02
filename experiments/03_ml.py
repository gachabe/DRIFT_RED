"""ML: LR y RF sobre crudo y PCA-r; evalúa fase 'developed' y 'transition' en test."""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_config
from src.evaluation import (
    classification_metrics,
    confusion,
    fig_components_vs_performance,
    fig_confusions,
    save_tables,
)
from src.features import prepare, rows
from src.models import make_model

EVAL_PHASES = ("developed", "transition", "pre")


def run(ds, split, reps, cfg, outdir: Path, save: bool = True) -> pd.DataFrame:
    seed = cfg["seed"]
    assert set(split.train).isdisjoint(split.test)  # antes de entrenar
    X_tr, y_tr = rows(ds, split.train, [cfg.get("ml", {}).get("train_phase", "developed")])
    cap = cfg.get("ml", {}).get("max_train_rows")
    if cap and len(X_tr) > cap:
        keep = np.random.default_rng(seed).choice(len(X_tr), cap, replace=False)
        X_tr, y_tr = X_tr[keep], y_tr[keep]
    X_test_all, _ = rows(ds, split.test)
    test = {p: rows(ds, split.test, [p]) for p in EVAL_PHASES}

    out, cms = [], {}
    for model_name in cfg["models"]:
        for rep in reps:
            clf = make_model(model_name, seed).fit(rep.transform(X_tr), y_tr)
            rmse, ev = rep.rmse(X_test_all), rep.explained_variance
            label = "Raw" if rep.pca is None else f"PCA-{rep.n_components}"
            for phase, (X_te, y_te) in test.items():
                y_pred = clf.predict(rep.transform(X_te))
                out.append(
                    dict(
                        model=model_name, phase=phase, representation=rep.name, components=rep.n_components,
                        **classification_metrics(y_te, y_pred), recon_rmse=rmse, explained_variance=ev,
                        n_test_rows=len(y_te),
                    )
                )
                cms[(model_name, phase, label)] = confusion(y_te, y_pred)
    results = pd.DataFrame(out)
    if save:
        save_tables(results, outdir)
        fig_confusions(cms, outdir)
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
