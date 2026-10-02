"""ROM: PCA ajustada solo con las realizaciones de entrenamiento."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data import load_config
from src.evaluation import fig_pca_variance, fig_reconstruction
from src.features import prepare, rows
from src.rom import fit_representation, rmse_curve


def run(ds, split, cfg, outdir: Path):
    """Devuelve [Raw, PCA-r...] ajustados con TODOS los días de las realizaciones train."""
    assert set(split.train).isdisjoint(split.test)
    X_train, _ = rows(ds, split.train)
    X_test, _ = rows(ds, split.test)
    reps = [fit_representation(X_train, None)] + [fit_representation(X_train, r) for r in cfg["rom"]["components"]]
    rep24 = fit_representation(X_train, 24)
    fig_pca_variance(rep24, cfg["rom"]["components"], outdir)
    fig_reconstruction(rmse_curve(rep24, X_test), reps, X_test[len(X_test) // 2], outdir)
    return reps


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    out = Path(cfg.get("results_dir", f"results/{cfg['mode']}"))
    out.mkdir(parents=True, exist_ok=True)
    ds, split = prepare(cfg, out)
    run(ds, split, cfg, out)
