"""Repite split + ROM + ML con varias semillas y agrega media ± desviación."""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import pandas as pd

from src.data import build_dataset, load_config
from src.evaluation import fig_components_seeds, to_markdown
from src.features import rows, split_realizations
from src.rom import fit_representation

METRICS = ("macro_f1", "balanced_accuracy", "f1_NS", "f1_EXP", "f1_REF", "recon_rmse")


def main(cfg: dict, n_seeds: int = 10) -> pd.DataFrame:
    ml = importlib.import_module("experiments.03_ml")
    out = Path(cfg.get("results_dir", f"results/{cfg['mode']}"))
    out.mkdir(parents=True, exist_ok=True)
    ds = build_dataset(cfg)  # se carga una sola vez
    tf = cfg.get("split", {}).get("train_fraction", 0.8)

    runs = []
    for i in range(n_seeds):
        seed = cfg["seed"] + i
        split = split_realizations(ds.realizations, tf, seed)
        X_train, _ = rows(ds, split.train)
        reps = [fit_representation(X_train, None)] + [fit_representation(X_train, r) for r in cfg["rom"]["components"]]
        res = ml.run(ds, split, reps, {**cfg, "seed": seed}, out, save=False)
        runs.append(res.assign(seed=seed))
        print(f"semilla {seed} ({i + 1}/{n_seeds}) hecha")
    raw = pd.concat(runs, ignore_index=True)
    raw.to_csv(out / "results_seeds_raw.csv", index=False)

    keys = ["model", "phase", "representation", "components"]
    g = raw.groupby(keys, sort=False)[list(METRICS)].agg(["mean", "std"])
    g.columns = [f"{m}_{s}" for m, s in g.columns]
    agg = g.reset_index()
    agg.to_csv(out / "results_seeds_summary.csv", index=False)

    parts = []
    for (model, phase), a in agg.groupby(["model", "phase"], sort=False):
        t = a[["representation", "components"]].copy()
        for m in ("macro_f1", "balanced_accuracy", "f1_NS"):
            t[m] = [f"{u:.3f} ± {v:.3f}" for u, v in zip(a[f"{m}_mean"], a[f"{m}_std"])]
        t["recon_rmse"] = a["recon_rmse_mean"].to_numpy()
        parts.append(f"### {model} — {phase} ({n_seeds} semillas)\n\n" + to_markdown(t))
    (out / "results_seeds_summary.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    fig_components_seeds(agg, out)
    print(f"\nListo. Resultados en {out}/ (results_seeds_summary.md, 06_*.png)")
    return agg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-seeds", type=int, default=10)
    a = ap.parse_args()
    main(load_config(a.config), a.n_seeds)
