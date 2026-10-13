"""Regenera las figuras sin reentrenar: usa los CSV/JSON ya guardados (y, con --data, recarga las series)."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_config
from src.evaluation import (
    fig_components_seeds,
    fig_components_vs_performance,
    fig_confusions,
    fig_progress,
)
from src.features import prepare


def main(cfg: dict, data: bool = False, confusion: bool = False) -> None:
    out = Path(cfg.get("results_dir", f"results/{cfg['mode']}"))
    done = []
    if (out / "results_table.csv").exists():
        fig_components_vs_performance(pd.read_csv(out / "results_table.csv"), out)
        done.append("05")
    if (out / "results_seeds_summary.csv").exists():
        fig_components_seeds(pd.read_csv(out / "results_seeds_summary.csv"), out)
        done.append("06")
    if (out / "progress_by_bin.csv").exists():
        fig_progress(pd.read_csv(out / "progress_by_bin.csv"), out)
        done.append("07")
    if (out / "confusion_matrices.json").exists():
        raw = json.loads((out / "confusion_matrices.json").read_text(encoding="utf-8"))
        fig_confusions({tuple(k.split("|")): np.array(v) for k, v in raw.items()}, out)
        done.append("04")
    if data or confusion:
        ds, split = prepare(cfg, out)  # recarga las series (~1.5 min con full)
        reps = importlib.import_module("experiments.02_rom").run(ds, split, cfg, out)  # figuras 02 y 03
        importlib.import_module("experiments.01_eda").run(ds, out)  # figura 01
        done += ["01", "02", "03"]
        if confusion:  # reentrena una vez y guarda las matrices (figuras 04 y 05)
            importlib.import_module("experiments.03_ml").run(ds, split, reps, cfg, out)
            done += ["04", "05"]
    print("Figuras regeneradas:", ", ".join(sorted(set(done))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", action="store_true", help="regenera también 01, 02 y 03 (recarga las series)")
    ap.add_argument("--confusion", action="store_true", help="reentrena una vez para regenerar 04 (y guarda las matrices)")
    a = ap.parse_args()
    main(load_config(a.config), a.data, a.confusion)
