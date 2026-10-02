"""Sanity plots: perfiles de ejemplo por escenario."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data import load_config
from src.evaluation import fig_example_profiles
from src.features import prepare


def run(ds, outdir: Path) -> None:
    fig_example_profiles(ds, outdir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    out = Path(cfg.get("results_dir", f"results/{cfg['mode']}"))
    out.mkdir(parents=True, exist_ok=True)
    ds, _ = prepare(cfg, out)
    run(ds, out)
