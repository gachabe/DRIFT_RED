"""Split por realizaciones completas y selección de filas por fase."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import Dataset, build_dataset


@dataclass
class Split:
    train: list
    test: list


def split_realizations(realizations: pd.DataFrame, train_fraction: float, seed: int) -> Split:
    """Split estratificado por (escenario, severidad) sobre realizaciones completas."""
    rng = np.random.default_rng(seed)
    train, test = [], []
    for _, g in realizations.groupby(["scenario", "severity"], sort=True):
        ids = np.array(sorted(g["realization"]))
        if len(ids) < 2:
            raise ValueError("Hacen falta >=2 realizaciones por (escenario, severidad) para hacer split")
        rng.shuffle(ids)
        n_test = min(max(1, int(round((1 - train_fraction) * len(ids)))), len(ids) - 1)
        test += ids[:n_test].tolist()
        train += ids[n_test:].tolist()
    assert set(train).isdisjoint(test)
    return Split(sorted(train), sorted(test))


def rows(ds: Dataset, ids, phases=None):
    """(X, y_scenario) de las realizaciones `ids` en las fases indicadas (None = todas)."""
    mask = ds.meta["realization"].isin(list(ids)).to_numpy().copy()
    if phases is not None:
        mask = mask & ds.meta["phase"].isin(list(phases)).to_numpy()
    return ds.X[mask], ds.meta.loc[mask, "scenario"].to_numpy()


def prepare(cfg: dict, outdir: Path):
    """Construye el dataset, hace el split y guarda los IDs."""
    ds = build_dataset(cfg)
    split = split_realizations(ds.realizations, cfg.get("split", {}).get("train_fraction", 0.8), cfg["seed"])
    sdir = Path(outdir) / "splits"
    sdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"realization": split.train}).to_csv(sdir / "train_realizations.csv", index=False)
    pd.DataFrame({"realization": split.test}).to_csv(sdir / "test_realizations.csv", index=False)
    return ds, split
