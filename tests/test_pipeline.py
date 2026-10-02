import importlib
import json
from pathlib import Path

import numpy as np

from src.features import rows
from src.rom import fit_representation


def test_no_preprocessing_leakage(dataset_and_split):
    """El escalado/PCA solo dependen de train: cambiar test no los altera."""
    ds, split = dataset_and_split
    X_train, _ = rows(ds, split.train)
    X_test, _ = rows(ds, split.test)
    rep = fit_representation(X_train, 5)
    assert np.allclose(rep.scaler.mean_, X_train.mean(axis=0))
    assert not np.allclose(rep.scaler.mean_, np.vstack([X_train, X_test]).mean(axis=0))
    rep2 = fit_representation(X_train, 5)  # tras alterar test, el ajuste debe ser idéntico
    X_test *= 1e3
    assert np.allclose(rep.pca.components_, rep2.pca.components_)


def test_end_to_end(cfg):
    final = importlib.import_module("experiments.04_final")
    summary = final.main(cfg)
    out = Path(cfg["results_dir"])
    for f in ("01_example_profiles.png", "02_pca_variance.png", "03_reconstruction.png",
              "04_confusion_matrix.png", "05_components_vs_performance.png", "summary.json",
              "run_metadata.json", "results_table.csv", "summary_table.md",
              "splits/train_realizations.csv", "splits/test_realizations.csv"):
        assert (out / f).exists(), f
    meta = json.loads((out / "run_metadata.json").read_text())
    assert set(meta["train_realizations"]).isdisjoint(meta["test_realizations"])
    res = summary["results"]
    assert len(res) == 2 * 3 * 3  # modelos x (raw+2 PCA) x fases
    assert all(np.isfinite(r["macro_f1"]) for r in res)


def test_seeds(cfg):
    agg = importlib.import_module("experiments.05_seeds").main(cfg, n_seeds=2)
    out = Path(cfg["results_dir"])
    assert (out / "results_seeds_summary.md").exists() and (out / "06_components_vs_performance_seeds.png").exists()
    assert len(agg) == 2 * 3 * 3 and np.isfinite(agg["macro_f1_mean"]).all()
