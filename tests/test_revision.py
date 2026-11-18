"""Tests P15 (guía de revisión): rejilla, NRMSE, normalización, tiempo, progreso, celdas, BL, regímenes PCA,
consistencia con el manuscrito y artefactos de figuras."""
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import recall_score

from src.analysis import (
    normalize_profiles, nrmse_table, prf_table, profile_errors, progress_bins, window_masks,
)
from src.consistency import check_numbers
from src.data import SCENARIOS
from src.features import rows, split_realizations
from src.rom import fit_representation
from src.runtime import load_dataset

seeds_mod = lambda: importlib.import_module("experiments.05_seeds")
prog_mod = lambda: importlib.import_module("experiments.06_progress")
ml_mod = lambda: importlib.import_module("experiments.03_ml")
rev_mod = lambda: importlib.import_module("experiments.07_revision")


@pytest.fixture(scope="module")
def seeds_done(cfg):
    seeds_mod().main(cfg, n_seeds=2)
    prog_mod().main(cfg)
    return Path(cfg["results_dir"])


def test_1_component_grid_once_per_model_phase(cfg, seeds_done):
    raw = pd.read_csv(seeds_done / "results_seeds_raw.csv")
    assert set(raw["components"]) == {24, *cfg["rom"]["components"]}
    assert not raw.duplicated(["seed", "model", "phase", "representation", "components"]).any()
    cnt = raw.groupby(["seed", "model", "phase"]).size()
    assert (cnt == len(cfg["rom"]["components"]) + 1).all()
    assert set(raw[raw["components"] == 24]["representation"]) == {"Raw"}  # el crudo no se etiqueta como PCA-24


def test_2_nrmse_finite_and_unit_invariant(dataset_and_split):
    ds, split = dataset_and_split
    X, _ = rows(ds, split.train)
    Xt, _ = rows(ds, split.test)
    e1 = profile_errors(fit_representation(X, 24), Xt, [2, 5])
    e2 = profile_errors(fit_representation(X * 1e-3, 24), Xt * 1e-3, [2, 5])  # W -> kW
    for k in (2, 5):
        assert np.isfinite(e1[k]["nrmse_sd"]).all()
        assert np.allclose(e1[k]["nrmse_sd"], e2[k]["nrmse_sd"], rtol=1e-6)
        assert np.allclose(e1[k]["rmse_w"] * 1e-3, e2[k]["rmse_w"], rtol=1e-6)
    t = nrmse_table(ds.meta[ds.meta["realization"].isin(split.test)].reset_index(drop=True), Xt, e1)
    assert np.isfinite(t["nrmse_sd_subset"]).all() and {"scenario", "severity", "phase", "components"} <= set(t.columns)


def test_3_normalization_mean_sum_and_guards(dataset_and_split):
    ds, _ = dataset_and_split
    X = ds.X[:500].copy()
    X[0] = 0.0  # perfil de ceros: no debe producir NaN
    for mode, stat in (("daily_mean_normalized", lambda a: a.mean(1)), ("daily_energy_normalized", lambda a: a.sum(1))):
        Z = normalize_profiles(X, mode)
        assert np.isfinite(Z).all() and np.allclose(stat(Z[1:]), 1.0)
    assert np.isfinite(normalize_profiles(X, "daily_zscore_shape")).all()
    from sklearn.preprocessing import StandardScaler  # x/mean y x/sum son equivalentes tras escalar
    a = StandardScaler().fit_transform(normalize_profiles(X[1:], "daily_mean_normalized"))
    b = StandardScaler().fit_transform(normalize_profiles(X[1:], "daily_energy_normalized"))
    assert np.allclose(a, b)


def test_4_weather_merge_preserves_rows_and_dates(cfg):
    base = load_dataset(cfg, weather=False)
    w = load_dataset(cfg, weather=True)
    assert len(w.meta) == len(base.meta) and (w.meta["date"].to_numpy() == base.meta["date"].to_numpy()).all()
    assert (w.meta["realization"].to_numpy() == base.meta["realization"].to_numpy()).all()
    assert w.meta["temp_mean"].notna().all() and (w.meta["hdd"] >= 0).all() and (w.meta["temp_min"] <= w.meta["temp_max"]).all()


def test_5_progress_bins_only_transition_and_bounded(dataset_and_split):
    ds, _ = dataset_and_split
    b = progress_bins(ds.meta, 5)
    tr = (ds.meta["phase"] == "transition").to_numpy()
    assert (b[~tr] == -1).all() and ((b[tr] >= 0) & (b[tr] <= 4)).all() and set(b[tr]) == set(range(5))


def test_6_class_severity_cells_and_support(cfg, seeds_done):
    t = pd.read_csv(seeds_done / "progress_class_severity_all_splits.csv")
    assert list(t.columns[:12]) == ["seed", "model", "representation", "scenario", "severity", "phase", "progress_bin",
                                    "precision", "recall", "f1", "n_profiles", "n_realizations"]
    assert set(t["scenario"]) == set(SCENARIOS) and set(t["severity"]) == {"L", "M", "H"}
    assert set(t[t["phase"] == "transition"]["progress_bin"]) == {"0-20%", "20-40%", "40-60%", "60-80%", "80-100%"}
    n = t.groupby(["seed", "model", "representation", "severity", "phase", "progress_bin"]).size()
    assert (n == 3).all()  # una fila por clase en cada celda
    assert (t["n_realizations"] >= 1).all() and t["recall"].between(0, 1).all() and t["precision"].dropna().between(0, 1).all()


def test_6b_prf_table_matches_sklearn(dataset_and_split):
    ds, split = dataset_and_split
    m = ds.meta[ds.meta["realization"].isin(split.test)].reset_index(drop=True)
    rng = np.random.default_rng(0)
    pred = np.where(rng.random(len(m)) < 0.7, m["scenario"], rng.choice(list(SCENARIOS), len(m)))
    t = prf_table(m, pred, ["phase"])
    for ph in ("pre", "developed"):
        s = (m["phase"] == ph).to_numpy()
        ref = recall_score(m["scenario"][s], pred[s], labels=list(SCENARIOS), average=None)
        got = t[t["phase"] == ph].set_index("scenario").loc[list(SCENARIOS), "recall"].to_numpy()
        assert np.allclose(ref, got)


def test_7_baseline_never_enters_training(cfg, dataset_and_split):
    ds, split = dataset_and_split
    Xbl = np.full((50, 24), 1e12)  # valores imposibles de confundir con los de entrenamiento
    d = pd.date_range("2020-01-01", periods=50)
    X_fit, _ = rows(ds, split.train)
    reps = [fit_representation(X_fit, None), fit_representation(X_fit, 2)]
    a = ml_mod().run(ds, split, reps, cfg, Path(cfg["results_dir"]), save=False)
    coll = {}
    b = ml_mod().run(ds, split, reps, cfg, Path(cfg["results_dir"]), save=False, collect=coll, bl=(Xbl, d))
    pd.testing.assert_frame_equal(a.drop(columns="fit_seconds"), b.drop(columns="fit_seconds"))
    assert coll["bl"] and all(r["n_days"] in (50, ) or r["source"] != "BL" for r in coll["bl"])
    assert reps[0].scaler.mean_.max() < 1e11 and ds.X.max() < 1e11
    assert not any("baseline" in r for r in ds.meta["realization"])


def test_7b_train_idx_matches_train_arrays(cfg, dataset_and_split):
    ds, split = dataset_and_split
    c = {**cfg, "ml": {"train_phase": "developed", "max_train_rows": 500}}
    idx = rev_mod().train_idx(ds, split, c, 42)
    X_tr, y_tr = ml_mod().train_arrays(ds, split, c, 42)
    assert np.array_equal(ds.X[idx], X_tr) and np.array_equal(ds.meta["scenario"].to_numpy()[idx], y_tr)


def test_8_pca_regimes_fit_on_train_only(dataset_and_split):
    ds, split = dataset_and_split
    for phases in (["pre"], ["developed"], None):
        X, _ = rows(ds, split.train, phases)
        mask = ds.meta["realization"].isin(split.train).to_numpy().copy()
        if phases:
            mask &= ds.meta["phase"].isin(phases).to_numpy()
        assert not ds.meta.loc[mask, "realization"].isin(split.test).any()
        rep = fit_representation(X, 5)
        assert np.allclose(rep.scaler.mean_, ds.X[mask].mean(axis=0))
    pre = fit_representation(rows(ds, split.train, ["pre"])[0], 5)
    allr = fit_representation(rows(ds, split.train)[0], 5)
    assert not np.allclose(pre.scaler.mean_, allr.scaler.mean_)


def test_9_manuscript_numbers_check(seeds_done, tmp_path):
    t = pd.read_csv(seeds_done / "results_table.csv") if (seeds_done / "results_table.csv").exists() else None
    if t is None:
        importlib.import_module("experiments.04_final")
        pytest.skip("results_table.csv no generado todavía")
    r = t[t["phase"] == "transition"].head(4)
    ref = pd.DataFrame({"source": "seed42", "model": r["model"], "phase": r["phase"], "representation": r["representation"],
                        "components": r["components"], "metric": "macro_f1", "value": r["macro_f1"].round(3), "tol": 0.0006})
    ref.to_csv(tmp_path / "ref.csv", index=False)
    assert check_numbers(seeds_done, tmp_path / "ref.csv")["ok"].all()
    ref["value"] += 0.01
    ref.to_csv(tmp_path / "bad.csv", index=False)
    assert not check_numbers(seeds_done, tmp_path / "bad.csv")["ok"].any()


def test_10_windows_masks_consistent(dataset_and_split):
    ds, _ = dataset_and_split
    m = window_masks(ds.meta, followups=(90,), buffers=(7,), matched=(180,))
    d, te = ds.meta["days_since_start"].to_numpy(), ds.meta["transition_end"].to_numpy()
    assert (d[m["followup_90d_after_endpoint"]] >= te[m["followup_90d_after_endpoint"]]).all()
    assert (d[m["matched_first_180d_from_onset"]] < 180).all() and (d[m["matched_first_180d_from_onset"]] >= 0).all()
    assert m["transition_excl_buffer_7d"].sum() < m["current:transition"].sum()


def test_11_cheap_revision_stages_and_figure_artifacts(cfg, seeds_done, dataset_and_split):
    ds = load_dataset(cfg)
    out = Path(cfg["results_dir"])
    importlib.import_module("experiments.04_final").main(cfg)
    rev = rev_mod()
    rev.stage_pca_interpretation(cfg, ds, out)
    rev.stage_hellinger(cfg, ds, out)
    rev.stage_workflow(cfg, None, out)
    pngs = list(out.rglob("*.png")) + list(out.rglob("*.svg")) + list(out.rglob("*.pdf"))
    assert len(pngs) >= 20
    for f in pngs:
        t = f.with_suffix(".txt")
        assert t.exists() and len(t.read_text(encoding="utf-8")) > 80, f"falta descripción de {f.name}"
    for name in ("14_rf_pca_curve.png", "07_progress.png", "08_class_progress_random_forest.png", "04_confusion_matrix.png"):
        assert (out / name).exists() and (out / name.replace(".png", ".txt")).exists()
    assert (out / "figures_manifest.json").exists()
    assert (out / "confusion_counts.csv").exists()
