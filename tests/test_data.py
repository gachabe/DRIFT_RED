import shutil

import numpy as np

from src.data import daily_profiles, read_heat_load, list_series, find_benchmark_dir
from src.validation import check_dataset
from tests.synthetic import make_synthetic


def test_validation_passes(synth_root):
    assert check_dataset(synth_root) is True


def test_validation_fails_when_folder_missing(tmp_path):
    root = make_synthetic(tmp_path, n_real=1)
    shutil.rmtree(root / "benchmark_data" / "heat_load" / "ns_h")
    assert check_dataset(root) is False


def test_daily_profiles(synth_root):
    s = list_series(find_benchmark_dir(synth_root))
    X, days = daily_profiles(read_heat_load(s["path"].iloc[0]))
    assert X.shape == (1461, 24) and np.isfinite(X).all() and len(days) == 1461


def test_dataset_labels(dataset_and_split):
    ds, _ = dataset_and_split
    assert set(ds.meta["phase"]) == {"pre", "transition", "developed"}
    assert set(ds.meta["scenario"]) == {"EXP", "REF", "NS"}
    assert ds.X.shape == (len(ds.meta), 24)


def test_phases_use_onset_and_last_step(dataset_and_split):
    ds, _ = dataset_and_split
    m = ds.meta.drop_duplicates("realization").set_index("realization")
    assert (m.loc[m["scenario"] == "EXP", "transition_end"] == 180).all()   # súbito: ventana fija
    assert (m.loc[m["scenario"] != "EXP", "transition_end"] == 360).all()   # onset -> último escalón
    r = ds.meta[ds.meta["scenario"] == "REF"]
    assert r["days_since_start"].min() < 0 and (r["phase"] == "transition").any()
