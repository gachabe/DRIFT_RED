import pytest

from tests.synthetic import make_synthetic


@pytest.fixture(scope="session")
def synth_root(tmp_path_factory):
    return make_synthetic(tmp_path_factory.mktemp("synth"), n_real=4)


@pytest.fixture(scope="session")
def cfg(synth_root, tmp_path_factory):
    return {
        "mode": "demo", "seed": 42, "dataset": "synthetic",
        "results_dir": str(tmp_path_factory.mktemp("results")),
        "cache_dir": str(tmp_path_factory.mktemp("cache")),
        "data": {"root": str(synth_root), "scenarios": ["EXP", "REF", "NS"],
                 "severities": ["L", "M", "H"], "realizations": 3},
        "split": {"train_fraction": 0.8},
        "phases": {"transition_days": 180},
        "rom": {"components": [2, 3, 5], "focus_components": [2, 5]},
        "figures": {"confusion_components": [2, 5], "progress_components": [2, 5]},
        "revision": {"n_seeds": 2, "n_seeds_secondary": 2, "n_seeds_perturb": 1, "bootstrap_resamples": 30,
                     "progress_bins": 5, "hdd_base_c": 15.5, "hellinger_bins": 10, "hellinger_bootstrap": 5,
                     "perturb_max_test_rows": 2000, "rf_cv_folds": 2, "rf_cv_rows": 2000, "rf_test_mode": "focused",
                     "bench_repeats": 2, "bench_pred_profiles": 500, "exp_transition_days_alt": [90, 360],
                     "rf_grid": {"n_estimators": [20, 30], "max_depth": [None, 8], "min_samples_leaf": [1, 5], "max_features": ["sqrt", 1.0]}},
        "ml": {"train_phase": "developed", "max_train_rows": None},
        "models": ["logistic_regression", "random_forest"],
    }


@pytest.fixture(scope="session")
def dataset_and_split(cfg, tmp_path_factory):
    from src.features import prepare
    return prepare(cfg, tmp_path_factory.mktemp("prep"))
