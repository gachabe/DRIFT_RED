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
        "data": {"root": str(synth_root), "scenarios": ["EXP", "REF", "NS"],
                 "severities": ["L", "M", "H"], "realizations": 3},
        "split": {"train_fraction": 0.8},
        "phases": {"transition_days": 180},
        "rom": {"components": [2, 5]},
        "ml": {"train_phase": "developed", "max_train_rows": None},
        "models": ["logistic_regression", "random_forest"],
    }


@pytest.fixture(scope="session")
def dataset_and_split(cfg, tmp_path_factory):
    from src.features import prepare
    return prepare(cfg, tmp_path_factory.mktemp("prep"))
