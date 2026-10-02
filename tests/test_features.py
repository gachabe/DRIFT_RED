from src.features import split_realizations, rows


def test_split_disjoint_stratified_deterministic(dataset_and_split):
    ds, _ = dataset_and_split
    a = split_realizations(ds.realizations, 0.8, 42)
    b = split_realizations(ds.realizations, 0.8, 42)
    assert a == b
    assert set(a.train).isdisjoint(a.test)
    assert set(a.train) | set(a.test) == set(ds.realizations["realization"])
    tst = ds.realizations[ds.realizations["realization"].isin(a.test)]
    assert set(tst["scenario"]) == {"EXP", "REF", "NS"}


def test_rows_respect_phase_and_ids(dataset_and_split):
    ds, split = dataset_and_split
    X, y = rows(ds, split.test, ["developed"])
    assert X.shape[1] == 24 and len(X) == len(y) > 0
