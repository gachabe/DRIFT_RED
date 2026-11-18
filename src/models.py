"""Los dos modelos base: regresión logística y random forest."""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

RF_DEFAULT = dict(n_estimators=200, max_depth=None, min_samples_leaf=1, max_features="sqrt")


def make_model(name: str, seed: int, params: dict | None = None):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    if name == "random_forest":
        p = {**RF_DEFAULT, **(params or {})}
        return RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=seed, **p)
    raise ValueError(f"Modelo desconocido: {name}")
