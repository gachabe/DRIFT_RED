"""Los dos modelos base: regresión logística y random forest."""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def make_model(name: str, seed: int):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=seed
        )
    raise ValueError(f"Modelo desconocido: {name}")
