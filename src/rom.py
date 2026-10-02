"""Representación reducida: StandardScaler + PCA ajustados solo con entrenamiento."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class Representation:
    name: str
    n_components: int  # 24 = crudo
    scaler: StandardScaler
    pca: PCA | None

    def transform(self, X):
        Z = self.scaler.transform(X)
        Z = Z if self.pca is None else self.pca.transform(Z)
        assert Z.shape[1] == self.n_components
        return Z

    def reconstruct(self, X):
        if self.pca is None:
            return X
        return self.scaler.inverse_transform(self.pca.inverse_transform(self.pca.transform(self.scaler.transform(X))))

    def rmse(self, X) -> float:
        if self.pca is None:
            return float("nan")
        r = float(np.sqrt(np.mean((X - self.reconstruct(X)) ** 2)))
        assert np.isfinite(r)
        return r

    @property
    def explained_variance(self) -> float:
        return float("nan") if self.pca is None else float(self.pca.explained_variance_ratio_.sum())


def fit_representation(X_train, n_components: int | None) -> Representation:
    """n_components=None -> perfil crudo (solo escalado)."""
    assert X_train.shape[1] == 24 and np.isfinite(X_train).all()
    scaler = StandardScaler().fit(X_train)
    if n_components is None:
        return Representation("Raw", 24, scaler, None)
    pca = PCA(n_components=n_components, svd_solver="full").fit(scaler.transform(X_train))
    return Representation("PCA", n_components, scaler, pca)


def rmse_curve(rep24: Representation, X, max_r: int = 24) -> np.ndarray:
    """RMSE de reconstrucción (unidades originales) truncando un PCA de 24 componentes."""
    Zs = rep24.pca.transform(rep24.scaler.transform(X))
    comps, mean = rep24.pca.components_, rep24.pca.mean_
    out = []
    for r in range(1, max_r + 1):
        Xs_hat = Zs[:, :r] @ comps[:r] + mean
        out.append(np.sqrt(np.mean((X - rep24.scaler.inverse_transform(Xs_hat)) ** 2)))
    out = np.array(out)
    assert np.all(np.diff(out) <= 1e-6 * max(1.0, out[0])), "RMSE debería decrecer con más componentes"
    return out
