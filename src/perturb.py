"""Perturbaciones de monitorización (ruido, sesgo, huecos, desfases, picos) e inyección TRANSPARENTE de fallos.

Todas operan sobre matrices de perfiles diarios (n, 24) y se aplican SOLO a los datos de evaluación (o a una copia
del entrenamiento para el modelo jerárquico); nunca modifican el dataset original.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- ruido de monitorización


def gaussian_noise(X, pct, rng):
    """Ruido aditivo N(0, (pct·media diaria del perfil)²)."""
    return X + rng.normal(0.0, 1.0, X.shape) * (pct * X.mean(axis=1, keepdims=True))


def calibration_bias(X, bias):
    """Sesgo multiplicativo de calibración (el mismo para todo el sensor)."""
    return X * (1.0 + bias)


def random_missing(X, pct, rng):
    Xn = X.astype(float).copy()
    Xn[rng.random(X.shape) < pct] = np.nan
    return Xn


def contiguous_outage(X, hours, rng):
    """Un corte contiguo de `hours` horas por día en una posición aleatoria."""
    start = rng.integers(0, 24 - hours + 1, size=len(X))
    cols = np.arange(24)[None, :]
    mask = (cols >= start[:, None]) & (cols < (start + hours)[:, None])
    Xn = X.astype(float).copy()
    Xn[mask] = np.nan
    return Xn


def timestamp_shift(X, realization_ids, shift):
    """Desplaza ±`shift` horas la serie continua de cada realización (las filas deben estar contiguas y ordenadas por día)."""
    out = X.copy()
    ids = np.asarray(realization_ids)
    change = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1], True])
    for a, b in zip(change[:-1], change[1:]):
        out[a:b] = np.roll(X[a:b].reshape(-1), shift).reshape(-1, 24)
    return out


def spikes_dropouts(X, rate, rng):
    """Con probabilidad `rate` por hora: pico (×2–4) o caída a cero (50/50)."""
    out = X.copy()
    hit = rng.random(X.shape) < rate
    spike = rng.random(X.shape) < 0.5
    out[hit & spike] *= rng.uniform(2.0, 4.0, X.shape)[hit & spike]
    out[hit & ~spike] = 0.0
    return out


# --------------------------------------------------------------------------- imputación (sin fuga de test)


def impute_linear(Xnan):
    out = pd.DataFrame(Xnan).interpolate(axis=1, limit_direction="both").to_numpy()
    bad = np.isnan(out)
    if bad.any():
        out[bad] = np.nanmean(Xnan, axis=1)[np.where(bad)[0]]
    return out


def fit_month_hour_median(X_train, months_train):
    """Mediana por (mes, hora) calculada SOLO con entrenamiento. Devuelve tabla (13, 24) (índice = mes)."""
    tab = np.tile(np.median(X_train, axis=0), (13, 1))
    for m in range(1, 13):
        sel = months_train == m
        if sel.any():
            tab[m] = np.median(X_train[sel], axis=0)
    return tab


def impute_median(Xnan, months, table):
    out = Xnan.copy()
    miss = np.isnan(out)
    out[miss] = table[np.asarray(months)][miss]
    return out


def augment_missing(X, rng):
    """Copia de entrenamiento con huecos aleatorios (tasa por fila ∈ {0, 1, 5, 10 %}) y/o corte de 1–6 h."""
    Xn = X.astype(float).copy()
    rate = rng.choice([0.0, 0.01, 0.05, 0.10], size=len(X))
    Xn[rng.random(X.shape) < rate[:, None]] = np.nan
    hrs = rng.choice([0, 0, 1, 3, 6], size=len(X))
    start = rng.integers(0, 19, size=len(X))
    cols = np.arange(24)[None, :]
    Xn[(cols >= start[:, None]) & (cols < (start + hrs)[:, None]) & (hrs[:, None] > 0)] = np.nan
    return Xn


# --------------------------------------------------------------------------- fallos (separados de las intervenciones)

MEASUREMENT = ("stuck_sensor", "calibration_drift")
PHYSICAL = ("valve_failure", "hx_degradation", "leakage")
FAULTS = MEASUREMENT + PHYSICAL
GROUP = {**{k: "measurement_anomaly" for k in MEASUREMENT}, **{k: "physical_fault" for k in PHYSICAL}}


def inject_fault(kind: str, X: np.ndarray, prog: np.ndarray, rng) -> np.ndarray:
    """Inyecta un mecanismo de fallo en los perfiles de UNA realización.

    X: (n, 24) en orden temporal; prog: (n,) avance 0..1 dentro del tramo afectado.
    Mecanismos (transparentes, sin calibrar con datos reales):
    - stuck_sensor: en el 30 % de los días, el sensor se congela 6–20 h en el último valor leído.   [medida]
    - calibration_drift: sesgo multiplicativo que crece linealmente hasta ±(5–20 %).                 [medida]
    - valve_failure: en el 25 % de los días, una ventana de 4–12 h se reduce a 0–30 % de la carga.   [físico]
    - hx_degradation: pérdida de capacidad del intercambiador: se recortan los picos a un tope que baja hasta un
      10–30 % por debajo del máximo de la realización.                                                [físico]
    - leakage: aumento aditivo de carga base (5–20 % de la media de la realización) a partir de un instante.[físico]
    """
    out = X.copy()
    n = len(X)
    if kind == "stuck_sensor":
        for i in np.flatnonzero(rng.random(n) < 0.30):
            h0 = int(rng.integers(1, 18))
            L = int(rng.integers(6, 21))
            out[i, h0:min(24, h0 + L)] = out[i, h0 - 1]
    elif kind == "calibration_drift":
        s = rng.choice([-1, 1]) * rng.uniform(0.05, 0.20)
        out = out * (1.0 + s * prog[:, None])
    elif kind == "valve_failure":
        for i in np.flatnonzero(rng.random(n) < 0.25):
            h0 = int(rng.integers(0, 13))
            L = int(rng.integers(4, 13))
            out[i, h0:min(24, h0 + L)] *= rng.uniform(0.0, 0.3)
    elif kind == "hx_degradation":
        d = rng.uniform(0.10, 0.30)
        cap = X.max() * (1.0 - d * prog)
        out = np.minimum(out, cap[:, None])
    elif kind == "leakage":
        c = rng.uniform(0.05, 0.20) * X.mean()
        t0 = rng.uniform(0.2, 0.6)
        out = out + c * (prog >= t0)[:, None]
    else:
        raise ValueError(kind)
    return out
