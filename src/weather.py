"""Tiempo (weather.csv): agregación diaria, grados-día de calefacción, estación y fusión por fecha."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data import find_benchmark_dir

WEATHER_COLS = ("temp_mean", "temp_min", "temp_max", "hdd")
SEASONS = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
           6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}  # hemisferio norte


def has_weather(meta: pd.DataFrame) -> bool:
    return all(c in meta.columns for c in WEATHER_COLS)


def _pick_cols(df: pd.DataFrame, wcfg: dict):
    tcol = wcfg.get("time_col") or next(
        (c for c in df.columns if any(k in str(c).lower() for k in ("timestamp", "datetime", "time", "date"))), df.columns[0])
    cand = [c for c in df.columns if c != tcol and "temp" in str(c).lower() and "dew" not in str(c).lower()]
    pref = [c for c in cand if any(k in str(c).lower() for k in ("out", "air", "ambient", "t_"))]
    xcol = wcfg.get("temp_col") or (pref or cand or [None])[0]
    return tcol, xcol


def daily_weather(path, wcfg: dict | None = None, hdd_base: float = 15.5) -> pd.DataFrame | None:
    """Índice = día (UTC naive, igual que los perfiles). Columnas: temp_mean/min/max y hdd."""
    wcfg = wcfg or {}
    df = pd.read_csv(path)
    tcol, xcol = _pick_cols(df, wcfg)
    if xcol is None:
        return None
    ts = pd.to_datetime(df[tcol])
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    t = pd.to_numeric(df[xcol], errors="coerce").astype(float)
    if np.nanmean(t) > 150:  # Kelvin
        t = t - 273.15
    g = pd.DataFrame({"day": ts.dt.floor("D"), "t": t}).groupby("day")["t"]
    out = pd.DataFrame({"temp_mean": g.mean(), "temp_min": g.min(), "temp_max": g.max()})
    out["hdd"] = np.maximum(0.0, hdd_base - out["temp_mean"])
    out.index = pd.DatetimeIndex(out.index)
    return out


def add_weather(meta: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Fusiona el tiempo diario por fecha (left join: no cambia filas ni orden). Si no hay tiempo, devuelve meta tal cual."""
    if has_weather(meta):
        return meta
    rc = cfg.get("revision") or {}
    wcfg = rc.get("weather") or {}
    try:
        bench = find_benchmark_dir(cfg["data"].get("root", "data/raw"))
        path = bench / wcfg.get("file", "weather.csv")
        if not path.exists():
            return meta
        dw = daily_weather(path, wcfg, rc.get("hdd_base_c", 15.5))
    except Exception as e:
        print("aviso: no se pudo leer el tiempo:", e)
        return meta
    if dw is None:
        print("aviso: weather.csv sin columna de temperatura reconocible; fija revision.weather.temp_col")
        return meta
    n = len(meta)
    m = meta.merge(dw, how="left", left_on="date", right_index=True)
    assert len(m) == n and (m["date"].to_numpy() == meta["date"].to_numpy()).all(), "la fusión del tiempo cambió filas"
    miss = m["temp_mean"].isna().mean()
    if miss > 0:
        print(f"aviso: {miss:.2%} de las filas sin tiempo para su fecha")
    m["month"] = pd.DatetimeIndex(m["date"]).month.astype(np.int8)
    m["season"] = m["month"].map(SEASONS)
    m["year"] = pd.DatetimeIndex(m["date"]).year.astype(np.int16)
    m["doy"] = pd.DatetimeIndex(m["date"]).dayofyear.astype(np.int16)
    return m.reset_index(drop=True)


def add_calendar(meta: pd.DataFrame) -> pd.DataFrame:
    """Columnas de calendario aunque no haya tiempo."""
    if "month" in meta.columns:
        return meta
    m = meta.copy()
    d = pd.DatetimeIndex(m["date"])
    m["month"], m["season"] = d.month.astype(np.int8), d.month.map(SEASONS)
    m["year"], m["doy"] = d.year.astype(np.int16), d.dayofyear.astype(np.int16)
    return m


def temperature_bins(temp_train: np.ndarray, temp: np.ndarray) -> np.ndarray:
    """cold/mild/warm con terciles definidos SOLO con la temperatura de entrenamiento."""
    q1, q2 = np.nanquantile(temp_train, [1 / 3, 2 / 3])
    return np.where(np.isnan(temp), "NA", np.where(temp <= q1, "cold", np.where(temp <= q2, "mild", "warm")))
