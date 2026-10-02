"""Carga de DHDrift, perfiles diarios de 24 h y etiquetas relativas al evento."""
from __future__ import annotations

import argparse
import hashlib
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

SCENARIOS = ("EXP", "REF", "NS")
SEVERITIES = ("L", "M", "H")
PHASES = ("pre", "transition", "developed")

ZENODO_URL = "https://zenodo.org/records/18128257/files/dhdrift_dataset.zip?download=1"
ZENODO_MD5 = "6b681bdebae41174d6d833e2d50e1a3f"

FILE_PATTERNS = ("file", "name", "path", "series", "id")
START_PATTERNS = ("drift_start", "start", "onset", "occurrence", "drift_time", "change", "drift", "time", "date")
START_EXCLUDE = ("end", "stop", "file", "name", "path", "scenario", "severity", "type", "id")


def load_config(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def download(dest="data/raw") -> None:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    zpath = dest / "dhdrift_dataset.zip"
    if not zpath.exists():
        print(f"Descargando {ZENODO_URL} (~570 MB)...")
        urllib.request.urlretrieve(ZENODO_URL, zpath)
    md5 = hashlib.md5(zpath.read_bytes()).hexdigest()
    assert md5 == ZENODO_MD5, f"md5 inesperado: {md5}"
    with zipfile.ZipFile(zpath) as z:
        z.extractall(dest)
    print("OK: dataset extraído en", dest)


# --------------------------------------------------------------------------- estructura


def find_benchmark_dir(root) -> Path:
    hits = sorted(Path(root).rglob("ground_truth_labels.csv"))
    if not hits:
        raise FileNotFoundError(
            f"No se encontró ground_truth_labels.csv bajo {root}. Ejecuta `make data`."
        )
    return hits[0].parent


def list_series(bench: Path) -> pd.DataFrame:
    """Una fila por serie. Escenario y severidad salen del nombre de carpeta (exp_h -> EXP, H)."""
    rows = []
    for d in sorted((bench / "heat_load").iterdir()):
        parts = d.name.lower().split("_")
        if not d.is_dir() or len(parts) != 2:  # salta 'baseline' (sin evento de drift)
            continue
        for f in sorted(d.glob("*.csv")):
            rows.append(
                dict(
                    scenario=parts[0].upper(),
                    severity=parts[1].upper(),
                    folder=d.name,
                    file=f.name,
                    path=f,
                    realization=f"{d.name}/{f.stem}",
                )
            )
    return pd.DataFrame(rows)


def _pick(columns, patterns, exclude=()):
    for p in patterns:
        for c in columns:
            low = str(c).lower()
            if p in low and not any(e in low for e in exclude):
                return c
    return None


def load_labels(bench: Path, cfg_labels: dict | None = None):
    df = pd.read_csv(bench / "ground_truth_labels.csv")
    cfg_labels = cfg_labels or {}
    file_col = cfg_labels.get("file_col") or _pick(df.columns, FILE_PATTERNS)
    start_col = cfg_labels.get("start_col") or _pick(df.columns, START_PATTERNS, START_EXCLUDE)
    if file_col is None or start_col is None:
        raise ValueError(
            "No pude identificar columnas en ground_truth_labels.csv. "
            f"Columnas: {list(df.columns)}. Fija labels.file_col y labels.start_col en el YAML."
        )
    # Formato real: una fila 'onset' por serie + filas 'step' (escalones de los drifts incrementales)
    ev_col = next((c for c in df.columns if "event" in str(c).lower() and "type" in str(c).lower()), None)
    if ev_col is not None:
        is_onset = df[ev_col].astype(str).str.lower() == "onset"
        if is_onset.any():
            steps = df[~is_onset]
            last = pd.to_datetime(steps[start_col], utc=True).groupby(steps[file_col]).max()
            df = df[is_onset].reset_index(drop=True)
            df["last_step"] = df[file_col].map(last)
    return df, file_col, start_col


def resolve_labels(series: pd.DataFrame, labels: pd.DataFrame, file_col: str):
    """Empareja filas de etiquetas con series. Devuelve (no_resueltas, {idx_serie: pos_etiqueta})."""
    by_rel = dict(zip(series["folder"] + "/" + series["file"], series.index))
    by_base: dict[str, list] = {}
    for i, f in series["file"].items():
        by_base.setdefault(f, []).append(i)
    unresolved, matched = [], {}
    for pos, v in enumerate(labels[file_col].astype(str)):
        v = v.replace("\\", "/")
        if not v.lower().endswith(".csv"):
            v += ".csv"
        parts = v.split("/")
        idx = by_rel.get("/".join(parts[-2:])) if len(parts) >= 2 else None
        if idx is None and len(by_base.get(parts[-1], [])) == 1:
            idx = by_base[parts[-1]][0]
        if idx is None:
            unresolved.append(v)
        else:
            matched[idx] = pos
    return unresolved, matched


# --------------------------------------------------------------------------- series


def read_heat_load(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert "heat_load_w" in df.columns, f"{path}: falta la columna heat_load_w"
    ts_col = _pick([c for c in df.columns if c != "heat_load_w"], ("timestamp", "datetime", "time", "date"))
    ts_col = ts_col or next(c for c in df.columns if c != "heat_load_w")
    ts = pd.to_datetime(df[ts_col])
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    return pd.DataFrame({"timestamp": ts, "heat_load_w": df["heat_load_w"].astype(float)})


def daily_profiles(df: pd.DataFrame):
    """Serie horaria -> matriz (n_días, 24). Descarta días sin las 24 horas."""
    assert df["timestamp"].is_monotonic_increasing
    assert df["heat_load_w"].notna().all()
    ts = df["timestamp"]
    tmp = pd.DataFrame({"day": ts.dt.floor("D"), "hour": ts.dt.hour, "y": df["heat_load_w"]})
    piv = tmp.pivot(index="day", columns="hour", values="y").reindex(columns=range(24)).dropna()
    X = piv.to_numpy(dtype=float)
    assert X.shape[1] == 24 and np.isfinite(X).all()
    return X, pd.DatetimeIndex(piv.index)


def _event_day(raw, first_ts: pd.Timestamp) -> pd.Timestamp:
    if isinstance(raw, (int, float, np.integer, np.floating)):  # índice horario desde el inicio
        ts = first_ts + pd.Timedelta(hours=float(raw))
    else:
        ts = pd.Timestamp(raw)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.floor("D")


def assign_phase(days_since_start, transition_end) -> np.ndarray:
    """pre: antes del onset; transition: del onset al último escalón; developed: después."""
    d = np.asarray(days_since_start)
    return np.where(d < 0, "pre", np.where(d < np.asarray(transition_end), "transition", "developed"))


def select_realizations(series: pd.DataFrame, n, seed: int) -> pd.DataFrame:
    if n in ("all", None):
        return series
    rng = np.random.default_rng(seed)
    keep = []
    for _, g in series.groupby(["scenario", "severity"], sort=True):
        idx = g.index.to_numpy()
        keep += rng.choice(idx, size=min(int(n), len(idx)), replace=False).tolist()
    return series.loc[sorted(keep)]


@dataclass
class Dataset:
    X: np.ndarray            # (N, 24)
    meta: pd.DataFrame       # realization, scenario, severity, date, days_since_start, phase
    realizations: pd.DataFrame  # una fila por realización


def build_dataset(cfg: dict) -> Dataset:
    dcfg = cfg["data"]
    bench = find_benchmark_dir(dcfg.get("root", "data/raw"))
    series = list_series(bench)
    labels, file_col, start_col = load_labels(bench, cfg.get("labels"))
    unresolved, matched = resolve_labels(series, labels, file_col)
    assert not unresolved, f"Etiquetas que apuntan a ficheros inexistentes: {unresolved[:5]}"
    series = series[series["scenario"].isin(dcfg["scenarios"]) & series["severity"].isin(dcfg["severities"])]
    assert set(series["scenario"]) <= set(SCENARIOS) and set(series["severity"]) <= set(SEVERITIES)
    assert all(i in matched for i in series.index), "Hay series sin etiqueta de drift"
    series = select_realizations(series, dcfg.get("realizations", "all"), cfg["seed"])

    t_default = cfg.get("phases", {}).get("transition_days", 180)
    Xs, metas = [], []
    for idx, r in tqdm(series.iterrows(), total=len(series), desc="Cargando series"):
        df = read_heat_load(r["path"])
        X, days = daily_profiles(df)
        first_ts = df["timestamp"].iloc[0]
        ev = _event_day(labels[start_col].iloc[matched[idx]], first_ts)
        last = labels["last_step"].iloc[matched[idx]] if "last_step" in labels.columns else pd.NaT
        t_end = (_event_day(last, first_ts) - ev).days if pd.notna(last) else 0
        if t_end <= 0:  # drift súbito (sin escalones): ventana fija de transición
            t_end = t_default
        Xs.append(X)
        metas.append(
            pd.DataFrame(
                {
                    "realization": r["realization"],
                    "scenario": r["scenario"],
                    "severity": r["severity"],
                    "date": days,
                    "days_since_start": np.asarray((days - ev).days),
                    "transition_end": t_end,
                }
            )
        )
    X = np.vstack(Xs)
    meta = pd.concat(metas, ignore_index=True)
    meta["phase"] = assign_phase(meta["days_since_start"], meta["transition_end"])
    assert len(meta) == len(X) and np.isfinite(X).all() and X.shape[1] == 24
    realizations = series[["realization", "scenario", "severity"]].reset_index(drop=True)
    return Dataset(X=X, meta=meta, realizations=realizations)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--dest", default="data/raw")
    a = ap.parse_args()
    if a.download:
        download(a.dest)
