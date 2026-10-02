"""Dataset sintético con la MISMA estructura documentada de DHDrift (solo para tests)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_synthetic(root: Path, n_real: int = 4, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    bench = Path(root) / "benchmark_data"
    (bench / "heat_load").mkdir(parents=True)
    ts = pd.date_range("2020-01-01", "2023-12-31 23:00", freq="h")
    doy, hour = ts.dayofyear.to_numpy(), ts.hour.to_numpy()
    season = 0.6 + 0.4 * np.cos(2 * np.pi * (doy - 15) / 365.25)
    shape = 1 + 0.5 * np.exp(-(((hour - 7) / 2) ** 2)) + 0.6 * np.exp(-(((hour - 19) / 2.5) ** 2))
    shape = shape - 0.3 * ((hour < 6) | (hour >= 23))
    base = 1e6 * season * shape
    night = (hour >= 22) | (hour < 6)
    sev_val = {"L": 0.25, "M": 0.5, "H": 1.0}
    labels = []

    def write(folder, name, y):
        d = bench / "heat_load" / folder
        d.mkdir(exist_ok=True)
        pd.DataFrame({"timestamp": ts, "heat_load_w": y}).to_csv(d / name, index=False)

    write("baseline", "baseline.csv", base)
    for sc in ("exp", "ref", "ns"):
        for sv in ("l", "m", "h"):
            for k in range(n_real):
                event = pd.Timestamp("2021-06-01") + pd.Timedelta(days=int(rng.integers(0, 200)))
                after = (ts >= event).astype(float)
                ramp = np.clip((ts - event) / pd.Timedelta(days=360), 0, 1)
                s = sev_val[sv.upper()]
                y = base * rng.uniform(0.9, 1.1)
                if sc == "exp":
                    y = y * (1 + 0.3 * s * after)
                elif sc == "ref":
                    y = y * (1 - 0.25 * s * ramp * (season > 0.5))
                else:
                    y = y * (1 - 0.4 * s * ramp * night + 0.05 * s * ramp * (~night))
                y = y * rng.lognormal(0, 0.03, len(ts))
                name = f"{sc}_{sv}_{k:03d}.csv"
                write(f"{sc}_{sv}", name, y)
                ev_utc = event.tz_localize("UTC")
                row = dict(filename=name, scenario=sc.upper(), scenario_severity=sv.upper())
                labels.append({**row, "timestamp": ev_utc, "event_type": "onset"})
                if sc != "exp":  # drift incremental: 12 escalones hasta event+330d
                    for j in range(1, 13):
                        labels.append({**row, "timestamp": ev_utc + pd.Timedelta(days=30 * j - 0), "event_type": "step"})
    pd.DataFrame(labels).to_csv(bench / "ground_truth_labels.csv", index=False)
    pd.DataFrame({"timestamp": ts, "temp": 0.0}).to_csv(bench / "weather.csv", index=False)
    return Path(root)
