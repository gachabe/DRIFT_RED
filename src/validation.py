"""Validación del dataset (`make check`) y verificación de la pipeline (`make verify`)."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.data import (
    SCENARIOS,
    SEVERITIES,
    find_benchmark_dir,
    list_series,
    load_config,
    load_labels,
    read_heat_load,
    resolve_labels,
)


def check_dataset(root="data/raw", cfg_labels=None) -> bool:
    results: list[tuple[str, str, str]] = []  # (estado, nombre, detalle)

    def rec(ok, name, detail=""):
        results.append(("PASS" if ok else "FAIL", name, detail if not ok else ""))

    def skip(name, why):
        results.append(("SKIP", name, why))

    def show() -> bool:
        print("\nDHDrift dataset validation\n==========================\n")
        for st, name, detail in results:
            print(f"[{st}] {name}" + (f"  -> {detail}" if detail else ""))
        ok = all(st != "FAIL" for st, _, _ in results)
        print("\nRESULT:", "DATASET OK" if ok else "DATASET INVALID")
        return ok

    bench = find_benchmark_dir(root)
    hl = bench / "heat_load"
    expected = [f"{s.lower()}_{v.lower()}" for s in SCENARIOS for v in SEVERITIES]
    missing = [e for e in expected if not (hl / e).is_dir()]
    rec(not missing, "expected scenario folders exist", f"faltan {missing}")
    series = list_series(bench)

    fails: dict[str, list] = {k: [] for k in ("parse", "hourly", "coverage", "column", "dups", "nan")}
    for _, r in tqdm(series.iterrows(), total=len(series), desc="Validando series"):
        p = r["path"]
        try:
            df = read_heat_load(p)
        except AssertionError:
            fails["column"].append(p.name)
            continue
        except Exception:
            fails["parse"].append(p.name)
            continue
        ts = df["timestamp"]
        if ts.duplicated().any():
            fails["dups"].append(p.name)
        if not (ts.diff().dropna() == pd.Timedelta(hours=1)).all():
            fails["hourly"].append(p.name)
        if not (ts.min().year == 2020 and ts.max().year == 2023):
            fails["coverage"].append(p.name)
        if df["heat_load_w"].isna().any():
            fails["nan"].append(p.name)
    for key, name in (
        ("parse", "timestamps parse correctly"),
        ("hourly", "hourly temporal structure"),
        ("coverage", "2020--2023 coverage"),
        ("column", "heat_load_w column exists"),
        ("dups", "no duplicated timestamps"),
        ("nan", "missing values flagged (none allowed)"),
    ):
        rec(not fails[key], name, f"{len(fails[key])} ficheros, p.ej. {fails[key][:3]}")

    rec((bench / "ground_truth_labels.csv").exists(), "ground_truth_labels.csv exists")
    labels, file_col, start_col = load_labels(bench, cfg_labels)
    unresolved, matched = resolve_labels(series, labels, file_col)
    rec(not unresolved, "every referenced filename exists", f"{len(unresolved)}, p.ej. {unresolved[:3]}")
    unlabeled = [i for i in series.index if i not in matched]
    rec(not unlabeled, "every series has a drift label", f"{len(unlabeled)} sin etiqueta")
    rec(set(series["scenario"]) <= set(SCENARIOS), "scenario labels are valid")
    rec(set(series["severity"]) <= set(SEVERITIES), "severity labels are valid")

    type_col = next((c for c in labels.columns if "drift" in str(c).lower() and "type" in str(c).lower()), None)
    if type_col is None:
        skip("EXP→sudden, REF/NS→incremental", f"sin columna de tipo en labels (columnas: {list(labels.columns)})")
    else:
        bad = []
        for idx, pos in matched.items():
            want = "sudden" if series.loc[idx, "scenario"] == "EXP" else "incremental"
            if want not in str(labels[type_col].iloc[pos]).lower():
                bad.append(series.loc[idx, "file"])
        rec(not bad, "EXP→sudden, REF/NS→incremental (desde labels)", f"{len(bad)} incoherentes")
    skip("L→25%, M→50%, H→100%", "formato de metadata/ no verificado automáticamente; revisar README de Zenodo")
    return show()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/full.yaml")
    ap.add_argument("--run-tests", action="store_true")
    a = ap.parse_args()
    if a.run_tests:
        print("DHDrift ROM-ML verification\n===========================\n")
        rc = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"]).returncode
        print("\nRESULT:", "PIPELINE READY" if rc == 0 else "PIPELINE NOT READY")
        sys.exit(rc)
    cfg = load_config(a.config)
    ok = check_dataset(cfg["data"].get("root", "data/raw"), cfg.get("labels"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
