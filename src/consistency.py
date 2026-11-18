"""Comprueba que cada cifra de cabecera del manuscrito coincide con los CSV generados.

reference/manuscript_numbers.csv: source (seed42 | seeds_mean), model, phase, representation, components, metric, value, tol.
- seed42      -> results_table.csv (una ejecución, semilla del YAML)
- seeds_mean  -> results_seeds_summary.csv (media sobre las semillas)
Añade ahí las cifras que aparezcan en el artículo. Sale con código 1 si alguna no cuadra.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def check_numbers(results_dir, reference_csv) -> pd.DataFrame:
    rd = Path(results_dir)
    ref = pd.read_csv(reference_csv, comment="#")
    t42 = pd.read_csv(rd / "results_table.csv") if (rd / "results_table.csv").exists() else None
    tsm = pd.read_csv(rd / "results_seeds_summary.csv") if (rd / "results_seeds_summary.csv").exists() else None
    out = []
    for _, r in ref.iterrows():
        src = t42 if r["source"] == "seed42" else tsm
        col = r["metric"] if r["source"] == "seed42" else f"{r['metric']}_mean"
        got = float("nan")
        if src is not None and col in src.columns:
            m = src[(src["model"] == r["model"]) & (src["phase"] == r["phase"]) & (src["representation"] == r["representation"]) & (src["components"] == r["components"])]
            if len(m) == 1:
                got = float(m[col].iloc[0])
        tol = float(r.get("tol", 0.0006))
        ok = bool(pd.notna(got) and abs(got - float(r["value"])) <= tol)
        out.append({**r.to_dict(), "generated": got, "abs_diff": abs(got - float(r["value"])) if pd.notna(got) else float("nan"), "ok": ok})
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/full")
    ap.add_argument("--reference", default="reference/manuscript_numbers.csv")
    a = ap.parse_args()
    df = check_numbers(a.results, a.reference)
    out = Path(a.results) / "manuscript_consistency.csv"
    df.to_csv(out, index=False)
    bad = df[~df["ok"]]
    print(f"{len(df) - len(bad)}/{len(df)} cifras coinciden. Detalle en {out}")
    if len(bad):
        print(bad[["source", "model", "phase", "representation", "components", "metric", "value", "generated"]].to_string(index=False))
    sys.exit(1 if len(bad) else 0)


if __name__ == "__main__":
    main()
