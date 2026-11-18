"""Acierto según el avance de la transición, la severidad y la clase, sobre TODOS los splits (semillas).

Lee las predicciones guardadas por 05_seeds (si faltan, las genera). Salidas:
- progress_by_bin.csv / progress_by_severity.csv / progress_summary.md / 07_progress.png (split de la semilla del YAML)
- progress_class_severity_all_splits.csv  (esquema: seed, model, representation, scenario, severity, phase,
  progress_bin, precision, recall, f1, n_profiles, n_realizations)  + _summary.csv (media ± SD entre splits)
- exp_false_positives.csv / exp_false_positive_analysis.md / 08_*.png / 09_*.png
"""
from __future__ import annotations

import argparse
import importlib

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

from src.analysis import bin_label, confusion_by_stratum, prf_table, progress_bins
from src.data import SCENARIOS, SEVERITIES, load_config
from src.evaluation import fig_class_progress, fig_exp_false_positives, fig_progress, to_markdown
from src.runtime import available_seeds, load_dataset, load_predictions, out_dir, rcfg


def _m(y, p) -> dict:
    return {
        "macro_f1": float(f1_score(y, p, labels=list(SCENARIOS), average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "n_rows": len(y),
    }


def _label(rep: str):
    return (24 if rep == "Raw" else int(rep.split("-")[1]))


def main(cfg: dict, bins: int = 5, n_seeds: int | None = None) -> None:
    out = out_dir(cfg)
    if not available_seeds(out):
        importlib.import_module("experiments.05_seeds").main(cfg, n_seeds or int(rcfg(cfg, "n_seeds", 10)))
    seeds = available_seeds(out)
    ds = load_dataset(cfg)
    main_seed = cfg["seed"]
    all_rows, fp_rows, by_bin, by_sev = [], [], [], []
    for seed in seeds:
        te, preds = load_predictions(out, seed)
        mt = ds.meta.iloc[te].reset_index(drop=True)
        b = progress_bins(mt, bins)
        mt = mt.assign(progress_bin=[bin_label(int(x), bins) for x in b], _bin=b)
        for (model, rep), code in preds.items():
            pred = np.array(SCENARIOS, dtype=object)[np.minimum(code, 2)]
            t = prf_table(mt, pred, ["severity", "phase", "progress_bin"])
            t.insert(0, "representation", rep), t.insert(0, "model", model), t.insert(0, "seed", seed)
            t["_bin"] = [(-1 if x == "NA" else int(x.split("-")[0]) * bins // 100) for x in t["progress_bin"]]
            all_rows.append(t)
            cb = confusion_by_stratum(mt.assign(_pred=pred), pred, ["severity", "phase", "_bin"])
            cb = cb[cb["pred"] == "EXP"].rename(columns={"_bin": "progress_bin"})
            cb.insert(0, "representation", rep), cb.insert(0, "model", model), cb.insert(0, "seed", seed)
            fp_rows.append(cb)
            if seed == main_seed:  # tablas históricas (un split)
                y, ph, sev = mt["scenario"].to_numpy(), mt["phase"].to_numpy(), mt["severity"].to_numpy()
                comps = _label(rep)
                for k in range(bins):
                    s = (ph == "transition") & (b == k)
                    by_bin.append(dict(model=model, representation=rep, components=comps,
                                       progress_pct=f"{100 * k // bins}-{100 * (k + 1) // bins}", bin=k, **_m(y[s], pred[s])))
                for phn in ("developed", "transition"):
                    for sv in SEVERITIES:
                        s = (ph == phn) & (sev == sv)
                        by_sev.append(dict(model=model, representation=rep, components=comps, phase=phn, severity=sv, **_m(y[s], pred[s])))
    full = pd.concat(all_rows, ignore_index=True).drop(columns="_bin")
    cols = ["seed", "model", "representation", "scenario", "severity", "phase", "progress_bin", "precision", "recall", "f1",
            "n_profiles", "n_realizations", "n_pred"]
    full[cols].to_csv(out / "progress_class_severity_all_splits.csv", index=False)
    g = full.groupby(["model", "representation", "scenario", "severity", "phase", "progress_bin"], sort=False)
    summ = g.agg(**{f"{m}_{s}": (m, f) for m in ("precision", "recall", "f1") for s, f in (("mean", "mean"), ("sd", "std"))},
                 n_profiles_mean=("n_profiles", "mean"), n_realizations_mean=("n_realizations", "mean"), n_splits=("seed", "nunique")).reset_index()
    summ.to_csv(out / "progress_class_severity_summary.csv", index=False)

    b_df, s_df = pd.DataFrame(by_bin), pd.DataFrame(by_sev)
    b_df.to_csv(out / "progress_by_bin.csv", index=False)
    s_df.to_csv(out / "progress_by_severity.csv", index=False)
    parts = []
    for model, gg in b_df.groupby("model", sort=False):
        t = gg.pivot(index="representation", columns="progress_pct", values="balanced_accuracy").reset_index()
        parts.append(f"### {model} — balanced accuracy por tramo de transición (% del avance)\n\n" + to_markdown(t))
    for (model, ph), gg in s_df.groupby(["model", "phase"], sort=False):
        t = gg.pivot(index="representation", columns="severity", values="balanced_accuracy").reset_index()
        parts.append(f"### {model} — {ph}: balanced accuracy por severidad\n\n" + to_markdown(t))
    (out / "progress_summary.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    fig_progress(b_df, out, (cfg.get("figures") or {}).get("progress_components"))

    # --- ¿El F1 bajo de EXP lo causan los falsos positivos EXP de REF/NS tempranos?
    fp = pd.concat(fp_rows, ignore_index=True)
    tr = fp[fp["phase"] == "transition"].groupby(["model", "representation", "progress_bin", "true"], sort=False)["count"].sum().reset_index()
    tr["share"] = tr["count"] / tr.groupby(["model", "representation", "progress_bin"])["count"].transform("sum")
    tr.to_csv(out / "exp_false_positives.csv", index=False)
    focus = ["Raw"] + [f"PCA-{k}" for k in (cfg.get("figures") or {}).get("progress_components", cfg["rom"].get("focus_components", [2, 5, 10]))]
    focus = [r for r in focus if r in set(full["representation"])]
    for model in cfg["models"]:
        s = summ[(summ["phase"] == "transition") & (summ["progress_bin"] != "NA")].copy()
        s["progress_bin"] = [int(x.split("-")[0]) * bins // 100 for x in s["progress_bin"]]
        s = s.groupby(["model", "representation", "scenario", "progress_bin"], sort=False)[
            [f"{m}_{k}" for m in ("precision", "recall", "f1") for k in ("mean", "sd")]].mean().reset_index()
        fig_class_progress(s, out, model, focus, bins)
        fig_exp_false_positives(tr, out, model, focus, bins)

    md = ["# EXP: ¿falsos positivos de REF/NS tempranos?\n",
          "Para cada representación: en cada tramo de la transición, precisión y recall de EXP (media entre splits) y "
          "composición de los días predichos como EXP.\n"]
    pe = summ[(summ["scenario"] == "EXP") & (summ["phase"] == "transition") & (summ["progress_bin"] != "NA")]
    for (model, rep), g in pe.groupby(["model", "representation"], sort=False):
        a = g.groupby("progress_bin", sort=False)[["precision_mean", "recall_mean", "f1_mean"]].mean().reset_index()
        f = tr[(tr["model"] == model) & (tr["representation"] == rep)].pivot(index="progress_bin", columns="true", values="share").reset_index()
        md.append(f"## {model} / {rep}\n\n" + to_markdown(a) + "\n\nComposición de las predicciones EXP:\n\n" + to_markdown(f.fillna(0.0)) + "\n")
    # veredicto por semilla (RF crudo): precisión < recall en la mitad inicial de la transición
    v = full[(full["scenario"] == "EXP") & (full["phase"] == "transition") & (full["progress_bin"] != "NA") &
             (full["model"] == "random_forest") & (full["representation"] == "Raw")]
    if len(v):
        early = [b for b in v["progress_bin"].unique() if int(b.split("-")[0]) * bins // 100 < max(bins // 2, 1)]
        e = v[v["progress_bin"].isin(early)].groupby("seed")[["precision", "recall"]].mean()
        k = int((e["precision"] < e["recall"]).sum())
        md.append(f"**Veredicto (RF crudo, tramos iniciales):** precisión < recall de EXP en {k} de {len(e)} splits "
                  f"(precisión media {e['precision'].mean():.3f}, recall medio {e['recall'].mean():.3f}). "
                  "Si es consistente, el F1 bajo de EXP es un problema de precisión (REF/NS tempranos etiquetados como EXP), no de recall.\n")
    (out / "exp_false_positive_analysis.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nListo. Resultados en {out}/ (progress_summary.md, progress_class_severity_*.csv, 07-09_*.png)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--bins", type=int, default=5)
    a = ap.parse_args()
    main(load_config(a.config), a.bins)
