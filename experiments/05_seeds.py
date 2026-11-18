"""Repite split + ROM + ML con varias semillas (rejilla PCA prefijada) y agrega media ± desviación.

Además guarda, por semilla: predicciones sobre todo el test (para 06_progress y 07_revision), diferencias pareadas
frente a Raw, bootstrap por realización, NRMSE y estadísticas del control negativo BL.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import bootstrap_metrics, nrmse_table, paired_deltas, profile_errors, realization_confusions
from src.data import SCENARIOS, load_baseline, load_config
from src.evaluation import fig_components_seeds, fig_curve_with_ci, to_markdown
from src.features import rows, split_realizations
from src.rom import fit_representation
from src.runtime import load_dataset, out_dir, rcfg

METRICS = ("macro_f1", "balanced_accuracy", "f1_NS", "f1_EXP", "f1_REF", "recon_rmse")
DELTA_METRICS = ("macro_f1", "balanced_accuracy", "f1_EXP", "f1_REF", "f1_NS")
PHASES = ("developed", "transition", "pre")


def _bootstrap_seed(ds, split, collect, reps_labels, models, B, seed) -> pd.DataFrame:
    meta_te = ds.meta.iloc[collect["test_idx"]].reset_index(drop=True)
    rid = ds.realizations.set_index("realization")
    rng = np.random.default_rng(seed)
    rows_ = []
    for model in models:
        for phase in PHASES:
            mask = (meta_te["phase"] == phase).to_numpy()
            Cs, ids = {}, None
            for lab in reps_labels:
                pred = np.array(SCENARIOS, dtype=object)[np.minimum(collect[(model, lab)], 2)]
                ids, Cs[lab] = realization_confusions(meta_te, pred, mask)
            strata = [f"{rid.loc[r, 'scenario']}|{rid.loc[r, 'severity']}" for r in ids]
            boot = bootstrap_metrics(Cs, strata, B, rng)
            for lab in reps_labels:
                if lab == "Raw":
                    continue
                for mi, mname in enumerate(("macro_f1", "balanced_accuracy")):
                    d = boot[lab][mi] - boot["Raw"][mi]
                    d = d[np.isfinite(d)]
                    rows_.append(dict(seed=seed, model=model, phase=phase, components=int(lab.split("-")[1]), metric=mname,
                                      delta_mean=d.mean(), ci_low=np.percentile(d, 2.5), ci_high=np.percentile(d, 97.5),
                                      frac_delta_below_0=float((d < 0).mean())))
    return pd.DataFrame(rows_)


def main(cfg: dict, n_seeds: int = 10, resume: bool = True) -> pd.DataFrame:
    ml = importlib.import_module("experiments.03_ml")
    out = out_dir(cfg)
    parts, pdir, sdir = out / "seed_parts", out / "predictions", out / "splits"
    for d in (parts, pdir, sdir):
        d.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(cfg)  # se carga una sola vez
    bl = load_baseline(cfg)
    tf = cfg.get("split", {}).get("train_fraction", 0.8)
    comps = list(cfg["rom"]["components"])
    B = int(rcfg(cfg, "bootstrap_resamples", 1000))

    sig = hashlib.md5(json.dumps({k: cfg.get(k) for k in ("data", "split", "phases", "rom", "ml", "models", "labels")} | {"B": B, "seed": cfg["seed"]},
                                 sort_keys=True, default=str).encode()).hexdigest()
    sigf = parts / "config_hash.txt"
    if resume and sigf.exists() and sigf.read_text() != sig:
        print("la configuración cambió desde la última ejecución: se ignoran las semillas guardadas (resume desactivado)")
        resume = False
    sigf.write_text(sig)

    for i in range(n_seeds):
        seed = cfg["seed"] + i
        if resume and all((parts / f"{k}_{seed}.csv").exists() for k in ("res", "bl", "nrmse", "boot")) and (pdir / f"seed_{seed}.npz").exists():
            print(f"semilla {seed} ({i + 1}/{n_seeds}) ya hecha, se reutiliza")
            continue
        split = split_realizations(ds.realizations, tf, seed)
        pd.DataFrame({"realization": split.train}).to_csv(sdir / f"seed_{seed}_train.csv", index=False)
        pd.DataFrame({"realization": split.test}).to_csv(sdir / f"seed_{seed}_test.csv", index=False)
        X_train, _ = rows(ds, split.train)
        reps = [fit_representation(X_train, None)] + [fit_representation(X_train, r) for r in comps]
        rep24 = fit_representation(X_train, 24)
        collect: dict = {}
        res = ml.run(ds, split, reps, {**cfg, "seed": seed}, out, save=False, collect=collect, bl=bl)
        labels = [ml.rep_label(r) for r in reps]
        np.savez_compressed(pdir / f"seed_{seed}.npz", test_idx=collect["test_idx"],
                            **{f"{m}|{lab}": collect[(m, lab)] for m in cfg["models"] for lab in labels})
        res.assign(seed=seed).to_csv(parts / f"res_{seed}.csv", index=False)
        pd.DataFrame(collect["bl"]).assign(seed=seed).to_csv(parts / f"bl_{seed}.csv", index=False)
        te = collect["test_idx"]
        meta_te = ds.meta.iloc[te].reset_index(drop=True)
        errs = profile_errors(rep24, ds.X[te], comps)
        nrmse_table(meta_te, ds.X[te], errs).assign(seed=seed).to_csv(parts / f"nrmse_{seed}.csv", index=False)
        _bootstrap_seed(ds, split, collect, labels, cfg["models"], B, seed).to_csv(parts / f"boot_{seed}.csv", index=False)
        print(f"semilla {seed} ({i + 1}/{n_seeds}) hecha", flush=True)

    seeds = [cfg["seed"] + i for i in range(n_seeds)]
    cat = lambda k: pd.concat([pd.read_csv(parts / f"{k}_{s}.csv") for s in seeds], ignore_index=True)
    raw = cat("res")
    raw.to_csv(out / "results_seeds_raw.csv", index=False)

    keys = ["model", "phase", "representation", "components"]
    g = raw.groupby(keys, sort=False)[list(METRICS)].agg(["mean", "std"])
    g.columns = [f"{m}_{s}" for m, s in g.columns]
    agg = g.reset_index()
    agg.to_csv(out / "results_seeds_summary.csv", index=False)

    parts_md = []
    for (model, phase), a in agg.groupby(["model", "phase"], sort=False):
        t = a[["representation", "components"]].copy()
        for m in ("macro_f1", "balanced_accuracy", "f1_NS"):
            t[m] = [f"{u:.3f} ± {v:.3f}" for u, v in zip(a[f"{m}_mean"], a[f"{m}_std"])]
        t["recon_rmse"] = a["recon_rmse_mean"].to_numpy()
        parts_md.append(f"### {model} — {phase} ({n_seeds} semillas)\n\n" + to_markdown(t))
    (out / "results_seeds_summary.md").write_text("\n\n".join(parts_md) + "\n", encoding="utf-8")
    fig_components_seeds(agg, out)
    fig_curve_with_ci(agg, None, out)

    # P1: diferencias pareadas por split frente a Raw (sensibilidad a la partición, no incertidumbre estadística plena)
    d = paired_deltas(raw, DELTA_METRICS)
    d.to_csv(out / "paired_deltas_vs_raw.csv", index=False)
    md = []
    for (model, phase), a in d[d["metric"].isin(["macro_f1", "balanced_accuracy"])].groupby(["model", "phase"], sort=False):
        t = a[["metric", "components", "delta_mean", "delta_sd", "p2_5", "p97_5"]].copy()
        md.append(f"### {model} — {phase}: metric(PCA-k) − metric(Raw), {n_seeds} splits (partition sensitivity)\n\n" + to_markdown(t))
    (out / "paired_deltas_vs_raw.md").write_text("\n\n".join(md) + "\n", encoding="utf-8")
    boot = cat("boot")
    boot.to_csv(out / "bootstrap_deltas_by_split.csv", index=False)
    bs = boot.groupby(["model", "phase", "components", "metric"], sort=False)[["delta_mean", "ci_low", "ci_high", "frac_delta_below_0"]].mean().reset_index()
    bs.to_csv(out / "bootstrap_deltas_summary.csv", index=False)
    cat("nrmse").groupby(["components", "scenario", "severity", "phase"], sort=False).mean(numeric_only=True).drop(columns="seed").reset_index() \
        .to_csv(out / "nrmse_by_group.csv", index=False)
    blr = cat("bl")
    blr.to_csv(out / "bl_negative_control_by_seed.csv", index=False)
    print(f"\nListo. Resultados en {out}/ (results_seeds_summary.md, 06_*.png, paired_deltas_vs_raw.*, nrmse_by_group.csv)")
    return agg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--no-resume", action="store_true")
    a = ap.parse_args()
    main(load_config(a.config), a.n_seeds, resume=not a.no_resume)
