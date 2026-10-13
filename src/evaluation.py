"""Métricas, tablas y figuras."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

from src.data import SCENARIOS


def classification_metrics(y_true, y_pred) -> dict:
    assert len(y_true) == len(y_pred)
    labels = list(SCENARIOS)
    f1s = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    m = {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        **{f"f1_{s}": float(v) for s, v in zip(labels, f1s)},
    }
    assert all(np.isfinite(v) for v in m.values())
    return m


def confusion(y_true, y_pred) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(SCENARIOS))


def to_markdown(df: pd.DataFrame) -> str:
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: "--" if np.isnan(v) else f"{v:.3f}")
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    body = ["| " + " | ".join(map(str, r)) + " |" for r in d.to_numpy()]
    return "\n".join([head, sep, *body])


def save_tables(results: pd.DataFrame, outdir: Path) -> None:
    results.to_csv(outdir / "results_table.csv", index=False)
    cols = ["representation", "components", "macro_f1", "balanced_accuracy", "recon_rmse", "f1_NS"]
    parts = []
    for (model, phase), g in results.groupby(["model", "phase"], sort=False):
        parts.append(f"### {model} — {phase}\n\n" + to_markdown(g[cols]))
    (outdir / "summary_table.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- figuras

MODEL_NAMES = {"logistic_regression": "Logistic regression", "random_forest": "Random forest"}


def _name(model: str) -> str:
    return MODEL_NAMES.get(model, model)


def _legend_below(fig, ax, ncol: int = 3) -> None:
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=ncol, fontsize=8, frameon=False)


def fig_example_profiles(ds, outdir: Path) -> None:
    scen = [s for s in SCENARIOS if s in set(ds.meta["scenario"])]
    fig, ax = plt.subplots(2, len(scen), figsize=(4.8 * len(scen), 7.5), squeeze=False)
    for j, s in enumerate(scen):
        r = ds.realizations[ds.realizations["scenario"] == s]
        sev = "H" if "H" in set(r["severity"]) else r["severity"].iloc[0]
        rid = r[r["severity"] == sev]["realization"].iloc[0]
        m = ds.meta[ds.meta["realization"] == rid]
        X = ds.X[m.index.to_numpy()]
        daily = pd.Series(X.mean(axis=1), index=m["date"].to_numpy()).rolling(7, min_periods=1).mean()
        a = ax[0, j]
        a.plot(daily.index, daily.values, lw=1)
        ev = m["date"].iloc[np.argmax((m["days_since_start"] >= 0).to_numpy())]
        a.axvline(ev, color="r", ls="--", label="Drift onset")
        a.set(title=f"{s}-{sev} ({rid.split('/')[1]})", xlabel="Date", ylabel="Daily mean load, 7-day average (W)")
        a.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        a.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        a.tick_params(axis="x", labelrotation=40)
        a.legend()
        for phase, c, lab in (("pre", "tab:blue", "Pre-drift"), ("developed", "tab:red", "Developed")):
            sel = (m["phase"] == phase).to_numpy()
            if sel.any():
                ax[1, j].plot(range(24), X[sel].mean(axis=0), c=c, label=lab)
        ax[1, j].set(xlabel="Hour of day", ylabel="Mean daily profile (W)")
        ax[1, j].legend()
    fig.tight_layout()
    fig.savefig(outdir / "01_example_profiles.png", dpi=130)
    plt.close(fig)


def fig_pca_variance(rep24, components, outdir: Path) -> None:
    cum = np.cumsum(rep24.pca.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(range(1, 25), cum, "o-")
    for r in components:
        ax.axvline(r, color="gray", ls=":")
    ax.set(xlabel="Number of components", ylabel="Cumulative explained variance", ylim=(0, 1.02))
    fig.tight_layout()
    fig.savefig(outdir / "02_pca_variance.png", dpi=130)
    plt.close(fig)


def fig_reconstruction(curve, reps, x_example, outdir: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(range(1, 25), curve, "o-")
    ax[0].set(xlabel="Number of components", ylabel="Reconstruction RMSE, test set (W)")
    ax[1].plot(range(24), x_example, "k", lw=2, label="Original")
    for rep in reps:
        if rep.pca is not None:
            ax[1].plot(range(24), rep.reconstruct(x_example[None, :])[0], label=f"PCA-{rep.n_components}")
    ax[1].set(xlabel="Hour of day", ylabel="Heat load (W)", title="Test profile and reconstructions")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(outdir / "03_reconstruction.png", dpi=130)
    plt.close(fig)


def fig_confusions(cms: dict, outdir: Path) -> None:
    """cms[(model, phase, representation_label)] = 3x3 matrix."""
    models = list(dict.fromkeys(k[0] for k in cms))
    phases = list(dict.fromkeys(k[1] for k in cms))
    reps = list(dict.fromkeys(k[2] for k in cms))
    n_rows = len(models) * len(phases)
    fig, ax = plt.subplots(n_rows, len(reps), figsize=(3.2 * len(reps), 3.1 * n_rows), squeeze=False)
    for i, (mo, ph) in enumerate((m, p) for m in models for p in phases):
        for j, rp in enumerate(reps):
            cm = cms[(mo, ph, rp)]
            a = ax[i, j]
            a.imshow(cm / cm.sum(axis=1, keepdims=True).clip(1), vmin=0, vmax=1, cmap="Blues")
            for (u, v), n in np.ndenumerate(cm):
                a.text(v, u, int(n), ha="center", va="center", fontsize=8)
            a.set_xticks(range(3), SCENARIOS, fontsize=8)
            a.set_yticks(range(3), SCENARIOS, fontsize=8)
            a.set_title(f"{_name(mo)} | {ph.capitalize()} | {rp}", fontsize=8)
            if i == n_rows - 1:
                a.set_xlabel("Predicted class", fontsize=8)
            if j == 0:
                a.set_ylabel("True class", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "04_confusion_matrix.png", dpi=130)
    plt.close(fig)


def _components_axes(ax, xs) -> None:
    for a in ax:
        a.set_xticks(sorted(set(xs)))
        a.set_xlabel("Number of components (24 = raw profile)")


def fig_components_vs_performance(results: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 5), sharex=True)
    for (model, phase), g in results.groupby(["model", "phase"], sort=False):
        g = g.sort_values("components")
        for a, col in zip(ax, ("macro_f1", "f1_NS")):
            a.plot(g["components"], g[col], marker="o", ls="--" if phase == "pre" else "-",
                   label=f"{_name(model)} / {phase}")
    _components_axes(ax, results["components"])
    ax[0].set_ylabel("Macro-F1")
    ax[1].set_ylabel("NS F1")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    _legend_below(fig, ax[0])
    fig.savefig(outdir / "05_components_vs_performance.png", dpi=130)
    plt.close(fig)


def fig_components_seeds(agg: pd.DataFrame, outdir: Path) -> None:
    """agg: columns model, phase, components, macro_f1_mean/std, f1_NS_mean/std."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 5), sharex=True)
    for (model, phase), g in agg.groupby(["model", "phase"], sort=False):
        g = g.sort_values("components")
        for a, col in zip(ax, ("macro_f1", "f1_NS")):
            a.errorbar(g["components"], g[f"{col}_mean"], yerr=g[f"{col}_std"], marker="o", capsize=3,
                       ls="--" if phase == "pre" else "-", label=f"{_name(model)} / {phase}")
    _components_axes(ax, agg["components"])
    ax[0].set_ylabel("Macro-F1 (mean ± std over seeds)")
    ax[1].set_ylabel("NS F1 (mean ± std over seeds)")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    _legend_below(fig, ax[0])
    fig.savefig(outdir / "06_components_vs_performance_seeds.png", dpi=130)
    plt.close(fig)


def fig_progress(b_df: pd.DataFrame, outdir: Path) -> None:
    """b_df: columns model, representation, bin, progress_pct, balanced_accuracy."""
    models = list(dict.fromkeys(b_df["model"]))
    fig, ax = plt.subplots(1, len(models), figsize=(6 * len(models), 4.6), squeeze=False, sharey=True)
    for a, model in zip(ax[0], models):
        g = b_df[b_df["model"] == model]
        for label, gg in g.groupby("representation", sort=False):
            gg = gg.sort_values("bin")
            a.plot(gg["bin"], gg["balanced_accuracy"], "o-", label=label)
        ticks = g.drop_duplicates("bin").sort_values("bin")
        a.set_xticks(ticks["bin"], [f"{p}%" for p in ticks["progress_pct"]], rotation=40, ha="right")
        a.set(title=_name(model), xlabel="Transition progress (onset to last step)", ylabel="Balanced accuracy")
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "07_progress.png", dpi=130)
    plt.close(fig)
