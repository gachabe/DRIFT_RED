"""Métricas, tablas y figuras."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
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


def fig_example_profiles(ds, outdir: Path) -> None:
    scen = [s for s in SCENARIOS if s in set(ds.meta["scenario"])]
    fig, ax = plt.subplots(2, len(scen), figsize=(4.5 * len(scen), 7), squeeze=False)
    for j, s in enumerate(scen):
        r = ds.realizations[ds.realizations["scenario"] == s]
        sev = "H" if "H" in set(r["severity"]) else r["severity"].iloc[0]
        rid = r[r["severity"] == sev]["realization"].iloc[0]
        m = ds.meta[ds.meta["realization"] == rid]
        X = ds.X[m.index.to_numpy()]
        daily = pd.Series(X.mean(axis=1), index=m["date"].to_numpy()).rolling(7, min_periods=1).mean()
        ax[0, j].plot(daily.index, daily.values, lw=1)
        ev = m["date"].iloc[np.argmax((m["days_since_start"] >= 0).to_numpy())]
        ax[0, j].axvline(ev, color="r", ls="--", label="inicio drift")
        ax[0, j].set(title=f"{s}-{sev} ({rid.split('/')[1]})", ylabel="carga diaria media, 7d (W)")
        ax[0, j].legend()
        for phase, c in (("pre", "tab:blue"), ("developed", "tab:red")):
            sel = (m["phase"] == phase).to_numpy()
            if sel.any():
                ax[1, j].plot(range(24), X[sel].mean(axis=0), c=c, label=phase)
        ax[1, j].set(xlabel="hora", ylabel="perfil medio (W)")
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
    ax.set(xlabel="componentes", ylabel="varianza explicada acumulada", ylim=(0, 1.02))
    fig.tight_layout()
    fig.savefig(outdir / "02_pca_variance.png", dpi=130)
    plt.close(fig)


def fig_reconstruction(curve, reps, x_example, outdir: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(range(1, 25), curve, "o-")
    ax[0].set(xlabel="componentes", ylabel="RMSE reconstrucción, test (W)")
    ax[1].plot(range(24), x_example, "k", lw=2, label="original")
    for rep in reps:
        if rep.pca is not None:
            ax[1].plot(range(24), rep.reconstruct(x_example[None, :])[0], label=f"PCA-{rep.n_components}")
    ax[1].set(xlabel="hora", ylabel="W", title="perfil de test y reconstrucciones")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(outdir / "03_reconstruction.png", dpi=130)
    plt.close(fig)


def fig_confusions(cms: dict, outdir: Path) -> None:
    """cms[(model, phase, representation_label)] = matriz 3x3."""
    models = list(dict.fromkeys(k[0] for k in cms))
    phases = list(dict.fromkeys(k[1] for k in cms))
    reps = list(dict.fromkeys(k[2] for k in cms))
    fig, ax = plt.subplots(
        len(models) * len(phases), len(reps), figsize=(3 * len(reps), 2.8 * len(models) * len(phases)), squeeze=False
    )
    for i, (mo, ph) in enumerate((m, p) for m in models for p in phases):
        for j, rp in enumerate(reps):
            cm = cms[(mo, ph, rp)]
            a = ax[i, j]
            a.imshow(cm / cm.sum(axis=1, keepdims=True).clip(1), vmin=0, vmax=1, cmap="Blues")
            for (u, v), n in np.ndenumerate(cm):
                a.text(v, u, int(n), ha="center", va="center", fontsize=8)
            a.set_xticks(range(3), SCENARIOS, fontsize=7)
            a.set_yticks(range(3), SCENARIOS, fontsize=7)
            a.set_title(f"{mo[:6]} | {ph[:5]} | {rp}", fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir / "04_confusion_matrix.png", dpi=130)
    plt.close(fig)


def fig_components_vs_performance(results: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for (model, phase), g in results.groupby(["model", "phase"], sort=False):
        g = g.sort_values("components")
        for a, col in zip(ax, ("macro_f1", "f1_NS")):
            a.plot(g["components"], g[col], "o-", label=f"{model} / {phase}")
    ax[0].set(xlabel="componentes (24 = crudo)", ylabel="Macro-F1")
    ax[1].set(xlabel="componentes (24 = crudo)", ylabel="F1 de NS")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir / "05_components_vs_performance.png", dpi=130)
    plt.close(fig)


def fig_components_seeds(agg: pd.DataFrame, outdir: Path) -> None:
    """agg: columnas model, phase, components, macro_f1_mean/std, f1_NS_mean/std."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for (model, phase), g in agg.groupby(["model", "phase"], sort=False):
        g = g.sort_values("components")
        for a, col in zip(ax, ("macro_f1", "f1_NS")):
            a.errorbar(g["components"], g[f"{col}_mean"], yerr=g[f"{col}_std"], marker="o", capsize=3,
                       ls="--" if phase == "pre" else "-", label=f"{model} / {phase}")
    ax[0].set(xlabel="componentes (24 = crudo)", ylabel="Macro-F1 (media ± std)")
    ax[1].set(xlabel="componentes (24 = crudo)", ylabel="F1 de NS (media ± std)")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir / "06_components_vs_performance_seeds.png", dpi=130)
    plt.close(fig)
