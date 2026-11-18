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
from src.figio import save_fig


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


def x_positions(components) -> dict:
    """Las dimensiones PCA van en orden numérico; el perfil crudo (24) va aparte, a la derecha."""
    ks = sorted(int(c) for c in set(components) if int(c) != 24)
    pos = {k: float(i) for i, k in enumerate(ks)}
    pos[24] = len(ks) + 0.8
    return pos


def _components_axes(ax, components) -> None:
    pos = x_positions(components)
    ks = [k for k in pos if k != 24]
    ticks = [pos[k] for k in ks] + ([pos[24]] if 24 in set(int(c) for c in components) else [])
    labels = [str(k) for k in ks] + (["Raw (24)"] if 24 in set(int(c) for c in components) else [])
    for a in ax:
        a.set_xticks(ticks, labels)
        if 24 in set(int(c) for c in components) and ks:
            a.axvline(pos[24] - 0.5 * (pos[24] - pos[ks[-1]]), color="lightgray", lw=0.8, ls=":")
        a.set_xlabel("Number of PCA components (Raw = original 24 hourly features)")


HOW_CURVE = ("x axis: PCA dimension k (left, numeric order) and the raw 24-feature profile (separate, right). "
             "y axis: held-out score. Higher is better. Dashed lines are the pre-drift chance-level control.")


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
    save_fig(fig, outdir / "01_example_profiles.png",
             "Sanity check: one example series per scenario (EXP, REF, NS) and its mean 24-h profile before and after the drift.",
             "Top row: 7-day-averaged daily mean load over time with the drift onset in red. Bottom row: average hourly profile in the pre-drift phase (blue) vs the developed phase (red).")


def fig_pca_variance(rep24, components, outdir: Path) -> None:
    cum = np.cumsum(rep24.pca.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(range(1, 25), cum, "o-", label="Cumulative explained variance")
    for r in components:
        ax.axvline(r, color="gray", ls=":")
    ax.set(xlabel="Number of components", ylabel="Cumulative explained variance", ylim=(0, 1.02))
    fig.tight_layout()
    save_fig(fig, outdir / "02_pca_variance.png",
             "How much of the training-set variance the first k principal components capture (scaler+PCA fitted on training realizations only).",
             "Curve rises towards 1; dotted vertical lines mark the PCA dimensions used in the experiments.")


def fig_reconstruction(curve, reps, x_example, outdir: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(range(1, 25), curve, "o-", label="Test RMSE")
    ax[0].set(xlabel="Number of components", ylabel="Reconstruction RMSE, test set (W)")
    ax[1].plot(range(24), x_example, "k", lw=2, label="Original")
    for rep in reps:
        if rep.pca is not None:
            ax[1].plot(range(24), rep.reconstruct(x_example[None, :])[0], label=f"PCA-{rep.n_components}")
    ax[1].set(xlabel="Hour of day", ylabel="Heat load (W)", title="Test profile and reconstructions")
    ax[1].legend()
    fig.tight_layout()
    save_fig(fig, outdir / "03_reconstruction.png",
             "Reconstruction quality of the PCA representation on held-out realizations.",
             "Left: RMSE (watts) falls as components are added. Right: one test profile (black) against its reconstruction from each PCA dimension.")


def fig_confusions(cms: dict, outdir: Path) -> None:
    """cms[(model, phase, representation_label)] = 3x3 matrix. Celdas: % por fila (recall); los conteos van a CSV."""
    models = list(dict.fromkeys(k[0] for k in cms))
    phases = list(dict.fromkeys(k[1] for k in cms))
    reps = list(dict.fromkeys(k[2] for k in cms))
    n_rows = len(models) * len(phases)
    fig, ax = plt.subplots(n_rows, len(reps), figsize=(3.4 * len(reps), 3.3 * n_rows), squeeze=False)
    for i, (mo, ph) in enumerate((m, p) for m in models for p in phases):
        for j, rp in enumerate(reps):
            cm = cms[(mo, ph, rp)]
            frac = cm / cm.sum(axis=1, keepdims=True).clip(1)
            a = ax[i, j]
            a.imshow(frac, vmin=0, vmax=1, cmap="Blues")
            for (u, v), f in np.ndenumerate(frac):
                a.text(v, u, f"{100 * f:.0f}%", ha="center", va="center", fontsize=11,
                       color="white" if f > 0.6 else "black")
            a.set_xticks(range(3), SCENARIOS, fontsize=10)
            a.set_yticks(range(3), SCENARIOS, fontsize=10)
            a.set_title(f"{_name(mo)} | {ph.capitalize()} | {rp}", fontsize=9)
            if i == n_rows - 1:
                a.set_xlabel("Predicted class", fontsize=10)
            if j == 0:
                a.set_ylabel("True class", fontsize=10)
    fig.tight_layout()
    save_fig(fig, outdir / "04_confusion_matrix.png",
             "Row-normalised confusion matrices (share of each true class assigned to each predicted class) per model, phase and representation. Raw counts are in confusion_counts.csv.",
             "Rows = true class, columns = predicted class; diagonal = recall of that class. Darker = larger share.")


def confusion_counts_csv(cms: dict, outdir: Path) -> None:
    rows = []
    for (mo, ph, rp), cm in cms.items():
        for i, t in enumerate(SCENARIOS):
            for j, p in enumerate(SCENARIOS):
                rows.append(dict(model=mo, phase=ph, representation=rp, true=t, pred=p, count=int(cm[i, j])))
    pd.DataFrame(rows).to_csv(outdir / "confusion_counts.csv", index=False)


def fig_components_vs_performance(results: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 5), sharex=True)
    pos = x_positions(results["components"])
    for (model, phase), g in results.groupby(["model", "phase"], sort=False):
        g = g.sort_values("components")
        x = [pos[int(c)] for c in g["components"]]
        for a, col in zip(ax, ("macro_f1", "f1_NS")):
            a.plot(x, g[col], marker="o", ls="--" if phase == "pre" else "-", label=f"{_name(model)} / {phase}")
    _components_axes(ax, results["components"])
    ax[0].set_ylabel("Macro-F1")
    ax[1].set_ylabel("NS F1")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    _legend_below(fig, ax[0])
    save_fig(fig, outdir / "05_components_vs_performance.png",
             "Identification performance (single split, seed from config) against the number of ROM components.", HOW_CURVE)


def fig_components_seeds(agg: pd.DataFrame, outdir: Path) -> None:
    """agg: columns model, phase, components, macro_f1_mean/std, f1_NS_mean/std."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 5), sharex=True)
    pos = x_positions(agg["components"])
    for (model, phase), g in agg.groupby(["model", "phase"], sort=False):
        g = g.sort_values("components")
        x = [pos[int(c)] for c in g["components"]]
        for a, col in zip(ax, ("macro_f1", "f1_NS")):
            a.errorbar(x, g[f"{col}_mean"], yerr=g[f"{col}_std"], marker="o", capsize=3,
                       ls="--" if phase == "pre" else "-", label=f"{_name(model)} / {phase}")
    _components_axes(ax, agg["components"])
    ax[0].set_ylabel("Macro-F1 (mean ± SD over data partitions)")
    ax[1].set_ylabel("NS F1 (mean ± SD over data partitions)")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    _legend_below(fig, ax[0])
    save_fig(fig, outdir / "06_components_vs_performance_seeds.png",
             "Identification performance averaged over repeated realization-level splits (partition sensitivity, not full statistical uncertainty).",
             HOW_CURVE + " Error bars = SD across seeds (test sets overlap between seeds).")


def fig_progress(b_df: pd.DataFrame, outdir: Path, components=None) -> None:
    """b_df: columns model, representation, components, bin, progress_pct, balanced_accuracy."""
    if components is not None:
        b_df = b_df[b_df["components"].isin(list(components) + [24])]
    models = list(dict.fromkeys(b_df["model"]))
    fig, ax = plt.subplots(1, len(models), figsize=(6 * len(models), 4.6), squeeze=False, sharey=True)
    for a, model in zip(ax[0], models):
        g = b_df[b_df["model"] == model]
        for label, gg in g.groupby("representation", sort=False):
            gg = gg.sort_values("bin")
            a.plot(gg["bin"], gg["balanced_accuracy"], "o-", label=label)
        ticks = g.drop_duplicates("bin").sort_values("bin")
        a.set_xticks(ticks["bin"], [f"{p}%" if not str(p).endswith("%") else p for p in ticks["progress_pct"]], rotation=40, ha="right")
        a.set(title=_name(model), xlabel="Transition progress (onset to last step)", ylabel="Balanced accuracy")
        a.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, outdir / "07_progress.png",
             "Balanced accuracy on transition days as a function of how far the drift has progressed (onset to end of transition).",
             "x axis: progress bins (EXP uses a fixed 180-day window, so its progress is not physical implementation progress). Higher = better.")


def fig_curve_with_ci(agg: pd.DataFrame, deltas: pd.DataFrame | None, outdir: Path, model="random_forest",
                      name="14_rf_pca_curve.png") -> None:
    """Curva principal: puntos PCA-k con ±SD y el crudo como línea horizontal con banda; fases developed/transition."""
    g = agg[agg["model"] == model]
    ks = sorted(int(c) for c in g.loc[g["representation"] == "PCA", "components"].unique())
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), sharey=False)
    for a, phase in zip(ax, ("developed", "transition")):
        h = g[g["phase"] == phase]
        p = h[h["representation"] == "PCA"].sort_values("components")
        a.errorbar(p["components"], p["macro_f1_mean"], yerr=p["macro_f1_std"], fmt="o-", capsize=3, label="PCA-k")
        r = h[h["representation"] == "Raw"]
        if len(r):
            m, s = float(r["macro_f1_mean"].iloc[0]), float(r["macro_f1_std"].iloc[0])
            a.axhline(m, color="k", ls="--", label="Raw 24 features")
            a.axhspan(m - s, m + s, color="gray", alpha=0.2)
        a.set(title=f"{_name(model)} - {phase}", xlabel="Number of PCA components k", ylabel="Macro-F1", xticks=ks)
        a.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, outdir / name, "Main result: Macro-F1 versus PCA dimension with partition-sensitivity SD, against the raw profile.",
             "Points = mean over data partitions, bars = SD. The dashed line is the raw 24-feature profile with its ±SD band; the raw profile is NOT a PCA-24 point.")


def fig_class_progress(summary: pd.DataFrame, outdir: Path, model: str, reps: list[str], bins: int) -> None:
    """summary: columns representation, scenario, progress_bin(int), precision_mean/sd, recall_mean/sd, f1_mean/sd."""
    metrics = ("precision", "recall", "f1")
    fig, ax = plt.subplots(len(metrics), len(reps), figsize=(3.9 * len(reps), 3.0 * len(metrics)), squeeze=False, sharey=True)
    s = summary[summary["model"] == model]
    for j, rp in enumerate(reps):
        for i, m in enumerate(metrics):
            a = ax[i, j]
            for c in SCENARIOS:
                g = s[(s["representation"] == rp) & (s["scenario"] == c)].sort_values("progress_bin")
                a.errorbar(g["progress_bin"], g[f"{m}_mean"], yerr=g[f"{m}_sd"], marker="o", capsize=2, label=c)
            a.set_ylim(0, 1.02)
            a.set_xticks(range(bins), [f"{100 * b // bins}-{100 * (b + 1) // bins}%" for b in range(bins)], rotation=40, ha="right", fontsize=8)
            if i == 0:
                a.set_title(rp, fontsize=10)
            if j == 0:
                a.set_ylabel(m.capitalize())
            if i == len(metrics) - 1:
                a.set_xlabel("Transition progress")
    ax[0, 0].legend(fontsize=8)
    fig.suptitle(f"{_name(model)}: per-class precision, recall and F1 during the transition (mean ± SD over splits)")
    fig.tight_layout()
    save_fig(fig, outdir / f"08_class_progress_{model}.png",
             "Per-class precision, recall and F1 across transition-progress bins, averaged over severities and data partitions.",
             "Rows: precision, recall, F1. Columns: representation. One line per true scenario. Precision counts all predictions of that class in the same bin, whatever their true class.")


def fig_exp_false_positives(fp: pd.DataFrame, outdir: Path, model: str, reps: list[str], bins: int) -> None:
    """fp: columns model, representation, progress_bin, true, share (de las predicciones EXP)."""
    fig, ax = plt.subplots(1, len(reps), figsize=(3.9 * len(reps), 4), squeeze=False, sharey=True)
    for a, rp in zip(ax[0], reps):
        g = fp[(fp["model"] == model) & (fp["representation"] == rp)]
        bottom = np.zeros(bins)
        for t, col in zip(SCENARIOS, ("tab:green", "tab:orange", "tab:red")):
            v = np.array([g[(g["progress_bin"] == b) & (g["true"] == t)]["share"].sum() for b in range(bins)])
            a.bar(range(bins), v, bottom=bottom, color=col, label=f"true {t}")
            bottom += v
        a.set_xticks(range(bins), [f"{100 * b // bins}-{100 * (b + 1) // bins}%" for b in range(bins)], rotation=40, ha="right", fontsize=8)
        a.set(title=rp, xlabel="Transition progress", ylim=(0, 1.0))
    ax[0, 0].set_ylabel("Composition of 'predicted EXP' days")
    ax[0, 0].legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, outdir / f"09_exp_false_positives_{model}.png",
             "Who the days predicted as EXP really are during the transition: tests whether low EXP F1 is driven by REF/NS days wrongly called EXP.",
             "Each bar sums to 1 (all EXP predictions in that progress bin). Green = correct (true EXP); orange/red = false positives from REF/NS.")
