"""Etapas de revisión (guía computacional de Kai). Cada etapa es una función `stage_*(cfg, ds, out)` independiente
que escribe en results/<modo>/revision/<etapa>/ (CSV + MD + figuras con su .txt). Orquestadas por run_all.py."""
from __future__ import annotations

import io
import itertools
import threading
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src import perturb as pt
from src.analysis import (
    CLASSES4, bin_label, gaussian_hellinger, hellinger_discrete, hist_prob, ns_indicators, normalize_profiles,
    prf_table, progress_bins, window_masks,
)
from src.data import SCENARIOS, Dataset, assign_phase, load_baseline
from src.evaluation import classification_metrics, to_markdown
from src.features import rows, split_realizations
from src.figio import save_fig
from src.models import RF_DEFAULT, make_model
from src.rom import fit_representation
from src.runtime import available_seeds, hardware, load_predictions, out_dir, rcfg, rev_dir
from src.weather import add_calendar, has_weather, temperature_bins

PHASES = ("developed", "transition", "pre")
NIGHT = [22, 23, 0, 1, 2, 3, 4, 5]
DAY = list(range(6, 22))
MORNING = [6, 7, 8, 9]
F1COLS = ("macro_f1", "balanced_accuracy", "f1_EXP", "f1_REF", "f1_NS")


# --------------------------------------------------------------------------- utilidades comunes


def seeds_for(cfg, key="n_seeds_secondary", default=5):
    return [cfg["seed"] + i for i in range(int(rcfg(cfg, key, default)))]


def do_split(ds, cfg, seed):
    return split_realizations(ds.realizations, cfg.get("split", {}).get("train_fraction", 0.8), seed)


def train_idx(ds, split, cfg, seed, phases=("developed",)):
    """Índices (en ds.meta) de las filas de entrenamiento; misma selección y tope que experiments/03_ml.train_arrays."""
    mask = ds.meta["realization"].isin(split.train).to_numpy() & ds.meta["phase"].isin(list(phases)).to_numpy()
    idx = np.flatnonzero(mask)
    cap = cfg.get("ml", {}).get("max_train_rows")
    if cap and len(idx) > cap:
        idx = idx[np.random.default_rng(seed).choice(len(idx), cap, replace=False)]
    return idx


def focus_reps(X_fit, cfg):
    return [fit_representation(X_fit, None)] + [fit_representation(X_fit, k) for k in cfg["rom"].get("focus_components", [2, 5, 10])]


def rlabel(rep):
    return "Raw" if rep.pca is None else f"PCA-{rep.n_components}"


def phase_metrics(meta_te, y_pred, y_true=None, extra=None):
    y_true = meta_te["scenario"].to_numpy() if y_true is None else y_true
    ph = meta_te["phase"].to_numpy()
    out = []
    for p in PHASES:
        s = ph == p
        if s.any():
            out.append({**(extra or {}), "phase": p, **classification_metrics(y_true[s], np.asarray(y_pred)[s]), "n_rows": int(s.sum())})
    return out


def relabel(df):
    """ml.run devuelve representation='PCA' para todos los k; aquí se usa 'Raw' / 'PCA-k'."""
    df = df.copy()
    df["representation"] = np.where(df["components"] == 24, "Raw", "PCA-" + df["components"].astype(str))
    return df


def summarize(df, keys, cols=F1COLS):
    g = df.groupby(keys, sort=False)[list(cols)].agg(["mean", "std"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    return g.reset_index()


def md_table(df, cols=None):
    return to_markdown(df if cols is None else df[cols])


def _grouped_lines(ax, df, x, y, hue, yerr=None, **kw):
    for h, g in df.groupby(hue, sort=False):
        g = g.sort_values(x)
        if yerr:
            ax.errorbar(g[x], g[y], yerr=g[yerr], marker="o", capsize=2, label=str(h), **kw)
        else:
            ax.plot(g[x], g[y], marker="o", label=str(h), **kw)


def _fit_predict(model, seed, Ztr, ytr, Zte, params=None):
    clf = make_model(model, seed, params).fit(Ztr, ytr)
    return clf, clf.predict(Zte)


# --------------------------------------------------------------------------- P4: tiempo y estación


def stage_weather(cfg, ds, out):
    d = rev_dir(cfg, "P4_weather")
    if not has_weather(ds.meta):
        (d / "SKIPPED.txt").write_text("No se encontró weather.csv con columna de temperatura reconocible. "
                                       "Fija revision.weather.temp_col / time_col en el YAML.\n", encoding="utf-8")
        return
    meta = ds.meta
    rng = np.random.default_rng(cfg["seed"])
    # --- 1. ¿Comparten todas las realizaciones la misma trayectoria meteorológica (y base)?
    cover = meta.groupby("realization")["date"].agg(["min", "max", "count"])
    sample = rng.choice(ds.realizations["realization"].to_numpy(), size=min(60, len(ds.realizations)), replace=False)
    temp = meta.drop_duplicates("date").set_index("date")["temp_mean"]
    ser, resid = {}, {}
    for r in sample:
        m = meta[(meta["realization"] == r) & (meta["phase"] == "pre")]
        if len(m) < 60:
            continue
        y = pd.Series(ds.X[m.index.to_numpy()].mean(axis=1), index=pd.DatetimeIndex(m["date"].values))
        T = temp.reindex(y.index).to_numpy()
        A = np.c_[np.ones(len(T)), T, T ** 2]
        beta, *_ = np.linalg.lstsq(A, y.to_numpy(), rcond=None)
        ser[r], resid[r] = y, pd.Series(y.to_numpy() - A @ beta, index=y.index)
    keys, pairs = list(ser), []
    for _ in range(300):
        if len(keys) < 2:
            break
        a, b = rng.choice(keys, size=2, replace=False)
        ix = ser[a].index.intersection(ser[b].index)
        if len(ix) >= 60:
            pairs.append((np.corrcoef(ser[a][ix], ser[b][ix])[0, 1], np.corrcoef(resid[a][ix], resid[b][ix])[0, 1]))
    pr = np.array(pairs) if pairs else np.full((1, 2), np.nan)
    own = [pd.Series(ser[r]).corr(temp.reindex(ser[r].index), method="spearman") for r in ser]
    q = lambda v: ", ".join(f"{x:.3f}" for x in np.nanquantile(v, [0.05, 0.5, 0.95]))
    txt = [
        "# ¿Trayectoria meteorológica / base compartida entre realizaciones?\n",
        f"- weather.csv es un único fichero: **todas las realizaciones comparten la misma serie de tiempo por construcción**.",
        f"- Cobertura de fechas: {cover['min'].nunique()} fechas de inicio y {cover['max'].nunique()} de fin distintas entre {len(cover)} realizaciones.",
        f"- Spearman (carga diaria media vs temperatura, días pre, {len(own)} realizaciones) p5/mediana/p95: {q(own)}",
        f"- Correlación entre pares de realizaciones (carga diaria media, días pre comunes; {len(pairs)} pares) p5/mediana/p95: {q(pr[:, 0])}",
        f"- Igual tras quitar un ajuste cuadrático en la temperatura (residuo): {q(pr[:, 1])}",
        "",
        "Lectura: correlación bruta alta = las series se mueven con el mismo clima/calendario. Si el **residuo** sigue alto, "
        "las realizaciones comparten además una base común (no solo el tiempo), de modo que fecha y trayectoria pueden "
        "inducir dependencia aunque el split sea por realización. El control `pre` tranquiliza pero no lo descarta.",
    ]
    (d / "shared_trajectory_check.md").write_text("\n".join(txt) + "\n", encoding="utf-8")

    # --- 2. Líneas base solo-fecha / solo-temperatura / ambas, y perfil vs perfil+temperatura (sin test en el ajuste)
    def datefeat(m):
        a = 2 * np.pi * m["doy"].to_numpy() / 365.25
        return np.c_[np.sin(a), np.cos(a), m["year"].to_numpy()]

    tcols = ["temp_mean", "temp_min", "temp_max", "hdd"]
    rows_b, rows_p = [], []
    for seed in seeds_for(cfg):
        split = do_split(ds, cfg, seed)
        ti = train_idx(ds, split, cfg, seed)
        mt = meta.iloc[ti]
        te_mask = meta["realization"].isin(split.test).to_numpy()
        mte = meta[te_mask].reset_index(drop=True)
        fill = mt[tcols].mean()
        feats = {"date_only": datefeat, "temperature_only": lambda m: m[tcols].fillna(fill).to_numpy(),
                 "date+temperature": lambda m: np.c_[datefeat(m), m[tcols].fillna(fill).to_numpy()]}
        for name, f in feats.items():
            clf = make_model("random_forest", seed).fit(f(mt), mt["scenario"].to_numpy())
            rows_b += phase_metrics(mte, clf.predict(f(mte)), extra=dict(seed=seed, features=name))
        X_fit = ds.X[meta["realization"].isin(split.train).to_numpy()]
        sc = StandardScaler().fit(mt[tcols].fillna(fill))
        tf = lambda m: sc.transform(m[tcols].fillna(fill))
        for rep in focus_reps(X_fit, cfg):
            Ztr, Zte = rep.transform(ds.X[ti]), rep.transform(ds.X[te_mask])
            for model in cfg["models"]:
                a = make_model(model, seed).fit(Ztr, mt["scenario"].to_numpy())
                b = make_model(model, seed).fit(np.c_[Ztr, tf(mt)], mt["scenario"].to_numpy())
                rows_p += phase_metrics(mte, a.predict(Zte), extra=dict(seed=seed, model=model, representation=rlabel(rep), variant="profile"))
                rows_p += phase_metrics(mte, b.predict(np.c_[Zte, tf(mte)]), extra=dict(seed=seed, model=model, representation=rlabel(rep), variant="profile+temperature"))
    b_df, p_df = pd.DataFrame(rows_b), pd.DataFrame(rows_p)
    b_df.to_csv(d / "date_temperature_only_baselines.csv", index=False)
    p_df.to_csv(d / "profile_vs_profile_plus_temperature.csv", index=False)
    sb, sp = summarize(b_df, ["features", "phase"]), summarize(p_df, ["model", "representation", "variant", "phase"])
    (d / "weather_summary.md").write_text(
        "# Líneas base solo-fecha / solo-temperatura (RF; azar ≈ 0.33)\n\n" + md_table(sb, ["features", "phase", "macro_f1_mean", "macro_f1_std", "balanced_accuracy_mean"]) +
        "\n\n# Perfil vs perfil + temperatura\n\n" + md_table(sp, ["model", "representation", "variant", "phase", "macro_f1_mean", "macro_f1_std"]) +
        "\n\nSi la temperatura mejora el resultado, **no** se puede concluir que la representación sola sea invariante al clima.\n", encoding="utf-8")

    # --- 3. Rendimiento por terciles de temperatura, estación, y progreso×estación (desde las predicciones guardadas)
    seeds = available_seeds(out)
    bins = int(rcfg(cfg, "progress_bins", 5))
    allr = []
    for seed in seeds:
        te, preds = load_predictions(out, seed)
        split = do_split(ds, cfg, seed)
        tr_temp = meta.loc[meta["realization"].isin(split.train), "temp_mean"].to_numpy()
        mt = meta.iloc[te].reset_index(drop=True)
        pb = progress_bins(mt, bins)
        mt = mt.assign(tbin=temperature_bins(tr_temp, mt["temp_mean"].to_numpy()), progress_bin=[bin_label(int(x), bins) for x in pb])
        for (model, rep), code in preds.items():
            pred = np.array(SCENARIOS, dtype=object)[np.minimum(code, 2)]
            for name, strata in (("temperature_tercile", ["phase", "tbin"]), ("season", ["phase", "season"]),
                                 ("progress_x_season", ["phase", "season", "progress_bin"])):
                t = prf_table(mt, pred, strata)
                t.insert(0, "breakdown", name), t.insert(0, "representation", rep), t.insert(0, "model", model), t.insert(0, "seed", seed)
                allr.append(t)
    if allr:
        full = pd.concat(allr, ignore_index=True)
        full.to_csv(d / "performance_by_weather_strata_all_splits.csv", index=False)
        keys = [c for c in full.columns if c in ("model", "representation", "breakdown", "phase", "tbin", "season", "progress_bin", "scenario")]
        s = full.groupby(keys, sort=False, dropna=False).agg(precision_mean=("precision", "mean"), recall_mean=("recall", "mean"),
                                                             f1_mean=("f1", "mean"), f1_sd=("f1", "std"), n_profiles_mean=("n_profiles", "mean")).reset_index()
        s.to_csv(d / "performance_by_weather_strata_summary.csv", index=False)
        _fig_weather(s, d)


def _fig_weather(s, d):
    for breakdown, col, fname in (("season", "season", "P4_f1_by_season.png"), ("temperature_tercile", "tbin", "P4_f1_by_temperature.png")):
        g = s[(s["breakdown"] == breakdown) & (s["model"] == "random_forest") & (s["phase"] == "transition")]
        if g.empty:
            continue
        reps = [r for r in g["representation"].unique()][:4]
        fig, ax = plt.subplots(1, len(reps), figsize=(3.8 * len(reps), 4), sharey=True, squeeze=False)
        order = sorted(g[col].unique())
        for a, rp in zip(ax[0], reps):
            gg = g[g["representation"] == rp]
            w = 0.25
            for i, c in enumerate(SCENARIOS):
                v = [gg[(gg[col] == o) & (gg["scenario"] == c)]["f1_mean"].mean() for o in order]
                a.bar(np.arange(len(order)) + i * w, v, w, label=c)
            a.set_xticks(np.arange(len(order)) + w, order, rotation=30)
            a.set(title=rp, ylim=(0, 1.02))
        ax[0, 0].set_ylabel("F1 on transition days (mean over splits)")
        ax[0, 0].legend(fontsize=8)
        fig.tight_layout()
        save_fig(fig, d / fname, f"Random-forest F1 per scenario on transition days, by {breakdown.replace('_', ' ')}.",
                 "Bars = F1 of each true scenario in each weather stratum; one panel per representation.")


# --------------------------------------------------------------------------- P5: normalización


def stage_normalization(cfg, ds, out):
    d = rev_dir(cfg, "P5_normalization")
    modes = list(rcfg(cfg, "normalization_modes", ["daily_mean_normalized", "daily_zscore_shape"]))
    ml = __import__("importlib").import_module("experiments.03_ml")
    allr = []
    for mode in modes:
        dsm = Dataset(X=normalize_profiles(ds.X, mode), meta=ds.meta, realizations=ds.realizations)
        for seed in seeds_for(cfg):
            split = do_split(ds, cfg, seed)
            X_fit, _ = rows(dsm, split.train)
            reps = focus_reps(X_fit, cfg)
            res = ml.run(dsm, split, reps, {**cfg, "seed": seed}, out, save=False)
            allr.append(relabel(res.assign(seed=seed, mode=mode)))
        print(f"  normalización {mode} hecha", flush=True)
    base = pd.read_csv(out / "results_seeds_raw.csv") if (out / "results_seeds_raw.csv").exists() else None
    if base is not None:
        keep = base[base["seed"].isin(seeds_for(cfg)) & (base["components"].isin([24] + list(cfg["rom"].get("focus_components", []))))]
        allr.append(relabel(keep.assign(mode="raw_level_shape")))
    r = pd.concat(allr, ignore_index=True)
    r.to_csv(d / "normalization_results.csv", index=False)
    s = summarize(r, ["model", "mode", "phase", "representation", "components"])
    s.to_csv(d / "normalization_summary.csv", index=False)
    ref = s[s["mode"] == "raw_level_shape"].set_index(["model", "phase", "representation", "components"])
    s2 = s.join(ref[[f"{c}_mean" for c in F1COLS]], on=["model", "phase", "representation", "components"], rsuffix="_rawlevel")
    for c in F1COLS:
        s2[f"delta_{c}"] = s2[f"{c}_mean"] - s2[f"{c}_mean_rawlevel"]
    s2.to_csv(d / "normalization_deltas_vs_raw_level_shape.csv", index=False)
    (d / "normalization_summary.md").write_text(
        "# Información en nivel vs forma (RF/LR; media sobre semillas)\n\n"
        "`daily_energy_normalized` (x/suma) no se ejecuta: tras el StandardScaler es idéntico a `daily_mean_normalized` "
        "(difieren en una constante ×24); se verifica en tests.\n\n"
        + "\n\n".join(f"## {m} — {p}\n\n" + md_table(g, ["mode", "representation", "macro_f1_mean", "macro_f1_std", "f1_EXP_mean", "f1_REF_mean", "f1_NS_mean"])
                      for (m, p), g in s.groupby(["model", "phase"], sort=False)) + "\n", encoding="utf-8")
    g = s[(s["model"] == "random_forest") & (s["phase"].isin(["developed", "transition"]))]
    if len(g):
        fig, ax = plt.subplots(2, 4, figsize=(15, 7), sharex=True, sharey="row")
        for i, ph in enumerate(("developed", "transition")):
            for j, col in enumerate(("macro_f1", "f1_EXP", "f1_REF", "f1_NS")):
                gg = g[g["phase"] == ph]
                for mode, h in gg.groupby("mode", sort=False):
                    h = h.sort_values("components")
                    ax[i, j].plot(range(len(h)), h[f"{col}_mean"], "o-", label=mode)
                    ax[i, j].set_xticks(range(len(h)), h["representation"], rotation=30, fontsize=8)
                ax[i, j].set_title(f"{col} - {ph}", fontsize=9)
        ax[0, 0].legend(fontsize=7)
        fig.tight_layout()
        save_fig(fig, d / "P5_normalization.png", "Information source test: how much identification depends on load level vs daily shape (random forest).",
                 "Each line is a profile representation mode. Columns: Macro-F1 and per-scenario F1. If EXP/REF drop when level is removed but NS holds, NS is identified by shape.")


# --------------------------------------------------------------------------- P6: interpretación PCA y NS


def _rank(v):
    return pd.Series(v).rank().to_numpy()


def _partial_spearman(a, b, c):
    ra, rb, rc = _rank(a), _rank(b), _rank(c)
    def res(x, z):
        A = np.c_[np.ones(len(z)), z]
        return x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
    return float(np.corrcoef(res(ra, rc), res(rb, rc))[0, 1])


def stage_pca_interpretation(cfg, ds, out):
    d = rev_dir(cfg, "P6_pca_ns_interpretation")
    seed = cfg["seed"]
    split = do_split(ds, cfg, seed)
    meta = add_calendar(ds.meta)
    X_fit, _ = rows(ds, split.train)
    rep = fit_representation(X_fit, 24)
    comps = rep.pca.components_[:6].copy()
    for i in range(len(comps)):  # el signo de una PC es arbitrario: se fija para que el mayor |peso| sea positivo
        if comps[i][np.abs(comps[i]).argmax()] < 0:
            comps[i] = -comps[i]
    fig, ax = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    for i, a in enumerate(ax.ravel()):
        a.plot(range(24), comps[i], "o-", label=f"PC{i + 1}")
        a.axhline(0, color="gray", lw=0.6)
        a.set(title=f"PC{i + 1} ({100 * rep.pca.explained_variance_ratio_[i]:.1f}% var.)", xlabel="Hour of day", ylabel="Loading (standardised hours)")
    fig.tight_layout()
    save_fig(fig, d / "P6_pca_loadings.png", "Hourly loadings of the first six principal components (scaler+PCA fitted on training realizations).",
             "Sign of each component is arbitrary (flipped so the largest-magnitude loading is positive). Look for night vs day contrasts (setback) or all-hours-same-sign (level).")
    # indicadores y correlaciones
    ind = ns_indicators(ds.X, rcfg(cfg, "night_hours", NIGHT), rcfg(cfg, "day_hours", DAY), rcfg(cfg, "morning_hours", MORNING))
    n = len(ds.X)
    sub = np.sort(np.random.default_rng(seed).choice(n, size=min(n, 200_000), replace=False))
    S = rep.transform(ds.X[sub])[:, :6]
    ang = 2 * np.pi * meta["doy"].to_numpy()[sub] / 365.25
    var = pd.DataFrame({
        "daily_mean": ind["daily_mean"].to_numpy()[sub], "night_day_ratio": ind["night_day_ratio"].to_numpy()[sub],
        "morning_rebound": ind["morning_rebound"].to_numpy()[sub], "peak_hour": ind["peak_hour"].to_numpy()[sub],
        "cos_doy": np.cos(ang), "sin_doy": np.sin(ang),
        "phase_ordinal": meta["phase"].map({"pre": 0, "transition": 1, "developed": 2}).to_numpy()[sub],
        "severity_ordinal": meta["severity"].map({"L": 1, "M": 2, "H": 3}).to_numpy()[sub]})
    if has_weather(meta):
        var["temp_mean"] = meta["temp_mean"].to_numpy()[sub]
    var = var.fillna(var.median())
    sp = pd.DataFrame({v: [np.corrcoef(_rank(S[:, i]), _rank(var[v]))[0, 1] for i in range(6)] for v in var}, index=[f"PC{i + 1}" for i in range(6)])
    sp.to_csv(d / "pc_score_spearman.csv")
    if "temp_mean" in var:
        pp = pd.DataFrame({v: [_partial_spearman(S[:, i], var[v], var["temp_mean"]) for i in range(6)] for v in var if v != "temp_mean"},
                          index=sp.index)
        pp.to_csv(d / "pc_score_partial_spearman_given_temperature.csv")
    fig, a = plt.subplots(figsize=(1.1 * len(sp.columns) + 2, 4.5))
    a.imshow(sp.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
    for (u, v), val in np.ndenumerate(sp.to_numpy()):
        a.text(v, u, f"{val:.2f}", ha="center", va="center", fontsize=9)
    a.set_xticks(range(len(sp.columns)), sp.columns, rotation=40, ha="right")
    a.set_yticks(range(len(sp.index)), sp.index)
    a.set_title("Spearman correlation of PC scores with physical variables")
    fig.tight_layout()
    save_fig(fig, d / "P6_pc_correlations.png", "Which physical quantities each principal component tracks.", "Red = positive, blue = negative rank correlation; |value| near 1 = the PC is essentially that variable.")
    mt = meta[["scenario", "phase", "severity"]].iloc[sub].reset_index(drop=True)
    pd.concat([mt, pd.DataFrame(S[:, :3], columns=["PC1", "PC2", "PC3"])], axis=1).groupby(["scenario", "phase", "severity"]).mean() \
        .to_csv(d / "pc_mean_scores_by_scenario_phase_severity.csv")
    # distribuciones de los indicadores
    full = pd.concat([meta[["scenario", "phase", "severity"]].reset_index(drop=True), ind], axis=1)
    desc = full.groupby(["scenario", "phase", "severity"]).agg(
        **{f"{c}_{s}": (c, f) for c in ("night_day_ratio", "morning_rebound") for s, f in (("mean", "mean"), ("sd", "std"), ("q05", lambda x: x.quantile(.05)), ("q50", "median"), ("q95", lambda x: x.quantile(.95)))}, n=("daily_mean", "size")).reset_index()
    desc.to_csv(d / "ns_indicators_distribution.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for a, c in zip(ax, ("night_day_ratio", "morning_rebound")):
        pos, labs = [], []
        for i, sc in enumerate(SCENARIOS):
            for j, ph in enumerate(("pre", "transition", "developed")):
                v = full[(full["scenario"] == sc) & (full["phase"] == ph)][c].dropna()
                if len(v):
                    bp = a.boxplot(v.sample(min(len(v), 20000), random_state=0), positions=[i * 4 + j], widths=0.8, showfliers=False)
                    pos.append(i * 4 + j), labs.append(f"{sc}\n{ph[:4]}")
        a.set_xticks(pos, labs, fontsize=7)
        a.set(ylabel=c.replace("_", " "), title=c.replace("_", " "))
    fig.tight_layout()
    save_fig(fig, d / "P6_ns_indicators.png", "Predefined NS indicators (night/day ratio, morning rebound) by scenario and phase.", "Box = interquartile range, line = median; groups of three boxes = pre / transition / developed for EXP, REF, NS.")
    # efecto horario (verifica a posteriori el horario de setback, solo con train)
    trm = meta["realization"].isin(split.train).to_numpy()
    fig, a = plt.subplots(figsize=(7, 4.5))
    for sc in SCENARIOS:
        pre = ds.X[trm & (meta["scenario"] == sc).to_numpy() & (meta["phase"] == "pre").to_numpy()].mean(0)
        dev = ds.X[trm & (meta["scenario"] == sc).to_numpy() & (meta["phase"] == "developed").to_numpy()].mean(0)
        a.plot(range(24), 100 * (dev - pre) / pre, "o-", label=sc)
    a.axhline(0, color="gray", lw=0.6)
    a.set(xlabel="Hour of day (timestamp as stored)", ylabel="Developed vs pre-drift change (%)")
    a.legend()
    fig.tight_layout()
    save_fig(fig, d / "P6_hourly_effect.png", "Where in the day each intervention changes the load (training realizations only).",
             "Use it to check that the predefined night hours match the real setback window; if timestamps are UTC, shift night_hours in the config.")
    # línea base de una variable: logística + umbral elegido en train; AUC, F1, Brier, ECE
    tr = np.flatnonzero(trm & (meta["phase"] == "developed").to_numpy())
    te = np.flatnonzero(meta["realization"].isin(split.test).to_numpy())
    yb = lambda idx: (meta["scenario"].to_numpy()[idx] == "NS").astype(int)
    res, curves = [], {}
    for name, cols in (("night_day_ratio", ["night_day_ratio"]), ("night_day+morning", ["night_day_ratio", "morning_rebound"])):
        F = ind[cols].to_numpy()
        ok = np.isfinite(F).all(axis=1)
        trk, tek = tr[ok[tr]], te[ok[te]]
        sc_ = StandardScaler().fit(F[trk])
        lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc_.transform(F[trk]), yb(trk))
        ptr = lr.predict_proba(sc_.transform(F[trk]))[:, 1]
        thr = max(np.linspace(0.05, 0.95, 91), key=lambda t: f1_score(yb(trk), ptr >= t, zero_division=0))
        for ph in PHASES:
            idx = tek[(meta["phase"].to_numpy()[tek] == ph)]
            if len(idx) == 0:
                continue
            p = lr.predict_proba(sc_.transform(F[idx]))[:, 1]
            y = yb(idx)
            edges = np.linspace(0, 1, 11)
            b = np.clip(np.digitize(p, edges) - 1, 0, 9)
            ece = sum(abs(y[b == k].mean() - p[b == k].mean()) * (b == k).mean() for k in range(10) if (b == k).any())
            res.append(dict(indicator=name, phase=ph, threshold_from_train=thr, f1_NS=f1_score(y, p >= thr, zero_division=0),
                            auc=roc_auc_score(y, p) if len(set(y)) > 1 else np.nan, brier=brier_score_loss(y, p), ece=ece, n=len(idx)))
            if ph == "developed":
                curves[name] = (np.array([p[b == k].mean() if (b == k).any() else np.nan for k in range(10)]),
                                np.array([y[b == k].mean() if (b == k).any() else np.nan for k in range(10)]))
    r = pd.DataFrame(res)
    base = pd.read_csv(out / "results_seeds_raw.csv") if (out / "results_seeds_raw.csv").exists() else None
    if base is not None:
        c = base[(base["seed"] == seed) & (base["model"] == "random_forest")][["phase", "representation", "components", "f1_NS"]]
        c.to_csv(d / "reference_rf_f1_NS_same_split.csv", index=False)
    r.to_csv(d / "ns_single_indicator_baseline.csv", index=False)
    fig, a = plt.subplots(figsize=(5, 5))
    a.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect calibration")
    for name, (pp, yy) in curves.items():
        a.plot(pp, yy, "o-", label=name)
    a.set(xlabel="Predicted P(NS)", ylabel="Observed NS frequency")
    a.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, d / "P6_calibration.png", "Reliability of the single-indicator NS-vs-rest logistic baseline on developed test days.", "Closer to the diagonal = better calibrated probabilities.")


# --------------------------------------------------------------------------- P7: Hellinger


def stage_hellinger(cfg, ds, out):
    """Distancia de Hellinger pre vs fase posterior, por realización (bootstrap por realización). DESCRIPTIVA: usa todas las
    realizaciones; los bins se definen con entrenamiento. 1-D para indicadores; gaussiana aproximada para PCA-5 (nunca 24-D)."""
    d = rev_dir(cfg, "P7_hellinger")
    seed = cfg["seed"]
    split = do_split(ds, cfg, seed)
    meta = add_calendar(ds.meta)
    ind = ns_indicators(ds.X, rcfg(cfg, "night_hours", NIGHT), rcfg(cfg, "day_hours", DAY), rcfg(cfg, "morning_hours", MORNING))
    nb = int(rcfg(cfg, "hellinger_bins", 20))
    trm = meta["realization"].isin(split.train).to_numpy()
    vars1 = ("daily_mean", "night_day_ratio", "morning_rebound")
    edges = {}
    for v in vars1:
        x = ind[v].to_numpy()
        e = np.unique(np.nanquantile(x[trm & np.isfinite(x)], np.linspace(0, 1, nb + 1)))
        edges[v] = e if len(e) > 2 else np.linspace(np.nanmin(x), np.nanmax(x) + 1e-9, nb + 1)
    X_fit, _ = rows(ds, split.train)
    Z = fit_representation(X_fit, 5).transform(ds.X)
    pbins = int(rcfg(cfg, "progress_bins", 5))
    pb = progress_bins(meta, pbins)
    ph, season = meta["phase"].to_numpy(), meta["season"].to_numpy() if "season" in meta else np.array([""] * len(meta))
    idx_by_r = meta.groupby("realization").indices
    rid = ds.realizations.set_index("realization")
    recs = []
    for r, idx in idx_by_r.items():
        pre = idx[ph[idx] == "pre"]
        if len(pre) < 20:
            continue
        comps = [("developed", idx[ph[idx] == "developed"])]
        comps += [(f"transition_bin{b}", idx[(ph[idx] == "transition") & (pb[idx] == b)]) for b in range(pbins)]
        for s in sorted(set(season[idx])):
            comps.append((f"developed_{s}_matched", (idx[(ph[idx] == "developed") & (season[idx] == s)], pre[season[pre] == s])))
        for name, post in comps:
            pre_i = pre
            if isinstance(post, tuple):
                post, pre_i = post
            if len(post) < 20 or len(pre_i) < 20:
                continue
            for v in vars1:
                e = edges[v]
                a, b = ind[v].to_numpy()[pre_i], ind[v].to_numpy()[post]
                a, b = a[np.isfinite(a)], b[np.isfinite(b)]
                if len(a) >= 20 and len(b) >= 20:
                    recs.append((r, rid.loc[r, "scenario"], rid.loc[r, "severity"], name, v, hellinger_discrete(hist_prob(a, e), hist_prob(b, e))))
            if min(len(post), len(pre_i)) >= 15:
                recs.append((r, rid.loc[r, "scenario"], rid.loc[r, "severity"], name, "pca5_gaussian", gaussian_hellinger(Z[pre_i], Z[post])))
    H = pd.DataFrame(recs, columns=["realization", "scenario", "severity", "comparison", "variable", "hellinger"])
    H.to_csv(d / "hellinger_by_realization.csv", index=False)
    rng = np.random.default_rng(seed)
    B = int(rcfg(cfg, "hellinger_bootstrap", 200))
    rows_ = []
    for k, g in H.groupby(["scenario", "severity", "comparison", "variable"], sort=False):
        v = g["hellinger"].to_numpy()
        bm = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
        rows_.append(dict(zip(["scenario", "severity", "comparison", "variable"], k), mean_H=v.mean(), ci_low=np.percentile(bm, 2.5),
                          ci_high=np.percentile(bm, 97.5), n_realizations=len(v)))
    S = pd.DataFrame(rows_)
    S.to_csv(d / "hellinger_summary.csv", index=False)
    (d / "hellinger_notes.md").write_text(
        "# Hellinger (descriptivo)\n\nHellinger ∈ [0,1]: 0 = misma distribución, 1 = disjuntas. Se mide pre vs fase posterior **dentro de cada "
        "realización**; IC por bootstrap de realizaciones. `*_matched` compara dentro de la misma estación para no mezclar estacionalidad. "
        "Las distancias discretas están sesgadas al alza con pocas muestras (bins definidos con entrenamiento). La de PCA-5 es una "
        "aproximación gaussiana (Bhattacharyya). Mide cambio estadístico comparable, **no** equivalencia física de L/M/H entre intervenciones.\n", encoding="utf-8")
    g = S[S["comparison"] == "developed"]
    if len(g):
        vs = list(dict.fromkeys(g["variable"]))
        fig, ax = plt.subplots(1, len(vs), figsize=(4.2 * len(vs), 4.2), sharey=True, squeeze=False)
        cells = [(sc, sv) for sc in SCENARIOS for sv in ("L", "M", "H")]
        for a, v in zip(ax[0], vs):
            gg = g[g["variable"] == v].set_index(["scenario", "severity"])
            y = [gg["mean_H"].get(c, np.nan) for c in cells]
            lo = [max(gg["mean_H"].get(c, np.nan) - gg["ci_low"].get(c, np.nan), 0) for c in cells]
            hi = [max(gg["ci_high"].get(c, np.nan) - gg["mean_H"].get(c, np.nan), 0) for c in cells]
            a.bar(range(len(cells)), y, yerr=[lo, hi], capsize=2)
            a.set_xticks(range(len(cells)), [f"{s}-{v_}" for s, v_ in cells], rotation=60, fontsize=8)
            a.set_title(v, fontsize=9)
        ax[0, 0].set_ylabel("Hellinger distance, pre vs developed")
        fig.tight_layout()
        save_fig(fig, d / "P7_hellinger_developed.png", "Distributional shift between pre-drift and developed days per scenario and severity.", "Bars = mean over realizations, whiskers = 95% bootstrap interval over realizations.")
    g = S[S["comparison"].str.startswith("transition_bin") & S["variable"].isin(["daily_mean", "pca5_gaussian"])]
    if len(g):
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for a, v in zip(ax, ("daily_mean", "pca5_gaussian")):
            for sc in SCENARIOS:
                gg = g[(g["variable"] == v) & (g["scenario"] == sc)].groupby("comparison")["mean_H"].mean()
                gg = gg.reindex(sorted(gg.index, key=lambda s: int(s.replace("transition_bin", ""))))
                a.plot(range(len(gg)), gg.values, "o-", label=sc)
            a.set(title=v, xlabel="Transition progress bin", ylabel="Hellinger vs pre")
        ax[0].legend()
        fig.tight_layout()
        save_fig(fig, d / "P7_hellinger_progress.png", "How the distributional shift grows along the transition.", "x axis: progress bins (0 = first fifth of the transition). Mean over severities and realizations.")


# --------------------------------------------------------------------------- P8: control negativo BL y cuarta clase


def _four_metrics(y_true, y_pred):
    present = [c for c in CLASSES4 if (y_true == c).any()]
    m = dict(macro_f1_present=float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)),
             accuracy=float((y_true == y_pred).mean()), frac_pred_NONE=float((y_pred == "NONE").mean()))
    m.update({f"recall_{c}": float((y_pred[y_true == c] == c).mean()) for c in present})
    return m


def stage_baseline(cfg, ds, out):
    d = rev_dir(cfg, "P8_baseline_negative_control")
    f = out / "bl_negative_control_by_seed.csv"
    if f.exists():
        df = pd.read_csv(f)
        s = df.groupby(["model", "representation", "source"], sort=False).mean(numeric_only=True).drop(columns="seed").reset_index()
        s.to_csv(d / "bl_negative_control_summary.csv", index=False)
        (d / "bl_negative_control_summary.md").write_text(
            "# Baseline (BL) por los modelos de 3 clases\n\nNo se calcula accuracy: ninguna etiqueta es correcta. Se muestran probabilidades medias, "
            "fracción de días asignada a cada clase y entropía normalizada, junto a los días `pre` de test (mismas fechas aprox.).\n\n"
            + md_table(s[s["source"].isin(["BL", "pre_test"])]) + "\n", encoding="utf-8")
        g = s[(s["model"] == "random_forest") & s["source"].isin(["BL", "pre_test"])]
        reps = list(dict.fromkeys(g["representation"]))
        if len(g):
            fig, ax = plt.subplots(figsize=(1.0 * len(reps) * 2 + 3, 4.2))
            w = 0.12
            for i, (src, c) in enumerate((s_, c_) for s_ in ("BL", "pre_test") for c_ in SCENARIOS):
                v = [g[(g["representation"] == r) & (g["source"] == src)][f"frac_pred_{c}"].mean() for r in reps]
                ax.bar(np.arange(len(reps)) + i * w, v, w, label=f"{src} → {c}", hatch="//" if src == "pre_test" else None)
            ax.set_xticks(np.arange(len(reps)) + 2.5 * w, reps, rotation=30)
            ax.set(ylabel="Share of days predicted as class", title="Random forest: BL vs pre-drift test days")
            ax.legend(fontsize=7, ncol=2)
            fig.tight_layout()
            save_fig(fig, d / "P8_bl_negative_control.png", "Negative control: what the 3-class models say about the intervention-free baseline series.",
                     "Solid bars = baseline (BL) days; hatched = pre-drift test days. Similar bars mean BL behaves like pre-drift data (not like any intervention).")
    # --- cuarta clase NONE (= días pre de series reales, muchas realizaciones independientes)
    bl = load_baseline(cfg)
    rows_ = []
    for seed in seeds_for(cfg):
        split = do_split(ds, cfg, seed)
        meta = ds.meta
        trm = meta["realization"].isin(split.train).to_numpy()
        idx = np.flatnonzero(trm & meta["phase"].isin(["pre", "developed"]).to_numpy())
        cap = cfg.get("ml", {}).get("max_train_rows")
        if cap and len(idx) > cap:
            idx = idx[np.random.default_rng(seed).choice(len(idx), cap, replace=False)]
        y4 = np.where(meta["phase"].to_numpy()[idx] == "pre", "NONE", meta["scenario"].to_numpy()[idx])
        te = meta["realization"].isin(split.test).to_numpy()
        mte = meta[te].reset_index(drop=True)
        yt = np.where(mte["phase"] == "pre", "NONE", mte["scenario"]).astype(object)
        X_fit = ds.X[trm]
        for rep in focus_reps(X_fit, cfg):
            Ztr, Zte = rep.transform(ds.X[idx]), rep.transform(ds.X[te])
            for model in cfg["models"]:
                clf = make_model(model, seed).fit(Ztr, y4)
                pred = clf.predict(Zte)
                for p in PHASES:
                    s = (mte["phase"] == p).to_numpy()
                    rows_.append(dict(seed=seed, model=model, representation=rlabel(rep), phase=p, **_four_metrics(yt[s], pred[s])))
                if bl is not None:
                    pb = clf.predict(rep.transform(bl[0]))
                    rows_.append(dict(seed=seed, model=model, representation=rlabel(rep), phase="BL", macro_f1_present=np.nan,
                                      accuracy=np.nan, frac_pred_NONE=float((pb == "NONE").mean())))
    r = pd.DataFrame(rows_)
    r.to_csv(d / "four_class_results.csv", index=False)
    s = r.groupby(["model", "representation", "phase"], sort=False).mean(numeric_only=True).drop(columns="seed").reset_index()
    (d / "four_class_summary.md").write_text(
        "# Modelo de 4 clases (EXP/REF/NS/NONE)\n\nNONE = días `pre` de las 900 realizaciones (muchas trayectorias independientes antes del evento, "
        "no una sola serie). Entrenamiento: días pre + developed de las realizaciones de train. `BL` = fracción de días del baseline predichos como NONE.\n\n"
        + md_table(s) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- P11: sensibilidad de la transición


def stage_windows(cfg, ds, out):
    d = rev_dir(cfg, "P11_transition_sensitivity")
    ml = __import__("importlib").import_module("experiments.03_ml")
    res = []
    for seed in available_seeds(out):
        te, preds = load_predictions(out, seed)
        mt = ds.meta.iloc[te].reset_index(drop=True)
        masks = window_masks(mt, rcfg(cfg, "followup_windows", [90, 180, 365]), rcfg(cfg, "boundary_buffers", [7, 14, 30]))
        y = mt["scenario"].to_numpy()
        for (model, rep), code in preds.items():
            pred = np.array(SCENARIOS, dtype=object)[np.minimum(code, 2)]
            for name, m in masks.items():
                if m.sum() >= 30:
                    res.append(dict(seed=seed, model=model, representation=rep, window=name, **classification_metrics(y[m], pred[m]), n_rows=int(m.sum())))
    W = pd.DataFrame(res)
    W.to_csv(d / "window_results_all_splits.csv", index=False)
    S = summarize(W, ["model", "representation", "window"]) if len(W) else W
    S.to_csv(d / "window_results_summary.csv", index=False)
    notes = ["# Sensibilidad de la definición de transición\n",
             "Modelos entrenados como siempre (días `developed` actuales). `followup_*` = ventanas de seguimiento tras el final de la transición "
             "(se reportan como seguimiento, **no** como transición). `matched_first_*d_from_onset` = misma ventana de calendario para los 3 escenarios. "
             "`*_excl_buffer_*` excluye una franja alrededor del onset/final para medir ambigüedad de etiqueta.\n"]
    if len(S):
        notes.append(md_table(S[S["model"] == "random_forest"], ["representation", "window", "macro_f1_mean", "macro_f1_std", "f1_EXP_mean", "f1_REF_mean", "f1_NS_mean"]))
    # refit con otras duraciones de la transición de EXP (cambia también el conjunto de entrenamiento)
    alt_rows = []
    for alt in rcfg(cfg, "exp_transition_days_alt", [90, 360]):
        m2 = ds.meta.copy()
        m2["transition_end"] = np.where(m2["scenario"] == "EXP", alt, m2["transition_end"])
        m2["phase"] = assign_phase(m2["days_since_start"], m2["transition_end"])
        ds2 = Dataset(X=ds.X, meta=m2, realizations=ds.realizations)
        for seed in seeds_for(cfg):
            split = do_split(ds, cfg, seed)
            reps = focus_reps(rows(ds2, split.train)[0], cfg)
            alt_rows.append(relabel(ml.run(ds2, split, reps, {**cfg, "seed": seed}, out, save=False).assign(seed=seed, exp_transition_days=alt)))
    if alt_rows:
        A = pd.concat(alt_rows, ignore_index=True)
        base = pd.read_csv(out / "results_seeds_raw.csv") if (out / "results_seeds_raw.csv").exists() else None
        if base is not None:
            A = pd.concat([A, relabel(base[base["seed"].isin(seeds_for(cfg)) & base["components"].isin([24] + list(cfg["rom"].get("focus_components", [])))]
                           .assign(exp_transition_days=cfg.get("phases", {}).get("transition_days", 180)))], ignore_index=True)
        A.to_csv(d / "exp_transition_length_results.csv", index=False)
        SA = summarize(A, ["model", "exp_transition_days", "phase", "representation", "components"])
        SA.to_csv(d / "exp_transition_length_summary.csv", index=False)
        notes.append("\n\n## EXP con distinta duración de transición (reentrenando)\n\nLas fases se redefinen con cada duración, así que las cifras de cada fila "
                     "son sobre *su* definición de transition/developed.\n\n" + md_table(SA[SA["model"] == "random_forest"],
                     ["exp_transition_days", "phase", "representation", "macro_f1_mean", "macro_f1_std", "f1_EXP_mean"]))
    (d / "transition_sensitivity.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    if len(S):
        g = S[(S["model"] == "random_forest") & S["representation"].isin(["Raw", "PCA-5", "PCA-10", "PCA-2"])]
        wins = [w for w in g["window"].unique() if w.startswith(("current", "followup", "matched"))]
        fig, a = plt.subplots(figsize=(11, 4.6))
        for rp, h in g.groupby("representation", sort=False):
            h = h.set_index("window").reindex(wins)
            a.errorbar(range(len(wins)), h["macro_f1_mean"], yerr=h["macro_f1_std"], marker="o", capsize=2, label=rp)
        a.set_xticks(range(len(wins)), wins, rotation=40, ha="right", fontsize=8)
        a.set(ylabel="Macro-F1 (mean ± SD over splits)")
        a.legend(fontsize=8)
        fig.tight_layout()
        save_fig(fig, d / "P11_windows.png", "Random-forest Macro-F1 under alternative evaluation windows.", "Each x tick is a window definition (current phases, follow-up windows after the endpoint, matched calendar windows).")


# --------------------------------------------------------------------------- P12: regímenes de ajuste de la PCA


def stage_pca_regimes(cfg, ds, out):
    d = rev_dir(cfg, "P12_pca_fit_regimes")
    ml = __import__("importlib").import_module("experiments.03_ml")
    regimes = {"all_train_days": None, "pre_only": ["pre"], "developed_only": ["developed"]}
    allr = []
    for seed in seeds_for(cfg):
        split = do_split(ds, cfg, seed)
        for name, phs in regimes.items():
            X_fit, _ = rows(ds, split.train, phs)
            reps = focus_reps(X_fit, cfg)
            allr.append(relabel(ml.run(ds, split, reps, {**cfg, "seed": seed}, out, save=False).assign(seed=seed, regime=name)))
    R = pd.concat(allr, ignore_index=True)
    R.to_csv(d / "pca_regime_results.csv", index=False)
    S = summarize(R, ["model", "regime", "phase", "representation", "components"])
    ref = S[S["regime"] == "all_train_days"].set_index(["model", "phase", "representation", "components"])
    S = S.join(ref[["macro_f1_mean"]], on=["model", "phase", "representation", "components"], rsuffix="_all_days")
    S["delta_macro_f1_vs_all_days"] = S["macro_f1_mean"] - S["macro_f1_mean_all_days"]
    S.to_csv(d / "pca_regime_summary.csv", index=False)
    (d / "pca_regime_summary.md").write_text(
        "# PCA ajustada con distintos conjuntos de entrenamiento (siempre sin test)\n\n`pre_only` = la PCA nunca ve perfiles afectados por la intervención "
        "(aplicabilidad prospectiva); `all_train_days` es el régimen actual. No es fuga convencional, pero cambia lo que significa el resultado.\n\n"
        + md_table(S[S["model"] == "random_forest"], ["regime", "phase", "representation", "macro_f1_mean", "macro_f1_std", "delta_macro_f1_vs_all_days"]) + "\n", encoding="utf-8")
    g = S[(S["model"] == "random_forest") & (S["phase"] == "transition")]
    if len(g):
        fig, a = plt.subplots(figsize=(7.5, 4.6))
        reps = list(dict.fromkeys(g["representation"]))
        for i, (rg, h) in enumerate(g.groupby("regime", sort=False)):
            h = h.set_index("representation").reindex(reps)
            a.bar(np.arange(len(reps)) + i * 0.27, h["macro_f1_mean"], 0.27, yerr=h["macro_f1_std"], capsize=2, label=rg)
        a.set_xticks(np.arange(len(reps)) + 0.27, reps)
        a.set(ylabel="Transition Macro-F1", ylim=(0.5, 1.0))
        a.legend(fontsize=8)
        fig.tight_layout()
        save_fig(fig, d / "P12_pca_regimes.png", "Effect of the data used to fit the scaler/PCA on transition performance (random forest).", "Bars = mean ± SD over splits for each fit regime and representation.")


# --------------------------------------------------------------------------- P9: ruido / sesgo / huecos / desfases


def _case_list(Xs, shifted_sub, rng):
    cases = [("clean", 0.0, Xs)]
    cases += [("gaussian_noise", p, pt.gaussian_noise(Xs, p, rng)) for p in (0.01, 0.05, 0.10)]
    cases += [("calibration_bias", b, pt.calibration_bias(Xs, b)) for b in (-0.10, -0.05, -0.02, 0.02, 0.05, 0.10)]
    cases += [("random_missing", p, pt.random_missing(Xs, p, rng)) for p in (0.01, 0.05, 0.10)]
    cases += [("contiguous_outage", h, pt.contiguous_outage(Xs, h, rng)) for h in (1, 3, 6)]
    cases += [("timestamp_shift", s, shifted_sub[s]) for s in (-1, 1)]
    cases += [("spikes_dropouts", r, pt.spikes_dropouts(Xs, r, rng)) for r in (0.01, 0.03)]
    return cases


def stage_perturbation(cfg, ds, out):
    d = rev_dir(cfg, "P9_noise_robustness")
    meta = add_calendar(ds.meta)
    max_rows = int(rcfg(cfg, "perturb_max_test_rows", 80000))
    res = []
    for seed in seeds_for(cfg, "n_seeds_perturb", 3):
        rng = np.random.default_rng(seed)
        split = do_split(ds, cfg, seed)
        trm = meta["realization"].isin(split.train).to_numpy()
        ti = train_idx(ds, split, cfg, seed)
        te_idx = np.flatnonzero(meta["realization"].isin(split.test).to_numpy())
        mte, Xte = meta.iloc[te_idx].reset_index(drop=True), ds.X[te_idx]
        shifted = {s: pt.timestamp_shift(Xte, mte["realization"].to_numpy(), s) for s in (-1, 1)}
        elig = np.flatnonzero(mte["phase"].isin(["developed", "transition"]).to_numpy())
        sub = np.sort(rng.choice(elig, size=min(max_rows, len(elig)), replace=False))
        Xs, ms = Xte[sub], mte.iloc[sub].reset_index(drop=True)
        table = pt.fit_month_hour_median(ds.X[trm], meta["month"].to_numpy()[trm])
        months = ms["month"].to_numpy()
        cases = _case_list(Xs, {s: v[sub] for s, v in shifted.items()}, rng)
        y = ms["scenario"].to_numpy()
        reps = focus_reps(ds.X[trm], cfg)
        # modelo con indicadores de falta: entrenado con una copia del train con huecos aleatorios
        cap = min(len(ti), 100_000)
        ta = np.sort(rng.choice(ti, size=cap, replace=False)) if len(ti) > cap else ti
        Xaug = pt.augment_missing(ds.X[ta], rng)
        filled_tr = pt.impute_median(Xaug, meta["month"].to_numpy()[ta], table)
        flags_tr = np.isnan(Xaug).astype(float)
        # imputaciones calculadas una sola vez por caso (no dependen de la representación)
        prepared = []
        for fam, par, Xp in cases:
            has_nan = bool(np.isnan(Xp).any())
            variants = {"linear_interp": pt.impute_linear(Xp), "month_hour_median": pt.impute_median(Xp, months, table)} if has_nan else {"none": Xp}
            flags_in = (pt.impute_median(Xp, months, table), np.isnan(Xp).astype(float)) if has_nan else None
            prepared.append((fam, par, variants, flags_in))
        for rep in reps:
            clfs = {m: make_model(m, seed).fit(rep.transform(ds.X[ti]), meta["scenario"].to_numpy()[ti]) for m in cfg["models"]}
            flag_clf = make_model("random_forest", seed).fit(np.c_[rep.transform(filled_tr), flags_tr], meta["scenario"].to_numpy()[ta])
            for fam, par, variants, flags_in in prepared:
                for hand, Xf in variants.items():
                    Z = rep.transform(Xf)
                    for model, clf in clfs.items():
                        res += phase_metrics(ms, clf.predict(Z), y, dict(seed=seed, model=model, representation=rlabel(rep), family=fam, param=par, handling=hand))
                if flags_in is not None:
                    Zf = np.c_[rep.transform(flags_in[0]), flags_in[1]]
                    res += phase_metrics(ms, flag_clf.predict(Zf), y, dict(seed=seed, model="random_forest", representation=rlabel(rep), family=fam, param=par, handling="median+missing_flags"))
        print(f"  ruido: semilla {seed} hecha", flush=True)
    R = pd.DataFrame(res)
    R = R[R["phase"].isin(["developed", "transition"])]
    R.to_csv(d / "perturbation_results_all_seeds.csv", index=False)
    S = summarize(R, ["model", "representation", "family", "param", "handling", "phase"])
    clean = S[S["family"] == "clean"].set_index(["model", "representation", "phase"])["macro_f1_mean"]
    S["delta_macro_f1_vs_clean"] = S["macro_f1_mean"] - S.set_index(["model", "representation", "phase"]).index.map(clean).to_numpy()
    S.to_csv(d / "perturbation_summary.csv", index=False)
    (d / "perturbation_summary.md").write_text(
        "# Robustez a ruido de monitorización (modelos entrenados con datos limpios, test perturbado)\n\nPerturbaciones aplicadas solo al test; las imputaciones usan "
        "únicamente estadísticos del entrenamiento. `median+missing_flags` = RF entrenado con huecos sintéticos + indicadores de falta.\n\n"
        + md_table(S[(S["model"] == "random_forest") & (S["phase"] == "transition")], ["representation", "family", "param", "handling", "macro_f1_mean", "delta_macro_f1_vs_clean"]) + "\n", encoding="utf-8")
    fams = ["gaussian_noise", "calibration_bias", "random_missing", "contiguous_outage", "timestamp_shift", "spikes_dropouts"]
    g = S[(S["model"] == "random_forest") & (S["phase"] == "transition")]
    if len(g):
        fig, ax = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
        for a, fam in zip(ax.ravel(), fams):
            gg = g[g["family"] == fam]
            for (rp, hand), h in gg.groupby(["representation", "handling"], sort=False):
                h = h.sort_values("param")
                a.plot(h["param"], h["macro_f1_mean"], marker="o", label=f"{rp} / {hand}", ls="-" if hand in ("none", "linear_interp") else "--")
            a.set_title(fam, fontsize=10)
            a.set_xlabel("Perturbation level")
        ax[0, 0].set_ylabel("Transition Macro-F1")
        ax[1, 0].set_ylabel("Transition Macro-F1")
        ax[0, 0].legend(fontsize=6)
        ax[1, 0].legend(fontsize=6)
        fig.tight_layout()
        save_fig(fig, d / "P9_noise_robustness.png", "Random-forest Macro-F1 on transition days as monitoring noise grows (clean-trained models, perturbed test).",
                 "One panel per perturbation family; x = perturbation level (fraction of daily mean, relative bias, missing fraction, outage hours, hour shift, spike rate). Lines per representation and imputation strategy.")


# --------------------------------------------------------------------------- P9: fallos físicos / anomalías de medida (aparte de las intervenciones)


def stage_faults(cfg, ds, out):
    d = rev_dir(cfg, "P9_fault_anomaly")
    meta = ds.meta
    idx_by_r = meta.groupby("realization").indices
    ph = meta["phase"].to_numpy()
    cap = int(cfg.get("ml", {}).get("max_train_rows") or 300_000)
    all_conf, all_kind, all_e2e = [], [], []
    for seed in seeds_for(cfg, "n_seeds_perturb", 3):
        rng = np.random.default_rng(seed)
        split = do_split(ds, cfg, seed)

        def assemble(ids):
            Xs, lab, kind, phase = [], [], [], []
            for r in ids:
                idx = idx_by_r[r]
                pre, post = idx[ph[idx] == "pre"], idx[ph[idx] != "pre"]
                k = str(rng.choice(pt.FAULTS)) if rng.random() < 0.5 else "none"
                Xp = ds.X[pre]
                if k != "none":
                    Xp = pt.inject_fault(k, Xp, np.arange(len(pre)) / max(len(pre) - 1, 1), rng)
                Xs += [Xp, ds.X[post]]
                lab += [np.full(len(pre), pt.GROUP.get(k, "none")), np.full(len(post), "structural")]
                kind += [np.full(len(pre), k), np.full(len(post), "structural")]
                phase += [np.full(len(pre), "pre"), ph[post]]
            return np.vstack(Xs), np.concatenate(lab), np.concatenate(kind), np.concatenate(phase)

        Xtr, ytr, _, ptr = assemble(sorted(split.train))
        keep = np.concatenate([rng.choice(np.flatnonzero(ytr == c), size=min((ytr == c).sum(), cap // 4), replace=False) for c in np.unique(ytr)])
        Xtr, ytr = Xtr[keep], ytr[keep]
        # los 'structural' de entrenamiento = developed (como en el resto del estudio)
        dev = ptr[keep] == "developed"
        notdev = (ytr == "structural") & ~dev
        Xtr, ytr = Xtr[~notdev], ytr[~notdev]
        Xte, yte, kte, pte = assemble(sorted(split.test))
        ti = train_idx(ds, split, cfg, seed)
        X_fit = ds.X[meta["realization"].isin(split.train).to_numpy()]
        for rep in focus_reps(X_fit, cfg):
            router = make_model("random_forest", seed).fit(rep.transform(Xtr), ytr)
            scen = make_model("random_forest", seed).fit(rep.transform(ds.X[ti]), meta["scenario"].to_numpy()[ti])
            pred = router.predict(rep.transform(Xte))
            lab = rlabel(rep)
            c = pd.crosstab(pd.Series(yte, name="true_super"), pd.Series(pred, name="pred_super")).stack().rename("count").reset_index()
            all_conf.append(c.assign(seed=seed, representation=lab))
            k = pd.crosstab(pd.Series(kte, name="true_kind"), pd.Series(pred, name="pred_super"), normalize="index").stack().rename("share").reset_index()
            all_kind.append(k.assign(seed=seed, representation=lab))
            # de extremo a extremo: lo que el router declara 'structural' pasa al clasificador de escenario
            truth = np.concatenate([meta["scenario"].to_numpy()[idx_by_r[r]][ph[idx_by_r[r]] != "pre"] for r in sorted(split.test)])
            s_idx = np.flatnonzero(yte == "structural")
            routed = pred[s_idx] == "structural"
            sc_pred = scen.predict(rep.transform(Xte[s_idx[routed]]))
            all_e2e.append(dict(seed=seed, representation=lab, structural_rows=len(s_idx), routed_to_structural=float(routed.mean()),
                                scenario_macro_f1_given_routed=float(f1_score(truth[routed], sc_pred, labels=list(SCENARIOS), average="macro", zero_division=0)),
                                faulty_pre_rows=int(np.isin(yte, ["measurement_anomaly", "physical_fault"]).sum()),
                                faulty_routed_to_structural=float((pred[np.isin(yte, ["measurement_anomaly", "physical_fault"])] == "structural").mean())))
        print(f"  fallos: semilla {seed} hecha", flush=True)
    C, K, E = pd.concat(all_conf), pd.concat(all_kind), pd.DataFrame(all_e2e)
    C.to_csv(d / "fault_router_confusion_counts.csv", index=False)
    K.to_csv(d / "fault_routing_by_kind_all_seeds.csv", index=False)
    E.to_csv(d / "fault_end_to_end_all_seeds.csv", index=False)
    Ks = K.groupby(["representation", "true_kind", "pred_super"], sort=False)["share"].mean().reset_index()
    Ks.to_csv(d / "fault_routing_by_kind_summary.csv", index=False)
    Es = E.groupby("representation", sort=False).mean(numeric_only=True).drop(columns="seed").reset_index()
    (d / "fault_summary.md").write_text(
        "# Flujo jerárquico: sin evento / anomalía de medida / fallo físico / intervención estructural → EXP/REF/NS\n\n"
        "Los fallos son **simulaciones transparentes** (no calibradas con datos reales) inyectadas en los días `pre` de la mitad de las realizaciones; "\
        "la medida (stuck_sensor, calibration_drift) y lo físico (valve_failure, hx_degradation, leakage) se mantienen separados. Un fallo clasificado como "
        "`structural` es el error peligroso (falsa intervención).\n\n## Enrutado por tipo (fracción media)\n\n" + md_table(Ks.pivot_table(index=["representation", "true_kind"], columns="pred_super", values="share").reset_index().fillna(0.0))
        + "\n\n## Extremo a extremo\n\n" + md_table(Es) + "\n", encoding="utf-8")
    g = Ks[Ks["representation"] == ("PCA-5" if "PCA-5" in set(Ks["representation"]) else Ks["representation"].iloc[0])]
    if len(g):
        kinds = list(dict.fromkeys(g["true_kind"]))
        fig, a = plt.subplots(figsize=(9, 4.5))
        bottom = np.zeros(len(kinds))
        for cls, col in zip(("none", "measurement_anomaly", "physical_fault", "structural"), ("tab:green", "tab:blue", "tab:orange", "tab:red")):
            v = np.array([g[(g["true_kind"] == k) & (g["pred_super"] == cls)]["share"].sum() for k in kinds])
            a.bar(range(len(kinds)), v, bottom=bottom, label=f"routed to {cls}", color=col)
            bottom += v
        a.set_xticks(range(len(kinds)), kinds, rotation=30, ha="right")
        a.set(ylabel="Share of days", title=f"Where each true condition is routed ({g['representation'].iloc[0]}, random forest)")
        a.legend(fontsize=8)
        fig.tight_layout()
        save_fig(fig, d / "P9_fault_routing.png", "Hierarchical workflow: where injected faults and structural interventions are sent.", "Each bar = one true condition (clean = none, five fault mechanisms, structural). Red share on a fault bar = false structural intervention.")


# --------------------------------------------------------------------------- P10: sensibilidad del RF y coste computacional


def _grid(cfg):
    g = rcfg(cfg, "rf_grid", {"n_estimators": [200, 500], "max_depth": [None, 20], "min_samples_leaf": [1, 5], "max_features": ["sqrt", 0.5, 1.0]})
    keys = list(g)
    return [dict(zip(keys, v)) for v in itertools.product(*[g[k] for k in keys])], g


def _pname(p):
    return "n{n_estimators}_d{max_depth}_l{min_samples_leaf}_f{max_features}".format(**p)


def stage_rf_sensitivity(cfg, ds, out):
    d = rev_dir(cfg, "P10_rf_sensitivity")
    ml = __import__("importlib").import_module("experiments.03_ml")
    seed = cfg["seed"]
    split = do_split(ds, cfg, seed)
    K = int(rcfg(cfg, "rf_cv_folds", 3))
    cap = int(rcfg(cfg, "rf_cv_rows", 60000))
    grid, g = _grid(cfg)
    rng = np.random.default_rng(seed)
    fold_of = {}
    for _, gg in ds.realizations[ds.realizations["realization"].isin(split.train)].groupby(["scenario", "severity"]):
        ids = np.array(sorted(gg["realization"]))
        rng.shuffle(ids)
        for i, r in enumerate(ids):
            fold_of[r] = i % K
    meta = ds.meta
    isr = lambda ids: meta["realization"].isin(ids).to_numpy()
    cv = []
    for f in range(K):  # validación agrupada SOLO con realizaciones de train; la PCA se ajusta sin la fold de validación
        val_ids = [r for r, k in fold_of.items() if k == f]
        fit_ids = [r for r, k in fold_of.items() if k != f]
        itr = np.flatnonzero(isr(fit_ids) & (meta["phase"] == "developed").to_numpy())
        if len(itr) > cap:
            itr = rng.choice(itr, cap, replace=False)
        iv = np.flatnonzero(isr(val_ids) & meta["phase"].isin(["developed", "transition"]).to_numpy())
        for rep in focus_reps(ds.X[isr(fit_ids)], cfg):
            Ztr, Zv = rep.transform(ds.X[itr]), rep.transform(ds.X[iv])
            for p in grid:
                clf = make_model("random_forest", seed, p).fit(Ztr, meta["scenario"].to_numpy()[itr])
                pv = clf.predict(Zv)
                for phn in ("developed", "transition"):
                    s = (meta["phase"].to_numpy()[iv] == phn)
                    cv.append(dict(fold=f, representation=rlabel(rep), config=_pname(p), phase=phn, **p,
                                   macro_f1=classification_metrics(meta["scenario"].to_numpy()[iv][s], pv[s])["macro_f1"]))
        print(f"  RF-CV: fold {f + 1}/{K} hecha", flush=True)
    CV = pd.DataFrame(cv)
    CV.to_csv(d / "rf_cv_results.csv", index=False)
    cs = CV.groupby(["representation", "config", "phase"] + list(g), sort=False, dropna=False)["macro_f1"].mean().reset_index()
    cs.to_csv(d / "rf_cv_summary.csv", index=False)
    sel = {}
    for rp, h in cs[cs["phase"] == "transition"].groupby("representation", sort=False):
        best = h.sort_values(["macro_f1", "n_estimators"], ascending=[False, True]).iloc[0]
        sel[rp] = {k: (None if pd.isna(best[k]) else best[k]) for k in g}
        for k in ("n_estimators", "min_samples_leaf"):
            sel[rp][k] = int(sel[rp][k])
        if sel[rp]["max_depth"] is not None:
            sel[rp]["max_depth"] = int(sel[rp]["max_depth"])
        if isinstance(sel[rp]["max_features"], str) and sel[rp]["max_features"] not in ("sqrt", "log2"):
            sel[rp]["max_features"] = float(sel[rp]["max_features"])
    pd.DataFrame([dict(representation=r, **p) for r, p in sel.items()]).to_csv(d / "rf_cv_selected_frozen.csv", index=False)
    # --- evaluación en test una vez congelada la selección (la tabla completa es solo sensibilidad, no selección)
    X_fit, _ = rows(ds, split.train)
    reps = focus_reps(X_fit, cfg)
    X_tr, y_tr = ml.train_arrays(ds, split, cfg, seed)
    te = isr(split.test)
    Xte, mte = ds.X[te], meta[te].reset_index(drop=True)
    default = {**RF_DEFAULT}
    res = []
    for rep in reps:
        lab = rlabel(rep)
        Ztr, Zte = rep.transform(X_tr), rep.transform(Xte)
        todo = {"default": default, "selected_by_cv": {**default, **sel.get(lab, {})}}
        if rcfg(cfg, "rf_test_mode", "focused") == "all":
            todo.update({f"grid:{_pname(p)}": p for p in grid})
        else:
            for k, vals in g.items():
                for v in vals:
                    if v != default.get(k):
                        todo[f"one_factor:{k}={v}"] = {**default, k: v}
        for role, p in todo.items():
            t0 = time.time()
            clf = make_model("random_forest", seed, p).fit(Ztr, y_tr)
            fit_s = time.time() - t0
            res += phase_metrics(mte, clf.predict(Zte), extra=dict(representation=lab, role=role, config=_pname({**default, **p}), fit_seconds=fit_s))
        print(f"  RF-test: {lab} hecha", flush=True)
    T = pd.DataFrame(res)
    T.to_csv(d / "rf_test_results_seed_main.csv", index=False)
    lines = ["# Sensibilidad del random forest\n",
             "Selección por validación agrupada **solo con realizaciones de train** (PCA ajustada sin la fold de validación; métrica = Macro-F1 en transición). "
             "La tabla de test se calcula una vez congelada la selección y es solo sensibilidad (nada se elige con ella). "
             "Se añadió `max_features=1.0` a la rejilla de la guía: con PCA-2, `sqrt` y `0.5` dan ambos 1 variable por división, así que no distinguen el efecto.\n"]
    tt = T[T["phase"] == "transition"]
    for rp in ("PCA-2",):
        a = tt[(tt["representation"] == rp) & (tt["role"] == "default")]["macro_f1"]
        b = tt[(tt["representation"] == rp) & (tt["role"] == "one_factor:max_features=1.0")]["macro_f1"]
        r0 = tt[(tt["representation"] == "Raw") & (tt["role"] == "default")]["macro_f1"]
        if len(a) and len(b) and len(r0):
            lines.append(f"**{rp} (transición, RF):** defecto (`sqrt`) {a.iloc[0]:.3f}; `max_features=1.0` {b.iloc[0]:.3f}; Raw defecto {r0.iloc[0]:.3f}. "
                         "Si 1.0 cierra buena parte de la brecha con Raw, la pérdida de PCA-2 es en parte un artefacto de `max_features`.\n")
    lines.append(md_table(T[T["phase"].isin(["developed", "transition"])], ["representation", "role", "phase", "macro_f1", "balanced_accuracy", "fit_seconds"]))
    (d / "rf_sensitivity_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    h = T[(T["phase"] == "transition") & T["role"].isin(["default", "selected_by_cv"] + [r for r in T["role"].unique() if r.startswith("one_factor:max_features")])]
    if len(h):
        reps_l = list(dict.fromkeys(h["representation"]))
        roles = list(dict.fromkeys(h["role"]))
        fig, a = plt.subplots(figsize=(10, 4.6))
        for i, ro in enumerate(roles):
            v = [h[(h["representation"] == r) & (h["role"] == ro)]["macro_f1"].mean() for r in reps_l]
            a.bar(np.arange(len(reps_l)) + i * 0.8 / len(roles), v, 0.8 / len(roles), label=ro)
        a.set_xticks(np.arange(len(reps_l)) + 0.4, reps_l)
        a.set(ylabel="Transition Macro-F1 (main split)", ylim=(0.4, 1.0))
        a.legend(fontsize=7)
        fig.tight_layout()
        save_fig(fig, d / "P10_rf_sensitivity.png", "Random-forest hyper-parameter sensitivity on transition days, by representation.", "Bars: default, CV-selected, and max_features variants. Differences between bars within a representation = sensitivity to the forest setting.")


class _PeakRSS:
    def __init__(self):
        try:
            import psutil

            self.p = psutil.Process()
        except Exception:
            self.p = None
        self.peak = self.start = 0

    def __enter__(self):
        if self.p is None:
            return self
        self.start = self.peak = self.p.memory_info().rss
        self._stop = False

        def loop():
            while not self._stop:
                self.peak = max(self.peak, self.p.memory_info().rss)
                time.sleep(0.05)

        self.t = threading.Thread(target=loop, daemon=True)
        self.t.start()
        return self

    def __exit__(self, *a):
        if self.p is not None:
            self._stop = True
            self.t.join()
            self.peak = max(self.peak, self.p.memory_info().rss)

    @property
    def delta_mb(self):
        return float("nan") if self.p is None else (self.peak - self.start) / 2**20


def stage_benchmark(cfg, ds, out):
    import json

    import joblib

    d = rev_dir(cfg, "P10_resource_benchmark")
    ml = __import__("importlib").import_module("experiments.03_ml")
    seed = cfg["seed"]
    split = do_split(ds, cfg, seed)
    X_fit, _ = rows(ds, split.train)
    X_tr, y_tr = ml.train_arrays(ds, split, cfg, seed)
    n = int(rcfg(cfg, "bench_pred_profiles", 10000))
    probe = ds.X[ds.meta["realization"].isin(split.test).to_numpy()]
    probe = probe[np.arange(n) % len(probe)]
    R = int(rcfg(cfg, "bench_repeats", 5))
    FR = int(rcfg(cfg, "bench_fit_repeats", 2))
    reps = [fit_representation(X_fit, None)] + [fit_representation(X_fit, k) for k in cfg["rom"]["components"]]
    warm = make_model("random_forest", seed, {"n_estimators": 5}).fit(reps[0].transform(X_tr[:2000]), y_tr[:2000])  # calentamiento
    warm.predict(reps[0].transform(probe[:100]))
    res = []
    for rep in reps:
        Ztr = rep.transform(X_tr)
        rep.transform(probe)
        tt = []
        for _ in range(R):
            t0 = time.perf_counter(); Z = rep.transform(probe); tt.append(time.perf_counter() - t0)
        fits, peaks = [], []
        for _ in range(FR):
            with _PeakRSS() as m:
                t0 = time.perf_counter(); clf = make_model("random_forest", seed).fit(Ztr, y_tr); fits.append(time.perf_counter() - t0)
            peaks.append(m.delta_mb)
        clf.predict(Z[:100])
        pp = []
        for _ in range(R):
            t0 = time.perf_counter(); clf.predict(Z); pp.append(time.perf_counter() - t0)
        buf = io.BytesIO()
        joblib.dump({"scaler": rep.scaler, "pca": rep.pca, "rf": clf}, buf)
        depths = [e.tree_.max_depth for e in clf.estimators_]
        res.append(dict(representation=rlabel(rep), components=rep.n_components, pca_transform_s_per_profiles=float(np.median(tt)),
                        rf_fit_s=float(np.median(fits)), predict_s_per_profiles=float(np.median(pp)),
                        pipeline_predict_s_per_profiles=float(np.median(tt) + np.median(pp)), serialized_mb=buf.tell() / 2**20,
                        peak_rss_delta_mb=float(np.nanmax(peaks)) if not np.all(np.isnan(peaks)) else float("nan"),
                        total_nodes=int(sum(e.tree_.node_count for e in clf.estimators_)), mean_tree_depth=float(np.mean(depths)),
                        max_tree_depth=int(np.max(depths)), n_profiles_timed=n, n_train_rows=len(y_tr)))
        print(f"  benchmark: {rlabel(rep)} hecho", flush=True)
    B = pd.DataFrame(res)
    B.to_csv(d / "resource_benchmark.csv", index=False)
    (d / "hardware.json").write_text(json.dumps({"hardware": hardware(), "n_jobs": "all cores (RF)", "repeats": R, "fit_repeats": FR,
                                                   "note": "una pasada de calentamiento previa; mediana de las repeticiones"}, indent=2), encoding="utf-8")
    (d / "resource_benchmark.md").write_text(
        "# Coste computacional (RF por defecto; Raw y PCA-k)\n\nTiempos = mediana tras calentamiento; latencia por 10 000 perfiles; tamaño = PCA+RF serializados con joblib; "
        "memoria = pico de RSS durante el ajuste menos el RSS inicial. **No se afirma reducción de sensores**: PCA sigue necesitando las 24 horas de entrada.\n\n" + md_table(B) + "\n", encoding="utf-8")
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    pos = {int(c): i for i, c in enumerate(sorted(B[B["representation"] != "Raw"]["components"]))}
    for a, col, lab in zip(ax, ("rf_fit_s", "pipeline_predict_s_per_profiles", "serialized_mb"), ("RF fit time (s)", f"Transform+predict time ({n} profiles, s)", "Serialized PCA+RF (MB)")):
        p = B[B["representation"] != "Raw"].sort_values("components")
        a.plot(p["components"], p[col], "o-", label="PCA-k")
        a.axhline(float(B[B["representation"] == "Raw"][col].iloc[0]), color="k", ls="--", label="Raw 24 features")
        a.set(xlabel="Number of PCA components k", ylabel=lab)
        a.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, d / "P10_resource_benchmark.png", "Computational cost of Raw vs PCA-k pipelines (random forest).", "Dashed line = raw 24-feature profile; points = PCA dimension k.")


# --------------------------------------------------------------------------- P14: diagrama del flujo de trabajo


def stage_workflow(cfg, ds, out):
    d = rev_dir(cfg, "figures")
    steps = [("DHDrift series\n(900 hourly,\n3 scenarios x 3 severities)", 0.02, 0.62), ("Daily 24-h profiles\n+ phase labels\n(pre / transition / developed)", 0.21, 0.62),
             ("Split by whole\nrealizations\n(80/20 stratified)", 0.40, 0.62), ("Scaler + PCA\nfitted on TRAIN only\n(k = 2..12) or Raw", 0.59, 0.62),
             ("LR / RF trained on\ndeveloped TRAIN days", 0.78, 0.62), ("Evaluation on TEST:\ndeveloped, transition,\npre (control)", 0.78, 0.22),
             ("Extra analyses:\nweather, shape-only,\nloadings, Hellinger, BL,\nnoise/faults, RF sensitivity", 0.40, 0.22), ("Outputs: CSV + MD +\nfigures with .txt\ndescriptions", 0.02, 0.22)]
    fig, a = plt.subplots(figsize=(13, 5))
    a.set_xlim(0, 1), a.set_ylim(0, 1), a.axis("off")
    for t, x, y in steps:
        a.text(x + 0.085, y + 0.12, t, ha="center", va="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.5", fc="#e8f0fe", ec="#4a6fa5"))
    arrows = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]
    hw, hh = 0.078, 0.11
    for i, j in arrows:
        (_, x0, y0), (_, x1, y1) = steps[i], steps[j]
        cx0, cy0, cx1, cy1 = x0 + 0.085, y0 + 0.12, x1 + 0.085, y1 + 0.12
        if y0 == y1:
            sgn = 1 if x1 > x0 else -1
            a.annotate("", xy=(cx1 - sgn * hw, cy1), xytext=(cx0 + sgn * hw, cy0), arrowprops=dict(arrowstyle="->", color="#4a6fa5"))
        else:
            a.annotate("", xy=(cx1, cy1 + hh), xytext=(cx0, cy0 - hh), arrowprops=dict(arrowstyle="->", color="#4a6fa5"))
    fig.tight_layout()
    how = "Boxes in order: " + " -> ".join(t.replace("\n", " ") for t, _, _ in steps) + "."
    for ext in ("svg", "pdf", "png"):
        save_fig(fig, d / f"00_workflow.{ext}", "Workflow of the ROM + ML study and of the revision analyses (vector SVG/PDF for the manuscript).", how, close=False)
    plt.close(fig)
