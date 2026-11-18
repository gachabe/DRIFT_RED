"""Análisis a partir de predicciones: P/R/F1 estratificados, deltas pareados, bootstrap, NRMSE, ventanas,
indicadores físicos, distancia de Hellinger."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import SCENARIOS

CLASSES4 = ("EXP", "REF", "NS", "NONE")
CODE = {c: i for i, c in enumerate(CLASSES4)}


def encode(labels) -> np.ndarray:
    return np.array([CODE[x] for x in labels], dtype=np.int8)


def decode(codes) -> np.ndarray:
    return np.array(CLASSES4, dtype=object)[np.asarray(codes, dtype=int)]


# --------------------------------------------------------------------------- progreso


def progress_bins(meta: pd.DataFrame, bins: int = 5) -> np.ndarray:
    """-1 fuera de la transición; 0..bins-1 dentro (fracción del avance onset -> fin de transición)."""
    prog = (meta["days_since_start"] / meta["transition_end"]).to_numpy()
    b = np.clip((prog * bins).astype(int), 0, bins - 1)
    return np.where((meta["phase"] == "transition").to_numpy(), b, -1)


def bin_label(b: int, bins: int) -> str:
    return "NA" if b < 0 else f"{100 * b // bins}-{100 * (b + 1) // bins}%"


# --------------------------------------------------------------------------- P/R/F1 estratificados


def prf_table(meta: pd.DataFrame, y_pred, strata, classes=SCENARIOS, true_col: str = "scenario") -> pd.DataFrame:
    """Precision, recall y F1 por clase dentro de cada estrato (el estrato contiene todas las clases).

    precision = TP / (predichos como c en el estrato, sea cual sea su clase real); recall = TP / (reales c).
    Columnas: *strata, scenario (= clase), precision, recall, f1, n_profiles (reales), n_pred, n_realizations.
    """
    strata = list(strata) or ["_all"]
    d = pd.DataFrame({c: meta[c].to_numpy() for c in strata if c != "_all"})
    if "_all" in strata:
        d["_all"] = "all"
    d["realization"] = meta["realization"].to_numpy()
    d["_t"] = meta[true_col].to_numpy()
    d["_p"] = np.asarray(y_pred)
    cnt = d.groupby(strata + ["_t", "_p"], observed=True, sort=False).size().rename("n").reset_index()
    parts = []
    for c in classes:
        tn = cnt[cnt["_t"] == c].groupby(strata, observed=True)["n"].sum().rename("n_profiles")
        pn = cnt[cnt["_p"] == c].groupby(strata, observed=True)["n"].sum().rename("n_pred")
        tp = cnt[(cnt["_t"] == c) & (cnt["_p"] == c)].groupby(strata, observed=True)["n"].sum().rename("tp")
        nr = d[d["_t"] == c].groupby(strata, observed=True)["realization"].nunique().rename("n_realizations")
        t = pd.concat([tn, pn, tp, nr], axis=1).fillna(0).reset_index()
        t["scenario"] = c
        parts.append(t)
    t = pd.concat(parts, ignore_index=True)
    t = t[t["n_profiles"] > 0].copy()
    t["recall"] = t["tp"] / t["n_profiles"]
    t["precision"] = np.where(t["n_pred"] > 0, t["tp"] / t["n_pred"].where(t["n_pred"] > 0, 1), np.nan)
    t["f1"] = 2 * t["tp"] / (t["n_profiles"] + t["n_pred"])
    for c in ("n_profiles", "n_pred", "n_realizations"):
        t[c] = t[c].astype(int)
    keep = strata + ["scenario", "precision", "recall", "f1", "n_profiles", "n_pred", "n_realizations"]
    out = t[keep]
    return out.drop(columns="_all") if "_all" in out.columns else out


def macro_by_stratum(tbl: pd.DataFrame, strata) -> pd.DataFrame:
    """Macro-F1 y balanced accuracy (= recall medio) por estrato a partir de prf_table."""
    strata = list(strata)
    g = tbl.groupby(strata, observed=True) if strata else tbl.assign(_a=1).groupby("_a")
    out = g.agg(macro_f1=("f1", "mean"), balanced_accuracy=("recall", "mean"), n_profiles=("n_profiles", "sum")).reset_index()
    return out.drop(columns="_a", errors="ignore")


def confusion_by_stratum(meta: pd.DataFrame, y_pred, strata, true_col: str = "scenario") -> pd.DataFrame:
    d = pd.DataFrame({c: meta[c].to_numpy() for c in strata})
    d["true"], d["pred"] = meta[true_col].to_numpy(), np.asarray(y_pred)
    return d.groupby(list(strata) + ["true", "pred"], observed=True, sort=False).size().rename("count").reset_index()


# --------------------------------------------------------------------------- deltas pareados y bootstrap


def paired_deltas(raw: pd.DataFrame, metrics, keys=("model", "phase")) -> pd.DataFrame:
    """delta = métrica(PCA-k) - métrica(Raw) en cada split (semilla); media, sd y percentiles 2.5/97.5."""
    base = raw[raw["representation"] == "Raw"].set_index(list(keys) + ["seed"])[list(metrics)]
    pca = raw[raw["representation"] == "PCA"].copy()
    j = pca.join(base, on=list(keys) + ["seed"], rsuffix="_raw")
    rows = []
    for m in metrics:
        j[f"d_{m}"] = j[m] - j[f"{m}_raw"]
    for k, g in j.groupby(list(keys) + ["components"], sort=False):
        for m in metrics:
            d = g[f"d_{m}"].to_numpy()
            rows.append(dict(zip(list(keys) + ["components"], k)) | dict(
                metric=m, n_splits=len(d), delta_mean=d.mean(), delta_sd=d.std(ddof=1) if len(d) > 1 else np.nan,
                p2_5=np.percentile(d, 2.5), p97_5=np.percentile(d, 97.5)))
    return pd.DataFrame(rows)


def realization_confusions(meta: pd.DataFrame, y_pred, mask=None, classes=SCENARIOS):
    """(ids, C) con C[r, i, j] = nº de perfiles de la realización r con clase real i predichos como j."""
    m = np.ones(len(meta), bool) if mask is None else np.asarray(mask)
    d = pd.DataFrame({"r": meta["realization"].to_numpy()[m], "t": meta["scenario"].to_numpy()[m], "p": np.asarray(y_pred)[m]})
    ids = np.array(sorted(meta["realization"].unique()))
    pos = {r: i for i, r in enumerate(ids)}
    cls = {c: i for i, c in enumerate(classes)}
    C = np.zeros((len(ids), len(classes), len(classes)))
    g = d.groupby(["r", "t", "p"]).size()
    for (r, t, p), n in g.items():
        if t in cls and p in cls:
            C[pos[r], cls[t], cls[p]] = n
    return ids, C


def _macro_bal(S: np.ndarray):
    tp = np.einsum("bii->bi", S)
    tn, pn = S.sum(2), S.sum(1)
    f1 = np.where(tn + pn > 0, 2 * tp / np.maximum(tn + pn, 1), 0.0)
    rec = np.where(tn > 0, tp / np.maximum(tn, 1), np.nan)
    return f1.mean(1), np.nanmean(rec, axis=1)


def bootstrap_metrics(C_by_rep: dict, strata_of_real, B: int, rng) -> dict:
    """Bootstrap por REALIZACIÓN (estratificado por escenario×severidad) dentro de un split fijo de test.
    Las mismas réplicas se usan para todas las representaciones (diferencias pareadas)."""
    strata_of_real = np.asarray(strata_of_real)
    R = len(strata_of_real)
    W = np.zeros((B, R))
    groups = [np.flatnonzero(strata_of_real == s) for s in np.unique(strata_of_real)]
    for b in range(B):
        for g in groups:
            W[b] += np.bincount(rng.choice(g, size=len(g), replace=True), minlength=R)
    out = {}
    for rep, C in C_by_rep.items():
        S = np.einsum("br,rij->bij", W, C)
        out[rep] = _macro_bal(S)
    return out


# --------------------------------------------------------------------------- NRMSE


def profile_errors(rep24, X: np.ndarray, ks, eps: float = 1e-9) -> dict:
    """Errores de reconstrucción por perfil para cada k (PCA-k = primeras k componentes de PCA-24).
    Devuelve {k: dict(sq_err, rmse_w, nrmse_sd, nrmse_mean)} con arrays por perfil."""
    Zs = rep24.pca.transform(rep24.scaler.transform(X))
    comps, mean = rep24.pca.components_, rep24.pca.mean_
    sd, mu = X.std(axis=1), X.mean(axis=1)
    out = {}
    for k in ks:
        Xh = rep24.scaler.inverse_transform(Zs[:, :k] @ comps[:k] + mean)
        mse = ((X - Xh) ** 2).mean(axis=1)
        rmse = np.sqrt(mse)
        out[k] = dict(
            sq_err=mse, rmse_w=rmse,
            nrmse_sd=np.where(sd > eps, rmse / np.maximum(sd, eps), np.nan),
            nrmse_mean=np.where(mu > eps, rmse / np.maximum(mu, eps), np.nan),
        )
    return out


def nrmse_table(meta: pd.DataFrame, X: np.ndarray, errs: dict, group_cols=("scenario", "severity", "phase")) -> pd.DataFrame:
    """Tabla por k y grupo. 'nrmse_sd_subset' = RMSE del grupo / SD de todos los valores originales del grupo
    (no mezcla escalas entre escenarios: cada grupo tiene su propia escala)."""
    g = list(group_cols)
    base = meta[g].reset_index(drop=True).copy()
    base["_sum"], base["_sumsq"], base["_n"] = X.sum(axis=1), (X ** 2).sum(axis=1), X.shape[1]
    sc = base.groupby(g, observed=True).agg(s=("_sum", "sum"), ss=("_sumsq", "sum"), n=("_n", "sum"))
    sc["sd_group"] = np.sqrt(np.maximum(sc["ss"] / sc["n"] - (sc["s"] / sc["n"]) ** 2, 0))
    rows = []
    for k, e in errs.items():
        d = base[g].copy()
        for key in ("sq_err", "rmse_w", "nrmse_sd", "nrmse_mean"):
            d[key] = e[key]
        a = d.groupby(g, observed=True).agg(
            n_profiles=("sq_err", "size"), mse=("sq_err", "mean"), rmse_w_mean=("rmse_w", "mean"),
            nrmse_sd_mean=("nrmse_sd", "mean"), nrmse_mean_mean=("nrmse_mean", "mean")).join(sc["sd_group"])
        a["rmse_w_pooled"] = np.sqrt(a["mse"])
        a["rmse_kw_pooled"] = a["rmse_w_pooled"] / 1000.0
        a["nrmse_sd_subset"] = a["rmse_w_pooled"] / a["sd_group"].replace(0, np.nan)
        a["components"] = k
        rows.append(a.drop(columns=["mse"]).reset_index())
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------- ventanas (P11)


def window_masks(meta: pd.DataFrame, followups=(90, 180, 365), buffers=(7, 14, 30), matched=(180, 360)) -> dict:
    d, te = meta["days_since_start"].to_numpy(), meta["transition_end"].to_numpy()
    ph = meta["phase"].to_numpy()
    out = {"current:transition": ph == "transition", "current:developed": ph == "developed", "current:pre": ph == "pre"}
    for w in followups:
        out[f"followup_{w}d_after_endpoint"] = (d >= te) & (d < te + w)
    for w in matched:
        out[f"matched_first_{w}d_from_onset"] = (d >= 0) & (d < w)
    for b in buffers:
        out[f"transition_excl_buffer_{b}d"] = (ph == "transition") & (d >= b) & (d < te - b)
        out[f"pre_excl_buffer_{b}d"] = (ph == "pre") & (d < -b)
    return out


# --------------------------------------------------------------------------- indicadores físicos


def ns_indicators(X: np.ndarray, night, day, morning, eps: float = 1e-9) -> pd.DataFrame:
    """night_day_ratio = media(horas de noche)/media(horas de día); morning_rebound = energía(mañana)/energía diaria."""
    night, day, morning = list(night), list(day), list(morning)
    dm = X[:, day].mean(axis=1)
    tot = X.sum(axis=1)
    return pd.DataFrame({
        "daily_mean": X.mean(axis=1),
        "night_day_ratio": np.where(dm > eps, X[:, night].mean(axis=1) / np.maximum(dm, eps), np.nan),
        "morning_rebound": np.where(tot > eps, X[:, morning].sum(axis=1) / np.maximum(tot, eps), np.nan),
        "peak_hour": X.argmax(axis=1).astype(float),
    })


def normalize_profiles(X: np.ndarray, mode: str, eps: float = 1e-9) -> np.ndarray:
    """Modos de representación: raw_level_shape, daily_mean_normalized (x/mean, media 1),
    daily_energy_normalized (x/sum, suma 1), daily_zscore_shape ((x-mean)/sd). Protegidos frente a ~0."""
    if mode == "raw_level_shape":
        return X
    m = X.mean(axis=1, keepdims=True)
    if mode == "daily_mean_normalized":
        return np.where(m > eps, X / np.maximum(m, eps), 1.0)
    if mode == "daily_energy_normalized":
        s = X.sum(axis=1, keepdims=True)
        return np.where(s > eps, X / np.maximum(s, eps), 1.0 / X.shape[1])
    if mode == "daily_zscore_shape":
        sd = X.std(axis=1, keepdims=True)
        return np.where(sd > eps, (X - m) / np.maximum(sd, eps), 0.0)
    raise ValueError(mode)


# --------------------------------------------------------------------------- Hellinger


def hellinger_discrete(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))


def hist_prob(v: np.ndarray, edges: np.ndarray) -> np.ndarray:
    c, _ = np.histogram(np.clip(v, edges[0], edges[-1]), bins=edges)
    return c / max(c.sum(), 1)


def gaussian_hellinger(a: np.ndarray, b: np.ndarray, ridge: float = 1e-6) -> float:
    """Hellinger entre dos nubes ajustadas con gaussianas multivariantes (aproximación, vía Bhattacharyya)."""
    mu1, mu2 = a.mean(0), b.mean(0)
    S1 = np.atleast_2d(np.cov(a, rowvar=False)) + ridge * np.eye(a.shape[1])
    S2 = np.atleast_2d(np.cov(b, rowvar=False)) + ridge * np.eye(a.shape[1])
    S = 0.5 * (S1 + S2)
    dm = mu1 - mu2
    db = 0.125 * dm @ np.linalg.solve(S, dm) + 0.5 * (np.linalg.slogdet(S)[1] - 0.5 * (np.linalg.slogdet(S1)[1] + np.linalg.slogdet(S2)[1]))
    return float(np.sqrt(max(0.0, 1.0 - np.exp(-max(db, 0.0)))))


def normalized_entropy(P: np.ndarray) -> np.ndarray:
    P = np.clip(P, 1e-12, 1.0)
    return -(P * np.log(P)).sum(axis=1) / np.log(P.shape[1])
