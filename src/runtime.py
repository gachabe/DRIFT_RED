"""Utilidades de ejecución: caché del dataset, metadatos, checksums, hardware, logging de etapas."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip())
    except Exception:
        return False


def package_versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for p in ("numpy", "pandas", "scipy", "sklearn", "matplotlib", "yaml", "tqdm", "psutil", "joblib"):
        try:
            m = __import__(p)
            out[p] = getattr(m, "__version__", "?")
        except Exception:
            out[p] = None
    return out


def hardware() -> dict:
    import os

    d = {"platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count()}
    try:
        import psutil

        d["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:
        d["ram_gb"] = None
    return d


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def checksums(root, patterns=("*.csv", "*.json", "*.md", "*.png", "*.txt", "*.svg", "*.pdf")) -> pd.DataFrame:
    root = Path(root)
    rows = []
    for pat in patterns:
        for f in sorted(root.rglob(pat)):
            if f.name in ("output_checksums.csv",):
                continue
            rows.append({"file": str(f.relative_to(root)).replace("\\", "/"), "bytes": f.stat().st_size, "sha256": sha256_file(f)})
    return pd.DataFrame(rows).drop_duplicates("file")


def rcfg(cfg: dict, key: str, default=None):
    """cfg['revision'][key] con valor por defecto."""
    return (cfg.get("revision") or {}).get(key, default)


def out_dir(cfg: dict) -> Path:
    p = Path(cfg.get("results_dir", f"results/{cfg['mode']}"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def rev_dir(cfg: dict, name: str | None = None) -> Path:
    p = out_dir(cfg) / "revision"
    if name:
        p = p / name
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------- caché del dataset


def _cache_prefix(cfg: dict) -> Path:
    key = json.dumps(
        {"data": cfg["data"], "phases": cfg.get("phases"), "labels": cfg.get("labels"), "seed": cfg["seed"]},
        sort_keys=True, default=str,
    )
    h = hashlib.md5(key.encode()).hexdigest()[:10]
    d = Path(cfg.get("cache_dir", "data/processed"))
    d.mkdir(parents=True, exist_ok=True)
    return d / f"ds_{cfg['mode']}_{h}"


def load_dataset(cfg: dict, use_cache: bool = True, weather: bool = True):
    """Dataset (cacheado en disco) con, si existe, el tiempo diario fusionado por fecha (antes del split)."""
    from src.data import Dataset, build_dataset
    from src.weather import add_weather

    pre = _cache_prefix(cfg)
    fx, fm, fr = Path(str(pre) + ".X.npy"), Path(str(pre) + ".meta.pkl"), Path(str(pre) + ".real.pkl")
    ds = None
    if use_cache and fx.exists() and fm.exists() and fr.exists():
        try:
            ds = Dataset(X=np.load(fx), meta=pd.read_pickle(fm), realizations=pd.read_pickle(fr))
        except Exception:
            ds = None
    if ds is None:
        ds = build_dataset(cfg)
        if use_cache:
            try:
                np.save(fx, ds.X)
                ds.meta.to_pickle(fm)
                ds.realizations.to_pickle(fr)
            except Exception as e:  # disco lleno, etc.
                print("aviso: no se pudo cachear el dataset:", e)
    if weather:
        ds = Dataset(X=ds.X, meta=add_weather(ds.meta, cfg), realizations=ds.realizations)
    return ds


# --------------------------------------------------------------------------- ejecución de etapas


class StageLog:
    """Ejecuta etapas capturando errores; escribe revision/run_log.txt y stage_status.json."""

    def __init__(self, outdir: Path, resume: bool = True):
        self.dir = Path(outdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.status_path = self.dir / "stage_status.json"
        self.log_path = self.dir / "run_log.txt"
        self.status = json.loads(self.status_path.read_text()) if (resume and self.status_path.exists()) else {}

    def _log(self, msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def run(self, name: str, fn, *args, force: bool = False, **kw):
        if not force and self.status.get(name, {}).get("state") == "ok":
            self._log(f"{name}: ya hecha (resume), se salta")
            return None
        self._log(f"{name}: INICIO")
        t0 = time.time()
        try:
            res = fn(*args, **kw)
            self.status[name] = {"state": "ok", "seconds": round(time.time() - t0, 1)}
            self._log(f"{name}: OK ({time.time() - t0:.0f}s)")
            return res
        except Exception as e:  # nunca abortar el resto de la ejecución nocturna
            tb = traceback.format_exc()
            self.status[name] = {"state": "error", "seconds": round(time.time() - t0, 1), "error": repr(e)}
            self._log(f"{name}: ERROR {e!r}\n{tb}")
            (self.dir / f"error_{name}.txt").write_text(tb, encoding="utf-8")
            return None
        finally:
            self.status_path.write_text(json.dumps(self.status, indent=2), encoding="utf-8")


def load_predictions(outdir, seed: int):
    """(test_idx, {(model, rep_label): códigos de clase}) guardados por 05_seeds."""
    z = np.load(Path(outdir) / "predictions" / f"seed_{seed}.npz")
    preds = {tuple(k.split("|")): z[k] for k in z.files if k != "test_idx"}
    return z["test_idx"], preds


def available_seeds(outdir) -> list[int]:
    return sorted(int(p.stem.split("_")[1]) for p in (Path(outdir) / "predictions").glob("seed_*.npz"))
