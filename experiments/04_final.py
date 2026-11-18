"""Ejecuta toda la pipeline (según el YAML) y escribe resultados + metadatos."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from src.data import load_config
from src.features import prepare
from src.runtime import git_commit, git_dirty, hardware, package_versions


def main(cfg: dict) -> dict:
    out = Path(cfg.get("results_dir", f"results/{cfg['mode']}"))
    out.mkdir(parents=True, exist_ok=True)
    ds, split = prepare(cfg, out)
    importlib.import_module("experiments.01_eda").run(ds, out)
    reps = importlib.import_module("experiments.02_rom").run(ds, split, cfg, out)
    results = importlib.import_module("experiments.03_ml").run(ds, split, reps, cfg, out)

    summary = {
        "mode": cfg["mode"],
        "note": "El modo demo es un test de integridad, no una fuente de conclusiones." if cfg["mode"] == "demo" else "",
        "n_train_realizations": len(split.train),
        "n_test_realizations": len(split.test),
        "results": results.round(4).to_dict(orient="records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    meta = {
        "mode": cfg["mode"],
        "seed": cfg["seed"],
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "package_versions": package_versions(),
        "hardware": hardware(),
        "python_version": package_versions()["python"],
        "dataset": cfg.get("dataset", "DHDrift 1.0.0"),
        "components": cfg["rom"]["components"],
        "models": cfg["models"],
        "transition_days": cfg.get("phases", {}).get("transition_days"),
        "config": cfg,
        "train_realizations": split.train,
        "test_realizations": split.test,
    }
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nListo. Resultados en {out}/")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(load_config(ap.parse_args().config))
