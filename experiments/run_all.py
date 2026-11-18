"""Ejecuta TODO lo de la guía de revisión (P0–P15) de una sola vez, sin supervisión.

    python -m experiments.run_all --config configs/full.yaml

- Cada etapa se ejecuta aislada: si una falla, se guarda el error (revision/error_<etapa>.txt) y se sigue con las demás.
- Reanudable: las etapas ya terminadas (revision/stage_status.json) y las semillas ya hechas se saltan; usa --no-resume para repetir.
- Al final escribe revision/REVISION_REPORT.md y output_checksums.csv.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from src.consistency import check_numbers
from src.data import ZENODO_MD5, load_config
from src.runtime import StageLog, checksums, git_commit, git_dirty, hardware, load_dataset, out_dir, package_versions, rcfg, rev_dir
from src.validation import check_dataset

REV = importlib.import_module("experiments.07_revision")


def p0_tests(cfg):
    rc = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"]).returncode
    if rc != 0:
        raise RuntimeError("pytest falló: la pipeline no está verificada (ver salida)")


def p0_validate(cfg):
    if not check_dataset(cfg["data"].get("root", "data/raw"), cfg.get("labels")):
        raise RuntimeError("la validación del dataset falló")


def p0_freeze(cfg):
    d = rev_dir(cfg, "P0_freeze")
    info = {"git_commit": git_commit(), "git_dirty": git_dirty(), "package_versions": package_versions(), "hardware": hardware(),
            "config": cfg, "date": time.strftime("%Y-%m-%d %H:%M:%S")}
    zpath = Path(cfg["data"].get("root", "data/raw")) / "dhdrift_dataset.zip"
    if zpath.exists():
        h = hashlib.md5()
        with open(zpath, "rb") as f:
            while b := f.read(1 << 20):
                h.update(b)
        info["dataset_zip_md5"], info["dataset_zip_md5_expected"] = h.hexdigest(), ZENODO_MD5
        info["dataset_is_original_zenodo_release"] = h.hexdigest() == ZENODO_MD5
    else:
        info["dataset_is_original_zenodo_release"] = "no verificable: falta data/raw/dhdrift_dataset.zip (ejecuta `make data`)"
    ign = subprocess.run(["git", "check-ignore", "-q", str(Path(cfg["data"].get("root", "data/raw")) / "x.csv")]).returncode == 0
    info["data_raw_is_gitignored"] = ign
    (d / "freeze_metadata.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    if info.get("dataset_is_original_zenodo_release") is False:
        raise RuntimeError("el md5 del zip no coincide con la versión original de Zenodo")


def p0_final(cfg):
    importlib.import_module("experiments.04_final").main(cfg)


def p0_split_check(cfg):
    sd = out_dir(cfg) / "splits"
    bad = []
    for tr in sorted(sd.glob("seed_*_train.csv")):
        te = tr.with_name(tr.name.replace("_train", "_test"))
        if set(pd.read_csv(tr)["realization"]) & set(pd.read_csv(te)["realization"]):
            bad.append(tr.name)
    if bad:
        raise AssertionError(f"solapamiento train/test en {bad}")


def p0_consistency(cfg):
    ref = Path("reference/manuscript_numbers.csv")
    df = check_numbers(out_dir(cfg), ref)
    df.to_csv(rev_dir(cfg, "P0_freeze") / "manuscript_consistency.csv", index=False)
    bad = df[~df["ok"]]
    if len(bad):
        raise AssertionError(f"{len(bad)}/{len(df)} cifras del manuscrito no coinciden con lo generado (ver P0_freeze/manuscript_consistency.csv)")


def p13_note(cfg):
    d = rev_dir(cfg, "P13_overlapping_interventions")
    (d / "NOT_IMPLEMENTED.md").write_text(
        "# P13 — intervenciones solapadas: NO implementado\n\nRequiere **simular datos nuevos** (EXP+REF, EXP+NS, REF+NS, las tres; orden de inicio, solape y severidad "
        "independientes) con el generador de DHDrift, que no forma parte de este repositorio, y un rediseño multietiqueta (precisión/recall/F1 por etiqueta y subset accuracy). "
        "Combinar series ya existentes a posteriori no sería una intervención simulada de verdad. Es trabajo futuro (la propia guía lo marca como extensión, no como comprobación de la revisión).\n\n"
        "Especificación para cuando haya datos: split por realización/red; controles de una sola intervención y sin intervención; métricas por etiqueta + exact-match.\n", encoding="utf-8")


def final_report(cfg, log):
    out = out_dir(cfg)
    rd = rev_dir(cfg)
    lines = ["# REVISION REPORT\n", f"Generado: {time.strftime('%Y-%m-%d %H:%M:%S')} | git {git_commit()[:10]}{' (dirty)' if git_dirty() else ''}\n",
             "## Estado de las etapas\n", "| etapa | estado | segundos |", "|---|---|---|"]
    for k, v in log.status.items():
        lines.append(f"| {k} | {v['state']}{' — ' + v['error'][:80] if v['state'] == 'error' else ''} | {v['seconds']} |")
    lines += ["", "Los errores completos están en `revision/error_<etapa>.txt`; el log, en `revision/run_log.txt`.", "", "## Figuras (cada una con su .txt)\n"]
    for png in sorted(list(out.rglob("*.png")) + list(out.rglob("*.svg")) + list(out.rglob("*.pdf"))):
        t = png.with_suffix(".txt")
        first = ""
        if t.exists():
            first = next((l[9:] for l in t.read_text(encoding="utf-8").splitlines() if l.startswith("PURPOSE:")), "")
        lines.append(f"- `{png.relative_to(out).as_posix()}` — {first}")
    (rd / "REVISION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    checksums(out).to_csv(rd / "output_checksums.csv", index=False)


def main(cfg: dict, n_seeds: int | None, only: set, skip: set, resume: bool, skip_tests: bool) -> None:
    n_seeds = n_seeds or int(rcfg(cfg, "n_seeds", 10))
    log = StageLog(rev_dir(cfg), resume=resume)
    cache: dict = {}

    def get_ds():
        if "ds" not in cache:
            cache["ds"] = load_dataset(cfg)
        return cache["ds"]

    out = out_dir(cfg)
    stages = [  # (nombre, etiquetas, función)
        ("P0_tests", {"P0", "P15"}, lambda: None if skip_tests else p0_tests(cfg)),
        ("P0_validate_dataset", {"P0"}, lambda: p0_validate(cfg)),
        ("P0_freeze_metadata", {"P0"}, lambda: p0_freeze(cfg)),
        ("P0_04_final", {"P0", "P14"}, lambda: p0_final(cfg)),
        ("P1_P2_P8_05_seeds", {"P0", "P1", "P2", "P8"}, lambda: importlib.import_module("experiments.05_seeds").main(cfg, n_seeds, resume=resume)),
        ("P3_06_progress", {"P0", "P3"}, lambda: importlib.import_module("experiments.06_progress").main(cfg, int(rcfg(cfg, "progress_bins", 5)))),
        ("P0_split_disjoint_check", {"P0"}, lambda: p0_split_check(cfg)),
        ("P0_manuscript_consistency", {"P0"}, lambda: p0_consistency(cfg)),
        ("P6_pca_ns_interpretation", {"P6"}, lambda: REV.stage_pca_interpretation(cfg, get_ds(), out)),
        ("P7_hellinger", {"P7"}, lambda: REV.stage_hellinger(cfg, get_ds(), out)),
        ("P14_workflow_figure", {"P14"}, lambda: REV.stage_workflow(cfg, None, out)),
        ("P8_baseline_four_class", {"P8"}, lambda: REV.stage_baseline(cfg, get_ds(), out)),
        ("P11_transition_sensitivity", {"P11"}, lambda: REV.stage_windows(cfg, get_ds(), out)),
        ("P4_weather_season", {"P4"}, lambda: REV.stage_weather(cfg, get_ds(), out)),
        ("P12_pca_fit_regimes", {"P12"}, lambda: REV.stage_pca_regimes(cfg, get_ds(), out)),
        ("P5_normalization", {"P5"}, lambda: REV.stage_normalization(cfg, get_ds(), out)),
        ("P9_noise_robustness", {"P9"}, lambda: REV.stage_perturbation(cfg, get_ds(), out)),
        ("P9_fault_anomaly", {"P9"}, lambda: REV.stage_faults(cfg, get_ds(), out)),
        ("P10_resource_benchmark", {"P10"}, lambda: REV.stage_benchmark(cfg, get_ds(), out)),
        ("P10_rf_sensitivity", {"P10"}, lambda: REV.stage_rf_sensitivity(cfg, get_ds(), out)),
        ("P13_overlapping_note", {"P13"}, lambda: p13_note(cfg)),
    ]
    for name, tags, fn in stages:
        if only and not (tags & only):
            continue
        if tags & skip and not (only and tags & only):
            continue
        log.run(name, fn)
        cache.pop("tmp", None)
    log.run("REPORT", final_report, cfg, log, force=True)
    log._log("TODO TERMINADO. Revisa results/<modo>/revision/REVISION_REPORT.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-seeds", type=int, default=None)
    ap.add_argument("--only", default="", help="p.ej. P3,P5 (por número de punto de la guía)")
    ap.add_argument("--skip", default="", help="p.ej. P9,P10")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--skip-tests", action="store_true")
    a = ap.parse_args()
    split = lambda s: {x.strip().upper() for x in s.split(",") if x.strip()}
    main(load_config(a.config), a.n_seeds, split(a.only), split(a.skip), not a.no_resume, a.skip_tests)
