"""Guardado de figuras: cada imagen se acompaña de un .txt que la describe (texto + datos dibujados)."""
from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.runtime import git_commit


def _num(v) -> str:
    try:
        v = float(v)
    except Exception:
        return str(v)
    if not np.isfinite(v):
        return "nan"
    return f"{v:.4g}"


def _trend(y: np.ndarray) -> str:
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 3:
        return ""
    d = np.diff(y)
    tol = 1e-9 + 0.005 * (np.ptp(y) or 1.0)
    if np.all(d >= -tol):
        return "monotonically increasing"
    if np.all(d <= tol):
        return "monotonically decreasing"
    return "non-monotonic"


def _line_text(ln) -> str | None:
    x, y = np.asarray(ln.get_xdata(), dtype=object), np.asarray(ln.get_ydata(), dtype=object)
    label = ln.get_label()
    try:
        xf, yf = x.astype(float), y.astype(float)
    except Exception:  # fechas: describir solo por tamaño
        return f"  - line '{label}': {len(y)} points (date axis), y from {_num(np.nanmin(y.astype(float)))} to {_num(np.nanmax(y.astype(float)))}"
    if len(xf) == 0:
        return None
    if len(xf) == 2 and xf[0] == xf[1]:
        return f"  - vertical reference line at x={_num(xf[0])}" + (f" ('{label}')" if not label.startswith("_") else "")
    if len(yf) == 2 and yf[0] == yf[1]:
        return f"  - horizontal reference line at y={_num(yf[0])}" + (f" ('{label}')" if not label.startswith("_") else "")
    if label.startswith("_"):
        return None
    return (f"  - series '{label}': {len(yf)} points, x {_num(xf.min())}..{_num(xf.max())}; "
            f"y first={_num(yf[0])}, last={_num(yf[-1])}, min={_num(np.nanmin(yf))}, max={_num(np.nanmax(yf))}, "
            f"mean={_num(np.nanmean(yf))}; {_trend(yf)}").rstrip("; ")


def describe_figure(fig, purpose: str = "", how_to_read: str = "") -> str:
    fig.canvas.draw()
    out = []
    if purpose:
        out.append(f"PURPOSE: {purpose}")
    if how_to_read:
        out.append(f"HOW TO READ: {how_to_read}")
    sup = getattr(fig, "_suptitle", None)
    if sup is not None and sup.get_text():
        out.append(f"FIGURE TITLE: {sup.get_text()}")
    w, h = fig.get_size_inches()
    out.append(f"LAYOUT: {len(fig.axes)} panel(s), figure size {w:.1f} x {h:.1f} in.")
    for k, ax in enumerate(fig.axes, 1):
        out.append(f"\nPANEL {k}" + (f" - title: '{ax.get_title()}'" if ax.get_title() else ""))
        if ax.get_xlabel():
            out.append(f"  x axis: '{ax.get_xlabel()}'")
        if ax.get_ylabel():
            out.append(f"  y axis: '{ax.get_ylabel()}'")
        xt = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
        if xt:
            out.append("  x tick labels: " + ", ".join(xt[:24]) + (" ..." if len(xt) > 24 else ""))
        leg = ax.get_legend()
        if leg is not None:
            out.append("  legend: " + "; ".join(t.get_text() for t in leg.get_texts()))
        for ln in ax.get_lines():
            t = _line_text(ln)
            if t:
                out.append(t)
        for im in ax.images:
            a = np.asarray(im.get_array())
            out.append(f"  - image/heatmap {a.shape}, values {_num(a.min())}..{_num(a.max())}")
        if ax.texts:
            rows: dict[float, list] = {}
            for t in ax.texts:
                rows.setdefault(round(t.get_position()[1], 3), []).append((t.get_position()[0], t.get_text()))
            out.append("  cell annotations (row by row): " + " | ".join(
                " ".join(s for _, s in sorted(r)) for _, r in sorted(rows.items())))
        bars = [p for p in ax.patches if p.get_width() > 0 and hasattr(p, "get_height")]
        if bars:
            out.append("  bars (height): " + ", ".join(_num(p.get_height()) for p in bars[:40]) + (" ..." if len(bars) > 40 else ""))
        for c in ax.collections:
            try:
                off = np.asarray(c.get_offsets())
                if len(off) > 1:
                    out.append(f"  - point cloud: {len(off)} points, x {_num(off[:, 0].min())}..{_num(off[:, 0].max())}, "
                               f"y {_num(off[:, 1].min())}..{_num(off[:, 1].max())}")
            except Exception:
                pass
    for lg in fig.legends:
        out.append("\nFigure legend: " + "; ".join(t.get_text() for t in lg.get_texts()))
    return "\n".join(out)


def save_fig(fig, path, purpose: str = "", how_to_read: str = "", dpi: int = 130, close: bool = True) -> Path:
    """Guarda la figura y escribe <nombre>.txt con su descripción; actualiza figures_manifest.json."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
        fig.savefig(path, dpi=dpi)
    else:
        fig.savefig(path)
    script = Path(sys.argv[0]).name if sys.argv and sys.argv[0] else "unknown"
    commit = git_commit()
    text = describe_figure(fig, purpose, how_to_read)
    header = (f"FILE: {path.name}\nGENERATED: {_dt.datetime.now().isoformat(timespec='seconds')} | script: {script} | git: {commit[:10]}\n\n")
    path.with_suffix(".txt").write_text(header + text + "\n", encoding="utf-8")
    man = path.parent / "figures_manifest.json"
    try:
        d = json.loads(man.read_text(encoding="utf-8")) if man.exists() else {}
        d[path.name] = {"description": path.with_suffix(".txt").name, "script": script, "git_commit": commit, "purpose": purpose}
        man.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    if close:
        plt.close(fig)
    return path
