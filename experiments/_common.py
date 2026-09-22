"""Shared plumbing for the experiment scripts: where results go, how they are
written, and the figure style they share.

The split between a run stage and a figure stage is deliberate. Sweeps take
minutes to hours; figures take seconds. Writing the raw per-condition results
to parquet and regenerating every figure from that file means a reviewer can
reproduce the plots on a laptop without re-running the study, and that a
change to a label or an axis never silently re-rolls the numbers underneath
it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

# Reference data-visualisation palette, used unchanged. Only one categorical
# hue is ever in play here: the observed quantity. Everything a chart compares
# it against is a theoretical reference, which wears neutral ink and a dashed
# stroke so it reads as the baseline rather than as a second series.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
OBSERVED = "#2a78d6"
BAND = "#d9d8d4"


def write_results(rows: list[dict], name: str) -> Path:
    """Write one row per condition to results/<name>.parquet."""
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{name}.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def read_results(name: str):
    """Read a results table back as a list of dicts."""
    return pq.read_table(RESULTS_DIR / f"{name}.parquet").to_pylist()


def figure(nrows: int = 1, ncols: int = 1, size=(9.0, 4.2)):
    """A figure on the chart surface, with recessive axes and no chart junk."""
    fig, axes = plt.subplots(nrows, ncols, figsize=size, facecolor=SURFACE)
    for ax in np.atleast_1d(np.asarray(axes)).ravel():
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(BAND)
            ax.spines[side].set_linewidth(1.0)
        ax.tick_params(colors=INK_MUTED, labelsize=9, length=3, width=1.0)
        ax.grid(True, color=BAND, linewidth=0.8, alpha=0.7)
        ax.set_axisbelow(True)
    return fig, axes


def save_figure(fig, name: str) -> Path:
    """Write figures/<name>.png."""
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def label(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=8)
    ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)
