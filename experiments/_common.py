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
from matplotlib.figure import Figure

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


def figure(nrows: int = 1, ncols: int = 1, size=(9.0, 4.2), sharey: bool = False):
    """A figure on the chart surface, with recessive axes and no chart junk.

    Built as a bare Figure rather than through pyplot, and laid out by the
    constrained engine rather than an explicit tight_layout call. Both choices
    keep figure *construction* free of any rendering backend, so the plotting
    logic can be exercised by the test suite on a machine that cannot
    rasterise -- only save_figure() below needs a working renderer. It also
    avoids pyplot's global figure registry, which scripts leak.
    """
    fig = Figure(figsize=size, facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(nrows, ncols, sharey=sharey)
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


def _svg_canvas():
    """The SVG canvas class, working around platforms that block Agg.

    matplotlib's backend_svg imports backend_mixed, which imports backend_agg
    at module level, so every output format transitively needs the Agg
    extension even when nothing is rasterised. Where that extension cannot
    load -- an unsigned native binary under Windows Smart App Control, for
    instance -- a pure-vector figure is still perfectly renderable, because
    RendererAgg is imported and then never instantiated.

    So the import is retried against a stub that raises if it is ever really
    used. The stub is installed only after the honest import has already
    failed, so nothing changes on a machine where Agg works.
    """
    try:
        from matplotlib.backends.backend_svg import FigureCanvasSVG
    except ImportError:
        import sys
        import types

        stub = types.ModuleType("matplotlib.backends._backend_agg")

        class _RasterisationUnavailable:
            def __init__(self, *_args, **_kwargs):
                raise RuntimeError(
                    "this figure needs raster rendering, which the Agg "
                    "extension provides and this platform has blocked; "
                    "remove the rasterised element or render elsewhere"
                )

        stub.RendererAgg = _RasterisationUnavailable
        sys.modules.setdefault("matplotlib.backends._backend_agg", stub)
        from matplotlib.backends.backend_svg import FigureCanvasSVG
    return FigureCanvasSVG


def save_figure(fig, name: str) -> Path:
    """Write figures/<name>.svg.

    Vector rather than raster: it renders natively in a README, stays sharp
    at any zoom, is usually smaller for line work, and needs no rasteriser --
    which is what makes the figures reproducible on a locked-down machine.
    """
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / f"{name}.svg"
    _svg_canvas()(fig)

    # Deterministic output. By default matplotlib stamps the SVG with the
    # current time and names its internal elements from a random salt, so
    # regenerating an unchanged figure rewrites most of the file and shows up
    # as a large diff that has to be read to discover it says nothing. Figures
    # are committed here, so that noise would be permanent.
    matplotlib.rcParams["svg.hashsalt"] = name
    fig.savefig(path, facecolor=SURFACE, metadata={"Date": None})
    return path


def label(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=8)
    ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)
