"""Shared analysis rules, results I/O and figure style for the experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from matplotlib.figure import Figure

from posetrust.stats import ConsistencyReport, benjamini_hochberg

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

# Analysis rules fixed in HYPOTHESES.md.
ALPHA = 0.05
MIN_CONVERGED_FRACTION = 0.5
# Usable conditions below this converged fraction are lower bounds.
SURVIVORSHIP_FRACTION = 0.9
BOOTSTRAP_RESAMPLES = 2000

# Reference palette: the observed series is blue, references are neutral.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
OBSERVED = "#2a78d6"
BAND = "#d9d8d4"


def bootstrap_interval(values: np.ndarray) -> tuple[float, float]:
    """95% bootstrap interval for the mean, resampling runs."""
    values = np.ascontiguousarray(values, dtype=float)
    rng = np.random.default_rng(np.frombuffer(values.tobytes(), dtype=np.uint32))
    boot = rng.choice(
        values, size=(BOOTSTRAP_RESAMPLES, values.size), replace=True
    ).mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return float(low), float(high)


def calibration(result) -> dict:
    """Calibration summary for one condition, excluding non-converged runs."""
    converged = result.converged
    fraction = float(converged.mean())
    usable = fraction >= MIN_CONVERGED_FRACTION
    counts = {
        "n_runs": int(result.n_runs),
        "converged": int(converged.sum()),
        "converged_fraction": fraction,
        "usable": usable,
    }
    values = result.nees_full[converged]
    if values.size < 2:
        # Too few converged runs for any statistic: a convergence failure.
        nan = float("nan")
        return {
            "dof": int(result.free_dof),
            **counts,
            **dict.fromkeys(
                ("mean_nees", "ratio", "band_low", "band_high", "ci_low",
                 "ci_high", "pvalue"),
                nan,
            ),
            "verdict": "convergence failure",
        }
    report = ConsistencyReport(values, result.free_dof, ALPHA)
    low, high = report.acceptance

    ci_low, ci_high = np.array(bootstrap_interval(values)) / report.dof
    return {
        "dof": report.dof,
        **counts,
        "mean_nees": report.mean,
        "ratio": report.mean / report.dof,
        "band_low": low / report.dof,
        "band_high": high / report.dof,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "pvalue": report.pvalue,
        "verdict": report.verdict if usable else "convergence failure",
    }


def mark_fdr(rows: list[dict]) -> None:
    """Mark which usable conditions survive Benjamini-Hochberg."""
    usable = [r for r in rows if r["usable"]]
    flagged = benjamini_hochberg(np.array([r["pvalue"] for r in usable]), ALPHA)
    for row, reject in zip(usable, flagged):
        row["significant_after_fdr"] = bool(reject)
    for row in rows:
        row.setdefault("significant_after_fdr", False)


def survivorship_warning(rows: list[dict], describe) -> None:
    """Flag usable conditions that lost enough runs to be biased."""
    suspect = [
        r for r in rows if r["usable"] and r["converged_fraction"] < SURVIVORSHIP_FRACTION
    ]
    if not suspect:
        return
    print(
        f"\n  Survivorship warning: these dropped more than "
        f"{1 - SURVIVORSHIP_FRACTION:.0%} of runs, so"
    )
    print("  their ratios are biased towards calibration and are lower bounds:")
    for row in suspect:
        print(
            f"    {describe(row)}: {row['converged']}/{row['n_runs']} converged, "
            f"ratio {row['ratio']:.1f}"
        )


def write_results(rows: list[dict], name: str) -> Path:
    """Write one row per condition to results/<name>.parquet."""
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{name}.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def read_results(name: str):
    """Read a results table back as a list of dicts."""
    return pq.read_table(RESULTS_DIR / f"{name}.parquet").to_pylist()


def figure(
    nrows: int = 1,
    ncols: int = 1,
    size=(9.0, 4.2),
    sharey: bool = False,
    sharex: bool = False,
):
    """Empty figure in the shared chart style."""
    fig = Figure(figsize=size, facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(nrows, ncols, sharey=sharey, sharex=sharex)
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
    """SVG canvas class, with a stub where the Agg extension is blocked."""
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
    """Write figures/<name>.svg, byte-identical across reruns."""
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / f"{name}.svg"
    _svg_canvas()(fig)

    # Fixed hash salt and no timestamp, so an unchanged figure gives no diff.
    matplotlib.rcParams["svg.hashsalt"] = name
    fig.savefig(path, facecolor=SURFACE, metadata={"Date": None})
    return path


def label(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=8)
    ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)
