"""Shared statistics, results files and figures for the experiments."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from posetrust.stats import ConsistencyReport, benjamini_hochberg

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
ALPHA = 0.05
# Below this share of converged runs a condition is a convergence failure, not a calibration result.
MIN_CONVERGED_FRACTION = 0.5


def calibration(result) -> dict:
    """Mean NEES per dof and its chi-squared verdict, over the converged runs."""
    converged = result.converged
    row = {"n_runs": int(result.n_runs), "converged": int(converged.sum()), "dof": int(result.free_dof)}
    row["usable"] = bool(converged.mean() >= MIN_CONVERGED_FRACTION and converged.sum() >= 2)
    if not row["usable"]:
        return {**row, "ratio": float("nan"), "pvalue": float("nan"), "verdict": "convergence failure"}
    report = ConsistencyReport(result.nees_full[converged], result.free_dof, ALPHA)
    return {**row, "ratio": report.mean / report.dof, "pvalue": report.pvalue, "verdict": report.verdict}


def mark_fdr(rows: list[dict], by: tuple[str, ...] = ()) -> None:
    """Benjamini-Hochberg over the usable rows, separately for each group sharing the keys in by."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in by), []).append(row)
    for group in groups.values():
        usable = [r for r in group if r["usable"]]
        for row, reject in zip(usable, benjamini_hochberg(np.array([r["pvalue"] for r in usable]), ALPHA)):
            row["significant_after_fdr"] = bool(reject)
        for row in group:
            row.setdefault("significant_after_fdr", False)


def overconfident(row) -> bool:
    return row["usable"] and row["significant_after_fdr"] and row["ratio"] > 1.0


def parallel(fn, tasks):
    """fn(*task) for every task, across processes, in task order."""
    with ProcessPoolExecutor() as pool:
        yield from (future.result() for future in [pool.submit(fn, *task) for task in tasks])


def write_results(rows: list[dict], name: str) -> None:
    (ROOT / "results").mkdir(exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), ROOT / "results" / f"{name}.parquet")


def read_results(name: str) -> list[dict]:
    return pq.read_table(ROOT / "results" / f"{name}.parquet").to_pylist()


def save_figure(fig, name: str) -> None:
    """figures/<name>.svg, identical across reruns."""
    (ROOT / "figures").mkdir(exist_ok=True)
    matplotlib.rcParams["svg.hashsalt"] = name
    fig.savefig(ROOT / "figures" / f"{name}.svg", metadata={"Date": None})
    plt.close(fig)


def layout_lines(ax, series: dict, x, original, colour: str) -> None:
    """One line per layout: the original graph in colour, the others grey."""
    for seed, y in series.items():
        first = seed == original
        ax.plot(
            x,
            y,
            "o-",
            ms=4,
            lw=2 if first else 1,
            color=colour if first else "grey",
            alpha=1 if first else 0.5,
            zorder=3 if first else 2,
            label="original graph" if first else None,
        )
    ax.plot([], [], color="grey", alpha=0.5, label="other layouts")
    ax.axhline(1.0, color="black", lw=1, ls=":")
    ax.set_yscale("log")
