"""E2 (Q1): sweep loop-closure density from odometry-only to densely
constrained; report NEES against density with the chi-squared band (F1).

H1, as pre-registered: overconfidence increases as loop-closure density
falls, the odometry-only end is the worst calibrated, and consistency
improves monotonically as constraints are added.

The sparse end is not an academic corner. An edge system spends most of its
life there: loop closures arrive rarely, and between them the estimate is a
chain of odometry with error accumulating along it. If the reported
covariance is going to mislead anywhere, that is the regime where it matters,
and it is also where the linearisation has the least to hold it down.

Analysis follows HYPOTHESES.md rather than being chosen here: alpha = 0.05
two-sided, Benjamini-Hochberg across the conditions of this sweep, at least
200 runs each, non-converged runs excluded and counted, and a condition with
under half its runs converged reported as a convergence failure rather than
as a calibration result.

Run:  python experiments/e2_loop_closure_density.py [--runs N] [--figures-only]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from _common import (
    BAND,
    INK,
    INK_MUTED,
    OBSERVED,
    figure,
    label,
    read_results,
    save_figure,
    write_results,
)

from posetrust.lie import se2, se3
from posetrust.simulate import NoiseModel, make_scenario, monte_carlo
from posetrust.stats import ConsistencyReport, benjamini_hochberg

GROUPS = [("SE(2)", se2), ("SE(3)", se3)]
DENSITIES = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2]
N_POSES = 12
NOISE = 0.03
TURN = 0.25
ALPHA = 0.05
MIN_CONVERGED_FRACTION = 0.5


def condition(lie, density: float, n_runs: int, seed: int) -> dict:
    """One density: Monte Carlo, then NEES against the band."""
    scenario = make_scenario(
        lie, n_poses=N_POSES, loop_density=density, seed=seed, turn=TURN
    )
    result = monte_carlo(
        lie,
        scenario,
        NoiseModel(np.full(lie.DOF, NOISE)),
        n_runs=n_runs,
        seed=seed + 1,
    )
    converged = result.converged
    fraction = float(converged.mean())
    usable = fraction >= MIN_CONVERGED_FRACTION

    report = ConsistencyReport(result.nees_full[converged], result.free_dof, ALPHA)
    low, high = report.acceptance
    closures = len(scenario.edges) - (N_POSES - 1)
    return {
        "group": None,
        "density": density,
        "closures": closures,
        "dof": report.dof,
        "n_runs": int(n_runs),
        "converged": int(converged.sum()),
        "converged_fraction": fraction,
        "usable": usable,
        "mean_nees": report.mean,
        "ratio": report.mean / report.dof,
        "band_low": low / report.dof,
        "band_high": high / report.dof,
        "pvalue": report.pvalue,
        "verdict": report.verdict if usable else "convergence failure",
    }


def run(n_runs: int) -> None:
    rows = []
    for name, lie in GROUPS:
        for index, density in enumerate(DENSITIES):
            row = condition(lie, density, n_runs, seed=100 + index)
            row["group"] = name
            rows.append(row)
            print(f"  {name} density {density:<5} -> {row['verdict']}")

    # Multiplicity: every condition in the sweep is a test, so at alpha = 0.05
    # roughly one clean condition in twenty would be flagged by chance alone.
    usable = [r for r in rows if r["usable"]]
    flagged = benjamini_hochberg(
        np.array([r["pvalue"] for r in usable]), alpha=ALPHA
    )
    for row, reject in zip(usable, flagged):
        row["significant_after_fdr"] = bool(reject)
    for row in rows:
        row.setdefault("significant_after_fdr", False)

    write_results(rows, "e2_density")
    report_console(rows)


def report_console(rows) -> None:
    print("\nE2 - loop-closure density (H1)")
    print("=" * 88)
    print(
        f"  {'group':<7} {'density':>8} {'closures':>9} {'conv':>9} {'NEES/dof':>9} "
        f"{'band':>15} {'p':>9}  {'verdict':<14} FDR"
    )
    for row in rows:
        band = f"[{row['band_low']:.2f}, {row['band_high']:.2f}]"
        mark = "*" if row["significant_after_fdr"] else ""
        print(
            f"  {row['group']:<7} {row['density']:8.2f} {row['closures']:9d} "
            f"{row['converged']:4d}/{row['n_runs']:<4d} {row['ratio']:9.3f} "
            f"{band:>15} {row['pvalue']:9.2e}  {row['verdict']:<14} {mark}"
        )
    print("\n  * survives Benjamini-Hochberg across the sweep")

    print("\n  H1 predicted: overconfidence grows as density falls, monotonically.")
    for name, _ in GROUPS:
        series = [r for r in rows if r["group"] == name and r["usable"]]
        series.sort(key=lambda r: r["density"])
        ratios = [r["ratio"] for r in series]
        if len(ratios) < 3:
            print(f"  {name}: too few usable conditions to judge")
            continue
        falling = all(a >= b - 1e-9 for a, b in zip(ratios, ratios[1:]))
        sparsest, densest = ratios[0], ratios[-1]
        print(
            f"  {name}: sparse end {sparsest:.3f}, dense end {densest:.3f}, "
            f"monotone {'yes' if falling else 'NO'}"
        )
    print()


def build_figure():
    """F1: NEES against loop-closure density, with the acceptance band shaded."""
    rows = read_results("e2_density")
    fig, axes = figure(nrows=1, ncols=2, size=(10.0, 4.2))

    for index, (name, _) in enumerate(GROUPS):
        ax = axes[index]
        series = sorted(
            (r for r in rows if r["group"] == name), key=lambda r: r["density"]
        )
        density = np.array([r["density"] for r in series])
        ratio = np.array([r["ratio"] for r in series])
        low = np.array([r["band_low"] for r in series])
        high = np.array([r["band_high"] for r in series])

        ax.fill_between(
            density,
            low,
            high,
            color=BAND,
            alpha=0.9,
            linewidth=0,
            label="chi-squared acceptance band",
        )
        ax.axhline(
            1.0, color=INK_MUTED, linewidth=2.0, linestyle="--", label="calibrated"
        )
        ax.plot(
            density,
            ratio,
            marker="o",
            markersize=8,
            linewidth=2.0,
            color=OBSERVED,
            label="observed",
        )
        # linear, not log: the whole sweep spans 0.97 to 1.04, and a log
        # scale would compress the only thing this figure has to show
        label(
            ax,
            f"{name} - calibration against constraint density",
            "loop closures per pose",
            "mean NEES / dof",
        )

    # One legend for the figure, not one per panel: both panels carry the same
    # three series, and on a full-width line chart there is no in-panel corner
    # that some mark does not pass through.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncols=3,
        frameon=False,
        fontsize=8.5,
        labelcolor=INK_MUTED,
    )
    fig.suptitle(
        "E2: does the reported covariance degrade as loop closures thin out?",
        color=INK,
        fontsize=12,
        x=0.02,
        ha="left",
    )
    return fig


def figures() -> bool:
    fig = build_figure()
    try:
        path = save_figure(fig, "e2_loop_closure_density")
    except ImportError as exc:
        print(f"figure built but not written -- no usable renderer ({exc})")
        return False
    print(f"figure written to {path}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()

    if args.figures_only:
        report_console(read_results("e2_density"))
    else:
        run(args.runs)
    figures()
    return 0


if __name__ == "__main__":
    sys.exit(main())
