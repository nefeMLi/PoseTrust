"""E2: calibration against loop-closure density."""

from __future__ import annotations

import argparse
import sys
from itertools import pairwise

import numpy as np

import se2
import se3
from experiment_utils import (
    BAND,
    INK,
    INK_MUTED,
    OBSERVED,
    calibration,
    figure,
    label,
    mark_fdr,
    read_results,
    save_figure,
    survivorship_warning,
    write_results,
)
from simulate import NoiseModel, make_scenario, monte_carlo

GROUPS = [("SE(2)", se2), ("SE(3)", se3)]
DENSITIES = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2]
N_POSES = 20
NOISE = 0.03
TURN = 0.25
SEED = 100


def condition(lie, density: float, n_runs: int) -> dict:
    """Run one density and summarise its calibration."""
    scenario = make_scenario(
        lie, n_poses=N_POSES, loop_density=density, seed=SEED, turn=TURN
    )
    result = monte_carlo(
        lie,
        scenario,
        NoiseModel(np.full(lie.DOF, NOISE)),
        n_runs=n_runs,
        seed=SEED + 1,
    )
    closures = len(scenario.edges) - (N_POSES - 1)
    return {"density": density, "closures": closures, **calibration(result)}


def run(n_runs: int) -> None:
    rows = []
    for name, lie in GROUPS:
        for density in DENSITIES:
            row = {"group": name, **condition(lie, density, n_runs)}
            rows.append(row)
            print(f"  {name} density {density:<5} -> {row['verdict']}")

    mark_fdr(rows)
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
    survivorship_warning(rows, lambda r: f"{r['group']} density {r['density']}")

    print("\n  H1 predicted: overconfidence grows as density falls, monotonically.")
    for name, _ in GROUPS:
        series = [r for r in rows if r["group"] == name and r["usable"]]
        series.sort(key=lambda r: r["density"])
        ratios = [r["ratio"] for r in series]
        if len(ratios) < 3:
            print(f"  {name}: too few usable conditions to judge")
            continue
        falling = all(a >= b - 1e-9 for a, b in pairwise(ratios))
        sparsest, densest = ratios[0], ratios[-1]
        print(
            f"  {name}: sparse end {sparsest:.3f}, dense end {densest:.3f}, "
            f"monotone {'yes' if falling else 'NO'}"
        )

    # Print the interval widths next to the spread they would need to resolve.
    widths = [r["ci_high"] - r["ci_low"] for r in rows if r["usable"]]
    spread = max(r["ratio"] for r in rows if r["usable"]) - min(
        r["ratio"] for r in rows if r["usable"]
    )
    covering = sum(r["ci_low"] <= 1.0 <= r["ci_high"] for r in rows if r["usable"])
    print(
        f"\n  Resolution: 95% intervals span {min(widths):.3f}-{max(widths):.3f}, "
        f"against a total spread of {spread:.3f} across the whole sweep."
    )
    print(
        f"  Differences below roughly {max(widths) / 2:.1%} of the state dimension "
        f"are not resolvable here."
    )
    print(f"  {covering} of {len(widths)} intervals cover perfect calibration.")
    print()


def build_figure():
    """NEES against loop-closure density."""
    rows = read_results("e2_density")
    # Shared y, so the two groups compare directly.
    fig, axes = figure(nrows=1, ncols=2, size=(10.0, 4.2), sharey=True)

    for index, (name, _) in enumerate(GROUPS):
        ax = axes[index]
        series = sorted(
            (r for r in rows if r["group"] == name), key=lambda r: r["density"]
        )
        ratio = np.array([r["ratio"] for r in series])
        ci_low = np.array([r["ci_low"] for r in series])
        ci_high = np.array([r["ci_high"] for r in series])
        # Densities are unevenly spaced, so plot them as ordered categories.
        x = np.arange(len(series))

        ax.axhspan(
            series[0]["band_low"],
            series[0]["band_high"],
            color=BAND,
            alpha=0.9,
            linewidth=0,
            label="chi-squared acceptance band",
        )
        ax.axhline(
            1.0, color=INK_MUTED, linewidth=2.0, linestyle="--", label="calibrated"
        )
        ax.errorbar(
            x,
            ratio,
            yerr=[ratio - ci_low, ci_high - ratio],
            fmt="o",
            markersize=8,
            linewidth=0,
            elinewidth=2.0,
            capsize=4,
            color=OBSERVED,
            ecolor=OBSERVED,
            label="observed, 95% interval",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r['density']:g}" for r in series])
        ax.set_xlim(-0.5, len(series) - 0.5)
        # linear: the whole sweep stays within a few percent of 1
        label(
            ax,
            f"{name} - calibration against constraint density",
            "loop closures per pose",
            "mean NEES / dof" if index == 0 else "",
        )

    # One legend for both panels; they show the same series.
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
        print(f"figure built but not written: no usable renderer ({exc})")
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
