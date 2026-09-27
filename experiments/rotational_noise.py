"""Calibration against rotational noise, in SE(2) and SE(3), on eight loop-closure layouts."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from experiments.common import (
    INK_MUTED,
    calibration,
    figure,
    label,
    layout_lines,
    legend_below,
    mark_fdr,
    parallel,
    read_results,
    save_figures,
    title,
    write_results,
)
from posetrust import se2, se3
from posetrust.optimizer import levenberg_marquardt
from posetrust.simulate import NoiseModel, make_scenario, monte_carlo
from posetrust.stats import ConsistencyReport

GROUPS = [("SE(2)", se2), ("SE(3)", se3)]
ROTATION_NOISE = [0.01, 0.03, 0.06, 0.10, 0.15, 0.22, 0.30, 0.45]
COVERAGE_LEVELS = (0.5, 0.75, 0.9, 0.95, 0.99)
TRANSLATION_NOISE = 0.02
N_POSES = 10
LOOP_DENSITY = 0.3
TURN = 0.25
SEED = 200
# Loop-closure layouts; SEED is the original graph.
GRAPH_SEEDS = [200, 300, 400, 500, 600, 700, 800, 900]

# Sequential blue ramp, light to dark with noise.
NOISE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]


def condition(group: int, rotation_sigma: float, n_runs: int, seed: int) -> dict:
    """One noise level on one layout: NEES and ellipsoid coverage."""
    name, lie = GROUPS[group]
    sigma = np.full(lie.DOF, TRANSLATION_NOISE)
    sigma[lie.TRANSLATION_DOF :] = rotation_sigma
    scenario = make_scenario(lie, n_poses=N_POSES, loop_density=LOOP_DENSITY, seed=seed, turn=TURN)
    # LM converges more often than GN at high noise, and non-converged runs are dropped.
    result = monte_carlo(lie, scenario, NoiseModel(sigma), n_runs=n_runs, seed=seed + 1, solver=levenberg_marquardt)
    nominal, empirical = ConsistencyReport(result.nees_full[result.converged], result.free_dof).coverage(
        COVERAGE_LEVELS
    )
    return {
        "graph": seed,
        "group": name,
        "rotation_sigma": rotation_sigma,
        **calibration(result),
        "coverage_nominal": list(nominal),
        "coverage_empirical": list(empirical),
    }


def run(n_runs: int) -> None:
    tasks = [(g, sigma, n_runs, seed) for seed in GRAPH_SEEDS for g in range(len(GROUPS)) for sigma in ROTATION_NOISE]
    rows = []
    for row in parallel(condition, tasks):
        rows.append(row)
        print(f"  graph {row['graph']} {row['group']} rot sigma {row['rotation_sigma']:<5} -> {row['verdict']}")
    mark_fdr(rows, by=("graph",))  # within each layout's sweep
    write_results(rows, "rotational_noise")
    report(rows)


def original(rows) -> list[dict]:
    return [r for r in rows if r["graph"] == SEED]


def overconfident(row) -> bool:
    return row["usable"] and row["significant_after_fdr"] and row["ratio"] > 1.0


def report(rows) -> None:
    print("\nRotational noise - NEES/dof (* overconfident after FDR)")
    print(
        f"  {'rot sd':>6}"
        + "".join(f"{name + ' original':>16}{'median [min, max] over layouts':>34}" for name, _ in GROUPS)
    )
    for sigma in ROTATION_NOISE:
        cells = []
        for name, _ in GROUPS:
            level = [r for r in rows if r["group"] == name and r["rotation_sigma"] == sigma]
            first = next(r for r in original(level))
            ratios = [r["ratio"] for r in level if r["usable"]]
            cells.append(f"{first['ratio']:15.2f}{'*' if overconfident(first) else ' '}")
            cells.append(f"{np.median(ratios):20.2f} [{min(ratios):5.2f}, {max(ratios):6.2f}]")
        print(f"  {sigma:6.2f}" + "".join(cells))
    for name, _ in GROUPS:
        breaks = [
            min(
                (r["rotation_sigma"] for r in rows if r["graph"] == seed and r["group"] == name and overconfident(r)),
                default=None,
            )
            for seed in GRAPH_SEEDS
        ]
        print(
            f"  {name} first overconfident sd per layout: "
            + ", ".join("none" if b is None else f"{b:g}" for b in breaks)
        )
    print()


def build_coverage_figure():
    """Coverage curves across rotational noise levels on the original graph."""
    rows = original(read_results("rotational_noise"))
    shown = [0.01, 0.06, 0.15, 0.30, 0.45]
    fig, axes = figure(nrows=1, ncols=2, size=(10.0, 4.4), sharey=True)

    for index, (name, _) in enumerate(GROUPS):
        ax = axes[index]
        ax.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=2.0, linestyle="--", label="ideal")
        # Rejected conditions are not measurements.
        usable = {r["rotation_sigma"]: r for r in rows if r["group"] == name and r["usable"]}
        for colour, sigma in zip(NOISE_RAMP, shown):
            if sigma in usable:
                ax.plot(
                    usable[sigma]["coverage_nominal"],
                    usable[sigma]["coverage_empirical"],
                    marker="o",
                    markersize=6,
                    linewidth=2.0,
                    color=colour,
                    label=f"sd {sigma:g}",
                )
        absent = [sigma for sigma in shown if sigma not in usable]
        if absent:
            ax.annotate(
                f"sd {', '.join(f'{s:g}' for s in absent)} omitted: did not converge",
                (0.5, 0.97),
                xycoords="axes fraction",
                ha="center",
                va="top",
                fontsize=8.5,
                color=INK_MUTED,
            )
        ax.set_xlim(0.45, 1.02)
        ax.set_ylim(0.0, 1.02)
        label(
            ax,
            f"{name} - ellipsoid coverage by noise level",
            "nominal level",
            "empirical coverage" if index == 0 else "",
        )

    legend_below(fig, *axes[0].get_legend_handles_labels(), ncols=6)
    title(fig, "Credible regions cover less than they claim as rotation grows")
    return fig


def build_graphs_figure():
    """NEES against rotational noise, one line per loop-closure layout."""
    rows = read_results("rotational_noise")
    fig, axes = figure(nrows=1, ncols=2, size=(10.0, 4.2), sharey=True)
    x = np.arange(len(ROTATION_NOISE))

    for index, (name, _) in enumerate(GROUPS):
        ax = axes[index]
        ax.axhline(1.0, color=INK_MUTED, linewidth=2.0, linestyle="--", label="calibrated")
        series = {}
        for seed in GRAPH_SEEDS:
            by_sigma = {
                r["rotation_sigma"]: r["ratio"] if r["usable"] else np.nan
                for r in rows
                if r["graph"] == seed and r["group"] == name
            }
            series[seed] = [by_sigma.get(s, np.nan) for s in ROTATION_NOISE]
        layout_lines(ax, series, x, SEED)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s:g}" for s in ROTATION_NOISE])
        ax.set_xlim(-0.5, len(x) - 0.5)
        label(
            ax,
            f"{name} - {len(GRAPH_SEEDS)} loop-closure layouts",
            "rotational noise (rad)",
            "mean NEES / dof" if index == 0 else "",
        )

    legend_below(fig, *axes[0].get_legend_handles_labels(), ncols=3)
    title(fig, "Where calibration breaks depends on the layout")
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()

    if args.figures_only:
        report(read_results("rotational_noise"))
    else:
        run(args.runs)
    save_figures((build_coverage_figure, "rotational_noise_coverage"), (build_graphs_figure, "rotational_noise"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
