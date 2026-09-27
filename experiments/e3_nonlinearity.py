"""E3: calibration against rotational noise, in SE(2) and SE(3), on eight layouts."""

from __future__ import annotations

import argparse
import sys
from itertools import pairwise

import numpy as np

from experiments.common import (
    INK_MUTED,
    OBSERVED,
    SURVIVORSHIP_FRACTION,
    calibration,
    figure,
    label,
    layout_lines,
    legend_below,
    mark_fdr,
    parallel,
    read_results,
    save_figures,
    survivorship_warning,
    title,
    write_results,
)
from posetrust import se2, se3
from posetrust.optimizer import levenberg_marquardt
from posetrust.simulate import NoiseModel, make_scenario, monte_carlo
from posetrust.stats import ConsistencyReport, nees_by_dof

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
    """One noise level on one layout: NEES, coverage and the per-DOF split."""
    name, lie = GROUPS[group]
    sigma = np.full(lie.DOF, TRANSLATION_NOISE)
    sigma[lie.TRANSLATION_DOF :] = rotation_sigma
    scenario = make_scenario(lie, n_poses=N_POSES, loop_density=LOOP_DENSITY, seed=seed, turn=TURN)
    # LM converges more often than GN at high noise. Non-converged runs are
    # dropped, which biases towards calibration, so fewer failures is better.
    result = monte_carlo(lie, scenario, NoiseModel(sigma), n_runs=n_runs, seed=seed + 1, solver=levenberg_marquardt)
    converged = result.converged
    summary = calibration(result)

    # RMS rotation error in radians, comparable across SE(2) and SE(3).
    free = [k for k in range(N_POSES) if k != result.anchor]
    rotation_error = result.errors[converged][:, free, lie.TRANSLATION_DOF :]
    rms_rotation = float(np.sqrt(np.mean(np.sum(rotation_error**2, axis=-1))))

    # Translation vs rotation, pooled across poses (descriptive only).
    flat_errors = result.errors[converged][:, free].reshape(-1, lie.DOF)
    flat_marginals = result.marginals[converged][:, free].reshape(-1, lie.DOF, lie.DOF)
    split = nees_by_dof(lie, flat_errors, flat_marginals)
    translation_dof = lie.TRANSLATION_DOF

    nominal, empirical = ConsistencyReport(result.nees_full[converged], result.free_dof).coverage(COVERAGE_LEVELS)
    return {
        "graph": seed,
        "group": name,
        "rotation_sigma": rotation_sigma,
        **summary,
        "rms_rotation_error": rms_rotation,
        "translation_ratio": float(split["translation"].mean() / translation_dof),
        "rotation_ratio": float(split["rotation"].mean() / (lie.DOF - translation_dof)),
        "coverage_nominal": list(nominal),
        "coverage_empirical": list(empirical),
    }


def run(n_runs: int) -> None:
    tasks = [(g, sigma, n_runs, seed) for seed in GRAPH_SEEDS for g in range(len(GROUPS)) for sigma in ROTATION_NOISE]
    rows = []
    for row in parallel(condition, tasks):
        rows.append(row)
        print(
            f"  graph {row['graph']} {row['group']} rot sigma {row['rotation_sigma']:<5} -> {row['verdict']}",
            flush=True,
        )
    # Correct within each layout's sweep.
    mark_fdr(rows, by=("graph",))
    write_results(rows, "e3_nonlinearity")
    report(rows)


def original(rows) -> list[dict]:
    return [r for r in rows if r["graph"] == SEED]


def usable_series(rows, name: str) -> list[dict]:
    return sorted((r for r in rows if r["group"] == name and r["usable"]), key=lambda r: r["rotation_sigma"])


def overconfident(row) -> bool:
    return row["usable"] and row["significant_after_fdr"] and row["ratio"] > 1.0


def report(rows) -> None:
    report_original(original(rows))
    report_layouts(rows)


def report_original(rows) -> None:
    print("\nE3 - rotational non-linearity (H2)")
    print("=" * 100)
    print(
        f"  {'group':<7} {'rot sd':>7} {'conv':>9} {'NEES/dof':>9} {'95% interval':>16} "
        f"{'p':>9}  {'verdict':<14} FDR  {'trans':>6} {'rot':>6}"
    )
    for row in rows:
        ci = f"[{row['ci_low']:.2f}, {row['ci_high']:.2f}]"
        mark = "*" if row["significant_after_fdr"] else ""
        print(
            f"  {row['group']:<7} {row['rotation_sigma']:7.2f} "
            f"{row['converged']:4d}/{row['n_runs']:<4d} {row['ratio']:9.3f} {ci:>16} "
            f"{row['pvalue']:9.2e}  {row['verdict']:<14} {mark:<4} "
            f"{row['translation_ratio']:6.2f} {row['rotation_ratio']:6.2f}"
        )
    print("\n  * survives Benjamini-Hochberg across the sweep")
    print("  trans / rot are the per-degree-of-freedom split, pooled and descriptive")
    survivorship_warning(rows, lambda r: f"{r['group']} sd {r['rotation_sigma']}")

    print("\n  H2 predicted: overconfidence grows with rotational noise, abruptly.")
    for name, _ in GROUPS:
        series = usable_series(rows, name)
        ratios = [r["ratio"] for r in series]
        if len(ratios) < 2:
            print(f"  {name}: too few usable conditions to judge")
            continue
        rising = all(a <= b + 1e-9 for a, b in pairwise(ratios))
        biggest = max((b / a, series[i + 1]["rotation_sigma"]) for i, (a, b) in enumerate(pairwise(ratios)))
        print(
            f"  {name}: {ratios[0]:.2f} at sd {series[0]['rotation_sigma']}, "
            f"{ratios[-1]:.1f} at sd {series[-1]['rotation_sigma']}; "
            f"monotone {'yes' if rising else 'no'}; "
            f"largest single step x{biggest[0]:.1f} at sd {biggest[1]}"
        )

    print("\n  EXPLORATORY (H3 withdrawn): rotation error where calibration is lost.")
    for name, _ in GROUPS:
        broke = [r for r in usable_series(rows, name) if overconfident(r)]
        if broke:
            print(
                f"  {name}: first overconfident at sd {broke[0]['rotation_sigma']}, "
                f"RMS rotation error {broke[0]['rms_rotation_error']:.3f} rad"
            )
        else:
            print(f"  {name}: never loses calibration across this sweep")
    print()


def report_layouts(rows) -> None:
    print("\nE3 across graphs (exploratory)")
    print("=" * 72)
    for name, _ in GROUPS:
        print(f"\n  {name}: NEES/dof across layouts")
        print(f"  {'rot sd':>7} {'median':>8} {'min':>8} {'max':>8} {'overconf.':>10}")
        for sigma in ROTATION_NOISE:
            level = [r for r in rows if r["group"] == name and r["rotation_sigma"] == sigma and r["usable"]]
            if not level:
                print(f"  {sigma:7.2f}  no usable layouts")
                continue
            ratios = np.array([r["ratio"] for r in level])
            over = sum(r["significant_after_fdr"] and r["ratio"] > 1.0 for r in level)
            print(
                f"  {sigma:7.2f} {np.median(ratios):8.2f} {ratios.min():8.2f} "
                f"{ratios.max():8.2f} {over:>5d}/{len(level):<4d}"
            )
        breaks = [
            min(
                (r["rotation_sigma"] for r in rows if r["graph"] == seed and r["group"] == name and overconfident(r)),
                default=None,
            )
            for seed in GRAPH_SEEDS
        ]
        shown = ", ".join("none" if b is None else f"{b:g}" for b in breaks)
        print(f"  first overconfident sd per layout: {shown}")
    print()


def build_figure():
    """NEES against rotational noise on the original graph."""
    rows = original(read_results("e3_nonlinearity"))
    fig, axes = figure(nrows=1, ncols=2, size=(10.0, 4.2), sharey=True)

    for index, (name, _) in enumerate(GROUPS):
        ax = axes[index]
        series = sorted((r for r in rows if r["group"] == name), key=lambda r: r["rotation_sigma"])
        x = np.arange(len(series))
        ratio = np.array([r["ratio"] for r in series])
        ci_low = np.array([r["ci_low"] for r in series])
        ci_high = np.array([r["ci_high"] for r in series])

        # No acceptance band: it would be invisible on a two-decade log axis.
        ax.axhline(1.0, color=INK_MUTED, linewidth=2.0, linestyle="--", label="calibrated")

        # Skip rejected conditions; draw ones that lost runs hollow (lower bounds).
        usable = np.array([r["usable"] for r in series])
        complete = np.array([r["converged_fraction"] >= SURVIVORSHIP_FRACTION for r in series])
        for mask, fill, tag in (
            (usable & complete, OBSERVED, "observed, 95% interval"),
            (usable & ~complete, "none", "lower bound, runs dropped"),
        ):
            if mask.any():
                ax.errorbar(
                    x[mask],
                    ratio[mask],
                    yerr=[(ratio - ci_low)[mask], (ci_high - ratio)[mask]],
                    fmt="o",
                    markersize=8,
                    markerfacecolor=fill,
                    markeredgecolor=OBSERVED,
                    markeredgewidth=2.0,
                    linewidth=0,
                    elinewidth=2.0,
                    capsize=4,
                    ecolor=OBSERVED,
                    label=tag,
                )
        for position, row in zip(x[~usable], np.array(series)[~usable]):
            ax.annotate(
                f"did not\nconverge\n({row['converged']}/{row['n_runs']})",
                (position, 1.0),
                textcoords="offset points",
                xytext=(0, 14),
                ha="center",
                va="bottom",
                fontsize=8,
                color=INK_MUTED,
            )
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r['rotation_sigma']:g}" for r in series])
        ax.set_xlim(-0.5, len(series) - 0.5)
        label(
            ax,
            f"{name} - calibration against rotational noise",
            "rotational noise (rad)",
            "mean NEES / dof" if index == 0 else "",
        )

    legend_below(fig, *axes[0].get_legend_handles_labels(), ncols=3)
    title(fig, "E3: overconfidence against rotational noise")
    return fig


def build_coverage_figure():
    """Coverage curves across rotational noise levels on the original graph."""
    rows = original(read_results("e3_nonlinearity"))
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
    title(fig, "E3: credible regions cover less than they claim as rotation grows")
    return fig


def build_graphs_figure():
    """NEES against rotational noise, one line per loop-closure layout."""
    rows = read_results("e3_nonlinearity")
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
    title(fig, "E3 across graphs: where calibration breaks depends on the layout")
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()

    if args.figures_only:
        report(read_results("e3_nonlinearity"))
    else:
        run(args.runs)
    save_figures(
        (build_figure, "e3_nonlinearity"), (build_coverage_figure, "e3_coverage"), (build_graphs_figure, "e3_graphs")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
