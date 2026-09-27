"""Robust back-ends under false loop closures, and why Huber's covariance is overconfident."""

from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace

import numpy as np
from scipy.stats import chi2

from experiments.common import (
    INK_MUTED,
    calibration,
    figure,
    label,
    layout_lines,
    mark_fdr,
    parallel,
    read_results,
    save_figures,
    title,
    write_results,
)
from posetrust import se2
from posetrust.graph import PoseGraph
from posetrust.optimizer import free_mask, gauss_newton
from posetrust.robust import (
    Cauchy,
    DynamicCovarianceScaling,
    Huber,
    chi2_threshold,
    graduated_non_convexity,
    irls,
    loop_closure_indices,
    squared_residuals,
)
from posetrust.simulate import NoiseModel, dead_reckon, make_scenario, monte_carlo, sample_graph
from posetrust.stats import tangent_error

LIE = se2
OUTLIER_RATES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
N_POSES = 20
LOOP_DENSITY = 1.0
NOISE = 0.05
TURN = 0.25
SEED = 300
# Loop-closure layouts; SEED is the original graph.
GRAPH_SEEDS = [300, 400, 500, 600, 700, 800, 900, 1000]
# Layout counts are judged at this outlier rate and above.
JUDGED_RATE = 0.10

THRESHOLD = chi2_threshold(LIE.DOF, 0.95)
DELTA = float(np.sqrt(THRESHOLD))
# Closures above this chi-squared quantile at Huber's solution are dropped before refitting.
FLAG = chi2.ppf(0.999, LIE.DOF)

METHOD_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def _with_kernel(kernel):
    def solve(graph, poses, anchor=0):
        return irls(graph, poses, kernel, anchor=anchor, robust_factors=loop_closure_indices(graph))

    return solve


def _gnc(graph, poses, anchor=0):
    return graduated_non_convexity(graph, poses, c=DELTA, anchor=anchor, robust_factors=loop_closure_indices(graph))


# (name, solver, is the baseline)
METHODS = [
    ("plain least squares", gauss_newton, True),
    ("Huber", _with_kernel(Huber(DELTA)), False),
    ("Cauchy", _with_kernel(Cauchy(DELTA)), False),
    ("DCS", _with_kernel(DynamicCovarianceScaling(THRESHOLD)), False),
    ("GNC", _gnc, False),
]
ROBUST = [name for name, _, baseline in METHODS if not baseline]


def scenario_for(rate: float, seed: int):
    return make_scenario(LIE, n_poses=N_POSES, loop_density=LOOP_DENSITY, outlier_rate=rate, seed=seed, turn=TURN)


# --- every back-end on every layout -------------------------------------------


def condition(method: int, rate: float, n_runs: int, seed: int) -> dict:
    """One method at one outlier rate on one layout."""
    name, solver, _ = METHODS[method]
    scenario = scenario_for(rate, seed)
    noise = NoiseModel(np.full(LIE.DOF, NOISE))
    result = monte_carlo(LIE, scenario, noise, n_runs=n_runs, seed=seed + 1, solver=solver, initialize="odometry")
    # RMS error over the free poses (the anchor is exact).
    free = [k for k in range(N_POSES) if k != result.anchor]
    per_run = np.mean(result.errors[result.converged][:, free] ** 2, axis=(1, 2))
    rms_error = float(np.sqrt(per_run.mean())) if per_run.size else float("nan")
    return {
        "graph": seed,
        "method": name,
        "rate": rate,
        "outliers": len(scenario.outliers),
        "rms_error": rms_error,
        **calibration(result),
    }


def overconfident(row) -> bool:
    return row["usable"] and row["significant_after_fdr"] and row["ratio"] > 1.0


def report_backends(rows) -> None:
    print("\nFalse loop closures - NEES/dof against the outlier rate (* overconfident after FDR, - did not converge)")
    header = "  " + " " * 7 + "".join(f"{r:>9g}" for r in OUTLIER_RATES)
    for name, _, _ in METHODS:
        print(f"\n  {name}" + (" (original graph only)" if name not in ROBUST else ""))
        print(header)
        for seed in GRAPH_SEEDS if name in ROBUST else [SEED]:
            by_rate = {r["rate"]: r for r in rows if r["graph"] == seed and r["method"] == name}
            cells = [
                f"{row['ratio']:8.2f}{'*' if overconfident(row) else ' '}" if row["usable"] else f"{'-':>9}"
                for row in (by_rate[rate] for rate in OUTLIER_RATES)
            ]
            print(f"  {seed:>5}  " + "".join(cells))

    def judged(seed, method):
        return [r for r in rows if r["graph"] == seed and r["method"] == method and r["rate"] >= JUDGED_RATE]

    n = len(GRAPH_SEEDS)
    huber = sum(all(overconfident(r) for r in judged(seed, "Huber")) for seed in GRAPH_SEEDS)
    dcs = sum(all(r["usable"] and not overconfident(r) for r in judged(seed, "DCS")) for seed in GRAPH_SEEDS)
    # Accuracy recovered = RMS error below twice the method's own error with no outliers.
    clean = {r["graph"]: r["rms_error"] for r in rows if r["method"] == "Huber" and r["rate"] == 0.0}
    corrupted = [r for r in rows if r["method"] == "Huber" and r["rate"] > 0 and r["usable"]]
    recovered = sum(r["rms_error"] < 2 * clean[r["graph"]] for r in corrupted)
    print(f"\n  From {JUDGED_RATE:g} outliers up: Huber overconfident at every rate on {huber}/{n} layouts,")
    print(f"  DCS at none on {dcs}/{n}. Huber keeps its error within twice its outlier-free error")
    print(f"  at {recovered}/{len(corrupted)} corrupted conditions.\n")


def build_backends_figure():
    """Calibration of each robust back-end, one line per loop-closure layout."""
    rows = read_results("false_loop_closures")
    fig, axes = figure(nrows=2, ncols=2, size=(10.0, 7.0), sharex=True, sharey=True)
    for ax, name, colour in zip(np.ravel(axes), ROBUST, METHOD_COLOURS):
        ax.axhline(1.0, color=INK_MUTED, linewidth=2.0, linestyle=":", label="calibrated")
        series = {}
        for seed in GRAPH_SEEDS:
            by_rate = {
                r["rate"]: r["ratio"] if r["usable"] else np.nan
                for r in rows
                if r["graph"] == seed and r["method"] == name
            }
            series[seed] = [by_rate.get(x, np.nan) for x in OUTLIER_RATES]
        layout_lines(ax, series, OUTLIER_RATES, SEED, colour=colour, sizes=(5, 3.5))
        ax.set_yscale("log")
        ax.set_xticks(OUTLIER_RATES, [f"{x:g}" for x in OUTLIER_RATES])
        label(ax, name, "outlier rate", "mean NEES / dof")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, loc="upper left")
    title(fig, f"Calibration of each robust back-end on {len(GRAPH_SEEDS)} loop-closure layouts")
    return fig


# --- why Huber is overconfident -----------------------------------------------


def errors_of(poses, truth) -> np.ndarray:
    return np.concatenate([tangent_error(LIE, poses[k], truth[k]) for k in range(1, N_POSES)])


def huber_condition(rate: float, seed: int, n_runs: int) -> list[dict]:
    """Huber at one outlier rate on one layout: its covariance, its bias, and a refit."""
    scenario = scenario_for(rate, seed)
    noise = NoiseModel(np.full(LIE.DOF, NOISE))
    free = free_mask(N_POSES, LIE.DOF, 0)
    errors, informations = [], []
    nees = {"huber": np.full(n_runs, np.nan), "refit": np.full(n_runs, np.nan)}

    # The same noise draws as the back-end sweep, so the Huber numbers match it.
    for r, child in enumerate(np.random.SeedSequence(seed + 1).spawn(n_runs)):
        graph = sample_graph(LIE, scenario, noise, np.random.default_rng(child))
        start = dead_reckon(LIE, graph, N_POSES)
        start[0] = scenario.truth[0]
        closures = loop_closure_indices(graph)
        result = irls(graph, start, Huber(DELTA), robust_factors=closures)
        if not result.converged:
            continue
        error, information = errors_of(result.poses, scenario.truth), result.information[np.ix_(free, free)]
        errors.append(error)
        informations.append(information)
        nees["huber"][r] = error @ information @ error

        s = squared_residuals(graph, result.poses)
        kept = PoseGraph(LIE)
        kept.poses = graph.poses
        kept.factors = [f for k, f in enumerate(graph.factors) if k not in closures or s[k] <= FLAG]
        refit = gauss_newton(kept, result.poses)
        if refit.converged:
            refit_error = errors_of(refit.poses, scenario.truth)
            nees["refit"][r] = refit_error @ refit.information[np.ix_(free, free)] @ refit_error

    dof = int(free.sum())
    # The mean error over runs is the bias; its NEES is what no covariance of the same estimate can absorb.
    mean_error = np.mean(errors, axis=0)
    bias = float(np.mean([mean_error @ information @ mean_error for information in informations])) / dof
    rows = []
    for name, values in nees.items():
        used = np.isfinite(values)
        stats = calibration(SimpleNamespace(converged=used, n_runs=n_runs, nees_full=values, free_dof=dof))
        rows.append({"graph": seed, "rate": rate, "estimate": name, "bias_ratio": bias, **stats})
    return rows


def select(rows, **match):
    return [r for r in rows if all(r[k] == v for k, v in match.items())]


def report_huber(rows) -> None:
    print("\nWhy Huber's covariance is overconfident")
    print(f"  {'rate':>6} {'Huber NEES/dof':>22} {'bias share':>11} {'refit NEES/dof':>22}")
    for rate in OUTLIER_RATES:
        huber, refit = select(rows, rate=rate, estimate="huber"), select(rows, rate=rate, estimate="refit")
        excess = sum(r["ratio"] - 1 for r in huber)
        share = f"{sum(r['bias_ratio'] for r in huber) / excess:.2f}" if rate > 0 else "-"
        cells = [
            f"{np.median([r['ratio'] for r in level]):8.2f} "
            f"[{min(r['ratio'] for r in level):5.2f}, {max(r['ratio'] for r in level):5.2f}]"
            for level in (huber, refit)
        ]
        print(f"  {rate:6.2f} {cells[0]:>22} {share:>11} {cells[1]:>22}")
    within = sum(
        all(0.9 <= r["ratio"] <= 1.1 for r in select(rows, graph=seed, estimate="refit") if r["rate"] >= JUDGED_RATE)
        for seed in GRAPH_SEEDS
    )
    print("\n  NEES/dof is the median over layouts [min, max]; the bias share is pooled over layouts.")
    print(f"  Refit within 10% of calibrated at every rate from {JUDGED_RATE:g}: {within}/{len(GRAPH_SEEDS)} layouts\n")


def build_huber_figure():
    rows = read_results("huber_bias")
    fig, (left, right) = figure(nrows=1, ncols=2, size=(10.0, 4.4))
    rates = np.array(OUTLIER_RATES)

    left.axhline(1.0, color=INK_MUTED, linewidth=1.5, linestyle=":")
    for name, colour, text in (("huber", INK_MUTED, "Huber's covariance"), ("refit", "#eb6834", "reject and refit")):
        per_rate = [[r["ratio"] for r in select(rows, rate=x, estimate=name)] for x in rates]
        left.fill_between(rates, [min(v) for v in per_rate], [max(v) for v in per_rate], color=colour, alpha=0.15, lw=0)
        left.plot(
            rates, [np.median(v) for v in per_rate], marker="o", markersize=5, linewidth=2.0, color=colour, label=text
        )
    left.set_yscale("log")
    left.set_xticks(rates, [f"{x:g}" for x in rates])
    label(left, "Huber on eight layouts, median and range", "outlier rate", "mean NEES / dof")
    left.legend(frameon=False, fontsize=8.5, labelcolor=INK_MUTED, loc="upper left")

    # One point per layout and rate: on the diagonal, all of the excess is bias.
    huber = [r for r in select(rows, estimate="huber") if r["rate"] > 0]
    excess, bias = [r["ratio"] - 1 for r in huber], [r["bias_ratio"] for r in huber]
    right.scatter(excess, bias, s=26, color="#2a78d6", zorder=3)
    ends = [0.8 * min(v for v in excess + bias if v > 0), 1.25 * max(excess + bias)]
    right.plot(ends, ends, color=INK_MUTED, linewidth=1.5, linestyle=":", label="all bias")
    right.set_xscale("log")
    right.set_yscale("log")
    label(right, "Bias against excess, per layout and rate", "excess NEES / dof", "bias term / dof")
    right.legend(frameon=False, fontsize=8.5, labelcolor=INK_MUTED, loc="upper left")

    title(fig, "Huber's excess is mostly bias, and removing the flagged closures fixes it")
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--figures-only", action="store_true")
    parser.add_argument("--huber-only", action="store_true", help="skip the (slow) sweep over every back-end")
    args = parser.parse_args()

    if args.figures_only:
        report_backends(read_results("false_loop_closures"))
        report_huber(read_results("huber_bias"))
    else:
        if not args.huber_only:
            tasks = [
                (m, rate, args.runs, s) for s in GRAPH_SEEDS for m in range(len(METHODS)) for rate in OUTLIER_RATES
            ]
            rows = []
            for row in parallel(condition, tasks):
                rows.append(row)
                print(f"  graph {row['graph']} {row['method']:<20} rate {row['rate']:<5} -> {row['verdict']}")
            mark_fdr(rows, by=("graph",))  # within each layout's sweep
            write_results(rows, "false_loop_closures")
            report_backends(rows)
        rows = [
            r
            for batch in parallel(huber_condition, [(x, s, args.runs) for s in GRAPH_SEEDS for x in OUTLIER_RATES])
            for r in batch
        ]
        write_results(rows, "huber_bias")
        report_huber(rows)
    save_figures((build_backends_figure, "false_loop_closures"), (build_huber_figure, "huber_bias"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
