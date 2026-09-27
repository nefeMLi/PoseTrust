"""Robust back-ends under false loop closures, and why Huber's covariance is overconfident."""

import argparse
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

from experiments.common import (
    calibration,
    layout_lines,
    mark_fdr,
    overconfident,
    parallel,
    read_results,
    save_figure,
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
N_POSES, LOOP_DENSITY, NOISE, TURN = 20, 1.0, 0.05, 0.25
SEED = 300  # the original graph
GRAPH_SEEDS = [300, 400, 500, 600, 700, 800, 900, 1000]
THRESHOLD = chi2_threshold(LIE.DOF, 0.95)
DELTA = float(np.sqrt(THRESHOLD))
# Closures above this chi-squared quantile at Huber's solution are dropped before refitting.
FLAG = chi2.ppf(0.999, LIE.DOF)


def robust(kernel):
    return lambda graph, poses, anchor=0: irls(
        graph, poses, kernel, anchor=anchor, robust_factors=loop_closure_indices(graph)
    )


def gnc(graph, poses, anchor=0):
    return graduated_non_convexity(graph, poses, c=DELTA, anchor=anchor, robust_factors=loop_closure_indices(graph))


METHODS = {
    "plain least squares": gauss_newton,
    "Huber": robust(Huber(DELTA)),
    "Cauchy": robust(Cauchy(DELTA)),
    "DCS": robust(DynamicCovarianceScaling(THRESHOLD)),
    "GNC": gnc,
}
COLOURS = {"Huber": "tab:blue", "Cauchy": "tab:orange", "DCS": "tab:green", "GNC": "tab:red"}


def scenario_for(rate: float, seed: int):
    return make_scenario(LIE, n_poses=N_POSES, loop_density=LOOP_DENSITY, outlier_rate=rate, seed=seed, turn=TURN)


def condition(method: str, rate: float, n_runs: int, seed: int) -> dict:
    """One back-end at one outlier rate on one layout."""
    noise = NoiseModel(np.full(LIE.DOF, NOISE))
    result = monte_carlo(
        LIE,
        scenario_for(rate, seed),
        noise,
        n_runs=n_runs,
        seed=seed + 1,
        solver=METHODS[method],
        initialize="odometry",
    )
    errors = result.errors[result.converged][:, 1:]  # the first pose is anchored
    rms = float(np.sqrt(np.mean(errors**2))) if errors.size else float("nan")
    return {"graph": seed, "method": method, "rate": rate, "rms_error": rms, **calibration(result)}


def errors_of(poses, truth) -> np.ndarray:
    return np.concatenate([tangent_error(LIE, poses[k], truth[k]) for k in range(1, N_POSES)])


def huber_condition(rate: float, seed: int, n_runs: int) -> list[dict]:
    """Huber on one layout: its NEES, the part of it that is bias, and the NEES after a refit."""
    scenario, noise = scenario_for(rate, seed), NoiseModel(np.full(LIE.DOF, NOISE))
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

        # Refit: drop the closures still far off at Huber's solution and solve again.
        s = squared_residuals(graph, result.poses)
        kept = PoseGraph(LIE)
        kept.poses = graph.poses
        kept.factors = [f for k, f in enumerate(graph.factors) if k not in closures or s[k] <= FLAG]
        refit = gauss_newton(kept, result.poses)
        if refit.converged:
            refit_error = errors_of(refit.poses, scenario.truth)
            nees["refit"][r] = refit_error @ refit.information[np.ix_(free, free)] @ refit_error

    dof = int(free.sum())
    # The mean error over runs is the bias; its share of the NEES is what no covariance can absorb.
    mean_error = np.mean(errors, axis=0)
    bias = float(np.mean([mean_error @ info @ mean_error for info in informations])) / dof
    return [
        {
            "graph": seed,
            "rate": rate,
            "estimate": name,
            "bias_ratio": bias,
            **calibration(SimpleNamespace(converged=np.isfinite(v), n_runs=n_runs, nees_full=v, free_dof=dof)),
        }
        for name, v in nees.items()
    ]


def report(rows, huber_rows) -> None:
    judged = [r for r in rows if r["rate"] >= 0.10]
    n = len(GRAPH_SEEDS)
    huber = sum(
        all(overconfident(r) for r in judged if r["graph"] == g and r["method"] == "Huber") for g in GRAPH_SEEDS
    )
    dcs = sum(
        all(r["usable"] and not overconfident(r) for r in judged if r["graph"] == g and r["method"] == "DCS")
        for g in GRAPH_SEEDS
    )
    print("\nFalse loop closures")
    print(f"  from 10% outliers: Huber overconfident at every rate on {huber}/{n} layouts, DCS at none on {dcs}/{n}")
    print(f"  Huber NEES/dof up to {max(r['ratio'] for r in rows if r['method'] == 'Huber'):.1f}")
    for method in ("Huber", "plain least squares"):
        at30 = [r["rms_error"] for r in rows if r["method"] == method and r["rate"] == 0.30 and r["usable"]]
        print(f"  {method} trajectory error at 30%: {min(at30):.2f}-{max(at30):.2f} ({len(at30)} layouts converged)")
    for method in ("Cauchy", "GNC"):
        ratios = [r["ratio"] for r in rows if r["method"] == method]
        print(
            f"  {method} NEES/dof: median {np.median(ratios):.2f}, overconfident in {sum(map(overconfident, [r for r in rows if r['method'] == method]))} conditions"
        )

    print("\n  Huber   rate   NEES/dof  bias share   refit NEES/dof")
    for rate in OUTLIER_RATES[1:]:
        level = [r for r in huber_rows if r["rate"] == rate]
        own, refit = [r for r in level if r["estimate"] == "huber"], [r for r in level if r["estimate"] == "refit"]
        share = sum(r["bias_ratio"] for r in own) / sum(r["ratio"] - 1 for r in own)
        print(
            f"          {rate:4.2f}  {np.median([r['ratio'] for r in own]):9.2f}  {share:10.2f}  "
            f"{np.median([r['ratio'] for r in refit]):15.2f}"
        )
    within = sum(
        all(
            0.9 <= r["ratio"] <= 1.1
            for r in huber_rows
            if r["graph"] == g and r["estimate"] == "refit" and r["rate"] >= 0.10
        )
        for g in GRAPH_SEEDS
    )
    print(f"  refit within 10% of calibrated at every rate from 10%: {within}/{n} layouts\n")


def plot(rows, huber_rows) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True, layout="constrained")
    for ax, method in zip(axes.flat, COLOURS):
        series = {}
        for seed in GRAPH_SEEDS:
            mine = {r["rate"]: r for r in rows if r["graph"] == seed and r["method"] == method}
            series[seed] = [mine[x]["ratio"] if mine[x]["usable"] else np.nan for x in OUTLIER_RATES]
        layout_lines(ax, series, OUTLIER_RATES, SEED, COLOURS[method])
        ax.set(title=method, xlabel="outlier rate", ylabel="mean NEES / dof")
        ax.legend(frameon=False, loc="upper left")
    fig.suptitle("Calibration of each robust back-end on eight loop-closure layouts")
    save_figure(fig, "false_loop_closures")

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
    for name, colour, text in (("huber", "grey", "Huber's covariance"), ("refit", "tab:orange", "reject and refit")):
        per_rate = [[r["ratio"] for r in huber_rows if r["rate"] == x and r["estimate"] == name] for x in OUTLIER_RATES]
        left.fill_between(
            OUTLIER_RATES, [min(v) for v in per_rate], [max(v) for v in per_rate], color=colour, alpha=0.2
        )
        left.plot(OUTLIER_RATES, [np.median(v) for v in per_rate], "o-", color=colour, label=text)
    left.axhline(1.0, color="black", lw=1, ls=":")
    left.set(
        yscale="log", title="Huber on eight layouts, median and range", xlabel="outlier rate", ylabel="mean NEES / dof"
    )
    left.legend(frameon=False)
    # One point per layout and rate: on the diagonal, all of the excess is bias.
    own = [r for r in huber_rows if r["estimate"] == "huber" and r["rate"] > 0]
    excess, bias = [r["ratio"] - 1 for r in own], [r["bias_ratio"] for r in own]
    right.scatter(excess, bias, s=20)
    ends = [0.8 * min(excess + bias), 1.25 * max(excess + bias)]
    right.plot(ends, ends, color="black", lw=1, ls=":", label="all bias")
    right.set(
        xscale="log",
        yscale="log",
        title="Bias against excess, per layout and rate",
        xlabel="excess NEES / dof",
        ylabel="bias term / dof",
    )
    right.legend(frameon=False)
    fig.suptitle("Huber's excess is mostly bias, and removing the flagged closures fixes it")
    save_figure(fig, "huber_bias")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--figures-only", action="store_true")
    parser.add_argument("--huber-only", action="store_true", help="skip the slow sweep over every back-end")
    args = parser.parse_args()
    if args.figures_only or args.huber_only:
        rows = read_results("false_loop_closures")
    else:
        tasks = [(m, x, args.runs, seed) for seed in GRAPH_SEEDS for m in METHODS for x in OUTLIER_RATES]
        rows = list(parallel(condition, tasks))
        mark_fdr(rows, by=("graph",))  # within each layout's sweep
        write_results(rows, "false_loop_closures")
    if args.figures_only:
        huber_rows = read_results("huber_bias")
    else:
        tasks = [(x, seed, args.runs) for seed in GRAPH_SEEDS for x in OUTLIER_RATES]
        huber_rows = [row for batch in parallel(huber_condition, tasks) for row in batch]
        write_results(huber_rows, "huber_bias")
    report(rows, huber_rows)
    plot(rows, huber_rows)
