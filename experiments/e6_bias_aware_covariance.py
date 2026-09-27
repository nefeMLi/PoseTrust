"""E6: report Huber's bias as well as its variance."""

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
    mark_fdr,
    parallel,
    read_results,
    save_figures,
    title,
    write_results,
)
from experiments.e4_perceptual_aliasing import (
    DELTA,
    JUDGED_RATE,
    LIE,
    LOOP_DENSITY,
    N_POSES,
    NOISE,
    OUTLIER_RATES,
    THRESHOLD,
    TURN,
)
from experiments.e4_perceptual_aliasing import GRAPH_SEEDS as DEV_SEEDS
from experiments.e5_robust_covariance import ANCHOR, solve
from posetrust.graph import PoseGraph
from posetrust.optimizer import free_mask, gauss_newton
from posetrust.robust import DynamicCovarianceScaling, Huber, loop_closure_indices
from posetrust.robust_covariance import pull_bias
from posetrust.simulate import NoiseModel, dead_reckon, make_scenario, sample_graph
from posetrust.stats import tangent_error

# (scenario seed, trajectory turn) per graph; the test trajectories are new.
SPLITS = {
    "dev": [(seed, TURN) for seed in DEV_SEEDS],
    "test": [(seed, turn) for turn in (0.10, 0.40) for seed in (1900, 2000, 2100, 2200)],
}
METHODS = [("Huber", Huber(DELTA)), ("DCS", DynamicCovarianceScaling(THRESHOLD))]
QUANTILES = (0.95, 0.99, 0.999)
# Fixed from the development run by the rule in HYPOTHESES.md; None until then.
CHOSEN_QUANTILE = 0.999
PRACTICAL = (0.9, 1.1)
# e6full (exploratory, added after the development run) uses the converged refit's
# shift in place of the one-step estimate.
ESTIMATORS = ["naive"] + [f"{kind}_{q}" for q in QUANTILES for kind in ("e6", "refit", "e6full")]
COLOURS = {"naive": INK_MUTED, "e6": "#2a78d6", "refit": "#eb6834"}


def errors_of(poses, truth) -> np.ndarray:
    return np.concatenate([tangent_error(LIE, poses[k], truth[k]) for k in range(N_POSES) if k != ANCHOR])


def without(graph: PoseGraph, drop) -> PoseGraph:
    kept = PoseGraph(graph.lie)
    kept.poses = graph.poses
    kept.factors = [f for k, f in enumerate(graph.factors) if k not in set(drop)]
    return kept


def condition(method: int, rate: float, seed: int, turn: float, n_runs: int) -> list[dict]:
    """One back-end at one outlier rate: naive, E6 and reject-and-refit on the same runs."""
    name, kernel = METHODS[method]
    scenario = make_scenario(LIE, n_poses=N_POSES, loop_density=LOOP_DENSITY, outlier_rate=rate, seed=seed, turn=turn)
    noise = NoiseModel(np.full(LIE.DOF, NOISE))
    free = free_mask(N_POSES, LIE.DOF, ANCHOR)
    nees = {c: np.full(n_runs, np.nan) for c in ESTIMATORS}
    sq_error = {c: np.full(n_runs, np.nan) for c in ESTIMATORS}

    for r, child in enumerate(np.random.SeedSequence(seed + 1).spawn(n_runs)):
        graph = sample_graph(LIE, scenario, noise, np.random.default_rng(child))
        start = dead_reckon(LIE, graph, N_POSES)
        start[ANCHOR] = scenario.truth[ANCHOR]
        result = solve(name, kernel, graph, start)
        if not result.converged:
            continue
        error = errors_of(result.poses, scenario.truth)
        information = result.information[np.ix_(free, free)]
        nees["naive"][r] = error @ information @ error
        sq_error["naive"][r] = np.mean(error**2)
        closures = loop_closure_indices(graph)
        for q in QUANTILES:
            shift, pulled = pull_bias(graph, result.poses, kernel, closures, chi2.ppf(q, LIE.DOF), anchor=ANCHOR)
            covariance = np.linalg.inv(information) + np.outer(shift, shift)
            nees[f"e6_{q}"][r] = error @ np.linalg.solve(covariance, error)
            sq_error[f"e6_{q}"][r] = sq_error["naive"][r]

            refit = gauss_newton(without(graph, pulled), result.poses, anchor=ANCHOR)
            if refit.converged:
                refit_error = errors_of(refit.poses, scenario.truth)
                refit_info = refit.information[np.ix_(free, free)]
                nees[f"refit_{q}"][r] = refit_error @ refit_info @ refit_error
                sq_error[f"refit_{q}"][r] = np.mean(refit_error**2)
                full = errors_of(refit.poses, result.poses)
                full_cov = np.linalg.inv(information) + np.outer(full, full)
                nees[f"e6full_{q}"][r] = error @ np.linalg.solve(full_cov, error)
                sq_error[f"e6full_{q}"][r] = sq_error["naive"][r]

    dof = int(free.sum())
    rows = []
    for c in ESTIMATORS:
        used = np.isfinite(nees[c])
        stats = calibration(SimpleNamespace(converged=used, n_runs=n_runs, nees_full=nees[c], free_dof=dof))
        rms = float(np.sqrt(np.mean(sq_error[c][used]))) if used.any() else float("nan")
        rows.append(
            {"graph": seed, "turn": turn, "method": name, "rate": rate, "estimator": c, "rms_error": rms, **stats}
        )
    return rows


def run(split: str, n_runs: int) -> None:
    tasks = [
        (m, rate, seed, turn, n_runs)
        for seed, turn in SPLITS[split]
        for m in range(len(METHODS))
        for rate in OUTLIER_RATES
    ]
    rows = []
    for batch in parallel(condition, tasks):
        rows.extend(batch)
        first = batch[0]
        shown = {r["estimator"]: r["ratio"] for r in batch}
        print(
            f"  graph {first['graph']} turn {first['turn']} {first['method']:<6} "
            f"rate {first['rate']:<5} naive {shown['naive']:.2f} "
            + " ".join(f"e6/refit@{q} {shown[f'e6_{q}']:.2f}/{shown[f'refit_{q}']:.2f}" for q in QUANTILES),
            flush=True,
        )
    mark_fdr(rows, by=("graph", "turn", "estimator"))
    write_results(rows, f"e6_{split}")
    report(rows)


def select(rows, **match):
    return [r for r in rows if all(r[k] == v for k, v in match.items())]


def graphs(rows):
    return sorted({(r["graph"], r["turn"]) for r in rows})


def within(row) -> bool:
    return row["usable"] and PRACTICAL[0] <= row["ratio"] <= PRACTICAL[1]


def calibrated(row) -> bool:
    return row["usable"] and not row["significant_after_fdr"]


def chosen_quantile(rows) -> float:
    """CHOSEN_QUANTILE, or the development rule: Huber E6 closest to 1 on average."""
    if CHOSEN_QUANTILE is not None:
        return CHOSEN_QUANTILE

    def distance(q):
        judged = [r for r in select(rows, method="Huber", estimator=f"e6_{q}") if r["rate"] >= JUDGED_RATE]
        return np.mean([abs(r["ratio"] - 1.0) for r in judged])

    return min(QUANTILES, key=distance)


def every_rate(rows, graph, method, estimator, test, judged_only=True) -> bool:
    seed, turn = graph
    level = [
        r
        for r in select(rows, graph=seed, turn=turn, method=method, estimator=estimator)
        if r["rate"] >= JUDGED_RATE or not judged_only
    ]
    return all(test(r) for r in level)


def report(rows) -> None:
    gs = graphs(rows)
    n = len(gs)
    q = chosen_quantile(rows)
    print("\nE6 - Huber's covariance with its estimated pull added")
    print("=" * 84)

    print(f"\n  Threshold rule: Huber E6, mean |NEES/dof - 1| at rates >= {JUDGED_RATE:g}")
    for quantile in QUANTILES:
        judged = [r for r in select(rows, method="Huber", estimator=f"e6_{quantile}") if r["rate"] >= JUDGED_RATE]
        mark = "  <- used" if quantile == q else ""
        print(f"  q = {quantile:<6} {np.mean([abs(r['ratio'] - 1) for r in judged]):.3f}{mark}")

    for method in ("Huber", "DCS"):
        print(f"\n  {method}: NEES/dof, median over graphs [min, max], q = {q}")
        shown = ["naive", f"e6_{q}", f"refit_{q}", f"e6full_{q}"]
        print(f"  {'rate':>6} " + "".join(f"{c:>22}" for c in shown))
        for rate in OUTLIER_RATES:
            cells = []
            for c in shown:
                ratios = np.array([r["ratio"] for r in select(rows, method=method, rate=rate, estimator=c)])
                cells.append(f"{np.nanmedian(ratios):8.2f} [{np.nanmin(ratios):5.2f},{np.nanmax(ratios):5.2f}]")
            print(f"  {rate:6.2f} " + "".join(f"{x:>22}" for x in cells))

    e6, refit = f"e6_{q}", f"refit_{q}"
    low, high = PRACTICAL
    print(f"\n  H7a: Huber under E6 within [{low}, {high}] at every rate >= {JUDGED_RATE:g} (need 6)")
    print(
        f"  practical {sum(every_rate(rows, g, 'Huber', e6, within) for g in gs)}/{n}; "
        f"strict {sum(every_rate(rows, g, 'Huber', e6, calibrated) for g in gs)}/{n}"
    )
    print(f"  naive, for comparison: practical {sum(every_rate(rows, g, 'Huber', 'naive', within) for g in gs)}/{n}")

    at_zero = [r for r in select(rows, method="Huber", rate=0.0, estimator=e6)]
    print(f"\n  H7b: Huber under E6 at 0% outliers within [{low}, {high}]: {sum(within(r) for r in at_zero)}/{n}")

    def unchanged(graph):
        seed, turn = graph
        pairs = zip(
            sorted(select(rows, graph=seed, turn=turn, method="DCS", estimator="naive"), key=lambda r: r["rate"]),
            sorted(select(rows, graph=seed, turn=turn, method="DCS", estimator=e6), key=lambda r: r["rate"]),
        )
        return all(abs(a["ratio"] - b["ratio"]) < 0.05 for a, b in pairs)

    print(f"\n  H7c: DCS changed by less than 0.05 at every rate: {sum(unchanged(g) for g in gs)}/{n}")
    print(
        f"\n  Exploratory, E6 with the converged refit's shift: within [{low}, {high}] at every rate "
        f">= {JUDGED_RATE:g}: {sum(every_rate(rows, g, 'Huber', f'e6full_{q}', within) for g in gs)}/{n}"
    )
    print(
        f"\n  H7d: reject-and-refit within [{low}, {high}] at every rate >= {JUDGED_RATE:g}: "
        f"{sum(every_rate(rows, g, 'Huber', refit, within) for g in gs)}/{n}"
    )
    print()


def build_figure(split: str):
    """Huber's NEES under the naive covariance, E6, and reject-and-refit."""
    rows = read_results(f"e6_{split}")
    q = chosen_quantile(rows)
    fig, ax = figure(size=(7.5, 4.6))
    rates = np.array(OUTLIER_RATES)
    ax.axhspan(*PRACTICAL, color="#d9d8d4", alpha=0.6, linewidth=0, label="within 10%")
    ax.axhline(1.0, color=INK_MUTED, linewidth=1.5, linestyle=":")
    for kind, estimator, name in (
        ("naive", "naive", "naive covariance"),
        ("e6", f"e6_{q}", "E6: naive plus estimated pull"),
        ("refit", f"refit_{q}", "reject and refit"),
    ):
        per_rate = [
            np.array([r["ratio"] for r in select(rows, method="Huber", rate=x, estimator=estimator)]) for x in rates
        ]
        median = np.array([np.nanmedian(v) for v in per_rate])
        ax.fill_between(
            rates,
            [np.nanmin(v) for v in per_rate],
            [np.nanmax(v) for v in per_rate],
            color=COLOURS[kind],
            alpha=0.15,
            linewidth=0,
        )
        ax.plot(rates, median, marker="o", markersize=5, linewidth=2.0, color=COLOURS[kind], label=name)
    ax.set_yscale("log")
    ax.set_xticks(rates, [f"{x:g}" for x in rates])
    label(ax, f"Huber on the {split} graphs, median and range", "outlier rate", "mean NEES / dof")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_MUTED, loc="upper left")
    title(fig, "E6: Huber's covariance with its estimated pull added")
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SPLITS, default="dev")
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()

    if args.figures_only:
        report(read_results(f"e6_{args.split}"))
    else:
        run(args.split, args.runs)
    save_figures((lambda: build_figure(args.split), f"e6_{args.split}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
