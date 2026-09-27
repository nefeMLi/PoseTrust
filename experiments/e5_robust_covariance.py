"""E5: which covariance a robust back-end should report."""

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
from posetrust.optimizer import gauss_newton
from posetrust.robust import (
    Cauchy,
    DynamicCovarianceScaling,
    GemanMcClure,
    Huber,
    Trivial,
    graduated_non_convexity,
    irls,
    loop_closure_indices,
    squared_residuals,
)
from posetrust.robust_covariance import NAMES, robust_covariances
from posetrust.simulate import NoiseModel, dead_reckon, make_scenario, sample_graph
from posetrust.stats import tangent_error

TEST_SEEDS = [1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800]
SPLITS = {"dev": DEV_SEEDS, "test": TEST_SEEDS}
ANCHOR = 0

# Plain least squares only enters the gate, at 0% outliers.
METHODS = [
    ("plain least squares", Trivial()),
    ("Huber", Huber(DELTA)),
    ("Cauchy", Cauchy(DELTA)),
    ("DCS", DynamicCovarianceScaling(THRESHOLD)),
    ("GNC", GemanMcClure(DELTA, 1.0)),
]
# The figure shows the pre-registered covariances; the expected sandwich is
# exploratory and stays in the tables.
COVARIANCE_COLOURS = {"naive": INK_MUTED, "sandwich": "#2a78d6", "inlier": "#eb6834"}


def solve(name: str, kernel, graph, start):
    closures = loop_closure_indices(graph)
    if name == "plain least squares":
        return gauss_newton(graph, start, anchor=ANCHOR)
    if name == "GNC":
        return graduated_non_convexity(
            graph, start, c=DELTA, anchor=ANCHOR, robust_factors=closures
        )
    return irls(graph, start, kernel, anchor=ANCHOR, robust_factors=closures)


def condition(method: int, rate: float, seed: int, n_runs: int) -> list[dict]:
    """One back-end at one outlier rate, with every covariance on the same runs."""
    name, kernel = METHODS[method]
    scenario = make_scenario(
        LIE,
        n_poses=N_POSES,
        loop_density=LOOP_DENSITY,
        outlier_rate=rate,
        seed=seed,
        turn=TURN,
    )
    noise = NoiseModel(np.full(LIE.DOF, NOISE))
    false = {k for k, edge in enumerate(scenario.edges) if edge in scenario.outliers}

    # Same noise draws as monte_carlo(seed + 1), so E4 and E5 share runs.
    children = np.random.SeedSequence(seed + 1).spawn(n_runs)
    errors = np.full((n_runs, (N_POSES - 1) * LIE.DOF), np.nan)
    inverses = {c: [None] * n_runs for c in NAMES}
    solved = np.zeros(n_runs, dtype=bool)
    exact = np.zeros(n_runs, dtype=bool)

    for r, child in enumerate(children):
        graph = sample_graph(LIE, scenario, noise, np.random.default_rng(child))
        start = dead_reckon(LIE, graph, N_POSES)
        start[ANCHOR] = scenario.truth[ANCHOR]
        result = solve(name, kernel, graph, start)
        if not result.converged:
            continue
        solved[r] = True
        errors[r] = np.concatenate(
            [
                tangent_error(LIE, result.poses[k], scenario.truth[k])
                for k in range(N_POSES)
                if k != ANCHOR
            ]
        )
        closures = loop_closure_indices(graph)
        s = squared_residuals(graph, result.poses)
        exact[r] = {int(k) for k in closures if s[k] > THRESHOLD} == false
        covariances = robust_covariances(
            graph, result.poses, kernel, closures, THRESHOLD, anchor=ANCHOR
        )
        for c, cov in covariances.items():
            if cov is not None:
                inverses[c][r] = np.linalg.inv(cov)

    per_run = np.mean(errors[solved] ** 2, axis=1)
    rms = float(np.sqrt(per_run.mean())) if solved.any() else float("nan")
    rows = []
    for c in NAMES:
        defined = np.array([inv is not None for inv in inverses[c]])
        used = solved & defined
        nees = np.full(n_runs, np.nan)
        for r in np.flatnonzero(used):
            nees[r] = errors[r] @ inverses[c][r] @ errors[r]
        # Mean error over runs: what a better covariance cannot remove.
        bias = float("nan")
        if used.any():
            mean_error = errors[used].mean(axis=0)
            bias = float(
                np.mean([mean_error @ inverses[c][r] @ mean_error for r in np.flatnonzero(used)])
            )
        dof = errors.shape[1]
        stats = calibration(
            SimpleNamespace(converged=used, n_runs=n_runs, nees_full=nees, free_dof=dof)
        )
        rows.append(
            {
                "graph": seed,
                "method": name,
                "rate": rate,
                "covariance": c,
                "solved": int(solved.sum()),
                "undefined": int((solved & ~defined).sum()),
                "flagged_exact": float(exact[solved].mean()) if solved.any() else float("nan"),
                "rms_error": rms,
                "bias_ratio": bias / dof,
                # Split by whether the threshold flagged exactly the false closures (H6c).
                "exact_runs": int((used & exact).sum()),
                "nees_sum_exact": float(np.sum(nees[used & exact])),
                "nees_sum_missed": float(np.sum(nees[used & ~exact])),
                **stats,
            }
        )
    return rows


def tasks(seeds, gate_only: bool):
    for seed in seeds:
        yield 0, 0.0, seed
        if gate_only:
            continue
        for m in range(1, len(METHODS)):
            for rate in OUTLIER_RATES:
                yield m, rate, seed


def run(split: str, n_runs: int, gate_only: bool) -> None:
    seeds = SPLITS[split]
    rows = []
    for batch in parallel(condition, [(m, r, s, n_runs) for m, r, s in tasks(seeds, gate_only)]):
        rows.extend(batch)
        first = batch[0]
        print(
            f"  graph {first['graph']} {first['method']:<20} rate {first['rate']:<5} "
            + " ".join(f"{r['covariance']} {r['ratio']:.2f}" for r in batch),
            flush=True,
        )

    mark_fdr(rows, by=("graph", "covariance"))
    name = f"e5_{split}_gate" if gate_only else f"e5_{split}"
    write_results(rows, name)
    report(rows)


def calibrated(row) -> bool:
    return row["usable"] and not row["significant_after_fdr"]


def select(rows, **match):
    return [r for r in rows if all(r[k] == v for k, v in match.items())]


def report(rows) -> None:
    seeds = sorted({r["graph"] for r in rows})
    n = len(seeds)
    # Only the covariances this split recorded; the development run predates "expected".
    names = [c for c in NAMES if any(r["covariance"] == c for r in rows)]
    print("\nE5 - which covariance a robust back-end should report")
    print("=" * 84)

    print("\n  Gate: plain least squares, no outliers, calibrated layouts (need 6)")
    for c in names:
        gate = select(rows, method="plain least squares", covariance=c)
        ratios = ", ".join(f"{r['ratio']:.2f}" for r in gate)
        print(f"  {c:<13} {sum(calibrated(r) for r in gate)}/{n}   NEES/dof {ratios}")

    if not select(rows, method="Huber"):
        print()
        return

    print("\n  NEES/dof, median over layouts [min, max]")
    for method in ("Huber", "Cauchy", "DCS", "GNC"):
        print(f"\n  {method}")
        print(f"  {'rate':>6} " + "".join(f"{c:>22}" for c in names))
        for rate in OUTLIER_RATES:
            cells = []
            for c in names:
                ratios = np.array(
                    [r["ratio"] for r in select(rows, method=method, rate=rate, covariance=c)]
                )
                cells.append(
                    f"{np.nanmedian(ratios):8.2f} [{np.nanmin(ratios):5.2f},{np.nanmax(ratios):5.2f}]"
                    if np.isfinite(ratios).any()
                    else f"{'undefined':>22}"
                )
            print(f"  {rate:6.2f} " + "".join(f"{x:>22}" for x in cells))

    judged = [rate for rate in OUTLIER_RATES if rate >= JUDGED_RATE]
    print(f"\n  H6a: bias term against excess NEES, Huber naive, rates >= {JUDGED_RATE:g}")
    for rate in judged:
        level = select(rows, method="Huber", rate=rate, covariance="naive")
        share = [
            r["bias_ratio"] / (r["ratio"] - 1.0) for r in level if r["ratio"] > 1.0
        ]
        mostly = sum(s > 0.5 for s in share)
        print(
            f"  rate {rate:.2f}: bias share of excess, median {np.median(share):.2f}; "
            f"above half on {mostly}/{len(share)} layouts"
        )

    print(f"\n  A covariance fixes Huber if calibrated at every rate >= {JUDGED_RATE:g}")
    print("  on at least 6 layouts")
    for c in names:
        fixed = sum(
            all(
                calibrated(r)
                for r in select(rows, graph=seed, method="Huber", covariance=c)
                if r["rate"] >= JUDGED_RATE
            )
            for seed in seeds
        )
        print(f"  {c:<13} {fixed}/{n}")

    print("\n  H6c: Huber under the inlier covariance, runs pooled over layouts, split by")
    print("  whether the threshold flagged exactly the false closures")
    if "exact_runs" not in rows[0]:
        print("  not recorded for this split (added before the test run)")
        judged = []
    for rate in judged:
        level = select(rows, method="Huber", rate=rate, covariance="inlier")
        dof = level[0]["dof"]
        cells = []
        for key, count in (
            ("nees_sum_exact", sum(r["exact_runs"] for r in level)),
            ("nees_sum_missed", sum(r["converged"] - r["exact_runs"] for r in level)),
        ):
            if count == 0:
                cells.append("no runs")
                continue
            ratio = sum(r[key] for r in level) / count / dof
            low, high = (chi2.ppf(q, count * dof) / count / dof for q in (0.025, 0.975))
            cells.append(f"{ratio:.2f} over {count} runs (band {low:.2f}-{high:.2f})")
        print(f"  rate {rate:.2f}: exact {cells[0]}; missed {cells[1]}")

    print("\n  H6d: Cauchy and GNC at 0% outliers, calibrated layouts")
    for method in ("Cauchy", "GNC"):
        cells = ", ".join(
            f"{c} {sum(calibrated(r) for r in select(rows, method=method, rate=0.0, covariance=c))}/{n}"
            for c in names
        )
        print(f"  {method:<7} {cells}")
    print()


def build_figure(split: str):
    """Huber's calibration under each covariance, and the bias term."""
    rows = read_results(f"e5_{split}")
    fig, (left, right) = figure(nrows=1, ncols=2, size=(10.0, 4.4))
    rates = np.array(OUTLIER_RATES)

    for c, colour in COVARIANCE_COLOURS.items():
        per_rate = [
            np.array([r["ratio"] for r in select(rows, method="Huber", rate=x, covariance=c)])
            for x in rates
        ]
        if not any(np.isfinite(v).any() for v in per_rate):
            continue
        median = np.array([np.nanmedian(v) if np.isfinite(v).any() else np.nan for v in per_rate])
        low = np.array([np.nanmin(v) if np.isfinite(v).any() else np.nan for v in per_rate])
        high = np.array([np.nanmax(v) if np.isfinite(v).any() else np.nan for v in per_rate])
        left.fill_between(rates, low, high, color=colour, alpha=0.15, linewidth=0)
        left.plot(rates, median, marker="o", markersize=5, linewidth=2.0, color=colour, label=c)
    left.axhline(1.0, color=INK_MUTED, linewidth=1.5, linestyle=":")
    left.set_yscale("log")
    left.set_xticks(rates, [f"{x:g}" for x in rates])
    label(left, "Huber under each covariance", "outlier rate", "mean NEES / dof")
    left.legend(frameon=False, fontsize=8.5, labelcolor=INK_MUTED, loc="upper left")

    # One point per layout and rate: on the diagonal, the excess is all bias.
    ramp = ["#b5d0f2", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
    shown = []
    for colour, rate in zip(ramp, rates[1:]):
        level = select(rows, method="Huber", rate=rate, covariance="naive")
        excess = np.array([r["ratio"] - 1.0 for r in level])
        bias = np.array([r["bias_ratio"] for r in level])
        right.scatter(excess, bias, s=30, color=colour, label=f"{rate:g}", zorder=3)
        shown.extend(v for v in np.r_[excess, bias] if v > 0)
    ends = [0.8 * min(shown), 1.25 * max(shown)]
    right.plot(ends, ends, color=INK_MUTED, linewidth=1.5, linestyle=":", label="all bias")
    right.set_xscale("log")
    right.set_yscale("log")
    label(right, "Bias term against excess, per layout (naive)", "excess NEES / dof", "bias term / dof")
    right.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, loc="upper left", title="outlier rate",
                 title_fontsize=8)

    title(fig, f"E5 ({split} layouts): no covariance fixes Huber, because its error is bias")
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SPLITS, default="dev")
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--gate", action="store_true", help="plain least squares only")
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()

    if args.figures_only:
        report(read_results(f"e5_{args.split}"))
    else:
        run(args.split, args.runs, args.gate)
    if not args.gate:
        save_figures((lambda: build_figure(args.split), f"e5_{args.split}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
