"""E1 (week 1): linear-Gaussian pose chain. Laplace covariance must be exact
to near machine precision; empirical NEES must sit inside the chi-squared
band. Write and pass this before anything else.

Everywhere else in this study the reported covariance is an approximation of
unknown quality -- that is the thing being measured. Here it is not. On a
linear-Gaussian problem the Laplace approximation is the exact posterior, so
the right answer is known in advance and any disagreement is a defect in this
code rather than a finding about SLAM.

Two claims, both known-answer:

  exactness   On a chain whose Jacobians are exactly -I / +I, the marginal of
              pose k must be k * Omega^-1 and the relative covariance over m
              steps must be m * Omega^-1. Variances add along a chain.

  consistency Under small noise and gentle curvature the problem stays close
              to linear, so the empirical spread of the estimates over many
              noise draws must match the covariance the solver reported.

The gate is deliberately run at a wider band than the analysis alpha fixed in
HYPOTHESES.md. At alpha = 0.05 a correct implementation fails one condition in
twenty by construction, and a gate that cries wolf gets ignored, which is
worse than not having one. Detecting a real defect does not need the extra
sensitivity: the failures this catches are order-of-magnitude, not marginal.

Run:  python experiments/e1_validation_gate.py [--runs N] [--figures-only]
Exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from _common import (
    INK,
    INK_MUTED,
    OBSERVED,
    figure,
    label,
    read_results,
    save_figure,
    write_results,
)
from scipy.stats import chi2

from posetrust.covariance import marginal_covariances, relative_covariance
from posetrust.graph import PoseGraph
from posetrust.lie import se2, se3
from posetrust.simulate import NoiseModel, make_scenario, monte_carlo
from posetrust.stats import ConsistencyReport

GROUPS = [("SE(2)", se2), ("SE(3)", se3)]
NOISE_LEVELS = [1e-4, 1e-3, 1e-2]
HEADLINE_NOISE = 1e-3
GATE_ALPHA = 0.001
EXACTNESS_TOLERANCE = 1e-10
CHAIN_LENGTH = 8


def exactness(lie) -> dict:
    """Marginals on an exactly linear-Gaussian chain against their closed form."""
    dof = lie.DOF
    omega = np.diag(np.linspace(1.0, 4.0, dof))
    omega_inv = np.linalg.inv(omega)
    identity = lie.exp(np.zeros(dof))

    graph = PoseGraph(lie)
    for _ in range(CHAIN_LENGTH):
        graph.add_pose(identity)
    for k in range(CHAIN_LENGTH - 1):
        graph.add_factor(k, k + 1, identity, omega)
    H, _ = graph.linearize(graph.poses)

    marginals = marginal_covariances(H, anchor=0, dof=dof)
    marginal_error = max(
        float(np.max(np.abs(block - k * omega_inv)))
        for k, block in enumerate(marginals)
    )
    relative_error = max(
        float(
            np.max(
                np.abs(
                    relative_covariance(lie, H, graph.poses, i, j, anchor=0)
                    - (j - i) * omega_inv
                )
            )
        )
        for i, j in [(0, 1), (2, 6), (1, 7), (3, 4)]
    )
    worst = max(marginal_error, relative_error)
    return {
        "group": None,  # filled by the caller
        "marginal_error": marginal_error,
        "relative_error": relative_error,
        "worst_error": worst,
        "tolerance": EXACTNESS_TOLERANCE,
        "passed": worst < EXACTNESS_TOLERANCE,
    }


def consistency(lie, sigma: float, n_runs: int):
    """Monte Carlo NEES against the chi-squared band, near the linear regime."""
    scenario = make_scenario(lie, n_poses=6, loop_density=0.3, seed=1, turn=0.05)
    result = monte_carlo(
        lie, scenario, NoiseModel(np.full(lie.DOF, sigma)), n_runs=n_runs, seed=7
    )
    converged = result.converged
    report = ConsistencyReport(
        result.nees_full[converged], result.free_dof, alpha=GATE_ALPHA
    )
    low, high = report.acceptance
    nominal, empirical = report.coverage()
    summary = {
        "group": None,
        "sigma": sigma,
        "dof": report.dof,
        "n_runs": int(n_runs),
        "converged": int(converged.sum()),
        "mean_nees": report.mean,
        "ratio": report.mean / report.dof,
        "band_low": low,
        "band_high": high,
        "pvalue": report.pvalue,
        "verdict": report.verdict,
        "coverage_nominal": nominal.tolist(),
        "coverage_empirical": empirical.tolist(),
        "passed": bool(low <= report.mean <= high and converged.all()),
    }
    samples = [
        {"group": None, "sigma": sigma, "run": i, "nees": float(v)}
        for i, v in enumerate(result.nees_full[converged])
    ]
    return summary, samples


def run(n_runs: int) -> bool:
    exact_rows, nees_rows, sample_rows = [], [], []

    for name, lie in GROUPS:
        row = exactness(lie)
        row["group"] = name
        exact_rows.append(row)

        for sigma in NOISE_LEVELS:
            summary, samples = consistency(lie, sigma, n_runs)
            summary["group"] = name
            nees_rows.append(summary)
            for s in samples:
                s["group"] = name
            sample_rows.extend(samples)

    write_results(exact_rows, "e1_exactness")
    write_results(nees_rows, "e1_nees")
    write_results(sample_rows, "e1_nees_samples")
    return report_console(exact_rows, nees_rows)


def report_console(exact_rows, nees_rows) -> bool:
    print("\nE1 - validation gate")
    print("=" * 74)
    print("\nExactness: marginals against their closed form on a linear-Gaussian chain")
    print(f"  {'group':<8} {'marginal':>12} {'relative':>12} {'tolerance':>12}  result")
    for row in exact_rows:
        print(
            f"  {row['group']:<8} {row['marginal_error']:12.2e} "
            f"{row['relative_error']:12.2e} {row['tolerance']:12.0e}  "
            f"{'PASS' if row['passed'] else 'FAIL'}"
        )

    print(f"\nConsistency: mean NEES against the chi-squared band (alpha={GATE_ALPHA})")
    print(
        f"  {'group':<8} {'sigma':>8} {'dof':>5} {'conv':>8} {'mean':>9} "
        f"{'ratio':>7} {'band':>17} {'p':>8}  result"
    )
    for row in nees_rows:
        band = f"[{row['band_low']:.2f}, {row['band_high']:.2f}]"
        print(
            f"  {row['group']:<8} {row['sigma']:8.0e} {row['dof']:5d} "
            f"{row['converged']:4d}/{row['n_runs']:<3d} {row['mean_nees']:9.3f} "
            f"{row['ratio']:7.4f} {band:>17} {row['pvalue']:8.3f}  "
            f"{'PASS' if row['passed'] else 'FAIL'}"
        )

    passed = all(r["passed"] for r in exact_rows) and all(
        r["passed"] for r in nees_rows
    )
    print("\n" + "=" * 74)
    print("E1 GATE: PASS" if passed else "E1 GATE: FAIL")
    if not passed:
        print("Nothing downstream is reportable until this passes.")
    print()
    return passed


def figures() -> bool:
    """Regenerate every figure from the stored results, in seconds.

    Separated from the gate's verdict on purpose: whether a chart could be
    rendered says nothing about whether the covariance is correct, so a
    missing plotting backend must not be able to fail the science.
    """
    fig = build_figure()
    try:
        path = save_figure(fig, "e1_validation_gate")
    except ImportError as exc:
        print(f"figure built but not written -- no usable renderer here ({exc})")
        print("results are in results/; rerun with --figures-only to write it.")
        return False
    print(f"figure written to {path}")
    return True


def build_figure():
    """Assemble the E1 figure. No rendering, so this runs anywhere."""
    summaries = read_results("e1_nees")
    samples = read_results("e1_nees_samples")

    fig, axes = figure(nrows=2, ncols=2, size=(10.0, 7.0))
    for row_index, (name, _) in enumerate(GROUPS):
        summary = next(
            s
            for s in summaries
            if s["group"] == name and s["sigma"] == HEADLINE_NOISE
        )
        values = np.array(
            [
                s["nees"]
                for s in samples
                if s["group"] == name and s["sigma"] == HEADLINE_NOISE
            ]
        )
        dof = summary["dof"]

        # Left: does the NEES distribution match the chi-squared it should be?
        ax = axes[row_index][0]
        ax.hist(
            values,
            bins=20,
            density=True,
            color=OBSERVED,
            alpha=0.85,
            label="observed",
        )
        grid = np.linspace(max(0.0, values.min() * 0.6), values.max() * 1.15, 400)
        ax.plot(
            grid,
            chi2(dof).pdf(grid),
            color=INK_MUTED,
            linewidth=2.0,
            linestyle="--",
            label=f"chi-squared({dof})",
        )
        ax.axvline(
            dof,
            color=INK_MUTED,
            linewidth=1.0,
            alpha=0.6,
            label=f"expected mean = {dof}",
        )
        label(ax, f"{name} - NEES distribution", "NEES", "density")
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_MUTED, loc="upper left")
        ax.text(
            0.97,
            0.93,
            f"mean/dof = {summary['ratio']:.4f}\n{summary['verdict']}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color=INK,
        )

        # Right: do the credible ellipsoids cover what they claim to?
        ax = axes[row_index][1]
        nominal = np.array(summary["coverage_nominal"])
        empirical = np.array(summary["coverage_empirical"])
        ax.plot(
            [0, 1], [0, 1], color=INK_MUTED, linewidth=2.0, linestyle="--", label="ideal"
        )
        ax.plot(
            nominal,
            empirical,
            marker="o",
            markersize=8,
            linewidth=2.0,
            color=OBSERVED,
            label="observed",
        )
        ax.set_xlim(0.4, 1.02)
        ax.set_ylim(0.4, 1.02)
        label(ax, f"{name} - ellipsoid coverage", "nominal level", "empirical coverage")
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_MUTED, loc="upper left")
        # the deviation is far smaller than the axis can resolve, so state it
        ax.text(
            0.97,
            0.06,
            f"max deviation {np.max(np.abs(nominal - empirical)):.3f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color=INK,
        )

    fig.suptitle(
        "E1: where the Laplace covariance is exact, the solver agrees with it",
        color=INK,
        fontsize=12,
        x=0.02,
        ha="left",
    )
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=300)
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="regenerate figures from stored results without re-running",
    )
    args = parser.parse_args()

    if args.figures_only:
        passed = report_console(read_results("e1_exactness"), read_results("e1_nees"))
    else:
        passed = run(args.runs)
    figures()
    return 0 if passed else 1  # the gate's verdict, not the plotting backend's


if __name__ == "__main__":
    sys.exit(main())
