"""E4 (Q3, the prize): false loop closures at rates 0-30%. Compare plain
least squares, Huber, Cauchy, switchable constraints, and graduated
non-convexity on trajectory error and NEES (F3).

The interesting quadrant is low trajectory error with bad NEES: a back-end
that looks like it worked and reports an uncertainty that is wrong anyway.
Any evaluation scoring only accuracy would call that a success, and a system
that then sized a safety margin from the covariance would be relying on a
number nothing had checked.

H4 as refined in HYPOTHESES.md, still open: whether calibration is restored
depends on whether the kernel redescends. A convex kernel never drives an
outlier's weight to zero, so the surviving weight keeps inflating H even as
the majority of good constraints pull the trajectory back. Redescending
kernels zero it and recover both. The gap should widen with the outlier rate.
The preview that motivated this saw it at a single rate; sweeping is what
turns that into a result or kills it.

Initialisation is dead reckoning, not ground truth. That is pre-registered:
everywhere else the question is whether the covariance at the optimum is
honest, and starting at the truth keeps convergence from confounding it. Here
convergence is part of the question -- a back-end that cannot find the optimum
from a realistic starting point has not solved the problem.

One deviation from the original figure plan, which asked for trajectory error
and NEES on twin axes. Two y-scales on one frame let a reader infer whichever
relationship the author wants, and the comparison here is exactly the kind
that invites it. The two measures get a panel each over a shared x instead.

Run:  python experiments/e4_perceptual_aliasing.py [--runs N] [--figures-only]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from _common import (
    INK,
    INK_MUTED,
    figure,
    label,
    read_results,
    save_figure,
    write_results,
)

from posetrust.lie import se2
from posetrust.optimize.optimizer import gauss_newton
from posetrust.optimize.robust import (
    Cauchy,
    Huber,
    SwitchableConstraints,
    chi2_threshold,
    graduated_non_convexity,
    irls,
    loop_closure_indices,
)
from posetrust.simulate import NoiseModel, make_scenario, monte_carlo
from posetrust.stats import ConsistencyReport, benjamini_hochberg

LIE = se2
OUTLIER_RATES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
N_POSES = 12
LOOP_DENSITY = 0.8
NOISE = 0.05
TURN = 0.25
ALPHA = 0.05
MIN_CONVERGED_FRACTION = 0.5

THRESHOLD = chi2_threshold(LIE.DOF, 0.95)
DELTA = float(np.sqrt(THRESHOLD))

# Categorical slots in the reference palette's fixed order. The baseline is
# not one of the methods under test, so it wears neutral ink and a dashed
# stroke: it is the thing they are being compared against.
BASELINE_STYLE = {"color": INK_MUTED, "linestyle": "--"}
METHOD_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def _with_kernel(kernel):
    def solve(graph, poses, anchor=0):
        return irls(
            graph,
            poses,
            kernel,
            anchor=anchor,
            robust_factors=loop_closure_indices(graph),
        )

    return solve


def _gnc(graph, poses, anchor=0):
    return graduated_non_convexity(
        graph, poses, c=DELTA, anchor=anchor, robust_factors=loop_closure_indices(graph)
    )


METHODS = [
    ("plain least squares", gauss_newton, True),
    ("Huber", _with_kernel(Huber(DELTA)), False),
    ("Cauchy", _with_kernel(Cauchy(DELTA)), False),
    ("switchable", _with_kernel(SwitchableConstraints(THRESHOLD)), False),
    ("GNC", _gnc, False),
]
REDESCENDING = {"Cauchy", "switchable", "GNC"}


def condition(method_name: str, solver, rate: float, n_runs: int, seed: int) -> dict:
    """One method at one outlier rate: trajectory error and calibration."""
    scenario = make_scenario(
        LIE,
        n_poses=N_POSES,
        loop_density=LOOP_DENSITY,
        outlier_rate=rate,
        seed=seed,
        turn=TURN,
    )
    result = monte_carlo(
        LIE,
        scenario,
        NoiseModel(np.full(LIE.DOF, NOISE)),
        n_runs=n_runs,
        seed=seed + 1,
        solver=solver,
        initialize="odometry",
    )
    converged = result.converged
    fraction = float(converged.mean())
    usable = fraction >= MIN_CONVERGED_FRACTION

    # Trajectory error, in the same tangent-space units the covariance uses.
    rms_error = float(np.sqrt(np.mean(result.errors[converged] ** 2)))

    values = result.nees_full[converged]
    report = ConsistencyReport(values, result.free_dof, ALPHA)
    rng = np.random.default_rng(seed)
    boot = rng.choice(values, size=(2000, values.size), replace=True).mean(axis=1)
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5]) / report.dof

    return {
        "method": method_name,
        "rate": rate,
        "dof": report.dof,
        "n_runs": int(n_runs),
        "converged": int(converged.sum()),
        "converged_fraction": fraction,
        "usable": usable,
        "rms_error": rms_error,
        "ratio": report.mean / report.dof,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "band_low": report.acceptance[0] / report.dof,
        "band_high": report.acceptance[1] / report.dof,
        "pvalue": report.pvalue,
        "verdict": report.verdict if usable else "convergence failure",
    }


def run(n_runs: int) -> None:
    rows = []
    for method_name, solver, _ in METHODS:
        for index, rate in enumerate(OUTLIER_RATES):
            row = condition(method_name, solver, rate, n_runs, seed=300 + index)
            rows.append(row)
            print(
                f"  {method_name:<20} rate {rate:<5} -> "
                f"rms {row['rms_error']:.3f}, {row['verdict']}"
            )

    usable = [r for r in rows if r["usable"]]
    flagged = benjamini_hochberg(np.array([r["pvalue"] for r in usable]), ALPHA)
    for row, reject in zip(usable, flagged):
        row["significant_after_fdr"] = bool(reject)
    for row in rows:
        row.setdefault("significant_after_fdr", False)

    write_results(rows, "e4_aliasing")
    report_console(rows)


def report_console(rows) -> None:
    print("\nE4 - perceptual aliasing (H4)")
    print("=" * 96)
    print(
        f"  {'method':<20} {'rate':>6} {'conv':>9} {'rms err':>8} {'NEES/dof':>9} "
        f"{'95% interval':>17} {'verdict':<14} FDR"
    )
    for row in rows:
        ci = f"[{row['ci_low']:.2f}, {row['ci_high']:.2f}]"
        mark = "*" if row["significant_after_fdr"] else ""
        print(
            f"  {row['method']:<20} {row['rate']:6.2f} "
            f"{row['converged']:4d}/{row['n_runs']:<4d} {row['rms_error']:8.3f} "
            f"{row['ratio']:9.2f} {ci:>17} {row['verdict']:<14} {mark}"
        )
    print("\n  * survives Benjamini-Hochberg across the sweep")

    suspect = [r for r in rows if r["usable"] and r["converged_fraction"] < 0.9]
    if suspect:
        print("\n  Survivorship warning: these dropped more than 10% of runs, so")
        print("  their ratios are biased towards calibration and are lower bounds:")
        for row in suspect:
            print(
                f"    {row['method']} rate {row['rate']}: "
                f"{row['converged']}/{row['n_runs']} converged, "
                f"ratio {row['ratio']:.1f}"
            )

    print("\n  The dangerous quadrant: accuracy recovered, covariance still wrong.")
    # "Recovered" is judged against the method's own uncorrupted run, not
    # against plain least squares at the same rate: the baseline fails to
    # converge at half the corrupted rates, so its trajectory error there is
    # not a number anything should be compared against.
    for method_name, _, is_baseline in METHODS:
        if is_baseline:
            continue
        series = sorted(
            (r for r in rows if r["method"] == method_name and r["usable"]),
            key=lambda r: r["rate"],
        )
        corrupted = [r for r in series if r["rate"] > 0]
        clean = next((r for r in series if r["rate"] == 0.0), None)
        if not corrupted or clean is None:
            continue
        recovered = sum(
            1 for r in corrupted if r["rms_error"] < 2.0 * clean["rms_error"]
        )
        miscalibrated = sum(
            1 for r in corrupted if r["verdict"] == "OVERCONFIDENT"
        )
        family = "redescending" if method_name in REDESCENDING else "convex"
        print(
            f"  {method_name:<20} ({family:<12}) recovered accuracy at "
            f"{recovered}/{len(corrupted)} corrupted rates, "
            f"overconfident at {miscalibrated}/{len(corrupted)}"
        )

    failed = [
        r
        for r in rows
        if r["method"] == "plain least squares" and r["rate"] > 0 and not r["usable"]
    ]
    if failed:
        rates = ", ".join(f"{r['rate']:g}" for r in failed)
        print(
            f"\n  Plain least squares did not converge at rates {rates}. Under "
            "aliasing\n  its failure is not a miscalibrated answer, it is no "
            "answer at all."
        )

    print("\n  H4 refined: convex kernels recover accuracy but not calibration;")
    print("  redescending kernels recover both. Judged on the corrupted rates only.")
    for family, members in (
        ("convex", ["Huber"]),
        ("redescending", sorted(REDESCENDING)),
    ):
        bad = [
            r
            for r in rows
            if r["method"] in members and r["usable"] and r["rate"] > 0
            and r["verdict"] == "OVERCONFIDENT"
        ]
        total = [
            r for r in rows if r["method"] in members and r["usable"] and r["rate"] > 0
        ]
        print(f"  {family:<13}: overconfident at {len(bad)}/{len(total)} conditions")
    print()


def build_figure():
    """F3: outlier rate against trajectory error and calibration.

    Two panels over a shared x rather than twin y-axes. The whole point is
    that the two measures disagree, and a dual-axis chart would let their
    relative scaling be chosen rather than read.
    """
    rows = read_results("e4_aliasing")
    # Stacked over a common x, so the axis is shared: ticks line up and the
    # label is written once for the column rather than twice.
    fig, axes = figure(nrows=2, ncols=1, size=(8.2, 7.0), sharex=True)
    top, bottom = axes

    colours = dict(zip([m[0] for m in METHODS if not m[2]], METHOD_COLOURS))

    rates = np.array(sorted({r["rate"] for r in rows}))
    for method_name, _, is_baseline in METHODS:
        by_rate = {
            r["rate"]: r
            for r in rows
            if r["method"] == method_name and r["usable"]
        }
        if not by_rate:
            continue
        # NaN at every rate this method has no usable result for, so the line
        # breaks there. Joining across an excluded condition would draw a
        # segment through values that were never measured -- plain least
        # squares fails to converge at three rates, and a continuous line
        # would claim it was tracked across all of them.
        errors = np.array([by_rate[x]["rms_error"] if x in by_rate else np.nan for x in rates])
        ratios = np.array([by_rate[x]["ratio"] if x in by_rate else np.nan for x in rates])
        style = (
            BASELINE_STYLE
            if is_baseline
            else {"color": colours[method_name], "linestyle": "-"}
        )

        for axis, values in ((top, errors), (bottom, ratios)):
            axis.plot(
                rates,
                values,
                marker="o",
                markersize=6,
                linewidth=2.0,
                label=method_name,
                **style,
            )

    # A reference line rather than the acceptance band. The band is three
    # percent wide and this axis spans two and a half decades, so it renders
    # as a sliver on top of the line and adds nothing a reader can use.
    bottom.axhline(
        1.0, color=INK_MUTED, linewidth=2.0, linestyle=":", label="calibrated"
    )
    bottom.set_yscale("log")

    label(top, "Trajectory error", "", "RMS error (tangent units)")
    label(bottom, "Calibration", "outlier rate", "mean NEES / dof")

    handles, labels = top.get_legend_handles_labels()
    extra_h, extra_l = bottom.get_legend_handles_labels()
    handles.append(extra_h[-1])
    labels.append(extra_l[-1])
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
        "E4: robust back-ends restore the trajectory. Not all restore the covariance.",
        color=INK,
        fontsize=12,
        x=0.02,
        ha="left",
    )
    return fig


def figures() -> bool:
    try:
        print(f"figure written to {save_figure(build_figure(), 'e4_perceptual_aliasing')}")
    except ImportError as exc:
        print(f"figure built but not written -- no usable renderer ({exc})")
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()

    if args.figures_only:
        report_console(read_results("e4_aliasing"))
    else:
        run(args.runs)
    figures()
    return 0


if __name__ == "__main__":
    sys.exit(main())
