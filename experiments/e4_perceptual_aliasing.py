"""E4: robust back-ends under false loop closures."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from _common import (
    BAND,
    INK,
    INK_MUTED,
    bootstrap_interval,
    calibration,
    figure,
    label,
    mark_fdr,
    read_results,
    save_figure,
    survivorship_warning,
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

LIE = se2
OUTLIER_RATES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
N_POSES = 20
LOOP_DENSITY = 1.0
NOISE = 0.05
TURN = 0.25
SEED = 300

THRESHOLD = chi2_threshold(LIE.DOF, 0.95)
DELTA = float(np.sqrt(THRESHOLD))

# Fixed palette order; the baseline is neutral and dashed.
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


def condition(method_name: str, solver, rate: float, n_runs: int) -> dict:
    """Run one method at one outlier rate."""
    scenario = make_scenario(
        LIE,
        n_poses=N_POSES,
        loop_density=LOOP_DENSITY,
        outlier_rate=rate,
        seed=SEED,
        turn=TURN,
    )
    result = monte_carlo(
        LIE,
        scenario,
        NoiseModel(np.full(LIE.DOF, NOISE)),
        n_runs=n_runs,
        seed=SEED + 1,
        solver=solver,
        initialize="odometry",
    )
    # RMS error over the free poses (the anchor is exact), with a bootstrap interval.
    free = [k for k in range(N_POSES) if k != result.anchor]
    per_run = np.mean(result.errors[result.converged][:, free] ** 2, axis=(1, 2))
    if per_run.size >= 2:
        rms_error = float(np.sqrt(per_run.mean()))
        rms_low, rms_high = np.sqrt(bootstrap_interval(per_run))
    else:
        rms_error = rms_low = rms_high = float("nan")
    return {
        "method": method_name,
        "rate": rate,
        "outliers": len(scenario.outliers),
        "rms_error": rms_error,
        "rms_ci_low": float(rms_low),
        "rms_ci_high": float(rms_high),
        **calibration(result),
    }


def _condition_by_index(method: int, rate: float, n_runs: int) -> dict:
    """condition() by method index, since the solvers can't be pickled."""
    name, solver, _ = METHODS[method]
    return condition(name, solver, rate, n_runs)


def run(n_runs: int) -> None:
    # Conditions are seeded independently, so parallel runs give the same results.
    tasks = [(m, rate) for m in range(len(METHODS)) for rate in OUTLIER_RATES]
    with ProcessPoolExecutor() as pool:
        futures = [pool.submit(_condition_by_index, m, rate, n_runs) for m, rate in tasks]
        rows = []
        for future in futures:
            row = future.result()
            rows.append(row)
            print(
                f"  {row['method']:<20} rate {row['rate']:<5} -> "
                f"rms {row['rms_error']:.3f}, {row['verdict']}",
                flush=True,
            )

    mark_fdr(rows)
    write_results(rows, "e4_aliasing")
    report_console(rows)


def _miscalibrated(row, direction: int) -> bool:
    """Significant after FDR, in the given direction (+1 over, -1 under)."""
    return (
        row["usable"]
        and row["significant_after_fdr"]
        and (row["ratio"] - 1.0) * direction > 0
    )


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

    survivorship_warning(rows, lambda r: f"{r['method']} rate {r['rate']}")

    print("\n  The dangerous quadrant: accuracy recovered, covariance still wrong.")
    # Recovered = below twice the method's own clean error. The threshold was
    # chosen after the first runs (see HYPOTHESES.md).
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
        over = sum(1 for r in corrupted if _miscalibrated(r, +1))
        under = sum(1 for r in corrupted if _miscalibrated(r, -1))
        family = "redescending" if method_name in REDESCENDING else "convex"
        print(
            f"  {method_name:<20} ({family:<12}) recovered accuracy at "
            f"{recovered}/{len(corrupted)} corrupted rates; "
            f"overconfident at {over}, conservative at {under}"
        )

    # Conservative with no outliers means good constraints are being down-weighted.
    inflated = [
        r["method"] for r in rows if r["rate"] == 0.0 and _miscalibrated(r, -1)
    ]
    if inflated:
        print(f"\n  Conservative with no outliers at all: {', '.join(inflated)}.")
        print("  These kernels down-weight correct constraints, which inflates")
        print("  the covariance they report.")

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
        total = [
            r for r in rows if r["method"] in members and r["usable"] and r["rate"] > 0
        ]
        over = sum(1 for r in total if _miscalibrated(r, +1))
        under = sum(1 for r in total if _miscalibrated(r, -1))
        print(
            f"  {family:<13}: of {len(total)} conditions, overconfident at {over}, "
            f"conservative at {under}, consistent at {len(total) - over - under}"
        )
    print()


def build_figure():
    """Trajectory error and calibration against the outlier rate."""
    rows = read_results("e4_aliasing")
    # Stacked panels over a shared x.
    fig, axes = figure(nrows=2, ncols=1, size=(8.2, 7.4), sharex=True)
    top, bottom = axes

    colours = dict(zip([m[0] for m in METHODS if not m[2]], METHOD_COLOURS))
    rates = np.array(sorted({r["rate"] for r in rows}))
    # Offset the methods slightly so their intervals don't hide each other.
    step = rates[1] - rates[0]
    offsets = (np.arange(len(METHODS)) - (len(METHODS) - 1) / 2) * step * 0.09

    for index, (method_name, _, is_baseline) in enumerate(METHODS):
        by_rate = {
            r["rate"]: r
            for r in rows
            if r["method"] == method_name and r["usable"]
        }
        if not by_rate:
            continue

        # NaN at rejected conditions, so the line breaks instead of bridging them.
        def column(key, table=by_rate):
            return np.array([table[x][key] if x in table else np.nan for x in rates])

        style = (
            BASELINE_STYLE
            if is_baseline
            else {"color": colours[method_name], "linestyle": "-"}
        )
        panels = [(top, "rms_error", "rms_ci_low", "rms_ci_high")]
        if not is_baseline:
            panels.append((bottom, "ratio", "ci_low", "ci_high"))
        for axis, key, low, high in panels:
            value = column(key)
            axis.errorbar(
                rates + offsets[index],
                value,
                yerr=[value - column(low), column(high) - value],
                marker="o",
                markersize=6,
                linewidth=2.0,
                elinewidth=1.5,
                capsize=3,
                ecolor=style["color"],
                label=method_name,
                **style,
            )

    robust = [r for r in rows if r["method"] != METHODS[0][0] and r["usable"]]
    bottom.axhspan(
        robust[0]["band_low"],
        robust[0]["band_high"],
        color=BAND,
        alpha=0.9,
        linewidth=0,
        label="chi-squared acceptance band",
        zorder=0,
    )
    bottom.axhline(
        1.0, color=INK_MUTED, linewidth=2.0, linestyle=":", label="calibrated"
    )
    # Log scale, so over- and under-confidence look symmetric.
    bottom.set_yscale("log")
    ticks = [0.9, 1.0, 1.5, 2.0, 3.0]
    bottom.set_yticks(ticks, [f"{t:g}" for t in ticks])
    bottom.minorticks_off()

    baseline = [
        r["ratio"] for r in rows if r["method"] == METHODS[0][0] and r["rate"] > 0
        and r["usable"]
    ]
    if baseline:
        bottom.text(
            0.02,
            0.97,
            f"{METHODS[0][0]} not shown: {min(baseline):.0f} to "
            f"{max(baseline):.0f} at every corrupted rate",
            transform=bottom.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            color=INK_MUTED,
        )

    top.set_xticks(rates, [f"{x:g}" for x in rates])
    n_closures = round(N_POSES * LOOP_DENSITY)
    label(top, "Trajectory error", "", "RMS error (tangent units)")
    label(
        bottom,
        "Calibration of the robust back-ends",
        f"outlier rate (share of the {n_closures} loop closures that are false)",
        "mean NEES / dof",
    )

    handles, labels = top.get_legend_handles_labels()
    for handle, text in zip(*bottom.get_legend_handles_labels()):
        if text not in labels:
            handles.append(handle)
            labels.append(text)
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncols=4,
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
        print(f"figure built but not written: no usable renderer ({exc})")
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
