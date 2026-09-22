"""E3 (Q2): sweep rotational noise, SE(2) and SE(3) separately. Report
whether failure is graceful or abrupt, and the per-degree-of-freedom split.

H2, as pre-registered: overconfidence grows with rotational noise and the
onset is abrupt rather than gradual. H2 was already seen in preview before
HYPOTHESES.md was written, and is recorded there as an observation rather
than a prediction; this run is the proper measurement of it.

H3 -- "SE(3) is materially worse calibrated than SE(2) at matched rotational
uncertainty" -- was withdrawn before this ran, because no fair matching
across groups of different dimension exists. See HYPOTHESES.md. In its place
this reports, EXPLORATORY and not predicted, the rotation-error magnitude at
which each group loses calibration. That is a threshold in radians, common to
both groups, so it compares where each breaks rather than their values at an
arbitrarily matched sigma.

Why rotation and not translation: the exponential map is linear in the
translational part and non-linear only through rotation, so rotation is the
only axis along which the Laplace approximation can degrade. Translation
noise is held fixed throughout to keep that attribution clean.

Run:  python experiments/e3_nonlinearity.py [--runs N] [--figures-only]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from _common import (
    BAND,
    INK,
    INK_MUTED,
    OBSERVED,
    figure,
    label,
    read_results,
    save_figure,
    write_results,
)

from posetrust.lie import se2, se3
from posetrust.optimize.optimizer import levenberg_marquardt
from posetrust.simulate import NoiseModel, make_scenario, monte_carlo
from posetrust.stats import ConsistencyReport, benjamini_hochberg, nees_by_dof

GROUPS = [("SE(2)", se2), ("SE(3)", se3)]
ROTATION_NOISE = [0.01, 0.03, 0.06, 0.10, 0.15, 0.22, 0.30, 0.45]
COVERAGE_LEVELS = (0.5, 0.75, 0.9, 0.95, 0.99)
TRANSLATION_NOISE = 0.02
N_POSES = 10
LOOP_DENSITY = 0.3
TURN = 0.25
ALPHA = 0.05
MIN_CONVERGED_FRACTION = 0.5

# Ordinal ramp from the reference palette's sequential blue, starting at the
# step that still clears 2:1 against the light surface. Noise level is a
# magnitude, so it is encoded light to dark in one hue rather than by category.
NOISE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]


def condition(lie, rotation_sigma: float, n_runs: int, seed: int) -> dict:
    """One rotational noise level: NEES, coverage, and the per-DOF split."""
    sigma = np.full(lie.DOF, TRANSLATION_NOISE)
    sigma[lie.TRANSLATION_DOF :] = rotation_sigma

    scenario = make_scenario(
        lie, n_poses=N_POSES, loop_density=LOOP_DENSITY, seed=seed, turn=TURN
    )
    # Levenberg-Marquardt rather than plain Gauss-Newton. Both find the same
    # optimum where both converge, so this does not change what is measured;
    # it converges more often at high noise, and every run that fails to
    # converge is dropped. Dropping them is necessary but not neutral: the
    # survivors are the draws whose noise happened to be benign, which biases
    # the estimate towards calibration exactly where miscalibration is worst.
    # Converging more often shrinks that bias.
    result = monte_carlo(
        lie,
        scenario,
        NoiseModel(sigma),
        n_runs=n_runs,
        seed=seed + 1,
        solver=levenberg_marquardt,
    )
    converged = result.converged
    fraction = float(converged.mean())
    usable = fraction >= MIN_CONVERGED_FRACTION

    values = result.nees_full[converged]
    report = ConsistencyReport(values, result.free_dof, ALPHA)
    low, high = report.acceptance

    rng = np.random.default_rng(seed)
    boot = rng.choice(values, size=(2000, values.size), replace=True).mean(axis=1)
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5]) / report.dof

    # Realised rotational error magnitude, in radians. This is the common unit
    # the two groups can be compared in, since it measures how far the
    # estimate actually turned away from the truth rather than what noise was
    # injected into how many axes.
    free = [k for k in range(N_POSES) if k != result.anchor]
    rotation_error = result.errors[converged][:, free, lie.TRANSLATION_DOF :]
    rms_rotation = float(np.sqrt(np.mean(np.sum(rotation_error**2, axis=-1))))

    # Translation against rotation. Pooled across poses, so correlated and
    # descriptive: the formal test is the full-state NEES above.
    flat_errors = result.errors[converged][:, free].reshape(-1, lie.DOF)
    flat_marginals = result.marginals[converged][:, free].reshape(
        -1, lie.DOF, lie.DOF
    )
    split = nees_by_dof(lie, flat_errors, flat_marginals)
    translation_dof = lie.TRANSLATION_DOF
    rotation_dof = lie.DOF - translation_dof

    nominal, empirical = report.coverage(COVERAGE_LEVELS)
    return {
        "group": None,
        "rotation_sigma": rotation_sigma,
        "dof": report.dof,
        "n_runs": int(n_runs),
        "converged": int(converged.sum()),
        "converged_fraction": fraction,
        "usable": usable,
        "mean_nees": report.mean,
        "ratio": report.mean / report.dof,
        "band_low": low / report.dof,
        "band_high": high / report.dof,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "pvalue": report.pvalue,
        "verdict": report.verdict if usable else "convergence failure",
        "rms_rotation_error": rms_rotation,
        "translation_ratio": float(split["translation"].mean() / translation_dof),
        "rotation_ratio": float(split["rotation"].mean() / rotation_dof),
        "coverage_nominal": list(nominal),
        "coverage_empirical": list(empirical),
    }


def run(n_runs: int) -> None:
    rows = []
    for name, lie in GROUPS:
        for index, sigma in enumerate(ROTATION_NOISE):
            row = condition(lie, sigma, n_runs, seed=200 + index)
            row["group"] = name
            rows.append(row)
            print(f"  {name} rot sigma {sigma:<5} -> {row['verdict']}")

    usable = [r for r in rows if r["usable"]]
    flagged = benjamini_hochberg(np.array([r["pvalue"] for r in usable]), ALPHA)
    for row, reject in zip(usable, flagged):
        row["significant_after_fdr"] = bool(reject)
    for row in rows:
        row.setdefault("significant_after_fdr", False)

    write_results(rows, "e3_nonlinearity")
    report_console(rows)


def report_console(rows) -> None:
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

    # Dropping non-converged runs is necessary but not neutral. The survivors
    # are the draws whose noise happened to be benign, so a condition losing
    # runs reads as better calibrated than it is -- which is why the ratio
    # falls at the top of the sweep rather than the effect levelling off.
    suspect = [r for r in rows if r["usable"] and r["converged_fraction"] < 0.9]
    if suspect:
        print("\n  Survivorship warning: these dropped more than 10% of runs, so")
        print("  their ratios are biased towards calibration and are lower bounds:")
        for row in suspect:
            print(
                f"    {row['group']} sd {row['rotation_sigma']}: "
                f"{row['converged']}/{row['n_runs']} converged, "
                f"ratio {row['ratio']:.1f}"
            )

    print("\n  H2 predicted: overconfidence grows with rotational noise, abruptly.")
    for name, _ in GROUPS:
        series = sorted(
            (r for r in rows if r["group"] == name and r["usable"]),
            key=lambda r: r["rotation_sigma"],
        )
        ratios = [r["ratio"] for r in series]
        rising = all(a <= b + 1e-9 for a, b in zip(ratios, ratios[1:]))
        biggest = max(
            (b / a, series[i + 1]["rotation_sigma"])
            for i, (a, b) in enumerate(zip(ratios, ratios[1:]))
        )
        print(
            f"  {name}: {ratios[0]:.2f} at sd {series[0]['rotation_sigma']}, "
            f"{ratios[-1]:.1f} at sd {series[-1]['rotation_sigma']}; "
            f"monotone {'yes' if rising else 'no'}; "
            f"largest single step x{biggest[0]:.1f} at sd {biggest[1]}"
        )

    print("\n  EXPLORATORY (H3 withdrawn): rotation error where calibration is lost.")
    for name, _ in GROUPS:
        series = sorted(
            (r for r in rows if r["group"] == name and r["usable"]),
            key=lambda r: r["rotation_sigma"],
        )
        broke = [r for r in series if r["significant_after_fdr"] and r["ratio"] > 1.0]
        if broke:
            first = broke[0]
            print(
                f"  {name}: first overconfident at sd {first['rotation_sigma']}, "
                f"RMS rotation error {first['rms_rotation_error']:.3f} rad"
            )
        else:
            print(f"  {name}: never loses calibration across this sweep")
    print()


def build_figure():
    """NEES against rotational noise, with the acceptance band."""
    rows = read_results("e3_nonlinearity")
    fig, axes = figure(nrows=1, ncols=2, size=(10.0, 4.2), sharey=True)

    for index, (name, _) in enumerate(GROUPS):
        ax = axes[index]
        series = sorted(
            (r for r in rows if r["group"] == name),
            key=lambda r: r["rotation_sigma"],
        )
        x = np.arange(len(series))
        ratio = np.array([r["ratio"] for r in series])
        ci_low = np.array([r["ci_low"] for r in series])
        ci_high = np.array([r["ci_high"] for r in series])

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
        "E3: overconfidence against rotational noise",
        color=INK,
        fontsize=12,
        x=0.02,
        ha="left",
    )
    return fig


def build_coverage_figure():
    """F2: coverage calibration curves across rotational noise levels."""
    rows = read_results("e3_nonlinearity")
    shown = [0.01, 0.06, 0.15, 0.30, 0.45]
    fig, axes = figure(nrows=1, ncols=2, size=(10.0, 4.4), sharey=True)

    for index, (name, _) in enumerate(GROUPS):
        ax = axes[index]
        ax.plot(
            [0, 1], [0, 1], color=INK_MUTED, linewidth=2.0, linestyle="--", label="ideal"
        )
        for colour, sigma in zip(NOISE_RAMP, shown):
            match = [
                r
                for r in rows
                if r["group"] == name and abs(r["rotation_sigma"] - sigma) < 1e-9
            ]
            if not match:
                continue
            row = match[0]
            ax.plot(
                row["coverage_nominal"],
                row["coverage_empirical"],
                marker="o",
                markersize=6,
                linewidth=2.0,
                color=colour,
                label=f"sd {sigma:g}",
            )
        ax.set_xlim(0.45, 1.02)
        ax.set_ylim(0.0, 1.02)
        label(
            ax,
            f"{name} - ellipsoid coverage by noise level",
            "nominal level",
            "empirical coverage" if index == 0 else "",
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncols=6,
        frameon=False,
        fontsize=8.5,
        labelcolor=INK_MUTED,
    )
    fig.suptitle(
        "F2: credible regions cover less than they claim as rotation grows",
        color=INK,
        fontsize=12,
        x=0.02,
        ha="left",
    )
    return fig


def figures() -> bool:
    written = True
    for builder, name in (
        (build_figure, "e3_nonlinearity"),
        (build_coverage_figure, "e3_coverage"),
    ):
        try:
            print(f"figure written to {save_figure(builder(), name)}")
        except ImportError as exc:
            print(f"{name} built but not written -- no usable renderer ({exc})")
            written = False
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()

    if args.figures_only:
        report_console(read_results("e3_nonlinearity"))
    else:
        run(args.runs)
    figures()
    return 0


if __name__ == "__main__":
    sys.exit(main())
