"""Calibration against rotational noise, in SE(2) and SE(3), on eight loop-closure layouts."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

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
from posetrust import se2, se3
from posetrust.optimizer import levenberg_marquardt
from posetrust.simulate import NoiseModel, make_scenario, monte_carlo
from posetrust.stats import ConsistencyReport

GROUPS = [("SE(2)", se2), ("SE(3)", se3)]
ROTATION_NOISE = [0.01, 0.03, 0.06, 0.10, 0.15, 0.22, 0.30, 0.45]
COVERAGE_LEVELS = (0.5, 0.75, 0.9, 0.95, 0.99)
TRANSLATION_NOISE = 0.02
N_POSES, LOOP_DENSITY, TURN = 10, 0.3, 0.25
SEED = 200  # the original graph
GRAPH_SEEDS = [200, 300, 400, 500, 600, 700, 800, 900]


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


def report(rows) -> None:
    print("\nRotational noise")
    for name, _ in GROUPS:
        mine = [r for r in rows if r["group"] == name]
        first = {
            seed: min((r["rotation_sigma"] for r in mine if r["graph"] == seed and overconfident(r)), default=None)
            for seed in GRAPH_SEEDS
        }
        conservative = sum(r["usable"] and r["significant_after_fdr"] and r["ratio"] < 1 for r in mine)
        print(
            f"  {name}: overconfident from {first[SEED]} rad on the original graph, "
            f"from {min(first.values())}-{max(first.values())} rad across layouts; conservative anywhere: {conservative}"
        )
    r = next(r for r in rows if r["graph"] == SEED and r["group"] == "SE(3)" and r["rotation_sigma"] == 0.22)
    # Coverage is of the joint region for the whole trajectory (full-state NEES), not of single poses.
    print(
        f"  SE(3) at 0.22 rad: the 95% joint region contains the true trajectory in {r['coverage_empirical'][3]:.1%} of runs\n"
    )


def plot(rows) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, layout="constrained")
    x = np.arange(len(ROTATION_NOISE))
    for ax, (name, _) in zip(axes, GROUPS):
        series = {}
        for seed in GRAPH_SEEDS:
            mine = {r["rotation_sigma"]: r for r in rows if r["graph"] == seed and r["group"] == name}
            series[seed] = [mine[s]["ratio"] if mine[s]["usable"] else np.nan for s in ROTATION_NOISE]
        layout_lines(ax, series, x, SEED, "tab:blue")
        ax.set_xticks(x, [f"{s:g}" for s in ROTATION_NOISE])
        ax.set(title=name, xlabel="rotational noise (rad)")
    axes[0].set_ylabel("mean NEES / dof")
    axes[0].legend(frameon=False)
    fig.suptitle("Where calibration breaks depends on the layout")
    save_figure(fig, "rotational_noise")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()
    if args.figures_only:
        rows = read_results("rotational_noise")
    else:
        tasks = [(g, s, args.runs, seed) for seed in GRAPH_SEEDS for g in range(len(GROUPS)) for s in ROTATION_NOISE]
        rows = list(parallel(condition, tasks))
        mark_fdr(rows, by=("graph",))  # within each layout's sweep
        write_results(rows, "rotational_noise")
    report(rows)
    plot(rows)
