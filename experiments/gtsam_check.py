"""Cross-check against GTSAM: the same graphs, estimates and marginal covariances.

GTSAM is not a dependency and has no Windows build; GitHub Actions runs this on
Linux (.github/workflows/tests.yml).
"""

import sys

import gtsam
import numpy as np

from posetrust import se2, se3
from posetrust.covariance import covariance_matrix
from posetrust.optimizer import gauss_newton
from posetrust.simulate import NoiseModel, make_scenario, sample_graph

# GTSAM orders the SE(3) tangent rotation first; posetrust translation first.
ORDER = {se2: np.arange(3), se3: np.array([3, 4, 5, 0, 1, 2])}


def to_gtsam(lie, T):
    if lie is se2:
        return gtsam.Pose2(T[0, 2], T[1, 2], np.arctan2(T[1, 0], T[0, 0]))
    return gtsam.Pose3(T)


def check(lie, density: float, seed: int) -> tuple[float, float]:
    """Largest pose and relative marginal-covariance difference on one graph."""
    scenario = make_scenario(lie, n_poses=10, loop_density=density, seed=seed)
    noise = NoiseModel(np.full(lie.DOF, 0.03))
    graph = sample_graph(lie, scenario, noise, np.random.default_rng(seed))
    ours = gauss_newton(graph, list(scenario.truth))
    sigma = covariance_matrix(ours.information, anchor=0, dof=lie.DOF)

    order = ORDER[lie]
    between = gtsam.BetweenFactorPose2 if lie is se2 else gtsam.BetweenFactorPose3
    prior = gtsam.PriorFactorPose2 if lie is se2 else gtsam.PriorFactorPose3
    factors, initial = gtsam.NonlinearFactorGraph(), gtsam.Values()
    # A prior this tight stands in for posetrust's anchored first pose.
    tight = gtsam.noiseModel.Isotropic.Sigma(lie.DOF, 1e-9)
    factors.add(prior(0, to_gtsam(lie, scenario.truth[0]), tight))
    for f in graph.factors:
        model = gtsam.noiseModel.Gaussian.Information(f.information[np.ix_(order, order)])
        factors.add(between(f.i, f.j, to_gtsam(lie, f.measurement), model))
    for k, T in enumerate(scenario.truth):
        initial.insert(k, to_gtsam(lie, T))
    params = gtsam.LevenbergMarquardtParams()
    params.setRelativeErrorTol(1e-15)
    params.setAbsoluteErrorTol(1e-15)
    result = gtsam.LevenbergMarquardtOptimizer(factors, initial, params).optimize()
    marginals = gtsam.Marginals(factors, result)

    at = result.atPose2 if lie is se2 else result.atPose3
    back = np.argsort(order)
    pose_gap = covariance_gap = 0.0
    for k in range(1, 10):
        pose_gap = max(pose_gap, np.max(np.abs(at(k).matrix() - ours.poses[k])))
        theirs = marginals.marginalCovariance(k)[np.ix_(back, back)]
        block = sigma[k * lie.DOF : (k + 1) * lie.DOF, k * lie.DOF : (k + 1) * lie.DOF]
        covariance_gap = max(covariance_gap, np.max(np.abs(theirs - block)) / np.max(np.abs(block)))
    return pose_gap, covariance_gap


if __name__ == "__main__":
    agree = True
    for name, lie in (("SE(2)", se2), ("SE(3)", se3)):
        for density in (0.0, 0.5):
            poses, covariances = check(lie, density, seed=11)
            agree &= poses < 1e-6 and covariances < 1e-3
            print(
                f"{name} closures/pose {density}: max pose difference {poses:.1e}, "
                f"max relative covariance difference {covariances:.1e}"
            )
    sys.exit(0 if agree else 1)
