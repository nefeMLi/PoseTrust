"""Core checks: the Jacobians, a case where the covariance is known to be honest, and GTSAM."""

import numpy as np
import pytest

from posetrust import se2, se3
from posetrust.covariance import covariance_matrix
from posetrust.optimizer import gauss_newton
from posetrust.simulate import NoiseModel, make_scenario, monte_carlo, sample_graph
from posetrust.stats import CONSISTENT, consistency

groups = pytest.mark.parametrize("lie", [se2, se3], ids=["SE2", "SE3"])


@groups
def test_factor_jacobians(lie):
    """Analytic residual Jacobians against central differences, away from the truth."""
    scenario = make_scenario(lie, n_poses=8, loop_density=0.5, seed=0)
    graph = sample_graph(lie, scenario, NoiseModel(np.full(lie.DOF, 0.1)), np.random.default_rng(0))
    rng = np.random.default_rng(1)
    poses = [lie.compose(T, lie.exp(0.3 * rng.normal(size=lie.DOF))) for T in scenario.truth]
    h = 1e-6
    for factor in graph.factors:
        for which, J in zip((factor.i, factor.j), graph.factor_jacobians(factor, poses)):
            numeric = np.zeros((lie.DOF, lie.DOF))
            for k in range(lie.DOF):
                step = np.zeros(lie.DOF)
                step[k] = h
                plus, minus = list(poses), list(poses)
                plus[which] = lie.compose(poses[which], lie.exp(step))
                minus[which] = lie.compose(poses[which], lie.exp(-step))
                numeric[:, k] = (graph.residual(factor, plus) - graph.residual(factor, minus)) / (2 * h)
            np.testing.assert_allclose(J, numeric, rtol=0, atol=1e-6)


@groups
def test_near_linear_case_is_consistent(lie):
    """Tiny noise and almost no turning: the reported covariance should be honest."""
    scenario = make_scenario(lie, n_poses=6, loop_density=0.5, seed=1, turn=0.05)
    result = monte_carlo(lie, scenario, NoiseModel(np.full(lie.DOF, 1e-3)), n_runs=150, seed=7)
    report = consistency(result, alpha=0.01)
    assert report.verdict == CONSISTENT
    assert report.nees.mean / report.nees.dof == pytest.approx(1.0, abs=0.06)


@groups
@pytest.mark.parametrize("density", [0.0, 0.5])
def test_agrees_with_gtsam(lie, density):
    """Same graph in GTSAM: the same estimate and marginal covariances. GTSAM has no Windows build,
    so this skips locally and runs on GitHub Actions."""
    gtsam = pytest.importorskip("gtsam")
    scenario = make_scenario(lie, n_poses=10, loop_density=density, seed=11)
    graph = sample_graph(lie, scenario, NoiseModel(np.full(lie.DOF, 0.03)), np.random.default_rng(11))
    ours = gauss_newton(graph, list(scenario.truth))
    sigma = covariance_matrix(ours.information, anchor=0, dof=lie.DOF)

    # GTSAM orders the SE(3) tangent rotation first; posetrust translation first.
    order = np.arange(3) if lie is se2 else np.array([3, 4, 5, 0, 1, 2])
    pose = (lambda T: gtsam.Pose2(T[0, 2], T[1, 2], np.arctan2(T[1, 0], T[0, 0]))) if lie is se2 else gtsam.Pose3
    between = gtsam.BetweenFactorPose2 if lie is se2 else gtsam.BetweenFactorPose3
    prior = gtsam.PriorFactorPose2 if lie is se2 else gtsam.PriorFactorPose3
    factors, initial = gtsam.NonlinearFactorGraph(), gtsam.Values()
    # A prior this tight stands in for posetrust's anchored first pose.
    factors.add(prior(0, pose(scenario.truth[0]), gtsam.noiseModel.Isotropic.Sigma(lie.DOF, 1e-9)))
    for f in graph.factors:
        model = gtsam.noiseModel.Gaussian.Information(f.information[np.ix_(order, order)])
        factors.add(between(f.i, f.j, pose(f.measurement), model))
    for k, T in enumerate(scenario.truth):
        initial.insert(k, pose(T))
    params = gtsam.LevenbergMarquardtParams()
    params.setRelativeErrorTol(1e-15)
    params.setAbsoluteErrorTol(1e-15)
    result = gtsam.LevenbergMarquardtOptimizer(factors, initial, params).optimize()
    marginals = gtsam.Marginals(factors, result)

    at, back, d = (result.atPose2 if lie is se2 else result.atPose3), np.argsort(order), lie.DOF
    for k in range(1, 10):
        np.testing.assert_allclose(at(k).matrix(), ours.poses[k], rtol=0, atol=1e-6)
        block = sigma[k * d : (k + 1) * d, k * d : (k + 1) * d]
        theirs = marginals.marginalCovariance(k)[np.ix_(back, back)]
        np.testing.assert_allclose(theirs, block, rtol=0, atol=1e-3 * np.max(np.abs(block)))
