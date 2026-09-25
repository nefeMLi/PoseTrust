"""Core checks: Lie maths, Jacobians, covariances and a calibrated case."""

import numpy as np
import pytest
from scipy.linalg import expm

import se2
import se3
from covariance import cholesky_factor, selected_inverse
from optimizer import gauss_newton, levenberg_marquardt
from simulate import (
    NoiseModel,
    loop_closure_edges,
    make_scenario,
    monte_carlo,
    sample_graph,
)
from stats import CONSISTENT, consistency

groups = pytest.mark.parametrize("lie", [se2, se3], ids=["SE2", "SE3"])


def hat(lie, xi):
    """Lie algebra matrix of xi, built without the library."""
    if lie is se2:
        x, y, t = xi
        return np.array([[0.0, -t, x], [t, 0.0, y], [0.0, 0.0, 0.0]])
    v, (a, b, c) = xi[:3], xi[3:]
    M = np.zeros((4, 4))
    M[:3, :3] = [[0.0, -c, b], [c, 0.0, -a], [-b, a, 0.0]]
    M[:3, 3] = v
    return M


def random_xi(lie, rng, angle):
    xi = rng.normal(size=lie.DOF)
    rotation = xi[lie.TRANSLATION_DOF:]
    xi[lie.TRANSLATION_DOF:] = angle * rotation / np.linalg.norm(rotation)
    return xi


@groups
@pytest.mark.parametrize("angle", [1e-9, 1e-4, 0.05, 1.0, 3.0, np.pi - 1e-6])
def test_exp_and_log(lie, angle):
    rng = np.random.default_rng(0)
    for _ in range(20):
        xi = random_xi(lie, rng, angle)
        np.testing.assert_allclose(lie.exp(xi), expm(hat(lie, xi)), rtol=0, atol=1e-12)
        np.testing.assert_allclose(lie.log(lie.exp(xi)), xi, rtol=0, atol=1e-8)


@groups
@pytest.mark.parametrize("angle", [1e-6, 0.05, 1.0, 2.5])
def test_right_jacobian(lie, angle):
    xi, h = random_xi(lie, np.random.default_rng(1), angle), 1e-6
    T_inv = lie.inverse(lie.exp(xi))
    numeric = np.zeros((lie.DOF, lie.DOF))
    for k in range(lie.DOF):
        d = np.zeros(lie.DOF)
        d[k] = h
        plus = lie.log(lie.compose(T_inv, lie.exp(xi + d)))
        minus = lie.log(lie.compose(T_inv, lie.exp(xi - d)))
        numeric[:, k] = (plus - minus) / (2 * h)
    np.testing.assert_allclose(lie.right_jacobian(xi), numeric, rtol=0, atol=1e-7)


@groups
def test_factor_jacobians(lie):
    scenario = make_scenario(lie, n_poses=8, loop_density=0.5, seed=0)
    graph = sample_graph(lie, scenario, NoiseModel(np.full(lie.DOF, 0.1)), np.random.default_rng(0))
    rng = np.random.default_rng(1)
    poses = [lie.compose(T, lie.exp(0.3 * rng.normal(size=lie.DOF))) for T in scenario.truth]
    h = 1e-6
    for factor in graph.factors:
        analytic = graph.factor_jacobians(factor, poses)
        for which, J in zip((factor.i, factor.j), analytic):
            numeric = np.zeros((lie.DOF, lie.DOF))
            for k in range(lie.DOF):
                e = np.zeros(lie.DOF)
                e[k] = h
                plus, minus = list(poses), list(poses)
                plus[which] = lie.compose(poses[which], lie.exp(e))
                minus[which] = lie.compose(poses[which], lie.exp(-e))
                numeric[:, k] = (graph.residual(factor, plus) - graph.residual(factor, minus)) / (2 * h)
            np.testing.assert_allclose(J, numeric, rtol=0, atol=1e-6)


def test_selected_inverse_matches_dense_inverse():
    rng = np.random.default_rng(2)
    for n in (3, 12, 30):
        A = rng.normal(size=(n, n))
        H = A @ A.T + n * np.eye(n)
        np.testing.assert_allclose(selected_inverse(cholesky_factor(H)), np.linalg.inv(H), rtol=1e-10, atol=1e-12)


@groups
def test_levenberg_marquardt_reaches_the_gauss_newton_optimum(lie):
    scenario = make_scenario(lie, n_poses=10, loop_density=0.5, seed=3)
    graph = sample_graph(lie, scenario, NoiseModel(np.full(lie.DOF, 0.05)), np.random.default_rng(3))
    gn = gauss_newton(graph, list(scenario.truth))
    lm = levenberg_marquardt(graph, list(scenario.truth))
    assert gn.converged and lm.converged
    for a, b in zip(gn.poses, lm.poses):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-9)


@groups
def test_near_linear_case_is_consistent(lie):
    scenario = make_scenario(lie, n_poses=6, loop_density=0.5, seed=1, turn=0.05)
    result = monte_carlo(lie, scenario, NoiseModel(np.full(lie.DOF, 1e-3)), n_runs=150, seed=7)
    report = consistency(result, lie, alpha=0.01)
    assert report.verdict == CONSISTENT
    assert report.nees.mean / report.nees.dof == pytest.approx(1.0, abs=0.06)


def test_sweep_conditions_are_exact_and_nested():
    with pytest.raises(ValueError, match="not a whole number"):
        loop_closure_edges(12, 0.05, np.random.default_rng(0))
    sparse = set(loop_closure_edges(20, 0.2, np.random.default_rng(9)))
    dense = set(loop_closure_edges(20, 0.8, np.random.default_rng(9)))
    assert len(sparse) == 4 and len(dense) == 16 and sparse <= dense

    previous = {}
    for rate in (0.0, 0.1, 0.2, 0.3):
        outliers = make_scenario(se2, n_poses=20, loop_density=1.0, outlier_rate=rate, seed=3).outliers
        assert len(outliers) == round(rate * 20)
        assert previous.items() <= outliers.items()
        previous = dict(outliers)
