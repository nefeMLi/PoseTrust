"""Tests for the robust kernels and back-ends."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import short_trajectory
from scipy.stats import chi2

from posetrust.graph import PoseGraph
from posetrust.optimize.optimizer import gauss_newton
from posetrust.optimize.robust import (
    Cauchy,
    GemanMcClure,
    Huber,
    SwitchableConstraints,
    Trivial,
    chi2_threshold,
    factor_weights,
    graduated_non_convexity,
    irls,
    loop_closure_indices,
    robust_cost,
    squared_residuals,
)

KERNELS = [
    Trivial(),
    Huber(1.5),
    Cauchy(2.0),
    SwitchableConstraints(3.0),
    GemanMcClure(2.0, 1.0),
    GemanMcClure(2.0, 7.0),
]
KERNEL_IDS = ["trivial", "huber", "cauchy", "switchable", "gm_mu1", "gm_mu7"]


@pytest.mark.parametrize("kernel", KERNELS, ids=KERNEL_IDS)
def test_weight_is_the_derivative_of_cost(kernel):
    s = np.logspace(-6, 3, 2000)
    h = s * 1e-6
    numeric = (kernel.cost(s + h) - kernel.cost(s - h)) / (2 * h)
    analytic = kernel.weight(s)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-9)


@pytest.mark.parametrize("kernel", KERNELS, ids=KERNEL_IDS)
def test_zero_residual_has_full_weight(kernel):
    assert kernel.weight(np.array([0.0]))[0] == pytest.approx(1.0)


@pytest.mark.parametrize("kernel", KERNELS, ids=KERNEL_IDS)
def test_cost_increases_and_weight_never_does(kernel):
    s = np.logspace(-6, 3, 2000)
    assert np.all(np.diff(kernel.cost(s)) >= -1e-12)
    assert np.all(np.diff(kernel.weight(s)) <= 1e-12)


@pytest.mark.parametrize(
    "kernel,join", [(Huber(1.5), 1.5**2), (SwitchableConstraints(3.0), 3.0)]
)
def test_piecewise_kernels_are_continuous(kernel, join):
    lo, hi = np.array([join * (1 - 1e-9)]), np.array([join * (1 + 1e-9)])
    assert kernel.cost(hi)[0] == pytest.approx(kernel.cost(lo)[0], abs=1e-7)
    assert kernel.weight(hi)[0] == pytest.approx(kernel.weight(lo)[0], abs=1e-7)


def test_redescending_kernels_have_bounded_cost():
    huge = np.array([1e12])
    assert Cauchy(2.0).weight(huge)[0] < 1e-11
    assert SwitchableConstraints(3.0).weight(huge)[0] < 1e-11
    assert GemanMcClure(2.0).weight(huge)[0] < 1e-11
    # Huber is convex: it down-weights but never rejects
    assert Huber(1.5).weight(huge)[0] > 0.0


def test_chi2_threshold_matches_the_distribution():
    for dof in (3, 6):
        assert chi2_threshold(dof, 0.95) == pytest.approx(chi2.ppf(0.95, dof))


def test_loop_closure_indices_exclude_odometry(lie):
    graph = PoseGraph(lie)
    for T in short_trajectory(lie, n_poses=6):
        graph.add_pose(T)
    identity, omega = lie.exp(np.zeros(lie.DOF)), np.eye(lie.DOF)
    for k in range(5):
        graph.add_factor(k, k + 1, identity, omega)
    graph.add_factor(0, 5, identity, omega)
    graph.add_factor(1, 4, identity, omega)

    np.testing.assert_array_equal(loop_closure_indices(graph), [5, 6])


def test_trivial_kernel_reproduces_plain_least_squares(outlier_free_graph):
    graph, _, start = outlier_free_graph
    plain = gauss_newton(graph, start, anchor=0)
    reweighted = irls(graph, start, Trivial(), anchor=0)
    for a, b in zip(plain.poses, reweighted.poses):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-9)
    np.testing.assert_allclose(plain.information, reweighted.information, rtol=0, atol=1e-8)


def test_weights_reach_the_information_matrix(outlier_free_graph):
    graph, truth, _ = outlier_free_graph
    weights = np.full(len(graph.factors), 0.25)
    H_plain, _ = graph.linearize(truth)
    H_weighted, _ = graph.linearize(truth, weights=weights)
    np.testing.assert_allclose(H_weighted, 0.25 * H_plain, rtol=0, atol=1e-10)


def test_robust_cost_leaves_unmasked_factors_quadratic(outlier_free_graph):
    graph, truth, _ = outlier_free_graph
    s = squared_residuals(graph, truth)
    mask = np.zeros(len(graph.factors), dtype=bool)
    assert robust_cost(Cauchy(1.0), s, mask) == pytest.approx(float(s.sum()))


def test_masked_factors_keep_unit_weight(outlier_free_graph):
    graph, _, _ = outlier_free_graph
    s = np.full(len(graph.factors), 1e6)
    mask = np.zeros(len(graph.factors), dtype=bool)
    mask[loop_closure_indices(graph)] = True
    weights = factor_weights(Cauchy(1.0), s, mask)
    assert np.all(weights[~mask] == 1.0)
    assert np.all(weights[mask] < 1e-5)


def outlier_scenario(lie, rate=0.3, seed=4):
    """A graph with planted false loop closures, started from dead reckoning."""
    from posetrust.simulate import NoiseModel, dead_reckon, make_scenario, sample_graph

    scenario = make_scenario(
        lie, n_poses=10, loop_density=1.0, outlier_rate=rate, seed=seed, turn=0.25
    )
    graph = sample_graph(
        lie, scenario, NoiseModel(np.full(lie.DOF, 0.05)), np.random.default_rng(7)
    )
    start = dead_reckon(lie, graph, 10)
    start[0] = scenario.truth[0]
    return scenario, graph, start


def rms_error(lie, poses, truth):
    return float(
        np.sqrt(
            np.mean(
                [
                    np.sum(lie.log(lie.compose(lie.inverse(a), b)) ** 2)
                    for a, b in zip(poses, truth)
                ]
            )
        )
    )


def test_redescending_kernels_survive_outliers_and_plain_least_squares_does_not(lie):
    scenario, graph, start = outlier_scenario(lie)
    closures = loop_closure_indices(graph)
    threshold = chi2_threshold(lie.DOF, 0.95)
    delta = np.sqrt(threshold)

    plain = rms_error(lie, gauss_newton(graph, start, anchor=0).poses, scenario.truth)
    cauchy = rms_error(
        lie, irls(graph, start, Cauchy(delta), robust_factors=closures).poses, scenario.truth
    )
    switchable = rms_error(
        lie,
        irls(graph, start, SwitchableConstraints(threshold), robust_factors=closures).poses,
        scenario.truth,
    )
    gnc = rms_error(
        lie,
        graduated_non_convexity(graph, start, c=delta, robust_factors=closures).poses,
        scenario.truth,
    )

    assert plain > 4 * cauchy, "plain least squares should be dragged off by outliers"
    for name, value in [("cauchy", cauchy), ("switchable", switchable), ("gnc", gnc)]:
        assert value < 0.3, f"{name} failed to reject the outliers: rms {value:.3f}"


def test_graduated_non_convexity_identifies_the_planted_outliers(lie):
    scenario, graph, start = outlier_scenario(lie)
    closures = loop_closure_indices(graph)
    delta = np.sqrt(chi2_threshold(lie.DOF, 0.95))

    result = graduated_non_convexity(graph, start, c=delta, robust_factors=closures)
    s = squared_residuals(graph, result.poses)
    mask = np.zeros(len(graph.factors), dtype=bool)
    mask[closures] = True
    weights = factor_weights(GemanMcClure(delta, 1.0), s, mask)

    planted = np.array([(f.i, f.j) in scenario.outliers for f in graph.factors])
    assert planted.any(), "scenario should contain outliers"
    assert weights[planted].mean() < 0.05
    assert weights[~planted].mean() > 0.5


def test_robustness_costs_nothing_when_there_are_no_outliers(lie):
    scenario, graph, start = outlier_scenario(lie, rate=0.0)
    closures = loop_closure_indices(graph)
    delta = np.sqrt(chi2_threshold(lie.DOF, 0.95))

    plain = rms_error(lie, gauss_newton(graph, start, anchor=0).poses, scenario.truth)
    robust = rms_error(
        lie, irls(graph, start, Cauchy(delta), robust_factors=closures).poses, scenario.truth
    )
    assert robust == pytest.approx(plain, rel=0.25)


def test_huber_restores_accuracy_without_restoring_calibration():
    from posetrust.lie import se2
    from posetrust.simulate import NoiseModel, make_scenario, monte_carlo
    from posetrust.stats import CONSISTENT, ConsistencyReport

    scenario = make_scenario(
        se2, n_poses=10, loop_density=1.0, outlier_rate=0.2, seed=4, turn=0.25
    )
    noise = NoiseModel(np.full(3, 0.05))
    threshold = chi2_threshold(3, 0.95)
    delta = np.sqrt(threshold)

    def run(kernel):
        solver = lambda g, p, anchor=0: irls(
            g, p, kernel, anchor=anchor, robust_factors=loop_closure_indices(g)
        )
        result = monte_carlo(se2, scenario, noise, n_runs=40, seed=3, solver=solver)
        ok = result.converged
        rms = float(np.sqrt(np.mean(result.errors[ok] ** 2)))
        report = ConsistencyReport(result.nees_full[ok], result.free_dof)
        return rms, report

    huber_rms, huber_report = run(Huber(delta))
    cauchy_rms, cauchy_report = run(Cauchy(delta))

    # Huber looks fine on accuracy ...
    assert huber_rms < 2.0 * cauchy_rms
    # ... while its covariance is still overconfident
    assert huber_report.verdict != CONSISTENT
    assert huber_report.mean / huber_report.dof > 1.5
    # the redescending kernel restores both
    assert cauchy_report.verdict == CONSISTENT



def test_slow_reweighting_is_not_recorded_as_failure(lie):
    from posetrust.simulate import NoiseModel, dead_reckon, make_scenario, sample_graph

    scenario = make_scenario(
        lie, n_poses=20, loop_density=1.0, outlier_rate=0.3, seed=300, turn=0.25
    )
    graph = sample_graph(
        lie, scenario, NoiseModel(np.full(lie.DOF, 0.05)), np.random.default_rng(0)
    )
    start = dead_reckon(lie, graph, 20)
    start[0] = scenario.truth[0]
    delta = np.sqrt(chi2_threshold(lie.DOF, 0.95))
    closures = loop_closure_indices(graph)

    result = irls(graph, start, Huber(delta), robust_factors=closures)
    assert result.converged
    assert result.iterations > 50, "no longer a slow case; the test lost its point"
    # and the stationary point is a genuine one, not a tolerance artefact
    s = squared_residuals(graph, result.poses)
    mask = np.zeros(len(graph.factors), dtype=bool)
    mask[closures] = True
    _, b = graph.linearize(result.poses, weights=factor_weights(Huber(delta), s, mask))
    free = np.ones(len(b), dtype=bool)
    free[: lie.DOF] = False
    assert np.linalg.norm(b[free]) < 1e-8
