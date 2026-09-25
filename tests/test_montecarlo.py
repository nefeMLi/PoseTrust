"""Tests for the Monte Carlo harness and scenarios."""

from __future__ import annotations

import numpy as np
import pytest

from posetrust.optimize.optimizer import levenberg_marquardt
from posetrust.simulate import (
    NoiseModel,
    curved_trajectory,
    dead_reckon,
    loop_closure_edges,
    make_scenario,
    monte_carlo,
    odometry_edges,
    sample_graph,
)
from posetrust.stats import CONSISTENT, ConsistencyReport, classify

N_POSES = 6
N_RUNS = 150


def test_noise_model_information_inverts_covariance(lie):
    noise = NoiseModel(np.linspace(0.01, 0.05, lie.DOF))
    np.testing.assert_allclose(
        noise.information @ noise.covariance, np.eye(lie.DOF), rtol=0, atol=1e-12
    )


def test_noise_model_is_anisotropic(lie):
    sigma = np.linspace(0.01, 0.05, lie.DOF)
    noise = NoiseModel(sigma)
    draws = np.array([noise.sample(np.random.default_rng(k)) for k in range(4000)])
    np.testing.assert_allclose(draws.std(axis=0), sigma, rtol=0.1)


def test_curved_trajectory_actually_turns(lie):
    poses = curved_trajectory(lie, n_poses=5, turn=0.3)
    assert len(poses) == 5
    relative = lie.log(lie.compose(lie.inverse(poses[0]), poses[1]))
    assert abs(relative[lie.DOF - 1]) == pytest.approx(0.3, rel=1e-9)


def test_odometry_edges_form_a_chain():
    assert odometry_edges(4) == [(0, 1), (1, 2), (2, 3)]


@pytest.mark.parametrize("density", [0.0, 0.2, 0.5, 1.0])
def test_loop_closure_density_controls_edge_count(density):
    rng = np.random.default_rng(0)
    edges = loop_closure_edges(20, density, rng)
    assert len(edges) == round(density * 20)
    assert len(set(edges)) == len(edges)
    for i, j in edges:
        assert j - i >= 3, "closures must not duplicate odometry"


@pytest.mark.parametrize("n_poses,density", [(12, 0.05), (12, 0.1), (12, 0.8)])
def test_density_that_is_not_a_whole_count_is_refused(n_poses, density):
    with pytest.raises(ValueError, match="not a whole number"):
        loop_closure_edges(n_poses, density, np.random.default_rng(0))


def test_more_closures_than_candidate_pairs_is_refused():
    with pytest.raises(ValueError, match="pose pairs"):
        loop_closure_edges(5, 1.0, np.random.default_rng(0))


def test_denser_graphs_contain_sparser_ones():
    previous: set = set()
    for density in [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2]:
        edges = set(loop_closure_edges(20, density, np.random.default_rng(9)))
        assert previous <= edges
        previous = edges


def test_zero_density_is_odometry_only(lie):
    scenario = make_scenario(lie, n_poses=8, loop_density=0.0, seed=0)
    assert scenario.edges == odometry_edges(8)


def test_outlier_rate_marks_the_right_number_of_closures(lie):
    scenario = make_scenario(lie, n_poses=30, loop_density=1.0, outlier_rate=0.3, seed=0)
    closures = [e for e in scenario.edges if e not in odometry_edges(30)]
    assert len(scenario.outliers) == 9
    assert set(scenario.outliers) <= set(closures), "odometry must never be corrupted"
    for (i, j), b in scenario.outliers.items():
        assert b not in (i, j), "a false closure must claim a different place"


def test_outlier_rate_that_is_not_a_whole_count_is_refused(lie):
    with pytest.raises(ValueError, match="not a whole number"):
        make_scenario(lie, n_poses=10, loop_density=1.0, outlier_rate=0.05, seed=0)


def test_higher_outlier_rates_add_to_the_lower_rates_outliers(lie):
    previous: dict = {}
    for rate in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        scenario = make_scenario(
            lie, n_poses=20, loop_density=1.0, outlier_rate=rate, seed=3
        )
        assert len(scenario.outliers) == round(rate * 20)
        assert previous.items() <= scenario.outliers.items()
        previous = dict(scenario.outliers)


def test_noise_is_common_across_outlier_rates(lie):
    noise = NoiseModel(np.full(lie.DOF, 0.02))
    graphs = []
    for rate in (0.0, 0.3):
        scenario = make_scenario(
            lie, n_poses=20, loop_density=1.0, outlier_rate=rate, seed=3
        )
        graphs.append(
            (scenario, sample_graph(lie, scenario, noise, np.random.default_rng(1)))
        )
    (_, clean), (corrupt_scenario, corrupt) = graphs
    for a, b in zip(clean.factors, corrupt.factors):
        if (a.i, a.j) not in corrupt_scenario.outliers:
            np.testing.assert_array_equal(a.measurement, b.measurement)


def test_measurement_is_the_true_relative_pose_perturbed_by_the_noise(lie):
    scenario = make_scenario(lie, n_poses=6, loop_density=0.5, seed=0)
    noise = NoiseModel(np.full(lie.DOF, 0.01))

    graph = sample_graph(lie, scenario, noise, np.random.default_rng(4))
    replay = np.random.default_rng(4)
    for factor in graph.factors:
        eps = noise.sample(replay)
        relative = lie.compose(
            lie.inverse(scenario.truth[factor.i]), scenario.truth[factor.j]
        )
        expected = lie.compose(relative, lie.exp(eps))
        np.testing.assert_allclose(factor.measurement, expected, rtol=0, atol=1e-12)
        # and the residual at truth is exactly the negated noise draw
        np.testing.assert_allclose(
            graph.residual(factor, scenario.truth), -eps, rtol=0, atol=1e-12
        )


def test_zero_noise_scale_is_rejected(lie):
    with pytest.raises(ValueError, match="strictly positive"):
        NoiseModel(np.zeros(lie.DOF))


def test_outliers_produce_wrong_measurements(lie):
    scenario = make_scenario(
        lie, n_poses=20, loop_density=1.0, outlier_rate=0.5, seed=3
    )
    sigma = 1e-4
    graph = sample_graph(
        lie, scenario, NoiseModel(np.full(lie.DOF, sigma)), np.random.default_rng(1)
    )
    inliers, outliers = [], []
    for factor in graph.factors:
        residual = np.linalg.norm(graph.residual(factor, scenario.truth))
        (outliers if (factor.i, factor.j) in scenario.outliers else inliers).append(residual)

    assert max(inliers) < 10 * sigma, "a true edge must be explained by the truth"
    assert min(outliers) > 0.1, "a false match must be grossly inconsistent"


def test_dead_reckoning_composes_the_odometry_measurements(lie):
    scenario = make_scenario(lie, n_poses=6, loop_density=0.0, seed=0)
    graph = sample_graph(
        lie, scenario, NoiseModel(np.full(lie.DOF, 0.01)), np.random.default_rng(0)
    )
    measurements = {(f.i, f.j): f.measurement for f in graph.factors}

    expected = [lie.exp(np.zeros(lie.DOF))]
    for k in range(5):
        expected.append(lie.compose(expected[-1], measurements[(k, k + 1)]))
    for got, want in zip(dead_reckon(lie, graph, 6), expected):
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


def test_monte_carlo_is_reproducible(lie):
    scenario = make_scenario(lie, n_poses=5, loop_density=0.4, seed=0, turn=0.05)
    noise = NoiseModel(np.full(lie.DOF, 0.01))
    a = monte_carlo(lie, scenario, noise, n_runs=8, seed=42)
    b = monte_carlo(lie, scenario, noise, n_runs=8, seed=42)
    c = monte_carlo(lie, scenario, noise, n_runs=8, seed=43)
    np.testing.assert_array_equal(a.nees_full, b.nees_full)
    assert not np.allclose(a.nees_full, c.nees_full)


def test_monte_carlo_shapes_and_gauge(lie):
    scenario = make_scenario(lie, n_poses=5, loop_density=0.4, seed=0, turn=0.05)
    result = monte_carlo(
        lie, scenario, NoiseModel(np.full(lie.DOF, 0.01)), n_runs=6, seed=0
    )
    dof = lie.DOF
    assert result.errors.shape == (6, 5, dof)
    assert result.marginals.shape == (6, 5, dof, dof)
    assert result.free_dof == 4 * dof
    assert result.converged.all()

    # the anchor is held at truth, so it has no error and no uncertainty
    np.testing.assert_allclose(result.errors[:, 0, :], 0.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.marginals[:, 0], 0.0, rtol=0, atol=0.0)


def test_near_linear_gaussian_is_consistent(lie):
    scenario = make_scenario(lie, n_poses=N_POSES, loop_density=0.5, seed=1, turn=0.05)
    noise = NoiseModel(np.full(lie.DOF, 1e-3))
    result = monte_carlo(lie, scenario, noise, n_runs=N_RUNS, seed=7)

    assert result.converged.all()
    report = ConsistencyReport(result.nees_full, result.free_dof, alpha=0.01)
    assert report.verdict == CONSISTENT, (
        f"mean NEES {report.mean:.3f} outside {report.acceptance} "
        f"for dof {result.free_dof}"
    )
    assert report.mean / result.free_dof == pytest.approx(1.0, abs=0.06)


def test_large_rotational_noise_becomes_overconfident():
    from posetrust.lie import se2

    scenario = make_scenario(se2, n_poses=8, loop_density=0.25, seed=1, turn=0.3)
    result = monte_carlo(
        se2, scenario, NoiseModel(np.array([0.02, 0.02, 0.5])),
        n_runs=60, seed=11, solver=levenberg_marquardt,
    )

    # Every run should converge; drops here would mean an LM regression.
    assert result.converged.all()
    values = result.nees_full
    assert classify(values, result.free_dof) != CONSISTENT
    assert values.mean() / result.free_dof > 2.0


def test_a_graph_too_short_to_close_accepts_only_zero_closures():
    assert loop_closure_edges(3, density=0.0, rng=np.random.default_rng(0)) == []
    with pytest.raises(ValueError, match="pose pairs"):
        loop_closure_edges(3, density=1.0, rng=np.random.default_rng(0))


def test_result_accessors_expose_per_pose_slices(lie):
    scenario = make_scenario(lie, n_poses=5, loop_density=0.4, seed=0, turn=0.05)
    result = monte_carlo(
        lie, scenario, NoiseModel(np.full(lie.DOF, 0.01)), n_runs=7, seed=0
    )
    assert result.n_runs == 7
    assert result.pose_errors(3).shape == (7, lie.DOF)
    assert result.pose_marginals(3).shape == (7, lie.DOF, lie.DOF)
    np.testing.assert_array_equal(result.pose_errors(3), result.errors[:, 3, :])


def test_odometry_initialization_reaches_the_same_optimum(lie):
    scenario = make_scenario(lie, n_poses=5, loop_density=0.6, seed=2, turn=0.05)
    noise = NoiseModel(np.full(lie.DOF, 5e-3))
    from_truth = monte_carlo(lie, scenario, noise, n_runs=12, seed=5)
    from_odometry = monte_carlo(
        lie, scenario, noise, n_runs=12, seed=5, initialize="odometry"
    )
    assert from_odometry.converged.all()
    np.testing.assert_allclose(
        from_odometry.nees_full, from_truth.nees_full, rtol=1e-6
    )

