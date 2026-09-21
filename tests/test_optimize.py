"""Tests for the manifold optimizers and gauge handling.

The exact-recovery test here is the precursor to the E1 validation gate: with
noise-free measurements the MAP estimate IS the ground truth, so anything less
than machine-precision recovery means something upstream is wrong.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import build_graph, short_trajectory, perturbed

from posetrust.optimize.gauge import free_mask
from posetrust.optimize.optimizer import gauss_newton, levenberg_marquardt, solve_step

SOLVERS = [gauss_newton, levenberg_marquardt]
SOLVER_IDS = ["gauss_newton", "levenberg_marquardt"]


@pytest.mark.parametrize("solver", SOLVERS, ids=SOLVER_IDS)
def test_exact_recovery_from_noise_free_measurements(lie, solver) -> None:
    """Known answer: perfect measurements must reproduce the true trajectory.

    The anchor starts at its true value, so the gauge is pinned to the truth
    and the recovered poses must match exactly rather than up to a transform.
    """
    truth = short_trajectory(lie)
    graph = build_graph(lie, truth)
    result = solver(graph, perturbed(lie, truth, sigma=0.25), anchor=0)

    assert result.converged
    assert result.chi2 < 1e-20
    for got, want in zip(result.poses, truth):
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


@pytest.mark.parametrize("solver", SOLVERS, ids=SOLVER_IDS)
def test_anchor_never_moves(lie, solver) -> None:
    truth = short_trajectory(lie)
    graph = build_graph(lie, truth)
    start = perturbed(lie, truth, sigma=0.25)
    result = solver(graph, start, anchor=0)
    np.testing.assert_allclose(result.poses[0], start[0], rtol=0, atol=1e-15)


@pytest.mark.parametrize("solver", SOLVERS, ids=SOLVER_IDS)
def test_converges_with_noisy_measurements(lie, solver) -> None:
    """With noise the optimum is not the truth, but it must still be a minimum."""
    truth = short_trajectory(lie)
    graph = build_graph(lie, truth, noise=0.02, seed=7)
    start = perturbed(lie, truth, sigma=0.15)

    result = solver(graph, start, anchor=0)
    assert result.converged
    assert result.chi2 < graph.chi2(start)

    # at a true minimum the gauge-fixed gradient vanishes
    _, b = graph.linearize(result.poses)
    free = np.ones(len(truth) * lie.DOF, dtype=bool)
    free[: lie.DOF] = False
    assert np.linalg.norm(b[free]) < 1e-8


def test_both_solvers_reach_the_same_optimum(lie) -> None:
    truth = short_trajectory(lie)
    graph = build_graph(lie, truth, noise=0.02, seed=3)
    start = perturbed(lie, truth, sigma=0.15)

    gn = gauss_newton(graph, start, anchor=0)
    lm = levenberg_marquardt(graph, start, anchor=0)
    assert abs(gn.chi2 - lm.chi2) < 1e-12 * max(1.0, abs(gn.chi2))
    for a, b in zip(gn.poses, lm.poses):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-9)


def test_anchored_system_is_positive_definite(lie) -> None:
    """Gauge fixing must leave a system that is actually invertible.

    A singular H would make the reported covariance meaningless, which is the
    failure this whole project is built to detect.
    """
    truth = short_trajectory(lie)
    graph = build_graph(lie, truth, noise=0.01, seed=5)
    result = gauss_newton(graph, perturbed(lie, truth, sigma=0.1), anchor=0)

    free = np.ones(len(truth) * lie.DOF, dtype=bool)
    free[: lie.DOF] = False
    Hf = result.information[np.ix_(free, free)]
    assert np.linalg.eigvalsh(Hf).min() > 0.0


def test_levenberg_marquardt_decreases_chi2_monotonically(lie) -> None:
    """LM only accepts a step that improves the objective."""
    truth = short_trajectory(lie)
    graph = build_graph(lie, truth, noise=0.05, seed=9)
    start = perturbed(lie, truth, sigma=0.6)

    chi2 = graph.chi2(start)
    poses = start
    for _ in range(12):
        result = levenberg_marquardt(graph, poses, anchor=0, max_iterations=1)
        assert result.chi2 <= chi2 + 1e-12
        chi2, poses = result.chi2, result.poses


def test_solve_step_leaves_anchor_block_zero(lie) -> None:
    truth = short_trajectory(lie, n_poses=5)
    graph = build_graph(lie, truth)
    H, b = graph.linearize(perturbed(lie, truth, sigma=0.2))
    free = np.ones(len(truth) * lie.DOF, dtype=bool)
    free[: lie.DOF] = False

    delta = solve_step(H, b, free)
    np.testing.assert_allclose(delta[: lie.DOF], np.zeros(lie.DOF), rtol=0, atol=0.0)


def test_recovery_is_independent_of_which_pose_is_anchored(lie) -> None:
    """Anchoring is a gauge choice: it must not change the graph's shape.

    Relative poses are gauge-invariant even though absolute ones are not, so
    this is the property that actually has to hold.
    """
    truth = short_trajectory(lie)
    graph = build_graph(lie, truth, noise=0.02, seed=4)
    start = perturbed(lie, truth, sigma=0.1)

    first = gauss_newton(graph, start, anchor=0)
    third = gauss_newton(graph, start, anchor=3)
    assert abs(first.chi2 - third.chi2) < 1e-9 * max(1.0, abs(first.chi2))

    for k in range(len(truth) - 1):
        rel_a = lie.compose(lie.inverse(first.poses[k]), first.poses[k + 1])
        rel_b = lie.compose(lie.inverse(third.poses[k]), third.poses[k + 1])
        np.testing.assert_allclose(rel_a, rel_b, rtol=0, atol=1e-8)


def test_invalid_anchor_is_rejected(lie) -> None:
    """A bad anchor must fail loudly, not silently mask the wrong block."""
    with pytest.raises(IndexError, match="anchor"):
        free_mask(5, lie.DOF, anchor=5)
    with pytest.raises(IndexError, match="anchor"):
        free_mask(5, lie.DOF, anchor=-1)


def test_levenberg_marquardt_reports_convergence_at_the_optimum(lie) -> None:
    """Starting at the optimum, every trial step is correctly rejected.

    chi2 cannot improve, so LM rejects, damping climbs, and the step vanishes.
    Testing convergence only on *accepted* steps made the solver report
    failure while sitting exactly on the minimum -- and because the Monte
    Carlo harness refuses to judge non-converged runs, that turned into
    silently discarded conditions rather than a visible error.
    """
    truth = short_trajectory(lie)
    graph = build_graph(lie, truth, noise=0.02, seed=21)
    optimum = gauss_newton(graph, perturbed(lie, truth, sigma=0.1), anchor=0)

    restarted = levenberg_marquardt(graph, optimum.poses, anchor=0)
    assert restarted.converged
    assert restarted.chi2 == pytest.approx(optimum.chi2, rel=1e-9)


def test_levenberg_marquardt_is_not_worse_than_gauss_newton(lie) -> None:
    """LM exists for the hard cases; it must not converge less often."""
    truth = short_trajectory(lie)
    graph = build_graph(lie, truth, noise=0.15, seed=31)
    start = perturbed(lie, truth, sigma=0.8)

    gn = gauss_newton(graph, start, anchor=0)
    lm = levenberg_marquardt(graph, start, anchor=0)
    assert lm.converged or not gn.converged
    assert lm.chi2 <= gn.chi2 + 1e-6 * max(1.0, abs(gn.chi2))


def test_levenberg_marquardt_gives_up_when_damping_is_exhausted(lie) -> None:
    """The failure path must exit, not spin to max_iterations.

    Reached here by setting tol=0 so the step-norm test can never fire,
    leaving the damping ceiling as the only way out. That branch is how a
    genuinely stuck solve reports itself, and the Monte Carlo harness refuses
    to compute a verdict over runs that did not converge -- so it has to be
    reported honestly rather than silently returning the last iterate.
    """
    truth = short_trajectory(lie)
    graph = build_graph(lie, truth, noise=0.02, seed=21)
    optimum = gauss_newton(graph, perturbed(lie, truth, sigma=0.1), anchor=0)

    stuck = levenberg_marquardt(graph, optimum.poses, anchor=0, tol=0.0)
    assert not stuck.converged
    assert stuck.iterations < 100, "should exit on damping, not exhaust iterations"
    # giving up must not corrupt the estimate it already had
    assert stuck.chi2 == pytest.approx(optimum.chi2, rel=1e-9)

