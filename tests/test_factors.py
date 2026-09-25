"""Tests for residuals and factor Jacobians."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
from conftest import build_graph, perturbed, short_trajectory


def numerical_factor_jacobians(graph, factor, poses, h=1e-6):
    """Central-difference factor Jacobians."""
    lie = graph.lie
    dof = lie.DOF
    Ni, Nj = np.zeros((dof, dof)), np.zeros((dof, dof))
    Ti, Tj = poses[factor.i], poses[factor.j]
    for k in range(dof):
        e = np.zeros(dof)
        e[k] = h
        plus = list(poses)
        minus = list(poses)
        plus[factor.i], minus[factor.i] = (
            lie.compose(Ti, lie.exp(e)),
            lie.compose(Ti, lie.exp(-e)),
        )
        Ni[:, k] = (graph.residual(factor, plus) - graph.residual(factor, minus)) / (2 * h)
        plus = list(poses)
        minus = list(poses)
        plus[factor.j], minus[factor.j] = (
            lie.compose(Tj, lie.exp(e)),
            lie.compose(Tj, lie.exp(-e)),
        )
        Nj[:, k] = (graph.residual(factor, plus) - graph.residual(factor, minus)) / (2 * h)
    return Ni, Nj


def test_residual_is_zero_at_truth(lie) -> None:
    truth = short_trajectory(lie)
    graph = build_graph(lie, truth)
    for factor in graph.factors:
        r = graph.residual(factor, truth)
        np.testing.assert_allclose(r, np.zeros(lie.DOF), rtol=0, atol=1e-13)
    assert graph.chi2(truth) < 1e-24


def test_residual_is_nonzero_when_wrong(lie) -> None:
    truth = short_trajectory(lie)
    graph = build_graph(lie, truth)
    bad = perturbed(lie, truth, sigma=0.3)
    assert graph.chi2(bad) > 1e-3


def test_factor_jacobians_match_finite_differences(lie) -> None:
    rng = np.random.default_rng(11)
    truth = short_trajectory(lie)
    graph = build_graph(lie, truth)
    # evaluate away from the optimum, where both Jacobian blocks are non-trivial
    poses = perturbed(lie, truth, sigma=0.3, seed=int(rng.integers(1 << 30)))
    for factor in graph.factors:
        Ji, Jj = graph.factor_jacobians(factor, poses)
        Ni, Nj = numerical_factor_jacobians(graph, factor, poses)
        np.testing.assert_allclose(Ji, Ni, rtol=0, atol=1e-7)
        np.testing.assert_allclose(Jj, Nj, rtol=0, atol=1e-7)


def test_factor_jacobians_converge_quadratically(lie) -> None:
    truth = short_trajectory(lie)
    graph = build_graph(lie, truth)
    poses = perturbed(lie, truth, sigma=0.3)
    factor = graph.factors[0]
    Ji, Jj = graph.factor_jacobians(factor, poses)

    errs = []
    for h in (1e-2, 1e-3, 1e-4):
        Ni, Nj = numerical_factor_jacobians(graph, factor, poses, h=h)
        errs.append(max(np.max(np.abs(Ji - Ni)), np.max(np.abs(Jj - Nj))))
    for coarse, fine in pairwise(errs):
        assert coarse / fine > 50.0, f"finite differences plateaued: {errs}"


def test_information_matrix_structure(lie) -> None:
    truth = short_trajectory(lie)
    graph = build_graph(lie, truth)
    H, b = graph.linearize(truth)
    n = len(truth) * lie.DOF

    assert H.shape == (n, n) and b.shape == (n,)
    np.testing.assert_allclose(H, H.T, rtol=0, atol=1e-12)
    # at the truth every residual is zero, so the gradient must vanish
    np.testing.assert_allclose(b, np.zeros(n), rtol=0, atol=1e-12)
    # H is positive SEMI-definite: no negative eigenvalues
    assert np.linalg.eigvalsh(H).min() > -1e-9


def test_gauge_freedom_is_exactly_the_group_dimension(lie) -> None:
    truth = short_trajectory(lie)
    graph = build_graph(lie, truth)
    H, _ = graph.linearize(truth)
    null_dim = int(np.sum(np.abs(np.linalg.eigvalsh(H)) < 1e-8))
    assert null_dim == lie.DOF, f"expected {lie.DOF} gauge directions, got {null_dim}"


def test_retract_moves_along_the_manifold(lie) -> None:
    truth = short_trajectory(lie, n_poses=4)
    graph = build_graph(lie, truth)
    delta = np.zeros(len(truth) * lie.DOF)
    delta[lie.DOF : 2 * lie.DOF] = 0.1
    moved = graph.retract(truth, delta)

    np.testing.assert_allclose(moved[0], truth[0], rtol=0, atol=1e-15)
    expected = lie.compose(truth[1], lie.exp(np.full(lie.DOF, 0.1)))
    np.testing.assert_allclose(moved[1], expected, rtol=0, atol=1e-13)
    # every retracted pose is still a valid group element
    for T in moved:
        d = T.shape[0] - 1
        R = T[:d, :d]
        np.testing.assert_allclose(R @ R.T, np.eye(d), rtol=0, atol=1e-13)
