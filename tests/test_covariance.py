"""Tests for covariances and gauge handling."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import build_graph, short_trajectory

from posetrust.covariance import (
    cholesky_factor,
    covariance_matrix,
    joint_covariance,
    marginal_covariances,
    relative_covariance,
    selected_inverse,
)
from posetrust.graph import PoseGraph
from posetrust.optimize.gauge import free_mask
from posetrust.optimize.optimizer import gauss_newton

CHAIN_LENGTH = 8


def identity_chain(lie, n: int = CHAIN_LENGTH, loop_closure: bool = False):
    """All poses and measurements identity."""
    dof = lie.DOF
    omega = np.diag(np.linspace(1.0, 4.0, dof))
    graph = PoseGraph(lie)
    identity = lie.exp(np.zeros(dof))
    for _ in range(n):
        graph.add_pose(identity)
    for k in range(n - 1):
        graph.add_factor(k, k + 1, identity, omega)
    if loop_closure:
        graph.add_factor(0, n - 1, identity, omega)
    H, _ = graph.linearize(graph.poses)
    return graph, H, omega


@pytest.mark.parametrize("n", [1, 2, 3, 5, 9, 17, 40])
def test_selected_inverse_matches_dense_inverse(n: int) -> None:
    rng = np.random.default_rng(n)
    for _ in range(10):
        A = rng.normal(size=(n, n))
        H = A @ A.T + n * np.eye(n)
        sigma = selected_inverse(cholesky_factor(H))
        np.testing.assert_allclose(sigma, np.linalg.inv(H), rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(sigma @ H, np.eye(n), rtol=0, atol=1e-9)


def test_selected_inverse_is_symmetric() -> None:
    rng = np.random.default_rng(4)
    A = rng.normal(size=(12, 12))
    sigma = selected_inverse(cholesky_factor(A @ A.T + 12 * np.eye(12)))
    np.testing.assert_allclose(sigma, sigma.T, rtol=0, atol=1e-14)


def test_chain_marginals_match_closed_form(lie) -> None:
    _, H, omega = identity_chain(lie)
    marginals = marginal_covariances(H, anchor=0, dof=lie.DOF)
    omega_inv = np.linalg.inv(omega)
    for k, block in enumerate(marginals):
        np.testing.assert_allclose(block, k * omega_inv, rtol=0, atol=1e-12)


def test_chain_relative_covariance_matches_closed_form(lie) -> None:
    graph, H, omega = identity_chain(lie)
    omega_inv = np.linalg.inv(omega)
    for i, j in [(0, 1), (2, 6), (1, 7), (3, 4)]:
        rel = relative_covariance(lie, H, graph.poses, i, j, anchor=0)
        np.testing.assert_allclose(rel, (j - i) * omega_inv, rtol=0, atol=1e-12)


def test_anchored_pose_has_exactly_zero_covariance(lie) -> None:
    truth = short_trajectory(lie)
    graph = build_graph(lie, truth, noise=0.02, seed=2)
    result = gauss_newton(graph, truth, anchor=0)
    for anchor in (0, 3, len(truth) - 1):
        blocks = marginal_covariances(result.information, anchor, lie.DOF)
        np.testing.assert_allclose(
            blocks[anchor], np.zeros((lie.DOF, lie.DOF)), rtol=0, atol=0.0
        )


def test_marginals_are_valid_covariances(lie) -> None:
    truth = short_trajectory(lie)
    graph = build_graph(lie, truth, noise=0.02, seed=6)
    result = gauss_newton(graph, truth, anchor=0)
    blocks = marginal_covariances(result.information, 0, lie.DOF)

    for k, block in enumerate(blocks):
        np.testing.assert_allclose(block, block.T, rtol=0, atol=1e-14)
        smallest = np.linalg.eigvalsh(block).min()
        if k == 0:
            assert smallest == 0.0
        else:
            assert smallest > 0.0, f"pose {k} marginal is not positive definite"


def test_relative_covariance_is_gauge_invariant(lie) -> None:
    truth = short_trajectory(lie)
    graph = build_graph(lie, truth, noise=0.02, seed=8)

    reference = None
    for anchor in (0, 2, 5, len(truth) - 1):
        result = gauss_newton(graph, truth, anchor=anchor)
        rel = relative_covariance(lie, result.information, result.poses, 3, 7, anchor)
        if reference is None:
            reference = rel
        else:
            np.testing.assert_allclose(rel, reference, rtol=1e-8, atol=1e-12)


def test_loop_closure_reduces_uncertainty(lie) -> None:
    _, H_open, _ = identity_chain(lie, loop_closure=False)
    _, H_closed, _ = identity_chain(lie, loop_closure=True)
    open_blocks = marginal_covariances(H_open, 0, lie.DOF)
    closed_blocks = marginal_covariances(H_closed, 0, lie.DOF)

    for k in range(2, CHAIN_LENGTH):
        assert (
            np.linalg.eigvalsh(closed_blocks[k]).max()
            < np.linalg.eigvalsh(open_blocks[k]).max()
        )


def test_cross_covariance_is_not_ignored(lie) -> None:
    graph, H, _ = identity_chain(lie)
    dof = lie.DOF
    blocks = marginal_covariances(H, 0, dof)
    rel = relative_covariance(lie, H, graph.poses, 2, 6, anchor=0)
    naive = blocks[2] + blocks[6]

    assert np.trace(rel) < np.trace(naive)
    joint = joint_covariance(H, 2, 6, anchor=0, dof=dof)
    assert np.max(np.abs(joint[:dof, dof:])) > 1e-6


def test_unanchored_system_is_rejected(lie) -> None:
    _, H, _ = identity_chain(lie)
    with pytest.raises(np.linalg.LinAlgError, match="gauge"):
        cholesky_factor(H)


def test_covariance_matrix_zeroes_only_the_anchor(lie) -> None:
    truth = short_trajectory(lie)
    graph = build_graph(lie, truth, noise=0.02, seed=12)
    result = gauss_newton(graph, truth, anchor=2)
    sigma = covariance_matrix(result.information, anchor=2, dof=lie.DOF)

    free = free_mask(len(truth), lie.DOF, 2)
    np.testing.assert_allclose(sigma[~free], 0.0, rtol=0, atol=0.0)
    np.testing.assert_allclose(sigma[:, ~free], 0.0, rtol=0, atol=0.0)
    # and the free block really is the inverse of the free information block
    Hf = result.information[np.ix_(free, free)]
    np.testing.assert_allclose(
        sigma[np.ix_(free, free)] @ Hf, np.eye(free.sum()), rtol=0, atol=1e-8
    )
