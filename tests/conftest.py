"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from posetrust.graph import PoseGraph
from posetrust.lie import se2, se3
from posetrust.simulate import curved_trajectory as _curved_trajectory

GROUPS = [se2, se3]
GROUP_IDS = ["se2", "se3"]


@pytest.fixture(params=GROUPS, ids=GROUP_IDS)
def lie(request):
    """Run against both SE(2) and SE(3)."""
    return request.param


def short_trajectory(lie, n_poses: int = 12, turn: float = 0.35) -> list[np.ndarray]:
    """curved_trajectory with test defaults."""
    return _curved_trajectory(lie, n_poses=n_poses, turn=turn)


def hat_matrix(lie, xi: np.ndarray) -> np.ndarray:
    """Lie algebra matrix of xi, built without the library."""
    xi = np.asarray(xi, dtype=float)
    if lie.DOF == 3:
        vx, vy, theta = xi
        return np.array([[0.0, -theta, vx], [theta, 0.0, vy], [0.0, 0.0, 0.0]])
    rho, phi = xi[:3], xi[3:]
    M = np.zeros((4, 4))
    M[:3, :3] = np.array(
        [
            [0.0, -phi[2], phi[1]],
            [phi[2], 0.0, -phi[0]],
            [-phi[1], phi[0], 0.0],
        ]
    )
    M[:3, 3] = rho
    return M


def numerical_right_jacobian(lie, xi: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """Central-difference right Jacobian."""
    dof = lie.DOF
    T0_inv = lie.inverse(lie.exp(xi))
    J = np.zeros((dof, dof))
    for i in range(dof):
        d = np.zeros(dof)
        d[i] = h
        plus = lie.log(lie.compose(T0_inv, lie.exp(xi + d)))
        minus = lie.log(lie.compose(T0_inv, lie.exp(xi - d)))
        J[:, i] = (plus - minus) / (2 * h)
    return J


def build_graph(
    lie,
    truth: list[np.ndarray],
    loop_closure: bool = True,
    noise: float = 0.0,
    seed: int = 0,
) -> PoseGraph:
    """Odometry chain plus one closure from the first pose to the last."""
    rng = np.random.default_rng(seed)
    graph = PoseGraph(lie)
    for T in truth:
        graph.add_pose(T)

    def measure(i: int, j: int) -> np.ndarray:
        Z = lie.compose(lie.inverse(truth[i]), truth[j])
        if noise > 0.0:
            Z = lie.compose(Z, lie.exp(rng.normal(0.0, noise, lie.DOF)))
        return Z

    information = np.eye(lie.DOF)
    for k in range(len(truth) - 1):
        graph.add_factor(k, k + 1, measure(k, k + 1), information)
    if loop_closure:
        graph.add_factor(0, len(truth) - 1, measure(0, len(truth) - 1), information)
    return graph


def perturbed(lie, truth: list[np.ndarray], sigma: float = 0.25, seed: int = 1):
    """Perturb every pose except the first, which stays put as the anchor."""
    rng = np.random.default_rng(seed)
    return [truth[0]] + [
        lie.compose(T, lie.exp(rng.normal(0.0, sigma, lie.DOF))) for T in truth[1:]
    ]


@pytest.fixture
def outlier_free_graph(lie):
    """Small clean graph and a perturbed start."""
    truth = short_trajectory(lie, n_poses=8)
    graph = build_graph(lie, truth, noise=0.02, seed=5)
    return graph, truth, perturbed(lie, truth, sigma=0.05)

