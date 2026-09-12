"""Week-1 validation gate for SE(2): round-trip, identity, group validity,
analytic-vs-numerical Jacobian, and small-angle stability.

If any of these fail, nothing built on top of se2.py (factors, optimizer,
covariance) can be trusted — this is the E1 gate from the project spec.
"""

from __future__ import annotations

import numpy as np
import pytest

from posetrust.lie import se2

RNG = np.random.default_rng(0)


def random_xi(n: int) -> np.ndarray:
    return RNG.uniform(-2.0, 2.0, size=(n, 3))


def numerical_right_jacobian(xi: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """Central-difference Jr(xi): exp(xi)^-1 @ exp(xi + d) ~= exp(Jr @ d)."""
    T0 = se2.exp(xi)
    T0_inv = se2.inverse(T0)
    J = np.zeros((3, 3))
    for i in range(3):
        d = np.zeros(3)
        d[i] = h
        T_plus = se2.exp(xi + d)
        T_minus = se2.exp(xi - d)
        delta_plus = se2.log(se2.compose(T0_inv, T_plus))
        delta_minus = se2.log(se2.compose(T0_inv, T_minus))
        J[:, i] = (delta_plus - delta_minus) / (2 * h)
    return J


@pytest.mark.parametrize("xi", random_xi(20))
def test_exp_log_round_trip(xi: np.ndarray) -> None:
    T = se2.exp(xi)
    xi_recovered = se2.log(T)
    np.testing.assert_allclose(xi_recovered, xi, atol=1e-10)


def test_identity() -> None:
    T = se2.exp(np.zeros(3))
    np.testing.assert_allclose(T, np.eye(3), atol=1e-12)
    xi = se2.log(np.eye(3))
    np.testing.assert_allclose(xi, np.zeros(3), atol=1e-12)


@pytest.mark.parametrize("xi", random_xi(20))
def test_group_validity(xi: np.ndarray) -> None:
    T = se2.exp(xi)
    R = T[:2, :2]

    # Rotation block is orthogonal with determinant 1 (i.e. actually in SO(2)).
    np.testing.assert_allclose(R @ R.T, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-12)

    # T composed with its inverse is the identity.
    np.testing.assert_allclose(se2.compose(T, se2.inverse(T)), np.eye(3), atol=1e-10)
    np.testing.assert_allclose(se2.compose(se2.inverse(T), T), np.eye(3), atol=1e-10)


@pytest.mark.parametrize("xi", random_xi(20))
def test_analytic_vs_numerical_jacobian(xi: np.ndarray) -> None:
    J_analytic = se2.right_jacobian(xi)
    J_numeric = numerical_right_jacobian(xi)
    np.testing.assert_allclose(J_analytic, J_numeric, atol=1e-6)


@pytest.mark.parametrize("theta", [0.0, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3])
def test_small_angle_stability(theta: float) -> None:
    xi = np.array([1.0, -0.5, theta])

    # Round trip must stay accurate even as theta -> 0, where the direct
    # (1 - cos theta) formula would lose precision without the Taylor fallback.
    T = se2.exp(xi)
    xi_recovered = se2.log(T)
    np.testing.assert_allclose(xi_recovered, xi, atol=1e-9)

    # The right Jacobian must not blow up or contain NaNs near theta = 0.
    J = se2.right_jacobian(xi)
    assert np.all(np.isfinite(J))
    np.testing.assert_allclose(J, numerical_right_jacobian(xi), atol=1e-6)
