"""Week-1 validation gate for SE(3): round-trip, identity, group validity,
analytic-vs-numerical Jacobian, and small-angle stability. Mirrors test_lie.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from posetrust.lie import se3

RNG = np.random.default_rng(0)


def random_xi(n: int) -> np.ndarray:
    return RNG.uniform(-1.5, 1.5, size=(n, 6))


def numerical_right_jacobian(xi: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """Central-difference Jr(xi): exp(xi)^-1 @ exp(xi + d) ~= exp(Jr @ d).

    This deliberately exercises se3.py at tiny relative angles (~h), which is
    exactly where the Taylor-series fallback matters, not just where it's
    convenient.
    """
    T0 = se3.exp(xi)
    T0_inv = se3.inverse(T0)
    J = np.zeros((6, 6))
    for i in range(6):
        d = np.zeros(6)
        d[i] = h
        T_plus = se3.exp(xi + d)
        T_minus = se3.exp(xi - d)
        delta_plus = se3.log(se3.compose(T0_inv, T_plus))
        delta_minus = se3.log(se3.compose(T0_inv, T_minus))
        J[:, i] = (delta_plus - delta_minus) / (2 * h)
    return J


@pytest.mark.parametrize("xi", random_xi(20))
def test_exp_log_round_trip(xi: np.ndarray) -> None:
    T = se3.exp(xi)
    xi_recovered = se3.log(T)
    np.testing.assert_allclose(xi_recovered, xi, atol=1e-9)


def test_identity() -> None:
    T = se3.exp(np.zeros(6))
    np.testing.assert_allclose(T, np.eye(4), atol=1e-12)
    xi = se3.log(np.eye(4))
    np.testing.assert_allclose(xi, np.zeros(6), atol=1e-12)


@pytest.mark.parametrize("xi", random_xi(20))
def test_group_validity(xi: np.ndarray) -> None:
    T = se3.exp(xi)
    R = T[:3, :3]

    # Rotation block is orthogonal with determinant 1 (i.e. actually in SO(3)).
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)

    # T composed with its inverse is the identity.
    np.testing.assert_allclose(se3.compose(T, se3.inverse(T)), np.eye(4), atol=1e-9)
    np.testing.assert_allclose(se3.compose(se3.inverse(T), T), np.eye(4), atol=1e-9)


@pytest.mark.parametrize("xi", random_xi(20))
def test_analytic_vs_numerical_jacobian(xi: np.ndarray) -> None:
    J_analytic = se3.right_jacobian(xi)
    J_numeric = numerical_right_jacobian(xi)
    np.testing.assert_allclose(J_analytic, J_numeric, atol=1e-5)


@pytest.mark.parametrize("theta", [0.0, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
def test_small_angle_stability(theta: float) -> None:
    axis = np.array([1.0, -2.0, 0.5])
    axis /= np.linalg.norm(axis)
    xi = np.concatenate([[1.0, -0.5, 0.3], theta * axis])

    # Round trip must stay accurate even as theta -> 0, where the direct
    # formulas would lose precision without the Taylor fallback.
    T = se3.exp(xi)
    xi_recovered = se3.log(T)
    np.testing.assert_allclose(xi_recovered, xi, atol=1e-8)

    # The right Jacobian must not blow up or contain NaNs near theta = 0.
    J = se3.right_jacobian(xi)
    assert np.all(np.isfinite(J))
    np.testing.assert_allclose(J, numerical_right_jacobian(xi), atol=1e-5)
