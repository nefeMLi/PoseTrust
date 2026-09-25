"""Tests for SE(2)."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import hat_matrix, numerical_right_jacobian
from scipy.linalg import expm

from posetrust.lie import se2

RNG = np.random.default_rng(0)


def random_xi(n: int) -> np.ndarray:
    return RNG.uniform(-2.0, 2.0, size=(n, 3))


@pytest.mark.parametrize("xi", random_xi(20))
def test_exp_log_round_trip(xi: np.ndarray) -> None:
    T = se2.exp(xi)
    xi_recovered = se2.log(T)
    np.testing.assert_allclose(xi_recovered, xi, rtol=0, atol=1e-13)


def test_identity() -> None:
    T = se2.exp(np.zeros(3))
    np.testing.assert_allclose(T, np.eye(3), rtol=0, atol=1e-15)
    xi = se2.log(np.eye(3))
    np.testing.assert_allclose(xi, np.zeros(3), rtol=0, atol=1e-15)


@pytest.mark.parametrize("xi", random_xi(20))
def test_group_validity(xi: np.ndarray) -> None:
    T = se2.exp(xi)
    R = T[:2, :2]

    # Rotation block is orthogonal with determinant 1 (i.e. actually in SO(2)).
    np.testing.assert_allclose(R @ R.T, np.eye(2), rtol=0, atol=1e-13)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, rtol=0, atol=1e-13)

    # T composed with its inverse is the identity.
    np.testing.assert_allclose(se2.compose(T, se2.inverse(T)), np.eye(3), rtol=0, atol=1e-13)
    np.testing.assert_allclose(se2.compose(se2.inverse(T), T), np.eye(3), rtol=0, atol=1e-13)


@pytest.mark.parametrize("xi", random_xi(20))
def test_analytic_vs_numerical_jacobian(xi: np.ndarray) -> None:
    J_analytic = se2.right_jacobian(xi)
    J_numeric = numerical_right_jacobian(se2, xi)
    np.testing.assert_allclose(J_analytic, J_numeric, rtol=0, atol=1e-8)


@pytest.mark.parametrize(
    "theta",
    [0.0, 1e-12, 1e-8, 1e-5, 1e-4, 1e-3, 1e-2, 0.049, se2._EPS, 0.051, 0.1, 1.0, 3.0],
)
def test_exp_matches_matrix_exponential(theta: float) -> None:
    xi = np.array([1.3, -0.7, theta])
    np.testing.assert_allclose(se2.exp(xi), expm(hat_matrix(se2, xi)), rtol=0, atol=1e-14)


def test_full_domain_sweep() -> None:
    thetas = np.concatenate(
        [
            np.logspace(-14, -0.5, 100),
            np.linspace(0.3, np.pi, 100),
            [0.0, np.pi],
        ]
    )
    rng = np.random.default_rng(20260920)
    for theta in thetas:
        v = 10 ** rng.uniform(-6, 3) * rng.normal(size=2)
        xi = np.array([v[0], v[1], theta])
        T = se2.exp(xi)
        scale = max(1.0, float(np.max(np.abs(T))))
        np.testing.assert_allclose(T, expm(hat_matrix(se2, xi)), rtol=0, atol=1e-12 * scale)
        np.testing.assert_allclose(se2.exp(se2.log(T)), T, rtol=0, atol=1e-12 * scale)


@pytest.mark.parametrize("fraction", [1.0, 0.8, 0.6, 0.4, 0.2])
def test_taylor_and_direct_branches_agree(monkeypatch, fraction: float) -> None:
    theta = fraction * se2._EPS
    xi = np.array([1.3, -0.7, theta])

    monkeypatch.setattr(se2, "_EPS", 0.0)  # force the direct trig branch
    direct_exp, direct_jr = se2.exp(xi), se2.right_jacobian(xi)

    monkeypatch.setattr(se2, "_EPS", np.inf)  # force the Taylor branch
    taylor_exp, taylor_jr = se2.exp(xi), se2.right_jacobian(xi)

    np.testing.assert_allclose(direct_exp, taylor_exp, rtol=0, atol=1e-14)
    np.testing.assert_allclose(direct_jr, taylor_jr, rtol=0, atol=1e-14)


@pytest.mark.parametrize("theta", [0.0, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.049, 0.051])
def test_small_angle_stability(theta: float) -> None:
    xi = np.array([1.0, -0.5, theta])

    # Round trip must stay accurate even as theta -> 0, where the direct
    # (1 - cos theta) formula would lose precision without the Taylor fallback.
    T = se2.exp(xi)
    xi_recovered = se2.log(T)
    np.testing.assert_allclose(xi_recovered, xi, rtol=0, atol=1e-14)

    # The right Jacobian must not blow up or contain NaNs near theta = 0.
    J = se2.right_jacobian(xi)
    assert np.all(np.isfinite(J))
    np.testing.assert_allclose(J, numerical_right_jacobian(se2, xi), rtol=0, atol=1e-8)
