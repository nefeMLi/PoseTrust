"""Tests for SE(3)."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import hat_matrix, numerical_right_jacobian
from scipy.linalg import expm

from posetrust.lie import se3

RNG = np.random.default_rng(0)


def random_xi(n: int) -> np.ndarray:
    return RNG.uniform(-1.5, 1.5, size=(n, 6))


@pytest.mark.parametrize("xi", random_xi(20))
def test_exp_log_round_trip(xi: np.ndarray) -> None:
    T = se3.exp(xi)
    xi_recovered = se3.log(T)
    np.testing.assert_allclose(xi_recovered, xi, rtol=0, atol=1e-13)


def test_identity() -> None:
    T = se3.exp(np.zeros(6))
    np.testing.assert_allclose(T, np.eye(4), rtol=0, atol=1e-15)
    xi = se3.log(np.eye(4))
    np.testing.assert_allclose(xi, np.zeros(6), rtol=0, atol=1e-15)


@pytest.mark.parametrize("xi", random_xi(20))
def test_group_validity(xi: np.ndarray) -> None:
    T = se3.exp(xi)
    R = T[:3, :3]

    # Rotation block is orthogonal with determinant 1 (i.e. actually in SO(3)).
    np.testing.assert_allclose(R @ R.T, np.eye(3), rtol=0, atol=1e-13)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, rtol=0, atol=1e-13)

    # T composed with its inverse is the identity.
    np.testing.assert_allclose(se3.compose(T, se3.inverse(T)), np.eye(4), rtol=0, atol=1e-13)
    np.testing.assert_allclose(se3.compose(se3.inverse(T), T), np.eye(4), rtol=0, atol=1e-13)


@pytest.mark.parametrize("xi", random_xi(20))
def test_analytic_vs_numerical_jacobian(xi: np.ndarray) -> None:
    J_analytic = se3.right_jacobian(xi)
    J_numeric = numerical_right_jacobian(se3, xi)
    np.testing.assert_allclose(J_analytic, J_numeric, rtol=0, atol=1e-8)


@pytest.mark.parametrize(
    "theta",
    [0.0, 1e-12, 1e-8, 1e-5, 1e-4, 1e-3, 1e-2, 0.049, se3._EPS, 0.051, 0.1, 1.0, 3.0],
)
def test_exp_matches_matrix_exponential(theta: float) -> None:
    axis = np.array([0.0, 0.6, 0.8])
    xi = np.concatenate([[1.3, -0.7, 0.4], theta * axis])
    np.testing.assert_allclose(se3.exp(xi), expm(hat_matrix(se3, xi)), rtol=0, atol=1e-14)


def test_full_domain_sweep() -> None:
    thetas = np.concatenate(
        [
            np.logspace(-14, -0.5, 80),
            np.linspace(0.3, np.pi, 80),
            np.pi - np.logspace(-9, -1, 60),
            [0.0, np.pi],
        ]
    )
    axes = np.array([[0.0, 0.0, 1.0], [1.0, -2.0, 0.5], [0.3, 0.3, 0.9], [1.0, 1.0, 0.0]])
    axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)
    rng = np.random.default_rng(20260920)

    for theta in thetas:
        for axis in axes:
            rho = 10 ** rng.uniform(-6, 3) * rng.normal(size=3)
            xi = np.concatenate([rho, theta * axis])
            T = se3.exp(xi)
            scale = max(1.0, float(np.max(np.abs(T))))
            # exp against an independent oracle
            np.testing.assert_allclose(T, expm(hat_matrix(se3, xi)), rtol=0, atol=1e-12 * scale)
            # log must invert exp everywhere, including at theta = pi
            np.testing.assert_allclose(
                se3.exp(se3.log(T)), T, rtol=0, atol=1e-12 * scale
            )


PI_AXES = [
    np.array([0.0, 0.0, 1.0]),
    np.array([1.0, -2.0, 0.5]),
    np.array([0.3, 0.3, 0.9]),
    np.array([1.0, 1.0, 0.0]),
]


@pytest.mark.parametrize("gap", [1e-1, 4.5e-2, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 0.0])
@pytest.mark.parametrize("axis", PI_AXES)
def test_log_near_pi(gap: float, axis: np.ndarray) -> None:
    axis = axis / np.linalg.norm(axis)
    theta = np.pi - gap
    T = se3.exp(np.concatenate([[0.5, 0.2, -0.3], theta * axis]))

    xi = se3.log(T)
    np.testing.assert_allclose(se3.exp(xi), T, rtol=0, atol=1e-12)
    # the recovered angle must still be the true one
    assert abs(np.linalg.norm(xi[3:]) - theta) < 1e-12


@pytest.mark.parametrize("fraction", [1.0, 0.8, 0.6, 0.4, 0.2])
def test_taylor_and_direct_branches_agree(monkeypatch, fraction: float) -> None:
    theta = fraction * se3._EPS
    axis = np.array([0.0, 0.6, 0.8])
    xi = np.concatenate([[1.3, -0.7, 0.4], theta * axis])

    monkeypatch.setattr(se3, "_EPS", 0.0)  # force the direct trig branch
    direct_exp, direct_jr = se3.exp(xi), se3.right_jacobian(xi)

    monkeypatch.setattr(se3, "_EPS", np.inf)  # force the Taylor branch
    taylor_exp, taylor_jr = se3.exp(xi), se3.right_jacobian(xi)

    np.testing.assert_allclose(direct_exp, taylor_exp, rtol=0, atol=1e-14)
    np.testing.assert_allclose(direct_jr, taylor_jr, rtol=0, atol=1e-14)


@pytest.mark.parametrize(
    "theta", [0.0, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.049, 0.051]
)
def test_small_angle_stability(theta: float) -> None:
    axis = np.array([1.0, -2.0, 0.5])
    axis /= np.linalg.norm(axis)
    xi = np.concatenate([[1.0, -0.5, 0.3], theta * axis])

    # Round trip must stay accurate even as theta -> 0, where the direct
    # formulas would lose precision without the Taylor fallback.
    T = se3.exp(xi)
    xi_recovered = se3.log(T)
    np.testing.assert_allclose(xi_recovered, xi, rtol=0, atol=1e-14)

    # The right Jacobian must not blow up or contain NaNs near theta = 0.
    J = se3.right_jacobian(xi)
    assert np.all(np.isfinite(J))
    np.testing.assert_allclose(J, numerical_right_jacobian(se3, xi), rtol=0, atol=1e-8)
