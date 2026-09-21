"""SE(2) Lie group: exp/log maps and the right Jacobian.

Convention: a pose is a 3x3 homogeneous matrix [[R, t], [0, 1]].
A tangent vector xi = [vx, vy, theta] maps to SE(2) via exp(), and
exp(xi + d) ~= exp(xi) @ exp(right_jacobian(xi) @ d) for small d —
this is the Jr used by the Gauss-Newton update step.
"""

from __future__ import annotations

import numpy as np

# Crossover between the Taylor series and the direct trigonometric formulas.
# Both branches must be accurate AT this radius, which puts it far higher than
# the naive "just above sqrt(machine epsilon)" reasoning suggests. The direct
# form of (theta - sin(theta))/theta**2 carries relative error ~eps/theta**2,
# so it is still losing ~8 significant digits at theta = 1e-4; the series
# below (five terms) is exact to ~1e-19 out to 5e-2. Measured worst-case
# error across the whole crossover is ~1e-15 (see the small-angle tests).
_EPS = 5e-2


# Degrees of freedom of the tangent space, so callers can stay group-agnostic.
DOF = 3

# xi is ordered [translation..., rotation...]; this is where it splits, so a
# per-degree-of-freedom breakdown can separate the two.
TRANSLATION_DOF = 2


def rotation_matrix(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _ab(theta: float) -> tuple[float, float]:
    """a = sin(theta)/theta, b = (1 - cos(theta))/theta, Taylor-stable near 0."""
    if abs(theta) < _EPS:
        t2 = theta**2
        a = 1.0 - t2 / 6.0 + t2**2 / 120.0 - t2**3 / 5040.0 + t2**4 / 362880.0
        b = theta * (
            0.5 - t2 / 24.0 + t2**2 / 720.0 - t2**3 / 40320.0 + t2**4 / 3628800.0
        )
    else:
        a = np.sin(theta) / theta
        # 1 - cos(theta) == 2*sin(theta/2)**2 avoids the cancellation entirely
        b = 2.0 * np.sin(0.5 * theta) ** 2 / theta
    return a, b


def _v_matrix(theta: float) -> np.ndarray:
    """Left Jacobian of SO(2): maps the translational part of xi to SE(2) t."""
    a, b = _ab(theta)
    return np.array([[a, -b], [b, a]])


def _v_inv_matrix(theta: float) -> np.ndarray:
    a, b = _ab(theta)
    det = a**2 + b**2
    return np.array([[a, b], [-b, a]]) / det


def exp(xi: np.ndarray) -> np.ndarray:
    """se(2) -> SE(2). xi = [vx, vy, theta]. Returns a 3x3 homogeneous matrix."""
    xi = np.asarray(xi, dtype=float)
    v, theta = xi[:2], xi[2]
    T = np.eye(3)
    T[:2, :2] = rotation_matrix(theta)
    T[:2, 2] = _v_matrix(theta) @ v
    return T


def log(T: np.ndarray) -> np.ndarray:
    """SE(2) -> se(2). Returns xi = [vx, vy, theta]."""
    R, t = T[:2, :2], T[:2, 2]
    theta = np.arctan2(R[1, 0], R[0, 0])
    v = _v_inv_matrix(theta) @ t
    return np.array([v[0], v[1], theta])


def compose(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
    return T1 @ T2


def inverse(T: np.ndarray) -> np.ndarray:
    R, t = T[:2, :2], T[:2, 2]
    Tinv = np.eye(3)
    Tinv[:2, :2] = R.T
    Tinv[:2, 2] = -R.T @ t
    return Tinv


def adjoint(T: np.ndarray) -> np.ndarray:
    """Adj(T), defined by T @ exp(x) @ inverse(T) == exp(Adj(T) @ x).

    Moves a tangent perturbation from one frame to another, which is what
    lets a relative-pose residual be differentiated with respect to both of
    the poses it connects.
    """
    R, t = T[:2, :2], T[:2, 2]
    A = np.eye(3)
    A[:2, :2] = R
    A[:2, 2] = np.array([t[1], -t[0]])
    return A


def right_jacobian(xi: np.ndarray) -> np.ndarray:
    """Right Jacobian Jr(xi): exp(xi + d) ~= exp(xi) @ exp(Jr(xi) @ d)."""
    xi = np.asarray(xi, dtype=float)
    v, theta = xi[:2], xi[2]
    a, b = _ab(theta)
    if abs(theta) < _EPS:
        t2 = theta**2
        c = theta * (
            1.0 / 6.0
            - t2 / 120.0
            + t2**2 / 5040.0
            - t2**3 / 362880.0
            + t2**4 / 39916800.0
        )
        d = 0.5 - t2 / 24.0 + t2**2 / 720.0 - t2**3 / 40320.0 + t2**4 / 3628800.0
    else:
        c = (theta - np.sin(theta)) / theta**2
        d = 2.0 * np.sin(0.5 * theta) ** 2 / theta**2
    J = np.eye(3)
    J[:2, :2] = np.array([[a, b], [-b, a]])
    J[:2, 2] = np.array([c * v[0] - d * v[1], c * v[1] + d * v[0]])
    return J
