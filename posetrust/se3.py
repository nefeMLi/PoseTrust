"""SE(3): exp, log, adjoint and right Jacobian."""

from __future__ import annotations

import numpy as np

# Below this angle use the Taylor series. Q divides by theta^4 and theta^5,
# so the direct form loses precision to cancellation well above 1e-3.
_EPS = 5e-2

# Below this cos(theta) the log reads the axis from the symmetric part,
# which stays accurate near pi.
_COS_PI_SWITCH = -0.999


# Degrees of freedom of the tangent space, so callers can stay group-agnostic.
DOF = 6

# xi = [translation, rotation]; the index where they split.
TRANSLATION_DOF = 3


def hat(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix of v."""
    v = np.asarray(v, dtype=float)
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


def vee(M: np.ndarray) -> np.ndarray:
    """Inverse of hat()."""
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


def _so3_coeffs(theta: float) -> tuple[float, float, float]:
    """sin(t)/t, (1 - cos t)/t^2 and (t - sin t)/t^3, stable near zero."""
    if abs(theta) < _EPS:
        t2 = theta**2
        a = 1.0 - t2 / 6.0 + t2**2 / 120.0 - t2**3 / 5040.0 + t2**4 / 362880.0
        b = 0.5 - t2 / 24.0 + t2**2 / 720.0 - t2**3 / 40320.0 + t2**4 / 3628800.0
        c = (
            1.0 / 6.0
            - t2 / 120.0
            + t2**2 / 5040.0
            - t2**3 / 362880.0
            + t2**4 / 39916800.0
        )
    else:
        s = np.sin(theta)
        a = s / theta
        # 1 - cos(theta) == 2*sin(theta/2)**2 avoids the cancellation entirely
        b = 2.0 * np.sin(0.5 * theta) ** 2 / theta**2
        c = (theta - s) / theta**3
    return a, b, c


def _v_inv_coeff(theta: float) -> float:
    """Coefficient of K^2 in the inverse SO(3) left Jacobian."""
    if abs(theta) < _EPS:
        t2 = theta**2
        return (
            1.0 / 12.0
            + t2 / 720.0
            + t2**2 / 30240.0
            + t2**3 / 1209600.0
            + t2**4 / 47900160.0
        )
    # Written with tan(theta/2): (1 + cos(theta)) cancels badly near pi.
    return 1.0 / theta**2 - 1.0 / (2.0 * theta * np.tan(0.5 * theta))


def _q_coeffs(theta: float) -> tuple[float, float, float]:
    """Coefficients of the SE(3) Q matrix (Barfoot & Furgale 2014)."""
    if abs(theta) < _EPS:
        t2 = theta**2
        c1 = (
            1.0 / 6.0
            - t2 / 120.0
            + t2**2 / 5040.0
            - t2**3 / 362880.0
            + t2**4 / 39916800.0
        )
        c2 = (
            -1.0 / 24.0
            + t2 / 720.0
            - t2**2 / 40320.0
            + t2**3 / 3628800.0
            - t2**4 / 479001600.0
        )
        c3 = (
            -1.0 / 60.0
            + t2 / 1260.0
            - t2**2 / 60480.0
            + t2**3 / 4989600.0
            - t2**4 / 622702080.0
        )
    else:
        s = np.sin(theta)
        c1 = (theta - s) / theta**3
        # 1 - theta**2/2 - cos(theta) == 2*sin(theta/2)**2 - theta**2/2, which
        # cancels one order less violently than the cos form at small theta
        c2 = (2.0 * np.sin(0.5 * theta) ** 2 - theta**2 / 2.0) / theta**4
        c3 = c2 - 3.0 * (theta - s - theta**3 / 6.0) / theta**5
    return c1, c2, c3


def rotation_matrix(phi: np.ndarray) -> np.ndarray:
    """so(3) -> SO(3) by Rodrigues' formula."""
    phi = np.asarray(phi, dtype=float)
    theta = np.linalg.norm(phi)
    K = hat(phi)
    a, b, _ = _so3_coeffs(theta)
    return np.eye(3) + a * K + b * (K @ K)


def _v_matrix(phi: np.ndarray) -> np.ndarray:
    """Left Jacobian of SO(3)."""
    theta = np.linalg.norm(phi)
    K = hat(phi)
    _, b, c = _so3_coeffs(theta)
    return np.eye(3) + b * K + c * (K @ K)


def _v_inv_matrix(phi: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(phi)
    K = hat(phi)
    e = _v_inv_coeff(theta)
    return np.eye(3) - 0.5 * K + e * (K @ K)


def exp(xi: np.ndarray) -> np.ndarray:
    """se(3) -> SE(3)."""
    xi = np.asarray(xi, dtype=float)
    rho, phi = xi[:3], xi[3:]
    T = np.eye(4)
    T[:3, :3] = rotation_matrix(phi)
    T[:3, 3] = _v_matrix(phi) @ rho
    return T


def _log_rotation(R: np.ndarray) -> np.ndarray:
    """SO(3) -> so(3), including angles near 0 and near pi."""
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    s = vee((R - R.T) / 2.0)  # = sin(theta) * axis
    theta = np.arctan2(np.linalg.norm(s), cos_theta)
    if cos_theta <= _COS_PI_SWITCH:
        return _log_rotation_near_pi(R, cos_theta, theta, s)
    a, _, _ = _so3_coeffs(theta)  # a = sin(theta)/theta, safe at theta = 0
    return s / a


def _log_rotation_near_pi(
    R: np.ndarray, cos_theta: float, theta: float, s: np.ndarray
) -> np.ndarray:
    """Rotation vector from the symmetric part, for angles near pi."""
    M = (R + R.T) / 2.0 - cos_theta * np.eye(3)  # = (1 - cos(theta)) a a^T
    k = int(np.argmax(np.diag(M)))
    axis = M[:, k] / np.sqrt(M[k, k] * (1.0 - cos_theta))
    axis = axis / np.linalg.norm(axis)
    # The symmetric part gives the axis up to sign; the antisymmetric part fixes it.
    if np.dot(axis, s) < 0.0:
        axis = -axis
    return theta * axis


def log(T: np.ndarray) -> np.ndarray:
    """SE(3) -> se(3)."""
    R, t = T[:3, :3], T[:3, 3]
    phi = _log_rotation(R)
    rho = _v_inv_matrix(phi) @ t
    return np.concatenate([rho, phi])


def compose(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
    return T1 @ T2


def inverse(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    Tinv = np.eye(4)
    Tinv[:3, :3] = R.T
    Tinv[:3, 3] = -R.T @ t
    return Tinv


def adjoint(T: np.ndarray) -> np.ndarray:
    """Adjoint matrix of T."""
    R, t = T[:3, :3], T[:3, 3]
    A = np.zeros((6, 6))
    A[:3, :3] = R
    A[:3, 3:] = hat(t) @ R
    A[3:, 3:] = R
    return A


def _q_matrix(rho: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Off-diagonal block of the SE(3) left Jacobian."""
    theta = np.linalg.norm(phi)
    K, P = hat(phi), hat(rho)
    c1, c2, c3 = _q_coeffs(theta)
    term0 = 0.5 * P
    term1 = c1 * (K @ P + P @ K + K @ P @ K)
    term2 = -c2 * (K @ K @ P + P @ K @ K - 3 * K @ P @ K)
    term3 = -0.5 * c3 * (K @ P @ K @ K + K @ K @ P @ K)
    return term0 + term1 + term2 + term3


def _left_jacobian(xi: np.ndarray) -> np.ndarray:
    rho, phi = xi[:3], xi[3:]
    Jl_phi = _v_matrix(phi)  # the SO(3) left Jacobian is exactly the V matrix above
    Q = _q_matrix(rho, phi)
    J = np.zeros((6, 6))
    J[:3, :3] = Jl_phi
    J[:3, 3:] = Q
    J[3:, 3:] = Jl_phi
    return J


def right_jacobian(xi: np.ndarray) -> np.ndarray:
    """Right Jacobian Jr(xi)."""
    xi = np.asarray(xi, dtype=float)
    return _left_jacobian(-xi)
