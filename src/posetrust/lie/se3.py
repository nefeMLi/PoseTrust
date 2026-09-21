"""SE(3) Lie group: exp/log maps and the right Jacobian, via Rodrigues' formula.

Convention: a pose is a 4x4 homogeneous matrix [[R, t], [0, 1]].
A tangent vector xi = [rho (3), phi (3)] maps to SE(3) via exp(), with
rho the translational part and phi the axis-angle rotation vector —
exp(xi + d) ~= exp(xi) @ exp(right_jacobian(xi) @ d) for small d, mirroring
se2.py's convention and contract.
"""

from __future__ import annotations

import numpy as np

# Crossover between the Taylor series and the direct trigonometric formulas.
# Both branches must be accurate AT this radius. The Q matrix below divides by
# theta**4 and theta**5, so its direct form is still losing double-digit
# significant figures well past theta = 1e-3; the five-term series used below
# stays exact to ~1e-19 out to here. Worst-case error of right_jacobian across
# the whole crossover is 6e-16, measured against 150-digit reference arithmetic
# (float64 needs that much headroom: the Q coefficients cancel ~50 digits).
_EPS = 5e-2

# Crossover to the symmetric-part axis extraction in the log map. The
# antisymmetric part carries the axis scaled by sin(theta), so its accuracy
# decays as 1/(pi - theta); the symmetric branch is well conditioned for any
# theta this close to pi. cos(theta) = -0.999 is theta ~ pi - 0.045, inside
# the band where both branches are accurate to ~1e-15.
_COS_PI_SWITCH = -0.999


# Degrees of freedom of the tangent space, so callers can stay group-agnostic.
DOF = 6

# xi is ordered [translation..., rotation...]; this is where it splits, so a
# per-degree-of-freedom breakdown can separate the two.
TRANSLATION_DOF = 3


def hat(v: np.ndarray) -> np.ndarray:
    """R^3 -> so(3): the skew-symmetric cross-product matrix of v."""
    v = np.asarray(v, dtype=float)
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


def vee(M: np.ndarray) -> np.ndarray:
    """so(3) -> R^3: inverse of hat()."""
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


def _so3_coeffs(theta: float) -> tuple[float, float, float]:
    """a = sin(theta)/theta, b = (1-cos(theta))/theta^2, c = (theta-sin(theta))/theta^3.

    These build the SO(3) rotation matrix and its left Jacobian; all three
    are smooth at theta = 0 but the direct formulas are 0/0 there.
    """
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
    """e in V^-1(phi) = I - 0.5*K + e*K^2 (inverse of the SO(3) left Jacobian)."""
    if abs(theta) < _EPS:
        t2 = theta**2
        return (
            1.0 / 12.0
            + t2 / 720.0
            + t2**2 / 30240.0
            + t2**3 / 1209600.0
            + t2**4 / 47900160.0
        )
    # (1 + cos(theta)) / (2*theta*sin(theta)) == 1 / (2*theta*tan(theta/2)).
    # The cot form has no cancellation: (1 + cos(theta)) loses all significance
    # near theta = pi, which is exactly where the log map's pi branch sends us.
    return 1.0 / theta**2 - 1.0 / (2.0 * theta * np.tan(0.5 * theta))


def _q_coeffs(theta: float) -> tuple[float, float, float]:
    """c1, c2, c3 for the SE(3) Q matrix (Barfoot & Furgale 2014, eq. 102)."""
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
    """so(3) -> SO(3) via Rodrigues' formula. phi is the axis-angle vector."""
    phi = np.asarray(phi, dtype=float)
    theta = np.linalg.norm(phi)
    K = hat(phi)
    a, b, _ = _so3_coeffs(theta)
    return np.eye(3) + a * K + b * (K @ K)


def _v_matrix(phi: np.ndarray) -> np.ndarray:
    """Left Jacobian of SO(3): maps the translational part of xi to SE(3) t."""
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
    """se(3) -> SE(3). xi = [rho (3), phi (3)]. Returns a 4x4 homogeneous matrix."""
    xi = np.asarray(xi, dtype=float)
    rho, phi = xi[:3], xi[3:]
    T = np.eye(4)
    T[:3, :3] = rotation_matrix(phi)
    T[:3, 3] = _v_matrix(phi) @ rho
    return T


def _log_rotation(R: np.ndarray) -> np.ndarray:
    """SO(3) -> so(3), robust at both ends of the range.

    Uses atan2 on the antisymmetric part rather than arccos((trace(R)-1)/2):
    the arccos form has an unbounded derivative near theta = 0, which is
    exactly the regime the optimizer and the Monte Carlo Jacobian check
    both live in for a well-converged estimate.

    Near theta = pi the antisymmetric part vanishes with sin(theta) and stops
    carrying the axis, so the axis is taken from the symmetric part instead
    (see _log_rotation_near_pi). Loop closures between opposite-facing views
    land exactly there, so this is a real operating point, not a corner case.
    """
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
    """Axis from the symmetric part, which stays well conditioned at theta = pi.

    Rodrigues in outer-product form gives
        (R + R.T)/2 = cos(theta) I + (1 - cos(theta)) a a^T,
    so a a^T is recoverable by dividing by (1 - cos(theta)) ~ 2 here. Reading
    the axis off the column with the largest diagonal keeps the square root
    away from zero.
    """
    M = (R + R.T) / 2.0 - cos_theta * np.eye(3)  # = (1 - cos(theta)) a a^T
    k = int(np.argmax(np.diag(M)))
    axis = M[:, k] / np.sqrt(M[k, k] * (1.0 - cos_theta))
    axis = axis / np.linalg.norm(axis)
    # The symmetric part fixes the axis only up to sign; the (tiny) remaining
    # antisymmetric part still carries it. At exactly pi, s == 0 and both
    # signs denote the same rotation, so either is correct.
    if np.dot(axis, s) < 0.0:
        axis = -axis
    return theta * axis


def log(T: np.ndarray) -> np.ndarray:
    """SE(3) -> se(3). Returns xi = [rho (3), phi (3)]."""
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
    """Adj(T), defined by T @ exp(x) @ inverse(T) == exp(Adj(T) @ x).

    Moves a tangent perturbation from one frame to another, which is what
    lets a relative-pose residual be differentiated with respect to both of
    the poses it connects.
    """
    R, t = T[:3, :3], T[:3, 3]
    A = np.zeros((6, 6))
    A[:3, :3] = R
    A[:3, 3:] = hat(t) @ R
    A[3:, 3:] = R
    return A


def _q_matrix(rho: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Off-diagonal block of the SE(3) left Jacobian (Barfoot & Furgale 2014)."""
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
    """Right Jacobian Jr(xi): exp(xi + d) ~= exp(xi) @ exp(Jr(xi) @ d).

    Built from the general Lie-group identity Jr(xi) = Jl(-xi) rather than
    hand-deriving the off-diagonal Q block's closed form under negation —
    fewer places for a sign error to hide. Verified against finite
    differences (see tests/test_se3.py).
    """
    xi = np.asarray(xi, dtype=float)
    return _left_jacobian(-xi)
