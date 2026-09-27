"""Covariances to report for a robust estimate: naive, sandwich and inlier."""

from __future__ import annotations

import numpy as np

from posetrust.graph import PoseGraph
from posetrust.optimizer import free_mask
from posetrust.robust import Trivial

NAMES = ("naive", "sandwich", "inlier")


def whitened_factors(
    graph: PoseGraph, poses: list[np.ndarray], anchor: int
) -> tuple[np.ndarray, np.ndarray]:
    """Whitened residuals e (m, d) and Jacobians J (m, d, n) over the free state."""
    d = graph.dof
    n = len(poses) * d
    free = free_mask(len(poses), d, anchor)
    residuals, jacobians = [], []
    for factor in graph.factors:
        # Omega = L L^T, so e = L^T r has e^T e = r^T Omega r.
        L = np.linalg.cholesky(factor.information)
        Ji, Jj = graph.factor_jacobians(factor, poses)
        J = np.zeros((d, n))
        J[:, factor.i * d : (factor.i + 1) * d] = Ji
        J[:, factor.j * d : (factor.j + 1) * d] = Jj
        residuals.append(L.T @ graph.residual(factor, poses))
        jacobians.append(L.T @ J[:, free])
    return np.array(residuals), np.array(jacobians)


def _inverse_spd(M: np.ndarray) -> np.ndarray | None:
    """Inverse of a symmetric positive definite matrix, or None if it isn't."""
    try:
        L = np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        return None
    L_inv = np.linalg.inv(L)
    return L_inv.T @ L_inv


def robust_covariances(
    graph: PoseGraph,
    poses: list[np.ndarray],
    kernel,
    robust_factors: np.ndarray,
    threshold: float,
    anchor: int = 0,
) -> dict[str, np.ndarray | None]:
    """Free-state covariances of one converged robust estimate.

    naive     (sum w J^T J)^-1, what the back-ends report
    sandwich  A^-1 (sum w^2 J^T J) A^-1, A the robust cost's Gauss-Newton Hessian
    inlier    (sum J^T J)^-1 over the factors below the threshold

    None marks a covariance that does not exist for this estimate, because
    its matrix is not positive definite.
    """
    e, J = whitened_factors(graph, poses, anchor)
    s = np.einsum("md,md->m", e, e)
    robust = np.zeros(len(s), dtype=bool)
    robust[robust_factors] = True
    kernel = kernel or Trivial()

    w = np.ones_like(s)
    curvature = np.zeros_like(s)
    w[robust] = kernel.weight(s[robust])
    curvature[robust] = kernel.curvature(s[robust])

    gram = np.einsum("mdi,mdj->mij", J, J)
    score = np.einsum("mdi,md->mi", J, e)
    outer = np.einsum("mi,mj->mij", score, score)

    information = np.einsum("m,mij->ij", w, gram)
    A = information + 2.0 * np.einsum("m,mij->ij", curvature, outer)
    B = np.einsum("m,mij->ij", w**2, gram)
    A_inv = _inverse_spd(A)

    kept = ~robust | (s <= threshold)
    return {
        "naive": _inverse_spd(information),
        "sandwich": None if A_inv is None else A_inv @ B @ A_inv,
        "inlier": _inverse_spd(gram[kept].sum(axis=0)),
    }
