"""The state covariance by selected inversion of the Cholesky factor."""

from __future__ import annotations

import numpy as np

from posetrust.optimizer import free_mask


def cholesky_factor(H: np.ndarray) -> np.ndarray:
    """Lower-triangular L with H = L L^T."""
    try:
        return np.linalg.cholesky(H)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            "information matrix is not positive definite; is the gauge fixed? "
            "An unanchored pose graph is singular by construction."
        ) from exc


def selected_inverse(L: np.ndarray) -> np.ndarray:
    """Inverse of L L^T by the Takahashi recursion."""
    n = L.shape[0]
    sigma = np.zeros((n, n))
    for i in range(n - 1, -1, -1):
        below = L[i + 1 :, i]
        if below.size:
            off = -(below @ sigma[i + 1 :, i + 1 :]) / L[i, i]
            sigma[i, i + 1 :] = off
            sigma[i + 1 :, i] = off
            sigma[i, i] = (1.0 / L[i, i] - below @ sigma[i + 1 :, i]) / L[i, i]
        else:
            sigma[i, i] = 1.0 / L[i, i] ** 2
    return sigma


def covariance_matrix(information: np.ndarray, anchor: int, dof: int) -> np.ndarray:
    """Full state covariance; the anchored pose's block is zero."""
    n_poses = information.shape[0] // dof
    free = free_mask(n_poses, dof, anchor)
    L = cholesky_factor(information[np.ix_(free, free)])
    sigma = np.zeros_like(information)
    sigma[np.ix_(free, free)] = selected_inverse(L)
    return sigma
