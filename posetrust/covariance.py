"""The state covariance: the inverse of the information matrix over the free poses."""

from __future__ import annotations

import numpy as np

from posetrust.optimizer import free_mask


def covariance_matrix(information: np.ndarray, anchor: int, dof: int) -> np.ndarray:
    """Full state covariance; the anchored pose's block is zero."""
    free = free_mask(information.shape[0] // dof, dof, anchor)
    sigma = np.zeros_like(information)
    sigma[np.ix_(free, free)] = np.linalg.inv(information[np.ix_(free, free)])
    return sigma
