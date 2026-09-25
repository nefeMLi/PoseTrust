"""Gauge fixing: drop one pose's block instead of adding a stiff prior."""

from __future__ import annotations

import numpy as np


def free_mask(n_poses: int, dof: int, anchor: int) -> np.ndarray:
    """Boolean index over the stacked state with the anchored pose removed."""
    if not 0 <= anchor < n_poses:
        raise IndexError(f"anchor {anchor} outside [0, {n_poses})")
    free = np.ones(n_poses * dof, dtype=bool)
    free[anchor * dof : (anchor + 1) * dof] = False
    return free

