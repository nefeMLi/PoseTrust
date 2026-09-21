"""Explicit gauge fixing. Without anchoring, the information matrix is
singular and the covariance is meaningless.

Every relative-pose measurement is invariant to rigidly transforming the whole
graph, so H always has a null space of exactly DOF dimensions -- 3 for SE(2),
6 for SE(3). That is not numerical noise to be regularised away; it is the
problem statement saying the absolute frame is unobservable. Nothing here
chooses a *better* trajectory, only a representative of the family the data
cannot distinguish between.

Two ways to do it, and the difference matters for this project:

  anchor (used here)  Drop the anchored pose's block from the system entirely.
                      Its covariance is then exactly zero, which is a property
                      that can be tested.

  stiff prior         Add a large information term on one pose. Easier to
                      implement, but it only makes that pose's covariance
                      small rather than zero, and the leftover mass leaks into
                      every other marginal. For a study measuring whether
                      covariances are honest, that is a contaminant.
"""

from __future__ import annotations

import numpy as np


def free_mask(n_poses: int, dof: int, anchor: int) -> np.ndarray:
    """Boolean index over the stacked state with the anchored pose removed."""
    if not 0 <= anchor < n_poses:
        raise IndexError(f"anchor {anchor} outside [0, {n_poses})")
    free = np.ones(n_poses * dof, dtype=bool)
    free[anchor * dof : (anchor + 1) * dof] = False
    return free


def gauge_dimension(information: np.ndarray, tol: float = 1e-8) -> int:
    """Number of near-null directions of H -- should equal the group's DOF.

    A diagnostic rather than a routine call: if this comes back larger than
    DOF the graph has a disconnected component or an unconstrained pose, and
    anchoring one pose will not be enough to make the system solvable.
    """
    return int(np.sum(np.abs(np.linalg.eigvalsh(information)) < tol))
