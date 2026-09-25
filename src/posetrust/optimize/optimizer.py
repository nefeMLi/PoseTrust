"""Gauss-Newton and Levenberg-Marquardt on the manifold."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from posetrust.graph import PoseGraph
from posetrust.optimize.gauge import free_mask

# Relative chi2 change too small to tell apart from rounding.
_CHI2_RESOLUTION = 1e-12


@dataclass
class Result:
    """Optimiser output, including the final H."""

    poses: list[np.ndarray]
    chi2: float
    iterations: int
    converged: bool
    information: np.ndarray
    anchor: int


def solve_step(
    H: np.ndarray, b: np.ndarray, free: np.ndarray, damping: float = 0.0
) -> np.ndarray:
    """Solve (H + damping diag(H)) delta = -b over the free poses."""
    Hf = H[np.ix_(free, free)]
    if damping > 0.0:
        Hf = Hf + damping * np.diag(np.diag(Hf))
    delta = np.zeros(H.shape[0])
    delta[free] = np.linalg.solve(Hf, -b[free])
    return delta


def gauss_newton(
    graph: PoseGraph,
    poses: list[np.ndarray] | None = None,
    anchor: int = 0,
    max_iterations: int = 2000,
    tol: float = 1e-12,
) -> Result:
    """Gauss-Newton."""
    poses = [np.array(T, dtype=float) for T in (poses or graph.poses)]
    free = free_mask(len(poses), graph.dof, anchor)
    converged = False
    iterations = 0

    H, b = graph.linearize(poses)
    for iterations in range(1, max_iterations + 1):
        delta = solve_step(H, b, free)
        poses = graph.retract(poses, delta)
        H, b = graph.linearize(poses)
        if np.linalg.norm(delta) < tol:
            converged = True
            break

    return Result(poses, graph.chi2(poses), iterations, converged, H, anchor)


def levenberg_marquardt(
    graph: PoseGraph,
    poses: list[np.ndarray] | None = None,
    anchor: int = 0,
    max_iterations: int = 2000,
    tol: float = 1e-12,
    damping: float = 1e-4,
) -> Result:
    """Levenberg-Marquardt with Nielsen's damping update."""
    poses = [np.array(T, dtype=float) for T in (poses or graph.poses)]
    free = free_mask(len(poses), graph.dof, anchor)
    chi2 = graph.chi2(poses)
    converged = False
    iterations = 0
    growth = 2.0

    H, b = graph.linearize(poses)
    for iterations in range(1, max_iterations + 1):
        # Test the undamped step, as gauss_newton does, and before accept/reject:
        # at the optimum every trial is rejected.
        newton = solve_step(H, b, free)
        if np.linalg.norm(newton) < tol:
            converged = True
            break

        delta = solve_step(H, b, free, damping)

        trial = graph.retract(poses, delta)
        trial_chi2 = graph.chi2(trial)

        if trial_chi2 < chi2:
            # With (H + damping*D) delta = -b, the model chi2 + 2 b.delta +
            # delta.H.delta predicts a drop of delta.(damping*D*delta - b).
            d = delta[free]
            scaled = damping * np.diag(H)[free] * d
            predicted = float(d @ (scaled - b[free]))
            rho = (chi2 - trial_chi2) / predicted
            poses, chi2 = trial, trial_chi2
            damping *= max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
            growth = 2.0
            H, b = graph.linearize(poses)
        else:
            # Rejected, but the Newton decrement is below chi2's rounding, so this is
            # the optimum. Take the last GN step so the result matches gauss_newton.
            decrement = -float(b[free] @ newton[free])
            if decrement <= _CHI2_RESOLUTION * max(chi2, 1.0):
                poses = graph.retract(poses, newton)
                chi2 = graph.chi2(poses)
                H, b = graph.linearize(poses)
                converged = True
                break
            damping *= growth
            growth *= 2.0
            if damping > 1e12:
                break

    return Result(poses, chi2, iterations, converged, H, anchor)
