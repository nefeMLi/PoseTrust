"""Gauss-Newton and Levenberg-Marquardt on the manifold, with a retraction
(exp update) each iteration.

Gauge handling lives in gauge.py: without it the normal equations are
singular and cannot be solved at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from posetrust.graph import PoseGraph
from posetrust.optimize.gauge import free_mask


@dataclass
class Result:
    """Outcome of an optimisation, including the system the covariance comes from."""

    poses: list[np.ndarray]
    chi2: float
    iterations: int
    converged: bool
    information: np.ndarray
    anchor: int


def solve_step(
    H: np.ndarray, b: np.ndarray, free: np.ndarray, damping: float = 0.0
) -> np.ndarray:
    """Solve (H + damping*diag(H)) delta = -b over the un-anchored poses only.

    The anchored block is dropped rather than penalised with a large prior;
    see gauge.py for why that distinction matters here.
    """
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
    max_iterations: int = 50,
    tol: float = 1e-12,
) -> Result:
    """Plain Gauss-Newton. Converges quadratically when started near the optimum."""
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
    max_iterations: int = 100,
    tol: float = 1e-12,
    damping: float = 1e-4,
) -> Result:
    """Levenberg-Marquardt: Gauss-Newton with a trust region.

    Needed where plain Gauss-Newton diverges — poor initialisation, and the
    heavily corrupted graphs of the perceptual-aliasing experiment, where the
    objective is far from quadratic.
    """
    poses = [np.array(T, dtype=float) for T in (poses or graph.poses)]
    free = free_mask(len(poses), graph.dof, anchor)
    chi2 = graph.chi2(poses)
    converged = False
    iterations = 0

    H, b = graph.linearize(poses)
    for iterations in range(1, max_iterations + 1):
        delta = solve_step(H, b, free, damping)

        # A step too small to matter means the optimum has been reached,
        # whether or not it would have been accepted. Testing this only on
        # accepted steps misreads the endgame: at the optimum chi2 can no
        # longer improve, so LM correctly rejects every trial, damping climbs
        # until the step vanishes, and the solver would report failure while
        # sitting exactly on the minimum it was asked to find.
        if np.linalg.norm(delta) < tol:
            converged = True
            break

        trial = graph.retract(poses, delta)
        trial_chi2 = graph.chi2(trial)

        if trial_chi2 < chi2:
            poses, chi2 = trial, trial_chi2
            damping = max(damping * 0.1, 1e-12)
            H, b = graph.linearize(poses)
        else:
            damping *= 10.0
            if damping > 1e12:
                break

    return Result(poses, chi2, iterations, converged, H, anchor)
