"""Robust kernels, IRLS and graduated non-convexity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from posetrust.graph import PoseGraph
from posetrust.optimize.gauge import free_mask
from posetrust.optimize.optimizer import Result, solve_step


def chi2_threshold(dof: int, quantile: float = 0.95) -> float:
    """Chi-squared quantile for dof degrees of freedom."""
    return float(chi2.ppf(quantile, dof))


@dataclass(frozen=True)
class Trivial:
    """Plain least squares."""

    def weight(self, s: np.ndarray) -> np.ndarray:
        return np.ones_like(s)

    def cost(self, s: np.ndarray) -> np.ndarray:
        return s


@dataclass(frozen=True)
class Huber:
    """Huber kernel (convex)."""

    delta: float

    def weight(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        return np.where(s <= self.delta**2, 1.0, self.delta / np.sqrt(np.maximum(s, 1e-300)))

    def cost(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        return np.where(s <= self.delta**2, s, 2.0 * self.delta * np.sqrt(s) - self.delta**2)


@dataclass(frozen=True)
class Cauchy:
    """Cauchy kernel (redescending)."""

    c: float

    def weight(self, s: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.asarray(s, dtype=float) / self.c**2)

    def cost(self, s: np.ndarray) -> np.ndarray:
        return self.c**2 * np.log1p(np.asarray(s, dtype=float) / self.c**2)


@dataclass(frozen=True)
class SwitchableConstraints:
    """Switchable constraints in closed form (dynamic covariance scaling)."""

    phi: float

    def _scale(self, s: np.ndarray) -> np.ndarray:
        return np.minimum(1.0, 2.0 * self.phi / (self.phi + np.asarray(s, dtype=float)))

    def weight(self, s: np.ndarray) -> np.ndarray:
        return self._scale(s) ** 2

    def cost(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        return np.where(s <= self.phi, s, 3.0 * self.phi - 4.0 * self.phi**2 / (self.phi + s))


@dataclass(frozen=True)
class GemanMcClure:
    """Geman-McClure kernel, with a scale mu for GNC."""

    c: float
    mu: float = 1.0

    def weight(self, s: np.ndarray) -> np.ndarray:
        a = self.mu * self.c**2
        return (a / (np.asarray(s, dtype=float) + a)) ** 2

    def cost(self, s: np.ndarray) -> np.ndarray:
        a = self.mu * self.c**2
        s = np.asarray(s, dtype=float)
        return a * s / (s + a)


def squared_residuals(graph: PoseGraph, poses: list[np.ndarray]) -> np.ndarray:
    """r^T Omega r for every factor."""
    out = np.empty(len(graph.factors))
    for index, factor in enumerate(graph.factors):
        r = graph.residual(factor, poses)
        out[index] = r @ factor.information @ r
    return out


def loop_closure_indices(graph: PoseGraph) -> np.ndarray:
    """Indices of the factors that are not odometry."""
    return np.array(
        [k for k, f in enumerate(graph.factors) if f.j != f.i + 1], dtype=int
    )


def _robust_mask(graph: PoseGraph, robust_factors: np.ndarray | None) -> np.ndarray:
    mask = np.zeros(len(graph.factors), dtype=bool)
    if robust_factors is None:
        mask[:] = True
    else:
        mask[np.asarray(robust_factors, dtype=int)] = True
    return mask


def factor_weights(
    kernel, s: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Kernel weights on the masked factors, 1 everywhere else."""
    weights = np.ones_like(s)
    if mask.any():
        weights[mask] = kernel.weight(s[mask])
    return weights


def robust_cost(kernel, s: np.ndarray, mask: np.ndarray) -> float:
    """Total cost: rho(s) on robust factors, s elsewhere."""
    total = float(s[~mask].sum())
    if mask.any():
        total += float(np.sum(kernel.cost(s[mask])))
    return total


def irls(
    graph: PoseGraph,
    poses: list[np.ndarray] | None = None,
    kernel=None,
    anchor: int = 0,
    robust_factors: np.ndarray | None = None,
    max_iterations: int = 500,
    tol: float = 1e-12,
) -> Result:
    """Iteratively reweighted least squares with a fixed kernel."""
    kernel = kernel or Trivial()
    poses = [np.array(T, dtype=float) for T in (poses or graph.poses)]
    free = free_mask(len(poses), graph.dof, anchor)
    mask = _robust_mask(graph, robust_factors)
    converged = False
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        s = squared_residuals(graph, poses)
        weights = factor_weights(kernel, s, mask)
        H, b = graph.linearize(poses, weights=weights)
        delta = solve_step(H, b, free)
        poses = graph.retract(poses, delta)
        if np.linalg.norm(delta) < tol:
            converged = True
            break

    s = squared_residuals(graph, poses)
    weights = factor_weights(kernel, s, mask)
    H, _ = graph.linearize(poses, weights=weights)
    return Result(
        poses, robust_cost(kernel, s, mask), iterations, converged, H, anchor
    )


def graduated_non_convexity(
    graph: PoseGraph,
    poses: list[np.ndarray] | None = None,
    c: float = 1.0,
    anchor: int = 0,
    robust_factors: np.ndarray | None = None,
    mu_factor: float = 1.4,
    inner_iterations: int = 3,
    max_iterations: int = 500,
    tol: float = 1e-12,
) -> Result:
    """Graduated non-convexity down to Geman-McClure, finished by IRLS."""
    poses = [np.array(T, dtype=float) for T in (poses or graph.poses)]
    free = free_mask(len(poses), graph.dof, anchor)
    mask = _robust_mask(graph, robust_factors)

    s = squared_residuals(graph, poses)
    mu = max(1.0, 2.0 * float(s.max()) / c**2)
    annealing = 0

    while mu > 1.0:
        kernel = GemanMcClure(c, mu)
        for _ in range(inner_iterations):
            annealing += 1
            s = squared_residuals(graph, poses)
            weights = factor_weights(kernel, s, mask)
            H, b = graph.linearize(poses, weights=weights)
            poses = graph.retract(poses, solve_step(H, b, free))
        mu = max(1.0, mu / mu_factor)

    final = irls(
        graph,
        poses,
        GemanMcClure(c, 1.0),
        anchor=anchor,
        robust_factors=robust_factors,
        max_iterations=max_iterations,
        tol=tol,
    )
    return Result(
        final.poses,
        final.chi2,
        annealing + final.iterations,
        final.converged,
        final.information,
        anchor,
    )
