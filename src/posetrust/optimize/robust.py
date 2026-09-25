"""Robust back-ends for perceptual aliasing (E4): Huber, Cauchy, switchable
constraints, and graduated non-convexity.

A single false loop closure is a confident, self-consistent constraint that
happens to be about the wrong place. Least squares has no defence against it:
the quadratic cost rewards splitting the difference, so one bad edge drags the
whole trajectory. Robust kernels replace the quadratic with something that
grows more slowly, so a residual large enough to be implausible stops
dominating.

Every kernel here is expressed as a pair -- a cost rho(s) and the weight it
induces, w(s) = d rho / ds, both as functions of the squared Mahalanobis
residual s = r^T Omega r. Iteratively reweighted least squares then reuses the
ordinary machinery unchanged, with Omega scaled by w at each relinearization.

The consequence that matters for this project: those weights reach H, so the
covariance a robust method reports describes a *reweighted* problem. Robust
estimation is known to rescue the trajectory. Whether it also rescues the
uncertainty is the question E4 exists to answer, and nothing in the derivation
of these kernels promises that it does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from posetrust.graph import PoseGraph
from posetrust.optimize.gauge import free_mask
from posetrust.optimize.optimizer import Result, solve_step


def chi2_threshold(dof: int, quantile: float = 0.95) -> float:
    """Squared-residual level a correct measurement exceeds only `1-quantile` of the time.

    Kernel parameters should be set from this rather than guessed: under a
    correct noise model s = r^T Omega r is chi-squared on the measurement's
    degrees of freedom, so the scale at which a residual becomes suspicious is
    a property of the problem, not a tuning knob.
    """
    return float(chi2.ppf(quantile, dof))


@dataclass(frozen=True)
class Trivial:
    """Plain least squares. The baseline E4 compares everything against."""

    def weight(self, s: np.ndarray) -> np.ndarray:
        return np.ones_like(s)

    def cost(self, s: np.ndarray) -> np.ndarray:
        return s


@dataclass(frozen=True)
class Huber:
    """Quadratic near zero, linear beyond `delta` -- convex, so it cannot create
    local minima, but it never fully rejects an outlier either.
    """

    delta: float

    def weight(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        return np.where(s <= self.delta**2, 1.0, self.delta / np.sqrt(np.maximum(s, 1e-300)))

    def cost(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        return np.where(s <= self.delta**2, s, 2.0 * self.delta * np.sqrt(s) - self.delta**2)


@dataclass(frozen=True)
class Cauchy:
    """Redescending: weight falls off as 1/s, so gross outliers are nearly
    ignored. Non-convex, so the result depends on where the solve started.
    """

    c: float

    def weight(self, s: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.asarray(s, dtype=float) / self.c**2)

    def cost(self, s: np.ndarray) -> np.ndarray:
        return self.c**2 * np.log1p(np.asarray(s, dtype=float) / self.c**2)


@dataclass(frozen=True)
class SwitchableConstraints:
    """Dynamic covariance scaling: the closed form of switchable constraints.

    Switchable constraints attach a latent switch in [0, 1] to every loop
    closure and optimize it jointly with the trajectory, under a prior that
    pulls it towards 1. Minimising that objective over the switch has a
    closed form, s_scale = min(1, 2*phi/(phi + s)), which gives the same
    answer without enlarging the state -- so the extra variables are omitted
    here rather than carried and immediately eliminated.
    """

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
    """Strongly redescending, bounded cost. The surrogate GNC anneals.

    `mu` controls how non-convex it is: large mu is almost plain least
    squares, mu = 1 is Geman-McClure proper. Graduated non-convexity walks
    from one to the other.
    """

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
    """s_i = r_i^T Omega_i r_i for every factor: chi-squared under a correct model."""
    out = np.empty(len(graph.factors))
    for index, factor in enumerate(graph.factors):
        r = graph.residual(factor, poses)
        out[index] = r @ factor.information @ r
    return out


def loop_closure_indices(graph: PoseGraph) -> np.ndarray:
    """Factors that are not consecutive odometry.

    Robust kernels are normally applied only to these: odometry comes from a
    different sensing process and is not subject to place-recognition error,
    so down-weighting it discards good information for nothing.
    """
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
    """Total objective: rho(s) where the kernel applies, s where it does not."""
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
    """Iteratively reweighted least squares with a fixed kernel.

    Each iteration recomputes the weights from the current residuals, then
    takes an ordinary Gauss-Newton step against the reweighted system. The
    returned `information` is that reweighted H -- the covariance a robust
    back-end would actually report, and the object E4 puts on trial.

    `chi2` on the result is the robust objective, not the least-squares one;
    the two are not comparable across kernels, which is why E4 compares
    trajectory error and NEES instead.

    The iteration budget is large on purpose. Reweighting converges linearly,
    not quadratically like Gauss-Newton: with several false closures a convex
    kernel can shrink the step by only ten percent an iteration and need a few
    hundred to settle. A tight budget would record those runs as failures,
    and the analysis would then drop them -- an artefact of the budget
    reported as a property of the kernel.
    """
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
    """Anneal from a near-convex surrogate down to Geman-McClure.

    The redescending kernels are non-convex, so a solve started from a
    trajectory that outliers have already dragged out of shape can settle into
    whichever bad minimum is nearest. Graduated non-convexity avoids choosing
    a starting point at all: it begins with a surrogate so heavily smoothed
    that it is effectively least squares, and sharpens it only as the estimate
    improves, carrying the solution along the way.

    mu starts at 2*max(s)/c^2, large enough that the surrogate is convex over
    the residuals actually present, and is divided by `mu_factor` until it
    reaches 1, taking `inner_iterations` reweighted steps at each level. At
    mu = 1 the surrogate is Geman-McClure proper, and the final solve is
    plain IRLS on it, so convergence is judged by the same rule and budget as
    every other robust back-end.
    """
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
