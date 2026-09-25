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

# Relative size of a chi2 change too small to be told apart from rounding in
# chi2 itself: a sum of a few dozen terms, each good to about 1e-16, with
# three orders of margin.
_CHI2_RESOLUTION = 1e-12


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
    max_iterations: int = 2000,
    tol: float = 1e-12,
) -> Result:
    """Plain Gauss-Newton. Converges quadratically when started near the optimum.

    Only quadratically when the residuals at the optimum are small, though.
    False loop closures leave large residuals, convergence becomes linear,
    and from dead reckoning with six false closures in twenty it takes about
    sixty iterations. A budget of fifty once recorded every such run as a
    failure, and E4 reported plain least squares as producing no answer at
    all when it produces a wrong one. The budget is a safety net, not a test.
    """
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
    """Levenberg-Marquardt: Gauss-Newton with a trust region.

    Needed where plain Gauss-Newton diverges — poor initialisation, and the
    heavily corrupted graphs of the perceptual-aliasing experiment, where the
    objective is far from quadratic.

    Damping follows Nielsen's gain-ratio rule (Madsen, Nielsen & Tingleff,
    "Methods for Non-Linear Least Squares Problems", 2004). rho compares the
    reduction in chi2 a step achieved with the reduction the quadratic model
    predicted; the damping shrinks smoothly when the model is trustworthy and
    grows geometrically, by a factor that doubles, after each rejection.

    The simpler rule -- divide by ten on success, multiply by ten on failure
    -- was used here first and is a trap in narrow curved valleys, which is
    exactly what large rotational noise produces. It alternated accept and
    reject for thousands of iterations, crawling towards the minimum, and at
    a budget of a hundred iterations it reported those runs as failures. E3
    then dropped them: the most non-linear runs of the sweep, discarded for
    being slow rather than for being wrong.

    Even with a good damping rule, convergence near a large-residual optimum
    is linear rather than quadratic, because Gauss-Newton drops the
    second-order residual terms. The iteration budget is therefore a safety
    net sized well beyond that tail; the failure that matters is damping
    exhaustion, where no step in any trust region improves chi2.
    """
    poses = [np.array(T, dtype=float) for T in (poses or graph.poses)]
    free = free_mask(len(poses), graph.dof, anchor)
    chi2 = graph.chi2(poses)
    converged = False
    iterations = 0
    growth = 2.0

    H, b = graph.linearize(poses)
    for iterations in range(1, max_iterations + 1):
        # Converged when the *undamped* Gauss-Newton step vanishes: the same
        # test gauss_newton applies, so the two solvers agree on what an
        # optimum is. Testing the damped step instead lets a large damping
        # shrink the step below tol short of the optimum. Testing it before
        # the accept/reject decision matters too: at the optimum chi2 can no
        # longer improve, every trial is rejected, and a test on accepted
        # steps only would report failure while sitting on the minimum.
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
            # A rejection close to the optimum can mean only that chi2 no
            # longer resolves the improvement. The full Gauss-Newton step
            # predicts a reduction of -b.newton (the Newton decrement); once
            # that is below the rounding in chi2 itself no trial can ever be
            # accepted, and the estimate is at the optimum to working
            # precision. The decrement is also the NEES-scale size of the
            # error still remaining, which is why it is the right yardstick.
            # That last Gauss-Newton step is then taken unconditionally: it
            # cannot change chi2 measurably, and it lands the estimate, and
            # the H returned with it, on the same optimum gauss_newton finds.
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
