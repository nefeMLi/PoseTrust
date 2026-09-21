"""Trajectory and noise simulation, and the Monte Carlo harness: parameterized
trajectory generators with controllable loop-closure density, noise
anisotropy and outlier rate, solved over many independent noise realizations
to build the empirical sampling distribution of the estimate.

This is why the study has to be simulated. The reported covariance claims to
describe the distribution the estimate would take if the same measurements
were collected again with fresh noise. On real data that distribution is
unobservable -- there is one dataset and one answer, and no way to ask what
else might have happened. Generating many noise realizations of the *same*
measurement set makes it observable, and the spread of the resulting estimates
is exactly the thing the covariance is a claim about.

What is held fixed across runs matters as much as what varies. The trajectory,
which poses are connected, and which of those connections are false are all
part of the experimental condition and are drawn once. Only the measurement
noise is redrawn. Resampling the graph structure each run would blur several
effects together and the resulting spread would answer no clean question.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from posetrust.covariance import covariance_matrix
from posetrust.graph import PoseGraph
from posetrust.optimize.gauge import free_mask
from posetrust.optimize.optimizer import gauss_newton
from posetrust.stats import nees, tangent_error


@dataclass(frozen=True)
class NoiseModel:
    """Zero-mean tangent-space noise with per-degree-of-freedom scales.

    Anisotropy is deliberate and is one of the axes the study sweeps: odometry
    is typically far more certain along the direction of travel than across
    it, and rotational and translational uncertainty are not even in the same
    units. A scalar noise level would quietly assume the problem away.
    """

    sigma: np.ndarray

    def __post_init__(self) -> None:
        sigma = np.asarray(self.sigma, dtype=float)
        if sigma.ndim != 1 or np.any(sigma <= 0.0):
            raise ValueError(
                "noise sigma must be a 1-D vector of strictly positive scales; "
                "a zero scale means infinite information, which has no "
                "information matrix. For effectively exact measurements use a "
                "small sigma instead."
            )

    @property
    def covariance(self) -> np.ndarray:
        return np.diag(np.asarray(self.sigma, dtype=float) ** 2)

    @property
    def information(self) -> np.ndarray:
        return np.diag(1.0 / np.asarray(self.sigma, dtype=float) ** 2)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(0.0, np.asarray(self.sigma, dtype=float))


@dataclass
class Scenario:
    """One experimental condition: the structure that stays fixed across runs."""

    truth: list[np.ndarray]
    edges: list[tuple[int, int]]
    outliers: frozenset[tuple[int, int]] = field(default_factory=frozenset)

    @property
    def n_poses(self) -> int:
        return len(self.truth)


def curved_trajectory(
    lie, n_poses: int = 20, step: float = 1.0, turn: float = 0.2
) -> list[np.ndarray]:
    """A chain that advances `step` and turns `turn` radians each pose.

    Curvature is not decoration. On a straight line the rotation blocks stay
    near identity, the problem is nearly linear, and the manifold effects the
    study exists to measure would never appear.
    """
    increment = np.zeros(lie.DOF)
    increment[0] = step
    increment[lie.DOF - 1] = turn
    poses = [lie.exp(np.zeros(lie.DOF))]
    for _ in range(n_poses - 1):
        poses.append(lie.compose(poses[-1], lie.exp(increment)))
    return poses


def odometry_edges(n_poses: int) -> list[tuple[int, int]]:
    """Consecutive constraints -- the backbone every pose graph has."""
    return [(k, k + 1) for k in range(n_poses - 1)]


def loop_closure_edges(
    n_poses: int,
    density: float,
    rng: np.random.Generator,
    min_separation: int = 3,
) -> list[tuple[int, int]]:
    """Sample loop closures at `density` closures per pose.

    density = 0 is the odometry-only regime an edge system spends most of its
    time in, and is the sparse end of the E2 sweep. Pairs closer together than
    `min_separation` are excluded because they duplicate odometry rather than
    closing anything.
    """
    candidates = [
        (i, j)
        for i in range(n_poses)
        for j in range(i + min_separation, n_poses)
    ]
    if not candidates:
        return []
    count = min(int(round(density * n_poses)), len(candidates))
    chosen = rng.choice(len(candidates), size=count, replace=False)
    return [candidates[k] for k in sorted(chosen)]


def make_scenario(
    lie,
    n_poses: int = 20,
    loop_density: float = 0.2,
    outlier_rate: float = 0.0,
    seed: int = 0,
    turn: float = 0.2,
) -> Scenario:
    """Draw one experimental condition: trajectory, connectivity, and which
    closures are false.

    Outliers are marked here rather than generated here so that the same
    corrupted edges are reused across every Monte Carlo run, which is what
    makes the aliasing condition a property of the condition rather than
    noise on top of it.
    """
    rng = np.random.default_rng(seed)
    truth = curved_trajectory(lie, n_poses, turn=turn)
    closures = loop_closure_edges(n_poses, loop_density, rng)
    edges = odometry_edges(n_poses) + closures

    n_bad = int(round(outlier_rate * len(closures)))
    bad = rng.choice(len(closures), size=n_bad, replace=False) if n_bad else []
    return Scenario(truth, edges, frozenset(closures[k] for k in bad))


def sample_graph(
    lie, scenario: Scenario, noise: NoiseModel, rng: np.random.Generator
) -> PoseGraph:
    """One noise realization of the fixed measurement set.

    A true measurement is the exact relative pose perturbed on the right,
    Z = (Ti^-1 Tj) @ exp(eps) with eps ~ N(0, noise.covariance), which makes
    the residual exactly -eps and so gives the information matrix its stated
    meaning.

    A false one keeps the observing pose and corrupts the endpoint: the graph
    is told that pose i saw place b when it actually saw place j. That is the
    shape of the real failure -- perceptual aliasing is a front end matching
    the wrong place confidently, not a large random error -- and it is why the
    constraint is self-consistent enough to fool a least-squares back end.

    The wrong endpoint is drawn away from j deliberately. Sampling an
    unrelated pair at random would occasionally reproduce the true relative
    pose, so a condition labelled "30% outliers" would quietly contain fewer,
    and E4's headline axis would not mean what it says.
    """
    graph = PoseGraph(lie)
    for T in scenario.truth:
        graph.add_pose(T)

    n = scenario.n_poses
    for i, j in scenario.edges:
        if (i, j) in scenario.outliers:
            wrong = [b for b in range(n) if b not in (i, j)]
            b = int(rng.choice(wrong))
            relative = lie.compose(
                lie.inverse(scenario.truth[i]), scenario.truth[b]
            )
        else:
            relative = lie.compose(
                lie.inverse(scenario.truth[i]), scenario.truth[j]
            )
        measurement = lie.compose(relative, lie.exp(noise.sample(rng)))
        graph.add_factor(i, j, measurement, noise.information)
    return graph


def dead_reckon(lie, graph: PoseGraph, n_poses: int) -> list[np.ndarray]:
    """Chain the odometry measurements together: what a real system starts from."""
    poses = [lie.exp(np.zeros(lie.DOF))]
    odometry = {(f.i, f.j): f.measurement for f in graph.factors}
    for k in range(n_poses - 1):
        poses.append(lie.compose(poses[-1], odometry[(k, k + 1)]))
    return poses


@dataclass
class MonteCarloResult:
    """Per-run errors and reported covariances -- the raw material for NEES.

    Full state covariances are not retained: at a thousand poses each one is
    larger than the whole rest of the run put together. The full-state NEES is
    computed while the matrix is still in hand and only the scalar is kept,
    alongside the per-pose marginals that the by-degree-of-freedom breakdown
    and the ellipsoid coverage plots need.
    """

    errors: np.ndarray
    marginals: np.ndarray
    nees_full: np.ndarray
    converged: np.ndarray
    anchor: int
    dof: int

    @property
    def n_runs(self) -> int:
        return self.errors.shape[0]

    @property
    def free_dof(self) -> int:
        """Degrees of freedom of the full-state NEES, after the gauge is fixed."""
        return (self.errors.shape[1] - 1) * self.dof

    def pose_errors(self, k: int) -> np.ndarray:
        return self.errors[:, k, :]

    def pose_marginals(self, k: int) -> np.ndarray:
        return self.marginals[:, k, :, :]


def monte_carlo(
    lie,
    scenario: Scenario,
    noise: NoiseModel,
    n_runs: int = 200,
    anchor: int = 0,
    seed: int = 0,
    solver=None,
    initialize: str = "truth",
) -> MonteCarloResult:
    """Solve the same measurement set under `n_runs` independent noise draws.

    The anchor is held at its true pose in every run. Without that the
    estimates would each sit in their own arbitrary frame and the spread
    across runs would be dominated by gauge freedom rather than by estimation
    error -- the covariance would look enormous and the comparison would mean
    nothing.

    `solver` is any callable (graph, poses, anchor) -> Result, so the same
    harness measures plain least squares, Levenberg-Marquardt, or any robust
    back-end. E4 needs exactly that: the comparison it makes is between
    estimators on identical noise draws, which only holds if nothing else
    about the run changes with the estimator.

    `initialize="truth"` starts each solve at the true trajectory. That is
    deliberate: the question here is whether the covariance at the optimum is
    honest, not whether the optimizer can find the optimum from far away.
    Starting elsewhere would fold convergence failures into a calibration
    measurement and make a bad result impossible to attribute. Use
    "odometry" to fold that in on purpose, as the aliasing experiment does.

    The loop is embarrassingly parallel; it is serial here because at the
    graph sizes the study uses each solve is milliseconds.
    """
    dof = lie.DOF
    n_poses = scenario.n_poses
    solver = solver or gauss_newton
    free = free_mask(n_poses, dof, anchor)
    seeds = np.random.SeedSequence(seed).spawn(n_runs)

    errors = np.zeros((n_runs, n_poses, dof))
    marginals = np.zeros((n_runs, n_poses, dof, dof))
    nees_full = np.zeros(n_runs)
    converged = np.zeros(n_runs, dtype=bool)

    for r, child in enumerate(seeds):
        rng = np.random.default_rng(child)
        graph = sample_graph(lie, scenario, noise, rng)

        if initialize == "truth":
            start = list(scenario.truth)
        else:
            start = dead_reckon(lie, graph, n_poses)
            start[anchor] = scenario.truth[anchor]

        result = solver(graph, start, anchor=anchor)
        converged[r] = result.converged

        for k in range(n_poses):
            errors[r, k] = tangent_error(lie, result.poses[k], scenario.truth[k])

        # one inversion per run: the marginals are blocks of the same matrix
        sigma = covariance_matrix(result.information, anchor, dof)
        marginals[r] = np.array(
            [sigma[k * dof : (k + 1) * dof, k * dof : (k + 1) * dof] for k in range(n_poses)]
        )
        nees_full[r] = nees(errors[r].reshape(-1)[free], sigma[np.ix_(free, free)])

    return MonteCarloResult(errors, marginals, nees_full, converged, anchor, dof)
