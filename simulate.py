"""Scenarios, measurement noise and the Monte Carlo harness."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from covariance import covariance_matrix
from graph import PoseGraph
from optimizer import free_mask, gauss_newton
from stats import nees, tangent_error


@dataclass(frozen=True)
class NoiseModel:
    """Zero-mean tangent-space noise with per-DOF sigmas."""

    sigma: np.ndarray

    def __post_init__(self) -> None:
        sigma = np.asarray(self.sigma, dtype=float)
        if sigma.ndim != 1 or np.any(sigma <= 0.0):
            raise ValueError("noise sigma must be a 1-D vector of strictly positive values")

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
    """Truth, edges and false closures for one condition."""

    truth: list[np.ndarray]
    edges: list[tuple[int, int]]
    outliers: Mapping[tuple[int, int], int] = field(default_factory=dict)

    @property
    def n_poses(self) -> int:
        return len(self.truth)


def _exact_count(value: float, what: str) -> int:
    """value as an int, or ValueError if it isn't a whole number."""
    count = round(float(value))
    if abs(value - count) > 1e-9:
        raise ValueError(
            f"{what} is {value:g}, not a whole number; choose values that give "
            "an exact count rather than letting it be rounded"
        )
    return count


def curved_trajectory(
    lie, n_poses: int = 20, step: float = 1.0, turn: float = 0.2
) -> list[np.ndarray]:
    """Poses that step forward and turn by a fixed angle each time."""
    increment = np.zeros(lie.DOF)
    increment[0] = step
    increment[lie.DOF - 1] = turn
    poses = [lie.exp(np.zeros(lie.DOF))]
    for _ in range(n_poses - 1):
        poses.append(lie.compose(poses[-1], lie.exp(increment)))
    return poses


def odometry_edges(n_poses: int) -> list[tuple[int, int]]:
    """Edges between consecutive poses."""
    return [(k, k + 1) for k in range(n_poses - 1)]


def loop_closure_edges(
    n_poses: int,
    density: float,
    rng: np.random.Generator,
    min_separation: int = 3,
) -> list[tuple[int, int]]:
    """density * n_poses loop closures, taken from a random ordering."""
    candidates = [
        (i, j)
        for i in range(n_poses)
        for j in range(i + min_separation, n_poses)
    ]
    count = _exact_count(density * n_poses, "density * n_poses")
    if count > len(candidates):
        raise ValueError(
            f"{count} loop closures requested but only {len(candidates)} pose "
            f"pairs are at least {min_separation} apart"
        )
    order = rng.permutation(len(candidates))
    return [candidates[k] for k in sorted(order[:count])]


def make_scenario(
    lie,
    n_poses: int = 20,
    loop_density: float = 0.2,
    outlier_rate: float = 0.0,
    seed: int = 0,
    turn: float = 0.2,
) -> Scenario:
    """Build a scenario. For a fixed seed, closures and outliers are nested."""
    rng = np.random.default_rng(seed)
    truth = curved_trajectory(lie, n_poses, turn=turn)
    closures = loop_closure_edges(n_poses, loop_density, rng)
    edges = odometry_edges(n_poses) + closures

    false_endpoint = [
        int(rng.choice([b for b in range(n_poses) if b not in (i, j)]))
        for i, j in closures
    ]
    n_bad = _exact_count(outlier_rate * len(closures), "outlier_rate * closures")
    order = rng.permutation(len(closures))
    outliers = {closures[k]: false_endpoint[k] for k in order[:n_bad]}
    return Scenario(truth, edges, outliers)


def sample_graph(
    lie, scenario: Scenario, noise: NoiseModel, rng: np.random.Generator
) -> PoseGraph:
    """One noisy measurement set for the scenario."""
    graph = PoseGraph(lie)
    for T in scenario.truth:
        graph.add_pose(T)

    for i, j in scenario.edges:
        observed = scenario.outliers.get((i, j), j)
        relative = lie.compose(
            lie.inverse(scenario.truth[i]), scenario.truth[observed]
        )
        measurement = lie.compose(relative, lie.exp(noise.sample(rng)))
        graph.add_factor(i, j, measurement, noise.information)
    return graph


def dead_reckon(lie, graph: PoseGraph, n_poses: int) -> list[np.ndarray]:
    """Initial guess from chaining the odometry."""
    poses = [lie.exp(np.zeros(lie.DOF))]
    odometry = {(f.i, f.j): f.measurement for f in graph.factors}
    for k in range(n_poses - 1):
        poses.append(lie.compose(poses[-1], odometry[(k, k + 1)]))
    return poses


@dataclass
class MonteCarloResult:
    """Per-run errors, marginals and full-state NEES."""

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
        """DOF of the full-state NEES after gauge fixing."""
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
    """Solve the scenario under n_runs noise draws."""
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
