"""Pose-graph structure: odometry and loop-closure factors with full
information matrices, over SE(2) or SE(3) poses.

The group is injected rather than hard-coded — pass the se2 or se3 module as
`lie` and everything below is group-agnostic. Q2 of the study compares the two
directly, so they must run through identical code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Factor:
    """A relative-pose constraint: from pose i, pose j is observed at `measurement`.

    `information` is Omega, the inverse covariance of the measurement expressed
    in the tangent space — a full matrix rather than a scalar weight, so
    anisotropic and correlated noise are both representable. Noise anisotropy
    is one of the axes the Monte Carlo harness sweeps.
    """

    i: int
    j: int
    measurement: np.ndarray
    information: np.ndarray


class PoseGraph:
    """Poses plus the constraints between them, and the linear system they induce."""

    def __init__(self, lie) -> None:
        self.lie = lie
        self.poses: list[np.ndarray] = []
        self.factors: list[Factor] = []

    @property
    def dof(self) -> int:
        """Tangent-space dimension per pose: 3 for SE(2), 6 for SE(3)."""
        return self.lie.DOF

    def add_pose(self, T: np.ndarray) -> int:
        self.poses.append(np.asarray(T, dtype=float))
        return len(self.poses) - 1

    def add_factor(
        self, i: int, j: int, measurement: np.ndarray, information: np.ndarray
    ) -> None:
        self.factors.append(
            Factor(
                i,
                j,
                np.asarray(measurement, dtype=float),
                np.asarray(information, dtype=float),
            )
        )

    def residual(self, factor: Factor, poses: list[np.ndarray]) -> np.ndarray:
        """r = log(Z^-1 @ Ti^-1 @ Tj): how far the estimate sits from the measurement.

        Expressed in the tangent space, so it is the manifold error rather than
        a naive difference of matrix entries — the same distinction the NEES
        computation depends on later.
        """
        lie = self.lie
        predicted = lie.compose(lie.inverse(poses[factor.i]), poses[factor.j])
        return lie.log(lie.compose(lie.inverse(factor.measurement), predicted))

    def factor_jacobians(
        self, factor: Factor, poses: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """d(residual)/d(delta_i), d(residual)/d(delta_j) for right perturbations.

        With Ti <- Ti @ exp(delta_i) and M = Ti^-1 @ Tj, pushing both
        perturbations to the right of the error pose gives
            E(delta) = E0 @ exp(-Adj(M^-1) delta_i) @ exp(delta_j),
        so to first order the combined perturbation is
        -Adj(M^-1) delta_i + delta_j, and log() contributes Jr^-1(r0).
        """
        lie = self.lie
        r0 = self.residual(factor, poses)
        jr_inv = np.linalg.inv(lie.right_jacobian(r0))
        m_inv = lie.compose(lie.inverse(poses[factor.j]), poses[factor.i])
        return -jr_inv @ lie.adjoint(m_inv), jr_inv

    def chi2(self, poses: list[np.ndarray]) -> float:
        """Sum of r^T Omega r — the objective Gauss-Newton is minimising."""
        total = 0.0
        for factor in self.factors:
            r = self.residual(factor, poses)
            total += float(r @ factor.information @ r)
        return total

    def linearize(
        self, poses: list[np.ndarray], weights: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Assemble H = sum w J^T Omega J and b = sum w J^T Omega r.

        `weights` is one scale per factor, used by the robust back-ends. It
        multiplies the information matrix, so a down-weighted constraint
        contributes less to the estimate *and* less to H -- which means the
        covariance a robust method reports is built from the reweighted
        information. Whether that covariance stays honest is precisely what
        the perceptual-aliasing experiment asks, so the weighting has to reach
        H rather than being applied only to the residuals.

        H is assembled dense. At the graph sizes this study uses each solve
        is milliseconds, so the sparsity has never been worth exploiting; if
        that changes, the interface does not.
        """
        n = len(self.poses) * self.dof
        H = np.zeros((n, n))
        b = np.zeros(n)
        d = self.dof

        for index, factor in enumerate(self.factors):
            r = self.residual(factor, poses)
            Ji, Jj = self.factor_jacobians(factor, poses)
            omega = factor.information
            if weights is not None:
                omega = weights[index] * omega
            si, sj = factor.i * d, factor.j * d

            H[si : si + d, si : si + d] += Ji.T @ omega @ Ji
            H[si : si + d, sj : sj + d] += Ji.T @ omega @ Jj
            H[sj : sj + d, si : si + d] += Jj.T @ omega @ Ji
            H[sj : sj + d, sj : sj + d] += Jj.T @ omega @ Jj
            b[si : si + d] += Ji.T @ omega @ r
            b[sj : sj + d] += Jj.T @ omega @ r

        return H, b

    def retract(self, poses: list[np.ndarray], delta: np.ndarray) -> list[np.ndarray]:
        """Apply a tangent-space step: T <- T @ exp(delta), pose by pose.

        The retraction is what keeps the estimate on the manifold instead of
        drifting off it the way a vector-space update would.
        """
        d = self.dof
        return [
            self.lie.compose(T, self.lie.exp(delta[k * d : (k + 1) * d]))
            for k, T in enumerate(poses)
        ]
