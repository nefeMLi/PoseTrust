"""Pose graph over SE(2) or SE(3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Factor:
    """Relative-pose measurement from pose i to pose j."""

    i: int
    j: int
    measurement: np.ndarray
    information: np.ndarray


class PoseGraph:
    """Poses, factors, and the linear system they give."""

    def __init__(self, lie) -> None:
        self.lie = lie
        self.poses: list[np.ndarray] = []
        self.factors: list[Factor] = []

    @property
    def dof(self) -> int:
        """Tangent dimension per pose."""
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
        """r = log(Z^-1 Ti^-1 Tj)."""
        lie = self.lie
        predicted = lie.compose(lie.inverse(poses[factor.i]), poses[factor.j])
        return lie.log(lie.compose(lie.inverse(factor.measurement), predicted))

    def factor_jacobians(
        self, factor: Factor, poses: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Residual Jacobians for right perturbations of Ti and Tj."""
        lie = self.lie
        r0 = self.residual(factor, poses)
        jr_inv = np.linalg.inv(lie.right_jacobian(r0))
        m_inv = lie.compose(lie.inverse(poses[factor.j]), poses[factor.i])
        return -jr_inv @ lie.adjoint(m_inv), jr_inv

    def chi2(self, poses: list[np.ndarray]) -> float:
        """Sum of r^T Omega r over all factors."""
        total = 0.0
        for factor in self.factors:
            r = self.residual(factor, poses)
            total += float(r @ factor.information @ r)
        return total

    def linearize(
        self, poses: list[np.ndarray], weights: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build H and b, optionally with per-factor weights."""
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
        """Apply T <- T exp(delta) pose by pose."""
        d = self.dof
        return [
            self.lie.compose(T, self.lie.exp(delta[k * d : (k + 1) * d]))
            for k, T in enumerate(poses)
        ]
