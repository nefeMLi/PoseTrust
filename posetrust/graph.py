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
    measurement_inverse: np.ndarray


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
        measurement = np.asarray(measurement, dtype=float)
        self.factors.append(
            Factor(
                i,
                j,
                measurement,
                np.asarray(information, dtype=float),
                self.lie.inverse(measurement),
            )
        )

    def residual(self, factor: Factor, poses: list[np.ndarray]) -> np.ndarray:
        """r = log(Z^-1 Ti^-1 Tj)."""
        return self._residual(factor, self.lie.inverse(poses[factor.i]), poses[factor.j])

    def _residual(self, factor: Factor, Ti_inv: np.ndarray, Tj: np.ndarray) -> np.ndarray:
        lie = self.lie
        return lie.log(lie.compose(factor.measurement_inverse, lie.compose(Ti_inv, Tj)))

    def residuals(self, poses: list[np.ndarray]) -> list[np.ndarray]:
        """Residual of every factor, inverting each pose once."""
        inverses = [self.lie.inverse(T) for T in poses]
        return [self._residual(f, inverses[f.i], poses[f.j]) for f in self.factors]

    def factor_jacobians(
        self, factor: Factor, poses: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Residual Jacobians for right perturbations of Ti and Tj."""
        r0 = self.residual(factor, poses)
        return self._jacobians(r0, self.lie.inverse(poses[factor.j]), poses[factor.i])

    def _jacobians(
        self, r0: np.ndarray, Tj_inv: np.ndarray, Ti: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        lie = self.lie
        jr_inv = np.linalg.inv(lie.right_jacobian(r0))
        m_inv = lie.compose(Tj_inv, Ti)
        return -jr_inv @ lie.adjoint(m_inv), jr_inv

    def chi2(self, poses: list[np.ndarray]) -> float:
        """Sum of r^T Omega r over all factors."""
        total = 0.0
        for factor, r in zip(self.factors, self.residuals(poses)):
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

        # Each pose is inverted once, not once per factor it touches.
        inverses = [self.lie.inverse(T) for T in poses]
        for index, factor in enumerate(self.factors):
            r = self._residual(factor, inverses[factor.i], poses[factor.j])
            Ji, Jj = self._jacobians(r, inverses[factor.j], poses[factor.i])
            omega = factor.information
            if weights is not None:
                omega = weights[index] * omega
            si, sj = factor.i * d, factor.j * d

            Ji_omega, Jj_omega = Ji.T @ omega, Jj.T @ omega
            H[si : si + d, si : si + d] += Ji_omega @ Ji
            H[si : si + d, sj : sj + d] += Ji_omega @ Jj
            H[sj : sj + d, si : si + d] += Jj_omega @ Ji
            H[sj : sj + d, sj : sj + d] += Jj_omega @ Jj
            b[si : si + d] += Ji_omega @ r
            b[sj : sj + d] += Jj_omega @ r

        return H, b

    def retract(self, poses: list[np.ndarray], delta: np.ndarray) -> list[np.ndarray]:
        """Apply T <- T exp(delta) pose by pose."""
        d = self.dof
        return [
            self.lie.compose(T, self.lie.exp(delta[k * d : (k + 1) * d]))
            for k, T in enumerate(poses)
        ]
