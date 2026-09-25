"""The consistency() report."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from posetrust.simulate import MonteCarloResult
from posetrust.stats import (
    ConsistencyReport,
    coverage_curve,
    nees_by_dof,
    nees_series,
)


@dataclass
class Report:
    """Consistency results for one condition."""

    result: MonteCarloResult
    lie: object
    alpha: float = 0.05

    @property
    def nees(self) -> ConsistencyReport:
        """Full-state NEES."""
        return ConsistencyReport(
            self.result.nees_full, self.result.free_dof, self.alpha
        )

    @property
    def verdict(self) -> str:
        """consistent, conservative or OVERCONFIDENT."""
        return self.nees.verdict

    def coverage_curve(
        self, levels: tuple[float, ...] = (0.5, 0.9, 0.95, 0.99)
    ) -> tuple[np.ndarray, np.ndarray]:
        """Empirical against nominal ellipsoid coverage."""
        return coverage_curve(self.result.nees_full, self.result.free_dof, levels)

    def pose(self, k: int) -> ConsistencyReport:
        """NEES for pose k."""
        values = nees_series(self.result.pose_errors(k), self.result.pose_marginals(k))
        return ConsistencyReport(values, self.result.dof, self.alpha)

    def by_dof(self, pose: int | None = None) -> dict[str, ConsistencyReport]:
        """NEES split into translation and rotation."""
        if pose is not None:
            errors = self.result.pose_errors(pose)
            marginals = self.result.pose_marginals(pose)
        else:
            free = [k for k in range(self.result.errors.shape[1]) if k != self.result.anchor]
            errors = self.result.errors[:, free, :].reshape(-1, self.result.dof)
            marginals = self.result.marginals[:, free].reshape(
                -1, self.result.dof, self.result.dof
            )

        split = nees_by_dof(self.lie, errors, marginals)
        translation = self.lie.TRANSLATION_DOF
        return {
            "translation": ConsistencyReport(
                split["translation"], translation, self.alpha
            ),
            "rotation": ConsistencyReport(
                split["rotation"], self.result.dof - translation, self.alpha
            ),
        }

    def summary(self) -> str:
        """One-line summary."""
        nees = self.nees
        lo, hi = nees.acceptance
        return (
            f"{self.verdict:<14} mean NEES {nees.mean:9.3f} "
            f"(dof {nees.dof}, band [{lo:.2f}, {hi:.2f}], p={nees.pvalue:.3g}, "
            f"{self.result.n_runs} runs)"
        )


def consistency(
    result: MonteCarloResult, lie, alpha: float = 0.05
) -> Report:
    """Build a Report, refusing results with non-converged runs."""
    if not result.converged.all():
        failed = int((~result.converged).sum())
        raise ValueError(
            f"{failed} of {result.n_runs} runs did not converge; a consistency "
            "verdict over non-converged solutions would be meaningless. "
            "Inspect the condition, or re-run with the robust back-end."
        )
    return Report(result, lie, alpha)
