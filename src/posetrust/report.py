"""Public API: posetrust.consistency(result) returning a report with
.nees, .coverage_curve(), .by_dof(), and .verdict.

One call that turns a Monte Carlo run into the answer the project exists to
give: is the covariance this solver reported honest, and if not, which way is
it wrong. Everything here composes stats.py rather than recomputing anything,
so there is a single implementation of the chi-squared machinery.

A note on what cannot be asked here. NEES needs the true poses, and the point
of the Monte Carlo harness is that simulation is the only place they exist. On
real data there is one dataset, one answer, and no way to ask what else might
have happened -- which is why E5 tests consistency a different way, by holding
constraints out and checking whether their residuals match the uncertainty the
graph predicted for them.
"""

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
    """The consistency verdict for one experimental condition."""

    result: MonteCarloResult
    lie: object
    alpha: float = 0.05

    @property
    def nees(self) -> ConsistencyReport:
        """Full-state NEES: the distribution, its band, and the chi-squared test.

        This is the formal test. Every run contributes one independent sample,
        so the acceptance band is exactly valid here in a way it is not for
        the per-pose breakdowns below.
        """
        return ConsistencyReport(
            self.result.nees_full, self.result.free_dof, self.alpha
        )

    @property
    def verdict(self) -> str:
        """consistent | conservative | OVERCONFIDENT."""
        return self.nees.verdict

    def coverage_curve(
        self, levels: tuple[float, ...] = (0.5, 0.9, 0.95, 0.99)
    ) -> tuple[np.ndarray, np.ndarray]:
        """Empirical against nominal coverage of the credible ellipsoids."""
        return coverage_curve(self.result.nees_full, self.result.free_dof, levels)

    def pose(self, k: int) -> ConsistencyReport:
        """NEES for a single pose. Valid as a test: one sample per run."""
        values = nees_series(self.result.pose_errors(k), self.result.pose_marginals(k))
        return ConsistencyReport(values, self.result.dof, self.alpha)

    def by_dof(self, pose: int | None = None) -> dict[str, ConsistencyReport]:
        """Translation against rotation -- Q2's hypothesis lives here.

        With `pose` given, the samples are one per run and the band applies
        directly. Pooled across poses (the default) the samples share a graph
        and are correlated, so the means stay informative but the band is
        optimistic; read the pooled form as a description of shape and the
        per-pose or full-state form as the test.
        """
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
        """One line per condition, for a sweep's log."""
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
    """Is the covariance this solver reported consistent with its actual error?

    Raises if any run failed to converge: a NEES computed over a mixture of
    converged and diverged solutions is not a statement about calibration, and
    silently averaging the two is how a broken condition gets reported as a
    merely overconfident one.
    """
    if not result.converged.all():
        failed = int((~result.converged).sum())
        raise ValueError(
            f"{failed} of {result.n_runs} runs did not converge; a consistency "
            "verdict over non-converged solutions would be meaningless. "
            "Inspect the condition, or re-run with the robust back-end."
        )
    return Report(result, lie, alpha)
