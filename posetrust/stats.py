"""NEES, chi-squared tests, coverage and multiple-testing correction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import chi2

if TYPE_CHECKING:
    from posetrust.simulate import MonteCarloResult

CONSISTENT = "consistent"
CONSERVATIVE = "conservative"
OVERCONFIDENT = "OVERCONFIDENT"


def tangent_error(lie, estimate: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """log(estimate^-1 truth)."""
    return lie.log(lie.compose(lie.inverse(estimate), truth))


def nees(error: np.ndarray, covariance: np.ndarray) -> float:
    """e^T Sigma^-1 e."""
    factor = cho_factor(covariance)
    return float(error @ cho_solve(factor, error))


def nees_series(errors: np.ndarray, covariances: np.ndarray) -> np.ndarray:
    """NEES for each run."""
    return np.array([nees(e, S) for e, S in zip(np.asarray(errors), np.asarray(covariances))])


def nees_by_dof(lie, errors: np.ndarray, covariances: np.ndarray) -> dict[str, np.ndarray]:
    """NEES split into translation and rotation parts."""
    errors = np.asarray(errors)
    covariances = np.asarray(covariances)
    t = lie.TRANSLATION_DOF
    return {
        "translation": nees_series(errors[:, :t], covariances[:, :t, :t]),
        "rotation": nees_series(errors[:, t:], covariances[:, t:, t:]),
    }


def benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg rejections at level alpha."""
    pvalues = np.asarray(pvalues, dtype=float)
    n = pvalues.size
    order = np.argsort(pvalues)
    thresholds = alpha * np.arange(1, n + 1) / n
    passed = pvalues[order] <= thresholds

    rejected = np.zeros(n, dtype=bool)
    if passed.any():
        cutoff = np.max(np.flatnonzero(passed))
        rejected[order[: cutoff + 1]] = True
    return rejected


@dataclass
class ConsistencyReport:
    """NEES samples with their verdict, band and p-value."""

    values: np.ndarray
    dof: int
    alpha: float = 0.05

    @property
    def mean(self) -> float:
        return float(self.values.mean())

    @property
    def acceptance(self) -> tuple[float, float]:
        """Interval the mean NEES should fall in if Sigma is honest."""
        n = self.values.size
        lo = chi2.ppf(self.alpha / 2.0, n * self.dof) / n
        hi = chi2.ppf(1.0 - self.alpha / 2.0, n * self.dof) / n
        return float(lo), float(hi)

    @property
    def verdict(self) -> str:
        lo, hi = self.acceptance
        if self.mean > hi:
            return OVERCONFIDENT
        if self.mean < lo:
            return CONSERVATIVE
        return CONSISTENT

    @property
    def pvalue(self) -> float:
        """Two-sided p-value for the mean NEES."""
        lower = chi2.cdf(self.values.sum(), self.values.size * self.dof)
        return float(2.0 * min(lower, 1.0 - lower))

    def coverage(self, levels: tuple[float, ...] = (0.5, 0.9, 0.95, 0.99)) -> tuple[np.ndarray, np.ndarray]:
        """Empirical against nominal ellipsoid coverage."""
        nominal = np.asarray(levels, dtype=float)
        empirical = np.array([float(np.mean(self.values <= chi2.ppf(p, self.dof))) for p in nominal])
        return nominal, empirical


@dataclass
class Report:
    """Consistency results for one condition."""

    result: MonteCarloResult
    alpha: float = 0.05

    @property
    def nees(self) -> ConsistencyReport:
        """Full-state NEES."""
        return ConsistencyReport(self.result.nees_full, self.result.free_dof, self.alpha)

    @property
    def verdict(self) -> str:
        """consistent, conservative or OVERCONFIDENT."""
        return self.nees.verdict

    def summary(self) -> str:
        """One-line summary."""
        nees = self.nees
        lo, hi = nees.acceptance
        return (
            f"{self.verdict:<14} mean NEES {nees.mean:9.3f} "
            f"(dof {nees.dof}, band [{lo:.2f}, {hi:.2f}], p={nees.pvalue:.3g}, "
            f"{self.result.n_runs} runs)"
        )


def consistency(result: MonteCarloResult, alpha: float = 0.05) -> Report:
    """Build a Report, refusing results with non-converged runs."""
    if not result.converged.all():
        failed = int((~result.converged).sum())
        raise ValueError(
            f"{failed} of {result.n_runs} runs did not converge; a consistency "
            "verdict over non-converged solutions would be meaningless. "
            "Inspect the condition, or re-run with the robust back-end."
        )
    return Report(result, alpha)
