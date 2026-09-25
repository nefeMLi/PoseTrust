"""NEES, chi-squared tests, coverage and multiple-testing correction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import chi2

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
    return np.array(
        [nees(e, S) for e, S in zip(np.asarray(errors), np.asarray(covariances))]
    )


def mean_acceptance_interval(
    dof: int, n_runs: int, alpha: float = 0.05
) -> tuple[float, float]:
    """Interval the mean NEES should fall in if Sigma is honest."""
    lo = chi2.ppf(alpha / 2.0, n_runs * dof) / n_runs
    hi = chi2.ppf(1.0 - alpha / 2.0, n_runs * dof) / n_runs
    return float(lo), float(hi)


def classify(values: np.ndarray, dof: int, alpha: float = 0.05) -> str:
    """consistent, conservative or OVERCONFIDENT."""
    values = np.asarray(values, dtype=float)
    lo, hi = mean_acceptance_interval(dof, values.size, alpha)
    mean = float(values.mean())
    if mean > hi:
        return OVERCONFIDENT
    if mean < lo:
        return CONSERVATIVE
    return CONSISTENT


def mean_pvalue(values: np.ndarray, dof: int) -> float:
    """Two-sided p-value for the mean NEES."""
    values = np.asarray(values, dtype=float)
    total = values.sum()
    df = values.size * dof
    lower = chi2.cdf(total, df)
    return float(2.0 * min(lower, 1.0 - lower))


def coverage_curve(
    values: np.ndarray,
    dof: int,
    levels: tuple[float, ...] = (0.5, 0.9, 0.95, 0.99),
) -> tuple[np.ndarray, np.ndarray]:
    """Empirical against nominal ellipsoid coverage."""
    values = np.asarray(values, dtype=float)
    nominal = np.asarray(levels, dtype=float)
    empirical = np.array(
        [float(np.mean(values <= chi2.ppf(p, dof))) for p in nominal]
    )
    return nominal, empirical


def nees_by_dof(
    lie, errors: np.ndarray, covariances: np.ndarray
) -> dict[str, np.ndarray]:
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
        return mean_acceptance_interval(self.dof, self.values.size, self.alpha)

    @property
    def verdict(self) -> str:
        return classify(self.values, self.dof, self.alpha)

    @property
    def pvalue(self) -> float:
        return mean_pvalue(self.values, self.dof)

    def coverage(
        self, levels: tuple[float, ...] = (0.5, 0.9, 0.95, 0.99)
    ) -> tuple[np.ndarray, np.ndarray]:
        return coverage_curve(self.values, self.dof, levels)
