"""The statistical core: normalized estimation error squared (using the
manifold error, with its chi-squared consistency test and acceptance band),
and empirical coverage of nominal credible-ellipsoid levels (50/90/95/99%).

NEES asks the one question the whole project is built around. A solver reports
a covariance Sigma and an estimate; the estimation error e is whatever it
actually turned out to be. If Sigma honestly describes the distribution of e,
then e^T Sigma^-1 e is chi-squared with as many degrees of freedom as the
state has, and its mean is the state dimension. Anything else is the solver
being wrong about its own uncertainty:

    mean NEES above the band   ->  Sigma too small  ->  OVERCONFIDENT
    mean NEES below the band   ->  Sigma too large  ->  conservative

Only one of those is dangerous. A planner that believes a too-tight ellipse
drives into things; one that believes a too-loose ellipse is merely slow.

Two details that are easy to get wrong and would invalidate everything:

  manifold error   e must be log(estimate^-1 @ truth), not a difference of
                   matrix entries or of raw parameters. Sigma lives in the
                   tangent space, so the error has to as well.

  the gauge        The anchored pose has exactly zero covariance, so it has
                   to be excluded. Including it puts a zero block into a
                   matrix that is about to be inverted, and the degrees of
                   freedom would be wrong even if it did not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import chi2

CONSISTENT = "consistent"
CONSERVATIVE = "conservative"
OVERCONFIDENT = "OVERCONFIDENT"


def tangent_error(lie, estimate: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """log(estimate^-1 @ truth): the error in the space the covariance describes.

    The sign convention is immaterial to NEES, which is quadratic, but it is
    fixed here so the per-degree-of-freedom breakdown means something.
    """
    return lie.log(lie.compose(lie.inverse(estimate), truth))


def nees(error: np.ndarray, covariance: np.ndarray) -> float:
    """e^T Sigma^-1 e, via a Cholesky solve rather than an explicit inverse."""
    factor = cho_factor(covariance)
    return float(error @ cho_solve(factor, error))


def nees_series(errors: np.ndarray, covariances: np.ndarray) -> np.ndarray:
    """NEES for each Monte Carlo run. errors (n_runs, d), covariances (n_runs, d, d)."""
    return np.array(
        [nees(e, S) for e, S in zip(np.asarray(errors), np.asarray(covariances))]
    )


def mean_acceptance_interval(
    dof: int, n_runs: int, alpha: float = 0.05
) -> tuple[float, float]:
    """Two-sided interval the MEAN NEES must fall inside if Sigma is honest.

    Each run contributes an independent chi-squared(dof), so their sum is
    chi-squared(n_runs * dof) and the interval for the mean is that scaled by
    1/n_runs. The band narrows as runs accumulate, which is what gives the
    test its power -- with too few runs almost nothing is detectable.
    """
    lo = chi2.ppf(alpha / 2.0, n_runs * dof) / n_runs
    hi = chi2.ppf(1.0 - alpha / 2.0, n_runs * dof) / n_runs
    return float(lo), float(hi)


def classify(values: np.ndarray, dof: int, alpha: float = 0.05) -> str:
    """Verdict for a set of NEES samples: consistent, conservative, or overconfident."""
    values = np.asarray(values, dtype=float)
    lo, hi = mean_acceptance_interval(dof, values.size, alpha)
    mean = float(values.mean())
    if mean > hi:
        return OVERCONFIDENT
    if mean < lo:
        return CONSERVATIVE
    return CONSISTENT


def mean_pvalue(values: np.ndarray, dof: int) -> float:
    """Two-sided p-value for the observed mean NEES under a correct covariance.

    Reported alongside the verdict so a near-miss is not read as a clean pass:
    the band is a decision rule, the p-value is the evidence behind it.
    """
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
    """Empirical vs nominal coverage of the credible ellipsoids.

    For each nominal level p, the fraction of runs whose error fell inside the
    p-credible ellipsoid. Under an honest covariance the empirical value
    tracks the nominal one and the curve sits on the diagonal. More legible to
    a general reader than a chi-squared statistic, and it exposes the *shape*
    of a miscalibration rather than collapsing it to one number.
    """
    values = np.asarray(values, dtype=float)
    nominal = np.asarray(levels, dtype=float)
    empirical = np.array(
        [float(np.mean(values <= chi2.ppf(p, dof))) for p in nominal]
    )
    return nominal, empirical


def nees_by_dof(
    lie, errors: np.ndarray, covariances: np.ndarray
) -> dict[str, np.ndarray]:
    """Split NEES into translation and rotation parts.

    A sub-block of a Gaussian is itself Gaussian with the corresponding
    sub-block of the covariance, so each part is chi-squared on its own
    degrees of freedom. Q2's hypothesis is that rotation drives the failure,
    and pooling the two would hide exactly that.

    Expects per-pose errors and marginals, since translation and rotation are
    interleaved once poses are stacked.
    """
    errors = np.asarray(errors)
    covariances = np.asarray(covariances)
    t = lie.TRANSLATION_DOF
    return {
        "translation": nees_series(errors[:, :t], covariances[:, :t, :t]),
        "rotation": nees_series(errors[:, t:], covariances[:, t:, t:]),
    }


def benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Which hypotheses survive false-discovery-rate control.

    The experiments sweep dozens of conditions and test each one, so at
    alpha = 0.05 roughly one in twenty clean conditions would be flagged
    overconfident by chance alone. Controlling the false discovery rate is
    what keeps a sweep from manufacturing its own headline.
    """
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
    """What a sweep condition produced, ready to be plotted or tabulated."""

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
