"""Tests for NEES and coverage."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import chi2, kstest

from posetrust.lie import se2, se3
from posetrust.stats import (
    CONSERVATIVE,
    CONSISTENT,
    OVERCONFIDENT,
    ConsistencyReport,
    benjamini_hochberg,
    classify,
    coverage_curve,
    mean_acceptance_interval,
    mean_pvalue,
    nees,
    nees_by_dof,
    nees_series,
    tangent_error,
)

DIM = 6
N_RUNS = 2000


@pytest.fixture(scope="module")
def gaussian_sample():
    """Errors drawn from N(0, Sigma) with Sigma known exactly."""
    rng = np.random.default_rng(0)
    A = rng.normal(size=(DIM, DIM))
    sigma = A @ A.T + DIM * np.eye(DIM)
    errors = (np.linalg.cholesky(sigma) @ rng.normal(size=(DIM, N_RUNS))).T
    return errors, sigma


def repeated(sigma, n=N_RUNS):
    return np.repeat(sigma[None], n, axis=0)


def test_nees_matches_explicit_quadratic_form():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(4, 4))
    sigma = A @ A.T + 4 * np.eye(4)
    e = rng.normal(size=4)
    assert nees(e, sigma) == pytest.approx(e @ np.linalg.inv(sigma) @ e)


def test_exact_covariance_is_consistent(gaussian_sample):
    errors, sigma = gaussian_sample
    values = nees_series(errors, repeated(sigma))
    assert classify(values, DIM) == CONSISTENT
    assert values.mean() == pytest.approx(DIM, rel=0.05)


def test_nees_follows_chi_squared(gaussian_sample):
    errors, sigma = gaussian_sample
    values = nees_series(errors, repeated(sigma))
    assert kstest(values, chi2(DIM).cdf).pvalue > 0.01


def test_inflated_covariance_reads_as_conservative(gaussian_sample):
    errors, sigma = gaussian_sample
    values = nees_series(errors, repeated(2.0 * sigma))
    assert classify(values, DIM) == CONSERVATIVE
    assert values.mean() == pytest.approx(DIM / 2.0, rel=0.05)


def test_deflated_covariance_reads_as_overconfident(gaussian_sample):
    errors, sigma = gaussian_sample
    values = nees_series(errors, repeated(sigma / 2.0))
    assert classify(values, DIM) == OVERCONFIDENT
    assert values.mean() == pytest.approx(2.0 * DIM, rel=0.05)


def test_acceptance_band_narrows_with_more_runs():
    widths = [
        np.diff(mean_acceptance_interval(DIM, n))[0] for n in (50, 500, 5000)
    ]
    assert widths[0] > widths[1] > widths[2]
    for n in (50, 500, 5000):
        lo, hi = mean_acceptance_interval(DIM, n)
        assert lo < DIM < hi


def test_coverage_curve_sits_on_the_diagonal(gaussian_sample):
    errors, sigma = gaussian_sample
    values = nees_series(errors, repeated(sigma))
    nominal, empirical = coverage_curve(values, DIM)
    # Monte Carlo noise on a proportion from N_RUNS draws is ~1/sqrt(N)
    assert np.max(np.abs(nominal - empirical)) < 3.0 / np.sqrt(N_RUNS)


def test_coverage_falls_below_nominal_when_overconfident(gaussian_sample):
    errors, sigma = gaussian_sample
    values = nees_series(errors, repeated(sigma / 2.0))
    nominal, empirical = coverage_curve(values, DIM)
    assert np.all(empirical < nominal)


def test_pvalue_is_large_when_calibrated_and_tiny_when_not(gaussian_sample):
    errors, sigma = gaussian_sample
    assert mean_pvalue(nees_series(errors, repeated(sigma)), DIM) > 0.05
    assert mean_pvalue(nees_series(errors, repeated(sigma / 2.0)), DIM) < 1e-6


def test_tangent_error_is_zero_for_identical_poses(lie):
    rng = np.random.default_rng(2)
    T = lie.exp(rng.uniform(-1, 1, lie.DOF))
    np.testing.assert_allclose(
        tangent_error(lie, T, T), np.zeros(lie.DOF), rtol=0, atol=1e-13
    )


def test_tangent_error_recovers_a_known_perturbation(lie):
    rng = np.random.default_rng(5)
    estimate = lie.exp(rng.uniform(-1, 1, lie.DOF))
    delta = rng.uniform(-0.1, 0.1, lie.DOF)
    truth = lie.compose(estimate, lie.exp(delta))
    np.testing.assert_allclose(tangent_error(lie, estimate, truth), delta, rtol=0, atol=1e-12)


@pytest.mark.parametrize("lie_module", [se2, se3], ids=["se2", "se3"])
def test_nees_by_dof_splits_into_chi_squared_parts(lie_module):
    dof = lie_module.DOF
    t = lie_module.TRANSLATION_DOF
    rng = np.random.default_rng(1)
    A = rng.normal(size=(dof, dof))
    sigma = A @ A.T + dof * np.eye(dof)
    errors = (np.linalg.cholesky(sigma) @ rng.normal(size=(dof, 4000))).T

    split = nees_by_dof(lie_module, errors, repeated(sigma, 4000))
    assert split["translation"].mean() == pytest.approx(t, rel=0.08)
    assert split["rotation"].mean() == pytest.approx(dof - t, rel=0.08)


def test_benjamini_hochberg_controls_false_discoveries():
    rng = np.random.default_rng(4)
    assert benjamini_hochberg(rng.uniform(size=200)).sum() <= 2

    mixed = np.concatenate([rng.uniform(size=90), np.full(10, 1e-9)])
    rejected = benjamini_hochberg(mixed)
    assert rejected[-10:].all()
    assert rejected.sum() <= 15


def test_consistency_report_exposes_the_summary(gaussian_sample):
    errors, sigma = gaussian_sample
    report = ConsistencyReport(nees_series(errors, repeated(sigma)), DIM)
    lo, hi = report.acceptance
    assert lo < report.mean < hi
    assert report.verdict == CONSISTENT
    assert report.pvalue > 0.05
    nominal, empirical = report.coverage()
    assert len(nominal) == len(empirical) == 4
