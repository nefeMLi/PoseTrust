"""Tests for the consistency() report."""

from __future__ import annotations

import dataclasses
import functools

import numpy as np
import pytest

import posetrust
from posetrust.report import Report, consistency
from posetrust.simulate import NoiseModel, make_scenario, monte_carlo
from posetrust.stats import CONSISTENT, ConsistencyReport

N_RUNS = 150


@functools.cache
def _calibrated_run(lie):
    """Cached near-linear run shared by these tests."""
    scenario = make_scenario(lie, n_poses=6, loop_density=0.5, seed=1, turn=0.05)
    return monte_carlo(
        lie, scenario, NoiseModel(np.full(lie.DOF, 1e-3)), n_runs=N_RUNS, seed=7
    )


def calibrated_run(lie):
    """Fresh copy of the cached run."""
    cached = _calibrated_run(lie)
    return dataclasses.replace(cached, converged=cached.converged.copy())


def test_consistency_is_exposed_at_package_level():
    assert posetrust.consistency is consistency


def test_report_verdict_on_a_calibrated_condition(lie):
    report = consistency(calibrated_run(lie), lie, alpha=0.01)
    assert report.verdict == CONSISTENT
    assert isinstance(report.nees, ConsistencyReport)
    assert report.nees.mean / report.nees.dof == pytest.approx(1.0, abs=0.06)


def test_coverage_curve_tracks_the_diagonal_when_calibrated(lie):
    report = consistency(calibrated_run(lie), lie)
    nominal, empirical = report.coverage_curve()
    assert nominal.shape == empirical.shape == (4,)
    assert np.max(np.abs(nominal - empirical)) < 4.0 / np.sqrt(N_RUNS)


def test_by_dof_splits_translation_from_rotation(lie):
    report = consistency(calibrated_run(lie), lie)
    split = report.by_dof()
    assert set(split) == {"translation", "rotation"}
    assert split["translation"].dof == lie.TRANSLATION_DOF
    assert split["rotation"].dof == lie.DOF - lie.TRANSLATION_DOF
    for part in split.values():
        assert part.mean / part.dof == pytest.approx(1.0, abs=0.25)


def test_by_dof_for_a_single_pose_is_a_valid_sample(lie):
    result = calibrated_run(lie)
    split = consistency(result, lie).by_dof(pose=4)
    assert split["translation"].values.size == result.n_runs


def test_per_pose_report(lie):
    result = calibrated_run(lie)
    report = consistency(result, lie)
    assert report.pose(3).dof == lie.DOF
    assert report.pose(3).values.size == result.n_runs


def test_summary_is_one_readable_line(lie):
    line = consistency(calibrated_run(lie), lie).summary()
    assert "mean NEES" in line and "\n" not in line


def test_non_converged_runs_are_refused(lie):
    result = calibrated_run(lie)
    result.converged[3] = False
    with pytest.raises(ValueError, match="did not converge"):
        consistency(result, lie)


def test_report_can_be_constructed_directly_for_partial_data(lie):
    result = calibrated_run(lie)
    result.converged[0] = False
    report = Report(result, lie)
    assert report.verdict in {"consistent", "conservative", "OVERCONFIDENT"}
