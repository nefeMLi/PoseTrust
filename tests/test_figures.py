"""Structural tests for the experiment figures.

Figure code is the easiest thing in a research repo to leave unexercised: it
runs once by hand, looks plausible, and is never touched again until the data
shape changes underneath it. These build the real figure from the committed
results and assert on what it contains.

They deliberately stop short of rendering. Asserting that a chart *reads*
well is not something a test can do -- that needs a person looking at the
image. What a test can do is catch the failures that are not about taste: the
wrong series count, a reference line that never got drawn, an identity
encoded in colour alone, or the dual-axis anti-pattern.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

EXPERIMENTS = Path(__file__).resolve().parent.parent / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

pytest.importorskip("matplotlib.figure")

from _common import BAND, INK_MUTED, OBSERVED, SURFACE  # noqa: E402
from e1_validation_gate import build_figure  # noqa: E402


@pytest.fixture(scope="module")
def e1_figure():
    """The real E1 figure, built from the committed parquet results."""
    return build_figure()


def test_figure_has_one_panel_per_group_and_view(e1_figure):
    """Two groups by two views, as small multiples rather than overlaid series."""
    assert len(e1_figure.axes) == 4


def test_every_panel_is_titled(e1_figure):
    # titles are set left-aligned, so get_title() must be asked for that
    # location -- its default reads the (empty) centre title
    titles = [ax.get_title(loc="left") for ax in e1_figure.axes]
    assert all(titles), "a panel without a title cannot be read on its own"
    assert any("SE(2)" in t for t in titles)
    assert any("SE(3)" in t for t in titles)


def test_no_panel_uses_a_second_y_axis(e1_figure):
    """The dual-axis anti-pattern, asserted against rather than trusted.

    Two y-scales on one frame let a reader infer any relationship the author
    wants. Two measures of different scale belong in two panels.
    """
    for ax in e1_figure.axes:
        siblings = ax.get_shared_x_axes().get_siblings(ax)
        overlapping = [
            other
            for other in siblings
            if other is not ax and other.bbox.bounds == ax.bbox.bounds
        ]
        assert not overlapping, "a twinned axis is a dual-axis chart"


def test_distribution_panels_compare_observed_against_theory(e1_figure):
    """A histogram of what happened, and the density it should have followed."""
    for ax in (e1_figure.axes[0], e1_figure.axes[2]):
        assert len(ax.patches) > 1, "no histogram drawn"
        assert len(ax.lines) >= 1, "no reference density drawn"

        reference = ax.lines[0]
        assert reference.get_linestyle() == "--"
        assert reference.get_color() == INK_MUTED, "theory must wear neutral ink"
        assert len(reference.get_xdata()) > 100, "density drawn too coarsely"


def test_coverage_panels_plot_observed_against_the_identity_line(e1_figure):
    for ax in (e1_figure.axes[1], e1_figure.axes[3]):
        assert len(ax.lines) == 2, "expected the ideal line and the observed curve"
        ideal, observed = ax.lines
        assert ideal.get_linestyle() == "--"
        assert ideal.get_color() == INK_MUTED
        assert observed.get_color() == OBSERVED
        assert len(observed.get_xdata()) == 4, "four nominal levels"


def test_identity_is_never_encoded_by_colour_alone(e1_figure):
    """Every panel carries a legend, so a colourblind reader is not stranded."""
    for ax in e1_figure.axes:
        legend = ax.get_legend()
        assert legend is not None, "a two-series panel needs a legend"
        assert len(legend.get_texts()) >= 2


def test_observed_series_uses_the_documented_palette_hue(e1_figure):
    """One categorical hue, taken unchanged from the validated palette."""
    for ax in (e1_figure.axes[0], e1_figure.axes[2]):
        assert any(patch.get_facecolor() for patch in ax.patches)
    assert OBSERVED == "#2a78d6"
    assert {SURFACE, INK_MUTED, BAND} == {"#fcfcfb", "#52514e", "#d9d8d4"}


def test_axes_are_labelled(e1_figure):
    for ax in e1_figure.axes:
        assert ax.get_xlabel(), "unlabelled x axis"
        assert ax.get_ylabel(), "unlabelled y axis"


def test_grid_is_recessive_and_behind_the_data(e1_figure):
    for ax in e1_figure.axes:
        assert ax.get_axisbelow() is True
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()
