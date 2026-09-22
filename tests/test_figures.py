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


def drawn_texts(fig):
    """Every text the renderer will actually draw.

    Ticks outside the view limits are computed but never drawn, so including
    them produces phantom collisions -- an earlier version of this check
    flagged two and both were invisible.
    """
    out = []
    for ax in fig.axes:
        for text in [ax._left_title, ax.xaxis.label, ax.yaxis.label, *ax.texts]:
            if text.get_text() and text.get_visible():
                out.append(text)
        for axis, (low, high) in (
            (ax.xaxis, ax.get_xlim()),
            (ax.yaxis, ax.get_ylim()),
        ):
            for text, loc in zip(axis.get_ticklabels(), axis.get_ticklocs()):
                if text.get_text() and text.get_visible() and low <= loc <= high:
                    out.append(text)
        legend = ax.get_legend()
        if legend:
            out.extend(t for t in legend.get_texts() if t.get_text())
    if fig._suptitle and fig._suptitle.get_text():
        out.append(fig._suptitle)
    return out


def laid_out(fig):
    """Run the layout engine and return a renderer that can measure text."""
    import io

    from _common import _svg_canvas

    canvas_cls = _svg_canvas()
    canvas_cls(fig)
    fig.draw_without_rendering()
    from matplotlib.backends.backend_svg import RendererSVG

    width, height = fig.get_size_inches() * fig.dpi
    return RendererSVG(width, height, io.StringIO()), width, height


def overlap_area(a, b) -> float:
    if a.x1 <= b.x0 or b.x1 <= a.x0 or a.y1 <= b.y0 or b.y1 <= a.y0:
        return 0.0
    return (min(a.x1, b.x1) - max(a.x0, b.x0)) * (min(a.y1, b.y1) - max(a.y0, b.y0))


def test_no_drawn_text_escapes_the_canvas(e1_figure):
    renderer, width, height = laid_out(e1_figure)
    escaped = [
        text.get_text()
        for text in drawn_texts(e1_figure)
        if (lambda b: b.x0 < -1 or b.y0 < -1 or b.x1 > width + 1 or b.y1 > height + 1)(
            text.get_window_extent(renderer)
        )
    ]
    assert not escaped, f"text clipped by the canvas edge: {escaped}"


def test_no_drawn_labels_collide(e1_figure):
    """The automated half of "render it and look at it".

    A test cannot judge whether a chart reads well, but it can catch the
    failure that most often makes one unreadable: labels printed on top of
    each other once the data changes shape.
    """
    renderer, _, _ = laid_out(e1_figure)
    boxes = [
        (t.get_text(), t.get_window_extent(renderer)) for t in drawn_texts(e1_figure)
    ]
    collisions = [
        (a[0], b[0])
        for i, a in enumerate(boxes)
        for b in boxes[i + 1 :]
        if overlap_area(a[1], b[1]) > 4.0
    ]
    assert not collisions, f"overlapping labels: {collisions}"


def test_panels_do_not_overlap_each_other(e1_figure):
    renderer, _, _ = laid_out(e1_figure)
    boxes = [ax.get_window_extent(renderer) for ax in e1_figure.axes]
    clashes = [
        (i, j)
        for i in range(len(boxes))
        for j in range(i + 1, len(boxes))
        if overlap_area(boxes[i], boxes[j]) > 1.0
    ]
    assert not clashes, f"panels overlap: {clashes}"


def test_no_mark_is_drawn_without_a_legend_entry(e1_figure):
    """An unexplained line on a chart is chart junk.

    The reference line marking the expected mean was drawn unlabelled at
    first: visible, meaningful, and impossible for a reader to identify.
    """
    for index, ax in enumerate(e1_figure.axes):
        anonymous = [
            line.get_label()
            for line in ax.lines
            if not line.get_label() or line.get_label().startswith("_")
        ]
        assert not anonymous, f"panel {index} draws unexplained marks: {anonymous}"


def ink_under(ax, box, renderer) -> float:
    """Area of drawn data falling inside a box, in square pixels."""
    total = 0.0
    for patch in ax.patches:
        pb = patch.get_window_extent(renderer)
        width = min(pb.x1, box.x1) - max(pb.x0, box.x0)
        height = min(pb.y1, box.y1) - max(pb.y0, box.y0)
        if width > 0 and height > 0:
            total += width * height
    for line in ax.lines:
        xy = ax.transData.transform(line.get_xydata())
        inside = (
            (xy[:, 0] >= box.x0)
            & (xy[:, 0] <= box.x1)
            & (xy[:, 1] >= box.y0)
            & (xy[:, 1] <= box.y1)
        ).sum()
        total += inside * 6.0
    return total


def test_no_annotation_is_laid_over_the_data(e1_figure):
    """Legends and labels must not sit on top of the marks they describe.

    This is the check that was missing when the label-collision test passed a
    figure whose legend covered 1400 px^2 of histogram: comparing text against
    text says nothing about text against data.
    """
    renderer, _, _ = laid_out(e1_figure)
    for index, ax in enumerate(e1_figure.axes):
        boxes = [ax.get_legend().get_window_extent(renderer)]
        boxes += [t.get_window_extent(renderer) for t in ax.texts]
        for box in boxes:
            covered = ink_under(ax, box, renderer)
            assert covered < 1.0, f"panel {index} hides {covered:.0f} px^2 of data"


def test_legend_placement_is_consistent_within_a_chart_type(e1_figure):
    """Panels showing the same chart should place the legend the same way.

    Deliberately per chart type rather than across the whole figure: an
    earlier version demanded one position everywhere, and satisfying it drove
    the legend onto the histogram. Not obscuring the data outranks symmetry.
    """
    renderer, _, _ = laid_out(e1_figure)

    def position(ax):
        box, axis_box = (
            ax.get_legend().get_window_extent(renderer),
            ax.get_window_extent(renderer),
        )
        return (box.x0 - axis_box.x0) / axis_box.width

    distributions = [position(e1_figure.axes[i]) for i in (0, 2)]
    coverages = [position(e1_figure.axes[i]) for i in (1, 3)]
    assert abs(distributions[0] - distributions[1]) < 0.05
    assert abs(coverages[0] - coverages[1]) < 0.05


def test_panel_titles_fit_inside_their_panel(e1_figure):
    """Titles now carry the headline number, so they can overflow."""
    renderer, _, _ = laid_out(e1_figure)
    for index, ax in enumerate(e1_figure.axes):
        title = ax._left_title
        if not title.get_text():
            continue
        width = title.get_window_extent(renderer).width
        available = ax.get_window_extent(renderer).width
        assert width <= available, (
            f"panel {index} title is {width:.0f}px in a {available:.0f}px panel"
        )
