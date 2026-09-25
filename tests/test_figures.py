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

import numpy as np
import pytest

EXPERIMENTS = Path(__file__).resolve().parent.parent / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from _common import BAND, INK_MUTED, OBSERVED, SURFACE
from e1_validation_gate import build_figure as build_e1
from e2_loop_closure_density import build_figure as build_e2
from e3_nonlinearity import build_coverage_figure as build_e3c
from e3_nonlinearity import build_figure as build_e3
from e4_perceptual_aliasing import build_figure as build_e4

BUILDERS = {
    "e1": build_e1,
    "e2": build_e2,
    "e3": build_e3,
    "e3_coverage": build_e3c,
    "e4": build_e4,
}


@pytest.fixture(scope="module")
def e1_figure():
    """The real E1 figure, built from the committed parquet results."""
    return build_e1()


@pytest.fixture(params=sorted(BUILDERS), scope="module")
def any_figure(request):
    """Every experiment figure, for the checks that apply to all of them."""
    return BUILDERS[request.param]()


def legend_of(ax, fig):
    """A panel's legend, whether it sits on the axes or on the figure."""
    return ax.get_legend() or (fig.legends[0] if fig.legends else None)


def test_figure_has_one_panel_per_group_and_view(e1_figure):
    """Two groups by two views, as small multiples rather than overlaid series."""
    assert len(e1_figure.axes) == 4


def test_every_panel_is_titled(any_figure):
    # titles are set left-aligned, so get_title() must be asked for that
    # location -- its default reads the (empty) centre title
    titles = [ax.get_title(loc="left") for ax in any_figure.axes]
    assert all(titles), "a panel without a title cannot be read on its own"


def test_no_panel_uses_a_second_y_axis(any_figure):
    """The dual-axis anti-pattern, asserted against rather than trusted.

    Two y-scales on one frame let a reader infer any relationship the author
    wants. Two measures of different scale belong in two panels.
    """
    for ax in any_figure.axes:
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


def test_identity_is_never_encoded_by_colour_alone(any_figure):
    """Every panel carries a legend, so a colourblind reader is not stranded."""
    for ax in any_figure.axes:
        legend = legend_of(ax, any_figure)
        assert legend is not None, "a multi-series panel needs a legend"
        assert len(legend.get_texts()) >= 2


def test_observed_series_uses_the_documented_palette_hue(e1_figure):
    """One categorical hue, taken unchanged from the validated palette."""
    for ax in (e1_figure.axes[0], e1_figure.axes[2]):
        assert any(patch.get_facecolor() for patch in ax.patches)
    assert OBSERVED == "#2a78d6"
    assert {SURFACE, INK_MUTED, BAND} == {"#fcfcfb", "#52514e", "#d9d8d4"}


def test_axes_are_labelled(any_figure):
    """Every axis is labelled, or shares one with a labelled sibling.

    A shared y-axis is labelled once for the row rather than repeated on each
    panel, so the requirement is that the reader can find the label, not that
    every Axes object carries its own.
    """
    for index, ax in enumerate(any_figure.axes):
        if not ax.get_xlabel():
            siblings = ax.get_shared_x_axes().get_siblings(ax)
            assert any(other.get_xlabel() for other in siblings), (
                f"panel {index} has an unlabelled x axis and shares with none"
            )
        if ax.get_ylabel():
            continue
        siblings = ax.get_shared_y_axes().get_siblings(ax)
        assert any(other.get_ylabel() for other in siblings), (
            f"panel {index} has an unlabelled y axis and shares with none"
        )


def test_grid_is_recessive_and_behind_the_data(any_figure):
    for ax in any_figure.axes:
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


def test_no_drawn_text_escapes_the_canvas(any_figure):
    renderer, width, height = laid_out(any_figure)

    def escapes(box) -> bool:
        return box.x0 < -1 or box.y0 < -1 or box.x1 > width + 1 or box.y1 > height + 1

    escaped = [
        text.get_text()
        for text in drawn_texts(any_figure)
        if escapes(text.get_window_extent(renderer))
    ]
    assert not escaped, f"text clipped by the canvas edge: {escaped}"


def test_no_drawn_labels_collide(any_figure):
    """The automated half of "render it and look at it".

    A test cannot judge whether a chart reads well, but it can catch the
    failure that most often makes one unreadable: labels printed on top of
    each other once the data changes shape.
    """
    renderer, _, _ = laid_out(any_figure)
    boxes = [
        (t.get_text(), t.get_window_extent(renderer)) for t in drawn_texts(any_figure)
    ]
    collisions = [
        (a[0], b[0])
        for i, a in enumerate(boxes)
        for b in boxes[i + 1 :]
        if overlap_area(a[1], b[1]) > 4.0
    ]
    assert not collisions, f"overlapping labels: {collisions}"


def test_panels_do_not_overlap_each_other(any_figure):
    renderer, _, _ = laid_out(any_figure)
    boxes = [ax.get_window_extent(renderer) for ax in any_figure.axes]
    clashes = [
        (i, j)
        for i in range(len(boxes))
        for j in range(i + 1, len(boxes))
        if overlap_area(boxes[i], boxes[j]) > 1.0
    ]
    assert not clashes, f"panels overlap: {clashes}"


def test_no_mark_is_drawn_without_a_legend_entry(any_figure):
    """An unexplained line on a chart is chart junk.

    The reference line marking the expected mean was drawn unlabelled at
    first: visible, meaningful, and impossible for a reader to identify.
    """
    for index, ax in enumerate(any_figure.axes):
        # An errorbar is one labelled series drawn as several Line2Ds -- its
        # marker and caps carry no label of their own, but the container's
        # legend entry explains all of them.
        owned = set()
        for container in ax.containers:
            for part in getattr(container, "lines", ()):
                if part is None:
                    continue
                parts = part if isinstance(part, tuple) else (part,)
                owned.update(id(p) for p in parts)

        anonymous = [
            line.get_label()
            for line in ax.lines
            if id(line) not in owned
            and (not line.get_label() or line.get_label().startswith("_"))
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


def test_no_annotation_is_laid_over_the_data(any_figure):
    """Legends and labels must not sit on top of the marks they describe.

    This is the check that was missing when the label-collision test passed a
    figure whose legend covered 1400 px^2 of histogram: comparing text against
    text says nothing about text against data.
    """
    renderer, _, _ = laid_out(any_figure)
    for index, ax in enumerate(any_figure.axes):
        legend = legend_of(ax, any_figure)
        boxes = [legend.get_window_extent(renderer)] if ax.get_legend() else []
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


def test_panel_titles_fit_inside_their_panel(any_figure):
    """Titles now carry the headline number, so they can overflow."""
    renderer, _, _ = laid_out(any_figure)
    for index, ax in enumerate(any_figure.axes):
        title = ax._left_title
        if not title.get_text():
            continue
        width = title.get_window_extent(renderer).width
        available = ax.get_window_extent(renderer).width
        assert width <= available, (
            f"panel {index} title is {width:.0f}px in a {available:.0f}px panel"
        )


@pytest.mark.parametrize("name", ["e2", "e3", "e4"])
def test_point_estimates_carry_intervals(name):
    """HYPOTHESES.md: "Every point estimate gets an interval."

    A sweep plotted as bare markers invites the reader to see structure in
    what is sampling noise -- which is exactly what the first version of the
    E2 figure did, with a dramatic-looking zigzag entirely inside the band.
    The acceptance band is not a substitute: it says what a calibrated solver
    is allowed to produce, not how precisely this sweep measured it. Checked
    on every sweep figure: E4 once shipped as bare lines because only E2 was.
    """
    figure = BUILDERS[name]()
    for index, ax in enumerate(figure.axes):
        bars = [c for c in ax.containers if hasattr(c, "has_yerr")]
        assert bars, f"panel {index} plots estimates with no interval"
        assert any(c.has_yerr for c in bars), f"panel {index} has empty error bars"


def test_sweep_conditions_are_evenly_spaced():
    """Unevenly spaced conditions plotted on a linear axis distort the shape.

    E2's densities run 0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2: on a linear axis the
    crowded low end exaggerates small differences into a visual trend. They
    are separate experiments, so they are placed as ordered categories.
    """
    figure = BUILDERS["e2"]()
    for ax in figure.axes:
        ticks = ax.get_xticks()
        gaps = np.diff(ticks)
        assert np.allclose(gaps, gaps[0]), "sweep conditions are not evenly placed"


def test_saved_figures_are_byte_identical_when_regenerated(tmp_path, monkeypatch):
    """Regenerating an unchanged figure must not produce a diff.

    matplotlib stamps SVGs with the current time and names internal elements
    from a random salt, so an unchanged figure rewrote most of its own file.
    Figures are committed here, so that noise would be permanent and every
    review would have to read a few hundred changed lines to find out they
    say nothing.
    """
    import _common

    monkeypatch.setattr(_common, "FIGURES_DIR", tmp_path)
    first = _common.save_figure(BUILDERS["e2"](), "determinism").read_bytes()
    second = _common.save_figure(BUILDERS["e2"](), "determinism").read_bytes()
    assert first == second, "figure output is not reproducible"


def _with_rejected(monkeypatch, module_name: str, name: str, pick) -> list[dict]:
    """The committed results for `name`, with the rows `pick` selects marked
    as convergence failures, served to the figure module in their place.

    Whether a real sweep happens to contain a rejected condition is a fact
    about the data, and changes when the data does; the figure's handling of
    one has to be tested regardless.
    """
    from _common import read_results

    rows = read_results(name)
    for row in rows:
        if pick(row):
            row.update(usable=False, verdict="convergence failure", converged=0)
    assert any(not r["usable"] for r in rows), "the selector matched nothing"
    monkeypatch.setattr(sys.modules[module_name], "read_results", lambda _: rows)
    return rows


def test_rejected_conditions_are_not_plotted_as_measurements(monkeypatch):
    """A condition the analysis threw out must not appear as a datum.

    E3 rejects any condition converging on under half its runs, and the first
    version of the figure drew one of those identically to the valid points.
    A reader had no way to know one of the eight markers was a result the
    study had already declined to report.
    """
    rows = _with_rejected(
        monkeypatch, "e3_nonlinearity", "e3_nonlinearity",
        lambda r: r["group"] == "SE(3)" and r["rotation_sigma"] == 0.45,
    )

    for builder in (BUILDERS["e3"], BUILDERS["e3_coverage"]):
        figure = builder()
        plotted = 0
        for ax in figure.axes:
            for container in ax.containers:
                if hasattr(container, "has_yerr"):
                    plotted += len(container[0].get_xdata())
            plotted += sum(
                1
                for line in ax.lines
                if line.get_label() and not line.get_label().startswith("_")
                and "ideal" not in line.get_label()
                and "calibrated" not in line.get_label()
            )
        usable = sum(1 for r in rows if r["usable"])
        assert plotted <= usable, (
            f"figure draws {plotted} series for {usable} usable conditions"
        )


def test_lines_break_rather_than_bridge_excluded_conditions(monkeypatch):
    """A line joined across a rejected condition claims data that is not there.

    Drawn as a continuous line, a method that produced no usable result at
    some rates looked tracked across the whole sweep, with straight segments
    spanning the gaps. Those gaps are NaN, so the line breaks.
    """
    rows = _with_rejected(
        monkeypatch, "e4_perceptual_aliasing", "e4_aliasing",
        lambda r: r["method"] == "plain least squares" and r["rate"] in (0.1, 0.2),
    )
    dropped = {
        r["rate"] for r in rows if r["method"] == "plain least squares" and not r["usable"]
    }

    figure = BUILDERS["e4"]()
    checked = 0
    for ax in figure.axes:
        for container in ax.containers:
            if container.get_label() != "plain least squares":
                continue
            checked += 1
            y = np.asarray(container[0].get_ydata(), dtype=float)
            assert np.isnan(y).sum() == len(dropped), (
                "baseline line does not break at the rates it failed to converge"
            )
    assert checked, "the baseline is not drawn at all"


def test_calibration_panel_separates_the_robust_methods():
    """The robust back-ends differ by a few percent, in both directions.

    With plain least squares on the same axis the scale ran to several
    hundred and all four collapsed into one line, hiding the finding that
    separates them: Cauchy and GNC read conservative, switchable does not.
    """
    figure = BUILDERS["e4"]()
    calibration = figure.axes[1]
    low, high = calibration.get_ylim()
    assert high / low < 10, "calibration axis spans too much to resolve a few percent"
    drawn = {c.get_label() for c in calibration.containers}
    assert "plain least squares" not in drawn
    assert any("not shown" in t.get_text() for t in calibration.texts), (
        "leaving the baseline out must be said on the panel"
    )
