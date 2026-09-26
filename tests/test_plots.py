"""Chart registry, smoke rendering, and byte-level reproducibility."""

from __future__ import annotations

from datetime import datetime, timezone

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from paclog import analyze, plots
from paclog.plots import Context, names, render, render_all
from paclog.plots.base import ACTION_COLORS, color_for
from paclog.plots.timeline import stable_unit

AS_OF = datetime(2021, 4, 10, tzinfo=timezone.utc)


@pytest.fixture
def ctx() -> Context:
    return Context(as_of=datetime(2021, 4, 10, tzinfo=timezone.utc), seed=0, options={"top_n": 150})


@pytest.fixture(autouse=True)
def close_figures():
    """Rendering tests each leak a figure, and matplotlib warns past 20 open.

    ``paclog.plot`` closes its own figures in ``save``; a test that calls
    ``render`` directly keeps the figure, so without this the suite trips the
    warning as soon as it grows past a certain number of chart tests.
    """
    yield
    plt.close("all")


def test_registry_is_populated():
    assert names()
    assert "events-per-hour" in names()
    assert "timeline" in names()


def test_timeline_is_opt_in_only():
    """It costs far more than the rest together, so a bare `paclog plot` skips it."""
    assert "timeline" not in plots.default_names()
    assert "timeline" in names()


def test_every_chart_has_a_distinct_filename():
    filenames = [plots.get(n).filename for n in names()]
    assert len(filenames) == len(set(filenames))
    assert all(f.endswith(".svg") for f in filenames)


def test_every_action_has_a_colour():
    from paclog.model import ACTION_ORDER

    for action in ACTION_ORDER:
        assert action.value in ACTION_COLORS


def test_unknown_chart_raises_with_the_available_list():
    with pytest.raises(KeyError, match="available"):
        plots.get("no-such-chart")


def test_every_chart_renders(frame, ctx):
    """Smoke test: a chart that raises is a bug, whatever it looks like."""
    for name in names():
        fig = render(name, frame, ctx)
        assert fig.get_axes(), f"{name} produced no axes"


def test_render_all_writes_files(tmp_path, frame, ctx):
    written = render_all(frame, ctx, tmp_path, only=["events-per-hour", "action-distribution"])
    assert [p.name for p in written] == ["events_per_hour.svg", "action_distribution.svg"]
    assert all(p.is_file() and p.stat().st_size > 0 for p in written)


def test_render_all_creates_the_output_directory(tmp_path, frame, ctx):
    target = tmp_path / "does" / "not" / "exist"
    render_all(frame, ctx, target, only=["events-per-hour"])
    assert target.is_dir()


def test_render_all_rejects_unknown_names(tmp_path, frame, ctx):
    with pytest.raises(KeyError, match="unknown chart"):
        render_all(frame, ctx, tmp_path, only=["nope"])


def test_charts_are_byte_reproducible(tmp_path, frame, ctx):
    """The property the original scripts could not offer.

    Two renders in the same process must produce identical bytes. Combined with
    ``svg.hashsalt`` and the suppressed ``dc:date``, this is what makes a chart
    safe to commit.
    """
    name = "events-per-month"
    first = render_all(frame, ctx, tmp_path / "a", only=[name])[0]
    second = render_all(frame, ctx, tmp_path / "b", only=[name])[0]
    assert first.read_bytes() == second.read_bytes()


def test_saved_svg_has_no_creation_timestamp(tmp_path, frame, ctx):
    written = render_all(frame, ctx, tmp_path, only=["action-distribution"])[0]
    assert b"dc:date" not in written.read_bytes()


def test_style_is_idempotent():
    from paclog.plots.base import apply_style

    apply_style()
    apply_style()  # must not raise


def test_timeline_colour_is_stable_across_processes():
    """``hash()`` is salted per process for str; crc32 is not."""
    assert stable_unit("linux") == stable_unit("linux")
    assert stable_unit("linux") != stable_unit("gimp")
    assert 0.0 <= stable_unit("anything") < 1.0


def test_timeline_respects_the_package_cap(frame, ctx):
    capped = Context(as_of=ctx.as_of, seed=0, options={"top_n": 1})
    render("timeline", frame, capped)
    render("timeline", frame, Context(as_of=ctx.as_of, seed=0, options={"top_n": 0}))


def test_timeline_plots_every_package_by_default(frame):
    """The default must not silently omit packages.

    An earlier version capped at 150 rows, which on a real log dropped 2 658 of
    2 808 packages and made the chart describe a different machine than the one it
    was run on. The cap is now opt-in via ``--top-n`` only.
    """
    periods = analyze.install_periods(frame, AS_OF)
    expected = periods["package"].nunique()

    fig = render("timeline", frame, Context(as_of=AS_OF, options={"top_n": 0}))
    ylim = fig.axes[0].get_ylim()
    # ylim spans [n-0.5, -0.5], so the row count is exactly its height.
    assert round(ylim[0] - ylim[1]) == expected


def test_timeline_cap_actually_reduces_rows(frame):
    total = analyze.install_periods(frame, AS_OF)["package"].nunique()
    capped = render("timeline", frame, Context(as_of=AS_OF, options={"top_n": 1}))
    ylim = capped.axes[0].get_ylim()
    assert round(ylim[0] - ylim[1]) == 1
    assert total > 1


def test_timeline_labels_every_package(big_frame):
    """Every package gets a row *and* its own label.

    Two earlier revisions dropped packages (a 150-package cap) and then strided
    the y labels to ~80 of 2 808. Both were the wrong trade: the cap changed what
    the chart was about, and the striding saved SVG bytes at the price of making
    the chart unreadable -- which is the chart's entire job. The canvas grows to
    fit instead.
    """
    fig = render("timeline", big_frame, Context(as_of=AS_OF, options={"top_n": 0}))
    ax = fig.axes[0]
    labels = [text.get_text() for text in ax.get_yticklabels()]
    assert sorted(labels) == sorted(analyze.install_periods(big_frame, AS_OF)["package"].unique())
    assert round(ax.get_ylim()[0] - ax.get_ylim()[1]) == big_packages(big_frame)


def test_timeline_height_grows_to_fit_rather_than_capping_rows(big_frame):
    """Tall means tall; it never means "drop rows to stay a sensible size"."""
    fig = render("timeline", big_frame, Context(as_of=AS_OF, options={"top_n": 0}))
    from paclog.config import INCHES_PER_PACKAGE, MAX_TIMELINE_HEIGHT_IN

    height = fig.get_size_inches()[1]
    assert height == pytest.approx(INCHES_PER_PACKAGE * big_packages(big_frame))
    assert height < MAX_TIMELINE_HEIGHT_IN  # 400 packages must not hit the ceiling


# -- install-period lifetime charts ----------------------------------------- #


def names_drawn(fig) -> list[str]:
    """The package names on the whole-time chart, which is text rather than bars."""
    return [t.get_text() for t in fig.axes[0].texts if t.get_fontfamily() == ["monospace"]]


def test_installed_whole_time_lists_every_qualifying_package(frame, ctx):
    """A list chart has to list all of them.

    331 of 2 808 packages qualify on the author's log, and the chart is the only
    place that set exists, so dropping names to keep the page tidy would drop the
    entire content of the chart.
    """
    whole = analyze.installed_whole_time(frame, ctx.as_of)
    fig = render("installed-whole-time", frame, ctx)
    drawn = names_drawn(fig)
    assert sorted(drawn) == sorted(whole["package"])
    assert len(drawn) == len(set(drawn)) == len(whole)


def test_installed_whole_time_never_drops_a_name(big_frame, ctx):
    """400 packages, all installed together and never removed: all 400 are drawn.

    The list chart lays names out in columns rather than capping them, for the
    same reason the timeline draws every row.
    """
    expected = analyze.installed_whole_time(big_frame, ctx.as_of)
    assert len(expected) == 400
    fig = render("installed-whole-time", big_frame, ctx)
    assert len(names_drawn(fig)) == 400
    # More names means a taller page, not a smaller font and fewer names.
    assert fig.get_size_inches()[1] > 6.0


def test_installed_whole_time_names_do_not_collide(frame, ctx):
    """Column pitch has to clear the longest name in the data.

    ``ca-certificates-mozilla`` is 23 characters, and monospace 7pt is 0.065in per
    character, so a column narrower than about 1.7in runs the longest names into
    their neighbour. Nothing asserts this by eye, so it is asserted here.
    """
    from paclog.plots.lifetime import COLUMN_WIDTH_IN, NAME_FONTSIZE, ROW_HEIGHT_IN

    fig = render("installed-whole-time", frame, ctx)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(renderer) for t in fig.axes[0].texts if t.get_fontfamily() == ["monospace"]]
    assert not any(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1 :])

    widest_in = max(b.width for b in boxes) / fig.dpi
    assert COLUMN_WIDTH_IN > widest_in
    assert fig.dpi * ROW_HEIGHT_IN > max(b.height for b in boxes)
    assert NAME_FONTSIZE > 0


def test_installed_whole_time_keeps_every_name_inside_the_canvas(frame, ctx):
    """The axes has to span the figure, or the last columns are silently cropped."""
    fig = render("installed-whole-time", frame, ctx)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    for text in fig.axes[0].texts:
        box = text.get_window_extent(renderer)
        assert box.x0 >= 0 and box.x1 <= width
        assert box.y0 >= 0 and box.y1 <= height


def test_installed_whole_time_reads_alphabetically(frame, ctx):
    """Column-major fill, so reading down a column and across stays alphabetical."""
    fig = render("installed-whole-time", frame, ctx)
    by_position = sorted(
        fig.axes[0].texts,
        key=lambda t: (t.get_position()[0], -t.get_position()[1]),
    )
    drawn = [t.get_text() for t in by_position if t.get_fontfamily() == ["monospace"]]
    assert drawn == sorted(drawn)


def test_package_lifetime_ranks_only_periods_that_ended(frame, ctx):
    """The bars are completed lifetimes, longest at the top.

    gptfdisk is the fixture's still-installed package. It is absent here and
    present on ``installed-whole-time``, which is the whole split.
    """
    completed = analyze.completed_lifetimes(frame, ctx.as_of)
    fig = render("package-lifetime", frame, ctx)
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()] == list(completed["package"].iloc[::-1])
    assert len(ax.patches) == len(completed)
    assert "gptfdisk" not in {t.get_text() for t in ax.get_yticklabels()}


def test_package_lifetime_bars_are_not_all_the_same_length(censored_frame, ctx):
    """The regression this rework exists for.

    60 packages installed at the first instant and never removed all score
    exactly the length of the log, so ranking every period put 40 identical bars
    on the chart and pushed the three real lifetimes off the bottom. The ranked
    periods must now differ from each other.
    """
    fig = render("package-lifetime", censored_frame, ctx)
    widths = [p.get_width() for p in fig.axes[0].patches]
    assert len(widths) == 3
    assert len(set(widths)) == 3
    assert sorted(widths) == [30, 200, 900]


def test_package_lifetime_caps_at_forty_and_says_so(many_completed_frame, ctx):
    """A summary view, labelled as one: the title carries the count it ranked from."""
    from paclog.plots.lifetime import TOP_PERIODS

    completed = analyze.completed_lifetimes(many_completed_frame, ctx.as_of)
    assert len(completed) == 50
    fig = render("package-lifetime", many_completed_frame, ctx)
    assert len(fig.axes[0].patches) == TOP_PERIODS
    assert f"top {TOP_PERIODS} of {len(completed)}" in fig.axes[0].get_title()


def test_lifetime_charts_handle_a_log_with_no_completed_periods(never_removed_frame, ctx):
    """Both charts have to render the empty case rather than raise."""
    assert analyze.completed_lifetimes(never_removed_frame, ctx.as_of).empty
    assert render("package-lifetime", never_removed_frame, ctx).axes
    assert render("installed-whole-time", never_removed_frame, ctx).axes


def _lifetime_frame(rows) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["package", "timestamp", "action"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["version_before"] = None
    frame["version_after"] = "1-1"
    return frame


@pytest.fixture
def censored_frame():
    """60 never-removed packages installed together, plus 3 that were removed.

    This is the shape that produced 40 identical bars: every still-installed
    package scores exactly ``as_of - installed``, so ranking all periods together
    put the base system on top and hid the only three lifetimes that vary.
    """
    installed = pd.Timestamp("2016-01-01", tz="UTC")
    rows = [(f"base{i:02d}", installed, "installed") for i in range(60)]
    for index, days in enumerate((30, 200, 900)):
        start = installed + pd.Timedelta(index + 1, unit="D")
        rows.append((f"gone{index}", start, "installed"))
        rows.append((f"gone{index}", start + pd.Timedelta(days, unit="D"), "removed"))
    return _lifetime_frame(rows)


@pytest.fixture
def many_completed_frame():
    """50 completed periods of distinct lengths, to exercise the top-40 cap."""
    installed = pd.Timestamp("2016-01-01", tz="UTC")
    rows = []
    for index in range(50):
        start = installed + pd.Timedelta(index + 1, unit="D")
        rows.append((f"pkg{index:02d}", start, "installed"))
        rows.append((f"pkg{index:02d}", start + pd.Timedelta(index + 1, unit="D"), "removed"))
    return _lifetime_frame(rows)


@pytest.fixture
def never_removed_frame():
    """Nothing is ever removed, so there are no completed lifetimes to rank."""
    installed = pd.Timestamp("2016-01-01", tz="UTC")
    rows = [(f"pkg{index}", installed + pd.Timedelta(index, unit="h"), "installed") for index in range(5)]
    return _lifetime_frame(rows)


# -- monthly charts --------------------------------------------------------- #


def test_events_per_month_is_one_panel_per_action(frame):
    """One panel per action, each on its own scale, stacked in ACTION_ORDER.

    This replaced the original's single stacked bar per month. Three earlier
    shapes were tried and all three lost information: a 150-package cap on the
    timeline, a plain total-events series with no action breakdown, and the
    stacked bar, which cannot show a mix where one action is 90% of the events.
    """
    from paclog.model import ACTION_ORDER

    table = analyze.events_per_month(frame)
    present = [a for a in ACTION_ORDER if a.value in table.columns]
    fig = render("events-per-month", frame, Context(as_of=AS_OF))
    assert [ax.get_ylabel() for ax in fig.axes] == [a.value for a in present]
    for ax, action in zip(fig.axes, present):
        assert len(ax.patches) == table.shape[0]
        assert all(p.get_facecolor()[:3] == matplotlib.colors.to_rgb(color_for(action.value))
                   for p in ax.patches)


def test_events_per_month_keeps_every_month(frame):
    """Splitting by action must not drop months, including the empty ones."""
    table = analyze.events_per_month(frame)
    fig = render("events-per-month", frame, Context(as_of=AS_OF))
    for ax in fig.axes:
        assert len(ax.patches) == table.shape[0]


def big_packages(frame) -> int:
    return analyze.install_periods(frame, AS_OF)["package"].nunique()


@pytest.fixture
def big_frame():
    """A few hundred packages, to exercise the all-labels-on-the-timeline path."""
    import pandas as pd

    rows = []
    start = pd.Timestamp("2016-01-01", tz="UTC")
    for index in range(400):
        name = f"pkg{index:04d}"
        rows.append((name, start, "installed", None, "1-1"))
        rows.append((name, start + pd.Timedelta(30, unit="D"), "upgraded", "1-1", "2-1"))
    frame = pd.DataFrame(
        rows, columns=["package", "timestamp", "action", "version_before", "version_after"]
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


@pytest.fixture
def skewed_frame():
    """One action at ~99.99% of events, the shape a real pacman log has.

    Upgrades are 90% of events on the author's log, and 3 000-to-1 here, so the
    downgrade series is the canary: on a shared y-axis it collapses to a
    zero-height line. The upgrades are deliberately piled into a single month
    rather than smeared over ten, because it is the *peak* of each action that
    sets that panel's scale -- spreading them out would make the fixture look
    harmless when it is meant to be pathological.
    """
    import pandas as pd

    base = pd.Timestamp("2016-01-01", tz="UTC")
    january = base + pd.Timedelta(20, unit="h")
    rows = [(f"pkg{i}", january, "upgraded", "1-1", "2-1") for i in range(3000)]
    # February holds a single event, and it is the minority action.
    rows.append(("pkg0", base + pd.Timedelta(35, unit="D"), "downgraded", "2-1", "1-1"))
    frame = pd.DataFrame(
        rows, columns=["package", "timestamp", "action", "version_before", "version_after"]
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


def test_a_dominant_action_does_not_flatten_the_others(skewed_frame):
    """Each panel scales to its own peak, so the minority action stays visible."""
    fig = render("events-per-month", skewed_frame, Context(as_of=AS_OF))
    scales = {ax.get_ylabel(): ax.get_ylim()[1] for ax in fig.axes}
    peaks = analyze.events_per_month(skewed_frame).max()
    assert scales["upgraded"] > 1000 * scales["downgraded"]
    # The downgraded panel must still resolve its single event.
    assert scales["downgraded"] >= peaks["downgraded"]
    # Each y-limit tracks that panel's own peak -- a shared axis would be 1:1.
    for action, scale in scales.items():
        assert 1 <= scale <= 1.1 * peaks[action]


def test_monthly_charts_handle_a_single_event_month(skewed_frame):
    assert not analyze.events_per_month(skewed_frame).empty
    render("events-per-month", skewed_frame, Context(as_of=AS_OF))
