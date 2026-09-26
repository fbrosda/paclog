"""Chart registry, smoke rendering, and byte-level reproducibility."""

from __future__ import annotations

from datetime import datetime, timezone

import matplotlib
import pytest

from paclog import analyze, plots
from paclog.plots import Context, names, render, render_all
from paclog.plots.base import ACTION_COLORS, color_for
from paclog.plots.timeline import stable_unit

AS_OF = datetime(2021, 4, 10, tzinfo=timezone.utc)


@pytest.fixture
def ctx() -> Context:
    return Context(as_of=datetime(2021, 4, 10, tzinfo=timezone.utc), seed=0, options={"top_n": 150})


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
