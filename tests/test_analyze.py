"""Analysis results.

These are the numbers the charts and ``paclog stats`` are built from, so they get
asserted directly rather than through pixels.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from paclog import analyze
from paclog.model import ACTION_ORDER, Action
from paclog.plots.base import ACTION_COLORS

AS_OF = datetime(2021, 4, 10, tzinfo=timezone.utc)


def test_action_counts_uses_the_canonical_order_not_data_order(frame):
    counts = analyze.action_counts(frame)
    assert list(counts.index) == [a.value for a in ACTION_ORDER]


def test_action_counts_reports_zero_for_absent_actions():
    """A log with no removals must still show a 'removed' row.

    This is what the original ``value_counts()`` got wrong, and why the pie chart
    lost slices on small logs.
    """
    tiny = pd.DataFrame(
        {
            "package": ["a", "a"],
            "timestamp": pd.to_datetime(["2020-01-01", "2020-02-01"], utc=True),
            "action": ["installed", "upgraded"],
            "version_before": [None, "1"],
            "version_after": ["1", "2"],
        }
    )
    counts = analyze.action_counts(tiny)
    assert counts["removed"] == 0
    assert int(counts.sum()) == 2


def test_every_action_has_a_colour():
    for action in ACTION_ORDER:
        assert action.value in ACTION_COLORS


def test_event_and_package_counts(frame):
    assert analyze.total_events(frame) == 14
    assert analyze.distinct_packages(frame) == 4  # filesystem, tzdata, st, gptfdisk


def test_time_span(frame):
    first, last = analyze.time_span(frame)
    assert first is not None and last is not None
    assert first < last


def test_events_per_hour_keeps_all_24_hours(frame):
    hourly = analyze.events_per_hour(frame)
    assert list(hourly.index) == list(range(24))
    assert hourly.sum() == analyze.total_events(frame)


def test_events_per_hour_fills_the_gaps_with_zero():
    """Hours with no activity must read as 0, not be missing.

    The original grouped and let absent hours disappear, so a machine asleep
    between 02:00 and 06:00 rendered as a chart with a hole in it.
    """
    stamps = pd.to_datetime(["2020-01-01 09:00", "2020-01-01 09:30"], utc=True)
    frame = pd.DataFrame(
        {
            "package": ["a", "a"],
            "timestamp": stamps,
            "action": ["installed", "upgraded"],
            "version_before": [None, "1"],
            "version_after": ["1", "2"],
        }
    )
    hourly = analyze.events_per_hour(frame)
    assert hourly[3] == 0
    assert hourly[9] == 2


def test_events_per_hour_respects_the_display_timezone(frame):
    """The hour buckets must follow the frame's timezone, not UTC."""
    from paclog.config import fixed_offset_tz
    from paclog.loader import to_display_tz

    in_utc = analyze.events_per_hour(frame)
    shifted = analyze.events_per_hour(to_display_tz(frame, fixed_offset_tz(-300)))
    assert int(shifted.sum()) == int(in_utc.sum())  # same events...
    assert int(shifted.idxmax()) != int(in_utc.idxmax())  # ...different clock


def test_events_per_weekday(frame):
    weekday = analyze.events_per_weekday(frame)
    assert list(weekday.index) == list(range(7))
    assert weekday.sum() == analyze.total_events(frame)


def test_events_per_month_has_a_column_per_action(frame):
    table = analyze.events_per_month(frame)
    assert list(table.columns) == [a.value for a in ACTION_ORDER]
    assert int(table.to_numpy().sum()) == analyze.total_events(frame)


def test_events_per_month_buckets_by_local_month(frame):
    table = analyze.events_per_month(frame)
    assert isinstance(table.index, pd.DatetimeIndex)
    assert all(ts.day == 1 for ts in table.index)


def test_events_per_month_without_actions(frame):
    total = analyze.events_per_month(frame, actions=False)
    assert list(total.columns) == ["count"]
    assert int(total["count"].sum()) == analyze.total_events(frame)


def test_top_packages(frame):
    top = analyze.top_packages(frame, Action.UPGRADED, n=1)
    assert list(top.index) == ["filesystem"]
    assert top.iloc[0] == 3


def test_top_packages_breaks_ties_deterministically(frame):
    first = analyze.top_packages(frame, Action.UPGRADED, n=5)
    second = analyze.top_packages(frame, Action.UPGRADED, n=5)
    assert list(first.index) == list(second.index)


def test_top_packages_grid_covers_every_action(frame):
    grid = analyze.top_packages_grid(frame, n=5)
    assert set(grid) == {a.value for a in ACTION_ORDER}


# -- intervals -------------------------------------------------------------- #


def test_upgrade_intervals_are_the_gaps_between_consecutive_upgrades(frame):
    intervals = analyze.upgrade_intervals(frame)
    assert set(intervals.columns) == {"package", "interval_days"}
    assert (intervals["interval_days"] > 0).all()
    # filesystem is upgraded three times, so two gaps.
    assert len(intervals[intervals["package"] == "filesystem"]) == 2


def test_upgrade_intervals_are_ordered_by_time_not_by_file(frame):
    """Two upgrades two days apart must each yield exactly 2.0 days."""
    stamps = pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-05"], utc=True)
    frame = pd.DataFrame(
        {
            "package": ["a"] * 3,
            "timestamp": stamps,
            "action": ["upgraded"] * 3,
            "version_before": ["1", "0", "2"],
            "version_after": ["2", "1", "3"],
        }
    )
    intervals = analyze.upgrade_intervals(frame)
    assert intervals["interval_days"].tolist() == [2.0, 2.0]


def test_upgrade_intervals_ignores_other_actions(frame):
    """A removal between two upgrades must not shorten the measured gap."""
    stamps = pd.to_datetime(["2020-01-01", "2020-01-05", "2020-01-11"], utc=True)
    frame = pd.DataFrame(
        {
            "package": ["a"] * 3,
            "timestamp": stamps,
            "action": ["upgraded", "removed", "upgraded"],
            "version_before": ["1", "2", None],
            "version_after": ["2", None, "3"],
        }
    )
    intervals = analyze.upgrade_intervals(frame)
    assert intervals["interval_days"].tolist() == [10.0]


def test_upgrade_intervals_of_nothing_is_empty_but_usable(frame):
    only_installs = frame[frame["action"] == "installed"]
    assert analyze.upgrade_intervals(only_installs).empty


def test_interval_summary(frame):
    summary = analyze.interval_summary(frame)
    assert "count" in summary.columns
    assert summary["count"].sum() == len(analyze.upgrade_intervals(frame))


# -- install periods -------------------------------------------------------- #


def test_install_periods_close_on_removal(frame):
    """tzdata goes in at 12:53 and comes out at 14:15 the same day."""
    periods = analyze.install_periods(frame, AS_OF)
    tzdata = periods[periods["package"] == "tzdata"].iloc[0]
    span = tzdata["end"] - tzdata["start"]
    assert span.total_seconds() == pytest.approx(82 * 60)
    assert span.days == 0  # a partial day, not a whole one


def test_install_periods_stay_open_until_as_of(frame):
    """gptfdisk is installed and never removed, so it runs to the cutoff."""
    periods = analyze.install_periods(frame, AS_OF)
    gpt = periods[periods["package"] == "gptfdisk"].iloc[0]
    assert gpt["end"] == pd.Timestamp(AS_OF)


def test_install_periods_handle_install_remove_reinstall(frame):
    """st is downgraded, upgraded, then installed and reinstalled: two periods."""
    periods = analyze.install_periods(frame, AS_OF)
    assert len(periods[periods["package"] == "st"]) == 1


def test_install_periods_leave_no_negative_lengths(frame):
    periods = analyze.install_periods(frame, AS_OF)
    assert (periods["end"] >= periods["start"]).all()


def test_install_periods_are_deterministic(frame):
    first = analyze.install_periods(frame, AS_OF)
    second = analyze.install_periods(frame, AS_OF)
    pd.testing.assert_frame_equal(first, second)


def test_currently_installed_excludes_removed_packages(frame):
    """tzdata and filesystem are both removed before the cutoff; st and gptfdisk are not."""
    held = set(analyze.currently_installed(frame, AS_OF))
    assert held == {"st", "gptfdisk"}


def test_package_lifetime_is_sorted_longest_first(frame):
    lifetimes = analyze.package_lifetime(frame, AS_OF)
    assert list(lifetimes["days"]) == sorted(lifetimes["days"], reverse=True)
    assert (lifetimes["days"] >= 0).all()


def test_long_gaps_finds_the_idle_stretch(frame):
    gaps = analyze.long_gaps(frame, days=300)
    assert len(gaps) >= 1
    assert (gaps["days"] >= 300).all()


# -- summary ---------------------------------------------------------------- #


def test_summary_reports_the_headline_numbers(frame):
    payload = analyze.summary(frame, AS_OF)
    assert payload["events"] == 14
    assert payload["packages"] == 4
    assert payload["actions"]["downgraded"] == 1
    assert payload["currently_installed"] == 2
    assert "upgrade_interval_days" in payload


def test_summary_is_json_serialisable(frame):
    import json

    json.dumps(analyze.summary(frame, AS_OF))


def test_busiest_hour(frame):
    assert analyze.busiest_hour(frame) is not None
    assert 0 <= analyze.busiest_hour(frame) <= 23


def test_empty_frame_does_not_explode():
    from paclog.loader import empty_frame

    empty = empty_frame()
    assert analyze.total_events(empty) == 0
    assert analyze.action_counts(empty).sum() == 0
    assert len(analyze.events_per_hour(empty)) == 24
    assert analyze.time_span(empty) == (None, None)
    assert analyze.install_periods(empty, AS_OF).empty
    assert analyze.summary(empty)["events"] == 0
    assert analyze.busiest_hour(empty) is None


def test_missing_timestamp_column_is_reported_clearly():
    with pytest.raises(KeyError, match="timestamp"):
        analyze.action_counts(pd.DataFrame({"package": ["a"]}))
