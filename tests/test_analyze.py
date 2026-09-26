"""Analysis results.

These are the numbers the charts and ``paclog stats`` are built from, so they get
asserted directly rather than through pixels.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
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


def test_package_lifetime_flags_the_periods_that_never_ended(frame):
    """Right-censoring is a property of the observation, so it rides along with it.

    gptfdisk is still installed, so its 1 854 days is a lower bound. tzdata was
    removed after 82 minutes, so its number is a lifetime. Ranking the two as
    if they were the same measurement is what made the lifetime chart useless.
    """
    lifetimes = analyze.package_lifetime(frame, AS_OF)
    censored = lifetimes.set_index("package")["censored"]
    assert censored["gptfdisk"]
    assert not censored["tzdata"]
    assert not censored["filesystem"]


# -- the two lifetime views -------------------------------------------------- #


def test_initial_install_window_ends_at_the_last_event_before_the_quiet_stretch(frame):
    """The fixture's build runs 10:53-12:15 on 2016-03-12, then goes quiet for 9 months.

    The window ends on the last event *before* the silence. Ending it on the
    first event after would swallow the next session, which is a different era
    of the machine's life and not part of the original system.
    """
    start, end = analyze.initial_install_window(frame)
    assert start == pd.Timestamp("2016-03-12 10:53", tz="UTC")
    assert end == pd.Timestamp("2016-03-12 12:15", tz="UTC")


def test_initial_install_window_falls_back_to_the_first_day_with_no_quiet_stretch():
    """A machine touched twice a day has no era boundary to find.

    The fallback is the first calendar day, because "this arrived with the
    system" is a stronger claim than a log with no silent day can support.
    """
    twice_daily = pd.DataFrame(
        {
            "package": [f"pkg{i}" for i in range(20)],
            "timestamp": pd.date_range("2020-01-01", periods=20, freq="12h", tz="UTC"),
            "action": ["installed"] * 20,
            "version_before": [None] * 20,
            "version_after": ["1-1"] * 20,
        }
    )
    start, end = analyze.initial_install_window(twice_daily)
    assert start == pd.Timestamp("2020-01-01", tz="UTC")
    assert end == pd.Timestamp("2020-01-02", tz="UTC")


def test_initial_install_window_collapses_when_events_are_a_day_apart():
    """One event per day is a day-long silence between every pair of them.

    So the build window is the single instant of the first event, and only what
    was installed in that instant counts as having come with the machine. The
    conservative outcome, the same reading as the fallback above.
    """
    daily = pd.DataFrame(
        {
            "package": [f"pkg{i}" for i in range(10)],
            "timestamp": pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC"),
            "action": ["installed"] * 10,
            "version_before": [None] * 10,
            "version_after": ["1-1"] * 10,
        }
    )
    start, end = analyze.initial_install_window(daily)
    assert start == end == pd.Timestamp("2020-01-01", tz="UTC")


def test_installed_whole_time_is_exactly_the_never_removed_base_system(frame):
    """gptfdisk and nothing else.

    filesystem and tzdata were installed during the build too, but both were
    removed, so neither is installed the whole time. st is still installed, but
    it arrived in 2021, five years into the log.
    """
    assert analyze.installed_whole_time(frame, AS_OF)["package"].tolist() == ["gptfdisk"]


def test_installed_whole_time_excludes_a_package_removed_and_reinstalled():
    """Present now and present at the build is not the same as present throughout.

    A removal in the middle splits the history in two. The package's open period
    starts at the reinstall, which is outside the window, so it is not on the
    list -- and the completed period it left behind does not drag it back on.
    """
    frame = pd.DataFrame(
        {
            "package": ["keeper", "churn", "churn", "churn", "late"],
            "timestamp": pd.to_datetime(
                [
                    "2016-03-12 10:00",  # build: two packages arrive
                    "2016-03-12 10:00",
                    "2020-05-05 12:00",  # churn is removed
                    "2021-01-01 09:00",  # and put back, so it is present again
                    "2016-06-06 08:00",  # late arrives after the build
                ],
                utc=True,
            ),
            "action": ["installed", "installed", "removed", "installed", "installed"],
            "version_before": [None, None, "1-1", None, None],
            "version_after": ["1-1", "1-1", None, "1-1", "1-1"],
        }
    )
    as_of = pd.Timestamp("2026-01-01", tz="UTC")
    assert analyze.installed_whole_time(frame, as_of)["package"].tolist() == ["keeper"]


def test_installed_whole_time_is_returned_alphabetically():
    """It is drawn as a list, so the order has to mean something."""
    whole = analyze.installed_whole_time(big_install_frame(), AS_OF)
    assert len(whole) == 60
    assert whole["package"].is_monotonic_increasing


def test_completed_lifetimes_excludes_still_installed_packages(frame):
    """The only periods with a real lifetime are the ones that ended."""
    completed = analyze.completed_lifetimes(frame, AS_OF)
    assert completed["package"].tolist() == ["filesystem", "tzdata"]
    assert not completed["censored"].any()
    assert len(completed) < len(analyze.package_lifetime(frame, AS_OF))


def test_the_two_lifetime_views_never_describe_the_same_period(frame):
    """The split is by censoring, so no period can be in both.

    Not quite the same as the two package sets being disjoint, and deliberately
    so: a package that was removed and reinstalled during the build has one
    completed period *and* one that never ended, and the author's log contains
    exactly one of those (``xf86-input-synaptics``, out and back four minutes
    later). It belongs on the whole-time list and its stale period is still a
    completed lifetime. What must hold is that the whole-time periods themselves
    are never ranked as completed ones.
    """
    lifetimes = analyze.package_lifetime(frame, AS_OF)
    _, window_end = analyze.initial_install_window(frame)
    whole = lifetimes[lifetimes["censored"] & (lifetimes["start"] <= window_end)]
    completed = analyze.completed_lifetimes(frame, AS_OF)
    keys = lambda table: set(zip(table["package"], table["start"]))
    assert keys(whole) & keys(completed) == set()
    assert len(whole) == len(analyze.installed_whole_time(frame, AS_OF))


def big_install_frame() -> pd.DataFrame:
    """60 packages that all arrived at the first instant and were never removed.

    The shape that made the original lifetime chart useless: 60 censored periods
    whose days are all the length of the log, and 3 completed ones that vary.
    """
    start = pd.Timestamp("2016-01-01", tz="UTC")
    rows = [(f"base{i:02d}", start, "installed") for i in range(60)]
    for index, days in enumerate((30, 200, 900)):
        began = start + pd.Timedelta(index, unit="D")
        rows.append((f"gone{index}", began, "installed"))
        rows.append((f"gone{index}", began + pd.Timedelta(days, unit="D"), "removed"))
    frame = pd.DataFrame(rows, columns=["package", "timestamp", "action"])
    frame["version_before"] = None
    frame["version_after"] = "1-1"
    return frame


def test_ranking_every_period_together_is_the_bug_the_split_avoids():
    """Pin the pathology itself, not just the fix.

    Ranking all periods by length puts the 60 still-installed base packages on
    top, every one of them the same length, and pushes all three real lifetimes
    off the chart. That is the chart the user reported: 40 identical bars.
    """
    as_of = pd.Timestamp("2026-01-01", tz="UTC")
    frame = big_install_frame()

    everything = analyze.package_lifetime(frame, as_of).head(40)
    assert everything["days"].nunique() == 1
    assert len(everything) == 40

    completed = analyze.completed_lifetimes(frame, as_of)
    assert completed["days"].tolist() == [900, 200, 30]
    assert len(analyze.installed_whole_time(frame, as_of)) == 60


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


# -- days, months, and the calendar's inputs --------------------------------- #


def test_events_per_day_covers_the_whole_span_including_the_empty_days(frame):
    """Every day between the first and last event, not only the days with events.

    This is the whole basis of the calendar chart. A series that carried only the
    active days would make a fortnight away invisible, which is the one thing the
    chart exists to show.
    """
    daily = analyze.events_per_day(frame)
    first, last = analyze.time_span(frame)
    assert daily.index[0] == first.normalize()
    assert daily.index[-1] == last.normalize()
    assert (daily.index.to_series().diff().dropna().dt.days == 1).all()
    assert daily.sum() == len(frame)
    assert (daily >= 0).all()


def test_active_days_counts_only_days_with_events(frame):
    daily = analyze.events_per_day(frame)
    active = analyze.active_days(frame)
    assert len(active) == int((daily > 0).sum())
    assert active.sum() == len(frame)
    assert len(active) < len(daily)


def test_month_start_snaps_to_the_first_of_the_month():
    stamps = pd.Series(pd.to_datetime(["2021-03-17 13:45", "2021-04-01 00:00"], utc=True))
    assert list(analyze.month_start(stamps)) == [
        pd.Timestamp("2021-03-01", tz="UTC"),
        pd.Timestamp("2021-04-01", tz="UTC"),
    ]


def test_month_range_is_inclusive_and_contiguous():
    months = analyze.month_range(
        pd.Timestamp("2021-01-15", tz="UTC"), pd.Timestamp("2021-04-02", tz="UTC")
    )
    assert list(months) == [pd.Timestamp(2021, month, 1, tz="UTC") for month in (1, 2, 3, 4)]


def test_month_range_collapses_when_both_ends_are_in_one_month():
    moment = pd.Timestamp("2021-02-14", tz="UTC")
    assert len(analyze.month_range(moment, moment + pd.Timedelta(1, unit="h"))) == 1


# -- transactions ------------------------------------------------------------ #


def test_transactions_breaks_on_the_gap_threshold(frame):
    """Sessions are contiguous runs of events; a quiet minute ends one.

    The threshold is a judgement call and it is the one number that decides what
    this chart means, so it is a named constant rather than a literal buried in
    the function, and the chart prints it.
    """
    assert analyze.SESSION_GAP_SECONDS == 60.0
    table = analyze.transactions(frame)
    assert list(table.columns) == ["start", "end", "events", "packages"]
    assert table["events"].sum() == len(frame)
    assert (table["events"] > 0).all()
    assert (table["packages"] <= table["events"]).all()


def test_a_wider_threshold_never_increases_the_session_count(frame):
    """Monotone in the threshold: a longer quiet period can only merge sessions."""
    narrow = len(analyze.transactions(frame, gap_seconds=0))
    default = len(analyze.transactions(frame))
    wide = len(analyze.transactions(frame, gap_seconds=3600))
    assert wide <= default <= narrow
    assert narrow >= 1


def test_transactions_carry_their_boundaries(frame):
    table = analyze.transactions(frame)
    ordered = table.sort_values("start")
    assert (ordered["start"].to_numpy() <= ordered["end"].to_numpy()).all()
    # Consecutive sessions must not overlap, or the same event is in two of them.
    assert (ordered["start"].to_numpy()[1:] >= ordered["end"].to_numpy()[:-1]).all()


def test_transaction_size_summary_describes_the_sessions(frame):
    summary = analyze.transaction_size_summary(frame)
    sizes = analyze.transactions(frame)["packages"]
    assert summary["count"] == len(sizes)
    assert summary["50%"] == sizes.median()
    assert summary["max"] == sizes.max()
    assert summary["min"] >= 1
    # A session cannot touch more distinct packages than it has events.
    assert sizes.max() <= len(frame)


def test_transactions_per_month_partitions_the_sessions(frame):
    monthly = analyze.transactions_per_month(frame)
    assert monthly["sessions"].sum() == len(analyze.transactions(frame))
    assert monthly["events"].sum() == len(frame)
    assert monthly.index.is_monotonic_increasing


# -- the pooled interval distribution ---------------------------------------- #


def test_upgrade_interval_distribution_shares_sum_to_one_hundred(frame):
    table = analyze.upgrade_interval_distribution(frame)
    assert table["count"].sum() == len(analyze.upgrade_intervals(frame))
    assert table["share"].sum() == pytest.approx(100.0)
    assert (table["low_days"] < table["high_days"]).all()
    assert (table["low_days"] > 0).all()


def test_upgrade_interval_distribution_starts_at_the_shortest_real_interval(frame):
    """The first bin opens at the smallest positive gap, so nothing is off-scale.

    A zero-day interval -- two upgrades in the same second -- is below that edge,
    which is why the counts are computed on clipped values: ``np.histogram``
    discards anything outside the edges rather than counting it in the nearest bin.
    """
    table = analyze.upgrade_interval_distribution(frame)
    days = analyze.upgrade_intervals(frame)["interval_days"]
    assert table["low_days"].iloc[0] == pytest.approx(days[days > 0].min())
    assert table["count"].iloc[0] > 0
    assert (table["low_days"] < table["high_days"]).all()


def test_upgrade_interval_distribution_is_empty_without_upgrades():
    frame = pd.DataFrame(
        {
            "package": ["a", "a"],
            "timestamp": pd.to_datetime(["2021-01-01", "2021-02-01"], utc=True),
            "action": ["installed", "removed"],
        }
    )
    assert analyze.upgrade_interval_distribution(frame).empty


# -- installed set over time -------------------------------------------------- #


def test_installed_over_time_ends_at_as_of_not_the_month_boundary(frame):
    """The last row is the state right now, so it equals the headline numbers."""
    table = analyze.installed_over_time(frame, AS_OF)
    final = table.iloc[-1]
    assert final.name == AS_OF
    assert int(final["installed"]) == len(analyze.currently_installed(frame, AS_OF))
    assert int(final["ever_seen"]) == analyze.distinct_packages(frame)


def test_installed_over_time_never_exceeds_ever_seen(frame):
    table = analyze.installed_over_time(frame, AS_OF)
    assert (table["installed"] <= table["ever_seen"]).all()
    assert (table["ever_seen"].diff().dropna() >= 0).all()  # a package is seen once


def test_installed_over_time_counts_a_removed_and_reinstalled_package_once(frame):
    """A period is in force while start <= t <= end, so a gap is not two packages."""
    table = analyze.installed_over_time(frame, AS_OF)
    assert (table["installed"] >= 0).all()
    assert table["installed"].max() <= table["ever_seen"].max()


# -- staleness --------------------------------------------------------------- #


def test_staleness_now_is_days_since_the_last_event(frame):
    now = analyze.staleness_now(frame, AS_OF)
    assert now.index.name == "package"
    assert set(now.index) == set(frame["package"])
    last = frame.groupby("package")["timestamp"].max()
    assert (now >= 0).all()
    # A package touched at the first instant of the log is the stalest thing here,
    # and its score has to be the whole span rather than something shorter.
    oldest = now.idxmax()
    assert now[oldest] == pytest.approx((AS_OF - last[oldest]).total_seconds() / 86400)
    assert now[oldest] >= now.min()


def test_staleness_is_measured_at_month_ends_so_it_is_never_negative(frame):
    """The bug this closes: measured at a month *start* a package touched on the
    20th scores minus twenty days, and 2 546 such cells came out negative."""
    grid = analyze.package_staleness(frame, AS_OF)
    finite = grid.days[~pd.isna(grid.days)]
    assert (finite >= 0).all()
    assert len(finite)


def test_staleness_leaves_pre_install_months_empty(frame):
    """A cell for a month that closed before the package existed is NaN, not zero.

    The predicate is the month's *closing* boundary, not its start: a package
    installed on the 12th is measured for that whole month, because the cell asks
    what the state was when the month handed over. Testing against the month start
    instead would demand a blank for the month the package arrived in.
    """
    grid = analyze.package_staleness(frame, AS_OF)
    firsts = frame.groupby("package")["timestamp"].min()
    closes = pd.DatetimeIndex(grid.months + pd.offsets.MonthBegin(1))
    for index, package in enumerate(grid.packages):
        row = grid.days[index]
        before = [c for c, close in enumerate(closes) if close < firsts[package]]
        assert pd.isna(row[before]).all(), package
        assert not pd.isna(row[len(before) :]).any(), package
        assert before == list(range(len(before)))  # the blanks are a prefix, not a hole


def test_staleness_counts_a_package_from_its_first_event_not_its_first_install(frame):
    """A package first seen mid-upgrade existed before the log knew about it.

    ``st`` arrives in the fixture as an ``upgraded`` event, so ``install_periods``
    has no install to point at and reports 2021 as its first period. Reading that
    into the heatmap would blank four years of a package that was plainly there.
    The grid measures from the first *event*, which understates nothing.
    """
    grid = analyze.package_staleness(frame, AS_OF)
    row = grid.days[grid.packages.index("st")]
    first = frame.loc[frame["package"] == "st", "timestamp"].min()
    assert first.year == 2016 and first.month == 12
    # Blank for the nine months before it was first seen, measured from then on --
    # four more years of history than the install period knows about.
    assert pd.isna(row[:9]).all()
    assert not pd.isna(row[9:]).any()
    assert analyze.install_periods(frame, AS_OF).query("package == 'st'")["start"].min().year == 2021


def test_staleness_last_column_is_the_current_staleness(frame):
    grid = analyze.package_staleness(frame, AS_OF)
    now = analyze.staleness_now(frame, AS_OF)
    assert grid.days[:, -1] == pytest.approx(now.reindex(grid.packages).to_numpy())


def test_staleness_carries_the_last_known_event_forward(frame):
    """Every cell is "boundary minus the most recent event before that boundary".

    Checked by brute force rather than by a property of the fast path. The tempting
    property -- staleness only ever goes up -- is false, and a test asserting it
    would have passed for the wrong reason: staleness *must* drop in a month where
    the package was touched, which is the entire point of the metric. gptfdisk
    falls from 475 days to 2 in July 2017 for exactly that reason.
    """
    grid = analyze.package_staleness(frame, AS_OF)
    boundaries = pd.DatetimeIndex(grid.months + pd.offsets.MonthBegin(1))
    boundaries = boundaries.where(boundaries <= grid.as_of, grid.as_of)
    for index, package in enumerate(grid.packages):
        stamps = frame.loc[frame["package"] == package, "timestamp"]
        for column, boundary in enumerate(boundaries):
            earlier = stamps[stamps < boundary]
            value = grid.days[index, column]
            if earlier.empty:
                assert np.isnan(value), (package, boundary)
            else:
                expected = (boundary - earlier.max()).total_seconds() / 86400
                assert value == pytest.approx(expected), (package, boundary)


def test_staleness_survives_a_month_with_no_events_at_all():
    """The regression: a month nothing happened in has no column to reindex from.

    Reindexing a grid onto a month range adds the missing column as an all-NaN
    column with no dtype to infer, so it arrives as ``object`` and the numpy
    conversion downstream raises. The author's log has one such month, July 2019.
    """
    rows = [
        ("a", "2019-06-15", "installed"),
        ("a", "2019-09-02", "upgraded"),
    ]
    frame = pd.DataFrame(rows, columns=["package", "stamp", "action"])
    frame["timestamp"] = pd.to_datetime(frame["stamp"], utc=True)
    frame = frame.drop(columns="stamp")

    grid = analyze.package_staleness(frame, pd.Timestamp("2019-10-01", tz="UTC"))
    july = grid.months.searchsorted(pd.Timestamp("2019-07-01", tz="UTC"))
    assert july in range(len(grid.months))
    # The month exists, is finite, and is one month further on than June's.
    assert grid.days[0, july] == pytest.approx(grid.days[0, july - 1] + 30, abs=2)


def test_staleness_row_order_is_a_choice_and_first_seen_is_chronological(frame):
    by_name = analyze.package_staleness(frame, AS_OF)
    by_arrival = analyze.package_staleness(frame, AS_OF, order="first_seen")

    assert by_name.packages == sorted(by_name.packages)
    firsts = frame.groupby("package")["timestamp"].min()
    ordered = [firsts[p] for p in by_arrival.packages]
    assert ordered == sorted(ordered)
    # Same packages either way; only the rows move.
    assert sorted(by_arrival.packages) == sorted(by_name.packages)
    assert by_arrival.days.shape == by_name.days.shape


def test_staleness_first_seen_breaks_ties_on_the_name():
    """Two packages installed in the same instant must have a stable order."""
    moment = pd.Timestamp("2021-01-01", tz="UTC")
    frame = pd.DataFrame(
        {
            "package": ["b", "a", "c"],
            "timestamp": [moment, moment, moment + pd.Timedelta(1, unit="h")],
            "action": ["installed"] * 3,
        }
    )
    assert analyze.package_staleness(frame, AS_OF, order="first_seen").packages == ["a", "b", "c"]


def test_staleness_rejects_an_unknown_order(frame):
    with pytest.raises(ValueError, match="order"):
        analyze.package_staleness(frame, AS_OF, order="by-vibes")


def test_staleness_describe_reports_percentiles():
    grid = analyze.package_staleness(big_frame(), AS_OF)
    described = grid.described()
    assert set(described) == {"median", "p90", "max"}
    assert described["median"] <= described["p90"] <= described["max"]
    assert described["max"] > 0


def big_frame() -> pd.DataFrame:
    """200 packages, so the heatmap's geometry is exercised on more than a handful."""
    rows = []
    start = pd.Timestamp("2016-01-01", tz="UTC")
    for index in range(200):
        name = f"pkg{index:04d}"
        rows.append((name, start, "installed"))
        rows.append((name, start + pd.Timedelta(index, unit="D"), "upgraded"))
    frame = pd.DataFrame(rows, columns=["package", "timestamp", "action"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


# -- the base install -------------------------------------------------------- #


def test_initial_install_packages_is_the_whole_base_system(frame):
    base = analyze.initial_install_packages(frame)
    start, end = analyze.initial_install_window(frame)
    assert not base.empty
    assert (base["start"] >= start).all() and (base["start"] <= end).all()
    assert base["package"].is_unique
    assert list(base.columns) == ["package", "start"]


def test_initial_install_packages_is_not_the_still_installed_subset(frame):
    """474 arrived with the machine and 331 are still on it; the two differ."""
    base = analyze.initial_install_packages(frame)
    whole = analyze.installed_whole_time(frame, AS_OF)
    assert len(base) > len(whole)
    assert set(whole["package"]) < set(base["package"])


def test_initial_install_packages_handles_an_empty_log():
    from paclog.loader import empty_frame

    assert analyze.initial_install_packages(empty_frame()).empty
