"""Analytics for pacman logs.

Every function here takes a frame and returns a frame or a series. None of them
import matplotlib, which is what makes the numbers testable without rendering a
chart and reusable by both ``paclog stats`` and the plots.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from .model import ACTION_ORDER, Action

#: One row per (package, install period).
PERIOD_COLUMNS = ("package", "start", "end")

#: A silence of at least this long with no package events ends the initial install
#: window. One day is the shortest gap that reliably separates "provisioning the
#: machine" from "using it": on the author's log the base install is followed by a
#: 28 h pause, while the largest gap *inside* the provisioning burst is 10 h.
INITIAL_WINDOW_GAP_DAYS = 1

#: Events closer together than this belong to one pacman invocation. pacman stamps
#: every ALPM line of a transaction with the same second, but not always: the
#: transactions of 50 rows or more on the author's log span a median of 6 seconds
#: and up to 59 minutes, so grouping on the timestamp second alone cuts the big
#: ones in half. 60 s rejoins those splits and still leaves a wide margin to the
#: 69 minutes between the base system's four transactions, which have to stay
#: apart for :func:`initial_install_window` to mean anything.
SESSION_GAP_SECONDS = 60.0

#: Nanoseconds in a day, as a float. Timestamps are compared as int64 nanoseconds
#: and only then divided, so the arithmetic stays exact until the last step.
NS_PER_DAY = 86_400_000_000_000.0


def _require(frame: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in frame.columns:
        raise KeyError("frame has no 'timestamp' column; did you load it with paclog?")
    return frame


def _as_of(as_of: datetime | pd.Timestamp, tz=None) -> pd.Timestamp:
    """``as_of`` as an aware timestamp, localized into the frame's timezone.

    Every function that closes an open install period needs this, and getting it
    subtly differently in each one is how a naive ``as_of`` ends up comparing
    instants instead of wall clocks.
    """
    stamp = pd.Timestamp(as_of)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC" if tz is None else tz)
    return stamp


def total_events(frame: pd.DataFrame) -> int:
    return int(len(_require(frame)))


def distinct_packages(frame: pd.DataFrame) -> int:
    return int(_require(frame)["package"].nunique())


def time_span(frame: pd.DataFrame) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if _require(frame).empty:
        return None, None
    stamps = frame["timestamp"]
    return stamps.min(), stamps.max()


def action_counts(frame: pd.DataFrame) -> pd.Series:
    """Counts per action, reindexed onto the canonical action order.

    Reindexing is what makes a pie chart of a 4-event log still show the same
    five slices as a 68k-event one, with zeroes instead of missing categories.
    """
    frame = _require(frame)
    counts = frame["action"].value_counts()
    ordered = [a.value for a in ACTION_ORDER]
    return counts.reindex(ordered, fill_value=0).rename("count")


def action_totals(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-action totals plus the share of all events."""
    counts = action_counts(frame)
    total = int(counts.sum())
    share = (counts / total * 100).round(2) if total else counts * 0.0
    return pd.DataFrame({"count": counts, "percent": share})


def events_per_hour(frame: pd.DataFrame) -> pd.Series:
    """Events by local hour of day, 0..23 with empty hours kept as zero.

    The original ``groupby(...).size()`` dropped hours with no events entirely, so
    a machine that is simply asleep 02:00-06:00 rendered as a chart with those
    hours *missing* rather than at zero -- which reads very differently.
    """
    frame = _require(frame)
    if frame.empty:
        return pd.Series(0, index=range(24), name="count", dtype="int64")
    counts = frame.groupby(frame["timestamp"].dt.hour).size()
    return counts.reindex(range(24), fill_value=0).rename("count")


def events_per_weekday(frame: pd.DataFrame) -> pd.Series:
    """Events by local weekday, Monday first."""
    frame = _require(frame)
    if frame.empty:
        return pd.Series(0, index=range(7), name="count", dtype="int64")
    counts = frame.groupby(frame["timestamp"].dt.weekday).size()
    return counts.reindex(range(7), fill_value=0).rename("count")


def events_per_day(frame: pd.DataFrame) -> pd.Series:
    """Events by local day, every day in the span kept as zero.

    The same argument as :func:`events_per_hour`, and it matters more at this
    resolution: on the author's log 1 968 of 3 851 days carry no package event at
    all, so a ``groupby(...).size()`` throws away the majority of the calendar and
    leaves a chart of the days the machine happened to be busy. The quiet days are
    the interesting part -- a trip abroad, a reinstall, a dead period.
    """
    frame = _require(frame)
    first, last = time_span(frame)
    if first is None:
        return pd.Series(
            0, index=pd.DatetimeIndex([], name="timestamp"), name="count", dtype="int64"
        )
    days = pd.date_range(
        first.normalize(), last.normalize(), freq="D", name="timestamp"
    )
    counts = frame.groupby(frame["timestamp"].dt.floor("D")).size()
    return counts.reindex(days, fill_value=0).rename("count")


def active_days(frame: pd.DataFrame) -> pd.Series:
    """The subset of :func:`events_per_day` that has at least one event."""
    daily = events_per_day(frame)
    return daily[daily > 0].rename("count")


def month_start(stamps: pd.Series) -> pd.Series:
    """First local midnight of each stamp's month, keeping the timezone.

    Plain arithmetic rather than ``dt.to_period("M")``: converting to a PeriodIndex
    makes pandas warn that timezone information is dropped, and the local month
    boundary is precisely what the monthly charts are about.
    """
    offset = pd.to_timedelta(stamps.dt.day - 1, unit="D")
    return (stamps - offset).dt.normalize()


def month_key(frame: pd.DataFrame) -> pd.Series:
    """First local midnight of each event's month, keeping the timezone."""
    return month_start(_require(frame)["timestamp"])


def month_range(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Every local month start from ``start``'s month to ``end``'s, inclusive.

    Used to keep empty months in the monthly tables. A ``groupby`` drops them, and
    on a bar chart a missing month is indistinguishable from a month with no
    activity in it.
    """
    first = start.normalize().replace(day=1)
    last = end.normalize().replace(day=1)
    return pd.date_range(first, last, freq="MS", name="month")


def events_per_month(frame: pd.DataFrame, actions: bool = True) -> pd.DataFrame:
    """Events per calendar month, one column per action.

    Indexed by the first instant of each local month, so it plots directly and
    round-trips through JSON without a period parser.
    """
    frame = _require(frame)
    if frame.empty:
        return pd.DataFrame(columns=[a.value for a in ACTION_ORDER])

    period = month_key(frame)
    if not actions:
        return frame.groupby(period).size().rename("count").to_frame()

    table = frame.groupby([period, "action"], observed=True).size().unstack(fill_value=0)
    ordered = [a.value for a in ACTION_ORDER if a.value in table.columns]
    extra = [c for c in table.columns if c not in ordered]
    return table.reindex(columns=ordered + extra)


def top_packages(frame: pd.DataFrame, action: Action | str, n: int = 20) -> pd.Series:
    """The n most-changed packages for one action."""
    frame = _require(frame)
    value = action.value if isinstance(action, Action) else action
    subset = frame[frame["action"] == value]
    return subset["package"].value_counts().head(n).rename("count")


def top_packages_grid(frame: pd.DataFrame, n: int = 20) -> dict[str, pd.Series]:
    """Top packages for every action in the canonical order."""
    return {a.value: top_packages(frame, a, n) for a in ACTION_ORDER}


def transactions(
    frame: pd.DataFrame, gap_seconds: float = SESSION_GAP_SECONDS
) -> pd.DataFrame:
    """Group events into pacman invocations: one row per session.

    Every other chart in the package counts *events*, which means a single
    ``pacman -Syu`` that touches 229 packages outweighs a month of single-package
    work in all of them. This is the unit that fixes that: on the author's log it
    turns 68 521 events into 3 157 sessions with a median of 7 packages each.

    A session ends at the last event and the next begins after a silence longer
    than ``gap_seconds``. The threshold is a parameter and not a constant because
    the log cannot settle it: timestamps carry one-second resolution, so a
    transaction that straddles a second boundary is indistinguishable from two
    transactions. On the author's log a zero-second gap gives 9 807 sessions, the
    60 s default gives 3 157, and 300 s gives 2 720 -- which is why the charts
    that draw this put the number they used in the title.
    """
    frame = _require(frame)
    columns = ["start", "end", "events", "packages"]
    if frame.empty:
        return pd.DataFrame({name: pd.Series(dtype="object") for name in columns})

    ordered = frame.sort_values("timestamp", kind="stable")
    # The gaps are compared as int64 nanoseconds rather than as timedeltas:
    # ``Series.diff() > pd.Timedelta(...)`` goes through numpy's *generic* timedelta
    # unit, which pandas 2.3 and numpy 2.5 deprecate and will turn into an error.
    # Prepending the first stamp makes its own gap zero, so the first event opens
    # session 0 rather than starting a spurious one -- which is what diff()'s NaT
    # was doing here, less legibly.
    stamps = ordered["timestamp"].to_numpy(dtype="datetime64[ns]").astype("int64")
    gaps = np.diff(stamps, prepend=stamps[0])
    session = pd.Series((gaps > gap_seconds * 1e9).cumsum(), index=ordered.index)
    grouped = ordered.groupby(session, observed=True)
    out = pd.DataFrame(
        {
            "start": grouped["timestamp"].min(),
            "end": grouped["timestamp"].max(),
            "events": grouped.size(),
            "packages": grouped["package"].nunique(),
        }
    )
    return out.reset_index(drop=True)


def transactions_per_month(
    frame: pd.DataFrame, gap_seconds: float = SESSION_GAP_SECONDS
) -> pd.DataFrame:
    """Sessions per local month, with the events and packages they account for.

    Months with no sessions are kept as zero rows. A ``groupby`` would drop them,
    and on a bar chart a missing month and a month with no maintenance are the
    same picture -- which is a claim about the machine's habits that the data does
    not support.

    ``events`` and ``packages`` are summed over the sessions in the month, so they
    are totals rather than per-session values: one big sweep in a month should
    read as one month with a lot of work in it, not as a month of busy ones.
    """
    sessions = transactions(frame, gap_seconds)
    columns = ["sessions", "events", "packages"]
    if sessions.empty:
        empty = pd.Series(0, index=pd.DatetimeIndex([], name="month"), dtype="int64")
        return pd.DataFrame({name: empty.copy() for name in columns})

    keyed = sessions.assign(month=month_start(sessions["start"]))
    bounds = month_range(sessions["start"].min(), sessions["end"].max())
    table = keyed.groupby("month", observed=True).agg(
        sessions=("events", "size"), events=("events", "sum"), packages=("packages", "sum")
    )
    return table.reindex(bounds, fill_value=0)


def transaction_size_summary(frame: pd.DataFrame, gap_seconds: float = SESSION_GAP_SECONDS) -> pd.Series:
    """Descriptive statistics of the packages touched per session.

    Returned rather than left to the caller because the median is the number
    worth quoting and recomputing it in three places is how three places end up
    disagreeing.
    """
    sessions = transactions(frame, gap_seconds)
    if sessions.empty:
        return pd.Series(dtype="float64", name="packages")
    return sessions["packages"].describe()


def upgrade_intervals(frame: pd.DataFrame, action: Action = Action.UPGRADED) -> pd.DataFrame:
    """Days between consecutive events of ``action`` for each package.

    Returns a tidy ``(package, interval_days)`` frame. This replaces the original
    ``groupby(...).apply(lambda x: ..., include_groups=False)``, whose behaviour
    depended on the pandas version and whose result was awkward to assert on.
    """
    frame = _require(frame)
    value = action.value if isinstance(action, Action) else action
    subset = frame[frame["action"] == value]
    if subset.empty:
        return pd.DataFrame({"package": pd.Series(dtype=str), "interval_days": pd.Series(dtype=float)})

    subset = subset.sort_values(["package", "timestamp"], kind="stable")
    elapsed = subset.groupby("package", observed=True)["timestamp"].diff()
    days = elapsed.dt.total_seconds() / 86400.0
    out = pd.DataFrame({"package": subset["package"], "interval_days": days})
    return out.dropna().reset_index(drop=True)


def interval_summary(frame: pd.DataFrame, action: Action = Action.UPGRADED) -> pd.DataFrame:
    """Per-package statistics of the intervals from :func:`upgrade_intervals`."""
    intervals = upgrade_intervals(frame, action)
    if intervals.empty:
        return pd.DataFrame(
            columns=["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        )
    return intervals.groupby("package", observed=True)["interval_days"].describe()


def upgrade_interval_distribution(
    frame: pd.DataFrame, action: Action = Action.UPGRADED, bins: int = 40
) -> pd.DataFrame:
    """Every interval between consecutive upgrades of one package, pooled.

    :func:`interval_summary` answers this per package, which on a real log is a
    grid of small multiples nobody can compare across packages. Pooled, the same
    numbers show the shape: on the author's log the median gap is 20 days and the
    mean is 66, and only the pooled view says that the mean is an artefact of a
    long tail rather than the typical wait.

    The bins are geometric because the values span three orders of magnitude --
    a package rebuilt the same afternoon and one left alone for a decade are both
    real observations. Linear bins over that range either drop the short ones or
    smear the long ones into the last bar. Bins below the observed minimum are
    still emitted, so a value of exactly zero days is counted rather than silently
    falling outside the first edge.
    """
    intervals = upgrade_intervals(frame, action)
    columns = ["low_days", "high_days", "count", "share"]
    if intervals.empty:
        return pd.DataFrame({name: pd.Series(dtype="float64") for name in columns})

    days = intervals["interval_days"].to_numpy(dtype="float64")
    positive = days[days > 0]
    if positive.size == 0 or bins < 1:
        return pd.DataFrame({name: pd.Series(dtype="float64") for name in columns})

    low = float(positive.min())
    high = float(days.max())
    if high <= low:  # every interval is the same value
        low, high = low / 10.0, low * 10.0
    edges = np.geomspace(low, high * (1 + 1e-9), bins + 1)
    # np.histogram drops anything outside the edges, and a zero-day interval is
    # below ``low``. Clamping counts it in the first bin instead of losing it.
    counts, _ = np.histogram(np.clip(days, low, high), bins=edges)
    total = int(counts.sum())
    return pd.DataFrame(
        {
            "low_days": edges[:-1],
            "high_days": edges[1:],
            "count": counts,
            "share": counts / total * 100.0 if total else counts * 0.0,
        }
    )


def install_periods(frame: pd.DataFrame, as_of: datetime | pd.Timestamp) -> pd.DataFrame:
    """Derive install periods per package by walking each package's history.

    ``installed`` opens a period, ``removed`` closes one. A package still installed
    at ``as_of`` gets an open-ended period, which is why ``as_of`` is an explicit
    input rather than a hidden ``now()`` call.
    """
    frame = _require(frame)
    end_default = _as_of(as_of, None if frame.empty else frame["timestamp"].dt.tz)

    if frame.empty:
        return pd.DataFrame({"package": [], "start": [], "end": []}).astype(
            {"start": "datetime64[ns, UTC]", "end": "datetime64[ns, UTC]"}
        )

    ordered = frame.sort_values(["package", "timestamp"], kind="stable")
    periods: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    for package, group in ordered.groupby("package", sort=True, observed=True):
        open_since: pd.Timestamp | None = None
        for row in group.itertuples(index=False):
            if row.action == Action.INSTALLED.value:
                open_since = row.timestamp
            elif row.action == Action.REMOVED.value and open_since is not None:
                periods.append((package, open_since, row.timestamp))
                open_since = None
        if open_since is not None:
            periods.append((package, open_since, end_default))

    if not periods:
        return pd.DataFrame({"package": [], "start": [], "end": []}).astype(
            {"start": "datetime64[ns, UTC]", "end": "datetime64[ns, UTC]"}
        )
    return pd.DataFrame(periods, columns=list(PERIOD_COLUMNS))


def initial_install_window(
    frame: pd.DataFrame, days: int = INITIAL_WINDOW_GAP_DAYS
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """The provisioning era: the span from the first event to the last event
    before the log's first silence of at least ``days``.

    Everything installed inside it belongs to the original system. Deriving the
    end from the data rather than hardcoding "the first day" is what makes it
    survive a slow install: on the author's log the base system arrives in four
    transactions spread over 21 hours, and a fixed hour count would cut the
    X.org stack off the list.

    With no such silence anywhere in the log the machine was being touched at
    least twice a day, so no era boundary exists to find and the window falls
    back to the first calendar day -- the conservative reading, since claiming a
    package arrived "with the system" is a stronger claim than the log supports.
    A log with events exactly a day apart has a silence between every pair of
    them, so its window collapses to the single instant of the first event.
    """
    frame = _require(frame)
    if frame.empty:
        return None, None

    stamps = frame["timestamp"].sort_values(kind="stable").reset_index(drop=True)
    start = stamps.iloc[0]
    gaps = stamps.diff().to_numpy()
    # ``diff`` puts each gap at its *later* event, so the window ends one row back:
    # the silence starts after the last event that still belongs to the build.
    quiet = np.flatnonzero(gaps >= pd.Timedelta(days, unit="D"))
    if quiet.size:
        end = stamps.iloc[int(quiet[0]) - 1]
    else:
        end = start.normalize() + pd.Timedelta(days, unit="D")
    return start, end


def package_lifetime(frame: pd.DataFrame, as_of: datetime | pd.Timestamp) -> pd.DataFrame:
    """Days each install period lasted, longest first, flagged for censoring.

    ``censored`` marks a period that had not ended at ``as_of``: the package was
    still installed, so its ``days`` is a lower bound rather than a lifetime. The
    two kinds are not comparable, which is why the flag is carried alongside the
    number instead of being left for each caller to re-derive. See
    :func:`completed_lifetimes` and :func:`installed_whole_time` for the two
    charts that need the kinds apart.
    """
    periods = install_periods(frame, as_of)
    if periods.empty:
        return pd.DataFrame({"package": [], "start": [], "end": [], "days": [], "censored": []})
    out = periods.copy()
    out["days"] = (out["end"] - out["start"]).dt.total_seconds() / 86400.0
    out["censored"] = out["end"] >= _as_of(as_of, out["end"].dt.tz)
    return out.sort_values("days", ascending=False).reset_index(drop=True)


def completed_lifetimes(frame: pd.DataFrame, as_of: datetime | pd.Timestamp) -> pd.DataFrame:
    """Install periods that actually ended, longest first.

    These are the only rows in :func:`package_lifetime` whose ``days`` is a
    lifetime rather than a lower bound. Ranking the two kinds together is what
    made the original lifetime chart useless: every still-installed package scores
    ``as_of - installed``, so the longest bars were whichever packages happened to
    be installed first, all pinned to the length of the log and all identical.
    The author's log had 1 666 censored periods -- 1 666 distinct packages, as no
    package can have two of them open at once -- against 2 278 completed ones, so
    the top 40 was entirely censored: 40 bars of exactly the same length, all of
    them from the 331 packages that have been installed since the machine was
    built. (1 378 is a different and less useful number: packages whose periods
    are *all* censored, the other 288 having been removed and reinstalled at
    least once.)
    """
    lifetimes = package_lifetime(frame, as_of)
    if lifetimes.empty:
        return lifetimes
    return lifetimes[~lifetimes["censored"]].reset_index(drop=True)


def installed_whole_time(frame: pd.DataFrame, as_of: datetime | pd.Timestamp) -> pd.DataFrame:
    """Packages that have been installed for the entire recorded history.

    Two conditions, both required:

    * the install period is still open at ``as_of`` -- the package was never
      removed, so there is no completed period to rank it by;
    * it opened inside the initial install window, so it was part of the system
      as built rather than an arrival later on.

    A package removed and later reinstalled satisfies neither: its open period
    starts at the reinstall, so it is not on this list, which is the whole point
    -- it was not installed the whole time.

    Returned alphabetically, because the chart that draws it is a list.
    """
    lifetimes = package_lifetime(frame, as_of)
    if lifetimes.empty:
        return lifetimes[["package", "start", "days"]]
    start, end = initial_install_window(frame)
    if start is None:  # pragma: no cover - a non-empty frame always has a window
        return lifetimes.iloc[0:0][["package", "start", "days"]]
    inside = lifetimes["start"] >= start
    if end is not None:
        inside &= lifetimes["start"] <= end
    whole = lifetimes[inside & lifetimes["censored"]]
    return whole.sort_values("package").reset_index(drop=True)[["package", "start", "days"]]


def currently_installed(frame: pd.DataFrame, as_of: datetime | pd.Timestamp) -> pd.Series:
    """Packages with an open install period at ``as_of``."""
    periods = install_periods(frame, as_of)
    if periods.empty:
        return pd.Series([], dtype=str, name="package")
    cutoff = _as_of(as_of, periods["end"].dt.tz)
    return periods[periods["end"] >= cutoff]["package"].drop_duplicates().rename("package")


def installed_over_time(
    frame: pd.DataFrame, as_of: datetime | pd.Timestamp, freq: str = "MS"
) -> pd.DataFrame:
    """How many packages are installed at each point on the calendar.

    Indexed by the ``freq`` boundary instants, default the first of each local
    month, and carrying four columns:

    ``installed``
        install periods in force at that instant.
    ``ever_seen``
        packages with any event at or before it. Always at least ``installed``,
        and the gap between the two is the churn.
    ``arrivals`` / ``departures``
        periods that opened or closed inside that bucket.

    Every other chart in the package is *activity* -- how much happened, when.
    This one is *state*: the size of the set the activity happened to. They answer
    different questions and the difference is visible -- a month with 2 000 events
    can leave the installed count untouched.

    A period is in force while ``start <= t <= end``, so a package removed and
    later reinstalled is counted once, not twice, and an open period runs to
    ``as_of``.

    ``as_of`` is appended as a final boundary, so the last row is the state at
    ``as_of`` itself rather than at the start of the current month. That is what
    makes ``installed`` there equal :func:`currently_installed` and ``ever_seen``
    equal :func:`distinct_packages`, which is a cheap way for a caller -- or a test
    -- to know the curve ends where the data does. The price is a final interval
    shorter than the others, which is the honest shape for a curve that has not
    finished its month.
    """
    frame = _require(frame)
    columns = ["installed", "ever_seen", "arrivals", "departures"]
    periods = install_periods(frame, as_of)
    if periods.empty:
        empty = pd.Series(0, index=pd.DatetimeIndex([], name="boundary"), dtype="int64")
        return pd.DataFrame({name: empty.copy() for name in columns})

    cutoff = _as_of(as_of, periods["end"].dt.tz)
    bounds = pd.date_range(
        periods["start"].min().normalize().replace(day=1),
        cutoff.normalize().replace(day=1),
        freq=freq,
        name="boundary",
    )
    if bounds[-1] < cutoff:
        bounds = bounds.append(pd.DatetimeIndex([cutoff], name="boundary"))
    instants = bounds.to_numpy(dtype="datetime64[ns]")

    # ``to_numpy()`` on a tz-aware column hands back an object array of
    # Timestamps, which compares slowly and rejects a datetime64 scalar outright.
    starts = periods["start"].to_numpy(dtype="datetime64[ns]")
    ends = periods["end"].to_numpy(dtype="datetime64[ns]")
    # started and not yet finished: a period ending exactly at t is still in force.
    installed = (starts[None, :] <= instants[:, None]).sum(axis=1) - (
        ends[None, :] < instants[:, None]
    ).sum(axis=1)

    firsts = (
        frame.groupby("package", observed=True)["timestamp"]
        .min()
        .to_numpy(dtype="datetime64[ns]")
    )
    ever_seen = (firsts[None, :] <= instants[:, None]).sum(axis=1)

    def _bucket(values: np.ndarray) -> np.ndarray:
        # -1 means "before the first bucket", which belongs in the first one.
        return np.clip(np.searchsorted(instants, values, side="right") - 1, 0, len(bounds) - 1)

    arrivals = np.bincount(_bucket(starts), minlength=len(bounds))
    # An open period ends *at* as_of by construction; counting that as a departure
    # would put one removal in every run's final bucket.
    closed = ends[ends < cutoff.to_datetime64()]
    departures = np.bincount(_bucket(closed), minlength=len(bounds))

    return pd.DataFrame(
        {
            "installed": installed,
            "ever_seen": ever_seen,
            "arrivals": arrivals,
            "departures": departures,
        },
        index=bounds,
    )


def staleness_now(frame: pd.DataFrame, as_of: datetime | pd.Timestamp) -> pd.Series:
    """Days since each package's last event at ``as_of``, longest first.

    The single-instant companion to :func:`package_staleness`, and the column a
    staleness chart quotes: the median here is the typical amount of neglect, and
    the tail is the packages that have not been touched since the machine was
    built. Packages with no events at all cannot occur, so there is nothing to
    mask.
    """
    frame = _require(frame)
    if frame.empty:
        return pd.Series(dtype="float64", name="days")
    cutoff = _as_of(as_of, frame["timestamp"].dt.tz)
    last = frame.groupby("package", observed=True)["timestamp"].max()
    days = (cutoff - last).dt.total_seconds() / 86400.0
    return days.sort_values(ascending=False).rename("days")


@dataclass(frozen=True, slots=True)
class Staleness:
    """Days since each package's last event, on a package-by-month grid.

    Returned as a small object rather than a frame because it is a matrix with two
    labelled axes, and forcing it into a frame is what turns ``NaN`` into a value
    that looks like an observation.

    ``days`` is a float array of shape ``(len(packages), len(months))``, sorted by
    package name, and ``NaN`` wherever the package did not exist yet.
    """

    packages: list[str]
    months: pd.DatetimeIndex
    days: np.ndarray
    as_of: pd.Timestamp

    @property
    def shape(self) -> tuple[int, int]:
        return self.days.shape

    def described(self) -> dict[str, float]:
        """Summary numbers for the chart to annotate itself with."""
        finite = self.days[~np.isnan(self.days)]
        if finite.size == 0:
            return {"median": float("nan"), "p90": float("nan"), "max": float("nan")}
        return {
            "median": float(np.median(finite)),
            "p90": float(np.percentile(finite, 90)),
            "max": float(finite.max()),
        }


def package_staleness(
    frame: pd.DataFrame, as_of: datetime | pd.Timestamp, order: str = "name"
) -> Staleness:
    """Days since each package's last event, at every month boundary.

    The timeline answers *when was this installed*. This answers *when did anyone
    last touch it*, which is a different question, and the one a maintenance-debt
    chart needs: a package installed in 2016 and upgraded yesterday is on the
    timeline for a decade and stale for none of it.

    Every package gets a row and every month a column. A cell is ``NaN`` until the
    package exists; filling it with zero would claim "last touched this month"
    about a package that had not been installed yet, which is the one thing a
    heatmap must never say. On the author's log 130 126 of 356 616 cells are
    ``NaN`` for that reason.

    Each column is measured at the month's *closing* boundary -- the instant the
    month hands over to the next -- and the last column at ``as_of``. Measuring at
    the month *start* was tried and is wrong: a package touched on the 20th scores
    minus twenty days at the start of that month, so 2 546 cells came out negative
    on the author's log. A closing boundary cannot go negative, because a package's
    last event is always inside the month that boundary closes.

    ``order`` picks the row order, and it is not cosmetic. ``"name"`` matches the
    timeline and is the default; ``"first_seen"`` sorts by the package's first
    event, which is what turns the grid from a list of 2 808 unrelated stripes
    into something with a shape -- the base system arrives as one block at the
    bottom, later arrivals stack above it, and the packages nothing has touched
    since the machine was built show up as long pale runs rather than being
    scattered at random. Ties break on name so the order is stable.

    The forward fill runs over the numpy values rather than through
    ``DataFrame.ffill(axis=1)``, which costs 5.3 s on a 2 808-by-127 tz-aware frame
    against 0.07 s for this function's other steps combined.
    """
    frame = _require(frame)
    if frame.empty:
        empty = pd.DatetimeIndex([], name="month")
        return Staleness(
            packages=[],
            months=empty,
            days=np.empty((0, 0), dtype="float64"),
            as_of=_as_of(as_of, None),
        )

    cutoff = _as_of(as_of, frame["timestamp"].dt.tz)
    stamps = frame["timestamp"]
    months = month_range(stamps.min(), cutoff)
    if order not in ("name", "first_seen"):
        raise ValueError(f"order must be 'name' or 'first_seen', not {order!r}")
    if order == "name":
        packages = sorted(frame["package"].unique())
    else:
        seen = frame.groupby("package", observed=True)["timestamp"].min()
        packages = sorted(seen.index, key=lambda p: (seen[p], p))

    keyed = pd.DataFrame(
        {"package": frame["package"], "timestamp": stamps, "month": month_start(stamps)}
    )
    # Last event per package per month, then carry each row's last known event
    # forward over the months that follow it.
    last = keyed.groupby(["package", "month"], observed=True)["timestamp"].max().unstack("month")
    grid = last.reindex(index=packages, columns=months)
    # A month with no events at all is absent from ``last``, and reindexing adds it
    # back as an all-NaN column with no dtype to infer: it comes in as ``object`` and
    # the numpy conversion below then refuses to touch it. The author's log has
    # exactly one such month, July 2019, which the machine spent switched off.
    grid = grid.astype(last.dtypes.iloc[0])
    values = grid.to_numpy(dtype="datetime64[ns]")
    carry = np.where(~np.isnat(values), np.arange(values.shape[1]), -1)
    np.maximum.accumulate(carry, axis=1, out=carry)
    last_seen = np.take_along_axis(values, np.maximum(carry, 0), axis=1)

    boundary = pd.DatetimeIndex(months + pd.offsets.MonthBegin(1))
    # The final month is not over yet, so it is measured at as_of rather than at a
    # boundary that lies in the future. That also makes the last column identical
    # to staleness_now(), which is a useful thing for a test to be able to assert.
    boundary = boundary.where(boundary <= cutoff, cutoff)
    closes = boundary.to_numpy(dtype="datetime64[ns]").astype("int64")

    days = (closes[None, :] - last_seen.astype("int64")) / NS_PER_DAY
    days[np.isnat(last_seen) | (carry < 0)] = np.nan
    return Staleness(packages=packages, months=months, days=days, as_of=cutoff)


def initial_install_packages(frame: pd.DataFrame) -> pd.DataFrame:
    """Every package whose install period opened inside the initial install window.

    This is the whole of the base system, as opposed to
    :func:`installed_whole_time`, which is only the part of it that has never been
    removed: on the author's log 549 packages arrived with the machine and 331 of
    them are still on it. Both numbers are wanted and they are not
    interchangeable, so neither is derived from the other.

    Install periods rather than first events, because a package that was installed
    with the system, removed, and reinstalled later counts once here -- it did
    arrive with the system.
    """
    frame = _require(frame)
    columns = ["package", "start"]
    if frame.empty:
        return pd.DataFrame({name: pd.Series(dtype="object") for name in columns})
    start, end = initial_install_window(frame)
    periods = install_periods(frame, _as_of(end, frame["timestamp"].dt.tz))
    inside = periods[(periods["start"] >= start) & (periods["start"] <= end)]
    return inside.sort_values(["start", "package"]).reset_index(drop=True)[columns]


def long_gaps(frame: pd.DataFrame, days: int = 180) -> pd.DataFrame:
    """Gaps of at least ``days`` with no package events at all.

    Cheap to compute and surprisingly informative: it lines up with reinstalls,
    trips abroad, and dead periods in a machine's life.
    """
    frame = _require(frame)
    if frame.empty:
        return pd.DataFrame({"start": [], "end": [], "days": []})
    stamps = frame["timestamp"].sort_values().reset_index(drop=True)
    gaps = stamps.diff()
    out = pd.DataFrame({"start": stamps.shift(1), "end": stamps, "days": gaps.dt.total_seconds() / 86400.0})
    return out[out["days"] >= days].reset_index(drop=True)


def summary(frame: pd.DataFrame, as_of: datetime | pd.Timestamp | None = None) -> dict[str, object]:
    """Headline numbers, as printed by ``paclog stats``."""
    frame = _require(frame)
    first, last = time_span(frame)
    out: dict[str, object] = {
        "events": total_events(frame),
        "packages": distinct_packages(frame),
        "first_event": first.isoformat() if first is not None else None,
        "last_event": last.isoformat() if last is not None else None,
        "actions": {str(k): int(v) for k, v in action_counts(frame).items()},
    }
    intervals = upgrade_intervals(frame)
    if not intervals.empty:
        out["upgrade_interval_days"] = {
            "median": round(float(intervals["interval_days"].median()), 2),
            "mean": round(float(intervals["interval_days"].mean()), 2),
        }
    if as_of is not None:
        out["currently_installed"] = int(len(currently_installed(frame, as_of)))
    return out


def busiest_hour(frame: pd.DataFrame) -> int | None:
    hourly = events_per_hour(frame)
    if hourly.empty or hourly.sum() == 0:
        return None
    return int(hourly.idxmax())


def month_index(frame: pd.DataFrame) -> pd.Index:  # pragma: no cover - plotting helper
    return pd.DatetimeIndex(month_key(frame).unique()).sort_values()


def as_datetime_array(values: pd.Series) -> np.ndarray:  # pragma: no cover - plotting helper
    return values.to_numpy(dtype="datetime64[ns]")
