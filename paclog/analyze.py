"""Analytics for pacman logs.

Every function here takes a frame and returns a frame or a series. None of them
import matplotlib, which is what makes the numbers testable without rendering a
chart and reusable by both ``paclog stats`` and the plots.
"""

from __future__ import annotations

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


def month_key(frame: pd.DataFrame) -> pd.Series:
    """First local midnight of each event's month, keeping the timezone.

    Done with plain arithmetic rather than ``dt.to_period("M")``: converting to a
    PeriodIndex makes pandas warn that timezone information is dropped, and the
    local month boundary is precisely what the monthly chart is about.
    """
    stamps = _require(frame)["timestamp"]
    offset = pd.to_timedelta(stamps.dt.day - 1, unit="D")
    return (stamps - offset).dt.normalize()


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
    The author's log had 1 378 such packages against 2 278 completed periods, so
    the top 40 was entirely censored -- 40 bars of exactly the same length.
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
