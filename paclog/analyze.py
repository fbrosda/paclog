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


def _require(frame: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in frame.columns:
        raise KeyError("frame has no 'timestamp' column; did you load it with paclog?")
    return frame


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
    end_default = pd.Timestamp(as_of)
    if end_default.tzinfo is None and not frame.empty:
        end_default = end_default.tz_localize(frame["timestamp"].dt.tz)
    if end_default.tzinfo is None:
        end_default = end_default.tz_localize("UTC")

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


def package_lifetime(frame: pd.DataFrame, as_of: datetime | pd.Timestamp) -> pd.DataFrame:
    """Days each install period lasted, longest first."""
    periods = install_periods(frame, as_of)
    if periods.empty:
        return pd.DataFrame({"package": [], "days": []})
    out = periods.copy()
    out["days"] = (out["end"] - out["start"]).dt.total_seconds() / 86400.0
    return out.sort_values("days", ascending=False).reset_index(drop=True)


def currently_installed(frame: pd.DataFrame, as_of: datetime | pd.Timestamp) -> pd.Series:
    """Packages with an open install period at ``as_of``."""
    periods = install_periods(frame, as_of)
    if periods.empty:
        return pd.Series([], dtype=str, name="package")
    cutoff = pd.Timestamp(as_of)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize(periods["end"].dt.tz)
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
