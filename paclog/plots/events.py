"""Charts about *when* things happened."""

from __future__ import annotations

import matplotlib.dates as mdates
import pandas as pd

from .. import analyze
from ..model import ACTION_ORDER
from .base import ACTION_COLOR_LIST, color_for, figure, finish
from .registry import Context, chart

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


@chart(
    "events-per-hour",
    "events_per_hour.svg",
    "Package Event Distribution Over The Day",
    "events by local hour, 0-23",
)
def events_per_hour(frame: pd.DataFrame, ctx: Context):
    counts = analyze.events_per_hour(frame)
    fig = figure("events-per-hour")
    ax = fig.add_subplot(111)
    ax.bar(counts.index, counts.values, color="#1565c0", width=0.85)
    ax.set_xticks(range(24))
    ax.set_xlabel("Hour of the day")
    ax.set_ylabel("# Events")
    ax.set_title("Package Event Distribution Over The Day")
    finish(ax)
    return fig


@chart(
    "events-per-weekday",
    "events_per_weekday.svg",
    "Package Events Per Weekday",
    "events by local weekday",
)
def events_per_weekday(frame: pd.DataFrame, ctx: Context):
    counts = analyze.events_per_weekday(frame)
    fig = figure("events-per-weekday")
    ax = fig.add_subplot(111)
    ax.bar(range(7), counts.values, color="#2e7d32", width=0.7)
    ax.set_xticks(range(7))
    ax.set_xticklabels(WEEKDAYS, rotation=30, ha="right")
    ax.set_ylabel("# Events")
    ax.set_title("Package Events Per Weekday")
    finish(ax)
    return fig


@chart(
    "events-per-month",
    "events_per_month.svg",
    "Package Events Per Month",
    "one panel per action, each with its own scale",
)
def events_per_month(frame: pd.DataFrame, ctx: Context):
    """Small multiples: one panel per action, each with an independent y-axis.

    This replaces the original's single stacked bar per month, which cannot be
    read on a real log. Upgrades are 90.42% of events here and downgrades are
    0.07% -- 45 of 68 521 -- so on a shared axis the four minority series
    collapse into a hairline at the base of every bar and ``downgraded`` is flat
    on zero. One panel per action with its own scale is the smallest change that
    makes all five legible at once, and it keeps the monthly bucketing, which is
    the thing the chart is for.
    """
    table = analyze.events_per_month(frame)
    present = [a for a in ACTION_ORDER if a.value in table.columns]
    fig = figure("events-per-month", figsize=(16, 3 * max(len(present), 1)))
    if table.empty:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "no events", ha="center", va="center", transform=ax.transAxes)
        return fig

    axes = fig.subplots(len(present), 1, sharex=True, squeeze=False).ravel()
    index = pd.DatetimeIndex(table.index)
    for ax, action in zip(axes, present):
        values = table[action.value].to_numpy()
        ax.bar(index, values, width=27, color=color_for(action.value), linewidth=0)
        peak = int(values.max()) if len(values) else 0
        ax.set_ylabel(action.value, fontsize=9)
        ax.text(
            0.995,
            0.9,
            f"peak {peak}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#555555",
        )
        finish(ax)

    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for label in axes[-1].get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")
    axes[-1].set_xlabel("Month")
    axes[0].set_title("Package Events Per Month")
    fig.tight_layout()
    return fig


@chart(
    "action-distribution",
    "action_distribution.svg",
    "Distribution of Package Actions",
    "share of events per action",
)
def action_distribution(frame: pd.DataFrame, ctx: Context):
    counts = analyze.action_counts(frame)
    fig = figure("action-distribution")
    ax = fig.add_subplot(111)
    values = counts.values
    if values.sum() == 0:
        ax.text(0.5, 0.5, "no events", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.pie(
            values,
            labels=[f"{name}\n{int(n)}" for name, n in counts.items() if n > 0],
            colors=[c for c, n in zip(ACTION_COLOR_LIST, values) if n > 0],
            autopct="%1.1f%%",
            startangle=140,
            wedgeprops={"edgecolor": "white", "linewidth": 1},
        )
        ax.axis("equal")
    ax.set_title("Distribution of Package Actions")
    return fig
