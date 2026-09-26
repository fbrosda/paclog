"""Charts about *when* things happened."""

from __future__ import annotations

import matplotlib.dates as mdates
import pandas as pd

from .. import analyze
from ..model import ACTION_ORDER
from .base import ACTION_COLOR_LIST, color_for, colors_for, figure, finish
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
    "stacked monthly event counts by action",
)
def events_per_month(frame: pd.DataFrame, ctx: Context):
    """Monthly event counts, stacked by action.

    This keeps the original chart's encoding: one stacked bar per month, split by
    action, so the mix is readable inside each month. The size is the original
    32x18 rather than the notebook's 16x9, because 126 months of five series
    stacked is a dense chart and at 16 inches the bars are too narrow to read.

    One real fix over the original: the legend sat at ``loc="upper left"``, which
    is exactly where the March-2016 first-install spike is -- 551 installs, the
    tallest thing in the left third of the chart. It is now placed above the axes
    so it cannot cover data.
    """
    table = analyze.events_per_month(frame)
    fig = figure("events-per-month")
    ax = fig.add_subplot(111)
    if table.empty:
        ax.text(0.5, 0.5, "no events", ha="center", va="center", transform=ax.transAxes)
    else:
        plotted = table.copy()
        plotted.index = pd.DatetimeIndex(plotted.index)
        plotted.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            width=27,
            color=colors_for(table.columns),
            legend=True,
        )
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")
        # Above the axes: the original's upper-left legend covered the first
        # install spike.
        ax.legend(
            title="",
            ncols=len(table.columns),
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            fontsize=10,
        )
    ax.set_xlabel("Month")
    ax.set_ylabel("# Events")
    ax.set_title("Package Events Per Month")
    finish(ax)
    return fig


@chart(
    "events-per-month-by-action",
    "events_per_month_by_action.svg",
    "Package Events Per Month, By Action",
    "one panel per action, each with its own scale",
)
def events_per_month_by_action(frame: pd.DataFrame, ctx: Context):
    """Small multiples: one panel per action, each with an independent y-axis.

    Sharing a single y-axis is what makes a stacked bar unreadable here, and
    sharing one panel per action is what makes `downgraded` (45 events out of
    68 521) visible at all.
    """
    table = analyze.events_per_month(frame)
    present = [a for a in ACTION_ORDER if a.value in table.columns]
    fig = figure("events-per-month-by-action", figsize=(16, 3 * max(len(present), 1)))
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
    axes[0].set_title("Package Events Per Month, By Action")
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
