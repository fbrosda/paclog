"""Charts about *when* things happened."""

from __future__ import annotations

import matplotlib.dates as mdates
import pandas as pd

from .. import analyze
from .base import ACTION_COLOR_LIST, colors_for, figure, finish
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
            width=25,
            color=colors_for(table.columns),
            legend=True,
        )
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.legend(title="", ncols=len(table.columns), loc="upper left", fontsize=9)
    ax.set_xlabel("Month")
    ax.set_ylabel("# Events")
    ax.set_title("Package Events Per Month")
    finish(ax)
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
