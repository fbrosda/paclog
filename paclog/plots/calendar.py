"""The activity calendar.

A calendar is a chart about *days*, and the monthly chart cannot be one: it averages
30 days into a single bar, which turns the 1 968 days on the author's log that
carry no package event at all into a slightly low number. Those days are the
content -- a trip abroad, a reinstall, a machine switched off for a month -- and a
chart that cannot show them cannot show a machine's rhythm.

The encoding is GitHub's contribution graph on purpose. A single hue that gets
darker, over buckets rather than a continuous ramp, because "how busy was this
Tuesday" is a question with a small number of useful answers and a rainbow
colormap answers none of them legibly.

The cell size is in points and the figure is sized from it, which is the only way
to get weeks that touch. A marker is fixed-size in points while a week is a fraction
of the axes, so the two only agree if the axes is made to fit the marker: at seven
points of square on a sixteen-inch canvas each week column came out twenty points
wide and the grid read as floating confetti rather than as a calendar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap

from .. import analyze
from .base import figure
from .registry import Context, chart

#: Side of one day cell, in points. Fixed, and every other dimension follows from
#: it, so the days are flush vertically and the weeks are flush horizontally.
CELL_POINTS = 11.0

#: GitHub's palette, in order: the colour of a day with no contributions, then the
#: five shades of green for increasing activity. Taking the empty-day grey as the
#: bottom of the scale is what lets a dead day be *drawn* rather than absent.
EMPTY_COLOR = "#ebedf0"
ACTIVITY_COLORS = ["#9be9a8", "#40c463", "#30a14e", "#216e39", "#0e4429"]

#: Bucket edges in events per day. Zero is its own bucket so that it lands on
#: :data:`EMPTY_COLOR` rather than on the palest green -- "nothing happened" and "a
#: little happened" must not look alike. Seven edges make six blocks, which is one
#: per colour in the palette above.
BUCKET_EDGES = [0, 1, 3, 6, 11, 21, 51]
BUCKET_LABELS = ["none", "1-2", "3-5", "6-10", "11-20", "21-50", "50+"]

#: Months named along the bottom, and only these four: a year's week columns are
#: about as wide as a five-letter word, so twelve labels would touch.
NAMED_MONTHS = (1, 4, 7, 10)

#: Fixed margins in inches, rather than figure fractions, because the cells are
#: sized in inches and a fraction of a figure whose size comes from the cells is
#: not a length anyone can reason about. ``LEFT_IN`` is the room the weekday names
#: need; the rest are the bands above, below and between the panels.
LEFT_IN = 0.28
MONTH_IN = 0.18
PANEL_TITLE_IN = 0.22
PANEL_GAP_IN = 0.06
#: The caption's band. It has to clear the first panel's year label, which sits
#: directly above the first panel: both are in this band, and a caption that ends
#: where the year label starts is the same collision one inch lower down.
HEAD_IN = 0.80
FOOT_IN = 0.12
RIGHT_IN = 0.10
CAPTION_TOP_IN = 0.14


def _scale():
    """The colormap and norm, shared by the cells and the count of levels.

    No key is drawn. The scale is a convention borrowed from a page people already
    read every day, and a legend in a chart nobody asked for a legend in is noise
    covering the data -- the caption carries the busiest day instead, which is the
    number a reader actually wants.
    """
    colors = [EMPTY_COLOR, *ACTIVITY_COLORS]
    norm = BoundaryNorm(BUCKET_EDGES, ncolors=len(colors), clip=True)
    return ListedColormap(colors), norm


def _monday_on_or_before(moment: pd.Timestamp) -> pd.Timestamp:
    # The explicit ``unit=`` form: ``pd.Timedelta(days=n)`` builds a timedelta with
    # numpy's *generic* unit, which pandas 2.3 and numpy 2.5 deprecate.
    return moment - pd.Timedelta(int(moment.weekday()), unit="D")


@chart(
    "activity-calendar",
    "activity_calendar.svg",
    "Package Activity Calendar",
    "one row per year, one cell per day",
)
def activity_calendar(frame: pd.DataFrame, ctx: Context):
    """One panel per year: weekday down, week across, one square per day.

    Every day in the span is drawn, including the empty ones. A day with no events
    is a fact about the machine, and it is drawn in the flat grey at the bottom of
    the scale -- the same treatment a contribution graph gives a blank square. It
    is what makes the dead spells visible as *shape*: a fortnight abroad reads as a
    grey gap in the middle of a row, which a bar chart cannot show at all.

    Columns are real calendar weeks, each starting on a Monday, and the cells touch.
    Both properties come from the same place: the cell is a fixed number of points,
    the week is made exactly as wide, and the panel is exactly seven cells tall.
    """
    daily = analyze.events_per_day(frame)
    fig = figure("activity-calendar")
    if daily.empty:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "no events", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        ax.set_title("Package activity calendar (no events)")
        return fig

    days = pd.DatetimeIndex(daily.index)
    values = daily.to_numpy()
    years = sorted({day.year for day in days})
    cmap, norm = _scale()

    # Each panel is as many week columns as the weeks its year actually touches,
    # which is 53 or 54 -- so the axes is sized per panel rather than shared, and
    # the cell size is what stays constant.
    bounds = {year: _monday_on_or_before(pd.Timestamp(year, 1, 1, tz=days.tz)) for year in years}
    spans = {
        year: int((_monday_on_or_before(pd.Timestamp(year, 12, 31, tz=days.tz)) - bounds[year]).days // 7) + 1
        for year in years
    }
    panel_in = 7 * CELL_POINTS / 72
    # One block is a panel's title band plus its seven rows of cells. Every
    # dimension below is a sum of these and of the fixed margins, so the panels
    # cannot drift off the canvas: an earlier version added the panel height to the
    # bottom offset *and* used it as the height, which put the first panel's top
    # edge 0.17 in above the figure and drove the caption into the 2016 label.
    block_in = PANEL_TITLE_IN + panel_in
    total_w = LEFT_IN + max(spans.values()) * CELL_POINTS / 72 + RIGHT_IN
    total_h = (
        HEAD_IN
        + len(years) * block_in
        + (len(years) - 1) * PANEL_GAP_IN
        + MONTH_IN
        + FOOT_IN
    )
    fig.set_size_inches(total_w, total_h)

    peak = int(values.max())
    for panel, year in enumerate(years):
        origin = bounds[year]
        bottom = (
            FOOT_IN
            + MONTH_IN
            + (len(years) - 1 - panel) * (block_in + PANEL_GAP_IN)
            + PANEL_TITLE_IN
        )
        grid_in = spans[year] * CELL_POINTS / 72
        ax = fig.add_axes(
            [
                LEFT_IN / total_w,
                bottom / total_h,
                grid_in / total_w,
                panel_in / total_h,
            ]
        )

        in_year = np.array([day.year == year for day in days])
        ordinal = np.array([(day - origin).days for day in days[in_year]])
        ax.scatter(
            ordinal // 7,
            ordinal % 7,  # a Monday-based week puts Monday at the top
            s=CELL_POINTS**2,
            c=values[in_year],
            cmap=cmap,
            norm=norm,
            marker="s",
            linewidths=0,
        )
        ax.set_xlim(-0.5, spans[year] - 0.5)
        ax.set_ylim(6.5, -0.5)
        ax.set_yticks([0, 2, 4, 6])
        ax.set_yticklabels(["Mon", "Wed", "Fri", "Sun"], fontsize=7)
        ax.set_xticks([(pd.Timestamp(year, month, 1, tz=days.tz) - origin).days // 7 for month in NAMED_MONTHS])
        ax.set_xticklabels([f"{pd.Timestamp(year, month, 1, tz=days.tz):%b}" for month in NAMED_MONTHS], fontsize=7)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0, pad=1)
        active = int((values[in_year] > 0).sum())
        ax.set_title(
            f"{year}: {active} active day{'s' if active != 1 else ''} of {len(ordinal)}",
            fontsize=8,
            loc="left",
            pad=3,
        )

    fig.suptitle(
        f"Package activity per day ({int((values > 0).sum()):,} active of {len(values):,} days, "
        f"busiest day {peak:,} events; darker means busier, flat grey means none)",
        va="top",
        fontsize=11,
        ha="left",
        x=LEFT_IN / total_w,
        y=1 - CAPTION_TOP_IN / total_h,
    )
    return fig
