"""Charts about how long packages stay installed.

Two charts, because "how long" has two answers and pooling them is meaningless. A
package that is still installed has been installed for *at least* as long as the
log covers, and that lower bound is nearly the same number for every long-lived
package on a machine. Ranking by it therefore produces a wall of identical bars
that says only "these were installed first", which is what the original lifetime
chart did: 40 bars of exactly the same length, all censored at the as-of date.

So the censored periods get listed -- they are a set, and a set is a list -- and
the completed periods get ranked, because those have lifetimes that actually
vary. See :func:`analyze.installed_whole_time` and
:func:`analyze.completed_lifetimes`.
"""

from __future__ import annotations

import math

import pandas as pd

from .. import analyze
from ..model import Action
from .base import color_for, figure
from .registry import Context, chart

#: How many completed periods the ranking chart draws. This is a summary view and
#: says so in its title; the timeline is where every package is drawn.
TOP_PERIODS = 40

#: The whole-time chart is a list, so its layout is a page of names. The canvas
#: grows with the count and the names are never dropped: on the author's log 331
#: of 2 808 packages qualify, and every one of them is on the chart.
MIN_COLUMNS = 4
MAX_COLUMNS = 12
TARGET_ROWS = 45
#: 24 characters is the longest name on the author's log
#: (``ca-certificates-mozilla``), and monospace 7pt is 0.065in per character, so
#: anything under ~1.7in per column runs the longest names into their neighbour.
COLUMN_WIDTH_IN = 1.75
ROW_HEIGHT_IN = 0.16
TITLE_IN = 1.1
NAME_FONTSIZE = 7


def column_count(total: int) -> int:
    """How many columns of names to lay out ``total`` of them in.

    Wider rather than taller once the list outgrows :data:`TARGET_ROWS`, up to a
    point: past :data:`MAX_COLUMNS` the page stops being a page.
    """
    if total <= 0:
        return 1
    return min(MAX_COLUMNS, max(MIN_COLUMNS, math.ceil(total / TARGET_ROWS)))


@chart(
    "installed-whole-time",
    "installed_whole_time.svg",
    "Installed The Whole Time",
    "packages present since the first install, never removed",
)
def installed_whole_time(frame: pd.DataFrame, ctx: Context):
    """Every package that has been installed for the whole logged history, as a list.

    Drawn as names rather than bars on purpose. These periods are all censored at
    the same as-of instant and all started within the initial install window, so
    their lengths are nearly equal by construction: a bar chart of them is one
    value repeated 331 times, and the names are the entire content.
    """
    whole = analyze.installed_whole_time(frame, ctx.as_of)
    columns = column_count(len(whole))
    rows = max(1, math.ceil(len(whole) / columns)) if len(whole) else 1

    fig = figure(
        "installed-whole-time",
        figsize=(columns * COLUMN_WIDTH_IN, rows * ROW_HEIGHT_IN + TITLE_IN),
    )
    ax = fig.add_subplot(111)
    # One data unit is one inch, which is what makes the column and row pitches
    # above mean anything. The axes has to span the whole figure for that: the
    # default subplot box leaves a 2in margin, and 250-odd of the names on the
    # author's log fall outside the canvas without this.
    ax.set_position([0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    ax.set_xlim(0, columns * COLUMN_WIDTH_IN)
    ax.set_ylim(-rows * ROW_HEIGHT_IN, ROW_HEIGHT_IN)

    if whole.empty:
        ax.text(
            columns * COLUMN_WIDTH_IN / 2,
            0,
            "no package was installed the whole time",
            ha="center",
            va="center",
        )
        ax.set_title("Installed the whole time (none)")
        return fig

    # Column-major, so reading down a column and then across stays alphabetical.
    for position, name in enumerate(whole["package"]):
        column, row = divmod(position, rows)
        ax.text(
            column * COLUMN_WIDTH_IN,
            -row * ROW_HEIGHT_IN,
            name,
            fontsize=NAME_FONTSIZE,
            family="monospace",
            va="top",
            ha="left",
        )

    ax.set_title(
        f"Installed the whole time ({len(whole)} of {analyze.distinct_packages(frame)} packages)",
        pad=14,
    )
    first, last = whole["start"].min(), whole["start"].max()
    ax.text(
        0,
        ROW_HEIGHT_IN,
        f"arrived between {first:%Y-%m-%d} and {last:%Y-%m-%d}, and have never been removed",
        fontsize=8,
        color="#555555",
        va="top",
        ha="left",
    )
    return fig


@chart(
    "package-lifetime",
    "package_lifetime.svg",
    "Longest Completed Install Periods",
    "days per install period that ended, longest first",
)
def package_lifetime(frame: pd.DataFrame, ctx: Context):
    """The longest install periods that actually ended, one bar per package.

    Restricted to :func:`analyze.completed_lifetimes`, which is what makes the
    bars differ from each other. The still-installed packages are not a shorter
    version of this ranking, they are a different kind of observation, and they
    are on the ``installed-whole-time`` chart instead.
    """
    lifetimes = analyze.completed_lifetimes(frame, ctx.as_of)
    fig = figure("package-lifetime")
    ax = fig.add_subplot(111)
    if lifetimes.empty:
        ax.text(
            0.5,
            0.5,
            "no completed install periods",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Longest completed install periods")
        return fig

    top = lifetimes.head(TOP_PERIODS).iloc[::-1]
    positions = range(len(top))
    # These periods ended in a removal, so they take the removal colour rather
    # than the "upgraded" blue the original used, which described nothing.
    ax.barh(positions, top["days"], color=color_for(Action.REMOVED.value))
    ax.set_yticks(list(positions))
    ax.set_yticklabels(top["package"])
    ax.set_ylim(len(top) - 0.5, -0.5)  # longest at the top, as the eye expects

    longest = float(top["days"].max())
    ax.set_xlim(0, longest * 1.14)
    for position, days in zip(positions, top["days"]):
        ax.text(float(days) + longest * 0.015, position, f"{days:,.0f}", va="center", fontsize=7)

    ax.set_xlabel("Days installed, up to removal")
    ax.set_ylabel("Package")
    ax.set_title(f"Longest completed install periods (top {len(top)} of {len(lifetimes)})")
    ax.xaxis.grid(True, alpha=0.25, linewidth=0.8)
    ax.yaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig
