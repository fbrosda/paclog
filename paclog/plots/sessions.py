"""Charts about *when the machine was maintained*, not how much was installed.

Every other chart counts events, which is the wrong unit for that question. One
``pacman -Syu`` that touches 229 packages outweighs a month of single-package
work in all of them, so a machine that upgrades everything in one weekly sweep and
one that does the same work in a dozen small steps look identical. The unit that
separates them is the invocation, and :func:`analyze.transactions` recovers it from
the log's one-second resolution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import analyze
from .base import figure, finish
from .registry import Context, chart

#: One bar per month on the top panel, so a decade is 120-odd of them. Wider than
#: that and the bars stop being countable by eye.
MONTH_TICK_EVERY = 12


def _month_ticks(ax, index: pd.DatetimeIndex) -> None:
    """Label year boundaries, or every month when there are only a few."""
    import matplotlib.dates as mdates

    if len(index) <= 14:
        locator, fmt = mdates.MonthLocator(), "%Y-%m"
    else:
        locator, fmt = mdates.YearLocator(), "%Y"
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")


@chart(
    "transactions",
    "transactions.svg",
    "Maintenance Sessions",
    "pacman invocations per month, and how many packages each touched",
)
def transactions(frame: pd.DataFrame, ctx: Context):
    """Two panels: when maintenance happened, and how much of it there was.

    The gap threshold is stated in the subtitle rather than buried, because it is a
    judgement the log cannot make on its own: the same data is 9 807 sessions if
    you split on the timestamp second and 3 157 at a minute. A reader comparing
    this chart against another machine's has to know which of those they are
    looking at.

    Drawn as two panels rather than three on purpose. A third panel of one tick per
    session would be 3 157 marks, and the two panels below already say everything
    it could: when sessions cluster, and how large they get.
    """
    gap = analyze.SESSION_GAP_SECONDS
    sessions = analyze.transactions(frame, gap)
    fig = figure("transactions")
    top, bottom = fig.subplots(2, 1, height_ratios=(2, 3))

    if sessions.empty:
        for ax in (top, bottom):
            ax.text(0.5, 0.5, "no events", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        fig.suptitle("Maintenance sessions (no events)")
        return fig

    # -- when: sessions per month ----------------------------------------- #
    per_month = analyze.transactions_per_month(frame, gap)
    index = pd.DatetimeIndex(per_month.index)
    top.bar(index, per_month["sessions"], width=27, color="#1565c0", linewidth=0)
    peak = int(per_month["sessions"].max())
    top.set_ylabel("Sessions", fontsize=9)
    top.text(
        0.995,
        0.9,
        f"busiest month {peak:,}",
        transform=top.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#555555",
    )
    _month_ticks(top, index)
    finish(top)

    # -- how much: packages per session, log y ----------------------------- #
    sizes = sessions["packages"]
    counts = sizes.value_counts().sort_index()
    bottom.bar(
        counts.index,
        counts.values,
        width=np.maximum(counts.index.to_numpy() * 0.12, 0.6),
        color="#2e7d32",
        linewidth=0,
    )
    bottom.set_xscale("log")
    bottom.set_yscale("log")
    bottom.set_xlabel("Packages touched in one session")
    bottom.set_ylabel("Sessions", fontsize=9)
    for label, value in (
        ("median", sizes.median()),
        ("mean", sizes.mean()),
        ("max", sizes.max()),
    ):
        bottom.axvline(value, color="#c62828", linewidth=0.8, linestyle="--", alpha=0.7)
        bottom.text(
            value,
            bottom.get_ylim()[1],
            f" {label} {value:,.0f}",
            fontsize=7,
            color="#c62828",
            va="top",
            ha="left",
        )
    finish(bottom)

    total = len(sessions)
    top.set_title(
        f"Maintenance sessions ({total:,} invocations, {len(frame):,} events, "
        f"a session is a gap of over {gap:.0f} s)"
    )
    fig.tight_layout()
    return fig
