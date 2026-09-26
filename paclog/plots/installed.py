"""The size of the installed set over time.

Every other chart here measures *activity*: how many events, when, to which
packages. This one measures *state* -- how many packages were on the machine at a
given moment -- and the two genuinely come apart. A month with 2 000 events can
leave the installed count untouched, which is what a full-system upgrade looks
like from here.
"""

from __future__ import annotations

import matplotlib.dates as mdates
import pandas as pd

from .. import analyze
from .base import figure, finish
from .registry import Context, chart


@chart(
    "installed-set-size",
    "installed_set_size.svg",
    "Installed Set Size Over Time",
    "packages installed and ever seen, per month",
)
def installed_set_size(frame: pd.DataFrame, ctx: Context):
    """The installed set and the ever-seen set, as two step curves.

    Step, not line: the count changes at discrete events, and a line drawn between
    two month boundaries invents a smooth growth that never happened.

    The second series is everything ever installed, so the gap between the curves is
    the churn -- packages that arrived and were later removed. The final point is
    ``as_of`` rather than the start of the current month, so the curve ends where
    the data does and the last value is the same number ``paclog stats`` reports as
    currently held.

    The base install is marked rather than left to be inferred. It arrives in a
    single step at the far left of the chart and is the largest single arrival in
    the log by an order of magnitude, so a reader who missed it would draw the
    wrong conclusion about what the rest of the curve is. The count comes from
    :func:`analyze.initial_install_packages` -- 474 packages on the author's log --
    and not from the first row of the table, which is the boundary *before* the
    first install and therefore reads zero.

    The legend is anchored bottom right and the annotation at mid height on the
    left, because the upper left is where they both wanted to be and they
    overlapped: a legend is a block, and text placed relative to it is a guess.
    """
    table = analyze.installed_over_time(frame, ctx.as_of)
    fig = figure("installed-set-size")
    ax = fig.add_subplot(111)

    if table.empty:
        ax.text(0.5, 0.5, "no install periods", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Installed set size (no install periods)")
        return fig

    index = pd.DatetimeIndex(table.index)
    total = int(table["ever_seen"].iloc[-1])
    final = int(table["installed"].iloc[-1])
    ax.set_xlim(index[0], index[-1])
    ax.set_ylim(0, total * 1.08)

    ax.fill_between(index, 0, table["installed"], step="post", color="#2e7d32", alpha=0.18, linewidth=0)
    ax.step(index, table["installed"], where="post", color="#2e7d32", label="installed now")
    ax.step(index, table["ever_seen"], where="post", color="#1565c0", linestyle="--", label="ever installed")

    base = analyze.initial_install_packages(frame)
    start, end = analyze.initial_install_window(frame)
    if len(base) and end is not None:
        ax.axvline(end, color="#c62828", linewidth=0.8, linestyle=":", alpha=0.8)
        # The label is positioned in axes fractions rather than with
        # ``get_yaxis_transform()``. A blended transform -- x in data coordinates,
        # y in axes fractions -- makes the two axes' scales differ by five orders of
        # magnitude on this chart, and freetype then computes a glyph size out of
        # that and dies with "raster overflow" when the figure is drawn. The x axis
        # is linear time, so the conversion is one subtraction -- and it has to go
        # through ``date2num``, because ``get_xlim`` on a date axis answers in days
        # since the epoch, not in timestamps. The fraction is clamped because a base
        # install ending at the very end of the log would put the text off canvas.
        left, right = ax.get_xlim()
        position = min(max((mdates.date2num(end) - left) / (right - left), 0.01), 0.55)
        ax.text(
            position,
            0.5,
            f"base install ends here: {len(base):,} of the {total:,} packages\n"
            f"ever installed arrived with the machine",
            transform=ax.transAxes,
            fontsize=8,
            color="#c62828",
            va="center",
            ha="left",
        )

    ax.annotate(
        f"{final:,} installed",
        xy=(index[-1], final),
        xytext=(-6, 10),
        textcoords="offset points",
        fontsize=9,
        color="#2e7d32",
        ha="right",
    )
    ax.set_ylabel("Packages")
    ax.set_xlabel("Month")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title(
        f"Installed set size ({final:,} of {total:,} packages ever installed are on "
        f"the machine at as-of)"
    )
    finish(ax)
    return fig
