"""The install-period timeline.

The original drew one ``ax.barh`` artist per install period, which is where the
4.2 MB / 92 750-line SVG came from. One ``broken_barh`` per package does the same
job with roughly one artist per package, and the figure height is capped so the
result is a chart rather than a scroll.
"""

from __future__ import annotations

import zlib

import matplotlib
import matplotlib.dates as mdates
import pandas as pd

from .. import analyze
from ..config import INCHES_PER_PACKAGE, MAX_TIMELINE_HEIGHT_IN
from .base import figure
from .registry import Context, chart

MIN_HEIGHT_IN = 6.0
ROW_HEIGHT = 0.6


def stable_unit(name: str) -> float:
    """A deterministic 0..1 value for a package name.

    ``hash()`` is randomized per process for ``str``, so it would reintroduce
    exactly the nondeterminism this replaced. ``zlib.crc32`` is stable forever.
    """
    return (zlib.crc32(name.encode("utf-8")) % 10_000) / 10_000.0


@chart(
    "timeline",
    "timeline.svg",
    "Package Installation Periods",
    "when each package was installed (excluded from `paclog plot` by default)",
    default=False,
)
def timeline(frame: pd.DataFrame, ctx: Context):
    periods = analyze.install_periods(frame, ctx.as_of)
    fig = figure("timeline", figsize=(16, MIN_HEIGHT_IN))
    ax = fig.add_subplot(111)

    if periods.empty:
        ax.text(0.5, 0.5, "no install periods", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Package Installation Periods")
        return fig

    total = periods["package"].nunique()
    top_n = int(ctx.options.get("top_n") or 0)
    if top_n and total > top_n:
        # Opt-in summary view. Rank by how much history the package actually has,
        # so the cap keeps the packages worth looking at rather than an arbitrary
        # alphabetical slice.
        weight = periods.assign(days=(periods["end"] - periods["start"]).dt.total_seconds())
        ranked = (
            weight.groupby("package", observed=True)
            .agg(spans=("days", "size"), days=("days", "sum"))
            .sort_values(["spans", "days"], ascending=False)
            .index[:top_n]
        )
        periods = periods[periods["package"].isin(ranked)]
        note = f"showing {top_n} of {total} packages, by install history"
    else:
        note = f"all {total} packages"

    packages = sorted(periods["package"].unique())
    y_of = {name: i for i, name in enumerate(packages)}
    row = ROW_HEIGHT / 2

    cmap = matplotlib.colormaps["viridis"]
    rotation = float(ctx.seed or 0) * 0.01
    colors = {name: cmap((stable_unit(name) + rotation) % 1.0) for name in packages}

    periods = periods.reset_index(drop=True)
    starts = mdates.date2num(periods["start"].to_numpy())
    ends = mdates.date2num(periods["end"].to_numpy())
    # groupby.indices gives *positional* rows, which is what lines up with the
    # numpy arrays above. group.index would not, after the top-n filter.
    for name, rows in periods.groupby("package", sort=True, observed=True).indices.items():
        spans = [(starts[i], max(ends[i] - starts[i], 1e-6)) for i in rows]
        ax.broken_barh(
            spans,
            (y_of[name] - row, ROW_HEIGHT),
            facecolors=colors[name],
            edgecolors="black",
            linewidths=0.3,
        )

    # Every package gets a row *and* its own label. The canvas grows to fit; it is
    # never the labels or the rows that get dropped to keep the file small. The
    # y-label text is the dominant cost in the SVG (~4.6 MB for 2 808 labels), and
    # that is the same order as the original chart, so there is nothing to win by
    # hiding them.
    height = min(max(MIN_HEIGHT_IN, INCHES_PER_PACKAGE * len(packages)), MAX_TIMELINE_HEIGHT_IN)
    fig.set_size_inches(16, height)

    ax.set_yticks(range(len(packages)))
    ax.set_yticklabels(packages, fontsize=4)
    ax.set_ylim(len(packages) - 0.5, -0.5)
    ax.set_xlabel("Time")
    ax.set_title(f"Package Installation Periods ({note})")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig
