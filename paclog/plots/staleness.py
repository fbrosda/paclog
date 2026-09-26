"""The staleness heatmap: maintenance debt, package by month.

The timeline answers *when was this installed*. This answers *when did anyone last
touch it*, which is a different question and the one that shows neglect: a package
installed in 2016 and upgraded yesterday sits on the timeline for a decade and is
stale for none of it.

It covers the same ground as the timeline -- every package, every one of them
labelled, no cap -- but spends the space on the shape of the data rather than on
package names alone.
"""

from __future__ import annotations

import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from .. import analyze
from ..config import INCHES_PER_PACKAGE, MAX_TIMELINE_HEIGHT_IN
from .base import figure
from .registry import Context, chart

#: Where the colour scale tops out, in days. Two years is past which "not touched
#: in a long time" stops distinguishing anything: on the author's log the 90th
#: percentile is 1 417 days and the median 167, so a scale to the maximum (3 850)
#: would render the median as indistinguishable from untouched. Cells past the cap
#: are drawn in the top colour rather than dropped, and the colourbar says so.
SCALE_CAP_DAYS = 730

#: One cell is a square this many inches on a side, which makes the grid legible
#: the same way the timeline's 18 pt rows are: the row pitch comes from
#: ``INCHES_PER_PACKAGE`` and the column pitch is set to match it.
MIN_WIDTH_IN = 8.0
MAX_WIDTH_IN = 60.0
MIN_HEIGHT_IN = 3.0

#: Fractions of the figure the cell grid occupies, as ``(left, bottom, width,
#: height)``. Stated rather than left to ``add_subplot``'s defaults because the
#: pitch has to come out *exact*: the raster dpi only means anything relative to
#: how many inches a cell actually occupies, and a default subplot is 0.9 x 0.77 of
#: whatever the figure happens to be, which is how the first attempt ended up with
#: 0.16-inch cells while believing they were 0.25.
GRID_BOX = (0.055, 0.030, 0.90, 0.950)

#: dpi for the embedded raster, giving two pixels per cell at the pitch above. The
#: grid is one big ``imshow``, so this is the only thing that decides how many
#: pixels a cell gets: at matplotlib's default 100 a 700-inch-tall figure crams 28
#: pixels into every row, which is not a heatmap but a gradient.
RASTER_DPI = 8

#: Label font size, matching the timeline's, so the two charts read at one zoom.
LABEL_FONTSIZE = 4

#: Thickness of the colour key, in inches, and the space above the grid it needs
#: for its own tick labels and caption.
BAR_THICKNESS_IN = 0.22
BAR_CLEARANCE_IN = 0.62


def _pitch(rows: int) -> float:
    """Inches per cell, shrinking only when the log is too long for the budget."""
    budget = MAX_TIMELINE_HEIGHT_IN * GRID_BOX[3]
    return min(INCHES_PER_PACKAGE, budget / rows) if rows else INCHES_PER_PACKAGE


def _figure_size(rows: int, columns: int) -> tuple[float, float]:
    """Figure size in inches for a grid of square cells, given the axes fraction.

    The floors are for the degenerate case only -- a log with no events has no cells
    to be square, and a zero-height figure is not something matplotlib can save.
    """
    pitch = _pitch(rows)
    width = min(max(MIN_WIDTH_IN, pitch * columns / GRID_BOX[2]), MAX_WIDTH_IN)
    height = max(MIN_HEIGHT_IN, pitch * rows / GRID_BOX[3])
    return width, height


@chart(
    "staleness-heatmap",
    "staleness_heatmap.svg",
    "Package Staleness Over Time",
    "days since each package's last event, one row per package",
    default=False,
    dpi=RASTER_DPI,
)
def staleness_heatmap(frame: pd.DataFrame, ctx: Context):
    """A package-by-month grid of days since the last event touching that package.

    Every package gets a row and every one of them is labelled, on the same terms
    as the timeline: dropping rows or striding the labels would make this a chart
    of a different machine than the one it was run on. Rows are ordered by first
    install rather than by name, which is what gives the grid a shape -- the base
    system is a solid block at the bottom, and the packages nothing has touched
    since show up as long pale runs instead of being scattered at random.

    The canvas grows to fit the rows exactly as the timeline's does. The first
    version of this chart was 16x20 inches for 2 808 packages, which works out to
    0.5 pt of vertical space per row: the labels overlapped into a grey smear and
    the 0.71 raster pixels per row turned the cells into a smooth vertical
    gradient that said nothing at all. Nothing was dropped to get from there to
    here; the figure simply got as tall as the data needed.

    Cells before a package existed are left blank, not zero-filled: a zero there
    would claim the package was touched during a month in which it was not
    installed. Each column is measured at the month's closing boundary and the last
    one at ``as_of``, so the right-hand edge of the chart is the same staleness
    :func:`analyze.staleness_now` reports.
    """
    grid = analyze.package_staleness(frame, ctx.as_of, order="first_seen")
    rows, columns = grid.shape
    width, height = _figure_size(rows, columns)
    # The figure is sized before the axes exist, because the axes fraction is fixed
    # in GRID_BOX and the figure is what that fraction is a fraction *of*.
    fig = figure("staleness-heatmap", figsize=(width, height))
    ax = fig.add_axes(list(GRID_BOX))

    if rows == 0:
        ax.text(0.5, 0.5, "no events", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        ax.set_title("Package staleness over time (no events)")
        return fig

    days = np.ma.masked_invalid(grid.days)
    image = ax.imshow(
        days,
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
        rasterized=True,
        vmin=0,
        vmax=SCALE_CAP_DAYS,
    )

    ax.set_yticks(range(rows))
    ax.set_yticklabels(grid.packages, fontsize=LABEL_FONTSIZE)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_xlabel("Month", labelpad=6)
    ax.set_title("Days since each package's last package event", pad=62)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")

    # The key sits in the band above the grid rather than beside it, because a
    # vertical key on a 1 100-inch-tall chart would spread four tick labels over
    # that distance. Its thickness is given in inches, not in figure fractions, so
    # it stays the same visual size whatever the log's length is.
    key = fig.add_axes(
        [
            GRID_BOX[0],
            GRID_BOX[1] + GRID_BOX[3] + BAR_CLEARANCE_IN / height,
            0.30,
            BAR_THICKNESS_IN / height,
        ]
    )
    bar = fig.colorbar(image, cax=key, orientation="horizontal")
    bar.set_label(
        f"days since last event (darkest = 0, capped at {SCALE_CAP_DAYS:,})",
        fontsize=8,
        labelpad=6,
    )
    bar.ax.tick_params(labelsize=7)

    stats = grid.described()
    blank = int(np.isnan(grid.days).sum())
    fig.text(
        GRID_BOX[0],
        0.008,
        f"{rows:,} packages x {columns} months, oldest first; "
        f"median {stats['median']:,.0f} d, 90th percentile {stats['p90']:,.0f} d, "
        f"oldest {stats['max']:,.0f} d; {blank:,} of {grid.days.size:,} cells predate the package",
        fontsize=8,
        color="#555555",
        va="bottom",
        ha="left",
    )
    return fig
