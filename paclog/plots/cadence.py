"""Charts about *how often* things happen.

Install-period lengths live in :mod:`paclog.plots.lifetime` instead, which needs
two charts to say one thing properly.
"""

from __future__ import annotations

import pandas as pd

from .. import analyze
from .base import figure
from .registry import Context, chart


@chart(
    "upgrade-interval",
    "upgrade_interval.svg",
    "Upgrade Interval",
    "days between consecutive upgrades, per package",
)
def upgrade_interval(frame: pd.DataFrame, ctx: Context):
    """Box plot of the gap between consecutive upgrades, one distribution per package.

    The original used ``groupby(...).apply(..., include_groups=False)`` to get here,
    which made the result depend on the pandas version. ``analyze.upgrade_intervals``
    does the arithmetic; this only draws it.
    """
    intervals = analyze.upgrade_intervals(frame)
    fig = figure("upgrade-interval")
    ax = fig.add_subplot(111)
    if intervals.empty:
        ax.text(0.5, 0.5, "no upgrades recorded", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Upgrade interval")
        return fig

    summary = analyze.interval_summary(frame)
    # Keep the busiest packages legible; a 2800-tick box plot is a smear.
    keep = summary.sort_values("count", ascending=False).head(60).index
    per_package = [group["interval_days"].values for _, group in intervals[intervals["package"].isin(keep)].groupby("package", sort=True)]
    labels = [p for p in keep if p in set(intervals["package"])]
    ax.boxplot(per_package, tick_labels=labels, showfliers=False)
    ax.set_ylabel("Days between upgrades")
    ax.set_title(f"Upgrade interval (top {len(labels)} packages by upgrade count)")
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.yaxis.grid(True, alpha=0.25, linewidth=0.8)
    ax.xaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig
