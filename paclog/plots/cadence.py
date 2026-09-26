"""Charts about *how often* things happen.

Install-period lengths live in :mod:`paclog.plots.lifetime` instead, which needs
two charts to say one thing properly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import analyze
from .base import figure, finish
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


@chart(
    "upgrade-interval-distribution",
    "upgrade_interval_distribution.svg",
    "Upgrade Interval Distribution",
    "every gap between consecutive upgrades of a package, pooled",
)
def upgrade_interval_distribution(frame: pd.DataFrame, ctx: Context):
    """All upgrade intervals in one histogram, plus the median and the mean.

    The companion to ``upgrade-interval``, not a replacement. That chart shows
    which packages are irregular; this one shows what the typical wait actually is.
    Neither answers the other's question, and the notebook named the gap when it
    called the box plot unreadable and then did not do anything about it.

    Log x, because the values span three orders of magnitude. Bars are anchored to
    their bin's low edge at the true bin width, so a bar's area is the count it
    stands for rather than an artefact of which edge the width was measured from --
    on a geometric grid those edges differ by a visible factor at the left of the
    range -- and so that adjacent bars tile the axis instead of leaving a sliver
    between each pair.

    The median and the mean are both drawn because on a real log they disagree by
    a factor of three. That disagreement is the finding: a mean of 64 days beside a
    median of 19 is a long tail, not a typical wait, and only the shape says so.
    """
    table = analyze.upgrade_interval_distribution(frame)
    fig = figure("upgrade-interval-distribution")
    ax = fig.add_subplot(111)

    if table.empty:
        ax.text(
            0.5, 0.5, "no upgrade intervals", ha="center", va="center", transform=ax.transAxes
        )
        ax.set_title("Upgrade interval distribution")
        return fig

    low = table["low_days"].to_numpy()
    high = table["high_days"].to_numpy()
    # ``align="edge"`` on the low edge, not a bar centred on the geometric midpoint.
    # On a log axis matplotlib measures a centred bar's width in the transformed
    # space, so the drawn rectangle lands a fraction inside its own bin and every
    # pair of bars is separated by a sliver of background; anchored to the low edge
    # the patch covers its bin exactly and the bars tile the axis. The width stays
    # the true bin width, so a bar's area is the count it stands for rather than an
    # artefact of which edge the width was measured from.
    ax.bar(low, table["count"], width=high - low, align="edge", color="#1565c0", linewidth=0)
    ax.set_xscale("log")
    ax.set_xlabel("Days between consecutive upgrades of one package")
    ax.set_ylabel("Intervals")

    intervals = analyze.upgrade_intervals(frame)
    median = float(intervals["interval_days"].median())
    mean = float(intervals["interval_days"].mean())
    # The two labels sit on opposite sides of their own lines and at different
    # heights. Both are annotated to the right at the same height they overlapped:
    # on a log axis over three decades a median of 20 days and a mean of 66 days are
    # close enough that the first label's right edge lands on the second one. The
    # mean is above the median in any right-skewed distribution, so the labels
    # point away from each other.
    markers = (
        (median, "median", "#2e7d32", "left", 0.97),
        (mean, "mean", "#c62828", "right", 0.88),
    )
    for value, label, color, align, height in markers:
        ax.axvline(value, color=color, linewidth=1.0, linestyle="--", alpha=0.85)
        ax.text(
            value,
            height,
            f"{label} {value:,.0f} d ",
            color=color,
            fontsize=9,
            ha=align,
            va="top",
            transform=ax.get_xaxis_transform(),
        )

    total = int(table["count"].sum())
    ax.set_title(
        f"Days between upgrades ({total:,} intervals across "
        f"{intervals['package'].nunique():,} packages; log scale)"
    )
    finish(ax)
    return fig
