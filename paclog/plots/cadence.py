"""Charts about *how often* things happen."""

from __future__ import annotations

import pandas as pd

from .. import analyze
from .base import colors_for, figure, finish
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
    "package-lifetime",
    "package_lifetime.svg",
    "How Long Packages Stay Installed",
    "days per install period, longest first",
)
def package_lifetime(frame: pd.DataFrame, ctx: Context):
    lifetimes = analyze.package_lifetime(frame, ctx.as_of)
    fig = figure("package-lifetime")
    ax = fig.add_subplot(111)
    if lifetimes.empty:
        ax.text(0.5, 0.5, "no install periods", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Install period length")
        return fig

    top = lifetimes.head(40).iloc[::-1]
    ax.barh(top["package"], top["days"], color=colors_for(["upgraded"] * len(top)))
    ax.set_xlabel("Days installed")
    ax.set_ylabel("Package")
    ax.set_title("Install period length (longest 40)")
    ax.xaxis.grid(True, alpha=0.25, linewidth=0.8)
    ax.yaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig
