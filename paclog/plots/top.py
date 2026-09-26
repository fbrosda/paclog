"""The per-action top-packages grid.

The original iterated ``df["action"].unique()`` and indexed ``axes[i]``, so the
subplot count and the loop count had to agree by luck. Here the grid is sized from
:data:`~paclog.model.ACTION_ORDER`, which is fixed.
"""

from __future__ import annotations

import math

import pandas as pd

from .. import analyze
from ..model import ACTION_ORDER
from .base import color_for, figure
from .registry import Context, chart

TOP_N = 20


@chart(
    "top-packages",
    "top_packages.svg",
    "Top 20 Most Modified Packages By Action",
    "busiest packages for each action",
)
def top_packages(frame: pd.DataFrame, ctx: Context):
    grid = analyze.top_packages_grid(frame, TOP_N)
    populated = [a for a in ACTION_ORDER if not grid[a.value].empty]
    n = max(len(populated), 1)
    cols = 2 if n > 1 else 1
    rows = math.ceil(n / cols)

    fig = figure("top-packages")
    axes = fig.subplots(rows, cols, squeeze=False).ravel()

    for ax, action in zip(axes, populated):
        counts = grid[action.value]
        ax.barh(counts.index, counts.values, color=color_for(action.value))
        ax.set_title(action.value.capitalize())
        ax.set_xlabel("# Changes")
        ax.set_ylabel("Package")
        ax.invert_yaxis()
        ax.xaxis.grid(True, alpha=0.25, linewidth=0.8)
        ax.yaxis.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(f"Top {TOP_N} Most Modified Packages By Action", y=0.995)
    fig.tight_layout()
    return fig
