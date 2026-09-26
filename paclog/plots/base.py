"""Shared chart styling.

One place for the palette, one place for figure sizes, one place that decides how
a figure becomes a file. Before this existed, the sizes lived in two places that
had already drifted apart (``32x18`` in the script vs ``16x9`` in the notebook).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend selection)

from ..model import ACTION_ORDER  # noqa: E402

#: Semantic colour per action, used by every chart that shows actions. The
#: original scripts let matplotlib pick, so the same action was a different colour
#: in the pie chart and the monthly bars.
ACTION_COLORS: dict[str, str] = {
    "installed": "#2e7d32",
    "upgraded": "#1565c0",
    "downgraded": "#6a1b9a",
    "removed": "#c62828",
    "reinstalled": "#616161",
}

ACTION_COLOR_LIST: list[str] = [ACTION_COLORS[a.value] for a in ACTION_ORDER]

#: Figure sizes in inches. These are the notebook's, chosen over the script's
#: larger ones because 32x18 renders unreadable text in a browser.
FIG_SIZES: dict[str, tuple[float, float]] = {
    "events-per-hour": (14, 8),
    "events-per-weekday": (12, 6),
    "events-per-month": (16, 9),
    "action-distribution": (7, 7),
    "upgrade-interval": (7, 10),
    "package-lifetime": (7, 10),
    "top-packages": (16, 12),
    "timeline": (16, 20),
}

GRID_KWARGS = {"axis": "y", "alpha": 0.25, "linewidth": 0.8}

_STYLE_APPLIED = False


def apply_style() -> None:
    """Install the rcParams every chart relies on. Idempotent.

    ``svg.hashsalt`` is the load-bearing one: it defaults to ``None``, which makes
    matplotlib salt SVG element ids from a fresh UUID, so two runs over identical
    data produce different bytes. Setting it is what makes charts diffable.
    """
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    plt.rcParams.update(
        {
            "svg.hashsalt": "paclog",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.frameon": False,
            "figure.autolayout": False,
            "savefig.bbox": "tight",
        }
    )
    _STYLE_APPLIED = True


def color_for(action: str) -> str:
    return ACTION_COLORS.get(action, "#455a64")


def colors_for(actions) -> list[str]:
    return [ACTION_COLORS.get(a, "#455a64") for a in actions]


def finish(ax, *, xgrid: bool = False, ygrid: bool = True) -> None:
    """Apply the house grid rules to an axes."""
    if ygrid:
        ax.grid(True, axis="y", **{k: v for k, v in GRID_KWARGS.items() if k != "axis"})
    if xgrid:
        ax.grid(True, axis="x", **{k: v for k, v in GRID_KWARGS.items() if k != "axis"})


def save(fig, out_dir: Path, filename: str) -> Path:
    """Write a figure as SVG, creating the output directory if needed.

    ``out_dir`` is created here because the original scripts assumed it already
    existed, which is the first thing that breaks on a fresh clone.

    ``metadata={"Date": None}`` suppresses the ``<dc:date>`` element matplotlib
    otherwise stamps into every SVG. Together with ``svg.hashsalt`` that is what
    makes two runs over unchanged data produce byte-identical files, so charts can
    be committed and diffed.
    """
    apply_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    return path


def figure(name: str, **kwargs):
    """Create a figure at the registered size for ``name``."""
    apply_style()
    size = FIG_SIZES.get(name, (12, 7))
    return plt.figure(figsize=kwargs.pop("figsize", size), **kwargs)
