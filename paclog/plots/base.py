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
#: larger ones because 32x18 renders unreadable text in a browser. Two entries are
#: only a floor: ``staleness-heatmap`` and ``timeline`` both size themselves to the
#: data, because a row per package is the whole point of them.
FIG_SIZES: dict[str, tuple[float, float]] = {
    "events-per-hour": (14, 8),
    "events-per-weekday": (12, 6),
    "events-per-month": (16, 15),
    "action-distribution": (7, 7),
    "upgrade-interval": (7, 10),
    "upgrade-interval-distribution": (12, 7),
    "installed-whole-time": (7, 8),
    "package-lifetime": (7, 10),
    "installed-set-size": (14, 8),
    "transactions": (14, 9),
    "activity-calendar": (16, 12),
    "staleness-heatmap": (16, 20),
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

    ``svg.fonttype`` is the second one that matters, and it is a size decision.
    The default, ``"path"``, converts every glyph to an outline and then emits one
    ``<use>`` element per character, so a 2 808-label y axis becomes 38 167 ``<use>``
    references plus 2 329 path definitions -- about 64% of the 4.6 MB timeline and
    17% more in glyph defs, for text that is illegible at any zoom on a 700-inch
    canvas. ``"none"`` writes a real ``<text>`` element instead and the viewer
    shapes the glyphs. Nothing is dropped and the bytes stay reproducible (this
    only changes how the *renderer's* font is applied, not what the file contains),
    but it does mean the committed chart is now at the mercy of the viewer's font
    metrics: matplotlib lays the labels out with DejaVu Sans and a machine without
    it substitutes something else. In practice the substitutes are narrower, so
    labels gain slack rather than colliding, and the two charts that would be hurt
    most -- ``timeline`` and ``staleness-heatmap`` -- use a 4pt label on a canvas
    where the row pitch is 18pt, so there is a lot of room to give away.
    """
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    plt.rcParams.update(
        {
            "svg.hashsalt": "paclog",
            "svg.fonttype": "none",
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


def save(fig, out_dir: Path, filename: str, dpi: int | None = None) -> Path:
    """Write a figure as SVG, creating the output directory if needed.

    ``out_dir`` is created here because the original scripts assumed it already
    existed, which is the first thing that breaks on a fresh clone.

    ``metadata={"Date": None}`` suppresses the ``<dc:date>`` element matplotlib
    otherwise stamps into every SVG. Together with ``svg.hashsalt`` that is what
    makes two runs over unchanged data produce byte-identical files, so charts can
    be committed and diffed.

    ``dpi`` reaches rasterized artists only, which is the point of it. A chart
    that is 700 inches tall embeds its cells at 100 dpi as one blurry smear, and
    the dpi that makes a cell land on its own pixel is a property of that chart
    rather than of the house style. Charts that are pure vector leave it ``None``
    and are byte-for-byte unaffected.
    """
    apply_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None}, dpi=dpi)
    plt.close(fig)
    return path


def figure(name: str, **kwargs):
    """Create a figure at the registered size for ``name``."""
    apply_style()
    size = FIG_SIZES.get(name, (12, 7))
    return plt.figure(figsize=kwargs.pop("figsize", size), **kwargs)
