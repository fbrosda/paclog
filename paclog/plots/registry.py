"""The chart registry.

Every chart is one entry: a name, a filename, a title, and a render function. The
CLI iterates this, ``--list`` reads it, and the notebook calls the same render
functions. That is the mechanism that stops the notebook and the CLI from
drifting apart -- there is only one implementation of each chart.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.figure import Figure


@dataclass(frozen=True, slots=True)
class Context:
    """Everything a chart may need beyond the data itself."""

    as_of: pd.Timestamp
    seed: int = 0
    options: dict = field(default_factory=dict)


RenderFn = Callable[[pd.DataFrame, Context], "Figure"]

REGISTRY: dict[str, "Chart"] = {}


@dataclass(frozen=True, slots=True)
class Chart:
    name: str
    filename: str
    title: str
    render: RenderFn
    description: str = ""
    default: bool = True
    #: dpi for rasterized artists. ``None`` is the house default and is correct for
    #: every pure-vector chart; a chart whose artwork is a raster over a very large
    #: canvas sets it, because otherwise its cells are smeared together.
    dpi: int | None = None

    def __call__(self, frame: pd.DataFrame, ctx: Context) -> "Figure":
        return self.render(frame, ctx)


def chart(
    name: str,
    filename: str,
    title: str,
    description: str = "",
    default: bool = True,
    dpi: int | None = None,
) -> Callable[[RenderFn], RenderFn]:
    """Register a render function under ``name``.

    ``default=False`` keeps a chart reachable by name but out of a bare
    ``paclog plot``, which is how the timeline stays a deliberate choice: it is
    the expensive one.
    """

    def decorate(fn: RenderFn) -> RenderFn:
        if name in REGISTRY:
            raise ValueError(f"chart {name!r} is already registered")
        REGISTRY[name] = Chart(
            name=name,
            filename=filename,
            title=title,
            render=fn,
            description=description,
            default=default,
            dpi=dpi,
        )
        return fn

    return decorate


def names() -> list[str]:
    return sorted(REGISTRY)


def default_names() -> list[str]:
    return sorted(n for n, c in REGISTRY.items() if c.default)


def get(name: str) -> Chart:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown chart {name!r}; available: {', '.join(names())}") from None


def render(name: str, frame: pd.DataFrame, ctx: Context) -> "Figure":
    return get(name)(frame, ctx)


def render_all(
    frame: pd.DataFrame, ctx: Context, out_dir: Path, only: Iterable[str] | None = None
) -> list[Path]:
    """Render the selected charts and return the files written, in order.

    With no selection, the default set runs -- the timeline is excluded because it
    costs far more than the rest put together.
    """
    from .base import save  # local import keeps matplotlib out of module import time

    selected = list(only) if only else default_names()
    unknown = [n for n in selected if n not in REGISTRY]
    if unknown:
        raise KeyError(f"unknown chart(s): {', '.join(unknown)}; available: {', '.join(names())}")

    written: list[Path] = []
    for name in selected:
        entry = get(name)
        fig = render(name, frame, ctx)
        written.append(save(fig, out_dir, entry.filename, dpi=entry.dpi))
    return written


def describe() -> str:
    lines = ["available charts:"]
    for name in names():
        entry = REGISTRY[name]
        marker = "" if entry.default else "  (opt-in)"
        lines.append(f"  {name:<22} {entry.filename:<28} {entry.description}{marker}")
    return "\n".join(lines)
