"""Chart registry for paclog.

Importing this package registers every chart, so ``paclog.plots.names()`` is the
authoritative list. Each chart is a ``(frame, Context) -> Figure`` function shared
by the CLI and the notebook.
"""

from __future__ import annotations

from . import cadence, events, lifetime, timeline, top  # noqa: F401  (imported for registration)
from .base import ACTION_COLORS, apply_style, save
from .registry import (
    REGISTRY,
    Chart,
    Context,
    chart,
    default_names,
    describe,
    get,
    names,
    render,
    render_all,
)

__all__ = [
    "ACTION_COLORS",
    "REGISTRY",
    "Chart",
    "Context",
    "apply_style",
    "chart",
    "default_names",
    "describe",
    "get",
    "names",
    "render",
    "render_all",
    "save",
]
