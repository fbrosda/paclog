"""paclog -- analytics and visualizations for Arch Linux pacman logs.

Typical use is the ``paclog`` command line tool::

    paclog parse
    paclog plot
    paclog timeline

The same pieces are importable, which is what the notebook does::

    from paclog import analyze, plots
"""

from __future__ import annotations

from .config import Config
from .model import ACTION_ORDER, SCHEMA_VERSION, Action, Event

__version__ = "0.2.0"

__all__ = [
    "ACTION_ORDER",
    "SCHEMA_VERSION",
    "Action",
    "Config",
    "Event",
    "__version__",
    "analyze",
    "loader",
    "parsing",
    "plots",
    "store",
]
