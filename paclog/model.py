"""Core data types for paclog.

Everything that needs to agree on "what is an event" and "what order do actions
appear in" lives here, so that no other module has to guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

#: Bumped whenever the CSV column set or the timestamp encoding changes.
#: Written into ``data/pacman_history.meta.json`` so a stale CSV can be detected
#: instead of silently misinterpreted.
SCHEMA_VERSION = 2

#: Column order of the CSV, and the row order of the event frame.
COLUMNS = ("package", "timestamp", "action", "version_before", "version_after")


class Action(str, Enum):
    """A package-affecting ALPM transaction entry.

    ``downgraded`` was missing from the original hand-written regex and its events
    were dropped without a word. It is a real ALPM verb, so it belongs here.
    """

    INSTALLED = "installed"
    UPGRADED = "upgraded"
    REMOVED = "removed"
    REINSTALLED = "reinstalled"
    DOWNGRADED = "downgraded"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


#: Canonical presentation order. Charts must iterate this, never
#: ``df["action"].unique()``: the order of first appearance in the log is data
#: dependent, and deriving a subplot grid from it is how the original
#: ``axes[i]`` indexing ended up one action away from an IndexError.
ACTION_ORDER: tuple[Action, ...] = (
    Action.INSTALLED,
    Action.UPGRADED,
    Action.DOWNGRADED,
    Action.REMOVED,
    Action.REINSTALLED,
)

#: Actions that move a package from one version to another.
VERSION_CHANGING = frozenset({Action.UPGRADED, Action.DOWNGRADED})


@dataclass(frozen=True, slots=True)
class Event:
    """A single package-affecting entry from ``pacman.log``.

    ``timestamp`` is always timezone-aware and normalized to UTC. Lines written
    in the pre-2019 offset-less format are localized with the configured
    ``assume_tz`` before being converted.
    """

    package: str
    timestamp: datetime
    action: Action
    version_before: str | None = None
    version_after: str | None = None

    def as_row(self) -> dict[str, object]:
        return {
            "package": self.package,
            "timestamp": self.timestamp,
            "action": self.action.value,
            "version_before": self.version_before,
            "version_after": self.version_after,
        }
