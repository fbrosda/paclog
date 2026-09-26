"""Pure, side-effect-free parsing of ``pacman.log`` lines.

The two hand-written regexes and two near-identical ``parse_timestamp_*``
functions from the original script are collapsed here into one envelope match, a
verb lookup, and a small table of timestamp formats.

The important behavioural change is that a line which is *not* understood no
longer disappears silently. :func:`parse_line` always says why.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import NamedTuple

from .model import VERSION_CHANGING, Action, Event

#: ``[2016-03-12 12:37] [ALPM] upgraded foo (1 -> 2)``
#:
#: The timestamp and tag are captured loosely on purpose: pacman changed the
#: timestamp format mid-life (offset-less ``%Y-%m-%d %H:%M`` before 2019-10-27,
#: ISO-8601 with an offset afterwards) and both forms appear in the same file.
ENVELOPE_RE = re.compile(r"^\[(?P<ts>[^\]]+)\]\s+\[(?P<tag>[^\]]+)\]\s+(?P<rest>.*)$")

#: Only reached once the first word of an ``[ALPM]`` line is known to be a verb.
PACKAGE_RE = re.compile(r"^(?P<package>\S+)\s+\((?P<version_info>.+)\)\s*$")

#: Offset-bearing timestamps only, used to infer a timezone for the naive lines.
OFFSET_RE = re.compile(r"[+-]\d{4}$")

#: Order matters only for reporting: the first format that parses wins.
TIMESTAMP_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M%z",
    "%Y-%m-%d %H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
)

ARROW = " -> "


class SkipReason(Enum):
    """Why a line did not become an :class:`~paclog.model.Event`."""

    NO_ENVELOPE = "no_envelope"
    NOT_ALPM = "not_alpm"
    NOT_PACKAGE_ACTION = "not_package_action"
    BAD_TIMESTAMP = "bad_timestamp"
    BAD_VERSION = "bad_version"
    MALFORMED = "malformed"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


#: Reasons that mean "this looked like a package event but was broken".
#:
#: ``NO_ENVELOPE`` is deliberately *not* here: those are lines with no timestamp at
#: all -- blank lines and wrapped ``[ALPM-SCRIPTLET]`` output such as
#: ``filesystem: 750  package: 755`` -- which are ordinary log noise, not defects.
#: Including them would make ``--strict`` fire on a perfectly healthy 10-year log.
STRUCTURAL_REASONS = frozenset({SkipReason.BAD_TIMESTAMP, SkipReason.BAD_VERSION, SkipReason.MALFORMED})


@dataclass
class ParseStats:
    """Coverage accounting, so a surprising row count is explainable."""

    total: int = 0
    matched: int = 0
    skipped: Counter = field(default_factory=Counter)
    examples: dict[SkipReason, list[str]] = field(default_factory=dict)
    multi_hop: int = 0

    def record_skip(self, reason: SkipReason, line: str) -> None:
        self.skipped[reason] += 1
        bucket = self.examples.setdefault(reason, [])
        if len(bucket) < 3:
            bucket.append(line)

    @property
    def structural_problems(self) -> int:
        return sum(self.skipped[r] for r in STRUCTURAL_REASONS)

    def as_dict(self) -> dict[str, object]:
        return {
            "total_lines": self.total,
            "events": self.matched,
            "skipped": {r.value: n for r, n in sorted(self.skipped.items(), key=lambda kv: kv[0].value)},
            "multi_hop_version_chains": self.multi_hop,
        }

    def summary(self) -> str:
        parts = [f"{self.total} lines", f"{self.matched} events"]
        if self.skipped:
            detail = ", ".join(f"{r.value}={n}" for r, n in self.skipped.most_common())
            parts.append(f"skipped ({detail})")
        if self.multi_hop:
            parts.append(f"multi-hop={self.multi_hop}")
        return ", ".join(parts)


class ParsedLine(NamedTuple):
    """Result of parsing one line: an event, or the reason there is none."""

    event: Event | None
    reason: SkipReason | None


def parse_timestamp(raw: str, assume_tz) -> datetime:
    """Parse a pacman timestamp into an aware UTC datetime.

    Offset-bearing stamps keep their offset. Offset-less stamps -- the format
    pacman used until 2019-10-27 -- are localized with ``assume_tz``, which the
    caller resolves to something concrete.
    """
    raw = raw.strip()
    for fmt in TIMESTAMP_FORMATS:
        try:
            parsed = datetime.strptime(raw, fmt)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=assume_tz)
        return parsed.astimezone(timezone.utc)
    raise ValueError(f"unrecognised timestamp: {raw!r}")


def infer_offset(raw: str) -> int | None:
    """Minutes east of UTC for an offset-bearing stamp, else ``None``."""
    if not OFFSET_RE.search(raw):
        return None
    sign = -1 if raw[-5] == "-" else 1
    hours, minutes = int(raw[-4:-2]), int(raw[-2:])
    return sign * (hours * 60 + minutes)


def split_versions(action: Action, version_info: str) -> tuple[str | None, str | None] | None:
    """Split a version field into ``(before, after)``, or ``None`` if malformed.

    An upgrade chain longer than two hops (``a -> b -> c``) collapses to the
    endpoints: what the user cares about is where the package came from and where
    it ended up.
    """
    version_info = version_info.strip()
    if action in VERSION_CHANGING:
        parts = [p.strip() for p in version_info.split(ARROW)]
        parts = [p for p in parts if p]
        if len(parts) < 2:
            return None
        return parts[0], parts[-1]
    if action is Action.INSTALLED:
        return None, version_info
    if action is Action.REMOVED:
        return version_info, None
    # reinstalled: one version, unchanged on both sides
    return version_info, version_info


def parse_line(line: str, assume_tz, stats: ParseStats | None = None) -> ParsedLine:
    """Parse one log line. Never raises; always explains a non-event."""
    line = line.rstrip("\n")

    envelope = ENVELOPE_RE.match(line)
    if envelope is None:
        if stats is not None:
            stats.record_skip(SkipReason.NO_ENVELOPE, line)
        return ParsedLine(None, SkipReason.NO_ENVELOPE)

    tag = envelope.group("tag")
    if tag != "ALPM":
        if stats is not None:
            stats.record_skip(SkipReason.NOT_ALPM, line)
        return ParsedLine(None, SkipReason.NOT_ALPM)

    rest = envelope.group("rest")
    verb = rest.split(" ", 1)[0]
    try:
        action = Action(verb)
    except ValueError:
        # 'transaction started', "running 'foo.hook'...", 'warning: ...'
        if stats is not None:
            stats.record_skip(SkipReason.NOT_PACKAGE_ACTION, line)
        return ParsedLine(None, SkipReason.NOT_PACKAGE_ACTION)

    match = PACKAGE_RE.match(rest[len(verb) :].lstrip())
    if match is None:
        if stats is not None:
            stats.record_skip(SkipReason.MALFORMED, line)
        return ParsedLine(None, SkipReason.MALFORMED)

    try:
        timestamp = parse_timestamp(envelope.group("ts"), assume_tz)
    except ValueError:
        if stats is not None:
            stats.record_skip(SkipReason.BAD_TIMESTAMP, line)
        return ParsedLine(None, SkipReason.BAD_TIMESTAMP)

    versions = split_versions(action, match.group("version_info"))
    if versions is None:
        if stats is not None:
            stats.record_skip(SkipReason.BAD_VERSION, line)
        return ParsedLine(None, SkipReason.BAD_VERSION)

    before, after = versions
    if stats is not None:
        if action in VERSION_CHANGING and ARROW in match.group("version_info"):
            if match.group("version_info").count(ARROW) > 1:
                stats.multi_hop += 1
        stats.matched += 1

    return ParsedLine(
        Event(
            package=match.group("package"),
            timestamp=timestamp,
            action=action,
            version_before=before or None,
            version_after=after or None,
        ),
        None,
    )


def iter_events(
    lines: Iterable[str], assume_tz, stats: ParseStats | None = None
) -> Iterator[Event]:
    """Yield events from an iterable of log lines, filling in ``stats``."""
    stats = stats if stats is not None else ParseStats()
    for line in lines:
        stats.total += 1
        parsed = parse_line(line, assume_tz, stats)
        if parsed.event is not None:
            yield parsed.event
