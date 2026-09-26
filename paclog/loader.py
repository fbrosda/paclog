"""Turning log files into a tidy, timezone-aware event frame."""

from __future__ import annotations

import gzip
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path

import pandas as pd

from .config import AUTO, Config, fixed_offset_tz, format_offset
from .model import COLUMNS, Event
from .parsing import (
    ENVELOPE_RE,
    ParseStats,
    infer_offset,
    iter_events,
    parse_line,
)

#: Columns that must agree for two rows to be considered the same event.
DEDUPE_KEY = ("package", "timestamp", "action", "version_before", "version_after")


class LogNotFound(Exception):
    """No readable log file could be located."""


@dataclass(frozen=True, slots=True)
class LogSource:
    """Provenance for one input file, recorded in the metadata sidecar."""

    path: Path
    size: int
    mtime: float
    lines: int

    def as_dict(self, root: Path) -> dict[str, object]:
        try:
            shown = str(self.path.relative_to(root))
        except ValueError:
            shown = str(self.path)
        return {"path": shown, "bytes": self.size, "mtime": self.mtime, "lines": self.lines}


def open_log(path: Path):
    """Open a plain or gzip-compressed log as text."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def discover_logs(config: Config, warn=lambda msg: None) -> list[Path]:
    """Resolve the set of log files to read.

    Explicit ``--log`` paths win. Otherwise the configured globs are expanded,
    which by default picks up rotated and gzipped history -- the original script
    read exactly one file and said nothing about the rest.
    """
    if config.log_paths:
        found = [p for p in config.log_paths if p.is_file()]
        for missing in (p for p in config.log_paths if not p.is_file()):
            warn(f"warning: log not found: {missing}")
    else:
        found = []
        for pattern in config.log_globs:
            matched = sorted(Path(pattern).parent.glob(Path(pattern).name))
            found.extend(p for p in matched if p.is_file())
        found = _dedupe(found)

    if not found:
        raise LogNotFound(
            "no pacman log found. Looked for: "
            + ", ".join(config.log_globs if not config.log_paths else (str(p) for p in config.log_paths))
            + "\nUse --log PATH to point at yours."
        )
    return found


def _dedupe(paths: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            out.append(path)
    return out


def infer_assume_tz(paths: Sequence[Path], config: Config) -> tuple[tzinfo, str]:
    """Decide what timezone the offset-less legacy lines belong to.

    Returns the tzinfo plus a human-readable description of how it was chosen,
    because the ``auto`` answer is an approximation and belongs in the metadata.

    Resolution order: explicit setting, then the most common UTC offset among the
    lines that *do* carry one, then UTC. The modal-offset answer is wrong for the
    minority of lines on the other side of a DST switch, so pass an IANA name to
    ``--assume-tz`` when you want it exact.
    """
    configured = config.assume_tzinfo
    if configured is not None:
        return configured, f"explicit ({format_offset(configured)})"

    counts: Counter[int] = Counter()
    for path in paths:
        with open_log(path) as handle:
            for line in handle:
                envelope = ENVELOPE_RE.match(line)
                if envelope is None:
                    continue
                minutes = infer_offset(envelope.group("ts"))
                if minutes is not None:
                    counts[minutes] += 1

    if not counts:
        return timezone.utc, "no offset-bearing lines found, assuming UTC"

    minutes, hits = counts.most_common(1)[0]
    return (
        fixed_offset_tz(minutes),
        f"auto (most common offset in log: {format_offset(fixed_offset_tz(minutes))}, "
        f"{hits}/{sum(counts.values())} lines)",
    )


def read_events(
    paths: Sequence[Path],
    assume_tz: tzinfo,
    config: Config,
    warn=lambda msg: None,
    dedupe: bool = False,
) -> tuple[list[Event], ParseStats, list[LogSource]]:
    """Read every log file into events, with coverage accounting.

    Duplicate rows are kept by default. Two identical rows are usually *not* a
    bug: pacman can run two transactions for the same package in the same second
    (a real ``pacman -U`` twice), and a reinstall inside a ``transaction failed``
    leaves an event that never took effect. Pass ``dedupe=True`` when you have
    deliberately pointed ``--log`` at overlapping rotations.
    """
    stats = ParseStats()
    events: list[Event] = []
    sources: list[LogSource] = []

    for path in paths:
        lines = 0
        file_stats = ParseStats()
        with open_log(path) as handle:
            for event in iter_events(handle, assume_tz, file_stats):
                events.append(event)
            lines = file_stats.total
        stats.total += file_stats.total
        stats.matched += file_stats.matched
        stats.multi_hop += file_stats.multi_hop
        stats.skipped.update(file_stats.skipped)
        for reason, examples in file_stats.examples.items():
            stats.examples.setdefault(reason, []).extend(examples)
        stat = path.stat()
        sources.append(LogSource(path=path, size=stat.st_size, mtime=stat.st_mtime, lines=lines))

    if dedupe:
        before = len(events)
        events = dedupe_events(events)
        duplicates = before - len(events)
        if duplicates and not config.quiet:
            warn(f"deduped {duplicates} exactly duplicated rows")
    return events, stats, sources


def dedupe_events(events: Sequence[Event]) -> list[Event]:
    """Remove exact duplicates, which overlapping rotations produce."""
    seen: set[tuple] = set()
    out: list[Event] = []
    for event in events:
        key = (event.package, event.timestamp, event.action.value, event.version_before, event.version_after)
        if key not in seen:
            seen.add(key)
            out.append(event)
    return out


def frame_from_events(events: Sequence[Event]) -> pd.DataFrame:
    """Build the canonical event frame, sorted deterministically.

    Timestamp first, because this is an event log: a monotonic ``timestamp``
    column is a useful invariant, and everything that needs per-package grouping
    does its own groupby anyway.
    """
    if not events:
        return empty_frame()
    frame = pd.DataFrame([e.as_row() for e in events], columns=list(COLUMNS))
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    # A total order keeps the CSV and every chart reproducible regardless of how
    # the log happened to be ordered on disk.
    return frame.sort_values(["timestamp", "package", "action"], kind="stable").reset_index(drop=True)


def empty_frame() -> pd.DataFrame:
    frame = pd.DataFrame(columns=list(COLUMNS))
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


def to_display_tz(frame: pd.DataFrame, tz: tzinfo) -> pd.DataFrame:
    """Convert the UTC frame into the timezone used for human-facing buckets.

    Bucketing by local wall clock is the whole point of the hour-of-day chart, so
    the conversion happens here, once, rather than being guessed at by each plot.
    """
    if frame.empty:
        return frame
    out = frame.copy()
    out["timestamp"] = out["timestamp"].dt.tz_convert(tz)
    return out


def resolve_as_of(config: Config, frame: pd.DataFrame) -> datetime:
    """Pick the timestamp that still-installed packages are considered alive at.

    Defaults to the newest event in the data rather than ``now()``, so that a
    re-run over unchanged input produces an unchanged chart. Pass ``--as-of now``
    to opt back into wall-clock time.
    """
    requested = config.as_of
    if requested:
        if requested.strip().lower() == "now":
            return datetime.now(timezone.utc).replace(microsecond=0)
        return _to_utc(datetime.fromisoformat(requested), config)
    if frame.empty:
        return datetime.now(timezone.utc).replace(microsecond=0)
    return frame["timestamp"].max().to_pydatetime()


def _to_utc(value: datetime, config: Config) -> datetime:
    if value.tzinfo is None:
        assumed = config.assume_tzinfo
        if assumed is None:
            assumed = config.display_tzinfo or timezone.utc
        value = value.replace(tzinfo=assumed)
    return value.astimezone(timezone.utc)


def resolve_display_tz(config: Config, frame: pd.DataFrame, assume_tz: tzinfo) -> tuple[tzinfo, str]:
    """The timezone charts bucket in, plus how it was chosen."""
    configured = config.display_tzinfo
    if configured is not None:
        return configured, f"explicit ({format_offset(configured)})"
    if config.tz == AUTO:
        return assume_tz, f"auto (same as assume-tz: {format_offset(assume_tz)})"
    return timezone.utc, "UTC"  # pragma: no cover - unreachable via CLI


def assert_utc(frame: pd.DataFrame) -> None:
    """Guard for the failure mode where a CSV sneaks in naive timestamps."""
    if frame.empty:
        return
    dtype = frame["timestamp"].dtype
    if not isinstance(dtype, pd.DatetimeTZDtype):
        raise TypeError(
            f"timestamp column is {dtype}, expected a timezone-aware dtype. "
            "Re-create the CSV with `paclog parse`."
        )


def timedelta_days(start: datetime, end: datetime) -> timedelta:  # pragma: no cover - helper
    return end - start


__all__ = [
    "COLUMNS",
    "DEDUPE_KEY",
    "LogNotFound",
    "LogSource",
    "assert_utc",
    "dedupe_events",
    "discover_logs",
    "empty_frame",
    "frame_from_events",
    "infer_assume_tz",
    "open_log",
    "parse_line",
    "read_events",
    "resolve_as_of",
    "resolve_display_tz",
    "to_display_tz",
]
