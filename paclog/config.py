"""Configuration: every path, timezone and knob the program has.

The original three scripts hardcoded ``/var/log/pacman.log``, ``data/`` and
``visualizations/`` as module constants and string literals, resolved against the
current working directory. Everything here is overridable, and relative paths
resolve against an explicit ``root`` instead of an implicit CWD.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import timedelta, timezone, tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

#: The live log plus the rotations a logrotate setup would leave behind. Without
#: the globs a rotated box silently analyses only the current file.
DEFAULT_LOG_GLOBS: tuple[str, ...] = (
    "/var/log/pacman.log",
    "/var/log/pacman.log.[0-9]*",
    "/var/log/pacman.log.*.gz",
)

AUTO = "auto"
UTC = "UTC"

#: Cap on how tall the timeline figure is allowed to get, in inches. The original
#: was ``len(packages) * 0.4`` with no ceiling, which is how a single chart turned
#: into a 4.2 MB SVG.
MAX_TIMELINE_HEIGHT_IN = 48.0
INCHES_PER_PACKAGE = 0.16
TIMELINE_TOP_N = 150


class ConfigError(Exception):
    """Raised for user-facing configuration mistakes."""


#: ``+01:00`` / ``+0100`` / ``-05:30``, for round-tripping a resolved offset out of
#: the metadata sidecar and back in.
OFFSET_RE = re.compile(r"^(?P<sign>[+-])(?P<hours>\d{1,2}):?(?P<minutes>\d{2})$")


def resolve_tzinfo(spec: str) -> tzinfo | None:
    """Turn a timezone spec into a ``tzinfo``.

    Returns ``None`` for ``auto``, meaning "the caller has to infer this from the
    data". Accepts ``UTC``, IANA names such as ``Europe/Berlin``, and fixed
    offsets such as ``+01:00``.
    """
    if spec == AUTO:
        return None
    if spec.upper() == UTC:
        return timezone.utc
    match = OFFSET_RE.match(spec)
    if match:
        minutes = int(match.group("hours")) * 60 + int(match.group("minutes"))
        if match.group("sign") == "-":
            minutes = -minutes
        return fixed_offset_tz(minutes)
    try:
        return ZoneInfo(spec)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ConfigError(
            f"unknown timezone {spec!r}: expected 'auto', 'UTC', a fixed offset like "
            f"'+01:00', or an IANA name such as 'Europe/Berlin'"
        ) from exc


def fixed_offset_tz(minutes: int) -> timezone:
    """Build a fixed-offset timezone from minutes east of UTC."""
    return timezone(timedelta(minutes=minutes))


def format_offset(tz: tzinfo) -> str:
    """Render a tzinfo as ``+HH:MM`` for human-readable metadata."""
    offset = tz.utcoffset(None)
    if offset is None:  # pragma: no cover - ZoneInfo always resolves for UTC dates
        return "+00:00"
    total = int(offset.total_seconds())
    sign = "-" if total < 0 else "+"
    total = abs(total)
    return f"{sign}{total // 3600:02d}:{(total % 3600) // 60:02d}"


@dataclass(frozen=True, slots=True)
class Config:
    """Fully resolved runtime configuration."""

    root: Path = Path(".")
    log_paths: tuple[Path, ...] = ()
    log_globs: tuple[str, ...] = DEFAULT_LOG_GLOBS
    data_dir: Path | None = None
    out_dir: Path | None = None
    assume_tz: str = AUTO
    tz: str = AUTO
    seed: int = 0
    as_of: str | None = None
    strict: bool = False
    dedupe: bool = False
    top_n: int = TIMELINE_TOP_N
    quiet: bool = False

    def __post_init__(self) -> None:
        root = self.root.expanduser().resolve()
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "log_paths", tuple(_abs(p, root) for p in self.log_paths))
        object.__setattr__(self, "log_globs", tuple(str(_abs(Path(g), root)) for g in self.log_globs))
        object.__setattr__(self, "data_dir", _abs(self.data_dir or Path("data"), root))
        object.__setattr__(self, "out_dir", _abs(self.out_dir or Path("visualizations"), root))

    # -- derived paths ----------------------------------------------------

    @property
    def csv_path(self) -> Path:
        assert self.data_dir is not None
        return self.data_dir / "pacman_history.csv"

    @property
    def meta_path(self) -> Path:
        assert self.data_dir is not None
        return self.data_dir / "pacman_history.meta.json"

    # -- timezones --------------------------------------------------------

    @property
    def display_tzinfo(self) -> tzinfo | None:
        """Timezone for hour-of-day and month bucketing, ``None`` if auto."""
        return resolve_tzinfo(self.tz)

    @property
    def assume_tzinfo(self) -> tzinfo | None:
        """Timezone for offset-less legacy lines, ``None`` if auto."""
        return resolve_tzinfo(self.assume_tz)

    def with_overrides(self, **kwargs: object) -> Config:
        return replace(self, **kwargs)  # type: ignore[arg-type]

    # -- CLI construction -------------------------------------------------

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> Config:
        """Build a config from parsed arguments, keeping the CWD-relative default."""
        return cls(
            root=Path(getattr(args, "root", ".")),
            log_paths=tuple(Path(p) for p in getattr(args, "log", ()) or ()),
            log_globs=tuple(getattr(args, "log_glob", ()) or DEFAULT_LOG_GLOBS),
            data_dir=Path(args.data) if getattr(args, "data", None) else None,
            out_dir=Path(args.out) if getattr(args, "out", None) else None,
            assume_tz=getattr(args, "assume_tz", AUTO),
            tz=getattr(args, "tz", AUTO),
            seed=getattr(args, "seed", 0),
            as_of=getattr(args, "as_of", None),
            strict=bool(getattr(args, "strict", False)),
            dedupe=bool(getattr(args, "dedupe", False)),
            top_n=getattr(args, "top_n", None) or TIMELINE_TOP_N,
            quiet=bool(getattr(args, "quiet", False)),
        )


def _abs(path: Path, root: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else (root / path)


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Arguments shared by every subcommand."""
    group = parser.add_argument_group("paths")
    group.add_argument(
        "--root",
        default=".",
        metavar="DIR",
        help="base directory for relative paths (default: current directory)",
    )
    group.add_argument(
        "--data",
        metavar="DIR",
        help="directory holding the parsed CSV and its metadata (default: ./data)",
    )
    group.add_argument(
        "--out",
        metavar="DIR",
        help="directory to write charts into (default: ./visualizations)",
    )
    group.add_argument(
        "--log",
        action="append",
        metavar="PATH",
        help="read this log instead of the discovered ones; repeatable",
    )
    group.add_argument(
        "--log-glob",
        dest="log_glob",
        action="append",
        metavar="GLOB",
        help=f"glob for log files, repeatable (default: {' '.join(DEFAULT_LOG_GLOBS)})",
    )

    tz_group = parser.add_argument_group("timezones")
    tz_group.add_argument(
        "--tz",
        default=AUTO,
        metavar="ZONE",
        help=(
            "display timezone for hour-of-day and month buckets: 'auto', 'UTC' or "
            "an IANA name like 'Europe/Berlin' (default: auto)"
        ),
    )
    tz_group.add_argument(
        "--assume-tz",
        dest="assume_tz",
        default=AUTO,
        metavar="ZONE",
        help=(
            "timezone for pre-2019 log lines that carry no offset: 'auto', 'UTC' "
            "or an IANA name (default: auto, which uses the most common offset "
            "seen in the offset-bearing lines)"
        ),
    )
    tz_group.add_argument(
        "--as-of",
        dest="as_of",
        metavar="TIMESTAMP",
        help=(
            "treat still-installed packages as installed until this time "
            "(default: the newest event in the data)"
        ),
    )

    parser.add_argument("-q", "--quiet", action="store_true", help="only report errors")
    parser.add_argument(
        "--seed", type=int, default=0, help="seed for any randomized chart styling (default: 0)"
    )


def parse_known_config(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    """Convenience for tests and notebooks that just want the defaults."""
    parser = argparse.ArgumentParser(prog="paclog")
    add_common_arguments(parser)
    return parser.parse_args(list(argv) if argv is not None else None)
