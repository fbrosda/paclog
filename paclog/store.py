"""Reading and writing the inter-stage artifacts.

Two files make up the cache: the CSV that everything downstream reads, and a JSON
sidecar that records where it came from. The sidecar matters because the CSV
alone cannot tell you whether it is current, which log produced it, or how many
lines the parser failed to understand.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .model import COLUMNS, SCHEMA_VERSION

#: Fixed so a re-run over unchanged input produces a byte-identical CSV.
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


class StaleData(Exception):
    """The CSV on disk was produced by an incompatible version of paclog."""


@dataclass
class Meta:
    """Provenance for one parsed CSV."""

    schema_version: int = SCHEMA_VERSION
    generated_at: str = ""
    paclog_version: str = ""
    row_count: int = 0
    source_logs: list[dict[str, Any]] = field(default_factory=list)
    parse_stats: dict[str, Any] = field(default_factory=dict)
    storage_tz: str = "UTC"
    display_tz: str = ""
    assume_tz: str = ""
    as_of: str = ""
    seed: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Meta:
        known = {f: raw[f] for f in cls.__dataclass_fields__ if f in raw}
        return cls(**known)


def write_table(frame: pd.DataFrame, path: Path) -> Path:
    """Write the event CSV, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    if not out.empty:
        out["timestamp"] = out["timestamp"].dt.strftime(TIMESTAMP_FORMAT)
    out.to_csv(path, index=False, columns=list(COLUMNS))
    return path


def read_table(path: Path, warn=lambda msg: None) -> pd.DataFrame:
    """Read the event CSV as a UTC-aware frame.

    ``utc=True`` is not optional. With offset-bearing stamps and no ``utc``, pandas
    returns ``object`` dtype and every ``.dt`` accessor downstream raises. A
    pre-0.2 naive CSV is still accepted, with a warning.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"no parsed data at {path}. Run `paclog parse` first."
        )
    frame = pd.read_csv(path, dtype={"package": "string"})
    frame["action"] = frame["action"].astype("string")
    for column in ("version_before", "version_after"):
        frame[column] = frame[column].astype("string").replace("", pd.NA)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["package"] = frame["package"].astype(str)
    frame["action"] = frame["action"].astype(str)
    return frame[list(COLUMNS)]


def write_meta(meta: Meta, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta.as_dict(), indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


def read_meta(path: Path) -> Meta | None:
    if not path.is_file():
        return None
    try:
        return Meta.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, TypeError):
        return None


def require_current(config, warn=lambda msg: None) -> Meta | None:
    """Load the sidecar and complain if the CSV predates this schema.

    A missing sidecar means a hand-made or pre-0.2 CSV. That is a warning, not an
    error: the data is still usable, we just cannot vouch for it.
    """
    meta = read_meta(config.meta_path)
    if meta is None:
        if not config.quiet:
            warn(
                f"note: no metadata at {config.meta_path}; treating the CSV as legacy "
                "input (timestamps assumed UTC)"
            )
        return None
    if meta.schema_version != SCHEMA_VERSION:
        warn(
            f"warning: {config.csv_path} was written with schema v{meta.schema_version}, "
            f"this is v{SCHEMA_VERSION}. Re-run `paclog parse`."
        )
    return meta


def build_meta(
    *,
    row_count: int,
    sources: list[dict[str, Any]],
    parse_stats: dict[str, Any],
    assume_tz: str,
    display_tz: str,
    as_of: datetime,
    seed: int,
    version: str,
) -> Meta:
    return Meta(
        generated_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        paclog_version=version,
        row_count=row_count,
        source_logs=sources,
        parse_stats=parse_stats,
        storage_tz="UTC",
        display_tz=display_tz,
        assume_tz=assume_tz,
        as_of=as_of.isoformat(),
        seed=seed,
    )
