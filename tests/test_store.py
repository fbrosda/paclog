"""CSV and metadata round-trips."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from paclog.config import Config
from paclog.model import SCHEMA_VERSION
from paclog.store import (
    Meta,
    build_meta,
    read_meta,
    read_table,
    require_current,
    write_meta,
    write_table,
)
from tests.conftest import EXPECTED_EVENTS


def test_round_trip_preserves_the_utc_dtype(tmp_path, frame):
    path = write_table(frame, tmp_path / "pacman_history.csv")
    loaded = read_table(path)
    assert isinstance(loaded["timestamp"].dtype, pd.DatetimeTZDtype)
    assert str(loaded["timestamp"].dt.tz) == "UTC"
    assert len(loaded) == len(frame)


def test_round_trip_preserves_every_value(tmp_path, frame):
    path = write_table(frame, tmp_path / "pacman_history.csv")
    loaded = read_table(path)
    columns = list(loaded.columns)
    left = loaded.sort_values(columns).reset_index(drop=True)
    right = frame.sort_values(columns).reset_index(drop=True)
    # An empty CSV cell reads back as <NA> while the in-memory frame holds None;
    # normalise so the comparison is about values, not null representation.
    for column in ("version_before", "version_after"):
        left[column] = left[column].astype(object).where(left[column].notna(), None)
        right[column] = right[column].astype(object).where(right[column].notna(), None)
    pd.testing.assert_frame_equal(left, right, check_dtype=False)


def test_missing_version_reads_back_as_null_not_empty_string(tmp_path, frame):
    path = write_table(frame, tmp_path / "h.csv")
    loaded = read_table(path)
    installed = loaded[loaded["action"] == "installed"]
    assert installed["version_before"].isna().all()
    assert not (installed["version_before"] == "").any()


def test_written_timestamps_carry_an_offset(tmp_path, frame):
    path = write_table(frame, tmp_path / "h.csv")
    text = path.read_text(encoding="utf-8").splitlines()[1]
    assert "+0000" in text


def test_write_creates_missing_directories(tmp_path, frame):
    path = write_table(frame, tmp_path / "deep" / "nested" / "h.csv")
    assert path.is_file()


def test_reading_a_missing_csv_explains_what_to_do(tmp_path):
    with pytest.raises(FileNotFoundError, match="paclog parse"):
        read_table(tmp_path / "absent.csv")


def test_legacy_naive_csv_is_still_readable(tmp_path):
    """A pre-0.2 CSV has no offset. It must load as UTC rather than explode."""
    legacy = tmp_path / "pacman_history.csv"
    legacy.write_text(
        "package,timestamp,action,version_before,version_after\n"
        "tzdata,2016-03-12 12:53:00,installed,,2016a-1\n",
        encoding="utf-8",
    )
    loaded = read_table(legacy)
    assert str(loaded["timestamp"].dt.tz) == "UTC"
    assert loaded["timestamp"].iloc[0].hour == 12


def test_empty_frame_round_trips(tmp_path):
    from paclog.loader import empty_frame

    path = write_table(empty_frame(), tmp_path / "h.csv")
    assert read_table(path).empty


# -- metadata --------------------------------------------------------------- #


def test_meta_round_trip(tmp_path):
    meta = build_meta(
        row_count=EXPECTED_EVENTS,
        sources=[{"path": "pacman.log", "bytes": 10}],
        parse_stats={"events": EXPECTED_EVENTS},
        assume_tz="+02:00 (auto)",
        display_tz="+02:00",
        as_of=pd.Timestamp("2021-01-01", tz="UTC").to_pydatetime(),
        seed=0,
        version="0.2.0",
    )
    path = write_meta(meta, tmp_path / "pacman_history.meta.json")
    loaded = read_meta(path)
    assert loaded is not None
    assert loaded.row_count == EXPECTED_EVENTS
    assert loaded.schema_version == SCHEMA_VERSION
    assert loaded.source_logs[0]["path"] == "pacman.log"


def test_meta_is_valid_json(tmp_path):
    meta = build_meta(
        row_count=1,
        sources=[],
        parse_stats={},
        assume_tz="UTC",
        display_tz="UTC",
        as_of=pd.Timestamp("2021-01-01", tz="UTC").to_pydatetime(),
        seed=0,
        version="0.2.0",
    )
    path = write_meta(meta, tmp_path / "m.json")
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == SCHEMA_VERSION


def test_corrupt_meta_is_ignored_rather_than_fatal(tmp_path):
    path = tmp_path / "pacman_history.meta.json"
    path.write_text("{not json", encoding="utf-8")
    assert read_meta(path) is None


def test_missing_meta_warns_but_does_not_fail(tmp_path):
    config = Config(data_dir=tmp_path)
    messages: list[str] = []
    assert require_current(config, messages.append) is None
    assert any("legacy" in m for m in messages)


def test_meta_from_a_future_schema_is_flagged(tmp_path):
    config = Config(data_dir=tmp_path)
    write_meta(Meta(schema_version=SCHEMA_VERSION + 99), config.meta_path)
    messages: list[str] = []
    assert require_current(config, messages.append) is not None
    assert any("Re-run" in m for m in messages)
