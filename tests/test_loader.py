"""Log discovery, rotation handling, frame construction and timezone policy."""

from __future__ import annotations

import gzip
from datetime import datetime, timezone

import pandas as pd
import pytest

from paclog.config import Config, fixed_offset_tz
from paclog.loader import (
    LogNotFound,
    dedupe_events,
    discover_logs,
    frame_from_events,
    infer_assume_tz,
    open_log,
    read_events,
    resolve_as_of,
    to_display_tz,
)
from paclog.model import Action, Event
from tests.conftest import ASSUME_TZ, EXPECTED_EVENTS


def test_discovers_the_explicit_log(fixture_log):
    config = Config(log_paths=(fixture_log,))
    assert discover_logs(config) == [fixture_log]


def test_discovers_by_glob(fixture_log):
    config = Config(log_globs=(str(fixture_log),))
    assert discover_logs(config) == [fixture_log]


def test_missing_explicit_log_warns_and_is_skipped(tmp_path):
    config = Config(log_paths=(tmp_path / "nope.log",))
    with pytest.raises(LogNotFound):
        discover_logs(config, warn=lambda msg: None)


def test_no_logs_at_all_raises_with_a_hint(tmp_path):
    config = Config(log_globs=(str(tmp_path / "*.log"),))
    with pytest.raises(LogNotFound, match="--log"):
        discover_logs(config, warn=lambda msg: None)


def test_rotated_and_compressed_logs_are_picked_up(tmp_path, fixture_log):
    plain = tmp_path / "pacman.log"
    plain.write_text(fixture_log.read_text(encoding="utf-8"), encoding="utf-8")
    with gzip.open(tmp_path / "pacman.log.1.gz", "wt", encoding="utf-8") as handle:
        handle.write("[2015-01-01 00:00] [ALPM] installed old-pkg (1-1)\n")

    config = Config(log_globs=(str(tmp_path / "pacman.log*"),))
    found = discover_logs(config)
    assert {p.name for p in found} == {"pacman.log", "pacman.log.1.gz"}


def test_gzipped_logs_are_read_transparently(tmp_path, fixture_log):
    path = tmp_path / "pacman.log.1.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(fixture_log.read_text(encoding="utf-8"))
    with open_log(path) as handle:
        assert "[ALPM] installed" in handle.read()


def test_reads_every_event_from_the_fixture(fixture_log):
    config = Config(log_paths=(fixture_log,))
    events, stats, sources = read_events([fixture_log], ASSUME_TZ, config)
    assert len(events) == EXPECTED_EVENTS
    assert stats.matched == EXPECTED_EVENTS
    assert [s.path for s in sources] == [fixture_log]
    assert sources[0].lines == stats.total


def test_duplicates_are_kept_by_default(fixture_log):
    """Two identical rows are usually two real transactions, not a parsing bug.

    pacman can run the same install twice in one second, and a reinstall inside a
    ``transaction failed`` leaves an event that never took effect.
    """
    config = Config(log_paths=(fixture_log,))
    events, _, _ = read_events([fixture_log], ASSUME_TZ, config)
    assert len(events) == EXPECTED_EVENTS


def test_dedupe_is_opt_in(fixture_log):
    config = Config(log_paths=(fixture_log,), dedupe=True)
    doubled = fixture_log.read_text(encoding="utf-8")
    doubled = doubled + doubled  # every row twice
    doubled_path = fixture_log.parent / "doubled.log"
    doubled_path.write_text(doubled, encoding="utf-8")
    try:
        kept, _, _ = read_events([doubled_path], ASSUME_TZ, config, dedupe=True)
        assert len(kept) == EXPECTED_EVENTS
        unkept, _, _ = read_events([doubled_path], ASSUME_TZ, config, dedupe=False)
        assert len(unkept) == 2 * EXPECTED_EVENTS
    finally:
        doubled_path.unlink()


def test_dedupe_events_preserves_order():
    when = datetime(2020, 1, 1, tzinfo=timezone.utc)
    first = Event("a", when, Action.INSTALLED, None, "1")
    second = Event("b", when, Action.INSTALLED, None, "1")
    assert dedupe_events([first, second, first]) == [first, second]


# -- timezones -------------------------------------------------------------- #


def test_auto_assume_tz_picks_the_modal_offset(fixture_log):
    config = Config(log_paths=(fixture_log,))
    tzinfo, how = infer_assume_tz([fixture_log], config)
    assert tzinfo == fixed_offset_tz(120)
    assert "auto" in how


def test_explicit_assume_tz_wins(fixture_log):
    config = Config(log_paths=(fixture_log,), assume_tz="UTC")
    tzinfo, how = infer_assume_tz([fixture_log], config)
    assert tzinfo == timezone.utc
    assert "explicit" in how


def test_assume_tz_falls_back_to_utc_without_offsets(tmp_path):
    only_naive = tmp_path / "pacman.log"
    only_naive.write_text("[2016-03-12 12:53] [ALPM] installed a (1-1)\n", encoding="utf-8")
    config = Config(log_paths=(only_naive,))
    tzinfo, how = infer_assume_tz([only_naive], config)
    assert tzinfo == timezone.utc
    assert "no offset-bearing" in how


def test_display_tz_conversion_preserves_the_instant(frame):
    converted = to_display_tz(frame, fixed_offset_tz(120))
    assert converted["timestamp"].dt.tz == fixed_offset_tz(120)
    assert (converted["timestamp"].astype("int64") == frame["timestamp"].astype("int64")).all()


# -- frame construction ----------------------------------------------------- #


def test_frame_is_utc_aware_with_the_documented_columns(frame):
    assert isinstance(frame["timestamp"].dtype, pd.DatetimeTZDtype)
    assert str(frame["timestamp"].dt.tz) == "UTC"
    assert list(frame.columns) == ["package", "timestamp", "action", "version_before", "version_after"]


def test_frame_is_sorted_deterministically(frame):
    stamps = frame["timestamp"].tolist()
    assert stamps == sorted(stamps)


def test_empty_input_yields_a_usable_frame():
    empty = frame_from_events([])
    assert empty.empty
    assert isinstance(empty["timestamp"].dtype, pd.DatetimeTZDtype)


def test_as_of_defaults_to_the_newest_event(frame):
    assert resolve_as_of(Config(), frame) == frame["timestamp"].max().to_pydatetime()


def test_as_of_now_is_opt_in(frame):
    before = datetime.now(timezone.utc)
    resolved = resolve_as_of(Config(as_of="now"), frame)
    assert resolved >= before.replace(microsecond=0)


def test_explicit_as_of_is_honoured(frame):
    resolved = resolve_as_of(Config(as_of="2020-06-01T12:00:00+00:00"), frame)
    assert resolved == datetime(2020, 6, 1, 12, 0, tzinfo=timezone.utc)
