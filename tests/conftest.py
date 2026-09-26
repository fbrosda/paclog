"""Shared fixtures.

The committed ``fixtures/pacman.log`` is a hand-written, tiny log that covers
every line shape the parser has to deal with -- both timestamp formats, all five
actions, a multi-hop upgrade chain, and the various kinds of noise a real log
contains. It exists because the repository ignores ``*.log``, so there was
otherwise no way to test anything.
"""

from __future__ import annotations

from datetime import timezone
from pathlib import Path

import pytest

from paclog.config import fixed_offset_tz
from paclog.loader import frame_from_events
from paclog.parsing import ParseStats, iter_events

FIXTURE = Path(__file__).parent / "fixtures" / "pacman.log"

#: The fixture's offset-bearing lines use +0100 and +0200, so this is the modal
#: offset a run with ``--assume-tz auto`` would pick.
ASSUME_TZ = fixed_offset_tz(120)

#: Hand-counted from the fixture; if a parser change breaks these, it broke
#: something real.
EXPECTED_EVENTS = 14
EXPECTED_COUNTS = {
    "installed": 4,
    "upgraded": 5,
    "downgraded": 1,
    "removed": 2,
    "reinstalled": 2,
}


@pytest.fixture
def fixture_log() -> Path:
    assert FIXTURE.is_file(), f"missing test fixture: {FIXTURE}"
    return FIXTURE


@pytest.fixture
def events(fixture_log: Path):
    with fixture_log.open(encoding="utf-8") as handle:
        return list(iter_events(handle, ASSUME_TZ))


@pytest.fixture
def frame(events):
    return frame_from_events(events)


@pytest.fixture
def stats(fixture_log: Path) -> ParseStats:
    collected = ParseStats()
    with fixture_log.open(encoding="utf-8") as handle:
        list(iter_events(handle, ASSUME_TZ, collected))
    return collected


@pytest.fixture
def utc():
    return timezone.utc
