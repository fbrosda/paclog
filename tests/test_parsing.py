"""Parser behaviour: formats, versions, and the reasons lines get skipped."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from paclog.model import Action
from paclog.parsing import (
    ARROW,
    ParseStats,
    SkipReason,
    infer_offset,
    iter_events,
    parse_line,
    parse_timestamp,
    split_versions,
)
from tests.conftest import ASSUME_TZ, EXPECTED_COUNTS, EXPECTED_EVENTS


def reason_of(line: str):
    return parse_line(line, ASSUME_TZ).reason


# -- timestamps ------------------------------------------------------------- #


def test_parses_offset_less_legacy_stamp_with_the_assumed_zone():
    event, reason = parse_line(
        "[2016-03-12 12:53] [ALPM] installed tzdata (2016a-1)", ASSUME_TZ
    )
    assert reason is None
    # 12:53 local at +02:00 is 10:53 UTC.
    assert event.timestamp == datetime(2016, 3, 12, 10, 53, tzinfo=timezone.utc)


def test_parses_iso_stamp_and_keeps_its_offset_before_normalising():
    event, reason = parse_line(
        "[2019-10-27T16:17:01+0100] [ALPM] upgraded tzdata (1 -> 2)", ASSUME_TZ
    )
    assert reason is None
    assert event.timestamp == datetime(2019, 10, 27, 15, 17, 1, tzinfo=timezone.utc)


def test_stored_timestamps_are_always_utc(events):
    assert all(event.timestamp.tzinfo == timezone.utc for event in events)


def test_assume_tz_actually_moves_naive_timestamps():
    naive = "[2016-03-12 12:53] [ALPM] installed tzdata (2016a-1)"
    as_utc, _ = parse_line(naive, timezone.utc)
    as_berlin, _ = parse_line(naive, ASSUME_TZ)
    assert (as_utc.timestamp - as_berlin.timestamp).total_seconds() == 7200


def test_parse_timestamp_rejects_nonsense():
    with pytest.raises(ValueError):
        parse_timestamp("not a timestamp", ASSUME_TZ)


@pytest.mark.parametrize(
    "raw,minutes",
    [("2019-10-27T16:17:01+0100", 60), ("2019-10-27T16:17:01-0500", -300), ("2016-03-12 12:53", None)],
)
def test_infer_offset(raw, minutes):
    assert infer_offset(raw) == minutes


# -- version splitting ------------------------------------------------------ #


def test_upgrade_splits_before_and_after():
    assert split_versions(Action.UPGRADED, "1.0-1 -> 2.0-1") == ("1.0-1", "2.0-1")


def test_multi_hop_upgrade_collapses_to_the_endpoints():
    assert split_versions(Action.UPGRADED, "1.0-1 -> 1.5-1 -> 2.0-1") == ("1.0-1", "2.0-1")


def test_downgrade_uses_the_same_shape_as_upgrade():
    assert split_versions(Action.DOWNGRADED, "2.12.3-1 -> 2.12.1-4") == ("2.12.3-1", "2.12.1-4")


def test_install_has_no_before_version():
    assert split_versions(Action.INSTALLED, "1.0-1") == (None, "1.0-1")


def test_remove_has_no_after_version():
    assert split_versions(Action.REMOVED, "1.0-1") == ("1.0-1", None)


def test_reinstall_keeps_one_version_on_both_sides():
    assert split_versions(Action.REINSTALLED, "1.0-1") == ("1.0-1", "1.0-1")


def test_upgrade_without_an_arrow_is_malformed():
    assert split_versions(Action.UPGRADED, "1.0-1") is None


def test_multi_hop_chain_is_counted(stats: ParseStats):
    assert stats.multi_hop == 1


# -- actions ---------------------------------------------------------------- #


def test_all_five_actions_are_recognised(events):
    assert {event.action for event in events} == set(Action)


def test_downgrade_is_not_dropped(events):
    """The original regex forgot this verb and lost every one of these events."""
    downgrades = [e for e in events if e.action is Action.DOWNGRADED]
    assert len(downgrades) == 1
    assert downgrades[0].package == "st"
    assert downgrades[0].version_before == "1.0.2-1"
    assert downgrades[0].version_after == "1.0.1-1"


def test_event_count_and_action_breakdown(events):
    assert len(events) == EXPECTED_EVENTS
    counts = {action.value: 0 for action in Action}
    for event in events:
        counts[event.action.value] += 1
    assert counts == EXPECTED_COUNTS


# -- skip accounting -------------------------------------------------------- #


@pytest.mark.parametrize(
    "line,expected",
    [
        ("[2016-03-12 12:37] [PACMAN] Running 'pacman -Syu'", SkipReason.NOT_ALPM),
        ("[2016-03-12 12:54] [ALPM-SCRIPTLET] Created symlink", SkipReason.NOT_ALPM),
        ("[2021-04-09T07:42:23+0200] [PACKAGEKIT] synchronizing", SkipReason.NOT_ALPM),
        ("[2016-03-12 12:53] [ALPM] transaction started", SkipReason.NOT_PACKAGE_ACTION),
        ("[2016-03-12 12:54] [ALPM] running 'foo.hook'...", SkipReason.NOT_PACKAGE_ACTION),
        ("[2016-03-12 13:10] [ALPM] warning: could not open", SkipReason.NOT_PACKAGE_ACTION),
        # A wrapped scriptlet body with no timestamp at all: noise, not a defect.
        ("filesystem: 750  package: 755", SkipReason.NO_ENVELOPE),
        ("", SkipReason.NO_ENVELOPE),
    ],
)
def test_noise_is_classified_not_swallowed(line, expected):
    assert reason_of(line) is expected


def test_ordinary_noise_is_not_a_structural_problem(stats: ParseStats):
    """``--strict`` must stay usable on a real log.

    Blank and scriptlet-wrapped lines have no envelope and no package intent, so
    they must not count as structural failures.
    """
    assert stats.structural_problems == 0
    assert stats.skipped[SkipReason.NO_ENVELOPE] > 0
    assert stats.skipped[SkipReason.NOT_ALPM] > 0
    assert stats.skipped[SkipReason.NOT_PACKAGE_ACTION] > 0


def test_broken_package_line_is_structural():
    line = "[2016-03-12 12:53] [ALPM] installed tzdata"
    assert reason_of(line) is SkipReason.MALFORMED


def test_unparseable_timestamp_is_structural():
    line = "[the twelfth of March] [ALPM] installed tzdata (2016a-1)"
    assert reason_of(line) is SkipReason.BAD_TIMESTAMP


def test_upgrade_without_arrow_is_structural():
    line = "[2016-03-12 12:53] [ALPM] upgraded tzdata (2016a-1)"
    assert reason_of(line) is SkipReason.BAD_VERSION


def test_stats_totals_add_up(stats: ParseStats):
    assert stats.matched == EXPECTED_EVENTS
    assert stats.total == stats.matched + sum(stats.skipped.values())


def test_stats_serialises(stats: ParseStats):
    payload = stats.as_dict()
    assert payload["events"] == EXPECTED_EVENTS
    assert payload["multi_hop_version_chains"] == 1
    assert "not_alpm" in payload["skipped"]


def test_stats_records_an_example_per_reason(stats: ParseStats):
    for reason, examples in stats.examples.items():
        assert examples, f"no example recorded for {reason}"
        assert all(isinstance(e, str) for e in examples)


def test_iter_events_without_stats_does_not_explode(fixture_log):
    with fixture_log.open(encoding="utf-8") as handle:
        assert len(list(iter_events(handle, ASSUME_TZ))) == EXPECTED_EVENTS


def test_arrow_constant_is_the_one_the_parser_expects():
    assert ARROW == " -> "
