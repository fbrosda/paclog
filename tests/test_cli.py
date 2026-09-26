"""End-to-end CLI behaviour."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paclog.cli import main
from tests.conftest import EXPECTED_EVENTS


def run(*argv: str) -> int:
    return main(list(argv))


@pytest.fixture
def workspace(tmp_path: Path, fixture_log: Path) -> Path:
    """A directory with the log available and nothing generated yet."""
    return tmp_path


def parse(workspace: Path, fixture_log: Path, *extra: str) -> int:
    return run("parse", "--root", str(workspace), "--log", str(fixture_log), "-q", *extra)


def test_parse_creates_data_dir_and_csv(workspace, fixture_log):
    assert parse(workspace, fixture_log) == 0
    assert (workspace / "data" / "pacman_history.csv").is_file()
    assert (workspace / "data" / "pacman_history.meta.json").is_file()


def test_parse_records_provenance(workspace, fixture_log):
    parse(workspace, fixture_log)
    meta = json.loads((workspace / "data" / "pacman_history.meta.json").read_text())
    assert meta["row_count"] == EXPECTED_EVENTS
    assert meta["source_logs"][0]["path"].endswith("pacman.log")
    assert meta["schema_version"] >= 2
    assert meta["parse_stats"]["events"] == EXPECTED_EVENTS
    assert meta["storage_tz"] == "UTC"
    assert "assume_tz" in meta and "display_tz" in meta


def test_parse_is_idempotent(workspace, fixture_log):
    parse(workspace, fixture_log)
    first = (workspace / "data" / "pacman_history.csv").read_bytes()
    parse(workspace, fixture_log)
    assert (workspace / "data" / "pacman_history.csv").read_bytes() == first


def test_strict_passes_on_a_healthy_log(workspace, fixture_log):
    """Wrapped scriptlet output must not be mistaken for a defect."""
    assert parse(workspace, fixture_log, "--strict") == 0


def test_strict_fails_on_a_corrupt_package_line(tmp_path, fixture_log):
    broken = tmp_path / "broken.log"
    broken.write_text(
        fixture_log.read_text(encoding="utf-8")
        + "[2016-03-12 12:53] [ALPM] upgraded tzdata (no-arrow-here)\n",
        encoding="utf-8",
    )
    assert run("parse", "--root", str(tmp_path), "--log", str(broken), "-q", "--strict") == 3


def test_parse_without_a_log_fails_cleanly(tmp_path, capsys):
    status = run("parse", "--root", str(tmp_path), "--log", str(tmp_path / "absent.log"), "-q")
    assert status == 2
    assert "error" in capsys.readouterr().err


def test_plot_before_parse_explains_itself(tmp_path, capsys):
    assert run("plot", "--root", str(tmp_path), "-q") == 2
    assert "paclog parse" in capsys.readouterr().err


def test_plot_creates_the_output_dir(workspace, fixture_log):
    parse(workspace, fixture_log)
    assert run("plot", "--root", str(workspace), "-q") == 0
    out = workspace / "visualizations"
    assert out.is_dir()
    assert (out / "events_per_hour.svg").is_file()
    assert (out / "events_per_month.svg").is_file()
    assert (out / "top_packages.svg").is_file()
    # opt-in only
    assert not (out / "timeline.svg").exists()


def test_plot_a_single_chart(workspace, fixture_log):
    parse(workspace, fixture_log)
    assert run("plot", "action-distribution", "--root", str(workspace), "-q") == 0
    files = sorted(p.name for p in (workspace / "visualizations").iterdir())
    assert files == ["action_distribution.svg"]


def test_plot_rejects_an_unknown_chart(workspace, fixture_log, capsys):
    parse(workspace, fixture_log)
    assert run("plot", "nope", "--root", str(workspace), "-q") == 2
    assert "unknown chart" in capsys.readouterr().err


def test_timeline_subcommand_writes_the_timeline(workspace, fixture_log):
    parse(workspace, fixture_log)
    assert run("timeline", "--root", str(workspace), "-q") == 0
    assert (workspace / "visualizations" / "timeline.svg").is_file()


def test_full_pipeline_is_reproducible_across_invocations(workspace, fixture_log):
    """Two `build` runs must leave byte-identical charts behind."""
    assert run("build", "--root", str(workspace), "--log", str(fixture_log), "-q") == 0
    first = {p.name: p.read_bytes() for p in (workspace / "visualizations").iterdir()}
    assert run("build", "--root", str(workspace), "--log", str(fixture_log), "-q") == 0
    second = {p.name: p.read_bytes() for p in (workspace / "visualizations").iterdir()}
    assert first == second
    assert "timeline.svg" in first


def test_stats_human_output(workspace, fixture_log, capsys):
    parse(workspace, fixture_log)
    assert run("stats", "--root", str(workspace), "-q") == 0
    out = capsys.readouterr().out
    assert "events" in out
    assert "downgraded" in out
    assert f"{EXPECTED_EVENTS}" in out


def test_stats_json_output_is_parseable(workspace, fixture_log, capsys):
    parse(workspace, fixture_log)
    assert run("stats", "--root", str(workspace), "-q", "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["events"] == EXPECTED_EVENTS
    assert payload["actions"]["downgraded"] == 1


def test_stats_honours_date_filters(workspace, fixture_log, capsys):
    parse(workspace, fixture_log)
    run("stats", "--root", str(workspace), "-q", "--json", "--since", "2020-01-01")
    payload = json.loads(capsys.readouterr().out)
    assert 0 < payload["events"] < EXPECTED_EVENTS


def test_list_shows_the_registry(capsys):
    assert run("list") == 0
    out = capsys.readouterr().out
    assert "events-per-hour" in out
    assert "opt-in" in out  # the timeline is marked as such


def test_bare_invocation_prints_help(capsys):
    assert run() == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        run("--version")
    assert exc.value.code == 0
    assert "paclog" in capsys.readouterr().out


def test_bad_timezone_is_reported(tmp_path, fixture_log, capsys):
    status = run("parse", "--root", str(tmp_path), "--log", str(fixture_log), "--tz", "Mars/Olympus")
    assert status == 2
    assert "unknown timezone" in capsys.readouterr().err
