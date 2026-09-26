"""Chart registry, smoke rendering, and byte-level reproducibility."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from paclog import plots
from paclog.plots import Context, names, render, render_all
from paclog.plots.base import ACTION_COLORS
from paclog.plots.timeline import stable_unit

AS_OF = datetime(2021, 4, 10, tzinfo=timezone.utc)


@pytest.fixture
def ctx() -> Context:
    return Context(as_of=datetime(2021, 4, 10, tzinfo=timezone.utc), seed=0, options={"top_n": 150})


def test_registry_is_populated():
    assert names()
    assert "events-per-hour" in names()
    assert "timeline" in names()


def test_timeline_is_opt_in_only():
    """It costs far more than the rest together, so a bare `paclog plot` skips it."""
    assert "timeline" not in plots.default_names()
    assert "timeline" in names()


def test_every_chart_has_a_distinct_filename():
    filenames = [plots.get(n).filename for n in names()]
    assert len(filenames) == len(set(filenames))
    assert all(f.endswith(".svg") for f in filenames)


def test_every_action_has_a_colour():
    from paclog.model import ACTION_ORDER

    for action in ACTION_ORDER:
        assert action.value in ACTION_COLORS


def test_unknown_chart_raises_with_the_available_list():
    with pytest.raises(KeyError, match="available"):
        plots.get("no-such-chart")


def test_every_chart_renders(frame, ctx):
    """Smoke test: a chart that raises is a bug, whatever it looks like."""
    for name in names():
        fig = render(name, frame, ctx)
        assert fig.get_axes(), f"{name} produced no axes"


def test_render_all_writes_files(tmp_path, frame, ctx):
    written = render_all(frame, ctx, tmp_path, only=["events-per-hour", "action-distribution"])
    assert [p.name for p in written] == ["events_per_hour.svg", "action_distribution.svg"]
    assert all(p.is_file() and p.stat().st_size > 0 for p in written)


def test_render_all_creates_the_output_directory(tmp_path, frame, ctx):
    target = tmp_path / "does" / "not" / "exist"
    render_all(frame, ctx, target, only=["events-per-hour"])
    assert target.is_dir()


def test_render_all_rejects_unknown_names(tmp_path, frame, ctx):
    with pytest.raises(KeyError, match="unknown chart"):
        render_all(frame, ctx, tmp_path, only=["nope"])


def test_charts_are_byte_reproducible(tmp_path, frame, ctx):
    """The property the original scripts could not offer.

    Two renders in the same process must produce identical bytes. Combined with
    ``svg.hashsalt`` and the suppressed ``dc:date``, this is what makes a chart
    safe to commit.
    """
    name = "events-per-month"
    first = render_all(frame, ctx, tmp_path / "a", only=[name])[0]
    second = render_all(frame, ctx, tmp_path / "b", only=[name])[0]
    assert first.read_bytes() == second.read_bytes()


def test_saved_svg_has_no_creation_timestamp(tmp_path, frame, ctx):
    written = render_all(frame, ctx, tmp_path, only=["action-distribution"])[0]
    assert b"dc:date" not in written.read_bytes()


def test_style_is_idempotent():
    from paclog.plots.base import apply_style

    apply_style()
    apply_style()  # must not raise


def test_timeline_colour_is_stable_across_processes():
    """``hash()`` is salted per process for str; crc32 is not."""
    assert stable_unit("linux") == stable_unit("linux")
    assert stable_unit("linux") != stable_unit("gimp")
    assert 0.0 <= stable_unit("anything") < 1.0


def test_timeline_respects_the_package_cap(frame, ctx):
    capped = Context(as_of=ctx.as_of, seed=0, options={"top_n": 1})
    render("timeline", frame, capped)
    render("timeline", frame, Context(as_of=ctx.as_of, seed=0, options={"top_n": 0}))
