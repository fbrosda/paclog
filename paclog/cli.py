"""The ``paclog`` command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import timezone as dt_timezone
from pathlib import Path

import pandas as pd

from . import __version__, analyze, loader, store
from .config import AUTO, Config, ConfigError, add_common_arguments, format_offset, resolve_tzinfo
from .loader import LogNotFound
from .parsing import STRUCTURAL_REASONS
from .plots import Context, default_names, describe, names, render_all


class Reporter:
    """All human-facing chatter goes to stderr, so stdout stays pipeable."""

    def __init__(self, quiet: bool = False) -> None:
        self.quiet = quiet

    def info(self, message: str) -> None:
        if not self.quiet:
            print(message, file=sys.stderr)

    def warn(self, message: str) -> None:
        print(f"paclog: {message}", file=sys.stderr)

    def error(self, message: str) -> None:
        print(f"paclog: error: {message}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #


def load_frame(config: Config, reporter: Reporter) -> pd.DataFrame:
    """Read the CSV and put its timestamps into the display timezone."""
    store.require_current(config, reporter.warn)
    frame = store.read_table(config.csv_path, reporter.warn)
    loader.assert_utc(frame)
    tz, how = _resolve_display_tz(config, reporter)
    reporter.info(f"display timezone: {tz_name(tz)} ({how})")
    return loader.to_display_tz(frame, tz)


def _resolve_display_tz(config: Config, reporter: Reporter):
    """Honour ``--tz``, else reuse whatever the parse run resolved."""
    configured = config.display_tzinfo
    if configured is not None:
        return configured, f"from --tz {config.tz}"

    meta = store.read_meta(config.meta_path)
    if meta is not None and meta.display_tz:
        try:
            resolved = resolve_tzinfo(meta.display_tz)
        except ConfigError:
            resolved = None
        if resolved is not None:
            return resolved, f"recorded in {config.meta_path.name} as {meta.display_tz}"

    assumed = config.assume_tzinfo
    if assumed is not None:
        return assumed, f"same as --assume-tz {config.assume_tz}"
    return dt_timezone.utc, "default, because the CSV has no recorded timezone"


def tz_name(tz) -> str:
    return format_offset(tz) if not hasattr(tz, "key") else str(tz)


def context_for(config: Config, frame: pd.DataFrame) -> Context:
    return Context(
        as_of=loader.resolve_as_of(config, frame),
        seed=config.seed,
        options={"top_n": config.top_n},
    )


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #


def cmd_parse(args: argparse.Namespace) -> int:
    reporter = Reporter(args.quiet)
    config = Config.from_args(args)

    try:
        paths = loader.discover_logs(config, reporter.warn)
        assume_tz, assume_how = loader.infer_assume_tz(paths, config)
        display_tz = config.display_tzinfo or assume_tz
        events, stats, sources = loader.read_events(
            paths, assume_tz, config, reporter.warn, dedupe=config.dedupe
        )
    except LogNotFound as exc:
        reporter.error(str(exc))
        return 2

    reporter.info(
        f"read {len(paths)} log file(s); assume-tz {format_offset(assume_tz)} ({assume_how})"
    )
    reporter.info(f"parsed {stats.summary()}")

    if stats.structural_problems and config.strict:
        for reason in sorted(STRUCTURAL_REASONS, key=lambda r: r.value):
            for example in stats.examples.get(reason, []):
                reporter.error(f"{reason.value}: {example}")
        reporter.error(
            f"--strict: {stats.structural_problems} line(s) looked like package events "
            "but could not be parsed"
        )
        return 3

    if not events:
        reporter.error("no package events found; is this a pacman log?")
        return 4

    frame = loader.frame_from_events(events)
    store.write_table(frame, config.csv_path)

    as_of = loader.resolve_as_of(config, frame)
    meta = store.build_meta(
        row_count=len(frame),
        sources=[s.as_dict(config.root) for s in sources],
        parse_stats=stats.as_dict(),
        assume_tz=f"{format_offset(assume_tz)} ({assume_how})",
        display_tz=format_offset(display_tz),
        as_of=as_of,
        seed=config.seed,
        version=__version__,
    )
    store.write_meta(meta, config.meta_path)

    reporter.info(f"wrote {len(frame)} rows to {config.csv_path}")
    reporter.info(f"wrote metadata to {config.meta_path}")

    skipped = stats.skipped.get
    for reason, count in stats.skipped.most_common():
        if count:
            reporter.info(f"  skipped {count} {reason.value}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    reporter = Reporter(args.quiet)
    config = Config.from_args(args)
    try:
        frame = load_frame(config, reporter)
    except (FileNotFoundError, LogNotFound) as exc:
        reporter.error(str(exc))
        return 2

    if args.since or args.until:
        frame = _filter_dates(frame, args.since, args.until)
        reporter.info(f"filtered to {len(frame)} events")

    as_of = loader.resolve_as_of(config, frame)
    payload = analyze.summary(frame, as_of)

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    actions = payload["actions"]
    total = sum(actions.values()) or 1
    biggest = max(actions.values()) if actions else 1
    print(f"events            {payload['events']}")
    print(f"packages          {payload['packages']}")
    print(f"first event       {payload['first_event']}")
    print(f"last event        {payload['last_event']}")
    print(f"currently held    {payload.get('currently_installed', 'n/a')}")
    print("actions")
    for name, count in actions.items():
        # Scaled against the largest action, not the total: upgrades are ~90% of
        # all events, so scaling to the total renders every other row as one cell.
        bar = "#" * max(1, round(count / biggest * 28)) if biggest else ""
        print(f"  {name:<13} {count:>8}  {count / total * 100:5.1f}%  {bar}")
    interval = payload.get("upgrade_interval_days")
    if interval:
        print(
            f"upgrade interval   median {interval['median']}d, mean {interval['mean']}d"
        )
    busiest = analyze.busiest_hour(frame)
    if busiest is not None:
        print(f"busiest hour      {busiest:02d}:00")
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    reporter = Reporter(args.quiet)
    config = Config.from_args(args)
    try:
        frame = load_frame(config, reporter)
    except (FileNotFoundError, LogNotFound) as exc:
        reporter.error(str(exc))
        return 2

    # `build` and `timeline` reuse this command without declaring the positional.
    selected = getattr(args, "charts", None) or None
    if selected:
        unknown = [c for c in selected if c not in names()]
        if unknown:
            reporter.error(f"unknown chart(s): {', '.join(unknown)}")
            reporter.error(describe())
            return 2

    ctx = context_for(config, frame)
    reporter.info(f"as-of {ctx.as_of.isoformat()}")
    try:
        written = render_all(frame, ctx, config.out_dir, selected)
    except KeyError as exc:
        reporter.error(str(exc))
        return 2
    for path in written:
        reporter.info(f"wrote {path} ({path.stat().st_size // 1024} KiB)")
    return 0


def cmd_timeline(args: argparse.Namespace) -> int:
    args.charts = ["timeline"]
    return cmd_plot(args)


def cmd_build(args: argparse.Namespace) -> int:
    """parse, then every default chart, then the timeline.

    The opt-in charts other than the timeline stay opt-in: ``build`` is what
    someone runs to refresh the committed output, and a 3.8 MB heatmap nobody
    asked for should not turn up in it.
    """
    status = cmd_parse(args)
    if status != 0:
        return status
    status = cmd_plot(args)
    if status != 0:
        return status
    args.charts = ["timeline"]
    return cmd_plot(args)


def cmd_list(args: argparse.Namespace) -> int:
    print(describe())
    return 0


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _filter_dates(frame: pd.DataFrame, since: str | None, until: str | None) -> pd.DataFrame:
    tz = frame["timestamp"].dt.tz
    start = pd.Timestamp(since).tz_localize(tz) if since and pd.Timestamp(since).tzinfo is None else (
        pd.Timestamp(since) if since else None
    )
    end = pd.Timestamp(until).tz_localize(tz) if until and pd.Timestamp(until).tzinfo is None else (
        pd.Timestamp(until) if until else None
    )
    mask = pd.Series(True, index=frame.index)
    if start is not None:
        mask &= frame["timestamp"] >= start
    if end is not None:
        mask &= frame["timestamp"] <= end
    return frame[mask].reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paclog",
        description="Analytics and visualizations for Arch Linux pacman logs.",
    )
    parser.add_argument("--version", action="version", version=f"paclog {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p_parse = sub.add_parser("parse", help="read pacman.log into a tidy CSV")
    add_common_arguments(p_parse)
    p_parse.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero if any line looked like a package event but failed to parse",
    )
    p_parse.add_argument(
        "--dedupe",
        action="store_true",
        help=(
            "drop exactly duplicated rows; only use when --log points at overlapping "
            "rotations, since two identical rows are usually two real transactions"
        ),
    )
    p_parse.set_defaults(func=cmd_parse)

    p_stats = sub.add_parser("stats", help="print headline numbers")
    add_common_arguments(p_stats)
    p_stats.add_argument("--since", metavar="TIMESTAMP", help="ignore events before this time")
    p_stats.add_argument("--until", metavar="TIMESTAMP", help="ignore events after this time")
    p_stats.add_argument("--json", action="store_true", help="machine-readable output")
    p_stats.set_defaults(func=cmd_stats)

    p_plot = sub.add_parser("plot", help="render charts from the parsed CSV")
    add_common_arguments(p_plot)
    # Built from the registry rather than written out, so adding a third opt-in
    # chart cannot leave the help claiming there is only one.
    opt_in = [name for name in names() if name not in set(default_names())]
    p_plot.add_argument(
        "charts",
        nargs="*",
        help=f"chart names to render (default: all except {', '.join(opt_in)})",
    )
    p_plot.add_argument(
        "--top-n",
        dest="top_n",
        type=int,
        default=None,
        help=(
            "cap the number of packages drawn in the timeline; by default every "
            "package is drawn, so this only makes the chart smaller on purpose"
        ),
    )
    p_plot.set_defaults(func=cmd_plot)

    p_timeline = sub.add_parser("timeline", help="render the install-period timeline")
    add_common_arguments(p_timeline)
    p_timeline.set_defaults(func=cmd_timeline)

    p_build = sub.add_parser(
        "build",
        help=(
            "parse, render every default chart, then the timeline; the other "
            "opt-in charts stay opt-in"
        ),
    )
    add_common_arguments(p_build)
    p_build.add_argument("--strict", action="store_true", help="see `paclog parse --strict`")
    p_build.add_argument("--dedupe", action="store_true", help="see `paclog parse --dedupe`")
    p_build.add_argument(
        "--top-n",
        dest="top_n",
        type=int,
        default=None,
        help="timeline package cap; by default every package is drawn",
    )
    p_build.set_defaults(func=cmd_build)

    p_list = sub.add_parser("list", help="list available charts")
    p_list.set_defaults(func=cmd_list, quiet=False)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    try:
        return int(args.func(args))
    except ConfigError as exc:
        print(f"paclog: error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        print("paclog: interrupted", file=sys.stderr)
        return 130
    except BrokenPipeError:  # pragma: no cover
        return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
