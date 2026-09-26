# paclog

An installable Python package that turns an Arch Linux `pacman.log` into a tidy
CSV, some numbers, and annotated SVG charts. One package, one CLI, a test suite,
and a committed `visualizations/` directory.

This file supersedes the pre-0.2 layout, which was three flat scripts
(`pacman_parse.py`, `pacman_visualize.py`, `pacman_timeline.py`) that each ran
their whole analysis at import time. Those are gone; see `git log` if you need
them.

## Setup

```bash
python -m venv --system-site-packages .venv   # system Python is PEP 668 managed
.venv/bin/pip install -e '.[dev]'
```

Without installing, `python -m paclog` works from the repository root.

## Pipeline

```bash
paclog parse      # /var/log/pacman.log[.*] -> data/pacman_history.csv + .meta.json
paclog stats      # headline numbers, --json for machine-readable
paclog plot       # every default chart -> visualizations/
paclog timeline   # the install-period timeline (opt-in, it is the expensive one)
paclog build      # all of the above, in order
```

`paclog list` shows the chart registry. Every path is overridable
(`--log`, `--data`, `--out`, `--root`); nothing depends on the CWD.

## Layout

```
paclog/
  cli.py        argparse subcommands, all chatter on stderr so stdout stays pipeable
  config.py     Config dataclass; every path, timezone and knob, resolved once
  model.py      Action enum, ACTION_ORDER, Event, SCHEMA_VERSION
  parsing.py    line -> Event | (None, reason). Pure, no I/O.
  loader.py     log discovery, gzip, frame construction, timezone policy
  store.py      CSV + metadata sidecar I/O
  analyze.py    aggregations. Imports no matplotlib, so it is testable without drawing.
  plots/        base.py (style) + registry.py + one module per chart family
tests/          pytest, driven by a hand-written fixtures/pacman.log
```

## Things that will waste your time if you do not know them

- **`pacman parse` takes ~6 s on a 12.8 MB / 157k-line log.** That is the regex
  pass plus the offset-inference pre-scan, not a hang. The pre-scan reads the file
  twice on purpose: it needs the modal UTC offset before it can interpret the
  offset-less lines, and those lines make up about a third of a real log.
- **A row count of 0 from `paclog parse` usually means the wrong file**, not an
  empty log. Check `parse_stats` in `data/pacman_history.meta.json`; it accounts
  for every line. `--strict` fails only on lines that *looked* like package events
  and failed, never on ordinary noise.
- **Do not trust `*.csv`-style gitignoring here.** `.gitignore` ignores `/data/`
  as a directory, because the metadata sidecar is JSON and a `*.csv` rule would
  let it get committed by accident.
- **The CSV stores UTC; charts bucket in a display timezone.** `loader.to_display_tz`
  is where that happens, and `--tz` / `--assume-tz` feed it. Comparing the raw
  `timestamp` column against local wall-clock expectations will mislead you.
- **`analyze.install_periods` treats an `installed` with no matching `removed` as
  running until `as_of`.** `as_of` defaults to `max(timestamp)` in the data, not
  `now()`, precisely so re-runs are stable. `--as-of now` opts back in.
- **Charts are byte-reproducible, and keeping them that way is load-bearing**,
  because `visualizations/*.svg` are committed. Three settings do the work:
  `svg.hashsalt` in `plots/base.py`, `metadata={"Date": None}` in `plots/base.py`'s
  `save()`, and `crc32`-derived timeline colours instead of `hash()` — which is
  salted per process for `str` and would reintroduce nondeterminism. There is a
  test for this; if a chart starts showing up in `git status` after a re-run,
  something in that list regressed.
- **Never derive chart layout or ordering from `df["action"].unique()`.** Use
  `model.ACTION_ORDER`. The original derived a subplot grid from the data's
  first-appearance order, which worked only while there happened to be exactly
  four actions; adding `downgraded` would have made it an `IndexError`.
- **The timeline plots and labels every package. Do not add a default cap.** An
  earlier revision capped it at 150 packages and a later one strided the y labels
  to ~80 of 2 808. Both were wrong and both were reverted: the cap changed what
  the chart was about, and the striding saved SVG bytes at the cost of making the
  chart unreadable, which is the only thing it is for. All 2 808 packages, all
  3 944 install periods and all 2 808 labels are drawn, which is 4.6 MB. The
  original 4.1 MB chart was not smaller because it hid nothing — it was smaller
  only because it drew one artist per *period* instead of one per *package*.
  `--top-n` remains available for a deliberate quick overview.
- **Do not shrink `visualizations/*.svg` by dropping content.** They are
  regenerated on `paclog plot`, so a size reduction is only worth having if the
  chart still says the same thing. Check the label count and the drawn-segment
  count against `analyze.install_periods` after touching `plots/timeline.py`.
- **Do not add `numpy` back as a dependency.** It was imported and never used in
  both the original script and the notebook; it is gone.

## Deps

`pandas>=2.2` and `matplotlib>=3.8`; `pytest` for the dev extra. `python>=3.10`.
Nothing installs them for you — see Setup.

## Tests

```bash
.venv/bin/python -m pytest
```

The suite runs against `tests/fixtures/pacman.log`, a ~35-line synthetic log that
covers both timestamp formats, all five actions, a multi-hop upgrade chain,
wrapped `[ALPM-SCRIPTLET]` output with no envelope, and both `[PACMAN]` and
`[PACKAGEKIT]` noise. It is hand-counted in `tests/conftest.py`; if a parser change
moves those numbers, something real changed. There is deliberately **no sample
real log** in the repository — `.gitignore` excludes `*.log` except under
`tests/fixtures/`, and no CSV is ever committed.

Tests assert on parsed values, analysis results and file bytes. They do not
compare rendered images.

## Docs

`README.org` is Emacs org-mode, not markdown (`#+title:`, `~code~`,
`[[file:...]]` links) — don't "fix" it with markdown syntax. The notebook markdown
is the narrative version of the same material; `paclog.ipynb` is written for the
author's personal history (Arch installed 03/12/2016, ~68.5k events), so its
commentary about habits does not generalize to other machines. Its code cells call
`paclog.plots` and `paclog.analyze` directly, so they cannot drift from the CLI.
