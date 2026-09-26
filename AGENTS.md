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
- **Never rank install periods by length without splitting on censoring.**
  `analyze.package_lifetime` returns every period and flags each one `censored`:
  a package still installed at `as_of` has a *lower bound*, not a lifetime. On
  the author's log 1 666 of the 3 944 periods are censored — 1 666 distinct
  packages, since no package has two open periods at once — so ranking the raw
  table put the 40 oldest survivors on top as 40 identical 3 850-day bars and
  pushed every real lifetime off the chart. Rank with
  `analyze.completed_lifetimes` (2 278 rows, median 120 days) and list the rest
  with `analyze.installed_whole_time` (331 packages). The two charts are
  `paclog plot package-lifetime` and `paclog plot installed-whole-time`.
  Do not confuse 1 666 with the 1 378 packages whose periods are *all* censored;
  the remaining 288 have both kinds, and it is the censored period that ranks.
- **`analyze.initial_install_window` is what "since the system was built" means**,
  and it is derived from the data on purpose: from the first event to the last
  event before the log's first day-long silence. The base install on the author's
  log is 21 hours and four transactions (10:53, 12:03, 13:06 for the X.org
  stack, 13:16), so any hardcoded hour count cuts the list in the wrong place,
  and comparing against the *first event* finds 73 of the 331 packages.
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
- **A month with no events in it has no column to reindex from.** July 2019 on
  the author's log is empty, so `unstack` never emits it; reindexing the grid
  back onto a `month_range` re-adds the column as all-NaN with no dtype to infer,
  it arrives as `object`, and the next `to_numpy(dtype="datetime64[ns]")` raises
  `ValueError: Could not convert object to NumPyy datetime`. `package_staleness`
  casts the reindexed frame back to the original column dtype; the test
  `test_staleness_survives_a_month_with_no_events_at_all` guards it.
- **The session gap is a parameter because the log cannot settle it.** Log
  timestamps have one-second resolution, so a transaction that straddles a second
  boundary is indistinguishable from two. `SESSION_GAP_SECONDS` is 60, and the
  author's log is 3 157 sessions at 60 s, 9 807 at 0 s, 2 720 at 300 s — which is
  why the charts that draw this put the number they used in the title. Do not
  "fix" the threshold to a rounder number without updating that title, the
  `transactions` docstring and the README together, or the chart and the prose
  will disagree about what it says. Sessions are a different unit from events and
  do not inherit the censoring rules of `install_periods`: a session is a
  contiguous run, so a package removed and reinstalled later appears in two of
  them and the counts deliberately double up.
- **Cell geometry has to be derived, not left to `add_subplot`.** The heatmap
  (`plots/staleness.py`) is one `imshow` on an explicit `add_axes(GRID_BOX)`,
  with the row pitch set to `config.INCHES_PER_PACKAGE`, the column pitch forced
  equal to it so the cells are square, and the figure sized from the two. Under
  `add_subplot`'s defaults a nominal 0.25 in pitch came out at 0.16 in, which is
  the difference between a heatmap and a vertical gradient. Three numbers are
  coupled here and must change together: `GRID_BOX`, the figure size, and
  `RASTER_DPI`. The last one is why the chart alone sets `Chart.dpi` (8) — at
  matplotlib's default 100 a 738-inch figure crams 28 pixels into every row, and
  the SVG is 3.8 MB either way. Leave `dpi=None` on every vector chart so the
  committed bytes of the older ones cannot move.
- **A 738-inch figure cannot be Agg-drawn at dpi 100.** That is over the 65 535
  pixel limit, and `fig.canvas.draw()` raises. Any test that needs a renderer on
  the heatmap has to use the fixture log, not the real one.
- **Size the calendar from the cell, and lay the bands out as one sum.**
  `plots/calendar.py` works in points and inches throughout: `CELL_POINTS = 11`
  fixes both the week pitch and the day pitch, and every margin is a named
  constant in inches. The panel geometry must be a sum of those constants with
  each block appearing exactly once. An earlier version added `panel_in` to the
  bottom offset *and* used it as the axes height, which put the first panel's top
  edge 0.17 in above the canvas and drove the caption into the 2016 label.
  `test_calendar_caption_clears_the_first_year_label` asserts the arithmetic, not
  the appearance, because the failure mode is invisible in a diff.
- **`ax.get_yaxis_transform()` kills freetype on these charts.** The blended
  x-data/y-axes transform is so anisotropic on a 700-inch canvas that
  `fig.canvas.draw()` dies with `RuntimeError: FT_Render_Glyph (ft2font_wrapper
  .cpp line 1947) failed with error 0x62: raster overflow`. For text at a data
  x and an axes-relative y, use `ax.transAxes` and feed it
  `mdates.date2num(...)` — a date axis's `get_xlim()` returns float days, not
  `Timestamp`s.
- **On a log axis, anchor bars to an edge, not to the bin midpoint.**
  `ax.bar(np.sqrt(low * high), ..., width=high - low)` measures the width in the
  transformed space, so each rectangle lands a fraction inside its own bin and
  every pair is separated by a sliver. `align="edge"` at `x=low` covers the bin
  exactly and the bars tile the axis.
- **Use the explicit `unit=` form for every `pd.Timedelta`.** `pd.Timedelta(hours=1)`
  and comparing a `Series.diff()` against a `pd.Timedelta` both go through numpy's
  *generic* timedelta unit, which pandas 2.3 and numpy 2.5 deprecate and will turn
  into an error. `pd.Timedelta(1, unit="h")` and int64-nanosecond comparisons are
  clean. Run the suite with `-W error::DeprecationWarning` when touching date
  arithmetic.
- **Overlap claims are worth asserting, because the eye is not a test.** Several
  defects here were only visible by measuring: the legend over the base-install
  annotation, the two interval labels on top of each other, the calendar's legend
  covering a decade of data. `get_renderer().get_window_extent()` on the text
  artists answers it, and `tests/test_plots.py` now has those checks so a
  regression is a failing test rather than a remark.
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

`README.md` is the documentation, and it is markdown. It was `README.org` until the
0.3 work, and the conversion was not cosmetic:

- **PyPI was the reason.** setuptools has no content type for `.org`, so the
  `readme` field had to be `content-type = "text/plain"`, which renders the raw
  markup — `#+title:`, `~code~`, `[[file:...]]` — on the package page.
- **GitHub was not a reason.** Linguist renders org-mode, so the repo page was
  already fine. The `[[file:...]]` links were the one thing org cost on GitHub:
  they render as plain text, not links, so `paclog.ipynb` and `LICENSE` were
  unreachable from the README either way.
- If you edit prose, keep the two-spaces-after-a-full-stop style; it is the
  author's org habit and predates the conversion.

The notebook markdown is the narrative version of the same material;
`paclog.ipynb` is written for the author's personal history (Arch installed
03/12/2016, ~68.5k events), so its commentary about habits does not generalize to
other machines. Its code cells call `paclog.plots` and `paclog.analyze` directly,
so they cannot drift from the CLI.
