# AGENTS.md — Guide for AI Agents Working in This Repository

This file orients AI coding agents (Claude Code, OpenCode, Copilot agents, …)
working in the `dmd_simulator` project. Read it before making changes.

## Project Overview

**dmdsim — "DMD Playground"**: an interactive Bokeh web application for
building synthetic modal signals, decomposing them with Dynamic Mode
Decomposition (DMD), and visually exploring/editing every DMD quantity
(eigenvalues, mode shapes, amplitudes, dynamics, reconstruction). The target
users are students who just heard the DMD lecture in the ML-CFD course
(TU Dresden, lecture 5 "Analysis of coherent structures") — the app must stay
**beginner-friendly** and match the lecture's terminology.

- **DMD engine:** flowTorch (the local repository `../flowtorch`, installed
  editable) — it is the *only* place DMD math happens (`dmdsim/core/engine.py`
  is the single wrapper).
- **GUI:** Bokeh 3.10 server application (`dmdsim/app/`).
- **Docs:** `README.md` (usage), `PLAN.md` (untracked roadmap),
  `documentation/` (internal docs: `architecture.md`, `modules.md`, `steps.md`
  — keep them updated with every step), `run/reports/` (untracked review
  reports), `ideas_for_review.txt` (user's idea list — see Workflow below).

## Environment and Commands

- **Everything runs in the project venv `.venv/`** (Python 3.12). Never use
  global tools. Key binaries: `.venv/bin/pytest`, `.venv/bin/dmdsim-app`,
  `.venv/bin/bokeh`.
- Tests: `.venv/bin/pytest` (15 tests; `testpaths = ["tests"]` is configured).
- Run the app: `.venv/bin/dmdsim-app` (serves on port 5006, opens a browser,
  stops itself ~2 s after the last session disconnects).
- flowTorch is installed **editable with `--no-deps`** from `../flowtorch`; the
  minimal dependency set is pinned in `requirements.txt` (CPU-only torch via
  the PyTorch wheel index).

## Architecture (see `documentation/architecture.md` for details)

```
src/dmdsim/
├── signals/generator.py   # synthetic modal line fields + generate_modes()
├── core/engine.py         # fit_dmd(): numpy → torch float64 → flowTorch → numpy
├── core/quantities.py     # DMDResult: quantities, properties, sort(), pairs
├── core/synth.py          # synthesize() (reconstruction) + mirror_conjugates()
└── app/                   # thin Bokeh layer
    ├── figures.py         # figure/data-source builders (no logic)
    ├── styling.py         # theme system (light/dark), palettes, CSS builders
    └── main.py            # DMDPlayground: widgets, callbacks, layout, entry points
```

Dependency direction: `app → core → signals → (flowTorch, numpy)`. The core is
GUI-independent and numpy-only; keep it that way.

## Project-Specific Rules and Pitfalls

### DMD / numerics
- **flowTorch is the only DMD engine.** Never implement DMD math elsewhere.
- **float64 through fits:** convert snapshot matrices to torch float64 before
  calling flowTorch (it preserves the input dtype; its default is float32).
- **`DMDResult.pairs` is stateful**: pair detection runs on the *current*
  values — cache it right after a fit and reuse the cached list while editing
  (editing breaks detection).
- **Sorting** uses flowTorch semantics: amplitude |b|, integral contribution
  (Kou et al. 2017 numbers captured from flowTorch), frequency |f|. Sorting
  only reorders; the reconstruction is invariant.
- **"1D" means a spatial line field** (traveling waves on x ∈ [0, 1]), not a
  scalar time series — plain DMD of a scalar series is rank-limited.
- Unit-norm mode shapes: fitted `|b|` values scale with the grid (a constant
  field of 2.0 on 64 points gives |b| = 16). Do not "fix" this.

### GUI (Bokeh 3.10) — verified limitations and patterns
- **`doc.clear()` is forbidden** — it wipes session callbacks and silently
  breaks the client. Theme switches restyle in place and regenerate the stored
  stylesheets' `css` strings; never rebuild roots on live sessions.
- **`GlobalInlineStyleSheet` is injected once** — updating its `css` later does
  NOT re-theme the page. Recreate the stylesheet object and re-attach it to the
  layout root on every theme switch.
- **Widgets theme via CSS custom properties** (`--background-color`,
  `--color`, `--hover-color`, …) — set them per widget in `_apply_theme`
  (inline custom properties beat the `:host` defaults). Accent-styled buttons
  keep plain inline styles instead.
- **`Spinner.bound_value` silently discards typed values outside the declared
  bounds** (returns the old value, no clamping) — typed values must be clamped
  server-side in `_on_parameter_change`. Typing an unchanged value also commits
  nothing (no visible feedback).
- **Decimal commas are rejected** by the numeric input filter ("2,5" becomes
  "25"). Document decimal points; do not "fix" without a client-side filter.
- **MathText (LaTeX) only works in canvas text** (plot titles, axis labels) —
  widget titles are plain text; use Unicode subscripts (fₘᵢₙ, Δt, σₙₒᵢₛₑ).
- **`TabPanel` titles are plain text too** — HTML entities render literally
  ("Real &amp; imaginary" showed as-is); use plain characters ("Real &
  imaginary").
- **`Switch`/`Toggle`/`Button`/`CheckboxGroup` have no `description` property**
  (only InputWidgets: Spinner, Select, Slider, TextInput do). For click-?
  behavior on buttons use a standalone `Tooltip` with care (target resolution
  across view trees fails — see the auto-rerun history; Select-based controls
  are the working pattern).
- **Wire `on_click` before `doc.add_root`** — bokeh resolves incoming events
  via the models subscribed at attach time (`_attach_document` →
  `_update_event_callbacks` → `document.callbacks.subscribe`); callbacks
  registered after the root is added are silently dead in the browser (no
  error anywhere — property-change callbacks via `on_change` are unaffected).
  This app wires callbacks in `_attach_callbacks` and adds the root afterwards
  (`__init__`); `run/verify_gui_polish.py` asserts the subscription.
- **`Slider`/`Select`/`Switch` do not clamp typed values** and `Slider` has no
  `description` — clamp server-side; put explanations in hints or neighboring
  Spinners.
- **Bokeh `Arrow`** uses `start=`/`end=` head models (`NormalHead`, …) — not
  `start_head`/`end_head`; `ArrowHead` only takes `size`.
- **`Line.line_color` accepts no field names** (unlike scatter colors) — set a
  line color through the renderer's glyph property instead.
- **Watchdog shutdown**: bokeh's session cleanup job never fires under manual
  io loop management (Tornado 6.6 binds periodic jobs to `IOLoop.current()` at
  start); the app's watchdog replaces it. Do not "simplify" it away.
- **Time sliders operate in seconds** and snap to the frame grid (step = Δt);
  round `value/dt` to get the frame index.
- **Re-entrancy guards** (`_syncing_time`, `_syncing_play`, `_syncing_mode`,
  `_syncing_drag`, `_syncing_amplitude`) protect the two-way synced widgets —
  keep them when touching callbacks.
- **The eigenvalue plane source drives the drag tool**: guard its refills with
  `_syncing_drag`; restore the source if a point is deleted via backspace
  (length mismatch check in `_on_eigenvalue_drag`).
- **In-place theme updates** must restyle *all* registered elements
  (`_themed_divs`) and both stored tab stylesheets + the page stylesheet;
  missed elements stay light in dark mode (user-visible bug).

### Workflow and conventions
- **Style** (enforced by the python-code-developer/python-code-review skills):
  Sphinx/reST docstrings on everything (with `:param:`/`:type:`/`:return:`/
  `:rtype:`), modern type hints (PEP 604/585), import blocks grouped and
  sorted by statement length, `_check_*` validation with `assert` and
  descriptive f-string messages, `_`-prefixed private attributes via
  `@property`, constants UPPER_SNAKE_CASE, every module ends with
  `if __name__ == "__main__": pass` (except declaration-only files).
- **Commit policy:** commit only after an explicit go-ahead from the user for
  that revision (e.g. "commit it", "go for it") — never commit proactively,
  even when a revision looks finished or verified; ask or wait when unsure.
  One commit per revision, never push (no remote configured).
- **PLAN.md is untracked** (user preference) but kept updated locally.
- **`ideas_for_review.txt`** (untracked, git-ignored): the user's running list
  of ideas/TODOs. **Check it at the start of every session for new or updated
  items.** Never delete the file and never commit it; remove individual items
  from it *only* after the user explicitly confirms they are completed
  (permission granted), and only then.
- **`run/` is untracked** internal tooling: `debug_server.py` (serves each tab
  at `/t0`–`/t5` on port **5099** — never 5006), `screenshot_tabs.py`
  (Playwright screenshots; the Playwright venv lives in
  `/tmp/opencode/shotenv`), `reports/` (review + UX reports), `screenshots/`.
  Clean up obsolete scripts and logs.
- **Vision reviews**: screenshots are checked by an image-reviewer agent
  (`scads/alias-vision`). Always verify the served process is fresh before
  reviewing captures (stale servers have produced false findings).
- **Internal `documentation/`** must be updated with every step: append to
  `steps.md`, revise `modules.md`/`architecture.md` when modules change.
- **App entry points**: `dmdsim-app` (console script) and
  `bokeh serve src/dmdsim/app` (directory app). The app modules use absolute
  imports (Bokeh's loader executes `main.py` outside the package context).

## Known Open Items (backlog in PLAN.md)

- Dark-mode revision (partial widget/hover coverage; goals and infrastructure
  documented).
- 2D OpenFOAM data via flowTorch readers (M5, postponed by the user).
- DMD variants (windowed, TLS, …) behind `core/engine.py`.
- Session save/load; per-figure toolbar grouping.

## Verification Checklist for Agents

1. `.venv/bin/pytest` — 15 tests must pass.
2. Headless app build: `python -c` snippet constructing `DMDPlayground(Document)`
   (used instead of GUI unit tests — the app layer is intentionally untested).
3. For GUI changes: capture screenshots with `run/debug_server.py` +
   `run/screenshot_tabs.py` (Playwright) and have them vision-reviewed; verify
   the served process is fresh (stale servers caused false findings).
4. Run `python-code-review` (skill) scoped to the changed files and write the
   report to `run/reports/`.
