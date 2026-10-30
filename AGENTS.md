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

- **DMD engine:** flowTorch (upstream
  `FlowModelingControl/flowtorch`, installed from the pinned git commit in
  `requirements.txt`) — it is the *only* place DMD math happens
  (`dmdsim/core/engine.py` is the single wrapper).
- **GUI:** Bokeh 3.10 server application (`dmdsim/app/`).
- **Docs:** `README.md` (usage), `PLAN.md` (untracked roadmap),
  `documentation/` (internal docs: `architecture.md`, `modules.md`, `steps.md`
  (untracked) — keep them updated with every step), `run/reports/` (untracked review
  reports), `ideas_for_review.txt` (user's idea list — see Workflow below).

## Environment and Commands

- **Everything runs in the project venv `.venv/`** (Python 3.12). Never use
  global tools. Key binaries: `.venv/bin/pytest`, `.venv/bin/dmdsim-app`,
  `.venv/bin/bokeh`.
- Tests: `.venv/bin/pytest` (20 tests; `testpaths = ["tests"]` is configured).
- Run the app: `.venv/bin/dmdsim-app` (serves on port 5006, opens a browser,
  stops itself ~2 s after the last session disconnects).
- flowTorch comes from the **upstream repository** via `requirements.txt`
  (`flowtorch-fluid @ git+…`, pinned by commit; the distribution name is
  `flowtorch-fluid`, the import package stays `flowtorch` — the PyPI release
  is outdated at 0.8). Verified end-to-end in a fresh venv (install → 20
  tests → headless build). The developer venv may still carry the user's
  fork `../flowtorch` installed **editable with `--no-deps`** — the fork
  (version 1.3) and upstream (1.10.1) both pass the suite; don't mix both
  installs in one venv.

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
- **Check flowTorch first**: before implementing any DMD-related math or
  functionality, check flowTorch for an existing implementation (its
  `flowtorch/analysis/` modules and the `DMD` API — e.g. `predict`,
  `reconstruction`, `top_modes`, `integral_contribution`) and reuse it or
  validate the own implementation against it in the tests; only implement in
  `dmdsim.core` what flowTorch does not provide. Reference sources: the
  installed upstream package (import `flowtorch`) or the user's fork checkout
  at `../flowtorch`.
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
- **Shadow-DOM theming (bokeh 3.x)**: page-level CSS can never reach widget
  internals — every visible surface lives inside a shadow root. Theme via CSS
  custom properties set on the model (`styles`, they inherit into the shadow
  root) plus per-widget `stylesheets` for what no variable covers. Verified
  specifics: buttons need the `--default*`/`--primary*`/`--inverted-color`
  families (plain inline styles only paint the host under the opaque
  `.bk-btn` — the accent "Open" buttons silently rendered white in BOTH
  themes until fixed); form-control text is forced black by the UA
  stylesheet and needs a shadow-root stylesheet (`.bk-input { color:
  var(--color) }`); sliders need one for the nouislider track/handle and
  `Slider.bar_color` for the fill (set inline by bokeh — CSS cannot win);
  switches consume `--inactive-*`/`--active-*`; hover tooltips are
  UIElements inside the plot's shadow root and follow `--background-color`/
  `--color` set on the figure. Helpers in `styling.py`
  (`button_styles`, `primary_button_styles`, `accent_button_styles`,
  `switch_styles`, `input_stylesheet`, `slider_stylesheet`); widgets are
  themed via `_style_widget` — iterate `layout_root.references()`, NOT
  `doc.models` (empty until the root is added), and style per-fit rebuilt
  widgets (weight sliders) at creation.
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
- **Figures get a `DataRange1d` unless given an explicit `Range1d`** — a
  `DataRange1d` recomputes its limits client-side from the rendered data and
  **ignores/overrides server-side `.start`/`.end` writes** in long-lived
  sessions (stale client state across server restarts made the reconstruction
  view clip at ±2 while the server computed correct limits). Any figure whose
  limits the app manages must be created with a plain `Range1d`; the app then
  owns the limits and they reach the browser verbatim. Check with
  `type(fig.y_range).__name__ == "Range1d"` in the regression script.
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
  (`_themed_divs` — Divs AND buttons/toggles) and both stored tab stylesheets
  + the page stylesheet; missed elements stay light in dark mode
  (user-visible bug).
- **CSS placeholder substitution** (`_TAB_CSS`/`_PAGE_CSS`/`_SLIDER_CSS`):
  replace longest keys first — `@PAGE` is a prefix of `@PAGETEXT` and
  `@HANDLE` of `@HANDLEBORDER`; naive dict order leaves literal `@` or
  corrupts values into invalid CSS that is silently ignored. Assert no `@`
  remains in generated CSS when touching these builders.

### Workflow and conventions
- **Style** (enforced by the python-code-developer/python-code-review skills):
  Sphinx/reST docstrings on everything (with `:param:`/`:type:`/`:return:`/
  `:rtype:`), modern type hints (PEP 604/585), import blocks grouped and
  sorted by statement length, `_check_*` validation with `assert` and
  descriptive f-string messages, `_`-prefixed private attributes via
  `@property`, constants UPPER_SNAKE_CASE, every module ends with
  `if __name__ == "__main__": pass` (except declaration-only files).
- **Commit and push policy:** always ask before committing and before
  pushing — commit only after an explicit go-ahead from the user for that
  revision (e.g. "commit it", "go for it"), and push only on an explicit
  request ("commit and push", "push it"); never do either proactively, even
  when a revision looks finished or verified. One commit per revision. The
  remote `origin` (GitHub, `JanisGeise/dmdsim`) is configured and pushes go
  to `main`.
- **PLAN.md is untracked** (user preference) but kept updated locally.
- **`ideas_for_review.txt`** (untracked, git-ignored): the user's running list
  of ideas/TODOs. **Check it at the start of every session for new or updated
  items.** Never delete the file and never commit it; remove individual items
  from it *only* after the user explicitly confirms they are completed
  (permission granted), and only then.
- **`run/` is untracked** internal tooling: `debug_server.py` (serves each
  tab at `/t0`–`/t6` on port **5099** — never 5006 — plus the dark-mode
  variants `/d0`–`/d6` (theme switch flipped after the build) and the static
  prediction cursor `/t6s`/`d6s`), `screenshot_tabs.py` (Playwright
  screenshots; base URL, output dir, and route prefix `t`/`d` as arguments;
  the Playwright venv lives in `/tmp/opencode/shotenv`), `reports/` (review +
  UX reports), `screenshots/`. Clean up obsolete scripts and logs.
- **Vision reviews**: screenshots are checked by an image-reviewer agent
  (`scads/alias-vision`). Always verify the served process is fresh before
  reviewing captures (stale servers have produced false findings).
- **Internal `documentation/`** must be updated with every step: append to
  `steps.md`, revise `modules.md`/`architecture.md` when modules change.
- **`README.md` scope**: user-facing only (installation, running the GUI with
  its panel screenshots, tests, license, internal-documentation pointer).
  **Never (re-)add roadmap, repository/hosting, or completed-steps sections**
  — plans, milestones, and the status of completed steps are always internal
  (`PLAN.md` untracked, `documentation/`, `steps.md` untracked). The panel
  screenshots are tracked in `docs/images/` (light theme, cropped to content
  + margin, ≤1200 px WebP); regenerate with `run/screenshot_tabs.py` +
  `run/crop_screenshots.py` when the UI changes in a visible way. Keep the
  README **course-agnostic**: never mention the ML-CFD course or specific
  lectures there — present the project as a general interactive DMD teaching
  tool (the origin/audience context lives in AGENTS.md and `PLAN.md`).
- **App entry points**: `dmdsim-app` (console script) and
  `bokeh serve src/dmdsim/app` (directory app). The app modules use absolute
  imports (Bokeh's loader executes `main.py` outside the package context).

## Known Open Items (backlog in PLAN.md)

- 2D OpenFOAM data via flowTorch readers (M5, postponed by the user).
- DMD variants (windowed, TLS, …) behind `core/engine.py`.
- Session save/load; per-figure toolbar grouping.
- (The dark-mode revision — widget shadow-DOM theming + themed figure
  reference colors — is done as of 2026-10-30; see
  `run/reports/2026-10-30-dark-mode-revision.md`.)

## Verification Checklist for Agents

1. `.venv/bin/pytest` — 20 tests must pass.
2. Headless app build: `python -c` snippet constructing `DMDPlayground(Document)`
   (used instead of GUI unit tests — the app layer is intentionally untested);
   for theme changes, also flip both theme switches and probe the computed
   styles (inputs/buttons/sliders/switch) — visual review alone has missed
   shadow-DOM leftovers.
3. For GUI changes: capture screenshots with `run/debug_server.py` +
   `run/screenshot_tabs.py` (Playwright) and have them vision-reviewed; verify
   the served process is fresh (stale servers caused false findings). Capture
   BOTH themes when the change touches theming (`/t…` and `/d…` routes).
4. Run `python-code-review` (skill) scoped to the changed files and write the
   report to `run/reports/`.
