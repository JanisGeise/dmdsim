# Implementation steps

Chronological log of the implementation. Each entry names the milestone, what was
built, and the decisions/insights that shaped it. New steps are appended at the
bottom; `modules.md` and `architecture.md` are updated in the same step whenever
modules change.

## M0 — project setup (2026-10-28)

- Created `.venv` (Python 3.12), `pyproject.toml` (src layout, package `dmdsim`),
  `requirements.txt` (pinned, CPU-only torch), `.gitignore`.
- Installed flowTorch editable from the local repository `../flowtorch` with
  `--no-deps` plus the minimal import requirements (scipy, plotly, pandas, pillow,
  vtk, h5py, netCDF4) — avoiding jupyterlab and other heavy optional deps.
- Smoke test: flowTorch's DMD preserves float64 input (complex128 eigenvalues,
  exact recovery of a 5 Hz traveling wave). Decision: always feed float64.

## M1 — core library (2026-10-28)

- `signals/generator.py`: `Mode` + `ModalSignal` — synthetic modal line fields
  (traveling waves, offset, seeded noise). Decision: "1D" means a spatial line
  field, not a scalar time series (plain DMD of a scalar series is rank-limited).
- `core/engine.py`: `fit_dmd` — the single numpy↔torch wrapper around
  `flowtorch.analysis.DMD`.
- `core/quantities.py`: `DMDResult` with frequencies, growth rates, and conjugate
  pair detection (`_PAIR_TOL = 1e-8`).
- `core/synth.py`: `synthesize` (rebuild `Re{Σ φᵢ·bᵢ·λᵢᵗ}`) and
  `mirror_conjugates` (conjugate symmetry after edits).
- Tests: 13 passing (eigenvalue recovery, round trip at machine precision,
  mirroring semantics), ~95% coverage. First code review: no bugs; coverage
  suggestions open (see `run/reports/2026-10-28.md`).
- Insight: `DMDResult.pairs` must be cached before editing quantities — pair
  detection runs on current values (documented in the docstring).

## M2 — GUI v1 (2026-10-28)

- `app/figures.py` + `app/main.py`: Bokeh application with signal panel (3 mode
  editors κ/f/A/σ, grid, offset, noise), DMD panel (rank with automatic option,
  auto-rerun toggle, Run button), eigenvalue plane, spectrum, mode shape viewer
  (Re/Im/|φ|/arg φ), time view (heatmap + cursor + curves + play animation),
  inline hints, per-mode readout.
- Entry points: console script `dmdsim-app` (programmatic `Server`) and
  `bokeh serve src/dmdsim/app` (directory app). Fix during implementation:
  `Application` needs a `FunctionHandler`, not a raw callable.
- Decision: absolute imports in `app/` because Bokeh executes `main.py` outside
  the package context under `bokeh serve`.
- Untracked `examples/demo_core.py` kept for internal testing (not committed, per
  decision that internal test scripts stay out of version control).

## M2.1 — server shutdown on browser close (2026-10-28)

- Requirement: the app server must stop when the browser is closed.
- Investigation with an instrumented server: Bokeh 3.10.1's session cleanup job
  never fires under manual io loop management — Tornado 6.6 binds
  `PeriodicCallback.start()` to `IOLoop.current()` at start time, which is not the
  server's private loop. Discard criteria were verifiably met while the session
  lingered forever.
- Solution: own watchdog (Tornado `PeriodicCallback` started from inside the
  running loop, 500 ms interval, 2 s grace) that stops the io loop once all
  sessions have zero connections. Verified: connected session keeps the server
  alive; closing the last session exits within ~2.5 s.
- `.idea/` removed from version control and ignored; example scripts untracked.

## M2.2 — sidebar layout and internal documentation (2026-10-28)

- GUI restructure: the crowded single-page layout became a `Tabs` widget with
  `tabs_location="left"` — a left sidebar with one submenu per step: "① Signal"
  (generators + fit controls + status), "② Eigenvalues", "③ Spectrum",
  "④ Mode shape", "⑤ Time view". All panels stay alive while hidden, so widget
  state survives switching; view figures got more space (620×560 for eigenvalue
  plane and spectrum, 1000×280 for mode shapes, 500×380 for the time views).
- Added this `documentation/` directory (README, architecture, modules, steps) as
  the living internal documentation, to be updated with every further step.
- Backlog additions (PLAN.md §8): dark mode (background `#131313`, dark-adapted
  diverging colormap, dimmed axes) and cross-platform support (Windows/macOS).

## M2.3 — signal preview and cleaner menus (2026-10-28)

- User feedback: the created signal was not visible in the setup panel, and the
  menu naming was confusing.
- Panels renamed to what they show: "Setup", "Eigenvalues", "Spectrum",
  "Mode shapes", "Reconstruction" (was "① Signal" ... "⑤ Time view"); circled
  numbers dropped.
- New live **signal preview** heatmap in the Setup panel (updates with every
  refit; `create_field_heatmap` gained `show_cursor` and `title` parameters so
  the preview reuses the builder without the time cursor).
- Setup panel reorganized with clear section headers: Modal components, Grid &
  noise, DMD fit, Signal preview.

## M2.4 — modern styling with per-step accent colors (2026-10-28)

- User feedback: the GUI should look more modern and the panels should be
  visually distinguishable through color.
- New `app/styling.py`: per-step `(tint, accent)` palette (setup blue, eigenvalues
  green, spectrum amber, modes purple, reconstruction red); `style_figure` (white
  plot area on tinted panel background, accent title, muted axes/grid/legend),
  `header`, `hint`, `info_box` element factories, and a tab strip stylesheet
  (rounded buttons, hover state, blue active tab).
- Applied in `main.py`: color-coded figures, styled section headers and hints,
  info-box status/readout, accent-colored Run button, panel spacing.
- Note for a future dark mode (backlog): the palette lives in one module, so a
  dark variant can be introduced by swapping `_STEP_COLORS` and the neutral
  constants there.

## M2.5 — vision-based GUI polish (2026-10-28)

- Method: headless screenshots of all five tabs (Playwright in a throwaway venv,
  scripts untracked in `run/`) reviewed with a vision model, two review rounds.
- Round 1 findings: Setup tab was a "wall of controls" (blocking), tab labels too
  small, tints slightly garish, Reconstruction selection controls disconnected
  from the plots.
- Fixes: two-column Setup (controls beside the signal preview), numbered tab
  titles ("1 · Setup" ... "5 · Reconstruction"), softened panel tints, larger tab
  labels, "Time & mode selection" header in the Reconstruction panel.
- Round 2 verdict: acceptable to show; final polish applied (more air between the
  mode editor rows, extra gap before the mode selection checkboxes).
- README now states transparently that the project is vibe coded; the disclosure
  names OpenCode and ScaDS.AI with links.

## M2.6 — start menu, nested mode submenus, sidebar alignment (2026-10-28)

- User feedback: the sidebar was not aligned, a start menu was missing (create
  signal, animate it, run DMD), and the mode quantities should live in their own
  submenus with another nesting depth.
- Sidebar alignment: the main tab stylesheet now uses flex column with
  `align-items: stretch` and left-aligned text, giving uniformly wide tab buttons;
  `tab_stylesheet` gained `accent` and `vertical` parameters so nested strips can
  differ (modes submenus use the purple accent horizontally).
- "1 · Signal" start menu: modal components + grid & noise on the left, new
  **signal animation** on the right (heatmap with time cursor + time slider +
  play), DMD fit and status below.
- "4 · Mode shapes": nested horizontal sub-tabs Real part / Imaginary part /
  Phase / Propagation direction (four single-quantity figures via
  `create_mode_component_figure`, replacing the combined viewer). New
  propagation-direction readout: phase slope κ (unwrapped linear fit) →
  wavelength λ_w = 2π/|κ|, phase velocity c = −2πf/κ with travel direction and
  arrow; physically validated against the generator (κ = 2, f = 5 →
  c = 15.7 units/s toward +s).
- Both time sliders and play toggles are two-way synchronized (re-entrancy
  guards) and drive the same shared time cursor, so the signal animation and the
  reconstruction animation always show the same instant.
- Verified with a third vision review round: sidebar aligned, start menu reads
  create → animate → run, nested sub-tabs clean; verdict "acceptable to show".

## M2.10 — status update, GUI revise phase declared (2026-10-28)

- PLAN.md, README.md and `documentation/architecture.md` updated to the current
  phase: the **GUI revise phase** is ongoing (M2.1–M2.9, panel-by-panel feedback
  rounds with vision-based screenshot reviews); the milestones table gained an
  M2R row whose completion criterion is the user declaring the revision
  complete.
- Commit policy in effect: the whole revision sits uncommitted in the working
  tree; a single commit is made when the user says the revision is complete.
  M3 (interactive eigenvalue editing) starts afterwards.

## M2.7 — Signal panel rework and panel-by-panel pass (2026-10-28)

- User feedback (panel-by-panel round, starting with "1 · Signal"): the 1D signal
  should animate as a 1D line (not a heatmap); the manual mode editors assume
  knowledge the user should not need; `torch.linalg` errors from rank experiments
  crashed the app.
- New core helper `signals.generator.generate_modes(n_modes, f_min, f_max, seed,
  phase_speed=10)`: evenly spaced frequencies, dispersion-tied wavenumbers
  (κ = 2πf/c), seeded random amplitudes/growth/phases — reproducible. Unit tests
  added (determinism, bounds, dispersion, single-mode centering, invalid args).
- GUI: the three per-mode editor rows became three spinners (n_modes, f_min,
  f_max) with an explanatory hint; the signal animation became a 1D line plot
  x(s) at the current time (fixed y-range so playback does not rescale).
- Exception handling: `_run_dmd` wraps build/fit/synthesis and reports failures
  (e.g., `torch.linalg.eig` NaN errors at high ranks) in red in the status box
  while previous views stay; verified programmatically for rank 64.
- Rank semantics: 0 = automatic now means the exact rank of the clean composition
  (2·n_modes, +1 with offset); flowTorch optimal truncation is used when noise is
  on (its optimal-rank selection overestimates on clean data: 26 modes for a
  3-mode signal). Status line shows the used rank.
- Panel-by-panel vision review (round 5) over Eigenvalues/Spectrum/Mode
  shapes/Propagation direction/Reconstruction: all "fine", verdict "ready for
  use"; two micro-polish items applied (larger direction-readout font, more gap
  before the mode checkboxes), toolbar grouping noted as optional.
- GUI revision declared complete; single commit per user request.

## M2.8 — unit window, internal grid, percent noise (2026-10-28)

- User feedback: typed number entry should work; dx/n_space need no user
  exposure; the animation should show y(x) over a fixed window x ∈ [0, 1]
  animated in time; noise should be a single percent box.
- The spatial domain is now fixed internally: x ∈ [0, 1] with 201 grid points
  (dx = 0.005); the composition's phase speed is chosen adaptively so the
  highest frequency appears as three wavelengths inside the window
  (phase_speed = f_max/3), keeping every mode visible in the animation.
- Noise is a single "noise [%]" box: 0 % = off, otherwise Gaussian noise with
  std = level/100 · max|clean signal| (seeded, reproducible); the separate
  toggle is gone and automatic rank falls back to flowTorch optimal truncation
  for any level > 0.
- The animation axes are labeled x and y(x) to match the fixed-window mental
  model; Bokeh Spinners accept typed numbers (commit with Enter).
- Fourth vision review of the panel: "ready for deployment", no must-fix items.
- Commit policy: the user declares when a GUI revision is complete; only then is
  a single commit made (the M2.7 commit was reset accordingly).

## M2.11 — wider bounds, hint removal, math labels (2026-10-28)

- User feedback: the n_modes/frequency limits were arbitrary and too tight; the
  hint texts are redundant now that the ? tooltips work; parameters should render
  as math.
- Bounds widened: n_modes 1–32, f_min/f_max 0–500 Hz, rank 0–256. The adaptive
  phase speed (f_max/3) makes the spatial resolution independent of f_max
  (κ_max = 6π rad constant), so only the temporal Nyquist limit 1/(2·dt) remains
  as a physics constraint (documented in the f_max tooltip; aliasing above it is
  a teaching moment).
- The Signal panel hint texts were removed — their content lives in the ?
  tooltips (composition semantics, time discretization, noise definition).
- Math rendering investigation: bokeh 3.10 supports MathText (LaTeX) ONLY in
  canvas text (plot titles, axis labels, tick labels) — widget titles are plain
  text (zero MathText references in the widgets bundle; a first attempt with
  "$$...$$" titles showed raw LaTeX, caught by vision review round 8). Widget
  titles now use Unicode subscripts (fₘᵢₙ, fₘₐₓ, Δt, σₙₒᵢₛₑ, nₜᵢₘₑ); the signal
  animation's axis labels use real LaTeX ("$$x$$", "$$y(x)$$", verified rendered).
- Vision review round 9: clean, no raw LaTeX, math axis labels rendered.

## M2.12 — Signal panel cosmetic fixes (2026-10-28)

- User feedback: the rank info should belong to the DMD boxes, the DMD fit
  controls felt randomly distributed, n_modes was not rendered mathematically,
  and the time sliders should show t in seconds.
- The "DMD fit" group (rank, auto-rerun, Run DMD) moved into the left controls
  column below "Noise" — the panel now reads as a single coherent stack
  (Modal composition → Time → Noise → DMD fit) beside the animation column.
- "n_modes" renamed to "Nₘ" (Unicode subscript; a full "modes" subscript is
  impossible in Unicode — no subscript 'd' glyph — the tooltip carries the
  meaning).
- Both time sliders now operate in **seconds**: title "t [s]", range
  [0, (n_time − 1)·dt], step = Δt (snapping to the frame grid); the animation
  advance and the two-way slider sync round to the frame grid (no float drift).
- Vision review round 10: "complete and production-ready", no must-fix items.

## M2.14 — eigenvalue dynamics subplot (2026-10-28)

- User request: an animation subplot for the eigenvalues to emphasize their
  influence (especially once dragging works in M3).
- The "2 · Eigenvalues" panel now shows the eigenvalue plane and a new
  **mode dynamics** subplot side by side (shared complex-plane scale and ranges):
  the trajectory λᵗ of the selected mode, animated with the global time cursor —
  the trail spirals inward (decay), outward (growth) or orbits (neutral), with
  the current point marked and hoverable; faded gray markers show all fitted
  eigenvalues for context.
- A second mode-index slider on this panel is two-way synchronized with the Mode
  shapes panel's slider (same re-entrancy-guard pattern as the time sliders).
- New builder `create_mode_dynamics_figure` in `app/figures.py`; trail/point
  updates live in `_update_time_view` (single source of the time cursor);
  range/contexts updates in `_update_quantities`.
- Vision review round 12: clean, balanced, pedagogically sound, no must-fix
  items. When M3 dragging lands, reshaping an eigenvalue reshapes this spiral
  live.
- Sorting aligned with flowTorch (user pointer): the fit now captures flowTorch's
  `integral_contribution` (Kou et al. 2017: mode-norm² · Σₜ|b·λᵗ|, stored on
  `DMDResult`) and the sort criteria are amplitude |b|, integral contribution
  (flowTorch numbers), and frequency |f| (flowTorch uses frequency only as a
  filter in `top_modes`); the previously hand-rolled L2 integral formula and the
  growth-rate criterion were dropped.

## M2.15 — panel 2 completed, panel 3 (Spectrum) revised (2026-10-28)

- User feedback: the mode sliders should be called "eigenvalue number"; panel 2
  revision declared complete; panel 3 revision next.
- Both mode sliders renamed to "eigenvalue number" (they pick an eigenvalue of
  the fitted set).
- Spectrum panel upgrades: stems/markers colored by stability (red growing, blue
  decaying, gray neutral), selected eigenvalue emphasized (larger marker,
  thicker stem, synced with the eigenvalue number slider), dashed Nyquist
  boundary lines at ±1/(2·dt) with the frequency axis widened to ±1.15·Nyquist;
  `create_spectrum_figure` returns the Nyquist spans for the app to update;
  spectrum filling extracted into `_update_spectrum` (called on refit and on
  eigenvalue selection changes).
- Vision review round 13: "major success", no must-fix items.

## M2.16 — panel 4 revision: phase wrapping + traveling-wave direction view (2026-10-28)

- User feedback: the propagation-direction panel was "weird" and the discontinuous
  (wrapped) phase is hard to understand.
- Phase sub-tab: now shows the UNWRAPPED phase as a continuous line (for a
  traveling wave simply a straight line with slope κ) and overlays the wrapped
  ±π values as faded dots — the jumps become an explainable feature
  (`create_mode_component_figure` renders `unwrapped` + `value` columns).
- Propagation direction sub-tab: replaced the abstract phase plot with the
  ANIMATED TRAVELING WAVE Re{φ·b·λᵗ} of the selected mode
  (`create_direction_figure`), driven by a synced time slider + play button
  inside the sub-tab; a purple direction arrow (bokeh `Arrow` + `NormalHead`)
  above the wave points toward ±x (hidden for non-propagating modes); the info
  box below derives κ (unwrapped linear fit), wavelength λ_w = 2π/|κ|, and phase
  velocity c = −2πf/κ with direction. Fixed y-range over the whole animation
  (envelope incl. growth) so playback never rescales. Physics validated: the
  fitted member (κ = −18.85, f = +10 Hz) yields c = +3.33 units/s toward +x,
  matching the generator's cos(18.85x − 62.8t).
- The Mode shapes panel gained its own synced time slider + play toggle (four-way
  synchronization with panels 1, 2 and 5).
- Vision review round 17: "a complete success… no must-fix items remain."

## M2.17 — panel 5 revision: pair-colored contributions (2026-10-28)

- User request: update the Reconstruction panel with the design language
  established in the previous panel revisions.
- The time view now draws a **thin per-eigenvalue contribution line in the
  pair color** for every checked eigenvalue (conjugate pairs share their color,
  so a physical mode's two lines visually group; hover shows eigenvalue number
  and frequency via a `MultiLine` + `HoverTool` with `line_policy="nearest"`).
- The **selected eigenvalue's** contribution is thicker and colored with its
  pair color (renderer glyph updated on selection changes, same pattern as the
  dynamics trail).
- Reference curves muted so the colored contributions are the focal point:
  original black solid, reconstruction gray dashed, Σ selected gray dash-dot.
- Legend labels: original / reconstruction / Σ selected / selected eigenvalue.
- Vision review round 18: "highly sophisticated and pedagogically superior —
  no must-fix items remain."

## M2.18 — panel 5 heatmap removed, fixed y-limits; revision committed (2026-10-28)

- User feedback: remove the 2D animation (the space-time heatmap) from the
  Reconstruction panel; the y-limits of the animation must remain constant.
- The heatmap (figure, image source, color mapper, cursor span) was removed from
  panel 5 and `create_field_heatmap` from `app/figures.py` (the layout slot is
  taken by the widened time-curves figure, 1000×420); the y-ranges of BOTH
  animated line plots (signal animation and reconstruction curves) are fixed to
  ±1.1·max|signal| on every refit, so playback never rescales.
- The GUI revise phase (M2R, rounds 1–18) is declared complete by the user; all
  accumulated changes are committed as a single commit. Next milestone: M3 —
  interactive eigenvalue editing (dragging), for which the λᵗ dynamics subplot
  and the reconstruction contribution lines are the live-feedback surfaces.
- Round 14/16 follow-ups on panel 2: eigenvalue markers in all complex-plane
  plots and the spectrum carry **pair colors** — each conjugate pair shares one
  categorical color (PAIR_PALETTE in `app/styling.py`, assigned via
  `DMDResult.pairs` in `_pair_colors()`), unpaired real eigenvalues are gray;
  the λᵗ trail carries the selected pair's color (set via the trail renderer's
  glyph — `Line.line_color` accepts no field names). Stability remains encoded
  by position (radius vs. unit circle) and the hover readouts. The λᵗ trail now
  starts at the selected eigenvalue (λ¹, previously the misleading λ⁰ = 1
  identity point). Vision rounds 14 (caught red/green inconsistency + the
  trail-origin bug) and 16 confirm: "the color scheme is perfectly implemented".
- Operational lesson: a stale debug server on the screenshot port served old
  code (vision round 15 flagged colors that were never in the new build); always
  verify the served process before reviewing captures — the debug server now
  runs on its own port 5099 and stray processes are killed before captures.

## M2.13 — DMD fit group fine-tuning (2026-10-28)

- User feedback: auto-rerun and Run DMD belong below the rank box; the info box
  belongs below the Run DMD button and should only show the rank (when
  automatic) and the error.
- The DMD fit section is now vertical: rank → row(auto-rerun, Run DMD) → info
  box, all inside the left controls column; the top status bar is gone.
- Info box content minimized per request: "rank automatic (N) · round-trip
  error" when automatic (N = fitted mode count, i.e. the resolved rank; optimal
  truncation count with noise), otherwise only "round-trip error". Fit failures
  still appear in red in the same box.
- Auto-rerun ? tooltip: bokeh's Toggle/Button/CheckboxGroup have no `description`
  property (only InputWidgets do), and a standalone Tooltip cannot anchor across
  view trees (target resolution falls back to the page body). Auto-rerun is
  therefore a compact `Select` (on/off) with the native ? popover explaining the
  behavior; semantics unchanged (on = refit on any change, off = stale marker +
  manual Run DMD).

## M2.9 — tooltips with bounds, clamped typing, section split (2026-10-28)

- User feedback: the ? tooltips showed nothing; Enter seemed to fail in the modal
  composition part; Time and Noise should be separate sections; the offset needed
  an explanation.
- Root causes found by reading the bokeh 3.10 bundle and browser-level testing:
  (1) a plain string `description` only sets the native `title` attribute — the
  interactive popover requires a `bokeh.models.Tooltip` instance (now used, with
  parameter meaning AND allowed range, auto-derived from the widget bounds via
  the `_spinner` helper); (2) bokeh's `Spinner.bound_value` silently DISCARDS
  typed values outside the declared bounds (returns the old value instead of
  clamping) — typing 99 into n_modes commits nothing; typing an unchanged value
  also commits nothing. Both documented as bokeh limitations; the tooltips now
  show the bounds so the discard is explainable, and the f_min/f_max relation
  clamp works server-side (those values do reach the server).
- Decimal commas ("2,5") are rejected by bokeh's input filter (keystrokes are
  reverted, "2,5" becomes "25") — noted as a limitation; type decimal points.
- "Time" (dt, n time) and "Noise" (noise [%]) are separate sections with hints.
- Verified in Firefox AND Chromium headless: hover flips the tooltip model to
  visible and renders the popover (screenshot `run/screenshots/tab0_tooltip.png`,
  vision-checked: readable, complete, well positioned); Enter commits on all
  seven spinners for in-bounds values.
