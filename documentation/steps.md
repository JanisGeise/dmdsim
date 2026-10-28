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

## M2.10 — status update, GUI revise phase declared (2026-10-28)



- PLAN.md, README.md and `documentation/architecture.md` updated to the current
  phase: the **GUI revise phase** is ongoing (M2.1–M2.9, panel-by-panel feedback
  rounds with vision-based screenshot reviews); the milestones table gained an
  M2R row whose completion criterion is the user declaring the revision
  complete.
- Commit policy in effect: the whole revision sits uncommitted in the working
  tree; a single commit is made when the user says the revision is complete.
  M3 (interactive eigenvalue editing) starts afterwards.

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

## M2.15b — panel 2 follow-up: selection highlight, trail origin, pair colors (2026-10-28)


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

## M3 — interactive eigenvalue editing (2026-10-28)


- **Draggable eigenvalues**: the eigenvalue plane's scatter carries a
  `PointDrawTool` (`add=False` — drag-only; backspace deletions are undone by
  restoring the source from the result). The drag callback
  (`_on_eigenvalue_drag`) fires once per drag gesture: it detects the dragged
  indices against a cached marker positions copy, writes the new eigenvalues
  into the result, **mirrors the conjugate partners** (cached pair structure of
  the current fit), refreshes the plane source (partner mirroring visible),
  patches the dynamics markers, recomputes the reconstruction via
  `synthesize`, and updates spectrum, λᵗ spiral, direction view, readout, and
  the status box (now reporting the max deviation from the original signal
  after edits).
- **Amplitude scaling**: a "|b| scale (selected)" spinner in panel 2 scales the
  selected eigenvalue's amplitude relative to its fitted value (baseline
  cached per fit; the conjugate partner is scaled and re-conjugated
  accordingly); the spinner follows the selection via the synced mode sliders.
- Semantics: edits apply to the CURRENT fit and are observable in the
  reconstruction (deviation from the original), the λᵗ spiral, and the
  spectrum; a refit (Run DMD or signal change) restores the fitted values.
- Verified headlessly (source-patch simulation = the exact server event the
  drag tool emits): mirroring, plane sync, reconstruction consistency against
  direct synthesis, spectrum frequency update, 2× amplitude scaling with
  partner, spinner per-mode sync, refit restore; browser round trip (JS source
  patch → server → views) with zero JS errors; vision review round 19
  ("implemented perfectly").
- Guard fixes during implementation: the plane-source refill in
  `_update_eigenvalue_markers` is guarded against the drag callback, and the
  partner-mirroring loop no longer mutates its iteration dict.
- Round 20 user feedback implemented: (1) **unit-circle lock** ("lock to unit
  circle" toggle in panel 2) — locked drags are projected onto |λ| = 1 so only
  the frequency changes, and locked edits are TEMPORARY: selecting a different
  eigenvalue restores the previously locked-edited one to its fitted value
  (tracked via `_locked_edit_indices`/`_fit_eigvals`; free drags without the
  lock persist deliberately). (2) Selecting a different eigenvalue resets the
  shared time cursor to zero. (3) The amplitude-scale tooltip now explains the
  semantics concretely (1 = as fitted, 0 = silenced, 2 = twice as strong; the
  original signal is unaffected). (4) Break-session hardening: typed amplitude
  scales are clamped and written back to the box (500 → 3, −5 → 0), drags into
  the origin are clamped to radius 1e-3 (log(0) undefined), constant signals
  produce a friendly "needs variation" message instead of the cryptic
  torch.linalg NaN error; drag-to-origin with and without the lock verified.

## M4 — UX revision: start panel, themes, beginner labels (2026-10-28, commit pending user verdict)

- Round 22 refinements: theme dropdown → "dark mode" toggle in the app bar
  (theme switch no longer redirects to Start — active tabs preserved across the
  layout rebuild), Start-panel "Open" buttons colored per step accent, panel 2
  controls consolidated into one row, λᵗ trail color-column length fixed
  (BokehUserWarning gone), dark mode uses the brighter `PAIR_PALETTE_DARK`
  cycle. Vision review round 22: "production-ready".
- Round 23 verification + fixes: in-place theming verified in the browser
  (body #131313, input boxes #1e1e1e with light text, figure backgrounds
  #1e1e1e); the white-input-box bug fixed via `!important` page CSS plus
  per-widget CSS custom properties.
- **Dark mode deferred**: some issues remain (partial coverage of widget
  internals, hover states); the user accepted the current state — a dedicated
  dark-mode revision is planned later (see backlog in PLAN.md).


- Method: beginner-persona audit (a student who just heard the DMD lecture —
  ML-CFD course, lecture 5) with vision-based review of all panels; report in
  `run/reports/ux_revision.md`.
- User feedback: the GUI should match the lecture website's style (light AND
  dark mode), feel less overloaded, and a start panel should be the default
  landing view.
- **Theme system**: `app/styling.py` centralizes all colors in `THEMES` (light +
  dark, dark = #131313 page with dark panel tints and brighter accents);
  `style_figure`/`header`/`hint`/`info_box` take a mode parameter; the page and
  widgets are styled via a `GlobalInlineStyleSheet` (inputs, buttons, sliders,
  tooltips); figures are restyled through a registered `(figure, step)` list and
  theme-dependent renderer colors (signal line, original/reconstruction/Σ
  lines). Theme switch rebuilds the layout container (models persist; the old
  root is removed after the new one is added — `doc.clear()` is forbidden, it
  wipes session callbacks).
- **Application bar**: title + theme dropdown (light/dark) above the tabs,
  persistent on every panel.
- **Start panel** (new default tab): "DMD in a nutshell" bridge (snapshot
  matrix X → time-shifted X′ → SVD → eigenvalue problem → modes φ, amplitudes
  b, λ → reconstruction formula), numbered workflow rows with Open buttons
  (navigate to panels), and tips. Default landing view.
- **Beginner labels**: "Σ selected" → "sum of selected modes"; spectrum y-axis
  "amplitude |b|"; sort tooltip explains the integral contribution in lecture
  terms; the shared-time-cursor line restored on panel 1.
- **Decluttering**: panel 2 controls in one row (eigenvalue number, sort,
  |b| scale, unit-circle lock); hints shortened.
- Debug tooling moved to port 5099; stale-server lesson documented (always
  verify the served build before reviewing captures).
- Vision review rounds 20–21: dark theme "clean and consistent", start panel
  "release-ready" (after a stale-server false alarm), pair colors "perfectly
  implemented".
- Round 22–23 user feedback implemented: (1) the theme dropdown became a
  **`Switch` pill toggle** ("dark mode", bokeh 3.10 `Switch` widget) in the app
  bar; (2) theme switching is now **in place** — `_apply_theme` restyles
  figures, renderers, 34 registered Divs, and updates the stored stylesheets'
  `css` strings; NO layout rebuild, NO root add/remove (the previous rebuild +
  root swap broke the client session: shared models across roots, silent
  render abort) — the active panel and all widget states are preserved;
  (3) the Start panel's "Open" buttons are colored per step accent;
  (4) panel 2's controls consolidated into ONE row; (5) the λᵗ trail's color
  column length now matches the point count (fixes BokehUserWarning); (6) dark
  mode uses a brighter pair-color cycle (`PAIR_PALETTE_DARK`); (7) widget boxes
  stay white in dark mode unless the global CSS uses `!important` (bokeh's own
  widget stylesheets win otherwise) — fixed. Vision review rounds 22–23:
  "production-ready", "fully functional, white input box bug resolved".
- Round 23 verification finding: updating a `GlobalInlineStyleSheet`'s `css`
  after injection does NOT re-theme the page (injected once); the fix recreates
  the stylesheet objects and re-attaches them to the layout root/tabs on every
  theme switch — verified in the browser: body #131313, input boxes #1e1e1e
  with light text, figure backgrounds #1e1e1e. Bokeh widgets theme via CSS
  custom properties (`--background-color`, `--color`, `--hover-color`, ...)
  that are set per widget in `_apply_theme`; accent-styled buttons keep their
  inline accent colors.
- Round 22–23 user feedback implemented: (1) the theme dropdown became a
  **`Switch` pill toggle** ("dark mode", bokeh 3.10 `Switch` widget) in the app
  bar; (2) theme switching is now **in place** — `_apply_theme` restyles
  figures, renderers, 34 registered Divs, and updates the stored stylesheets'
  `css` strings; NO layout rebuild, NO root add/remove (the previous rebuild +
  root swap broke the client session: shared models across roots, silent
  render abort) — the active panel and all widget states are preserved;
  (3) the Start panel's "Open" buttons are colored per step accent;
  (4) panel 2's controls consolidated into ONE row; (5) the λᵗ trail's color
  column length now matches the point count (fixes BokehUserWarning); (6) dark
  mode uses a brighter pair-color cycle (`PAIR_PALETTE_DARK`); (7) widget boxes
  stay white in dark mode unless the global CSS uses `!important` (bokeh's own
  widget stylesheets win otherwise) — fixed. Vision review rounds 22–23:
  "production-ready", "fully functional, white input box bug resolved".
- Round 22 user feedback implemented: (1) the theme dropdown became a **"dark
  mode" toggle** in the app bar; (2) theme switching NO LONGER redirects to the
  Start panel — `_apply_theme` preserves the active tabs (main + nested) and
  swaps the layout root after rebuilding; (3) the Start panel's "Open" buttons
  are colored per step accent; (4) panel 2's controls consolidated into ONE row;
  (5) the λᵗ trail's color column length now matches the point count (fixes
  BokehUserWarning); (6) dark mode uses a brighter pair-color cycle
  (`PAIR_PALETTE_DARK`). Vision review round 22: "production-ready".
- Open decisions: slider naming (kept "eigenvalue number" per earlier user
  choice), default sort (kept integral contribution, tooltip explains it).


## M4b — GUI polish: per-panel resets, title, continuous-time plane, direction link (2026-10-28)

Scope: items 1–4 of the user's `ideas_for_review.txt` (GUI polishing);
decisions confirmed with the user (per-panel reset semantics, continuous-time
plane as the "discrete vs. continuous" addition, half-plane shading for the
direction link).

- **Reset buttons (item 1)**: panel 1 gets "↺ reset parameters" (restores
  `_DEFAULT_SIGNAL_PARAMS` — Nₘ=3, fₘᵢₙ=1, fₘₐₓ=10, Δt=0.01, nₜᵢₘₑ=300,
  offset=0, σₙₒᵢₛₑ=0, rank=0 — plus auto-rerun "on", then one refit); panels
  2–5 get "↺ reset to DMD fit" (`_reset_to_fit`: restores the cached fit
  baselines `_fit_eigvals`/`_fit_amplitudes`, clears locked edits and the drag
  cache, |b| scale → 1, recomputes the reconstruction, refreshes the quantity
  and time views; panel 5 additionally re-selects all modes). New
  `_syncing_params` guard keeps the change callbacks idle during the
  programmatic restore (no refit storms, no mutual clamping of the frequency
  bounds — those are set explicitly).
- **Title (item 2)**: app bar now shows only "DMD Playground" (lecture
  reference removed).
- **Continuous-time plane (item 3)**: new
  `create_continuous_spectrum_figure()` — μ = ln(λ)/Δt plane with axes
  σ = Re(μ) [1/s] and f = Im(μ)/(2π) [Hz], dashed zero axes, pair colors and
  selection emphasis shared with the spectrum, hover with mode/σ/f; placed
  beside the amplitude spectrum in panel 3 (both 620×560); y-range shares the
  Nyquist range, σ-range follows the fitted data; refreshed wherever the
  spectrum is refreshed (refit, sort, drag, amplitude edit).
- **Direction link (item 4)**: the eigenvalue plane shades the upper/lower
  half planes (`HALF_PLANE_COLORS` in styling.py, labels "f > 0"/"f < 0"
  carrying the same colors; hover scoped to the scatter so the patches don't
  capture it); the propagation-direction readout gained the explicit line
  "Im λ = … → eigenvalue above/below the real axis (f ≷ 0)" and shows the
  formula c = −2πf/κ; hints updated (dragging across the real axis flips the
  propagation direction).
- **Fix included**: `_update_quantities` now also clears `_fit_eigvals` in the
  zero-mode branch (stale baseline would break `_reset_to_fit`).
- Debug tooling: `run/debug_server.py` extended to six tabs `/t0`–`/t5`
  (Start panel shift had left it at five); `run/screenshot_tabs.py` waits for
  `canvas, .bk-btn` (the Start tab has no canvas) and clicks the direction
  sub-tab on `/t4`.
- Verification: pytest 15 passed; headless verification script
  `run/verify_gui_polish.py` (11 checks: title, wired buttons, amplitude/drag
  reset round trips, panel-5 reselect, parameter reset with consistent
  bounds, continuous-plane data/ranges, direction side line, real-axis drag
  flip of side + arrow, half-plane colors across a theme switch) under
  `-W error::UserWarning`; vision review (image-reviewer agent, six fresh
  captures): "acceptable to show", all changes present and clean; review
  report appended to `run/reports/2026-10-28.md` (0 bugs, O16–O19 new
  suggestions).

## M4b follow-up — panel 3: propagation direction instead of f on the vertical axis (2026-10-28)

User remark on the growth-direction plane introduced in M4b: the vertical axis
should carry the propagation direction directly, not the signed frequency
(whose direction meaning depends on the sign of κ).

- `create_continuous_spectrum_figure` renamed to
  `create_growth_direction_figure`: the plane now plots growth rate
  σ = Re(ln λ)/dt (x) over the signed phase velocity c = −2πf/κ (y); the sign
  of c IS the direction (above the zero line toward +x, below toward −x), so
  dragging an eigenvalue across the real axis moves its point across the zero
  line — universally, independent of κ's sign. Conjugate partners coincide
  exactly (σ and c are equal for a pair: one physical wave, one point).
- New `_phase_slopes()` helper (unwrapped arg(φ) slope per mode, the
  wavenumber κ); `_update_spectrum` derives c = −2πf/κ per mode (zeroed for
  shapes without phase slope — they do not propagate) plus a hover direction
  column ("+x"/"−x"/"—"); both plane axes are data-driven now. The direction
  readout reuses the helper (no duplicated polyfit).
- Axis labels: "σ [1/s]" and "c = −2πf/κ [units/s]"; hover shows mode, σ, f,
  c, and the travel direction; spectrum hint and Start-panel description
  updated.
- Verification: pytest 15 passed; `run/verify_gui_polish.py` extended (velocity
  formula check per mode, direction strings, data-driven y-range, and a
  real-axis drag flipping the plane velocity sign) — all green under
  `-W error::UserWarning`; vision review of tab3: "acceptable to show", no
  regressions in the other panels.

## M4b follow-up 2 — travel-direction triangles in the eigenvalue plane (2026-10-28)

User clarification: the previous "instead of f" remark targeted the shaded
background of the eigenvalue plane (panel 2), not panel 3. Design decision
confirmed with the user after a numeric check showed that a static
half-plane direction label cannot be correct: both members of a fitted
conjugate pair travel the SAME direction (verified: all fitted generated
waves share +x regardless of half-plane), and the half-plane↔direction
association of a pair flips once it is dragged across the real axis.

- The eigenvalue plane now shows a small travel-direction triangle above each
  eigenvalue marker (same pair color, selection emphasis, rotated toward +x or
  −x; NaN coordinates hide it for non-propagating modes such as the offset
  mode). The triangles live in the SAME data source (columns ``dir_x``/``dir_y``/
  ``dir_angle``/``dir_color``/``dir_size``), are refilled with the markers, and
  the drag handler updates them so they follow a dragged eigenvalue and flip
  when it crosses the real axis.
- Shading labels changed from "f > 0"/"f < 0" to "Im λ > 0"/"Im λ < 0" (the
  shading stays the frequency-sign backdrop; the triangles carry the
  direction). New helpers: `_phase_velocities()` (shared by the triangles and
  the growth-direction plane) and `_direction_marker_columns()`; the
  growth-direction plane in panel 3 stays as approved.
- Hints updated; verification extended (triangle geometry, NaN hiding for the
  offset mode, label texts, triangle follow + flip during a real-axis drag) —
  all green under `-W error::UserWarning`, pytest 15 passed; vision review of
  tab2: "acceptable to show", no regressions.

## M4b follow-up 3 — reset buttons did not fire in the browser (event subscription bug) (2026-10-28)

User report: the reset button was not resetting the eigenvalues in the
figure. Root cause found by browser-level debugging (client-side data patch +
real button clicks via Playwright, server-log and event-pipeline tracing):

- **Root cause**: `on_click` handlers were registered in `_attach_callbacks`
  AFTER `doc.add_root(...)` (inside `_build_layout`). Bokeh resolves incoming
  events via `document.callbacks._subscribed_models`, populated by
  `_update_event_callbacks()` — which runs only in `Model._attach_document`,
  i.e. at attach time. Models whose callbacks are registered after the attach
  are never subscribed, so incoming `ButtonClick` events are silently dropped
  (no client or server error). Property-change callbacks (`on_change`) use a
  different path and were unaffected — which is why dragging, sliders, and
  auto-rerun all worked while every `Button` (Run DMD included, masked by
  auto-rerun; the Start panel's Open buttons excepted — they were wired
  before the root was added) was dead in the browser.
- **Fix**: `__init__` now adds the root AFTER `_attach_callbacks` (with an
  explanatory comment); `_build_layout` only builds and stores the layout.
- **Verification**: headless check of `document.callbacks._subscribed_models`
  (11 buttons subscribed) and document-level event dispatch
  (`doc.callbacks.trigger_event(ButtonClick(...))`) in
  `run/verify_gui_polish.py` (all reset invocations now go through the real
  dispatch path); browser-level Playwright test: client-side patch (as the
  PointDrawTool commits) → real reset click → eigenvalues restored in the
  figure and status "reset to the fitted values"; Run DMD, reset parameters
  (defaults restored: rank automatic (10) → (6)), and Open navigation all
  fire. pytest 15 passed.
- Pitfall documented in AGENTS.md (wire `on_click` before `doc.add_root`).

## M4c — traveling ↔ standing morph (mode-shape quadrature editing) (2026-10-28)

First item of the "teaching concepts" ideas (user decision from a brainstorm:
start with the morph only, slider-controlled, x–t heatmap skipped for now).

- **Concept**: a mode shape's quadrature mix decides traveling vs standing:
  `φ′ = Re(φ_fit) + i·t·Im(φ_fit)`, t ∈ [0, 1]. The decomposition is exact
  for linear-phase shapes: φ′ = (1+t)/2 · e^{+iκx} + (1−t)/2 · e^{−iκx} — a
  standing wave IS two equal opposite-traveling waves. t = 0 freezes the
  traveling-wave animation into an in-place oscillation; the phase view shows
  the flattening quadrature; the growth-direction plane keeps the effective
  c of the traveling component (verified numerically that the fitted slope
  stays ≈ κ across t).
- **Widget**: "standing ←→ traveling" slider (0–1, step 0.01) in panel 4's
  control row; typed values clamped server-side; `_syncing_morph` guard.
- **State**: edited shapes are written into `result.modes` (same pattern as
  eigenvalue/amplitude edits), the conjugate partner is conj-mirrored (the
  pair stays one physical wave), and the morph is relative to a new
  `_fit_modes` baseline (cached in `_update_quantities`, cleared in the
  zero-mode branch, restored by `_reset_to_fit` together with the slider → 1;
  the slider also syncs to the selected mode's mix factor on mode switching
  via `_morph_factor`, a projection onto the fitted imaginary part).
- **Readout**: for t < 1 the direction readout reports the exact split, e.g.
  "mixed wave: 75% traveling toward −x → + 25% counter-traveling (standing
  fraction)".
- **Semantics**: re-sorting bakes the current morph into the new baseline
  (same as the |b| scale); shapes without imaginary part (offset mode) are a
  visual no-op.
- **Bug fixed during the audit**: the non-reselect path of `_reset_to_fit`
  never refreshed the mode-shape views (latent gap — shapes only changed via
  sort before); `_update_mode_view()` added, regression-guarded in the verify
  script.
- Verification: pytest 15 passed (no core changes); `run/verify_gui_polish.py`
  extended (morph math, partner mirroring, factor recovery, mode-switch sync,
  readout split, clamping, reset + stale-shape-view guard, sort stability,
  real-shape no-op) — all green under `-W error::UserWarning`; vision review
  of panel 4: "acceptable to show", no regressions in panels 2/3; review
  report appended (B2 fixed, R2/O24–O26 new).

## M4c follow-up — stacked Re/Im subplots (see the morph at a glance) (2026-10-28)

User remark: show the real and imaginary parts in two subplots at the same
time so the morph slider's effect is directly visible.

- The "Real part" and "Imaginary part" sub-tabs merged into one "Real &
  imaginary" sub-tab with two stacked subplots (Re above, Im below, each
  1000×300) sharing a single `Range1d` x-range object — one zoom/pan syncs
  both. "Phase" and "Propagation direction" keep their own sub-tabs.
- `create_mode_component_figure(quantity, x_range=None, height=420)` gained
  optional shared-range and height parameters.
- Gotcha (documented in AGENTS.md): `TabPanel` titles are plain text — an
  HTML entity ("Real &amp; imaginary") rendered literally; plain "&" is fine.
- Verification: pytest 15 passed; `run/verify_gui_polish.py` extended (shared
  x-range identity, subplot heights, sub-tab titles); vision review of the
  default sub-tab with the morph slider at 0.4: "acceptable to show" — Re
  unchanged, Im visibly scaled to ~40%, tab label clean (one stale-server
  capture produced a false "&amp;" finding first — recaptured fresh).

## M4c follow-up 2 — constant quadrature ruler, standing arrow, merged propagation view (2026-10-28)

User remarks on the morph revision:

- **Constant ylim for the quadrature subplots**: the Re/Im subplots now share
  ONE `Range1d` y-ruler sized by the fitted quadratures of the selected mode
  (`_update_mode_view`, margin 1.1) — the morph slider scales the Im curve
  inside a fixed frame instead of rescaling the axis (vision-verified:
  identical axis limits at t = 1 and t = 0.4).
- **No direction arrow for standing waves**: below
  `_MORPH_STANDING_THRESHOLD` (0.05) the arrow is hidden and the readout
  reports "standing wave — no net propagation (equal superposition of
  opposite-traveling waves)" with κ/λ_w from the fitted shape (the morphed
  phase has no meaningful slope at t = 0).
- **Arrow length scales with the net propagation**: length =
  clamp(0.09·t·|c|, 0.04, 0.9), centered on the plot — full length for the
  pure traveling wave, shrinking proportionally as the morph slider mixes in
  the counter-traveling wave (vision-verified ratio ≈ 0.31 at t = 0.3),
  hidden below the standing threshold.
- **Propagation view merged into the shape sub-tab**: the separate
  "Propagation direction" sub-tab is gone; the traveling-wave figure, its
  time slider + play, the readout, and the direction hint now live inside the
  first sub-tab (renamed "Shape & propagation"), so Re, Im, and the wave
  animation all react to the morph slider at once. The direction figure is
  now also registered for theming (it had been missed — dark-mode gap).
- Verification: pytest 15 passed; `run/verify_gui_polish.py` extended (y-ruler
  constancy + size, direction figure placement, arrow hidden/standing
  readout, arrow length formula); vision review of tab4 at t = 1 and t = 0.4:
  "acceptable to show" (the flagged 70/30 vs 60/40 split was an error in the
  review prompt — the app's (1+t)/2 split is correct).

## M4c follow-up 3 — one-row layout without scrolling, ghost reference curves (2026-10-28)

User remarks: the morph scaling was not clearly visible, and all three
subplots (Re, Im, traveling wave) should be visible at the same time without
scrolling.

- **One-row layout**: the "Shape & propagation" sub-tab now shows Re | Im |
  traveling wave SIDE BY SIDE (410×340 each, spacing 5) with a shared
  `Range1d` x-range (one zoom/pan syncs all three); below it a row with the
  time slider + play (left) and the direction readout (right), then the three
  hints. Measured vertical overflow beyond a 900 px laptop viewport: 0 px —
  the whole panel fits without scrolling.
- **Ghost reference curves**: the Im subplot and the traveling-wave view draw
  a faded dotted line of the FITTED state (source column ``reference``, filled
  from the `_fit_modes` baseline; the wave reference uses the fitted shape at
  the current time level) — the morph scaling and the standing transition are
  now visible against the original, not from memory. The phase view (own
  sub-tab) gained the same ghost plus a "fitted (t = 1)" legend entry.
- Builder signatures extended: `create_mode_component_figure(..., width=1000)`
  and `create_direction_figure(x_range=None, height=420, width=1000)`.
- Verification: pytest 15 passed; `run/verify_gui_polish.py` extended (shared
  x-range across all three figures, row placement, sizes, reference columns
  incl. the fitted wave reference); vision review of viewport captures at
  t = 1 and t = 0.4: "acceptable to show" — three subplots fully visible,
  constant ruler confirmed, Im/wave scaling vs the ghost references and the
  70/30 mixed-wave line verified.

## M4c follow-up 4 — combined quadrature figure (Re & Im in one axes) (2026-10-28)

User idea (confirmed as Variant A): combine the Re and Im subplots into ONE
figure — more space, and the phase difference between the quadratures is
directly visible.

- New `create_quadrature_figure`: solid Re(φ) (blue) and Im(φ) (orange) in
  one axes plus the faded dotted fitted-Im reference, with a click-to-hide
  legend. The quarter-wavelength spatial offset between the quadratures —
  their phase relation, whose sign encodes the propagation direction — can
  be read directly off the plot; the morph scaling is judged against Re and
  the ghost in the same frame.
- Layout: the "Shape & propagation" sub-tab now has TWO grouped subplots
  (quadratures | traveling wave, 615×340 each, shared x-range, constant
  shared y-ruler sized by the fit); the phase view keeps its own sub-tab.
  State: `self._quadrature_figure/_quadrature_source` and
  `self._phase_figure/_phase_source` replace the `_mode_views` dict.
- Verification: pytest 15 passed; `run/verify_gui_polish.py` updated (new
  attributes, sizes, shared ranges, reference checks, stale-view guard);
  vision review of viewport captures at t = 1 and t = 0.4: "acceptable to
  show" — three curves distinguishable, π/2 shift visible, Im at ~40 % of
  the ghost, legend clean, no-scroll fit kept (0 px overflow).

## M4c follow-up 5 — quadrature legend moved outside (2026-10-28)

User remark: the legend should be three columns, centered top, outside the
plot. The quadrature figure's legend is now laid out horizontally with
`ncols = 3` and moved outside the frame via `fig.add_layout(fig.legend[0],
"above")` with `location = "center"` — the curves are no longer covered.
Verification: pytest 15 passed; headless verify green; vision review
("acceptable to show": legend above the frame, three entries side by side,
no curve overlap, 0 px overflow).

## M4d — space-time heatmap (the space/time phase decoupling, visualized) (2026-10-28)

User-flagged confusing point: arg(λ) = 2πf·Δt is a phase rotation per time
step (temporal), arg(φ) = κx + θ₀ is the spatial phase — dragging an
eigenvalue changes the animation speed and c = −2πf/κ but never the shape or
wavelength. Planned via brainstorm (options: x–t heatmap, snapshot trail,
dispersion tie, phase dial); user picked the heatmap first (selected mode
only), richer explanations later.

- New **"Space–time" sub-tab** in panel 4: heatmap of the selected mode's
  contribution Re{φ·b·λᵗ} over x (horizontal) and t (vertical, time
  increasing upward), diverging blue-white-red color map with color bar.
  The lesson: horizontal stripe spacing = wavelength λ_w = 2π/κ (a property
  of the SHAPE), vertical spacing = temporal period T = 1/f (a property of
  the EIGENVALUE), stripe slope = propagation speed c = −λ_w/T — the
  decoupling as two independent spacings.
- `create_space_time_figure(x_range=None, height=420, width=615)` with a
  smooth diverging palette (`_SPACE_TIME_PALETTE`, interpolated from bokeh's
  `RdBu[11]` to 256 entries) and a `ColorBar`; returns
  `(figure, source, mapper)`.
- Glyph orientation: bokeh's Image glyph renders array row 0 at the TOP and
  anchors at the bottom-left — the app stores the TIME-FLIPPED field
  (rows = time, columns = space) plus the vertical extent in the source
  (columns ``field``/``dh``) and updates the mapper's symmetric limits and
  the y-range per refresh.
- `_update_space_time_view`: refreshed only when its inputs change —
  `_update_mode_view` (refit/sort/mode switch/morph/reset) plus explicit
  calls in `_on_eigenvalue_drag` and `_on_amplitude_change` — deliberately
  NOT per animation tick (the heatmap shows all times and does not depend
  on the time cursor).
- Physics note verified in-browser: the default mode (f = 10 Hz) shows
  nearly horizontal stripes — correct, the wave crosses the 1-unit domain
  ~3.3 times per second; dragging the eigenvalue down to 1 Hz unfolds
  clearly diagonal bottom-left→top-right stripes (3 temporal periods over
  3 wavelengths). Vertical density changes with f, horizontal spacing with
  the shape — the decoupling at a glance.
- Verification: pytest 15 passed; `run/verify_gui_polish.py` extended (field
  shape and analytic match at t = 0/50/last with the time flip, symmetric
  mapper limits, t-range, refresh on eigenvalue edits, standing degeneration
  into vertical stripes); vision review of fast and slow captures:
  "acceptable to show" (an intermediate "blocking" finding was the review
  prompt's wrong expectation about the tilt at f = 10 Hz — the shallow tilt
  is the correct physics).

## M4d follow-up — panel 5: checkbox wrapping + reconstruction y-limits (2026-10-28)

User report: with many modes active, panel 5 ran off the screen and the
reconstruction plot's ylim was not adjusted.

- **Checkbox wrapping**: the mode-selection `CheckboxGroup` (inline) now has
  `width = 1000` and a stylesheet forcing `flex-wrap: wrap` on the inner
  `.bk-input-group` container (the group's flex row previously did not wrap,
  so 24 labels pushed the page ~1850 px beyond the viewport). With 24 modes
  the labels wrap into three rows within the figure width (0 px horizontal
  overflow, vision-verified).
- **Reconstruction y-limits**: the fixed y-range of the reconstruction view
  was sized from the ORIGINAL signal only (`_update_field_view`), so edits
  (|b| scaling, eigenvalue drags with growth, morphs) could push curves
  outside the frame. New `_update_time_limits()` sizes the range from the
  reconstruction and the original signal over ALL time levels — per user
  refinement it is based on the reconstruction and stays CONSTANT under
  checkbox toggles (the checked selection sum and per-mode terms were
  dropped; the original is included so silencing the reconstruction cannot
  clip it). Refreshed on refit and eigenvalue/amplitude/morph edits — not
  per animation tick. The signal view (panel 1) keeps its signal-sized
  range.
- Verification: pytest 15 passed; `run/verify_gui_polish.py` extended (12
  signal modes → 24 checkboxes wrapped within the width; y-range matches the
  analytic limit for all-checked and single-checked selections at |b| scale
  2.0); vision review of panel 5 with 24 modes (default and amplified):
  "acceptable to show" (checkboxes in three neat rows, y-range expanded
  ~[-10, 10] → ~[-15, 15], no clipping).

## M4d follow-up 3 — reconstruction view clipped at ±2: DataRange1d pitfall (2026-10-28)

User report: with Nₘ = 20 and playback, the original and reconstruction
curves went out of bounds in panel 5 (verified on the user's actual screen
via a desktop screenshot: y ticks 2/0, curves clipped, mode 0 at
f = −5.70 Hz, mode 39 at f = 15.48 Hz — the session had heavy eigenvalue
editing history).

- **Root cause**: the signal/reconstruction/wave figures were created WITHOUT
  an explicit y-range, so bokeh assigned its auto-ranging `DataRange1d`. A
  `DataRange1d` recomputes its limits CLIENT-side from the rendered data and
  ignores/overrides server-side `.start`/`.end` writes — in the user's
  long-lived browser session (which survived server restarts and many
  edits) the client's computed range (±2) no longer matched the server's
  data, so the curves clipped.
- **Fix**: the three figures are now created with plain `Range1d` objects —
  the application owns the limits and they reach the browser verbatim
  (the same pattern the quadrature view's constant ruler already used).
  Headless type check added to the regression script
  (`type(fig.y_range) is Range1d` for all app-managed figures).
- Process note: the first browser verification ran against a STALE debug
  server holding port 5099 (my pattern-based pkill missed it; the new
  server died with "Address already in use") — the measurements were from
  old code. Killed by PID from `ss -ltnp`, re-verified on a fresh server:
  `range_type Range1d`, reconstruction fits through |b| scale 2.0 and an
  extreme growth drag (radius × 1.5 → range ±1e53, honest physics of the
  exploding mode).
- Pitfall documented in AGENTS.md (DataRange1d vs app-owned Range1d).
- Verification: pytest 15 passed; `run/verify_gui_polish.py` all green under
  `-W error::UserWarning`.

## M4d follow-up 4 — reconstruction view: outside legend + tight margin (2026-10-28)

User remarks on panel 5's "field at current time" plot: move the legend
outside (same as panel 4) and decrease the y-range margin.

- The four-entry legend ("original", "reconstruction", "sum of selected
  modes", "selected eigenvalue") is now a four-column horizontal legend
  centered OUTSIDE the frame above the plot (`ncols = 4` +
  `add_layout(..., "above")` — the same pattern as the quadrature view).
  The plot was widened 1000 → 1150 px so the four long labels fit in ONE row
  (at 1000 px bokeh wrapped them 2+2).
- The y-range margin was tightened 1.1 → 1.05: the reconstruction uses the
  plot height better while playback stays inside (the range is sized by the
  global maximum over all time levels).
- Verification: pytest 15 passed; `run/verify_gui_polish.py` updated (margin
  1.05, legend-outside assertion); vision review with |b| scale 2.0:
  "acceptable to show" — one-row legend above the frame, curves fill the
  height without clipping.

## Cleanup — documentation, plans, artifacts, dead code (2026-10-28)

Housekeeping pass before the next feature:

- **PLAN.md**: the original v1.0 plan sections (1–8) are now explicitly marked
  as historical reference; the architecture tree matches the real module
  layout (samples, styling.py, DMDResult); the milestones table gained M4b/
  M4c/M4d rows with commit hashes; the decisions/drag-semantics tables note
  what is implemented vs. still a future idea (arg(b) editing, add/remove
  modes).
- **modules.md**: `_update_field_view`/`_update_time_limits` split documented;
  the phase builder entry updated after the rename.
- **Artifacts**: all screenshots in `run/screenshots/` deleted (regenerable
  via `run/screenshot_tabs.py`), `.coverage` removed.
- **Dead code removed** (review findings O16/O28): the unused `_MODE_DEFAULTS`
  constant and the generic `create_mode_component_figure` (only its phase
  view was used) replaced by a dedicated `create_phase_figure()`.
- Verification: pytest 15 passed; `run/verify_gui_polish.py` all green under
  `-W error::UserWarning`.
