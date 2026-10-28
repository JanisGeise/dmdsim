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
