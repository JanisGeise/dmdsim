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
