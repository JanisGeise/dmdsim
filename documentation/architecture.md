# Architecture

## Purpose and guiding principle

`dmdsim` is an interactive playground for Dynamic Mode Decomposition (DMD): it
generates synthetic modal line fields, decomposes them with flowTorch, and lets the
user explore and modify every DMD quantity visually — with the goal of building an
intuitive understanding of eigenvalues, mode shapes, amplitudes, phases, and the
reconstruction.

The guiding principle is a **GUI-independent core**: all math lives in
`dmdsim.signals` and `dmdsim.core` and operates on plain numpy arrays. The Bokeh
application (`dmdsim.app`) is a thin layer that calls the same functions a notebook
or CLI script would call. This keeps the GUI replaceable and the core testable.

## Layers

```
dmdsim/
├── signals/    synthetic data generation (numpy only)
│   └── generator.py   Mode + ModalSignal
├── core/       DMD engine wrapper, quantity container, field synthesis (numpy only)
│   ├── engine.py      fit_dmd() — the single entry point into flowTorch
│   ├── quantities.py  DMDResult — eigenvalues, modes, amplitudes, pairs
│   └── synth.py       synthesize() + mirror_conjugates() — the playground math
├── app/        Bokeh server application (thin layer)
│   ├── figures.py     figure/data-source builders
│   └── main.py        DMDPlayground: widgets, callbacks, layout, entry points
└── tests/      pytest suite for the core (app layer intentionally untested)
```

Dependency direction: `app → core → signals` and `core/signals → flowTorch, numpy`.
Nothing in the core knows about Bokeh; nothing in `app` implements math.

## Data flow in the GUI

1. **Parameter change** (any widget in the "Signal" step changes):
   `DMDPlayground._on_parameter_change` → `_run_dmd()` (immediately when the
   auto-rerun toggle is active, otherwise only via the *Run DMD* button).
2. **`_run_dmd`**:
   `_build_signal()` reads the widgets and creates a `ModalSignal`
   → `fit_dmd(signal.samples, dt, rank)` (numpy → torch float64 → flowTorch DMD →
   numpy quantities) → `synthesize(...)` caches the full reconstruction.
3. **View updates** (`_update_*` methods): eigenvalue plane, spectrum, mode viewer,
   heatmap, sliders/checkbox bounds, status line, and readout are refilled from the
   new `DMDResult`; Bokeh pushes the patches to the browser.
4. **View-only interactions** (time slider, play animation, mode selection
   checkboxes, mode index slider) call `_update_mode_view` / `_update_time_view`
   only — **no refit**. The reconstruction is sliced from the cached field.

The layout is a `Tabs` widget with `tabs_location="left"`; each step
(1 · Signal, 2 · Eigenvalues, 3 · Spectrum, 4 · Mode shapes, 5 · Reconstruction)
is one `TabPanel`. All panels stay alive while hidden, so widget state survives
tab switching. The Signal panel is the start menu (create the signal, animate it,
run the fit) and ends with a live signal animation; the Mode shapes panel nests
horizontal sub-tabs (Real part / Imaginary part / Phase / Propagation direction)
that share one mode slider. The Signal and Reconstruction views use two
synchronized time sliders/play toggles driving one shared time cursor.

## The playground math

The DMD reconstruction reads
`:math:`x(s,t) ≈ Re{ Σ_i φ_i(s) · b_i · λ_i^t }`` with mode shapes `φ_i`, complex
spectral coefficients `b_i`, and discrete eigenvalues `λ_i = exp((σ ± 2πif)·dt)`.
`core/synth.py:synthesize` implements exactly this; it is both the verification of a
fit (round trip) and the engine of the playground.

Editing semantics (used by the GUI and available in the core):

- **Eigenvalue radius** ↔ growth rate `σ = ln|λ| / dt` (unit circle = neutral).
- **Eigenvalue angle** ↔ frequency `f = arg(λ) / (2π·dt)`.
- Mode shapes `φ_i` and amplitudes `b_i` are kept from the fit; only `λ_i^t` (and
  optionally `b_i`) change.
- **Conjugate pairs**: real data produce pairs `(λ, λ̄)` with `(b, b̄)`, `(φ, φ̄)`;
  `mirror_conjugates(values, pairs, sources)` enforces this after an edit so the
  rebuilt field stays real-valued.

.. note:: Pair structure pitfall
    `DMDResult.pairs` is detected from the *current* values. Cache it right after a
    fit and reuse the cached list while editing; a recomputation after a manual
    edit would no longer see the pair that is being edited.

## Key design decisions

| Decision | Rationale |
|---|---|
| numpy at the engine boundary | `fit_dmd` converts numpy → torch float64 for flowTorch and returns numpy; the GUI and tests never see torch |
| float64 through the fit | flowTorch defaults to float32; the SVD/operator path preserves the input dtype, so float64 input gives double-precision eigenvalues |
| flowTorch as the only DMD engine | `core/engine.py` is the single wrapper; no DMD math anywhere else; variants can be added in one place |
| Generator produces spatial line fields | "1D" means one spatial dimension (traveling waves on a line), directly analogous to later 2D OpenFOAM fields; a scalar time series would make plain DMD rank-limited (≤ 1 mode) |
| Unit-norm mode shapes | flowTorch returns modes normalized to unit norm; amplitudes `|b|` scale accordingly (e.g., a constant field of 2.0 on 64 points gives \|b\| = 16) — documented to avoid confusion in the GUI |
| Absolute imports in `app/` | Bokeh's script/directory loader executes `main.py` outside the package context; relative imports would break under `bokeh serve` |
| Watchdog server shutdown | Bokeh's built-in session cleanup job never fires when the io loop is managed manually (Tornado 6.6 binds periodic jobs to `IOLoop.current()` at start time, not to the server's private loop); the app therefore runs its own watchdog that stops the server ~2 s after the last session disconnects |
| `Tabs(tabs_location="left")` for the GUI | native left sidebar with one submenu per step; panels stay alive while hidden |

## Roadmap position

See `PLAN.md` for milestones. M0 (setup) and M1 (core) are done, M2 (GUI v1) is
done including the sidebar layout; next is M3 (interactive eigenvalue dragging),
then an improvement round (M4) driven by hands-on feedback, and 2D OpenFOAM data
(M5) later.
