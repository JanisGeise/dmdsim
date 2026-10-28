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
3. **View updates** (`_update_*` methods): eigenvalue plane, spectrum,
   continuous-time eigenvalue plane, mode viewer, sliders/checkbox bounds,
   status line, and readout are refilled from the new `DMDResult`; Bokeh pushes
   the patches to the browser.
4. **View-only interactions** (time slider, play animation, mode selection
   checkboxes, mode index slider) call `_update_mode_view` / `_update_time_view`
   only — **no refit**. The reconstruction is sliced from the cached field.

The layout is a `Tabs` widget with `tabs_location="left"` inside an application
bar (title + light/dark theme `Switch`); the tabs are ordered **Start** (default
landing tab: DMD in a nutshell, workflow navigation with Open buttons, tips),
**1 · Signal**, **2 · Eigenvalues**, **3 · Spectrum** (amplitude spectrum and
growth-direction plane — growth rate σ over phase velocity c = −2πf/κ — side
by side), **4 · Mode shapes**
(nested sub-tabs: Real part / Imaginary part / Phase / Propagation direction),
and **5 · Reconstruction**. All panels stay alive while hidden, so widget state
survives tab switching. Four time sliders and play toggles (panels 1, 2, 4, 5)
are two-way synchronized and drive one shared time cursor. Every panel carries
a reset button: panel 1 restores the default signal parameters (and refits),
panels 2–5 restore the fitted DMD values (panel 5 additionally re-selects all
modes).

**Editing loop (M3):** dragging an eigenvalue in the plane (PointDrawTool)
commits once per gesture; `_on_eigenvalue_drag` writes the new eigenvalues into
the result with conjugate mirroring (cached pair structure), recomputes the
reconstruction via `synthesize`, and refreshes spectrum, λᵗ spiral, direction
view, and readout — the same feedback path is used by the |b| scale spinner.
Edits apply to the current fit; a refit restores the fitted values, and the
per-panel reset buttons (`_reset_to_fit`) restore them on demand. Locked
drags (unit-circle lock) are temporary experiments and reset when another
eigenvalue is selected. The eigenvalue plane shades the two half planes to
mark the sign of Im λ (equivalently the frequency sign), and a small triangle
above each eigenvalue points in that wave's travel direction — computed from
the signed phase velocity c = −2πf/κ, so it flips live when the eigenvalue is
dragged across the real axis (a static half-plane direction label would be
wrong: both members of a conjugate pair travel the same way). The
propagation-direction readout states the link explicitly
(`Im λ` sign → side → travel direction via `c = −2πf/κ`), so dragging across
the real axis visibly flips the direction. The spectrum panel reinforces the
same link: its growth-direction plane plots each mode's growth rate σ over
the signed phase velocity c = −2πf/κ (above the zero line toward +x, below
toward −x), so a real-axis drag moves the mode's point across the zero line;
conjugate partners coincide there (one physical wave).

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
| `Tabs(tabs_location="left")` + Start panel | the sidebar holds a Start landing tab (DMD in a nutshell + workflow navigation) followed by the numbered steps; panels stay alive while hidden |
| Pair colors across views | each conjugate eigenvalue pair shares one categorical color (`PAIR_PALETTE`, brighter `PAIR_PALETTE_DARK` in dark mode) so a physical mode is traceable across eigenvalue plane, λᵗ dynamics, spectrum, and reconstruction; unpaired modes are gray |
| In-place theming | light/dark themes are applied without rebuilding: figures/Divs are restyled from `THEMES` and the stored stylesheets' `css` strings are regenerated. `doc.clear()` is forbidden (it wipes session callbacks); a `GlobalInlineStyleSheet` is injected once, so theme switches **recreate and re-attach** the stylesheet objects |
| Editing semantics | dragged eigenvalues and |b| scale edits apply to the current fit and are fed back to the reconstruction live; unit-circle-locked drags are temporary (reset on eigenvalue selection); a refit restores the fitted values |

## Roadmap position

See `PLAN.md` for milestones. M0 (setup), M1 (core), M2 (GUI), the GUI revise
phase (M2R), M3 (interactive eigenvalue editing) and M4 (UX revision: start
panel, light/dark themes, beginner labels) are done. The dark-mode polish and
real 2D OpenFOAM data (M5) are deferred in the backlog; DMD variants and
session save/load are further backlog items.
