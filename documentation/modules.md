# Modules

Module-by-module reference. Signatures are abbreviated; see the docstrings for
full details. Convention reminders: Sphinx/reST docstrings everywhere, modern type
hints, import layout grouped and sorted by statement length, validation via
`_check_*` methods with `assert`, every module ends with the
`if __name__ == "__main__": pass` guard (except declaration-only files).

## `src/dmdsim/signals/generator.py`

Synthetic modal line fields on a uniform space-time grid.

- `Mode(wavenumber, frequency, amplitude=1.0, growth_rate=0.0, phase=0.0)` — one
  modal component contributing
  `A·exp(σ·t)·cos(κ·s − 2πf·t + φ)`. In the DMD picture this is a conjugate
  eigenvalue pair `λ = exp((σ ∓ 2πif)·dt)` with shapes `exp(±iκs)` and amplitudes
  `A·exp(±iφ)/2`. Validation: finite parameters, non-negative amplitude.
- `ModalSignal(modes, dt, n_space, n_time, dx=1.0, offset=0.0, noise_std=0.0,
  seed=None)` — evaluates the modal superposition (plus offset and optional seeded
  Gaussian noise) once at construction; `samples` is the real-valued field with
  shape `(n_space, n_time)`, one snapshot per column. Properties `space` and
  `time` return the grids.

## `src/dmdsim/core/engine.py`

The single wrapper around flowTorch.

- `fit_dmd(samples, dt, rank=None, optimal=False, tlsq=False) -> DMDResult` —
  converts the snapshot matrix to a torch float64 tensor (flowTorch preserves the
  input dtype in the SVD/operator path), runs `flowtorch.analysis.DMD`, and
  converts modes, amplitudes, and eigenvalues back to numpy. `rank=0`-style
  "automatic" is expressed by passing `rank=None` (flowTorch optimal truncation).

## `src/dmdsim/core/quantities.py`

- `DMDResult(modes, amplitudes, eigvals, dt)` — container with everything the GUI
  and synthesis need; stores complex numpy arrays (complex128) and validates
  shapes/dt in `_check_dimensions`.
  - attributes: `modes` `(n_space, n_modes)`, `amplitudes` `(n_modes,)`,
    `eigvals` `(n_modes,)`, `dt`
  - properties: `n_modes`, `n_space`, `frequencies` (= Im(log λ)/(2π·dt), Nyquist
    bounded via principal log branch), `growth_rates` (= Re(log λ)/dt), `pairs`
    (conjugate pair index tuples, detected from the *current* values — cache it
    before editing)
  - method: `mode(index)` returns the complex shape `φ_i` `(n_space,)`
- `_PAIR_TOL = 1e-8` — relative tolerance of the pair detection (eigenvalues,
  amplitudes, and mode shapes must all be conjugate).

## `src/dmdsim/core/synth.py`

- `synthesize(modes, amplitudes, eigvals, dt, n_time) -> np.ndarray` — rebuilds the
  real field `Re{Σ φᵢ·bᵢ·λᵢᵗ}` with integer time levels; works with modified
  quantities (the playground). The `dt` argument is currently unused by the
  formula (integer powers of λ) and kept for API symmetry — see review finding O1.
- `mirror_conjugates(values, pairs, sources=None) -> np.ndarray` — returns a copy
  with `values[j] = conj(values[i])` per pair; `sources` marks dragged indices so
  the edited member acts as the source. Used for eigenvalues and amplitudes alike.

## `src/dmdsim/app/figures.py`

Figure and `ColumnDataSource` builders; no application logic. Each creator returns
the figure plus the data source (and extras where needed):

- `create_eigenvalue_figure()` — eigenvalue scatter, unit circle, hover with mode
  index/λ/f/σ; 620×560.
- `create_spectrum_figure()` — stem-style |b| over frequency with hover; the x
  range is set to ±Nyquist by the app; 620×560.
- `create_mode_component_figure(quantity)` — single-quantity mode shape figure;
  ``quantity`` is one of ``real``, ``imag``, ``phase`` (plain views) or
  ``direction`` (phase view titled for the propagation analysis); phase-like
  views have a fixed ±π y-range; source columns ``s`` and ``value``; 1000×420.
- `create_field_heatmap(show_cursor=True, title="field x(s, t)")` — `image` glyph of
  `x(s, t)` with shared `LinearColorMapper` (RdBu, symmetric limits) and an
  optional `Span` as time cursor (`show_cursor=False` for the static signal
  preview); returns `(figure, source, mapper, span)`; 500×380.
- `create_time_curves_figure()` — original, reconstruction, Σ selected modes, and
  selected mode at the current time; 500×380.

Expected source columns are documented in each docstring; the app fills them in
`_update_*` methods.

## `src/dmdsim/app/styling.py`

Central visual styling of the application (modern light theme, one accent color
per step for visual distinction):

- `_STEP_COLORS` — per-step `(tint, accent)` pairs: setup blue, eigenvalues green,
  spectrum amber, modes purple, reconstruction red.
- `step_colors(step)` — accessor for the pair above.
- `style_figure(fig, step)` — white plot area on the tinted panel background,
  accent-colored bold title, muted gray axes/grid/legend (legend styling only
  applied when the figure actually has legends).
- `header(text, step)` — colored section header element for panels.
- `hint(text, width=None)` — muted inline explanation block.
- `info_box(text="")` — neutral box for dynamic status/readout text.
- `tab_stylesheet(accent="#1a73e8", vertical=True)` — `InlineStyleSheet` for a tab
  strip (rounded buttons, hover state, accent-colored active tab); vertical
  strips get uniformly wide, left-aligned buttons (main sidebar), horizontal
  strips compact buttons (nested submenus); uses the `.bk-tab`/`.bk-active`
  classes.

## `src/dmdsim/app/main.py`

The application wiring. Uses absolute imports (Bokeh loader constraint) and
silences flowTorch's INFO logging at import time.

- Constants: `_MAX_MODE_EDITORS = 3`, `_NOISE_SEED = 42`, `_PLAY_INTERVAL_MS = 60`,
  `_WATCHDOG_INTERVAL_MS = 500`, `_SHUTDOWN_GRACE_MS = 2000`, `_MODE_DEFAULTS`
  (wavenumber, frequency, amplitude, growth, active per editor slot), and the
  `_HINT_*` inline explanations (per panel and per mode quantity).
- `DMDPlayground(doc)` — builds widgets, figures, layout, callbacks and runs the
  initial fit:
  - `_build_widgets`: per-mode editors (toggle + κ/f/A/σ spinners), grid/offset/
    noise widgets, rank (0 = automatic), auto-rerun toggle, styled Run button,
    status info box, shared mode slider, TWO synchronized time sliders and play
    toggles (signal animation in the start menu, reconstruction view), mode
    checkbox group, readout and propagation-direction info boxes.
  - `_build_layout`: `Tabs(tabs_location="left")` with one `TabPanel` per step —
    "1 · Signal" (start menu: modal components + grid & noise on the left,
    signal animation with cursor/slider/play on the right, DMD fit + status
    below), "2 · Eigenvalues", "3 · Spectrum", "4 · Mode shapes" (nested
    horizontal sub-tabs: Real part / Imaginary part / Phase / Propagation
    direction, sharing the mode slider), "5 · Reconstruction". All panels stay
    alive while hidden. Figures are color-coded per step; the main sidebar and
    the nested strip carry separate tab stylesheets (vertical/horizontal).
  - `_attach_callbacks`: parameter widgets → `_on_parameter_change` (refit on
    auto-rerun, otherwise mark stale); Run button → `_run_dmd`; mode slider and
    checkbox → `_on_view_change` (view refresh without refit); both time sliders
    → `_on_time_change` (two-way sync + time view refresh); both play toggles →
    `_on_recon_play`/`_on_signal_play` → `_set_playback` (synced labels/states,
    one shared periodic callback advancing the time).
  - `_run_dmd`: `_build_signal` → `fit_dmd` → cache reconstruction → update all
    views, slider/checkbox bounds, and the status line (mode count + round-trip
    error).
  - `_mode_field(index, t_index)`: real contribution of one mode at one time level
    (`Re{φᵢ·bᵢ·λᵢᵗ}`) used by the time view.
  - `_update_*` methods: `_update_field_view` (signal animation heatmap + time
    view heatmap, color limits, slider bounds), `_update_quantities` (eigenvalues,
    spectrum, axis ranges), `_update_mode_slider`/`_update_time_slider` (bounds +
    clamping of both sliders), `_update_mode_view` (fills the four single-quantity
    mode sources), `_update_direction_readout` (phase slope κ → wavelength, phase
    velocity, travel direction with arrow), `_update_mode_select` (resets
    selection before relabeling to avoid stale indices), `_update_time_view`
    (both time cursors + curves), `_update_readout`.
  - `_on_time_change` / `_set_playback` / `_advance_time`: two-way slider sync
    with a re-entrancy guard and one shared animation via
    `Document.add_periodic_callback` (the returned token is kept for removal).
- `modify_doc(doc)` — entry point for `bokeh serve` (script or directory app).
- `main()` — entry point for the `dmdsim-app` console script: starts the server on
  localhost, opens the browser, and arms a watchdog (Tornado `PeriodicCallback`
  started from inside the running loop) that stops the server ~2 s after the last
  browser session disconnects. Background: Bokeh's built-in session cleanup job
  does not run under manual io loop management (Tornado 6.6 binds periodic jobs to
  `IOLoop.current()` at start time), so the watchdog replaces it.

## `tests/`

Core coverage only (app layer intentionally untested, see PLAN.md §7):

- `test_generator.py` — shapes/grids, offset-only field, analytic single-mode
  values, seeded noise determinism, invalid parameters.
- `test_engine.py` — recovery of known frequencies/growth rates/eigenvalues,
  conjugate pair detection, invalid inputs.
- `test_synth.py` — fit round trip, analytic ground-truth synthesis, effect of
  eigenvalue modifications, conjugate mirroring semantics.

## Packaging

- `pyproject.toml` — setuptools, src layout, `requires-python >= 3.10`, runtime
  deps numpy + bokeh (flowTorch installed editable from `../flowtorch`, see
  README), console script `dmdsim-app = "dmdsim.app.main:main"`.
- `requirements.txt` — pinned environment incl. CPU-only torch and the minimal
  flowTorch import requirements (scipy, plotly, pandas, pillow, vtk, h5py,
  netCDF4).
