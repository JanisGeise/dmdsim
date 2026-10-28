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
- `generate_modes(n_modes, f_min, f_max, seed=None, phase_speed=10.0)` — automatic
  modal composition for the GUI: evenly spaced frequencies within the bounds
  (predictable DMD spectrum), dispersion-tied wavenumbers
  :math:`\kappa = 2\pi f / c` (all waves share the phase speed and travel in
  :math:`+s`), seeded random amplitudes (0.5–1.5), growth rates (±0.5), and
  phases. Reproducible for identical arguments.

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
    `eigvals` `(n_modes,)`, `integral_contribution` `(n_modes,)` (Kou et al.
    2017 numbers provided by flowTorch, required for the ``"integral"`` sort),
    `dt`
  - properties: `n_modes`, `n_space`, `frequencies` (= Im(log λ)/(2π·dt), Nyquist
    bounded via principal log branch), `growth_rates` (= Re(log λ)/dt), `pairs`
    (conjugate pair index tuples, detected from the *current* values — cache it
    before editing)
  - method: `mode(index)` returns the complex shape `φ_i` `(n_space,)`
  - method: `sort(by)` reorders all stored quantities in place; criteria follow
    flowTorch: ``"amplitude"`` (descending |b|), ``"integral"`` (descending
    flowTorch integral contribution), ``"frequency"`` (ascending |f|); the
    reconstruction (sum over all modes) is invariant under sorting
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
  index/λ/f/σ; markers carry per-point pair colors and selection emphasis via the
  ``color``/``size`` columns; a `PointDrawTool` (``add=False`` — drag-only)
  makes the eigenvalues draggable; source columns ``re``, ``im``, ``index``,
  ``frequency``, ``growth``, ``color``, and ``size``; 620×560.
- `create_spectrum_figure()` — stem-style DMD spectrum of the mode amplitudes
  over frequency; stems and markers carry per-point stability colors (red
  growing, blue decaying, gray neutral) and selection emphasis via the
  ``color``/``width``/``size`` columns; dashed Nyquist spans are returned for
  the app to place; source columns ``frequency``, ``amplitude``, ``index``,
  ``growth``, ``color``, ``width``, ``size``; returns
  `(figure, source, nyquist_spans)`; 620×560.
- `create_mode_component_figure(quantity)` — single-quantity mode shape figure;
  ``quantity`` is one of ``real``, ``imag``, or ``phase``; the phase view renders
  the unwrapped phase as a line plus the wrapped ±π values as faded points;
  source columns ``s``, ``value``, and ``unwrapped``; 1000×420.
- `create_direction_figure()` — traveling-wave view of the selected mode with a
  direction arrow annotation (bokeh `Arrow` + `NormalHead`); source columns
  ``s`` and ``value``; returns `(figure, source, arrow)`; 1000×420.
- `create_mode_dynamics_figure()` — complex-plane animation of the selected
  mode's eigenvalue powers λᵗ (t starting at 1, i.e., at the eigenvalue itself):
  dashed unit circle, faded pair-colored markers for all eigenvalues, trail
  up to the current time and hoverable current-point marker; the trail color is
  set by the application (pair color of the selected eigenvalue); returns
  `(figure, trail_source, point_source, all_source, trail_renderer)`; 620×560,
  ranges set by the app to match the eigenvalue plane.
- `create_signal_curve_figure()` — 1D line view of the signal x(s) at the current
  animation time on the fixed unit window; source columns ``s`` and ``value``;
  returns `(figure, source, line_renderer)` (the app sets the line color per
  theme); 500×380 (y-range fixed by the app).
- `create_time_curves_figure()` — time view with the field along space at the
  current time: original (black), reconstruction (gray dashed), sum of selected
  modes (gray dash-dot), selected eigenvalue contribution (thicker, pair color set by
  the app via the returned renderer), and thin per-eigenvalue contribution
  lines in pair colors (`MultiLine` + hover) for all checked eigenvalues;
  returns `(figure, source, pair_source, selected_renderer, curve_renderers)`
  with ``curve_renderers`` holding the theme-dependent reference lines;
  1000×420.

Expected source columns are documented in each docstring; the app fills them in
`_update_*` methods.

## `src/dmdsim/app/styling.py`

Central visual styling of the application with a **theme system** (light and
dark; the dark theme follows the lecture website's dark variant, page
background #131313):

- `THEMES` — per-mode color dictionaries (page/text/muted/grid/box/tab/line
  colors plus per-step panel tints and accents).
- `PAIR_PALETTE` / `PAIR_PALETTE_DARK` — categorical colors for conjugate
  eigenvalue pairs (one color per physical mode); the dark cycle is brighter.
- `theme_colors(mode)`, `step_colors(step, mode)` — accessors.
- `style_figure(fig, step, mode)` — theme plot background on the tinted panel
  border, accent-colored bold title, theme axes/grid/legend (legend styling
  only applied when the figure actually has legends).
- `header(text, step, mode)` / `header_styles(step, mode)` — colored section
  header element and its style dictionary (for re-theming existing widgets).
- `hint(text, width=None, mode)` / `hint_styles(mode)` — muted inline
  explanation block.
- `info_box(text, font_size, mode)` / `info_styles(mode, font_size)` — neutral
  box for dynamic status/readout text.
- `tab_css(mode, accent, vertical)` — CSS string for a tab strip (rounded
  buttons, hover state, accent-colored active tab; uses the
  `.bk-tab`/`.bk-active` classes); the app wraps it in an `InlineStyleSheet`
  and updates the `css` on theme switches.
- `page_css(mode)` — CSS string for the page-wide widget theming (inputs,
  buttons, sliders, tooltips, menus); the app wraps it in a
  `GlobalInlineStyleSheet`. Properties competing with bokeh's own widget
  stylesheets use ``!important`` (bokeh's win otherwise — the white-input-box
  bug).

.. note:: Spinner tooltip mechanics
    Widget `description` must be a `bokeh.models.Tooltip` instance to render the
    interactive popover; a plain string only sets the native browser `title`
    attribute (long hover delay, no click). The `?` tooltips are built via the
    `_spinner` helper in `app/main.py`, which composes the explanation and the
    allowed range from the actual widget bounds.

.. note:: Spinner typed-value and math limitations (bokeh 3.10)
    `SpinnerView.bound_value` silently discards typed values outside the declared
    bounds (returns the old value instead of clamping), and the numeric input
    filter rejects decimal commas (typing "2,5" reverts the comma, yielding
    "25") — type decimal points. MathText (LaTeX) is only supported in canvas
    text (plot titles, axis labels); widget titles are plain text, so they use
    Unicode subscripts (fₘᵢₙ, Δt, σₙₒᵢₛₑ).

## `src/dmdsim/app/main.py`

The application wiring. Uses absolute imports (Bokeh loader constraint) and
silences flowTorch's INFO logging at import time.

- State: `_theme` (``"light"``/``"dark"``), `_cached_pairs`/`_fit_amplitudes`/
  `_fit_eigvals` (editing baselines cached per fit and on re-sort),
  `_locked_edit_indices` (unit-circle-locked drags are temporary and reset when
  another eigenvalue is selected), `_drag_cache` (marker positions for drag
  detection), `_themed_divs` (registered theme-dependent elements), and the
  stored stylesheets `_page_sheet`/`_tab_sheet`/`_mode_tab_sheet` (created once;
  a theme switch regenerates their `css`).
- `_build_widgets`: automatic-composition spinners (Nₘ, f_min, f_max, offset),
  time/noise widgets, rank (0 = automatic), auto-rerun select, styled Run
  button, status info box, shared mode sliders, FOUR synchronized time sliders
  in **seconds** ("t [s]", range [0, (n_time − 1)·dt], step = Δt) and play
  toggles (panels 1, 2, 4, 5), mode checkbox group, readout and
  propagation-direction info boxes, |b| scale spinner, unit-circle lock toggle.
  Persistent info boxes are registered for theming; the theme stylesheets are
  created here.
- `_build_layout`: application bar (title + dark-mode `Switch`) above
  `Tabs(tabs_location="left")`: "Start" (default landing tab: DMD in a nutshell,
  workflow rows with per-step-accent Open buttons navigating via `_make_goto`,
  tips), "1 · Signal" (composition + grid/noise + DMD fit on the left, signal
  animation on the right), "2 · Eigenvalues" (one control row: eigenvalue
  number, sort, |b| scale, unit-circle lock; eigenvalue plane + λᵗ dynamics),
  "3 · Spectrum", "4 · Mode shapes" (nested sub-tabs: Real part / Imaginary
  part / Phase / Propagation direction with its own time slider + play), 
  "5 · Reconstruction". All panels stay alive while hidden; the layout root,
  main tabs, and nested tabs are stored for in-place theming.
- `_attach_callbacks`: parameter widgets → `_on_parameter_change` (refit on
  auto-rerun, otherwise mark stale); Run button → `_run_dmd`; eigenvalue-number
  sliders → `_on_mode_slider_change` (two-way sync); checkbox →
  `_on_view_change`; time sliders (four) → `_on_time_change` (two-way sync);
  play toggles → `_set_playback` (synced labels/states, one shared periodic
  callback); eigenvalue source data → `_on_eigenvalue_drag`; |b| scale →
  `_on_amplitude_change`; sort select → `_on_sort_change`.
  - `_run_dmd`: `_build_signal` (uses `generate_modes`) → constant-signal guard
    (friendly message) → rank resolution (0 = automatic: exact rank 2·n_modes
    (+1 with offset) of the clean composition; flowTorch optimal truncation when
    noise is on) → `fit_dmd` wrapped in try/except (`AssertionError`,
    `IndexError`, `RuntimeError`, `ValueError`) so `torch.linalg` failures land
    as a red message in the status box while the previous views stay → sort by
    the selected criterion → cache pairs/fit amplitudes/fit eigenvalues →
    recompute reconstruction → update all views and the info box (rank
    automatic + round-trip error when automatic, otherwise only the error).
  - `_mode_field(index, t_index)`: real contribution of one mode at one time level
    (`Re{φᵢ·bᵢ·λᵢᵗ}`) used by the time view.
  - `_pair_colors()`: one color per conjugate pair (theme-dependent palette;
    unpaired modes gray) shared by eigenvalue plane, dynamics markers,
    spectrum, and reconstruction.
  - `_update_*` methods: `_update_field_view` (fixed y-ranges for both animated
    line plots), `_update_quantities` (editing-state cache + marker/spectrum
    refresh), `_update_eigenvalue_markers` (stability pair colors + selection
    emphasis in both complex-plane plots; guarded against the drag callback),
    `_update_spectrum` (pair colors, selection emphasis, Nyquist spans and
    axis), `_update_mode_slider`/`_update_time_slider` (bounds + clamping of
    the synced sliders), `_update_mode_view` (fills the three single-quantity
    mode sources incl. unwrapped phase), `_update_direction_view` (traveling
    wave at the current time, fixed envelope y-range, direction arrow, and the
    κ/λ_w/c readout), `_update_mode_select` (resets selection before
    relabeling to avoid stale indices), `_update_time_view` (curves + per-
    eigenvalue pair lines + selected-line color + λᵗ trail), `_update_readout`.
  - `_on_eigenvalue_drag`: applies dragged eigenvalues with conjugate
    mirroring (cached pairs), restores the source after backspace deletions,
    recomputes the reconstruction, and refreshes spectrum + time views; the
    status box reports the max deviation from the original.
  - `_on_amplitude_change`: scales the selected eigenvalue's amplitude (and its
    conjugate partner) relative to the fitted baseline; clamps typed values.
  - `_on_theme_switch` / `_apply_theme`: in-place theming — restyles the
    registered figures and Divs (headers, hints, info boxes, workflow entries,
    buttons), updates the theme-dependent renderer colors, and regenerates the
    stored stylesheets' `css`. No rebuild: the active panel and all widget
    states are preserved (`doc.clear()` is forbidden — it wipes session
    callbacks).
  - `_on_time_change` / `_set_playback` / `_advance_time`: four-way slider sync
    with a re-entrancy guard and one shared animation via
    `Document.add_periodic_callback` (the returned token is kept for removal).
  - `_make_goto(index)`: navigation callback factory for the Start panel.
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
  values, seeded noise determinism, invalid parameters, automatic composition
  (`generate_modes`: determinism, bounds, dispersion, invalid args).
- `test_engine.py` — recovery of known frequencies/growth rates/eigenvalues,
  conjugate pair detection, sorting criteria (amplitude/integral/frequency,
  reconstruction invariance), invalid inputs.
- `test_synth.py` — fit round trip, analytic ground-truth synthesis, effect of
  eigenvalue modifications, conjugate mirroring semantics.

## Packaging

- `pyproject.toml` — setuptools, src layout, `requires-python >= 3.10`, runtime
  deps numpy + bokeh (flowTorch installed editable from `../flowtorch`, see
  README), console script `dmdsim-app = "dmdsim.app.main:main"`.
- `requirements.txt` — pinned environment incl. CPU-only torch and the minimal
  flowTorch import requirements (scipy, plotly, pandas, pillow, vtk, h5py,
  netCDF4).
