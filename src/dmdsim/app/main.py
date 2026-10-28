"""
    Implements the DMD playground Bokeh application.
"""
import logging

import numpy as np

from functools import partial
from bokeh.document import Document
from bokeh.layouts import column, row
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.server.callbacks import PeriodicCallback
from bokeh.application.handlers import FunctionHandler
from bokeh.models import Button, CheckboxGroup, Select, Slider, Spinner, TabPanel, Tabs, Toggle, Tooltip
from tornado.ioloop import PeriodicCallback as TornadoPeriodicCallback

from dmdsim.core.engine import fit_dmd
from dmdsim.core.quantities import DMDResult
from dmdsim.core.synth import mirror_conjugates, synthesize
from dmdsim.signals.generator import ModalSignal, generate_modes

from dmdsim.app.figures import (create_direction_figure, create_eigenvalue_figure,
                                create_mode_component_figure, create_mode_dynamics_figure,
                                create_signal_curve_figure, create_spectrum_figure,
                                create_time_curves_figure)
from dmdsim.app.styling import (PAIR_PALETTE, header, hint, info_box, style_figure,
                                step_colors, tab_stylesheet)

# keep the flowTorch info messages out of the application output
logging.getLogger("flowtorch").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

_NOISE_SEED = 42
_N_SPACE = 201
_DOMAIN_LENGTH = 1.0
_PLAY_INTERVAL_MS = 60
_WATCHDOG_INTERVAL_MS = 500
_SHUTDOWN_GRACE_MS = 2000

# per-mode editor defaults: wavenumber, frequency, amplitude, growth rate, active
_MODE_DEFAULTS: list[tuple[float, float, float, float, bool]] = [
    (2.0, 5.0, 1.5, -0.2, True),
    (3.0, 12.0, 0.8, 0.1, True),
    (1.0, 20.0, 1.0, 0.0, False),
]

_HINT_EIGENVALUES = ("Radius |λ| encodes growth/decay: outside the unit circle the mode "
                     "grows, inside it decays. The angle encodes the frequency "
                     "f = arg(λ) / (2π·dt). Drag an eigenvalue to edit it — its "
                     "conjugate partner mirrors automatically and the reconstruction, "
                     "spectrum, and spiral update live. With 'lock to unit circle' "
                     "active, dragged eigenvalues stay neutral (|λ| = 1) and only "
                     "their frequency changes. A refit (Run DMD or a signal change) "
                     "restores the fitted values.")
_HINT_DYNAMICS = ("Powers λᵗ of the selected mode in the complex plane, animated with "
                  "the global time cursor: the trail spirals inward (decay), outward "
                  "(growth) or orbits on a circle (neutral); the rotation speed is the "
                  "frequency. Dragging an eigenvalue later reshapes this spiral live.")
_HINT_SPECTRUM = ("Mode strength |b| over frequency; conjugate pairs appear as mirror "
                  "peaks and share their color. Position encodes stability (radius = "
                  "growth/decay), the large marker is the selected eigenvalue, and "
                  "the dashed lines mark the Nyquist limits ±1/(2·dt).")
_HINT_MODE = ("Complex spatial shape φ(s): Re and Im form the wave profile, |φ| its "
              "magnitude, and arg(φ) the local phase shift. A linear arg(φ) corresponds "
              "to a wavenumber.")
_HINT_TIME = ("The curves show the field at the current time: the original (black), "
              "the reconstruction (gray dashed, Σ φᵢ·bᵢ·λᵢᵗ), the per-eigenvalue "
              "contributions of all checked eigenvalues in their pair colors "
              "(hover to identify), and the selected eigenvalue's contribution "
              "(thick, in its pair color). The time cursor is shared with all "
              "panels.")
_HINT_SIGNAL_ANIMATION = ("Animation of the created signal in the fixed window "
                          "x ∈ [0, 1]: press play or drag the time slider to move in "
                          "time. The same time cursor drives the reconstruction view.")
_HINT_REAL = ("Real part of the mode shape φ(s); for a traveling wave it is a cosine "
              "whose zero crossings are spaced by half a wavelength.")
_HINT_IMAG = ("Imaginary part of the mode shape φ(s); real and imaginary part "
              "together encode amplitude and spatial phase of the wave.")
_HINT_PHASE = ("The phase arg(φ) is only defined modulo 2π, so it wraps at ±π "
               "(faded points). The unwrapped line is continuous: its slope is the "
               "wavenumber κ and its offset is the spatial phase shift of the wave.")
_HINT_DIRECTION = ("The wave contribution Re{φ·b·λᵗ} of the selected mode, animated "
                   "with the time cursor: watch it travel. The arrow shows the "
                   "propagation direction; the info box derives the wavenumber κ, "
                   "the wavelength λ_w, and the phase velocity c from the phase "
                   "slope.")


def _spinner(title: str, value: float | int, low: float | int, high: float | int,
             step: float | int, width: int, info: str,
             description: str | None = None) -> Spinner:
    """
    Create a spinner whose question-mark tooltip explains parameter and bounds.

    :param title: widget title
    :type title: str
    :param value: initial value
    :type value: float | int
    :param low: lowest allowed value
    :type low: float | int
    :param high: highest allowed value
    :type high: float | int
    :param step: step size of the arrows
    :type step: float | int
    :param width: widget width in pixels
    :type width: int
    :param info: short explanation of what the parameter sets
    :type info: str
    :param description: full tooltip text overriding the default composition of
        ``info`` and the allowed range, defaults to None
    :type description: str | None, optional
    :return: configured spinner
    :rtype: Spinner
    """
    text = description if description is not None else f"{info} (allowed: {low} to {high})"
    # a Tooltip model renders the interactive question-mark tooltip; a plain
    # string would only set the native browser title attribute
    return Spinner(title=title, value=value, low=low, high=high, step=step, width=width,
                   description=Tooltip(content=text, target="auto", position="right"))


class DMDPlayground:
    def __init__(self, doc: Document) -> None:
        """
        Implement a class for the DMD playground Bokeh application.

        The application builds a synthetic modal line field from the widget state,
        fits the DMD via flowTorch, and displays eigenvalues, spectrum, mode shapes,
        and the time-dependent field. Whenever the auto-rerun toggle is active, any
        parameter change triggers a full recomputation.

        :param doc: Bokeh document the application is added to
        :type doc: Document
        :return: None
        :rtype: None
        """
        self._doc = doc
        self._signal: ModalSignal | None = None
        self._result: DMDResult | None = None
        self._reconstruction: np.ndarray | None = None
        self._periodic_callback: PeriodicCallback | None = None
        self._syncing_time = False
        self._syncing_play = False
        self._syncing_mode = False
        self._syncing_drag = False
        self._syncing_amplitude = False
        self._cached_pairs: list[tuple[int, int]] = []
        self._fit_amplitudes: np.ndarray | None = None
        self._fit_eigvals: np.ndarray | None = None
        self._locked_edit_indices: set[int] = set()
        self._drag_cache: tuple[np.ndarray, np.ndarray] = (np.array([]), np.array([]))
        self._build_widgets()
        self._build_figures()
        self._build_layout()
        self._attach_callbacks()
        self._run_dmd()

    def _build_widgets(self) -> None:
        """
        Create all input widgets and status elements of the application.

        :return: None
        :rtype: None
        """
        # automatic modal composition: the user picks the count and frequency range
        self._n_modes_spinner = _spinner("Nₘ", 3, 1, 32, 1, 80,
                                         "number of traveling waves (modes) superposed; "
                                         "their frequencies are evenly spaced between "
                                         "f_min and f_max")
        self._f_min_spinner = _spinner("fₘᵢₙ [Hz]", 1.0, 0.0, 500.0, 0.5, 90,
                                       "lowest frequency of the composition (must stay "
                                       "below f_max)")
        self._f_max_spinner = _spinner("fₘₐₓ [Hz]", 10.0, 0.0, 500.0, 0.5, 90,
                                       "highest frequency of the composition (must stay "
                                       "above f_min); frequencies above the Nyquist "
                                       "limit 1/(2·dt) alias in time")
        self._f_min_spinner.high = self._f_max_spinner.value
        self._f_max_spinner.low = self._f_min_spinner.value
        # time grid, offset, and noise (the spatial grid is fixed internally)
        self._dt_spinner = _spinner("Δt [s]", 0.01, 0.0001, 1.0, 0.001, 80,
                                    "time step between two animation frames")
        self._n_time_spinner = _spinner("nₜᵢₘₑ", 300, 8, 5000, 1, 80,
                                        "number of time levels; the animation covers "
                                        "n_time · dt seconds")
        self._offset_spinner = _spinner("offset", 0.0, -5.0, 5.0, 0.1, 80,
                                        "constant value added everywhere in space and "
                                        "time; shows up in the DMD as a mode with "
                                        "lambda = 1 (zero frequency, zero growth)")
        self._noise_spinner = _spinner("σₙₒᵢₛₑ [%]", 0, 0, 100, 1, 90,
                                       "Gaussian noise level in percent of the maximum "
                                       "clean signal amplitude; 0 % disables noise")
        # DMD controls
        self._rank_spinner = _spinner("rank (0 = automatic)", 0, 0, 256, 1, 160,
                                      "SVD truncation rank",
                                      description="SVD truncation rank; 0 = automatic "
                                                  "(exact rank of the composition, or "
                                                  "optimal truncation when noise is on); "
                                                  "allowed: 0 to 256")
        self._auto_toggle = Select(title="auto-rerun", value="on", options=["on", "off"],
                                   width=160,
                                   description="refit the DMD immediately whenever a "
                                               "signal or fit parameter changes; 'off' "
                                               "lets you change several parameters "
                                               "first and update via the Run DMD "
                                               "button (stale views are marked in the "
                                               "info box below)")
        self._run_button = Button(label="Run DMD", button_type="primary", width=160)
        self._run_button.styles = {
            "background-color": step_colors("setup")[1],
            "color": "#ffffff",
            "border": "none",
            "border-radius": "6px",
            "font-weight": "500",
        }
        self._status_div = info_box()
        # interactive elements of the quantity views
        self._mode_slider = Slider(title="eigenvalue number", start=0, end=0, value=0,
                                   step=1, width=300)
        self._dynamics_mode_slider = Slider(title="eigenvalue number", start=0, end=0,
                                            value=0, step=1, width=300)
        self._time_slider = Slider(title="t [s]", start=0, end=2.99, value=0, step=0.01,
                                   width=300)
        self._play_toggle = Toggle(label="▶ play", active=False, width=100)
        self._signal_time_slider = Slider(title="t [s]", start=0, end=2.99, value=0,
                                          step=0.01, width=300)
        self._signal_play_toggle = Toggle(label="▶ play", active=False, width=100)
        self._mode_select = CheckboxGroup(labels=[], active=[], inline=True)
        self._sort_select = Select(title="sort modes by", value="integral",
                                   options=[("integral", "integral contribution"),
                                            ("amplitude", "amplitude |b|"),
                                            ("frequency", "frequency |f|")],
                                   width=220,
                                   description="ordering of the modes in all views; "
                                               "the integral contribution follows Kou "
                                               "et al. (2017) as computed by flowTorch; "
                                               "sorting only rearranges the modes, the "
                                               "reconstruction is unchanged")
        self._eigen_time_slider = Slider(title="t [s]", start=0, end=2.99, value=0,
                                         step=0.01, width=300)
        self._eigen_play_toggle = Toggle(label="▶ play", active=False, width=100)
        self._amplitude_spinner = Spinner(title="|b| scale (selected)", value=1.0,
                                          low=0.0, high=3.0, step=0.01, width=160,
                                          description="scales the contribution of the "
                                                      "selected eigenvalue (and its "
                                                      "conjugate partner) to the "
                                                      "reconstruction: 1 = as fitted, "
                                                      "0 = silenced, 2 = twice as "
                                                      "strong; the original signal is "
                                                      "not affected")
        self._lock_toggle = Toggle(label="lock to unit circle", active=False, width=160)
        # a little extra air below the play/slider row improves the grouping
        self._mode_select.styles = {"margin-top": "10px"}
        self._readout_div = info_box()
        # slightly larger text improves the glanceability of the derived numbers
        self._direction_div = info_box(font_size="13px")
        self._modes_time_slider = Slider(title="t [s]", start=0, end=2.99, value=0,
                                         step=0.01, width=300)
        self._modes_play_toggle = Toggle(label="▶ play", active=False, width=100)

    def _build_figures(self) -> None:
        """
        Create all figures and their data sources.

        :return: None
        :rtype: None
        """
        self._eigenvalue_figure, self._eigenvalue_source = create_eigenvalue_figure()
        (self._dynamics_figure, self._dynamics_trail_source, self._dynamics_point_source,
         self._dynamics_all_source, self._dynamics_trail_renderer) = \
            create_mode_dynamics_figure()
        self._direction_figure, self._direction_source, self._direction_arrow = \
            create_direction_figure()
        self._spectrum_figure, self._spectrum_source, self._nyquist_spans = \
            create_spectrum_figure()
        self._signal_figure, self._signal_line_source = create_signal_curve_figure()
        self._time_figure, self._time_source, self._pair_source, \
            self._selected_renderer = create_time_curves_figure()
        # one single-quantity figure per mode submenu
        self._mode_views = {}
        for quantity in ("real", "imag", "phase"):
            mode_figure, mode_source = create_mode_component_figure(quantity)
            style_figure(mode_figure, "modes")
            self._mode_views[quantity] = (mode_figure, mode_source)
        # color-code the figures by step for visual distinction
        style_figure(self._eigenvalue_figure, "eigenvalues")
        style_figure(self._dynamics_figure, "eigenvalues")
        style_figure(self._spectrum_figure, "spectrum")
        style_figure(self._time_figure, "reconstruction")
        style_figure(self._signal_figure, "setup")

    def _build_layout(self) -> None:
        """
        Arrange the widgets and figures into a tabbed layout with one panel per step.

        The tab headers act as a left sidebar with numbered workflow steps; every
        panel keeps its state while another panel is shown. The Signal panel is
        the start menu (create the signal, animate it, run the fit); the Mode
        shapes panel contains nested submenus for the individual mode quantities.

        :return: None
        :rtype: None
        """
        # one compact editor row per mode
        composition = column(
            header("Modal composition", "setup"),
            row(self._n_modes_spinner, self._f_min_spinner, self._f_max_spinner,
                self._offset_spinner),
            spacing=10,
        )
        controls = column(
            composition,
            header("Time", "setup"),
            row(self._dt_spinner, self._n_time_spinner),
            header("Noise", "setup"),
            row(self._noise_spinner),
            header("DMD fit", "setup"),
            self._rank_spinner,
            self._auto_toggle,
            self._run_button,
            self._status_div,
            spacing=8,
        )
        animation = column(
            header("Signal animation", "setup"),
            self._signal_figure,
            row(self._signal_time_slider, self._signal_play_toggle),
            spacing=8,
        )
        signal_panel = row(controls, animation, spacing=14)
        eigen_panel = column(
            row(self._dynamics_mode_slider, self._sort_select),
            row(self._amplitude_spinner, self._lock_toggle),
            row(self._eigenvalue_figure, self._dynamics_figure),
            row(self._eigen_time_slider, self._eigen_play_toggle),
            row(hint(_HINT_EIGENVALUES, width=620), hint(_HINT_DYNAMICS, width=620)),
            spacing=10,
        )
        spectrum_panel = column(self._spectrum_figure, hint(_HINT_SPECTRUM, width=620),
                                spacing=10)
        mode_tabs = Tabs(tabs=[
            TabPanel(child=column(self._mode_views["real"][0],
                                  hint(_HINT_REAL, width=1000), spacing=8),
                     title="Real part"),
            TabPanel(child=column(self._mode_views["imag"][0],
                                  hint(_HINT_IMAG, width=1000), spacing=8),
                     title="Imaginary part"),
            TabPanel(child=column(self._mode_views["phase"][0],
                                  hint(_HINT_PHASE, width=1000), spacing=8),
                     title="Phase"),
            TabPanel(child=column(self._direction_figure,
                                  row(self._modes_time_slider, self._modes_play_toggle),
                                  self._direction_div, spacing=8),
                     title="Propagation direction"),
        ], tabs_location="above",
            stylesheets=[tab_stylesheet(accent=step_colors("modes")[1], vertical=False)])
        mode_panel = column(self._mode_slider, mode_tabs, spacing=10)
        time_panel = column(
            header("Time &amp; mode selection", "reconstruction"),
            row(self._time_slider, self._play_toggle),
            self._mode_select,
            self._time_figure,
            row(hint(_HINT_TIME, width=500), self._readout_div),
            spacing=10,
        )
        tabs = Tabs(tabs=[
            TabPanel(child=signal_panel, title="1 · Signal"),
            TabPanel(child=eigen_panel, title="2 · Eigenvalues"),
            TabPanel(child=spectrum_panel, title="3 · Spectrum"),
            TabPanel(child=mode_panel, title="4 · Mode shapes"),
            TabPanel(child=time_panel, title="5 · Reconstruction"),
        ], tabs_location="left", stylesheets=[tab_stylesheet()])
        self._doc.add_root(tabs)

    def _attach_callbacks(self) -> None:
        """
        Connect the widget callbacks for recomputation and view updates.

        :return: None
        :rtype: None
        """
        # signal and DMD parameters trigger a recomputation or mark the views stale
        parameter_widgets = [self._n_modes_spinner, self._f_min_spinner,
                             self._f_max_spinner, self._dt_spinner, self._n_time_spinner,
                             self._offset_spinner, self._noise_spinner,
                             self._rank_spinner]
        for widget in parameter_widgets:
            attribute = "active" if isinstance(widget, Toggle) else "value"
            widget.on_change(attribute, partial(self._on_parameter_change, widget=widget))
        self._eigenvalue_source.on_change("data", self._on_eigenvalue_drag)
        self._amplitude_spinner.on_change("value", self._on_amplitude_change)
        self._run_button.on_click(self._run_dmd)
        self._auto_toggle.on_change("value", self._on_auto_toggle)
        # keep the frequency bounds mutually consistent
        self._f_min_spinner.on_change("value", self._on_f_min_change)
        self._f_max_spinner.on_change("value", self._on_f_max_change)
        # view interactions update the curves without a refit
        self._mode_slider.on_change("value", self._on_mode_slider_change)
        self._dynamics_mode_slider.on_change("value", self._on_mode_slider_change)
        self._sort_select.on_change("value", self._on_sort_change)
        self._time_slider.on_change("value", self._on_time_change)
        self._signal_time_slider.on_change("value", self._on_time_change)
        self._eigen_time_slider.on_change("value", self._on_time_change)
        self._modes_time_slider.on_change("value", self._on_time_change)
        self._mode_select.on_change("active", self._on_view_change)
        self._play_toggle.on_change("active", self._on_recon_play)
        self._signal_play_toggle.on_change("active", self._on_signal_play)
        self._eigen_play_toggle.on_change("active", self._on_eigen_play)
        self._modes_play_toggle.on_change("active", self._on_modes_play)

    def _build_signal(self) -> ModalSignal:
        """
        Create the modal signal from the automatic composition and the widget state.

        The spatial domain is the fixed unit window :math:`x \\in [0, 1]` with an
        internal resolution; the phase speed of the composition is chosen so the
        highest frequency appears as three wavelengths inside the window. The
        noise level is given in percent of the maximum clean amplitude.

        :return: configured synthetic line field
        :rtype: ModalSignal
        """
        f_max = self._f_max_spinner.value
        modes = generate_modes(self._n_modes_spinner.value, self._f_min_spinner.value,
                               f_max, seed=_NOISE_SEED,
                               phase_speed=max(f_max, 1.0) / 3.0)
        dx = _DOMAIN_LENGTH / (_N_SPACE - 1)
        clean = ModalSignal(modes, dt=self._dt_spinner.value, n_space=_N_SPACE,
                            n_time=self._n_time_spinner.value, dx=dx,
                            offset=self._offset_spinner.value, noise_std=0.0,
                            seed=_NOISE_SEED)
        level = self._noise_spinner.value
        if level <= 0.0:
            return clean
        # noise is specified in percent of the maximum clean amplitude
        noise_std = level / 100.0 * float(np.abs(clean.samples).max())
        return ModalSignal(modes, dt=self._dt_spinner.value, n_space=_N_SPACE,
                           n_time=self._n_time_spinner.value, dx=dx,
                           offset=self._offset_spinner.value, noise_std=noise_std,
                           seed=_NOISE_SEED)

    def _run_dmd(self) -> None:
        """
        Rebuild the signal from the widget state, refit the DMD, and refresh all
        views. Fit failures (e.g., degenerate matrices from rank experiments) are
        reported in the status box while the previous views stay untouched.

        :return: None
        :rtype: None
        """
        try:
            signal = self._build_signal()
            if float(np.ptp(signal.samples)) == 0.0:
                raise ValueError("the signal is constant — the DMD needs temporal or "
                                 "spatial variation (add modes, offset, or noise)")
            rank = self._rank_spinner.value
            if rank == 0:
                # automatic: exact rank of the clean composition; optimal
                # truncation is only meaningful once noise blurs the spectrum
                rank = 2 * len(signal.modes) + (1 if signal.offset != 0.0 else 0)
                if self._noise_spinner.value > 0:
                    rank = None
            result = fit_dmd(signal.samples, dt=signal.dt, rank=rank)
            result.sort(by=self._sort_select.value)
            reconstruction = synthesize(result.modes, result.amplitudes, result.eigvals,
                                        signal.dt, signal.n_time)
        except (AssertionError, IndexError, RuntimeError, ValueError) as error:
            logger.exception("The DMD fit failed")
            self._status_div.text = (f"<span style='color:#c5221f'><b>the DMD fit "
                                     f"failed:</b> {error}</span>")
            return
        self._signal = signal
        self._result = result
        self._reconstruction = reconstruction
        # keep the slider ends consistent before any view reads them
        self._update_time_slider()
        self._update_mode_slider()
        self._update_field_view()
        self._update_quantities()
        self._update_mode_view()
        self._update_mode_select()
        error = float(np.abs(self._reconstruction - signal.samples).max())
        # only the automatic rank and the error are shown; an explicitly set rank
        # is known to the user already
        if self._rank_spinner.value == 0:
            self._status_div.text = (f"rank automatic ({result.n_modes}) · "
                                     f"round-trip error {error:.1e}")
        else:
            self._status_div.text = f"round-trip error {error:.1e}"

    def _mode_field(self, index: int, t_index: int) -> np.ndarray:
        """
        Compute the real contribution of one mode to the field at a single time level.

        :param index: mode index
        :type index: int
        :param t_index: time level index
        :type t_index: int
        :return: real-valued field contribution along space, shape ``(n_space,)``
        :rtype: np.ndarray
        """
        result = self._result
        dynamics = result.amplitudes[index] * result.eigvals[index] ** t_index
        return np.real(result.mode(index) * dynamics)

    def _update_field_view(self) -> None:
        """
        Update the fixed y-ranges of the animated line plots (signal and
        reconstruction views) so playback never rescales.

        :return: None
        :rtype: None
        """
        signal = self._signal
        vmax = max(float(np.abs(signal.samples).max()), 1e-12)
        # fixed y-range so the animated signal line does not rescale every step
        self._signal_figure.y_range.start = -1.1 * vmax
        self._signal_figure.y_range.end = 1.1 * vmax
        # the reconstruction view shows subsets of the signal, the same range fits
        self._time_figure.y_range.start = -1.1 * vmax
        self._time_figure.y_range.end = 1.1 * vmax

    def _update_quantities(self) -> None:
        """
        Update the eigenvalue plane, the spectrum, and their axis ranges, and
        refresh the cached editing state (pairs, fitted amplitudes, marker
        positions) for the eigenvalue editing.

        :return: None
        :rtype: None
        """
        result = self._result
        if result is None or result.n_modes == 0:
            self._eigenvalue_source.data = dict(re=[], im=[], index=[], frequency=[],
                                                growth=[], color=[], size=[])
            self._spectrum_source.data = dict(frequency=[], amplitude=[], index=[],
                                              growth=[], color=[], width=[], size=[])
            self._cached_pairs = []
            self._fit_amplitudes = None
            self._drag_cache = (np.array([]), np.array([]))
            return
        self._update_eigenvalue_markers()
        # cache the fitted baseline (eigenvalues, amplitudes), the pair structure,
        # and marker positions for the eigenvalue editing (dragging, amplitude
        # scaling, and the unit-circle lock reset)
        self._cached_pairs = result.pairs
        self._fit_amplitudes = result.amplitudes.copy()
        self._fit_eigvals = result.eigvals.copy()
        self._locked_edit_indices = set()
        self._drag_cache = (result.eigvals.real.copy(), result.eigvals.imag.copy())
        self._syncing_amplitude = True
        self._amplitude_spinner.value = 1.0
        self._syncing_amplitude = False
        # keep the eigenvalue plane centered on the unit circle with margin
        radius = max(1.2, float(np.abs(result.eigvals).max()) * 1.1)
        self._eigenvalue_figure.x_range.start = -radius
        self._eigenvalue_figure.x_range.end = radius
        self._eigenvalue_figure.y_range.start = -radius
        self._eigenvalue_figure.y_range.end = radius
        # the dynamics view shares the eigenvalue plane scale
        self._dynamics_figure.x_range.start = -radius
        self._dynamics_figure.x_range.end = radius
        self._dynamics_figure.y_range.start = -radius
        self._dynamics_figure.y_range.end = radius
        self._update_spectrum()

    def _pair_colors(self) -> list[str]:
        """
        Assign one color per conjugate eigenvalue pair.

        Each detected pair shares a categorical color so a physical mode can be
        followed across all views; unpaired modes (real eigenvalues, e.g., a
        constant offset) are gray.

        :return: one marker color per fitted mode
        :rtype: list[str]
        """
        result = self._result
        if result is None:
            return []
        assignment = {}
        for pair_index, (i, j) in enumerate(result.pairs):
            color = PAIR_PALETTE[pair_index % len(PAIR_PALETTE)]
            assignment[i] = color
            assignment[j] = color
        return [assignment.get(i, "#9aa0a6") for i in range(result.n_modes)]

    def _update_eigenvalue_markers(self) -> None:
        """
        Fill the eigenvalue markers of both complex-plane plots.

        Markers are colored by pair (each conjugate pair shares one categorical
        color) and the selected eigenvalue is drawn larger in both the
        eigenvalue plane and the mode dynamics view. The plane source carries
        the drag tool, so its refill is guarded against the drag callback.

        :return: None
        :rtype: None
        """
        result = self._result
        if result is None or result.n_modes == 0:
            self._eigenvalue_source.data = dict(re=[], im=[], index=[], frequency=[],
                                                growth=[], color=[], size=[])
            self._dynamics_all_source.data = dict(x=[], y=[], index=[], color=[],
                                                  size=[])
            return
        index = min(int(self._mode_slider.value), result.n_modes - 1)
        colors = self._pair_colors()
        sizes = [14 if i == index else 8 for i in range(result.n_modes)]
        # the plane source drives the drag tool: guard its refill against the
        # drag callback (fired by this very assignment)
        self._syncing_drag = True
        self._eigenvalue_source.data = dict(re=result.eigvals.real,
                                            im=result.eigvals.imag,
                                            index=list(range(result.n_modes)),
                                            frequency=result.frequencies,
                                            growth=result.growth_rates,
                                            color=colors, size=sizes)
        self._syncing_drag = False
        self._dynamics_all_source.data = dict(x=result.eigvals.real,
                                              y=result.eigvals.imag,
                                              index=list(range(result.n_modes)),
                                              color=colors, size=sizes)

    def _update_spectrum(self) -> None:
        """
        Update the DMD spectrum with stability colors and the selection emphasis.

        Stems and markers are red for growing, blue for decaying and gray for
        neutral modes; the selected eigenvalue is drawn larger. The dashed
        Nyquist spans and the frequency axis follow the current time step.

        :return: None
        :rtype: None
        """
        result = self._result
        if result is None or result.n_modes == 0:
            self._spectrum_source.data = dict(frequency=[], amplitude=[], index=[],
                                              growth=[], color=[], width=[], size=[])
            return
        index = min(int(self._mode_slider.value), result.n_modes - 1)
        colors = self._pair_colors()
        sizes = [14 if i == index else 7 for i in range(result.n_modes)]
        widths = [3 if i == index else 2 for i in range(result.n_modes)]
        self._spectrum_source.data = dict(frequency=result.frequencies,
                                          amplitude=np.abs(result.amplitudes),
                                          index=list(range(result.n_modes)),
                                          growth=result.growth_rates,
                                          color=colors, width=widths, size=sizes)
        nyquist = 0.5 / self._signal.dt
        self._spectrum_figure.x_range.start = -1.15 * nyquist
        self._spectrum_figure.x_range.end = 1.15 * nyquist
        self._nyquist_spans[0].location = -nyquist
        self._nyquist_spans[1].location = nyquist

    def _update_mode_slider(self) -> None:
        """
        Adjust the bounds of both synchronized mode sliders to the fitted modes.

        :return: None
        :rtype: None
        """
        result = self._result
        end = max(result.n_modes - 1, 0) if result is not None else 0
        for slider in (self._mode_slider, self._dynamics_mode_slider):
            slider.end = end
            if slider.value > end:
                slider.value = end

    def _update_time_slider(self) -> None:
        """
        Adjust the bounds of both synchronized time sliders to the signal length.

        The sliders operate in seconds and snap to the frame grid (step = dt).

        :return: None
        :rtype: None
        """
        signal = self._signal
        end = max((signal.n_time - 1) * signal.dt, 0.0) if signal is not None else 0.0
        step = signal.dt if signal is not None else 0.01
        for slider in (self._time_slider, self._signal_time_slider,
                       self._eigen_time_slider, self._modes_time_slider):
            slider.step = step
            slider.end = end
            if slider.value > end:
                slider.value = end

    def _update_mode_view(self) -> None:
        """
        Update the mode shape sub-views and the propagation direction readout.

        :return: None
        :rtype: None
        """
        result = self._result
        if result is None or result.n_modes == 0:
            for _, source in self._mode_views.values():
                source.data = dict(s=[], value=[], unwrapped=[])
            self._direction_div.text = "no modes fitted"
            return
        index = min(int(self._mode_slider.value), result.n_modes - 1)
        shape = result.mode(index)
        s = self._signal.space
        wrapped = np.angle(shape)
        values = {"real": np.real(shape), "imag": np.imag(shape),
                  "phase": wrapped, "unwrapped": np.unwrap(wrapped)}
        for quantity, (_, source) in self._mode_views.items():
            source.data = dict(s=s, value=values[quantity],
                               unwrapped=values["unwrapped"])

    def _update_direction_view(self) -> None:
        """
        Update the traveling-wave animation, the direction arrow, and the readout.

        The wave contribution of the selected mode is shown at the current time
        with a fixed y-range covering the whole animation, so growth is visible
        without rescaling. The propagation analysis follows from the phase slope
        :math:`\\kappa` (unwrapped linear fit) and the signed frequency.

        :return: None
        :rtype: None
        """
        result = self._result
        signal = self._signal
        if result is None or result.n_modes == 0 or signal is None:
            self._direction_source.data = dict(s=[], value=[])
            self._direction_div.text = "no modes fitted"
            self._direction_arrow.visible = False
            return
        index = min(int(self._mode_slider.value), result.n_modes - 1)
        t_index = min(round(self._time_slider.value / signal.dt), signal.n_time - 1)
        shape = result.mode(index)
        amplitude = result.amplitudes[index]
        wave = np.real(shape * (amplitude * result.eigvals[index] ** t_index))
        self._direction_source.data = dict(s=self._signal.space, value=wave)
        # fixed y-range over the whole animation so growth is visible without rescaling
        growth = result.growth_rates[index]
        envelope = (np.abs(amplitude) * float(np.abs(shape).max())
                    * max(1.0, float(np.exp(growth * signal.time[-1]))))
        self._direction_figure.y_range.start = -1.35 * envelope
        self._direction_figure.y_range.end = 1.35 * envelope
        # propagation analysis from the unwrapped phase slope
        frequency = float(result.frequencies[index])
        header = (f"<b>mode {index}</b> · f = {frequency:.3f} Hz · "
                  f"|b| = {np.abs(amplitude):.3f}<br>")
        slope = float(np.polyfit(self._signal.space, np.unwrap(np.angle(shape)), 1)[0])
        if abs(slope) < 1e-12:
            self._direction_arrow.visible = False
            self._direction_div.text = (header
                                        + "spatially constant shape — no propagation "
                                        "direction (wavenumber κ ≈ 0)")
            return
        wavelength = 2.0 * np.pi / abs(slope)
        if abs(frequency) < 1e-12:
            self._direction_arrow.visible = False
            self._direction_div.text = (header
                                        + f"κ = {slope:.3f} rad per unit length → "
                                        f"wavelength λ_w = {wavelength:.2f}"
                                        " — standing pattern (zero frequency)")
            return
        velocity = -2.0 * np.pi * frequency / slope
        direction = "+x" if velocity > 0.0 else "−x"
        self._direction_arrow.visible = True
        if velocity > 0.0:
            self._direction_arrow.x_start = 0.35
            self._direction_arrow.x_end = 0.65
        else:
            self._direction_arrow.x_start = 0.65
            self._direction_arrow.x_end = 0.35
        self._direction_arrow.y_start = 1.1 * envelope
        self._direction_arrow.y_end = 1.1 * envelope
        arrow_char = "→" if velocity > 0.0 else "←"
        self._direction_div.text = (
            header
            + f"κ = {slope:+.3f} rad per unit length → λ_w = {wavelength:.2f}<br>"
            f"phase velocity c = {abs(velocity):.2f} units per second toward "
            f"{direction} {arrow_char}")

    def _update_mode_select(self) -> None:
        """
        Rebuild the mode selection checkboxes; all modes are selected by default.

        :return: None
        :rtype: None
        """
        result = self._result
        if result is None or result.n_modes == 0:
            self._mode_select.active = []
            self._mode_select.labels = []
            return
        # reset the selection before relabeling to avoid stale indices in between
        self._mode_select.active = []
        self._mode_select.labels = [f"mode {i}: f = {f:.2f} Hz"
                                    for i, f in enumerate(result.frequencies)]
        self._mode_select.active = list(range(result.n_modes))

    def _update_time_view(self) -> None:
        """
        Update both time cursors and the field curves at the current time.

        :return: None
        :rtype: None
        """
        signal = self._signal
        result = self._result
        if signal is None:
            return
        t_index = min(round(self._time_slider.value / signal.dt), signal.n_time - 1)
        current_time = signal.time[t_index]
        self._signal_line_source.data = dict(s=self._signal.space,
                                             value=signal.samples[:, t_index])
        original = signal.samples[:, t_index]
        reconstruction = (self._reconstruction[:, t_index] if self._reconstruction is not None
                          else np.zeros_like(original))
        selected = np.zeros_like(original)
        selection = np.zeros_like(original)
        index = 0
        if result is not None and result.n_modes > 0:
            index = min(int(self._mode_slider.value), result.n_modes - 1)
            selected = self._mode_field(index, t_index)
            for i in self._mode_select.active:
                if i < result.n_modes:
                    selection += self._mode_field(i, t_index)
        self._time_source.data = dict(s=self._signal.space, original=original,
                                      reconstruction=reconstruction, selected=selected,
                                      selection=selection)
        # thin per-eigenvalue contribution lines in pair colors for all checked
        # eigenvalues; the selected eigenvalue's line is thicker in its pair color
        pair_xs, pair_ys, pair_colors, pair_index, pair_f = [], [], [], [], []
        if result is not None and result.n_modes > 0:
            colors = self._pair_colors()
            for i in self._mode_select.active:
                if i < result.n_modes:
                    pair_xs.append(self._signal.space)
                    pair_ys.append(self._mode_field(i, t_index))
                    pair_colors.append(colors[i])
                    pair_index.append(i)
                    pair_f.append(float(result.frequencies[i]))
        self._pair_source.data = dict(xs=pair_xs, ys=pair_ys, colors=pair_colors,
                                      index=pair_index, f=pair_f)
        self._selected_renderer.glyph.line_color = colors[index] \
            if result is not None and result.n_modes > 0 else "#9aa0a6"
        self._update_readout(index)
        # eigenvalue power trajectory of the selected mode, starting at lambda^1;
        # the trail carries the stability color of the selected mode
        if result is not None and result.n_modes > 0:
            color = self._pair_colors()[index]
            powers = result.eigvals[index] ** np.arange(1, max(t_index, 1) + 1)
            self._dynamics_trail_source.data = dict(x=powers.real, y=powers.imag)
            self._dynamics_point_source.data = dict(x=[powers.real[-1]],
                                                    y=[powers.imag[-1]],
                                                    color=[color])
            self._dynamics_trail_renderer.glyph.line_color = color
        else:
            self._dynamics_trail_source.data = dict(x=[], y=[], color=[])
            self._dynamics_point_source.data = dict(x=[], y=[], color=[])
        self._update_direction_view()

    def _update_readout(self, index: int) -> None:
        """
        Update the quantity readout of the selected mode.

        :param index: mode index to display
        :type index: int
        :return: None
        :rtype: None
        """
        result = self._result
        if result is None or result.n_modes == 0:
            self._readout_div.text = "no modes fitted"
            return
        index = min(index, result.n_modes - 1)
        eigval = result.eigvals[index]
        amplitude = result.amplitudes[index]
        self._readout_div.text = (
            f"<b>mode {index}</b><br>"
            f"λ = {eigval.real:.4f} {eigval.imag:+.4f}j · |λ| = {np.abs(eigval):.4f}<br>"
            f"f = {result.frequencies[index]:.3f} Hz · "
            f"σ = {result.growth_rates[index]:.3f} 1/s<br>"
            f"|b| = {np.abs(amplitude):.4f} · "
            f"arg(b) = {np.angle(amplitude):.3f} rad"
        )

    def _on_eigenvalue_drag(self, attr: str, old: object, new: object) -> None:
        """
        Apply dragged eigenvalues with conjugate mirroring and live feedback.

        The drag tool commits the edited positions once per drag gesture. The
        dragged eigenvalues are written into the result, their conjugate
        partners are mirrored, and the reconstruction, spectrum, and dynamics
        views are updated so the effects are immediately observable. A deletion
        via the backspace key is undone by restoring the source.

        :param attr: name of the changed source attribute
        :type attr: str
        :param old: previous source data
        :type old: object
        :param new: new source data
        :type new: object
        :return: None
        :rtype: None
        """
        if self._syncing_drag or self._result is None or self._signal is None:
            return
        re = np.asarray(new["re"], dtype=np.float64)
        im = np.asarray(new["im"], dtype=np.float64)
        if len(re) != self._result.n_modes:
            # a point was deleted via backspace: restore the fitted eigenvalues
            self._syncing_drag = True
            self._update_eigenvalue_markers()
            self._syncing_drag = False
            return
        old_re, old_im = self._drag_cache
        dragged = np.where((np.abs(re - old_re) > 1e-12)
                           | (np.abs(im - old_im) > 1e-12))[0]
        self._drag_cache = (re.copy(), im.copy())
        if len(dragged) == 0:
            return
        self._syncing_drag = True
        # write the dragged eigenvalues and mirror their conjugate partners;
        # a unit-circle lock projects them onto |lambda| = 1 and a drag into the
        # origin is clamped to a small radius (log(0) would be undefined)
        edited = {}
        for i in dragged:
            old_lam = complex(self._result.eigvals[i])
            lam = complex(re[i], im[i])
            radius = abs(lam)
            if self._lock_toggle.active:
                if radius > 1e-12:
                    lam = lam / radius
                elif abs(old_lam) > 1e-12:
                    lam = old_lam / abs(old_lam)
                else:
                    lam = complex(1.0, 0.0)
                # locked drags are temporary experiments: the eigenvalue is
                # restored to its fitted value when another one is selected
                self._locked_edit_indices.add(i)
            elif radius < 1e-3:
                angle = np.angle(lam) if radius > 1e-12 else np.angle(old_lam)
                lam = 1e-3 * np.exp(1j * angle)
            self._result.eigvals[i] = lam
            edited[i] = lam
        for i in dragged:
            for a, b in self._cached_pairs:
                if a == i and b not in edited:
                    edited[b] = np.conj(edited[i])
                elif b == i and a not in edited:
                    edited[a] = np.conj(edited[i])
        for i, lam in edited.items():
            self._result.eigvals[i] = lam
            re[i] = lam.real
            im[i] = lam.imag
        # refresh the plane source (partner mirroring) and the dynamics markers
        updated = dict(new)
        updated["re"] = re
        updated["im"] = im
        updated["frequency"] = self._result.frequencies
        updated["growth"] = self._result.growth_rates
        self._eigenvalue_source.data = updated
        markers = self._dynamics_all_source.data
        x = np.asarray(markers["x"], dtype=np.float64).copy()
        y = np.asarray(markers["y"], dtype=np.float64).copy()
        for i, lam in edited.items():
            x[i] = lam.real
            y[i] = lam.imag
        updated = dict(markers)
        updated["x"] = x
        updated["y"] = y
        self._dynamics_all_source.data = updated
        self._syncing_drag = False
        # feed the edited eigenvalues back into the reconstruction and all views
        self._reconstruction = synthesize(self._result.modes, self._result.amplitudes,
                                          self._result.eigvals, self._signal.dt,
                                          self._signal.n_time)
        error = float(np.abs(self._reconstruction - self._signal.samples).max())
        self._status_div.text = f"edited — max deviation from the original: {error:.1e}"
        self._update_spectrum()
        self._update_time_view()

    def _on_amplitude_change(self, attr: str, old: object, new: object) -> None:
        """
        Scale the amplitude of the selected eigenvalue and mirror its partner.

        The scale factor is applied relative to the fitted amplitude so the
        slider position corresponds to the current scaling; the reconstruction
        and all views are updated live.

        :param attr: name of the changed widget attribute
        :type attr: str
        :param old: previous attribute value
        :type old: object
        :param new: new attribute value (the scale factor)
        :type new: object
        :return: None
        :rtype: None
        """
        if self._syncing_amplitude or self._result is None or self._signal is None \
                or self._fit_amplitudes is None:
            return
        index = min(int(self._mode_slider.value), self._result.n_modes - 1)
        # typed values are not clamped by the widget: enforce the declared bounds
        scale = min(max(float(new), 0.0), 3.0)
        if scale != float(new):
            self._syncing_amplitude = True
            self._amplitude_spinner.value = scale
            self._syncing_amplitude = False
        self._result.amplitudes[index] = self._fit_amplitudes[index] * scale
        # mirror the conjugate partner (conjugated amplitude, same scale)
        for a, b in self._cached_pairs:
            if a == index:
                self._result.amplitudes[b] = np.conj(self._result.amplitudes[index])
            elif b == index:
                self._result.amplitudes[a] = np.conj(self._result.amplitudes[index])
        self._reconstruction = synthesize(self._result.modes, self._result.amplitudes,
                                          self._result.eigvals, self._signal.dt,
                                          self._signal.n_time)
        error = float(np.abs(self._reconstruction - self._signal.samples).max())
        self._status_div.text = f"edited — max deviation from the original: {error:.1e}"
        self._update_spectrum()
        self._update_time_view()

    def _on_sort_change(self, attr: str, old: object, new: object) -> None:
        """
        Re-sort the current DMD result and refresh all mode-dependent views.

        :param attr: name of the changed widget attribute
        :type attr: str
        :param old: previous sorting criterion
        :type old: object
        :param new: new sorting criterion
        :type new: object
        :return: None
        :rtype: None
        """
        if self._result is None or self._signal is None:
            return
        self._result.sort(by=str(new))
        self._update_quantities()
        self._update_mode_view()
        self._update_time_view()

    def _on_f_min_change(self, attr: str, old: object, new: object) -> None:
        """
        Keep f_min consistent with f_max: typed values above f_max are clamped.

        :param attr: name of the changed widget attribute
        :type attr: str
        :param old: previous attribute value
        :type old: object
        :param new: new attribute value
        :type new: object
        :return: None
        :rtype: None
        """
        if float(new) > float(self._f_max_spinner.value):
            # re-triggers this callback with the clamped value
            self._f_min_spinner.value = self._f_max_spinner.value
            return
        self._f_max_spinner.low = max(0.0, float(new))

    def _on_f_max_change(self, attr: str, old: object, new: object) -> None:
        """
        Keep f_max consistent with f_min: typed values below f_min are clamped.

        :param attr: name of the changed widget attribute
        :type attr: str
        :param old: previous attribute value
        :type old: object
        :param new: new attribute value
        :type new: object
        :return: None
        :rtype: None
        """
        if float(new) < float(self._f_min_spinner.value):
            # re-triggers this callback with the clamped value
            self._f_max_spinner.value = self._f_min_spinner.value
            return
        self._f_min_spinner.high = float(new)

    def _on_parameter_change(self, attr: str, old: object, new: object,
                             widget: Spinner | Toggle | None = None) -> None:
        """
        React to a signal or DMD parameter change by rerunning or marking stale.

        Typed spinner values are not clamped by the widget itself, so the allowed
        range is enforced here before anything is recomputed.

        :param attr: name of the changed widget attribute
        :type attr: str
        :param old: previous attribute value
        :type old: object
        :param new: new attribute value
        :type new: object
        :param widget: widget that fired the change, used for clamping typed
            values against its bounds
        :type widget: Spinner | Toggle | None
        :return: None
        :rtype: None
        """
        if widget is not None and not isinstance(widget, Toggle) \
                and isinstance(new, (int, float)):
            # enforce the declared bounds on typed values and refit once, clamped
            if new < widget.low:
                widget.value = widget.low
                return
            if new > widget.high:
                widget.value = widget.high
                return
        if self._auto_toggle.value == "on":
            self._run_dmd()
        else:
            self._status_div.text = ("parameters changed — press <b>Run DMD</b> to "
                                     "update the quantities")

    def _on_auto_toggle(self, attr: str, old: object, new: object) -> None:
        """
        React to the auto-rerun selection by refreshing immediately when turned on.

        :param attr: name of the changed widget attribute
        :type attr: str
        :param old: previous selection
        :type old: object
        :param new: new selection (``"on"`` or ``"off"``)
        :type new: object
        :return: None
        :rtype: None
        """
        if new == "on":
            self._run_dmd()

    def _on_mode_slider_change(self, attr: str, old: object, new: object) -> None:
        """
        Synchronize both mode sliders and refresh the mode-dependent views.

        :param attr: name of the changed widget attribute
        :type attr: str
        :param old: previous attribute value
        :type old: object
        :param new: new attribute value
        :type new: object
        :return: None
        :rtype: None
        """
        if self._syncing_mode:
            return
        self._syncing_mode = True
        for slider in (self._mode_slider, self._dynamics_mode_slider):
            if slider.value != new:
                slider.value = new
        self._syncing_mode = False
        # locked drags are temporary: restore the previously selected eigenvalue
        # to its fitted state when a different one is selected
        if self._fit_eigvals is not None and self._locked_edit_indices:
            for i in self._locked_edit_indices:
                if i < self._result.n_modes:
                    self._result.eigvals[i] = self._fit_eigvals[i]
            self._locked_edit_indices = set()
            self._reconstruction = synthesize(self._result.modes, self._result.amplitudes,
                                              self._result.eigvals, self._signal.dt,
                                              self._signal.n_time)
        self._update_mode_view()
        self._update_eigenvalue_markers()
        self._update_spectrum()
        # selecting a different eigenvalue restarts the shared time cursor
        self._time_slider.value = 0.0
        # the amplitude spinner follows the scale of the newly selected eigenvalue
        index = min(int(self._mode_slider.value), self._result.n_modes - 1) \
            if self._result is not None and self._result.n_modes > 0 else 0
        if self._fit_amplitudes is not None and index < len(self._fit_amplitudes) \
                and abs(self._fit_amplitudes[index]) > 0.0:
            scale = abs(self._result.amplitudes[index]) / abs(self._fit_amplitudes[index])
        else:
            scale = 1.0
        self._syncing_amplitude = True
        self._amplitude_spinner.value = scale
        self._syncing_amplitude = False

    def _on_view_change(self, attr: str, old: object, new: object) -> None:
        """
        React to a selection change by updating the views without a refit.

        :param attr: name of the changed widget attribute
        :type attr: str
        :param old: previous attribute value
        :type old: object
        :param new: new attribute value
        :type new: object
        :return: None
        :rtype: None
        """
        self._update_mode_view()
        self._update_time_view()

    def _on_recon_play(self, attr: str, old: bool, new: bool) -> None:
        """
        React to the reconstruction play toggle by updating the shared playback.

        :param attr: name of the changed widget attribute
        :type attr: str
        :param old: previous toggle state
        :type old: bool
        :param new: new toggle state
        :type new: bool
        :return: None
        :rtype: None
        """
        self._set_playback(new, self._play_toggle)

    def _on_signal_play(self, attr: str, old: bool, new: bool) -> None:
        """
        React to the signal animation play toggle by updating the shared playback.

        :param attr: name of the changed widget attribute
        :type attr: str
        :param old: previous toggle state
        :type old: bool
        :param new: new toggle state
        :type new: bool
        :return: None
        :rtype: None
        """
        self._set_playback(new, self._signal_play_toggle)

    def _on_eigen_play(self, attr: str, old: bool, new: bool) -> None:
        """
        React to the eigenvalue play toggle by updating the shared playback.

        :param attr: name of the changed widget attribute
        :type attr: str
        :param old: previous toggle state
        :type old: bool
        :param new: new toggle state
        :type new: bool
        :return: None
        :rtype: None
        """
        self._set_playback(new, self._eigen_play_toggle)

    def _on_modes_play(self, attr: str, old: bool, new: bool) -> None:
        """
        React to the mode shapes play toggle by updating the shared playback.

        :param attr: name of the changed widget attribute
        :type attr: str
        :param old: previous toggle state
        :type old: bool
        :param new: new toggle state
        :type new: bool
        :return: None
        :rtype: None
        """
        self._set_playback(new, self._modes_play_toggle)

    def _set_playback(self, active: bool, source: Toggle) -> None:
        """
        Synchronize both play toggles and start or stop the time animation.

        :param active: whether the animation should run
        :type active: bool
        :param source: toggle that triggered the change
        :type source: Toggle
        :return: None
        :rtype: None
        """
        if self._syncing_play:
            return
        self._syncing_play = True
        for toggle in (self._play_toggle, self._signal_play_toggle,
                       self._eigen_play_toggle, self._modes_play_toggle):
            toggle.label = "⏸ pause" if active else "▶ play"
            if toggle is not source and toggle.active != active:
                toggle.active = active
        self._syncing_play = False
        if active and self._periodic_callback is None:
            self._periodic_callback = self._doc.add_periodic_callback(
                self._advance_time, _PLAY_INTERVAL_MS)
        elif not active and self._periodic_callback is not None:
            self._doc.remove_periodic_callback(self._periodic_callback)
            self._periodic_callback = None

    def _on_time_change(self, attr: str, old: object, new: object) -> None:
        """
        Synchronize both time sliders and refresh the time-dependent views.

        :param attr: name of the changed widget attribute
        :type attr: str
        :param old: previous attribute value
        :type old: object
        :param new: new attribute value
        :type new: object
        :return: None
        :rtype: None
        """
        if self._syncing_time:
            return
        self._syncing_time = True
        for slider in (self._time_slider, self._signal_time_slider,
                       self._eigen_time_slider, self._modes_time_slider):
            if abs(slider.value - new) > 1e-12:
                slider.value = new
        self._syncing_time = False
        self._update_time_view()

    def _advance_time(self) -> None:
        """
        Advance the synchronized time index by one step for the animation.

        :return: None
        :rtype: None
        """
        if self._signal is None:
            return
        dt = self._signal.dt
        index = round(self._signal_time_slider.value / dt) % self._signal.n_time
        self._signal_time_slider.value = ((index + 1) % self._signal.n_time) * dt


def modify_doc(doc: Document) -> None:
    """
    Add the DMD playground application to a Bokeh document.

    :param doc: Bokeh document to add the application to
    :type doc: Document
    :return: None
    :rtype: None
    """
    DMDPlayground(doc)


def main() -> None:
    """
    Start the DMD playground as a standalone Bokeh server on localhost and open
    it in the default browser. The server stops itself about two seconds after
    the last browser session disconnected.

    :return: None
    :rtype: None
    """
    server = Server({"/": Application(FunctionHandler(modify_doc))}, num_procs=1)
    state = {"empty_ticks": 0}

    def watchdog() -> None:
        """
        Stop the io loop once every browser session has disconnected for the
        configured grace period.

        :return: None
        :rtype: None
        """
        sessions = server.get_sessions("/")
        if not sessions:
            return
        # require a short grace period to tolerate quick reconnects
        if all(session.connection_count == 0 for session in sessions):
            state["empty_ticks"] += 1
        else:
            state["empty_ticks"] = 0
        if state["empty_ticks"] * _WATCHDOG_INTERVAL_MS >= _SHUTDOWN_GRACE_MS:
            server.io_loop.stop()

    server.start()
    # Bokeh's own session cleanup does not run when the io loop is managed
    # manually (the periodic jobs bind to the wrong loop), so the shutdown is
    # handled by the watchdog instead; it must start from inside the running loop
    server.io_loop.add_callback(TornadoPeriodicCallback(watchdog, _WATCHDOG_INTERVAL_MS).start)
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()


if __name__ == "__main__":
    pass
