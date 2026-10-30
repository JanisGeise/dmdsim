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
from bokeh.models import (Button, Column, Div, GlobalInlineStyleSheet,
                          InlineStyleSheet, Label, Model, Range1d, Select,
                          Slider, Spinner, Switch, TabPanel, Tabs, Toggle,
                          Tooltip, UIElement)
from tornado.ioloop import PeriodicCallback as TornadoPeriodicCallback

from dmdsim.core.engine import fit_dmd
from dmdsim.core.quantities import DMDResult
from dmdsim.core.synth import synthesize, synthesize_window
from dmdsim.signals.generator import ModalSignal, generate_modes

from dmdsim.app.figures import (create_direction_figure, create_eigenvalue_figure,
                                create_growth_direction_figure,
                                create_mode_dynamics_figure,
                                create_phase_figure, create_prediction_error_figure,
                                create_prediction_probe_figure, create_quadrature_figure,
                                create_signal_curve_figure, create_space_time_figure,
                                create_spectrum_figure, create_time_curves_figure)
from dmdsim.app.styling import (PAIR_PALETTE, PAIR_PALETTE_DARK, THEMES,
                                accent_button_styles, button_styles, header,
                                header_styles, halfplane_colors, hint,
                                hint_styles, info_box, info_styles,
                                input_stylesheet, page_css,
                                primary_button_styles, slider_stylesheet,
                                step_colors, style_figure, switch_styles,
                                tab_css, theme_colors)

# keep the flowTorch info messages out of the application output
logging.getLogger("flowtorch").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

_NOISE_SEED = 42
_N_SPACE = 201
_DOMAIN_LENGTH = 1.0
_PLAY_INTERVAL_MS = 60
_WATCHDOG_INTERVAL_MS = 500
_SHUTDOWN_GRACE_MS = 2000
# vertical offset of the travel-direction triangles above the eigenvalue markers
_DIRECTION_MARKER_OFFSET = 0.13
# below this quadrature mix the wave counts as standing (no direction arrow)
_MORPH_STANDING_THRESHOLD = 0.05
# travel-direction arrow length: proportional to the net propagation of the
# pattern (traveling fraction t times the phase velocity |c|), clamped to
# stay visible and inside the plot domain x ∈ [0, 1]
_ARROW_SPEED_SCALE = 0.09
_ARROW_MIN_LENGTH = 0.04
_ARROW_MAX_LENGTH = 0.9
# default prediction horizon of the new panel in time levels beyond the data
_DEFAULT_PREDICTION_STEPS = 150
# the static prediction frame peeks this fraction of the data-window length
# into the forecast (capped to a quarter of the prediction window)
_PREDICTION_PEEK_DATA = 0.1
_PREDICTION_PEEK_PREDICTION = 0.25
# number of weight-slider boxes (one per conjugate pair or single mode) per row
_WEIGHT_BOXES_PER_ROW = 4

# signal and fit parameters restored by the panel-1 reset button; the keys name
# the spinner attributes (``_<key>_spinner``)
_DEFAULT_SIGNAL_PARAMS: dict[str, float | int] = {
    "n_modes": 3, "f_min": 1.0, "f_max": 10.0, "dt": 0.01, "n_time": 300,
    "offset": 0.0, "noise": 0, "rank": 0,
}

_HINT_EIGENVALUES = ("Radius |λ| encodes growth/decay: outside the unit circle the mode "
                     "grows, inside it decays. The angle encodes the frequency "
                     "f = arg(λ) / (2π·dt). The shaded half planes mark the sign of "
                     "Im λ (equivalently the frequency sign), and the small triangle "
                     "above each eigenvalue points in the travel direction of its "
                     "wave — it flips when the eigenvalue crosses the real axis. "
                     "Drag an eigenvalue to edit it — its conjugate partner mirrors "
                     "automatically and the reconstruction, spectrum, and spiral "
                     "update live. With 'lock to unit circle' active, dragged "
                     "eigenvalues stay neutral (|λ| = 1) and only their frequency "
                     "changes. A refit (Run DMD or a signal change) restores the "
                     "fitted values.")
_HINT_DYNAMICS = ("Powers λᵗ of the selected mode in the complex plane, animated with "
                  "the global time cursor: the trail spirals inward (decay), outward "
                  "(growth) or orbits on a circle (neutral); the rotation speed is the "
                  "frequency. Dragging an eigenvalue later reshapes this spiral live.")
_HINT_SPECTRUM = ("Mode amplitude |b| (how strong each mode is) over frequency; "
                  "conjugate pairs appear as mirror peaks and share their color. "
                  "Position encodes stability (radius = growth/decay), the large "
                  "marker is the selected eigenvalue, and the dashed lines mark the "
                  "Nyquist limits ±1/(2·dt). The right-hand plane shows each "
                  "mode's growth rate σ = Re(ln λ)/dt (horizontal) over its phase "
                  "velocity c = −2πf/κ (vertical): above the zero line the wave "
                  "travels toward +x, below it toward −x — conjugate partners "
                  "coincide there (one physical wave), and dragging an eigenvalue "
                  "across the real axis moves its point across the zero line, "
                  "flipping the direction.")
_HINT_MODE = ("Complex spatial shape φ(s): Re and Im form the wave profile, |φ| its "
              "magnitude, and arg(φ) the local phase shift. A linear arg(φ) corresponds "
              "to a wavenumber.")
_HINT_TIME = ("The curves show the field at the current time: the original (black), "
              "the weighted reconstruction (gray dashed, Σ φᵢ·wᵢ·gᵢ·bᵢ·λᵢᵗ), the "
              "per-eigenvalue contributions of all modes with a positive weight in "
              "their pair colors (hover to identify), and the selected eigenvalue's "
              "contribution (thick, in its pair color). The time cursor is shared "
              "with all panels.")
_HINT_WEIGHTS = ("One slider per eigenvalue, 0 = silenced to 1 = full fitted "
                 "contribution; conjugate partners move together (one physical wave, "
                 "one box, one color). The weights multiply the fitted amplitudes — "
                 "the reconstruction, the spectrum, and the prediction (panel 6) "
                 "respond live. The panel-2 '|b| gain' is an extra gain on top: "
                 "total scale = weight · gain.")
_HINT_SIGNAL_ANIMATION = ("Animation of the created signal in the fixed window "
                          "x ∈ [0, 1]: press play or drag the time slider to move in "
                          "time. The same time cursor drives the reconstruction view.")
_HINT_REAL = ("Real part of the mode shape φ(s); for a traveling wave it is a cosine "
              "whose zero crossings are spaced by half a wavelength.")
_HINT_IMAG = ("Imaginary part of the mode shape φ(s); real and imaginary part "
              "together encode amplitude and spatial phase of the wave. The "
              "'standing ←→ traveling' slider above scales this part: 1 is the "
              "fitted traveling wave, 0 a pure standing pattern — the shape is "
              "then an equal superposition of two opposite-traveling waves.")
_HINT_PHASE = ("The phase arg(φ) is only defined modulo 2π, so it wraps at ±π "
               "(faded points). The unwrapped line is continuous: its slope is the "
               "wavenumber κ and its offset is the spatial phase shift of the wave.")
_HINT_SPACETIME = ("The contribution Re{φ·b·λᵗ} of the selected mode over space "
                   "(x) and time (t): the horizontal stripe spacing is the "
                   "wavelength λ_w = 2π/κ — a property of the shape; the vertical "
                   "spacing is the temporal period T = 1/f — a property of the "
                   "eigenvalue; the stripe slope is the propagation speed "
                   "c = −λ_w/T. Editing the eigenvalue rescales the vertical "
                   "spacing and the slope while the horizontal spacing stays "
                   "frozen; the 'standing ←→ traveling' slider changes the "
                   "horizontal spacing. At t = 0 the pattern degenerates to "
                   "vertical stripes (standing).")
_HINT_DIRECTION = ("The wave contribution Re{φ·b·λᵗ} of the selected mode, animated "
                   "with the time cursor: watch it travel. The arrow shows the "
                   "propagation direction; the info box derives the wavenumber κ, "
                   "the wavelength λ_w, and the phase velocity c from the phase "
                   "slope. The eigenvalue's half plane sets the sign of f, which "
                   "together with the sign of κ fixes the direction via "
                   "c = −2πf/κ. The 'standing ←→ traveling' slider mixes in a "
                   "counter-traveling wave: the animation slows down and turns "
                   "into an in-place oscillation, and the arrow shrinks with "
                   "the remaining net propagation.")
_HINT_PREDICTION_PROBE = ("The model curve Σ φᵢ·bᵢ·λᵢᵗ continues seamlessly from the "
                          "data window (reconstruction) into the future "
                          "(prediction): the original data end at the dashed "
                          "boundary and the clean truth continues beyond it "
                          "(black dotted) — future noise is unpredictable by "
                          "definition. Mode weights (panel 5) and eigenvalue "
                          "edits reshape the prediction live: eigenvalues "
                          "outside the unit circle amplify it, inside they "
                          "damp it. The 'prediction steps' box sets how far "
                          "the model is advanced and the slider end follows.")
_HINT_PREDICTION_ERROR = ("RMS error of the model over space against the reference "
                          "(original inside the window, clean truth beyond the "
                          "dashed boundary): inside the window this is the "
                          "reconstruction error; beyond it the prediction error "
                          "reflects how the eigenvalues extrapolate — modes with "
                          "|λ| > 1 amplify the error exponentially, modes with "
                          "|λ| < 1 keep it bounded. With noise, the fit is "
                          "contaminated and the prediction drifts from the clean "
                          "physics.")


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
        self._theme = "light"
        self._signal: ModalSignal | None = None
        self._result: DMDResult | None = None
        self._reconstruction: np.ndarray | None = None
        self._prediction: np.ndarray | None = None
        self._truth: np.ndarray | None = None
        self._prediction_error: np.ndarray | None = None
        self._periodic_callback: PeriodicCallback | None = None
        self._predict_periodic_callback: PeriodicCallback | None = None
        self._syncing_time = False
        self._syncing_play = False
        self._syncing_mode = False
        self._syncing_drag = False
        self._syncing_amplitude = False
        self._syncing_params = False
        self._syncing_morph = False
        self._syncing_weights = False
        self._syncing_probe = False
        self._cached_pairs: list[tuple[int, int]] = []
        self._fit_amplitudes: np.ndarray | None = None
        self._fit_eigvals: np.ndarray | None = None
        self._fit_modes: np.ndarray | None = None
        self._mode_weights: np.ndarray | None = None
        self._mode_gains: np.ndarray | None = None
        self._weight_sliders: list[Slider] = []
        self._weight_slider_by_mode: dict[int, Slider] = {}
        self._pair_boxes: list[tuple[Column, int]] = []
        self._locked_edit_indices: set[int] = set()
        self._drag_cache: tuple[np.ndarray, np.ndarray] = (np.array([]), np.array([]))
        self._themed_divs: list[tuple[UIElement, str, str]] = []
        # theme stylesheets (created once; a theme switch updates their css)
        self._page_sheet = GlobalInlineStyleSheet(css=page_css(self._theme))
        self._tab_sheet = InlineStyleSheet(css=tab_css(self._theme, vertical=True))
        self._mode_tab_sheet = InlineStyleSheet(
            css=tab_css(self._theme, accent=step_colors("modes", self._theme)[1],
                        vertical=False))
        self._styled_figures: list[tuple[figure, str]] = []
        self._headers: list[tuple[Div, str]] = []
        self._hints: list[Div] = []
        self._build_widgets()
        self._build_figures()
        self._build_layout()
        # initial theming: the _themed-registered elements (Start-panel workflow
        # names/buttons etc.) carry no inline styles until this runs — without
        # it they stay unstyled until the first theme toggle
        self._apply_theme()
        self._attach_callbacks()
        # the root must be added after the callbacks are wired: bokeh resolves
        # incoming events (e.g. ButtonClick) via the models subscribed at
        # attach time, so on_click handlers registered after add_root would
        # never fire in the browser
        self._doc.add_root(self._layout_root)
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
        self._themed(self._run_button, "btn_primary")
        # reset buttons: panel 1 restores the default signal parameters, panels
        # 2-6 restore the fitted DMD values (the layout places one per panel)
        self._reset_signal_button = Button(label="↺ reset parameters", width=170)
        self._themed(self._reset_signal_button, "btn_default")
        self._reset_fit_buttons: dict[str, Button] = {}
        for key in ("eigen", "spectrum", "modes", "recon", "predict"):
            self._reset_fit_buttons[key] = Button(label="↺ reset to DMD fit",
                                                  width=170)
            self._themed(self._reset_fit_buttons[key], "btn_default")
        self._status_div = info_box()
        # interactive elements of the quantity views
        self._mode_slider = Slider(title="eigenvalue number", start=0, end=0, value=0,
                                   step=1, width=300)
        # quadrature mix of the selected mode shape: t = 1 keeps the fitted
        # traveling wave, t = 0 scales the imaginary part to zero (pure
        # standing pattern); Slider has no description, the hints explain it
        self._morph_slider = Slider(title="standing ←→ traveling", start=0.0,
                                    end=1.0, value=1.0, step=0.01, width=220)
        self._dynamics_mode_slider = Slider(title="eigenvalue number", start=0, end=0,
                                            value=0, step=1, width=300)
        self._time_slider = Slider(title="t [s]", start=0, end=2.99, value=0, step=0.01,
                                   width=300)
        self._play_toggle = Toggle(label="▶ play", active=False, width=100)
        self._signal_time_slider = Slider(title="t [s]", start=0, end=2.99, value=0,
                                          step=0.01, width=300)
        self._signal_play_toggle = Toggle(label="▶ play", active=False, width=100)
        # continuous mode weights of panel 5: one slider per eigenvalue (0 =
        # silenced, 1 = fitted contribution), grouped into one bordered box per
        # conjugate pair so the pairing stays visible; the sliders are rebuilt
        # on every fit and sort
        self._weights_layout = column(spacing=8, width=1150)
        self._weights_layout.styles = {"margin-top": "10px"}
        self._sort_select = Select(title="sort modes by", value="integral",
                                   options=[("integral", "integral contribution"),
                                            ("amplitude", "amplitude |b|"),
                                            ("frequency", "frequency |f|")],
                                   width=220,
                                   description="ordering of the modes in all views; "
                                               "'integral contribution' ranks modes by "
                                               "how much of the signal's fluctuation "
                                               "energy they explain over the whole "
                                               "signal (Kou et al. 2017); sorting only "
                                               "rearranges the modes, the "
                                               "reconstruction is unchanged")
        self._eigen_time_slider = Slider(title="t [s]", start=0, end=2.99, value=0,
                                         step=0.01, width=300)
        self._eigen_play_toggle = Toggle(label="▶ play", active=False, width=100)
        self._amplitude_spinner = Spinner(title="|b| gain (selected)", value=1.0,
                                          low=0.0, high=3.0, step=0.01, width=160,
                                          description="extra gain on the contribution "
                                                      "of the selected eigenvalue "
                                                      "(and its conjugate partner), "
                                                      "applied on top of the mode "
                                                      "weight set in panel 5: "
                                                      "total scale = weight · gain "
                                                      "(1 = no extra gain, 0 = "
                                                      "silenced, 2 = doubled); the "
                                                      "original signal is not "
                                                      "affected")
        self._lock_toggle = Toggle(label="lock to unit circle", active=False, width=160)
        self._readout_div = info_box()
        # slightly larger text improves the glanceability of the derived numbers
        self._direction_div = info_box(font_size="13px")
        self._themed(self._status_div, "info12")
        self._themed(self._readout_div, "info12")
        self._themed(self._direction_div, "info13")
        self._modes_time_slider = Slider(title="t [s]", start=0, end=2.99, value=0,
                                         step=0.01, width=300)
        self._modes_play_toggle = Toggle(label="▶ play", active=False, width=100)
        # prediction panel (6): the number of prediction steps (full user
        # control, 1 level upwards), an independent time cursor covering data
        # + prediction, and a probe point for the forecast time series
        self._n_pred_spinner = _spinner("prediction steps", _DEFAULT_PREDICTION_STEPS,
                                        1, 5000, 1, 160,
                                        "number of future time levels n_pred the "
                                        "fitted model is advanced beyond the data "
                                        "window; the prediction covers "
                                        "n_pred · Δt seconds")
        self._probe_slider = Slider(title="probe x", start=0.0, end=1.0, value=0.5,
                                    step=0.005, width=300)
        self._predict_time_slider = Slider(title="t [s]", start=0, end=4.49, value=0,
                                           step=0.01, width=300)
        self._predict_play_toggle = Toggle(label="▶ play", active=False, width=100)
        self._predict_div = info_box()
        self._predict_div.width = 700
        self._themed(self._predict_div, "info12")
        # play/lock toggles render a default button surface, which the plain
        # widget custom properties cannot reach (the .bk-btn paints over them)
        for toggle in (self._play_toggle, self._signal_play_toggle,
                       self._eigen_play_toggle, self._lock_toggle,
                       self._modes_play_toggle, self._predict_play_toggle):
            self._themed(toggle, "btn_default")

    def _build_figures(self) -> None:
        """
        Create all figures and their data sources.

        :return: None
        :rtype: None
        """
        # one shared x-range syncs the two grouped mode-shape figures
        # (quadratures, traveling wave); one shared y-ruler (sized by the
        # fitted quadratures in _update_mode_view) keeps the morph scaling
        # inside a fixed frame
        shape_x_range = Range1d(0.0, 1.0)
        self._shape_y_range = Range1d(-1.0, 1.0)
        self._eigenvalue_figure, self._eigenvalue_source, \
            self._upper_halfplane_renderer, self._lower_halfplane_renderer, \
            self._upper_halfplane_label, self._lower_halfplane_label = \
            create_eigenvalue_figure()
        (self._dynamics_figure, self._dynamics_trail_source, self._dynamics_point_source,
         self._dynamics_all_source, self._dynamics_trail_renderer) = \
            create_mode_dynamics_figure()
        self._quadrature_figure, self._quadrature_source = create_quadrature_figure(
            x_range=shape_x_range, y_range=self._shape_y_range, height=340,
            width=615)
        self._direction_figure, self._direction_source, self._direction_arrow = \
            create_direction_figure(x_range=shape_x_range, height=340, width=615)
        self._spectrum_figure, self._spectrum_source, self._nyquist_spans = \
            create_spectrum_figure()
        self._continuous_figure, self._continuous_source = \
            create_growth_direction_figure()
        self._signal_figure, self._signal_line_source, self._signal_line_renderer = \
            create_signal_curve_figure()
        (self._time_figure, self._time_source, self._pair_source,
         self._selected_renderer, self._curve_renderers) = create_time_curves_figure()
        # the phase view keeps its own sub-tab (1000 px wide)
        self._phase_figure, self._phase_source = create_phase_figure()
        self._space_time_figure, self._space_time_source, self._space_time_mapper = \
            create_space_time_figure(x_range=shape_x_range, height=420, width=615)
        # prediction panel: the original-vs-prediction forecast and the error
        (self._predict_probe_figure, self._predict_probe_source,
         self._probe_boundary_span, self._probe_cursor_span,
         self._probe_data_box, self._probe_labels,
         self._predict_probe_renderers) = create_prediction_probe_figure()
        (self._predict_error_figure, self._predict_error_source,
         self._error_boundary_span, self._error_cursor_span,
         self._error_data_box, self._error_labels,
         self._predict_error_renderer) = create_prediction_error_figure()
        # register figures with their step colors for theming
        self._styled_figures = [
            (self._eigenvalue_figure, "eigenvalues"),
            (self._dynamics_figure, "eigenvalues"),
            (self._spectrum_figure, "spectrum"),
            (self._continuous_figure, "spectrum"),
            (self._time_figure, "reconstruction"),
            (self._signal_figure, "setup"),
            (self._quadrature_figure, "modes"),
            (self._direction_figure, "modes"),
            (self._phase_figure, "modes"),
            (self._space_time_figure, "modes"),
            (self._predict_probe_figure, "prediction"),
            (self._predict_error_figure, "prediction"),
        ]

    def _build_layout(self) -> None:
        """
        Arrange the widgets and figures into the page layout.

        The layout consists of an application bar (title and theme toggle) and a
        tabbed area: a start panel as the default landing tab (DMD in a nutshell,
        workflow navigation, and tips) followed by one panel per step. The tab
        headers act as a left sidebar; every panel keeps its state while another
        panel is shown. All chrome is styled with the current theme.

        :return: None
        :rtype: None
        """
        mode = self._theme
        # theme stylesheets (stored so a theme switch can update their css)
        self._page_sheet = GlobalInlineStyleSheet(css=page_css(mode))
        self._tab_sheet = InlineStyleSheet(css=tab_css(mode, vertical=True))
        self._mode_tab_sheet = InlineStyleSheet(
            css=tab_css(mode, accent=step_colors("modes", mode)[1], vertical=False))
        # application bar: title and theme switch
        title_div = Div(text="<b>DMD Playground</b>",
                        styles={"color": theme_colors(mode)["text"],
                                "font-size": "15px", "padding": "8px 6px"},
                        width=900)
        self._theme_switch = Switch(label="dark mode", active=(mode == "dark"),
                                    width=100)
        self._theme_switch.on_change("active", self._on_theme_switch)
        app_bar = row(title_div, self._theme_switch, spacing=10)
        # start panel: nutshell, workflow navigation, and tips
        nutshell = self._themed_info(
            "<b>DMD in a nutshell</b><br>"
            "You measure a signal y(x, t) at many points in time — the snapshots are "
            "stacked into the <b>snapshot matrix</b> "
            "X = [y(x, t₀) … y(x, t<sub>N−1</sub>)]. DMD compares every snapshot with "
            "the next one (the time-shifted matrix X′) and finds the linear operator "
            "A ≈ X′X<sup>+</sup> that best advances the snapshots: X′ ≈ A X. The SVD "
            "of X solves this eigenvalue problem: the <b>eigenvalues λ</b> tell each "
            "mode how to evolve (angle = frequency, radius = growth/decay), the "
            "<b>DMD modes φ</b> are the shapes, and the <b>amplitudes b</b> set their "
            "strength. Every mode contributes φ·b·λ<sup>t</sup>, and the sum of all "
            "modes reconstructs the signal: "
            "y(x, t) ≈ Σ<sub>i</sub> φ<sub>i</sub>(x) · b<sub>i</sub> · λ<sub>i</sub>"
            "<sup>t</sup>. Because every mode carries its own λ<sup>t</sup>, the "
            "same formula also <b>predicts beyond the data window</b> — growing "
            "eigenvalues amplify it, decaying ones damp it (panel 6).",
            font_size="13px")
        step_for_panel = {1: "setup", 2: "eigenvalues", 3: "spectrum", 4: "modes",
                          5: "reconstruction", 6: "prediction"}
        descriptions = [
            ("1 · Signal", "design a traveling-wave signal and run the DMD fit", 1),
            ("2 · Eigenvalues", "inspect the fitted eigenvalues — drag them and watch "
                                "the reconstruction respond", 2),
            ("3 · Spectrum", "see each mode's amplitude, growth, and propagation "
                             "direction", 3),
            ("4 · Mode shapes", "look at Re/Im/phase of each mode and its traveling "
                                "wave", 4),
            ("5 · Reconstruction", "compare the reconstruction against the original "
                                   "and blend mode contributions with weight "
                                   "sliders", 5),
            ("6 · Prediction", "advance the fitted model beyond the data window and "
                               "watch the prediction error grow", 6),
        ]
        buttons = [Button(label="Open", width=90) for _ in descriptions]
        workflow_rows = []
        for button, (name, description, target) in zip(buttons, descriptions):
            button.on_click(self._make_goto(target))
            self._themed(button, "btn", step_for_panel[target])
            workflow_rows.append(row(
                name_div := Div(text=f"<b>{name}</b>", width=170),
                desc_div := Div(text=description, width=690),
                button,
                spacing=10,
            ))
            self._themed(name_div, "wf_name", step_for_panel[target])
            self._themed(desc_div, "wf_desc")
        tips = hint(
            "Tips: drag the eigenvalue dots in panel 2 (the toggle 'lock to unit "
            "circle' keeps them neutral) · blend mode contributions with the "
            "weight sliders in panel 5 · advance the model into the future in "
            "panel 6 · hover curves and markers for details · switch light/dark "
            "above.",
            width=1000)
        start_panel = column(
            nutshell,
            self._themed_info("Work through the steps in order — or jump straight "
                              "to any panel with the buttons on the right.",
                              font_size="12px"),
            *workflow_rows,
            tips,
            width=1000,
            spacing=12,
        )
        # step panels (unchanged structure, themed chrome)
        composition = column(
            self._themed_header("Modal composition", "setup"),
            row(self._n_modes_spinner, self._f_min_spinner, self._f_max_spinner,
                self._offset_spinner),
            spacing=10,
        )
        controls = column(
            composition,
            self._themed_header("Time", "setup"),
            row(self._dt_spinner, self._n_time_spinner),
            self._themed_header("Noise", "setup"),
            row(self._noise_spinner),
            self._themed_header("DMD fit", "setup"),
            self._rank_spinner,
            self._auto_toggle,
            row(self._run_button, self._reset_signal_button),
            self._status_div,
            spacing=8,
        )
        animation = column(
            self._themed_header("Signal animation", "setup"),
            self._signal_figure,
            row(self._signal_time_slider, self._signal_play_toggle),
            self._themed_hint("The same time cursor drives every panel.", width=500),
            spacing=8,
        )
        signal_panel = row(controls, animation, spacing=14)
        eigen_panel = column(
            row(self._dynamics_mode_slider, self._sort_select, self._amplitude_spinner,
                self._lock_toggle, self._reset_fit_buttons["eigen"]),
            row(self._eigenvalue_figure, self._dynamics_figure),
            row(self._eigen_time_slider, self._eigen_play_toggle),
            row(self._themed_hint(_HINT_EIGENVALUES, width=620),
                self._themed_hint(_HINT_DYNAMICS, width=620)),
            spacing=10,
        )
        spectrum_panel = column(
            self._reset_fit_buttons["spectrum"],
            row(self._spectrum_figure, self._continuous_figure),
            self._themed_hint(_HINT_SPECTRUM, width=620),
            spacing=10,
        )
        mode_tabs = Tabs(tabs=[
            TabPanel(child=column(
                row(self._quadrature_figure, self._direction_figure, spacing=5),
                row(column(self._modes_time_slider, self._modes_play_toggle,
                           width=300),
                    self._direction_div),
                self._themed_hint(_HINT_REAL, width=1230),
                self._themed_hint(_HINT_IMAG, width=1230),
                self._themed_hint(_HINT_DIRECTION, width=1230),
                spacing=8),
                title="Shape & propagation"),
            TabPanel(child=column(self._phase_figure,
                                  self._themed_hint(_HINT_PHASE, width=1000), spacing=8),
                     title="Phase"),
            TabPanel(child=column(self._space_time_figure,
                                  self._themed_hint(_HINT_SPACETIME, width=1000),
                                  spacing=8),
                     title="Space–time"),
        ], tabs_location="above",
            stylesheets=[self._mode_tab_sheet])
        self._mode_tabs = mode_tabs
        mode_panel = column(row(self._mode_slider, self._morph_slider,
                                self._reset_fit_buttons["modes"]),
                            mode_tabs, spacing=10)
        time_panel = column(
            self._themed_header("Time &amp; mode weights", "reconstruction"),
            row(self._time_slider, self._play_toggle, self._reset_fit_buttons["recon"]),
            self._weights_layout,
            self._time_figure,
            row(self._themed_hint(_HINT_TIME, width=500), self._readout_div),
            self._themed_hint(_HINT_WEIGHTS, width=1150),
            spacing=10,
        )
        prediction_panel = column(
            self._themed_header("Prediction beyond the data window", "prediction"),
            row(self._n_pred_spinner, self._probe_slider,
                self._reset_fit_buttons["predict"]),
            self._predict_probe_figure,
            row(self._predict_time_slider, self._predict_play_toggle,
                self._predict_div),
            self._predict_error_figure,
            row(self._themed_hint(_HINT_PREDICTION_PROBE, width=570),
                self._themed_hint(_HINT_PREDICTION_ERROR, width=570),
                spacing=10),
            spacing=10,
        )
        self._tabs = Tabs(tabs=[
            TabPanel(child=start_panel, title="Start"),
            TabPanel(child=signal_panel, title="1 · Signal"),
            TabPanel(child=eigen_panel, title="2 · Eigenvalues"),
            TabPanel(child=spectrum_panel, title="3 · Spectrum"),
            TabPanel(child=mode_panel, title="4 · Mode shapes"),
            TabPanel(child=time_panel, title="5 · Reconstruction"),
            TabPanel(child=prediction_panel, title="6 · Prediction"),
        ], tabs_location="left",
            stylesheets=[self._tab_sheet])
        self._mode_tabs.stylesheets = [self._mode_tab_sheet]
        self._layout_root = column(app_bar, self._tabs, spacing=8,
                                   stylesheets=[self._page_sheet])
        # the root is added to the document by __init__ AFTER the callbacks
        # are wired (bokeh's event subscription happens at attach time)

    def _themed(self, element: UIElement, kind: str, step: str = "") -> None:
        """
        Register a UI element for theme-dependent restyling.

        :param element: themed element (Div or Button/Toggle)
        :type element: UIElement
        :param kind: style kind; one of ``"header"``, ``"hint"``, ``"info12"``,
            ``"info13"``, ``"app_title"``, ``"wf_name"``, ``"wf_desc"``,
            ``"btn"`` (accent button, with ``step`` selecting the accent),
            ``"btn_default"``, or ``"btn_primary"``
        :type kind: str
        :param step: step key for kinds that use a step accent, defaults to ""
        :type step: str, optional
        :return: None
        :rtype: None
        """
        self._themed_divs.append((element, kind, step))

    def _themed_header(self, text: str, step: str) -> Div:
        """
        Create and register a themed section header.

        :param text: header text (HTML, escape special characters)
        :type text: str
        :param step: step key selecting the panel colors
        :type step: str
        :return: themed header element
        :rtype: Div
        """
        div = header(text, step, self._theme)
        self._themed(div, "header", step)
        return div

    def _themed_hint(self, text: str, width: int | None = None) -> Div:
        """
        Create and register a themed hint text block.

        :param text: explanation text (HTML, escape special characters)
        :type text: str
        :param width: fixed width of the block in pixels; unconstrained if ``None``
        :type width: int | None
        :return: themed hint element
        :rtype: Div
        """
        div = hint(text, width, self._theme)
        self._themed(div, "hint")
        return div

    def _themed_info(self, text: str, font_size: str = "12px") -> Div:
        """
        Create and register a themed info box.

        :param text: content of the box (HTML, escape special characters)
        :type text: str
        :param font_size: font size of the box text
        :type font_size: str
        :return: themed info box element
        :rtype: Div
        """
        div = info_box(text, font_size, self._theme)
        self._themed(div, "info13" if font_size == "13px" else "info12")
        return div

    def _apply_theme(self) -> None:
        """
        Apply the current theme in place to all figures, widgets, and text elements.

        The figure colors (canvas) are set from the theme, the registered Divs
        (headers, hints, info boxes, workflow entries) are restyled, the theme
        dependent renderer colors are updated, and the stored stylesheets' CSS
        strings are regenerated. No models are added or removed, so the session,
        the active panel, and all widget states are preserved.

        :return: None
        :rtype: None
        """
        theme = THEMES[self._theme]
        for fig, step in self._styled_figures:
            style_figure(fig, step, self._theme)
        self._signal_line_renderer.glyph.line_color = theme["line_signal"]
        self._curve_renderers["original"].glyph.line_color = theme["line_original"]
        self._curve_renderers["reconstruction"].glyph.line_color = \
            theme["line_reconstruction"]
        # the prediction panel's model curve follows the prediction accent, the
        # references follow the original line color, and the error is red
        renderers = self._predict_probe_renderers
        renderers["model"].glyph.line_color = theme["line_prediction"]
        renderers["original"].glyph.line_color = theme["line_original"]
        renderers["truth"].glyph.line_color = theme["line_original"]
        self._predict_error_renderer.glyph.line_color = theme["line_error"]
        # the shaded data-window background follows the theme's muted gray;
        # the stage labels are chips: reconstruction in the text color,
        # prediction in the prediction accent, on the plot background
        for box in (self._probe_data_box, self._error_data_box):
            box.fill_color = theme["muted"]
        for labels in (self._probe_labels, self._error_labels):
            reconstruction = labels["reconstruction"]
            reconstruction.text_color = theme["text"]
            reconstruction.border_line_color = theme["muted"]
            prediction = labels["prediction"]
            prediction.text_color = theme["line_prediction"]
            prediction.border_line_color = theme["accents"]["prediction"]
            for label in labels.values():
                label.background_fill_color = theme["plot_bg"]
        for span in (self._probe_cursor_span, self._error_cursor_span):
            span.line_color = theme["accents"]["prediction"]
        for span in (self._probe_boundary_span, self._error_boundary_span):
            span.line_color = theme["muted"]
        # the half-plane shading of the eigenvalue plane follows the theme
        upper_color, lower_color = halfplane_colors(self._theme)
        self._upper_halfplane_renderer.glyph.fill_color = upper_color
        self._lower_halfplane_renderer.glyph.fill_color = lower_color
        self._upper_halfplane_label.text_color = upper_color
        self._lower_halfplane_label.text_color = lower_color
        # the pair boxes of the weight sliders carry the pair colors
        colors = self._pair_colors()
        for box, index in self._pair_boxes:
            box.styles = {**box.styles, "border": f"2px solid {colors[index]}"}
        for element, kind, step in self._themed_divs:
            if kind == "header":
                element.styles = header_styles(step, self._theme)
            elif kind == "hint":
                element.styles = hint_styles(self._theme)
            elif kind == "info12":
                element.styles = info_styles(self._theme, "12px")
            elif kind == "info13":
                element.styles = info_styles(self._theme, "13px")
            elif kind == "app_title":
                element.styles = {"color": theme["text"], "font-size": "15px",
                                  "padding": "8px 6px"}
            elif kind == "wf_name":
                element.styles = {"color": theme["accents"][step], "font-weight": "600"}
            elif kind == "wf_desc":
                element.styles = {"color": theme["text"], "font-size": "12px"}
            elif kind == "btn":
                # accent buttons (workflow "Open") consume bokeh's button
                # custom properties; plain inline styles never reach the
                # .bk-btn surface inside the widget's shadow root
                element.styles = accent_button_styles(self._theme, step)
            elif kind == "btn_default":
                element.styles = button_styles(self._theme)
            elif kind == "btn_primary":
                element.styles = primary_button_styles(self._theme)
        # recreate the stylesheets and re-attach them: a GlobalInlineStyleSheet
        # is injected into the page only once, so updating its css alone does
        # not re-theme the page
        self._page_sheet = GlobalInlineStyleSheet(css=page_css(self._theme))
        self._tab_sheet = InlineStyleSheet(css=tab_css(self._theme, vertical=True))
        self._mode_tab_sheet = InlineStyleSheet(
            css=tab_css(self._theme, accent=theme["accents"]["modes"],
                        vertical=False))
        self._layout_root.stylesheets = [self._page_sheet]
        self._tabs.stylesheets = [self._tab_sheet]
        self._mode_tabs.stylesheets = [self._mode_tab_sheet]
        # theme every widget currently in the layout (doc.models would be empty
        # at the initial call — the root is only added to the document later)
        for model in self._layout_root.references():
            self._style_widget(model)

    def _style_widget(self, model: Model) -> None:
        """
        Apply the current theme to one widget model.

        Bokeh widgets theme via CSS custom properties that inherit into their
        shadow roots; buttons/toggles are excluded — their ``.bk-btn`` surface
        consumes the ``--default*``/``--primary*`` families and is themed
        through the btn kinds of the themed registry. Input widgets additionally
        need a shadow-root stylesheet: the browser UA stylesheet forces a black
        text color on form controls, and sliders need one for the nouislider
        track and handle.

        :param model: widget model to style (no-op for other models)
        :type model: Model
        :return: None
        :rtype: None
        """
        if not isinstance(model, (Spinner, Select, Slider, Switch)):
            return
        theme = THEMES[self._theme]
        widget_styles = {"--background-color": theme["box_bg"],
                         "--surface-background-color": theme["box_bg"],
                         "--color": theme["text"],
                         "--hover-color": theme["tab_hover_bg"],
                         "--disabled-color": theme["muted"]}
        if isinstance(model, Switch):
            model.styles = {**widget_styles, **switch_styles(self._theme)}
        elif isinstance(model, Slider):
            model.styles = widget_styles
            model.bar_color = theme["slider_connect"]
            model.stylesheets = [InlineStyleSheet(
                css=slider_stylesheet(self._theme))]
        else:
            model.styles = widget_styles
            model.stylesheets = [InlineStyleSheet(
                css=input_stylesheet(self._theme))]

    def _on_theme_switch(self, attr: str, old: bool, new: bool) -> None:
        """
        Switch the application theme and re-apply all colors in place.

        :param attr: name of the changed widget attribute
        :type attr: str
        :param old: previous switch state
        :type old: bool
        :param new: new switch state (``True`` = dark, ``False`` = light)
        :type new: bool
        :return: None
        :rtype: None
        """
        new_theme = "dark" if new else "light"
        if new_theme == self._theme:
            return
        self._theme = new_theme
        self._apply_theme()

    def _make_goto(self, index: int):
        """
        Create a navigation callback that opens the given tab.

        :param index: tab index to activate
        :type index: int
        :return: callback suitable for a Bokeh button
        :rtype: callable
        """
        def goto() -> None:
            """
            Activate the target tab.

            :return: None
            :rtype: None
            """
            self._tabs.active = index
        return goto

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
        self._morph_slider.on_change("value", self._on_morph_change)
        self._run_button.on_click(self._run_dmd)
        self._reset_signal_button.on_click(self._reset_signal_parameters)
        for key, button in self._reset_fit_buttons.items():
            button.on_click(self._reset_to_fit)
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
        self._play_toggle.on_change("active", self._on_recon_play)
        self._signal_play_toggle.on_change("active", self._on_signal_play)
        self._eigen_play_toggle.on_change("active", self._on_eigen_play)
        self._modes_play_toggle.on_change("active", self._on_modes_play)
        # prediction panel: horizon (recomputes the prediction, no refit),
        # probe point, independent time cursor, and playback
        self._n_pred_spinner.on_change("value", self._on_n_pred_change)
        self._probe_slider.on_change("value", self._on_probe_change)
        self._predict_time_slider.on_change("value", self._on_predict_time_change)
        self._predict_play_toggle.on_change("active", self._on_predict_play)

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
        except (AssertionError, IndexError, RuntimeError, ValueError) as error:
            logger.exception("The DMD fit failed")
            error_color = theme_colors(self._theme)["line_error"]
            self._status_div.text = (f"<span style='color:{error_color}'><b>the DMD fit "
                                     f"failed:</b> {error}</span>")
            return
        self._signal = signal
        self._result = result
        # the clean continuation of the modal composition is the prediction's
        # ground truth (future noise is unpredictable by definition)
        self._recompute_truth()
        self._recompute_fields()
        # keep the slider ends consistent before any view reads them
        self._update_time_slider()
        self._update_predict_slider()
        self._update_mode_slider()
        self._update_field_view()
        self._update_quantities()
        self._update_mode_view()
        self._update_mode_weights()
        self._update_time_limits()
        # the time view must be filled explicitly: with weight sliders (created
        # at their current value) no change callback fires on a fresh fit — the
        # old checkbox reset did that as a side effect of re-selecting
        self._update_time_view()
        self._update_prediction_views()
        error = float(np.abs(self._reconstruction - signal.samples).max())
        # only the automatic rank and the error are shown; an explicitly set rank
        # is known to the user already
        if self._rank_spinner.value == 0:
            self._status_div.text = (f"rank automatic ({result.n_modes}) · "
                                     f"round-trip error {error:.1e}")
        else:
            self._status_div.text = f"round-trip error {error:.1e}"

    def _reset_signal_parameters(self) -> None:
        """
        Restore the default signal and fit parameters and refit the DMD.

        The spinner values are written directly under the ``_syncing_params``
        guard so the change callbacks neither refit repeatedly nor clamp the
        frequency bounds against each other; the bounds are restored explicitly
        and one single refit follows.

        :return: None
        :rtype: None
        """
        if self._syncing_params:
            return
        self._syncing_params = True
        for name, value in _DEFAULT_SIGNAL_PARAMS.items():
            setattr(getattr(self, f"_{name}_spinner"), "value", value)
        self._f_min_spinner.high = _DEFAULT_SIGNAL_PARAMS["f_max"]
        self._f_max_spinner.low = _DEFAULT_SIGNAL_PARAMS["f_min"]
        self._auto_toggle.value = "on"
        self._syncing_params = False
        self._run_dmd()

    def _reset_to_fit(self) -> None:
        """
        Restore the fitted eigenvalues, amplitudes, and mode shapes and refresh
        all views.

        The reset undoes every eigenvalue drag (including unit-circle-locked
        ones), every |b| gain, every mode-shape morph, and every mode weight:
        the cached fit baselines are written back, the weight and gain factors
        return to one, the reconstruction and prediction are recomputed, and
        the eigenvalue plane, spectrum, growth-direction plane, time/dynamics
        views, and the prediction views are updated. The weight sliders are
        rebuilt at their neutral value one.

        :return: None
        :rtype: None
        """
        result = self._result
        if result is None or self._signal is None or self._fit_eigvals is None \
                or self._fit_amplitudes is None or self._fit_modes is None:
            return
        result.eigvals = self._fit_eigvals.copy()
        result.amplitudes = self._fit_amplitudes.copy()
        result.modes = self._fit_modes.copy()
        self._locked_edit_indices = set()
        self._drag_cache = (result.eigvals.real.copy(), result.eigvals.imag.copy())
        self._syncing_amplitude = True
        self._amplitude_spinner.value = 1.0
        self._syncing_amplitude = False
        self._syncing_morph = True
        self._morph_slider.value = 1.0
        self._syncing_morph = False
        if self._mode_weights is not None and self._mode_gains is not None:
            self._mode_weights[:] = 1.0
            self._mode_gains[:] = 1.0
        self._recompute_fields()
        self._update_eigenvalue_markers()
        self._update_spectrum()
        # the shape views must follow the restored modes (the morph edits them
        # in place); rebuilding the weight sliders restores their neutral values
        self._update_mode_view()
        self._update_mode_weights()
        self._update_time_limits()
        self._update_time_view()
        self._update_prediction_views()
        error = float(np.abs(self._reconstruction - self._signal.samples).max())
        self._status_div.text = (f"reset to the fitted values — round-trip error "
                                 f"{error:.1e}")

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

    def _phase_slopes(self) -> np.ndarray:
        """
        Compute the unwrapped phase slope :math:`\\kappa` of every mode shape.

        The slope of :math:`\\arg(\\phi_i)` along space is the wavenumber of the
        mode; its sign together with the sign of the frequency fixes the
        propagation direction via :math:`c = -2\\pi f / \\kappa`.

        :return: phase slope of each mode in radians per unit length, shape ``(n_modes,)``
        :rtype: np.ndarray
        """
        result = self._result
        s = self._signal.space
        return np.array([float(np.polyfit(s, np.unwrap(np.angle(result.mode(i))), 1)[0])
                         for i in range(result.n_modes)])

    def _phase_velocities(self) -> np.ndarray:
        """
        Compute the signed phase velocity :math:`c = -2\\pi f / \\kappa` of every mode.

        The sign of :math:`c` is the propagation direction (positive toward
        :math:`+x`); modes whose shape has no spatial phase variation do not
        propagate and get a velocity of zero.

        :return: phase velocity of each mode in units per second, shape ``(n_modes,)``
        :rtype: np.ndarray
        """
        slopes = self._phase_slopes()
        propagating = np.abs(slopes) > 1e-12
        frequencies = self._result.frequencies
        return np.where(propagating,
                        -2.0 * np.pi * frequencies
                        / np.where(propagating, slopes, 1.0),
                        0.0)

    def _direction_marker_columns(self) -> dict[str, list]:
        """
        Build the travel-direction triangle columns of the eigenvalue plane.

        Each triangle sits above its eigenvalue marker and points toward the
        propagation direction of the mode's wave (rotated toward :math:`+x` or
        :math:`-x`); it shares the pair color and the selection emphasis of the
        marker, and NaN coordinates hide it for non-propagating modes.

        :return: data-source columns ``dir_x``, ``dir_y``, ``dir_angle``,
            ``dir_color``, and ``dir_size`` with one entry per mode
        :rtype: dict[str, list]
        """
        result = self._result
        index = min(int(self._mode_slider.value), result.n_modes - 1)
        velocities = self._phase_velocities()
        colors = self._pair_colors()
        columns: dict[str, list] = dict(dir_x=[], dir_y=[], dir_angle=[],
                                        dir_color=[], dir_size=[])
        for i in range(result.n_modes):
            columns["dir_color"].append(colors[i])
            columns["dir_size"].append(11 if i == index else 8)
            if abs(velocities[i]) <= 1e-12:
                # no propagation: hide the triangle via NaN coordinates
                columns["dir_x"].append(float("nan"))
                columns["dir_y"].append(float("nan"))
                columns["dir_angle"].append(0.0)
                continue
            columns["dir_x"].append(float(result.eigvals[i].real))
            columns["dir_y"].append(float(result.eigvals[i].imag)
                                    + _DIRECTION_MARKER_OFFSET)
            # the default triangle points up: rotate it toward the travel direction
            columns["dir_angle"].append(-0.5 * np.pi if velocities[i] > 0.0
                                        else 0.5 * np.pi)
        return columns

    def _update_field_view(self) -> None:
        """
        Update the fixed y-range of the animated signal view so playback never
        rescales.

        :return: None
        :rtype: None
        """
        signal = self._signal
        vmax = max(float(np.abs(signal.samples).max()), 1e-12)
        # fixed y-range so the animated signal line does not rescale every step
        self._signal_figure.y_range.start = -1.1 * vmax
        self._signal_figure.y_range.end = 1.1 * vmax

    def _update_time_limits(self) -> None:
        """
        Size the fixed y-range of the reconstruction view from the
        reconstruction and the original signal.

        Both curves are drawn in the view over all time levels, and both only
        change when the quantities are edited (refit, |b| scaling, eigenvalue
        or morph edits) — the range is therefore constant under checkbox
        toggles and during playback.

        :return: None
        :rtype: None
        """
        signal = self._signal
        result = self._result
        if signal is None or result is None or result.n_modes == 0:
            return
        vmax = max(float(np.abs(signal.samples).max()), 1e-12)
        if self._reconstruction is not None:
            vmax = max(vmax, float(np.abs(self._reconstruction).max()))
        # a tight margin: the reconstruction should use the frame (the user
        # found 1.1 too generous); playback stays inside since the range is
        # sized by the global maximum over all time levels
        self._time_figure.y_range.start = -1.05 * vmax
        self._time_figure.y_range.end = 1.05 * vmax

    def _prediction_steps(self) -> int:
        """
        Return the number of prediction steps from the spinner.

        :return: number of future time levels beyond the data window
        :rtype: int
        """
        return int(min(max(self._n_pred_spinner.value, 1), 5000))

    def _recompute_fields(self) -> None:
        """
        Recompute the reconstruction and the prediction from the current state.

        The reconstruction covers the fitted data window (time levels
        :math:`0, \\ldots, n_{time} - 1`) and the prediction continues the same
        synthesis formula beyond it with the current (possibly edited)
        quantities — the DMD model advanced into the future.

        :return: None
        :rtype: None
        """
        result = self._result
        signal = self._signal
        if result is None or signal is None:
            return
        self._reconstruction = synthesize(result.modes, result.amplitudes,
                                          result.eigvals, signal.dt, signal.n_time)
        self._prediction = synthesize_window(result.modes, result.amplitudes,
                                             result.eigvals, signal.dt,
                                             signal.n_time, self._prediction_steps())

    def _recompute_truth(self) -> None:
        """
        Recompute the clean continuation of the signal over the prediction window.

        The truth is the modal composition of the generated signal evaluated at
        the future time levels — the noise realization of future snapshots is
        unpredictable by definition, the clean physics is not.

        :return: None
        :rtype: None
        """
        signal = self._signal
        if signal is None:
            return
        steps = self._prediction_steps()
        future_time = signal.dt * np.arange(signal.n_time, signal.n_time + steps)
        self._truth = signal.evaluate_clean(future_time)

    def _apply_scale_factors(self) -> None:
        """
        Write the weighted and gained amplitudes into the current result.

        The effective amplitude of every mode is the fitted amplitude times its
        weight (panel-5 slider, :math:`[0, 1]`) times its gain (panel-2 spinner,
        :math:`[0, 3]`); both factors are real and equal within a conjugate
        pair, so the conjugate symmetry of the amplitudes is preserved exactly.

        :return: None
        :rtype: None
        """
        result = self._result
        if result is None or self._fit_amplitudes is None \
                or self._mode_weights is None or self._mode_gains is None:
            return
        result.amplitudes = self._fit_amplitudes * self._mode_weights * self._mode_gains

    def _update_predict_slider(self) -> None:
        """
        Adjust the bounds of the prediction time slider to data plus horizon.

        The slider operates in seconds and snaps to the frame grid
        (step = Δt); it covers the data window and the prediction window and is
        independent of the shared time cursors of the other panels.

        :return: None
        :rtype: None
        """
        signal = self._signal
        if signal is None:
            return
        end = max((signal.n_time - 1 + self._prediction_steps()) * signal.dt, 0.0)
        self._predict_time_slider.step = signal.dt
        self._predict_time_slider.end = end
        if self._predict_time_slider.value > end:
            self._predict_time_slider.value = end

    def _update_quantities(self) -> None:
        """
        Update the eigenvalue plane, the spectrum, and the continuous-time
        eigenvalue plane with their axis ranges, and refresh the cached editing
        state (pairs, fitted amplitudes, fitted mode shapes, marker positions)
        for the quantity editing.

        :return: None
        :rtype: None
        """
        result = self._result
        if result is None or result.n_modes == 0:
            self._eigenvalue_source.data = dict(re=[], im=[], index=[], frequency=[],
                                                growth=[], color=[], size=[],
                                                dir_x=[], dir_y=[], dir_angle=[],
                                                dir_color=[], dir_size=[])
            self._spectrum_source.data = dict(frequency=[], amplitude=[], index=[],
                                              growth=[], color=[], width=[], size=[])
            self._continuous_source.data = dict(sigma=[], velocity=[], index=[],
                                                color=[], size=[], frequency=[],
                                                direction=[])
            self._cached_pairs = []
            self._fit_amplitudes = None
            self._fit_eigvals = None
            self._fit_modes = None
            self._mode_weights = None
            self._mode_gains = None
            self._drag_cache = (np.array([]), np.array([]))
            return
        self._update_eigenvalue_markers()
        # cache the fitted baseline (eigenvalues, amplitudes, mode shapes), the
        # pair structure, and marker positions for the quantity editing
        # (dragging, amplitude scaling, shape morphing, unit-circle lock reset)
        self._cached_pairs = result.pairs
        self._fit_amplitudes = result.amplitudes.copy()
        self._fit_eigvals = result.eigvals.copy()
        self._fit_modes = result.modes.copy()
        # a fresh fit (or a re-sort, which bakes the current amplitudes into
        # the new baseline) starts with neutral weights and gains
        self._mode_weights = np.ones(result.n_modes)
        self._mode_gains = np.ones(result.n_modes)
        self._locked_edit_indices = set()
        self._drag_cache = (result.eigvals.real.copy(), result.eigvals.imag.copy())
        self._syncing_amplitude = True
        self._amplitude_spinner.value = 1.0
        self._syncing_amplitude = False
        # a refit restores fresh shapes (t = 1); a re-sort bakes the current
        # morph into the new baseline, so the slider reads 1 in both cases
        self._syncing_morph = True
        self._morph_slider.value = 1.0
        self._syncing_morph = False
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
        constant offset) are gray. The palette follows the theme (brighter
        colors in dark mode).

        :return: one marker color per fitted mode
        :rtype: list[str]
        """
        result = self._result
        if result is None:
            return []
        palette = PAIR_PALETTE_DARK if self._theme == "dark" else PAIR_PALETTE
        assignment = {}
        for pair_index, (i, j) in enumerate(result.pairs):
            color = palette[pair_index % len(palette)]
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
                                                growth=[], color=[], size=[],
                                                dir_x=[], dir_y=[], dir_angle=[],
                                                dir_color=[], dir_size=[])
            self._dynamics_all_source.data = dict(x=[], y=[], index=[], color=[],
                                                  size=[])
            return
        index = min(int(self._mode_slider.value), result.n_modes - 1)
        colors = self._pair_colors()
        sizes = [14 if i == index else 8 for i in range(result.n_modes)]
        # the plane source drives the drag tool: guard its refill against the
        # drag callback (fired by this very assignment)
        self._syncing_drag = True
        data = dict(re=result.eigvals.real,
                    im=result.eigvals.imag,
                    index=list(range(result.n_modes)),
                    frequency=result.frequencies,
                    growth=result.growth_rates,
                    color=colors, size=sizes)
        data.update(self._direction_marker_columns())
        self._eigenvalue_source.data = data
        self._syncing_drag = False
        self._dynamics_all_source.data = dict(x=result.eigvals.real,
                                              y=result.eigvals.imag,
                                              index=list(range(result.n_modes)),
                                              color=colors, size=sizes)

    def _update_spectrum(self) -> None:
        """
        Update the DMD spectrum and the growth-direction plane with stability
        colors and the selection emphasis.

        Stems and markers are red for growing, blue for decaying and gray for
        neutral modes; the selected eigenvalue is drawn larger. The dashed
        Nyquist spans and the frequency axis follow the current time step; the
        growth-direction plane plots the growth rate over the signed phase
        velocity :math:`c = -2\\pi f/\\kappa` and sizes both of its axes from
        the fitted data.

        :return: None
        :rtype: None
        """
        result = self._result
        if result is None or result.n_modes == 0:
            self._spectrum_source.data = dict(frequency=[], amplitude=[], index=[],
                                              growth=[], color=[], width=[], size=[])
            self._continuous_source.data = dict(sigma=[], velocity=[], index=[],
                                                color=[], size=[], frequency=[],
                                                direction=[])
            return
        index = min(int(self._mode_slider.value), result.n_modes - 1)
        colors = self._pair_colors()
        sizes = [14 if i == index else 7 for i in range(result.n_modes)]
        widths = [3 if i == index else 2 for i in range(result.n_modes)]
        frequencies = result.frequencies
        self._spectrum_source.data = dict(frequency=frequencies,
                                          amplitude=np.abs(result.amplitudes),
                                          index=list(range(result.n_modes)),
                                          growth=result.growth_rates,
                                          color=colors, width=widths, size=sizes)
        # phase velocity c = -2*pi*f/kappa: its sign is the propagation
        # direction; modes without spatial phase variation do not propagate
        velocities = self._phase_velocities()
        directions = ["+x" if v > 1e-12 else ("−x" if v < -1e-12 else "—")
                      for v in velocities]
        self._continuous_source.data = dict(sigma=result.growth_rates,
                                            velocity=velocities,
                                            index=list(range(result.n_modes)),
                                            color=colors, size=sizes,
                                            frequency=frequencies,
                                            direction=directions)
        nyquist = 0.5 / self._signal.dt
        self._spectrum_figure.x_range.start = -1.15 * nyquist
        self._spectrum_figure.x_range.end = 1.15 * nyquist
        self._nyquist_spans[0].location = -nyquist
        self._nyquist_spans[1].location = nyquist
        # both axes of the growth-direction plane follow the fitted data
        c_limit = max(1.15 * float(np.abs(velocities).max()), 1.0)
        self._continuous_figure.y_range.start = -c_limit
        self._continuous_figure.y_range.end = c_limit
        sigma_limit = max(1.15 * float(np.abs(result.growth_rates).max()), 1.0)
        self._continuous_figure.x_range.start = -sigma_limit
        self._continuous_figure.x_range.end = sigma_limit

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
        Update the quadrature view, the phase view, and the direction readout.

        The shared y-ruler of the quadrature view is sized by the fitted
        quadratures of the selected mode and stays constant while the morph
        slider scales the imaginary part inside that fixed frame; the faded
        reference line shows the fitted imaginary part.

        :return: None
        :rtype: None
        """
        result = self._result
        if result is None or result.n_modes == 0:
            self._quadrature_source.data = dict(s=[], real=[], imag=[],
                                                reference=[])
            self._phase_source.data = dict(s=[], value=[], unwrapped=[],
                                           reference=[])
            self._direction_div.text = "no modes fitted"
            return
        index = min(int(self._mode_slider.value), result.n_modes - 1)
        shape = result.mode(index)
        s = self._signal.space
        wrapped = np.angle(shape)
        # the faded reference is the fitted shape: morph edits show against it
        fit_shape = (self._fit_modes[:, index] if self._fit_modes is not None
                     else shape)
        fit_wrapped = np.angle(fit_shape)
        self._quadrature_source.data = dict(s=s, real=np.real(shape),
                                            imag=np.imag(shape),
                                            reference=np.imag(fit_shape))
        self._phase_source.data = dict(s=s, value=wrapped,
                                       unwrapped=np.unwrap(wrapped),
                                       reference=fit_wrapped)
        if self._fit_modes is not None:
            # one constant ruler for the quadrature view, sized by the fit
            re_fit = np.real(self._fit_modes[:, index])
            im_fit = np.imag(self._fit_modes[:, index])
            vmax = max(float(np.abs(re_fit).max()), float(np.abs(im_fit).max()),
                       1e-12)
            self._shape_y_range.start = -1.1 * vmax
            self._shape_y_range.end = 1.1 * vmax
        self._update_space_time_view()

    def _update_space_time_view(self) -> None:
        """
        Update the space-time heatmap of the selected mode's contribution.

        The heatmap shows :math:`\\Re\\{\\phi b \\lambda^t\\}` for all time
        levels at once: the horizontal stripe spacing is the wavelength
        :math:`2\\pi/\\kappa` (a property of the shape), the vertical spacing
        the temporal period :math:`1/f` (a property of the eigenvalue), and
        the stripe slope the propagation speed. The color limits are
        symmetric around zero and sized by the field. The view is refreshed
        only when its inputs change (refit, sort, mode switch, morph,
        eigenvalue or amplitude edits) — not per animation tick, since it
        shows all times and does not depend on the time cursor.

        :return: None
        :rtype: None
        """
        result = self._result
        signal = self._signal
        if result is None or result.n_modes == 0 or signal is None:
            self._space_time_source.data = dict(field=[], dh=[])
            return
        index = min(int(self._mode_slider.value), result.n_modes - 1)
        t_values = np.arange(signal.n_time)
        dynamics = result.amplitudes[index] * result.eigvals[index] ** t_values
        # the image glyph expects rows = vertical (time, height) and columns =
        # horizontal (space, width), with row 0 rendered at the top: build
        # F[t, x] and store it time-flipped so time increases upward
        field = np.real(result.mode(index)[None, :] * dynamics[:, None])[::-1, :]
        t_max = (signal.n_time - 1) * signal.dt
        vmax = max(float(np.abs(field).max()), 1e-12)
        self._space_time_mapper.low = -vmax
        self._space_time_mapper.high = vmax
        self._space_time_figure.y_range.end = t_max
        self._space_time_source.data = dict(field=[field], dh=[t_max])

    def _update_direction_view(self) -> None:
        """
        Update the traveling-wave animation, the direction arrow, and the readout.

        The wave contribution of the selected mode is shown at the current time
        with a fixed y-range covering the whole animation, so growth is visible
        without rescaling. The propagation analysis follows from the phase slope
        :math:`\\kappa` (unwrapped linear fit) and the signed frequency. Below
        the standing threshold of the morph slider the arrow is hidden and the
        readout reports a standing wave; otherwise the arrow length scales with
        the net propagation of the pattern (traveling fraction :math:`t` times
        the speed :math:`|c|`, clamped to the plot domain).

        :return: None
        :rtype: None
        """
        result = self._result
        signal = self._signal
        if result is None or result.n_modes == 0 or signal is None:
            self._direction_source.data = dict(s=[], value=[], reference=[])
            self._direction_div.text = "no modes fitted"
            self._direction_arrow.visible = False
            return
        index = min(int(self._mode_slider.value), result.n_modes - 1)
        t_index = min(round(self._time_slider.value / signal.dt), signal.n_time - 1)
        shape = result.mode(index)
        amplitude = result.amplitudes[index]
        wave = np.real(shape * (amplitude * result.eigvals[index] ** t_index))
        # faded reference: the contribution of the fitted shape at the same
        # time level (isolates the morph effect on the traveling wave)
        fit_shape = (self._fit_modes[:, index] if self._fit_modes is not None
                     else shape)
        reference = np.real(fit_shape * (amplitude * result.eigvals[index] ** t_index))
        self._direction_source.data = dict(s=self._signal.space, value=wave,
                                           reference=reference)
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
        slope = float(self._phase_slopes()[index])
        t = self._morph_factor(index)
        if abs(slope) < 1e-12:
            self._direction_arrow.visible = False
            self._direction_div.text = (header
                                        + "spatially constant shape — no propagation "
                                        "direction (wavenumber κ ≈ 0)")
            return
        if t < _MORPH_STANDING_THRESHOLD:
            # pure standing: no net propagation and no direction arrow; the
            # wavelength follows from the fitted shape (the morphed phase has
            # no meaningful slope as t approaches zero)
            self._direction_arrow.visible = False
            text = header
            fit_slope = float(np.polyfit(self._signal.space,
                                         np.unwrap(np.angle(
                                             self._fit_modes[:, index])), 1)[0])
            if abs(fit_slope) > 1e-12:
                text += (f"κ = {fit_slope:+.3f} rad per unit length → "
                         f"λ_w = {2.0 * np.pi / abs(fit_slope):.2f}<br>")
            text += ("standing wave — no net propagation (equal superposition "
                     "of opposite-traveling waves)")
            self._direction_div.text = text
            return
        wavelength = 2.0 * np.pi / abs(slope)
        if abs(frequency) < 1e-12:
            self._direction_arrow.visible = False
            self._direction_div.text = (header
                                        + f"κ = {slope:.3f} rad per unit length → "
                                        f"wavelength λ_w = {wavelength:.2f}"
                                        " — standing pattern (eigenvalue on the "
                                        "real axis, zero frequency)")
            return
        velocity = -2.0 * np.pi * frequency / slope
        direction = "+x" if velocity > 0.0 else "−x"
        eigval = result.eigvals[index]
        side = "above" if eigval.imag > 0.0 else "below"
        sign = ">" if eigval.imag > 0.0 else "<"
        self._direction_arrow.visible = True
        # the arrow length scales with the net propagation of the pattern
        # (traveling fraction t times the speed |c|), clamped to the domain
        # and centered on the plot
        length = min(max(_ARROW_SPEED_SCALE * abs(velocity) * t,
                         _ARROW_MIN_LENGTH), _ARROW_MAX_LENGTH)
        if velocity > 0.0:
            self._direction_arrow.x_start = 0.5 - 0.5 * length
            self._direction_arrow.x_end = 0.5 + 0.5 * length
        else:
            self._direction_arrow.x_start = 0.5 + 0.5 * length
            self._direction_arrow.x_end = 0.5 - 0.5 * length
        self._direction_arrow.y_start = 1.1 * envelope
        self._direction_arrow.y_end = 1.1 * envelope
        arrow_char = "→" if velocity > 0.0 else "←"
        # a morphed shape is an exact superposition of two opposite-traveling
        # waves with amplitudes (1 ± t) / 2: report the traveling/standing split
        if t < 1.0 - 1e-9:
            mixed = (f"<br>mixed wave: {50.0 * (1.0 + t):.0f}% traveling toward "
                     f"{direction} {arrow_char} + {50.0 * (1.0 - t):.0f}% "
                     "counter-traveling (standing fraction)")
        else:
            mixed = ""
        self._direction_div.text = (
            header
            + f"κ = {slope:+.3f} rad per unit length → λ_w = {wavelength:.2f}<br>"
            + f"Im λ = {eigval.imag:+.4f} → eigenvalue {side} the real axis "
              f"(f {sign} 0)<br>"
            + f"phase velocity c = −2πf/κ = {abs(velocity):.2f} units per second "
              f"toward {direction} {arrow_char}"
            + mixed)

    def _update_mode_weights(self) -> None:
        """
        Rebuild the weight-slider grid; all modes start at their fitted weight.

        One slider per eigenvalue spans 0 (silenced) to 1 (fitted contribution);
        the sliders of a conjugate pair are grouped inside one pair-colored box
        so the pairing stays visible, unpaired modes (e.g. a constant offset)
        get a gray box. The boxes are laid out in rows of four.

        :return: None
        :rtype: None
        """
        result = self._result
        if result is None or result.n_modes == 0:
            self._weights_layout.children = []
            self._weight_sliders = []
            self._weight_slider_by_mode = {}
            self._pair_boxes = []
            return
        # order the sliders pair-first so the two members of every conjugate
        # pair sit adjacent inside their shared box; unpaired modes (e.g. a
        # constant offset) get their own gray box
        ordered = [index for pair in result.pairs for index in pair]
        paired = {index for pair in result.pairs for index in pair}
        singles = [i for i in range(result.n_modes) if i not in paired]
        colors = self._pair_colors()
        self._weight_sliders = []
        self._pair_boxes = []
        boxes = []
        for k in range(0, len(ordered), 2):
            members = ordered[k:k + 2]
            box = self._weight_box(members, colors[members[0]])
            boxes.append(box)
            self._pair_boxes.append((box, members[0]))
        for index in singles:
            box = self._weight_box([index], colors[index])
            boxes.append(box)
            self._pair_boxes.append((box, index))
        rows = [row(*boxes[k:k + _WEIGHT_BOXES_PER_ROW], spacing=8)
                for k in range(0, len(boxes), _WEIGHT_BOXES_PER_ROW)]
        self._weights_layout.children = rows

    def _weight_box(self, indices: list[int], color: str) -> Column:
        """
        Create one bordered weight-slider box for a mode or a conjugate pair.

        :param indices: mode indices of the box (one or two members)
        :type indices: list[int]
        :param color: border color identifying the conjugate pair
        :type color: str
        :return: box layout with one weight slider per mode
        :rtype: Column
        """
        result = self._result
        sliders = []
        for index in indices:
            frequency = result.frequencies[index]
            slider = Slider(title=f"mode {index} · f = {frequency:+.2f} Hz",
                            start=0.0, end=1.0, value=float(self._mode_weights[index]),
                            step=0.01, width=258)
            slider.on_change("value", partial(self._on_weight_change, index=index))
            # rebuilt per fit: apply the current theme right away (a theme
            # switch would only reach them on the next toggle otherwise)
            self._style_widget(slider)
            sliders.append(slider)
            self._weight_sliders.append(slider)
            self._weight_slider_by_mode[index] = slider
        box = column(*sliders, spacing=4,
                     styles={"border": f"2px solid {color}", "border-radius": "6px",
                             "padding": "2px 4px"})
        return box

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
        index = 0
        if result is not None and result.n_modes > 0:
            index = min(int(self._mode_slider.value), result.n_modes - 1)
            selected = self._mode_field(index, t_index)
        self._time_source.data = dict(s=self._signal.space, original=original,
                                      reconstruction=reconstruction, selected=selected)
        # thin per-eigenvalue contribution lines in pair colors for all modes
        # with a positive weight; the selected eigenvalue's line is thicker in
        # its pair color
        pair_xs, pair_ys, pair_colors, pair_index, pair_f = [], [], [], [], []
        if result is not None and result.n_modes > 0:
            colors = self._pair_colors()
            for i in range(result.n_modes):
                if self._mode_weights is not None and self._mode_weights[i] > 0.0:
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
        # the trail carries the pair color of the selected eigenvalue
        if result is not None and result.n_modes > 0:
            color = self._pair_colors()[index]
            powers = result.eigvals[index] ** np.arange(1, max(t_index, 1) + 1)
            self._dynamics_trail_source.data = dict(x=powers.real, y=powers.imag,
                                                    color=[color] * len(powers))
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
        # the direction triangles follow the dragged eigenvalues (position and,
        # after a real-axis crossing, the flipped travel direction)
        updated.update(self._direction_marker_columns())
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
        # feed the edited eigenvalues back into the reconstruction, prediction,
        # and all views
        self._recompute_fields()
        error = float(np.abs(self._reconstruction - self._signal.samples).max())
        self._status_div.text = f"edited — max deviation from the original: {error:.1e}"
        self._update_spectrum()
        self._update_space_time_view()
        self._update_time_limits()
        self._update_time_view()
        self._update_prediction_views()

    def _on_amplitude_change(self, attr: str, old: object, new: object) -> None:
        """
        Apply the extra gain of the selected eigenvalue and mirror its partner.

        The gain multiplies the fitted amplitude on top of the mode weight
        (total scale = weight · gain), so the spinner position corresponds to
        the current extra gain; the reconstruction, prediction, and all views
        are updated live.

        :param attr: name of the changed widget attribute
        :type attr: str
        :param old: previous attribute value
        :type old: object
        :param new: new attribute value (the extra gain)
        :type new: object
        :return: None
        :rtype: None
        """
        if self._syncing_amplitude or self._result is None or self._signal is None \
                or self._fit_amplitudes is None or self._mode_gains is None:
            return
        index = min(int(self._mode_slider.value), self._result.n_modes - 1)
        # typed values are not clamped by the widget: enforce the declared bounds
        gain = min(max(float(new), 0.0), 3.0)
        if gain != float(new):
            self._syncing_amplitude = True
            self._amplitude_spinner.value = gain
            self._syncing_amplitude = False
        # the gain applies to the selected eigenvalue and its conjugate partner
        self._mode_gains[index] = gain
        for a, b in self._cached_pairs:
            if a == index:
                self._mode_gains[b] = gain
            elif b == index:
                self._mode_gains[a] = gain
        self._apply_scale_factors()
        self._recompute_fields()
        error = float(np.abs(self._reconstruction - self._signal.samples).max())
        self._status_div.text = f"edited — max deviation from the original: {error:.1e}"
        self._update_spectrum()
        self._update_space_time_view()
        self._update_time_limits()
        self._update_time_view()
        self._update_prediction_views()

    def _on_weight_change(self, attr: str, old: object, new: object,
                          index: int = 0) -> None:
        """
        Apply the weight of one eigenvalue and mirror its conjugate partner.

        The weight multiplies the fitted amplitude (times the extra gain of the
        panel-2 spinner); the partner's slider follows so both members of the
        pair always carry the same weight, keeping the reconstructed field real.

        :param attr: name of the changed widget attribute
        :type attr: str
        :param old: previous attribute value
        :type old: object
        :param new: new attribute value (the weight)
        :type new: object
        :param index: mode index of the slider that fired the change
        :type index: int
        :return: None
        :rtype: None
        """
        if self._syncing_weights or self._result is None or self._signal is None \
                or self._mode_weights is None or self._mode_gains is None \
                or index >= self._result.n_modes:
            return
        # typed values are not clamped by the widget: enforce the declared bounds
        weight = min(max(float(new), 0.0), 1.0)
        if weight != float(new):
            self._syncing_weights = True
            self._weight_slider_by_mode[index].value = weight
            self._syncing_weights = False
        self._mode_weights[index] = weight
        # mirror the conjugate partner (one physical wave, one weight)
        for a, b in self._cached_pairs:
            if a == index:
                self._syncing_weights = True
                self._weight_slider_by_mode[b].value = weight
                self._syncing_weights = False
                self._mode_weights[b] = weight
            elif b == index:
                self._syncing_weights = True
                self._weight_slider_by_mode[a].value = weight
                self._syncing_weights = False
                self._mode_weights[a] = weight
        self._apply_scale_factors()
        self._recompute_fields()
        error = float(np.abs(self._reconstruction - self._signal.samples).max())
        self._status_div.text = f"edited — max deviation from the original: {error:.1e}"
        self._update_spectrum()
        self._update_space_time_view()
        self._update_time_limits()
        self._update_time_view()
        self._update_prediction_views()

    def _morph_factor(self, index: int) -> float:
        """
        Recover the quadrature mix factor of a mode from the edited shape.

        The morphed shape stores the imaginary part scaled by the factor, so
        the factor follows from a projection of the current imaginary part
        onto the fitted one; shapes without an imaginary part (e.g. a constant
        offset mode) report the neutral factor 1.

        :param index: mode index
        :type index: int
        :return: quadrature mix factor in ``[0, 1]`` (1 = fitted traveling wave)
        :rtype: float
        """
        if self._fit_modes is None or index >= self._fit_modes.shape[1]:
            return 1.0
        fit_imag = np.imag(self._fit_modes[:, index])
        norm = float(fit_imag @ fit_imag)
        if norm <= 1e-30:
            return 1.0
        shape_imag = np.imag(self._result.modes[:, index])
        return float(min(max((shape_imag @ fit_imag) / norm, 0.0), 1.0))

    def _apply_morph(self, index: int, t: float) -> None:
        """
        Apply the quadrature mix factor to a mode shape and refresh all views.

        The morphed shape is ``Re(φ) + i·t·Im(φ)`` of the fitted baseline: t = 1
        keeps the fitted traveling wave and t = 0 a pure standing pattern
        (equal superposition of two opposite-traveling waves). The conjugate
        partner is mirrored so the pair stays one physical wave, the
        reconstruction is recomputed, and the mode-shape, spectrum, and
        time/direction views are updated.

        :param index: mode index to morph
        :type index: int
        :param t: quadrature mix factor in ``[0, 1]``
        :type t: float
        :return: None
        :rtype: None
        """
        result = self._result
        shape = self._fit_modes[:, index]
        morphed = shape.real + 1j * t * shape.imag
        result.modes[:, index] = morphed
        # mirror the conjugate partner (same physical wave, conj shape)
        for a, b in self._cached_pairs:
            if a == index:
                result.modes[:, b] = np.conj(morphed)
            elif b == index:
                result.modes[:, a] = np.conj(morphed)
        self._recompute_fields()
        self._update_mode_view()
        self._update_spectrum()
        self._update_space_time_view()
        self._update_time_limits()
        self._update_time_view()
        self._update_prediction_views()
        error = float(np.abs(self._reconstruction - self._signal.samples).max())
        self._status_div.text = f"edited — max deviation from the original: {error:.1e}"

    def _on_morph_change(self, attr: str, old: object, new: object) -> None:
        """
        Apply the slider's quadrature mix to the selected mode shape live.

        Typed slider values are not clamped by the widget, so the allowed
        range is enforced here before anything is recomputed.

        :param attr: name of the changed widget attribute
        :type attr: str
        :param old: previous attribute value
        :type old: object
        :param new: new attribute value (the quadrature mix factor)
        :type new: object
        :return: None
        :rtype: None
        """
        if self._syncing_morph or self._result is None or self._signal is None \
                or self._fit_modes is None:
            return
        t = min(max(float(new), 0.0), 1.0)
        if t != float(new):
            self._syncing_morph = True
            self._morph_slider.value = t
            self._syncing_morph = False
        index = min(int(self._mode_slider.value), self._result.n_modes - 1)
        self._apply_morph(index, t)

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
        self._update_mode_weights()
        self._update_time_view()
        self._update_prediction_views()

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
        if self._syncing_params:
            return
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
        if self._syncing_params:
            return
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
        if self._syncing_params:
            return
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
        if self._syncing_params:
            return
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
            self._recompute_fields()
        self._update_mode_view()
        self._update_eigenvalue_markers()
        self._update_spectrum()
        # selecting a different eigenvalue restarts the shared time cursor
        self._time_slider.value = 0.0
        # the amplitude spinner follows the extra gain of the newly selected
        # eigenvalue (the mode weight lives on its panel-5 slider)
        index = min(int(self._mode_slider.value), self._result.n_modes - 1) \
            if self._result is not None and self._result.n_modes > 0 else 0
        if self._mode_gains is not None and index < len(self._mode_gains):
            gain = float(self._mode_gains[index])
        else:
            gain = 1.0
        self._syncing_amplitude = True
        self._amplitude_spinner.value = gain
        self._syncing_amplitude = False
        # the morph slider follows the selected mode's current quadrature mix
        self._syncing_morph = True
        self._morph_slider.value = self._morph_factor(index)
        self._syncing_morph = False

    def _prediction_window(self) -> tuple[float, float]:
        """
        Compute the visible time window of the prediction views.

        The frame is the data window plus a small PEEK into the prediction
        (so the static view shows what is about to happen); it stays STATIC
        while the cursor is inside the data window and starts SLIDING once
        the prediction begins — a strip chart with the cursor at a fixed
        offset from the left edge and the peek always ahead of it, clamped at
        the end of the horizon. Both prediction views share this rule.

        :return: visible time window ``(start, end)`` in seconds
        :rtype: tuple[float, float]
        """
        signal = self._signal
        boundary = (signal.n_time - 1) * signal.dt
        horizon = boundary + self._prediction_steps() * signal.dt
        peek = min(_PREDICTION_PEEK_DATA * boundary,
                   _PREDICTION_PEEK_PREDICTION * self._prediction_steps() * signal.dt)
        width = boundary + peek
        start = max(0.0, self._predict_time_slider.value - boundary)
        end = min(start + width, horizon)
        return max(0.0, end - width), end

    def _position_prediction_labels(self, labels: dict[str, Label], start: float,
                                    end: float, y_end: float) -> None:
        """
        Place the stage labels over their visible region of the time window.

        The "reconstruction" label is centered over the visible intersection
        of the window with the shaded data window and the "prediction" label
        over the visible unshaded part; a label is hidden while its region is
        not visible at all. Both sit at 70 % of the y-range (upper-middle of
        the frame, clearly below the top edge).

        :param labels: the figure's stage labels (``"reconstruction"`` and
            ``"prediction"``)
        :type labels: dict[str, Label]
        :param start: visible time-window start in seconds
        :type start: float
        :param end: visible time-window end in seconds
        :type end: float
        :param y_end: upper end of the figure's y-range
        :type y_end: float
        :return: None
        :rtype: None
        """
        boundary = (self._signal.n_time - 1) * self._signal.dt
        shaded_hi = min(end, boundary)
        if shaded_hi > start:
            labels["reconstruction"].visible = True
            labels["reconstruction"].x = 0.5 * (start + shaded_hi)
            labels["reconstruction"].y = 0.7 * y_end
        else:
            labels["reconstruction"].visible = False
        unshaded_lo = max(start, boundary)
        if end > unshaded_lo:
            labels["prediction"].visible = True
            labels["prediction"].x = 0.5 * (unshaded_lo + end)
            labels["prediction"].y = 0.7 * y_end
        else:
            labels["prediction"].visible = False

    def _update_prediction_probe_view(self) -> None:
        """
        Update the prediction time series of the prediction panel.

        The model curve covers the whole horizon (reconstruction inside the
        data window, prediction beyond), the original data ends at the
        boundary, and the clean truth continues beyond it; missing segments
        are masked with NaN so the lines break instead of closing. The title
        carries the current probe position.

        :return: None
        :rtype: None
        """
        result = self._result
        signal = self._signal
        if result is None or result.n_modes == 0 or signal is None \
                or self._prediction is None or self._truth is None:
            self._predict_probe_source.data = dict(t=[], model=[], original=[],
                                                   truth=[])
            return
        steps = self._prediction_steps()
        probe = min(round(self._probe_slider.value / signal.dx), signal.n_space - 1)
        horizon_time = signal.dt * np.arange(signal.n_time + steps)
        original = np.concatenate([signal.samples[probe, :],
                                   np.full(steps, np.nan)])
        truth = np.concatenate([np.full(signal.n_time, np.nan), self._truth[probe, :]])
        model = np.concatenate([self._reconstruction[probe, :],
                                self._prediction[probe, :]])
        self._predict_probe_source.data = dict(t=horizon_time, model=model,
                                               original=original, truth=truth)
        boundary = (signal.n_time - 1) * signal.dt
        self._probe_boundary_span.location = boundary
        self._probe_cursor_span.location = self._predict_time_slider.value
        self._probe_data_box.right = boundary
        self._predict_probe_figure.title.text = (f"original vs. prediction at the "
                                                 f"probe point x = "
                                                 f"{self._probe_slider.value:.2f}")
        # fixed y-range over the whole horizon (honest scaling under playback);
        # the x-window is static inside the data window and slides with the
        # cursor once the prediction begins (strip chart, cursor at the right
        # edge) — identical in both prediction views
        vmax = max(float(np.abs(original[np.isfinite(original)]).max()), 1e-12)
        vmax = max(vmax, float(np.abs(model).max()),
                   float(np.abs(truth[np.isfinite(truth)]).max()))
        self._predict_probe_figure.y_range.start = -1.05 * vmax
        self._predict_probe_figure.y_range.end = 1.05 * vmax
        window_start, window_end = self._prediction_window()
        self._predict_probe_figure.x_range.start = window_start
        self._predict_probe_figure.x_range.end = window_end
        self._position_prediction_labels(self._probe_labels, window_start,
                                         window_end,
                                         self._predict_probe_figure.y_range.end)

    def _update_prediction_error_view(self) -> None:
        """
        Update the prediction-error curve and the readout of the panel.

        The error is the root mean square of the model deviation over space
        against the reference: the original data inside the window
        (reconstruction error) and the clean truth beyond the boundary
        (prediction error). The readout reports the current time, the stage,
        and the local and window-maximum error.

        :return: None
        :rtype: None
        """
        result = self._result
        signal = self._signal
        if result is None or result.n_modes == 0 or signal is None \
                or self._reconstruction is None or self._prediction is None \
                or self._truth is None:
            self._predict_error_source.data = dict(t=[], error=[])
            self._prediction_error = None
            return
        steps = self._prediction_steps()
        error_window = np.sqrt(np.mean((self._reconstruction - signal.samples) ** 2,
                                       axis=0))
        error_future = np.sqrt(np.mean((self._prediction - self._truth) ** 2, axis=0))
        error = np.concatenate([error_window, error_future])
        self._prediction_error = error
        horizon_time = signal.dt * np.arange(signal.n_time + steps)
        self._predict_error_source.data = dict(t=horizon_time, error=error)
        boundary = (signal.n_time - 1) * signal.dt
        self._error_boundary_span.location = boundary
        self._error_cursor_span.location = self._predict_time_slider.value
        self._error_data_box.right = boundary
        # fixed y-range owned by the app (honest scaling of exploding errors);
        # the x-window follows the same static-then-sliding rule as the
        # forecast figure so both time axes stay aligned
        vmax = max(float(error.max()), 1e-12)
        self._predict_error_figure.y_range.start = 0.0
        self._predict_error_figure.y_range.end = 1.05 * vmax
        window_start, window_end = self._prediction_window()
        self._predict_error_figure.x_range.start = window_start
        self._predict_error_figure.x_range.end = window_end
        self._position_prediction_labels(self._error_labels, window_start,
                                         window_end,
                                         self._predict_error_figure.y_range.end)

    def _update_prediction_readout(self) -> None:
        """
        Update the prediction readout for the current cursor position.

        The readout reports the stage (data window or prediction), the distance
        beyond the data window, the local RMS error, and the maximum error over
        the prediction window.

        :return: None
        :rtype: None
        """
        result = self._result
        signal = self._signal
        if result is None or result.n_modes == 0 or signal is None \
                or self._prediction is None or self._prediction_error is None:
            self._predict_div.text = "no modes fitted"
            return
        steps = self._prediction_steps()
        t_index = min(round(self._predict_time_slider.value / signal.dt),
                      signal.n_time - 1 + steps)
        stage = ("prediction" if t_index >= signal.n_time else "data window")
        beyond = max(t_index - (signal.n_time - 1), 0)
        beyond_text = (f" · {beyond * signal.dt:.2f} s beyond the data window"
                       if beyond > 0 else "")
        self._predict_div.text = (
            f"<b>t = {self._predict_time_slider.value:.2f} s</b> · {stage}"
            f"{beyond_text}<br>"
            f"RMS error at t: {self._prediction_error[t_index]:.3e} · "
            f"max over the prediction window: "
            f"{float(self._prediction_error[signal.n_time:].max()):.3e}"
        )

    def _update_prediction_views(self) -> None:
        """
        Update both prediction views and the readout.

        Called whenever the model quantities change (refit, sort, eigenvalue or
        amplitude edits, weights, morph, reset, horizon change) — not per
        animation tick of the other panels, whose shared cursor does not touch
        the prediction views.

        :return: None
        :rtype: None
        """
        self._update_prediction_probe_view()
        self._update_prediction_error_view()
        self._update_prediction_readout()

    def _on_n_pred_change(self, attr: str, old: object, new: object) -> None:
        """
        React to a prediction-horizon change by recomputing the prediction.

        The horizon only resizes the prediction window: no refit is needed,
        but the truth, the fields, the slider bounds, and all prediction views
        follow. Typed values are clamped against the declared bounds.

        :param attr: name of the changed widget attribute
        :type attr: str
        :param old: previous attribute value
        :type old: object
        :param new: new attribute value
        :type new: object
        :return: None
        :rtype: None
        """
        steps = min(max(int(new), 1), 5000)
        if steps != int(new):
            self._n_pred_spinner.value = steps
            return
        if self._result is None or self._signal is None:
            return
        self._recompute_truth()
        self._recompute_fields()
        self._update_predict_slider()
        self._update_prediction_views()

    def _on_probe_change(self, attr: str, old: object, new: object) -> None:
        """
        React to a probe-point change by refreshing the probe time series.

        The probe snaps to the spatial grid; typed values outside the grid are
        written back clamped.

        :param attr: name of the changed widget attribute
        :type attr: str
        :param old: previous attribute value
        :type old: object
        :param new: new attribute value
        :type new: object
        :return: None
        :rtype: None
        """
        if self._syncing_probe or self._signal is None:
            return
        signal = self._signal
        index = min(max(round(float(new) / signal.dx), 0), signal.n_space - 1)
        snapped = index * signal.dx
        if abs(snapped - float(new)) > 1e-12:
            self._syncing_probe = True
            self._probe_slider.value = snapped
            self._syncing_probe = False
        self._update_prediction_probe_view()

    def _on_predict_time_change(self, attr: str, old: object, new: object) -> None:
        """
        React to the prediction time cursor by refreshing the dependent views.

        The cursor is independent of the shared time sliders of the other
        panels (it reaches beyond the data window); the visible time window,
        the cursor spans, and the readout follow it — the plot stays static
        inside the data window and the axis slides once the prediction begins.

        :param attr: name of the changed widget attribute
        :type attr: str
        :param old: previous attribute value
        :type old: object
        :param new: new attribute value
        :type new: object
        :return: None
        :rtype: None
        """
        self._update_prediction_probe_view()
        self._update_prediction_error_view()
        self._update_prediction_readout()

    def _on_predict_play(self, attr: str, old: bool, new: bool) -> None:
        """
        React to the prediction play toggle by starting or stopping its animation.

        The prediction playback is independent of the shared playback of the
        other panels: its cursor wraps over data window plus prediction window.

        :param attr: name of the changed widget attribute
        :type attr: str
        :param old: previous toggle state
        :type old: bool
        :param new: new toggle state
        :type new: bool
        :return: None
        :rtype: None
        """
        self._predict_play_toggle.label = "⏸ pause" if new else "▶ play"
        if new and self._predict_periodic_callback is None:
            self._predict_periodic_callback = self._doc.add_periodic_callback(
                self._advance_predict_time, _PLAY_INTERVAL_MS)
        elif not new and self._predict_periodic_callback is not None:
            self._doc.remove_periodic_callback(self._predict_periodic_callback)
            self._predict_periodic_callback = None

    def _advance_predict_time(self) -> None:
        """
        Advance the prediction time cursor by one step, wrapping at the horizon.

        :return: None
        :rtype: None
        """
        signal = self._signal
        if signal is None:
            return
        dt = signal.dt
        total = signal.n_time + self._prediction_steps()
        index = (round(self._predict_time_slider.value / dt) + 1) % total
        self._predict_time_slider.value = index * dt

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
