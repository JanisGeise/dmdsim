"""
    Implements the DMD playground Bokeh application.
"""
import logging

import numpy as np

from bokeh.document import Document
from bokeh.layouts import column, row
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.server.callbacks import PeriodicCallback
from bokeh.application.handlers import FunctionHandler
from bokeh.models import Button, CheckboxGroup, Div, Slider, Spinner, Toggle
from tornado.ioloop import PeriodicCallback as TornadoPeriodicCallback

from dmdsim.core.engine import fit_dmd
from dmdsim.core.quantities import DMDResult
from dmdsim.signals.generator import Mode, ModalSignal
from dmdsim.core.synth import mirror_conjugates, synthesize

from dmdsim.app.figures import (create_eigenvalue_figure, create_field_heatmap,
                                create_mode_figure, create_spectrum_figure,
                                create_time_curves_figure)

# keep the flowTorch info messages out of the application output
logging.getLogger("flowtorch").setLevel(logging.WARNING)

_MAX_MODE_EDITORS = 3
_NOISE_SEED = 42
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
                     "f = arg(λ) / (2π·dt).")
_HINT_SPECTRUM = ("Mode strength |b| over frequency; conjugate pairs appear as mirror "
                  "peaks at ±f. The axis is bounded by the Nyquist limit ±1/(2·dt).")
_HINT_MODE = ("Complex spatial shape φ(s): Re and Im form the wave profile, |φ| its "
              "magnitude, and arg(φ) the local phase shift. A linear arg(φ) corresponds "
              "to a wavenumber.")
_HINT_TIME = ("The cursor in the heatmap marks the current time of the curves on the "
              "right. The reconstruction is Σ φᵢ·bᵢ·λᵢᵗ; compare it against the "
              "original field.")


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
        # per-mode editors: active toggle plus wavenumber, frequency, amplitude, growth
        self._mode_toggles = []
        self._wavenumber_spinners = []
        self._frequency_spinners = []
        self._amplitude_spinners = []
        self._growth_spinners = []
        for i, (wavenumber, frequency, amplitude, growth, active) in enumerate(_MODE_DEFAULTS):
            self._mode_toggles.append(Toggle(label=f"mode {i}", active=active, width=90))
            self._wavenumber_spinners.append(
                Spinner(title="κ", value=wavenumber, low=0.0, high=40.0, step=0.1, width=70))
            self._frequency_spinners.append(
                Spinner(title="f [Hz]", value=frequency, low=0.0, high=30.0, step=0.1, width=70))
            self._amplitude_spinners.append(
                Spinner(title="A", value=amplitude, low=0.0, high=3.0, step=0.01, width=70))
            self._growth_spinners.append(
                Spinner(title="σ [1/s]", value=growth, low=-2.0, high=2.0, step=0.01, width=70))
        # grid, offset, and noise
        self._dt_spinner = Spinner(title="dt [s]", value=0.01, low=0.0001, high=1.0,
                                   step=0.001, width=80)
        self._dx_spinner = Spinner(title="dx", value=0.1, low=0.001, high=10.0,
                                   step=0.01, width=80)
        self._n_space_spinner = Spinner(title="n space", value=64, low=8, high=2048,
                                        step=1, width=80)
        self._n_time_spinner = Spinner(title="n time", value=300, low=8, high=5000,
                                       step=1, width=80)
        self._offset_spinner = Spinner(title="offset", value=0.0, low=-5.0, high=5.0,
                                       step=0.1, width=80)
        self._noise_toggle = Toggle(label="add noise", active=False, width=100)
        self._noise_spinner = Spinner(title="noise σ", value=0.05, low=0.0, high=2.0,
                                      step=0.01, width=80)
        # DMD controls
        self._rank_spinner = Spinner(title="rank (0 = automatic)", value=5, low=0,
                                     high=64, step=1, width=140)
        self._auto_toggle = Toggle(label="auto-rerun on change", active=True, width=160)
        self._run_button = Button(label="Run DMD", button_type="primary", width=100)
        self._status_div = Div(text="", width=460)
        # interactive elements of the quantity views
        self._mode_slider = Slider(title="mode index", start=0, end=0, value=0, step=1,
                                   width=300)
        self._time_slider = Slider(title="time index", start=0, end=1, value=0, step=1,
                                   width=300)
        self._play_toggle = Toggle(label="▶ play", active=False, width=100)
        self._mode_select = CheckboxGroup(labels=[], active=[], inline=True)
        self._readout_div = Div(text="", width=460)

    def _build_figures(self) -> None:
        """
        Create all figures and their data sources.

        :return: None
        :rtype: None
        """
        self._eigenvalue_figure, self._eigenvalue_source = create_eigenvalue_figure()
        self._spectrum_figure, self._spectrum_source = create_spectrum_figure()
        self._mode_figure, self._mode_source = create_mode_figure()
        (self._heatmap_figure, self._heatmap_source, self._color_mapper,
         self._time_span) = create_field_heatmap()
        self._time_figure, self._time_source = create_time_curves_figure()

    def _build_layout(self) -> None:
        """
        Arrange the widgets and figures into the page layout.

        :return: None
        :rtype: None
        """
        # one compact editor row per mode
        mode_rows = [row(self._mode_toggles[i], self._wavenumber_spinners[i],
                         self._frequency_spinners[i], self._amplitude_spinners[i],
                         self._growth_spinners[i]) for i in range(_MAX_MODE_EDITORS)]
        signal_panel = column(
            Div(text="<b>Signal generator</b>"),
            *mode_rows,
            Div(text="<b>Grid, offset, and noise</b>"),
            row(self._dt_spinner, self._dx_spinner, self._n_space_spinner,
                self._n_time_spinner),
            row(self._offset_spinner, self._noise_toggle, self._noise_spinner),
            Div(text="<b>DMD</b>"),
            row(self._rank_spinner, self._auto_toggle, self._run_button),
            self._status_div,
            width=460,
        )
        plots = column(
            row(self._eigenvalue_figure, self._spectrum_figure),
            row(Div(text=_HINT_EIGENVALUES, width=460), Div(text=_HINT_SPECTRUM, width=460)),
            self._mode_slider,
            self._mode_figure,
            Div(text=_HINT_MODE, width=940),
            row(self._time_slider, self._play_toggle),
            self._mode_select,
            row(self._heatmap_figure, self._time_figure),
            row(Div(text=_HINT_TIME, width=460), self._readout_div),
        )
        self._doc.add_root(row(signal_panel, plots))

    def _attach_callbacks(self) -> None:
        """
        Connect the widget callbacks for recomputation and view updates.

        :return: None
        :rtype: None
        """
        # signal and DMD parameters trigger a recomputation or mark the views stale
        parameter_widgets = (self._mode_toggles + self._wavenumber_spinners
                             + self._frequency_spinners + self._amplitude_spinners
                             + self._growth_spinners
                             + [self._dt_spinner, self._dx_spinner, self._n_space_spinner,
                                self._n_time_spinner, self._offset_spinner,
                                self._noise_toggle, self._noise_spinner,
                                self._rank_spinner])
        for widget in parameter_widgets:
            attribute = "active" if isinstance(widget, Toggle) else "value"
            widget.on_change(attribute, self._on_parameter_change)
        self._run_button.on_click(self._run_dmd)
        self._auto_toggle.on_change("active", self._on_auto_toggle)
        # view interactions update the curves without a refit
        self._mode_slider.on_change("value", self._on_view_change)
        self._time_slider.on_change("value", self._on_view_change)
        self._mode_select.on_change("active", self._on_view_change)
        self._play_toggle.on_change("active", self._on_play_toggle)

    def _build_signal(self) -> ModalSignal:
        """
        Create the modal signal from the current widget state.

        :return: configured synthetic line field
        :rtype: ModalSignal
        """
        modes = []
        for i in range(_MAX_MODE_EDITORS):
            if not self._mode_toggles[i].active:
                continue
            modes.append(Mode(wavenumber=self._wavenumber_spinners[i].value,
                              frequency=self._frequency_spinners[i].value,
                              amplitude=self._amplitude_spinners[i].value,
                              growth_rate=self._growth_spinners[i].value))
        return ModalSignal(modes, dt=self._dt_spinner.value,
                           n_space=self._n_space_spinner.value,
                           n_time=self._n_time_spinner.value, dx=self._dx_spinner.value,
                           offset=self._offset_spinner.value,
                           noise_std=self._noise_spinner.value if self._noise_toggle.active else 0.0,
                           seed=_NOISE_SEED)

    def _run_dmd(self) -> None:
        """
        Rebuild the signal from the widget state, refit the DMD, and refresh all views.

        :return: None
        :rtype: None
        """
        signal = self._build_signal()
        rank = self._rank_spinner.value
        result = fit_dmd(signal.samples, dt=signal.dt, rank=rank if rank > 0 else None)
        self._signal = signal
        self._result = result
        self._reconstruction = synthesize(result.modes, result.amplitudes, result.eigvals,
                                          signal.dt, signal.n_time)
        # keep the slider ends consistent before any view reads them
        self._update_time_slider()
        self._update_mode_slider()
        self._update_field_view()
        self._update_quantities()
        self._update_mode_view()
        self._update_mode_select()
        error = float(np.abs(self._reconstruction - signal.samples).max())
        self._status_div.text = (f"fitted {result.n_modes} modes · round-trip error "
                                 f"{error:.1e} · dt = {signal.dt} s")

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
        Update the space-time heatmap, its color range, and the time slider bounds.

        :return: None
        :rtype: None
        """
        signal = self._signal
        self._heatmap_source.data = dict(image=[signal.samples], x=[0.0], y=[0.0],
                                         dw=[(signal.n_time - 1) * signal.dt],
                                         dh=[(signal.n_space - 1) * signal.dx])
        self._heatmap_figure.x_range.end = (signal.n_time - 1) * signal.dt
        self._heatmap_figure.y_range.end = (signal.n_space - 1) * signal.dx
        vmax = max(float(np.abs(signal.samples).max()), 1e-12)
        self._color_mapper.low = -vmax
        self._color_mapper.high = vmax

    def _update_quantities(self) -> None:
        """
        Update the eigenvalue plane, the spectrum, and their axis ranges.

        :return: None
        :rtype: None
        """
        result = self._result
        if result is None or result.n_modes == 0:
            self._eigenvalue_source.data = dict(re=[], im=[], index=[], frequency=[],
                                                growth=[])
            self._spectrum_source.data = dict(frequency=[], amplitude=[], index=[],
                                              growth=[])
            return
        self._eigenvalue_source.data = dict(re=result.eigvals.real, im=result.eigvals.imag,
                                            index=list(range(result.n_modes)),
                                            frequency=result.frequencies,
                                            growth=result.growth_rates)
        self._spectrum_source.data = dict(frequency=result.frequencies,
                                          amplitude=np.abs(result.amplitudes),
                                          index=list(range(result.n_modes)),
                                          growth=result.growth_rates)
        # keep the eigenvalue plane centered on the unit circle with margin
        radius = max(1.2, float(np.abs(result.eigvals).max()) * 1.1)
        self._eigenvalue_figure.x_range.start = -radius
        self._eigenvalue_figure.x_range.end = radius
        self._eigenvalue_figure.y_range.start = -radius
        self._eigenvalue_figure.y_range.end = radius
        # the spectrum is bounded by the Nyquist frequency of the current time step
        nyquist = 0.5 / self._signal.dt
        self._spectrum_figure.x_range.start = -nyquist
        self._spectrum_figure.x_range.end = nyquist

    def _update_mode_slider(self) -> None:
        """
        Adjust the mode index slider bounds to the fitted number of modes.

        :return: None
        :rtype: None
        """
        result = self._result
        end = max(result.n_modes - 1, 0) if result is not None else 0
        self._mode_slider.end = end
        if self._mode_slider.value > end:
            self._mode_slider.value = end

    def _update_time_slider(self) -> None:
        """
        Adjust the time index slider bounds to the current signal length.

        :return: None
        :rtype: None
        """
        signal = self._signal
        end = max(signal.n_time - 1, 0) if signal is not None else 0
        self._time_slider.end = end
        if self._time_slider.value > end:
            self._time_slider.value = end

    def _update_mode_view(self) -> None:
        """
        Update the mode shape viewer for the currently selected mode.

        :return: None
        :rtype: None
        """
        result = self._result
        if result is None or result.n_modes == 0:
            self._mode_source.data = dict(s=[], re=[], im=[], abs=[], phase=[])
            self._mode_figure.title.text = "mode shape"
            return
        index = min(int(self._mode_slider.value), result.n_modes - 1)
        shape = result.mode(index)
        self._mode_source.data = dict(s=self._signal.space, re=np.real(shape),
                                      im=np.imag(shape), abs=np.abs(shape),
                                      phase=np.angle(shape))
        self._mode_figure.title.text = (f"mode {index}: f = "
                                        f"{result.frequencies[index]:.2f} Hz, σ = "
                                        f"{result.growth_rates[index]:.3f} 1/s, |b| = "
                                        f"{np.abs(result.amplitudes[index]):.3f}")

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
        Update the heatmap cursor and the field curves at the current time.

        :return: None
        :rtype: None
        """
        signal = self._signal
        result = self._result
        if signal is None:
            return
        t_index = min(int(self._time_slider.value), signal.n_time - 1)
        self._time_span.location = signal.time[t_index]
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
        self._update_readout(index)

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

    def _on_parameter_change(self, attr: str, old: object, new: object) -> None:
        """
        React to a signal or DMD parameter change by rerunning or marking stale.

        :param attr: name of the changed widget attribute
        :type attr: str
        :param old: previous attribute value
        :type old: object
        :param new: new attribute value
        :type new: object
        :return: None
        :rtype: None
        """
        if self._auto_toggle.active:
            self._run_dmd()
        else:
            self._status_div.text = ("parameters changed — press <b>Run DMD</b> to "
                                     "update the quantities")

    def _on_auto_toggle(self, attr: str, old: bool, new: bool) -> None:
        """
        React to the auto-rerun toggle by refreshing immediately when activated.

        :param attr: name of the changed widget attribute
        :type attr: str
        :param old: previous toggle state
        :type old: bool
        :param new: new toggle state
        :type new: bool
        :return: None
        :rtype: None
        """
        if new:
            self._run_dmd()

    def _on_view_change(self, attr: str, old: object, new: object) -> None:
        """
        React to a mode, time, or selection change by updating the views without a refit.

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

    def _on_play_toggle(self, attr: str, old: bool, new: bool) -> None:
        """
        Start or stop the time animation of the time view.

        :param attr: name of the changed widget attribute
        :type attr: str
        :param old: previous toggle state
        :type old: bool
        :param new: new toggle state
        :type new: bool
        :return: None
        :rtype: None
        """
        self._play_toggle.label = "⏸ pause" if new else "▶ play"
        if new and self._periodic_callback is None:
            self._periodic_callback = self._doc.add_periodic_callback(
                self._advance_time, _PLAY_INTERVAL_MS)
        elif not new and self._periodic_callback is not None:
            self._doc.remove_periodic_callback(self._periodic_callback)
            self._periodic_callback = None

    def _advance_time(self) -> None:
        """
        Advance the time slider by one step for the animation.

        :return: None
        :rtype: None
        """
        if self._signal is None:
            return
        self._time_slider.value = (int(self._time_slider.value) + 1) % self._signal.n_time


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
