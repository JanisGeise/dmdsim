"""
    Implements the figure builders of the DMD playground application.
"""
import numpy as np

from bokeh.palettes import RdBu
from bokeh.plotting import figure
from bokeh.models import (Span, Arrow, Label, Range1d, ColorBar, HoverTool,
                          NormalHead, GlyphRenderer, PointDrawTool,
                          ColumnDataSource, LinearColorMapper)

from dmdsim.app.styling import halfplane_colors


def create_eigenvalue_figure() -> tuple[figure, ColumnDataSource, GlyphRenderer,
                                        GlyphRenderer, Label, Label]:
    """
    Create the eigenvalue plane with the unit circle and an empty mode scatter.

    The upper and lower half planes are shaded to emphasize the sign of
    ``Im λ`` (equivalently the sign of the frequency); the shading colors
    follow the theme and are updated by the application on theme switches.
    A small triangle above each eigenvalue marker points in the travel
    direction of the mode's wave (the application fills the ``dir_*`` columns
    from the signed phase velocity; NaN coordinates hide the triangle for
    non-propagating modes).

    :return: figure, the data source for the eigenvalue scatter; the source
        expects the columns ``re``, ``im``, ``index``, ``frequency``,
        ``growth``, ``color``, ``size``, and the direction-marker columns
        ``dir_x``, ``dir_y``, ``dir_angle``, ``dir_color``, and ``dir_size``;
        the two half-plane shading renderers (upper first, lower second; the
        application updates their glyph fill color on theme switches), and the
        two half-plane labels (upper first, lower second; the application
        updates their text color on theme switches)
    :rtype: tuple[figure, ColumnDataSource, GlyphRenderer, GlyphRenderer, Label,
        Label]
    """
    fig = figure(width=620, height=560, title="eigenvalue plane", match_aspect=True,
                 x_range=(-1.6, 1.6), y_range=(-1.6, 1.6))
    # shaded half planes below all other glyphs mark the frequency sign; the
    # rectangles extend far beyond the axis ranges so any zoom level is covered
    upper_color, lower_color = halfplane_colors()
    upper_renderer = fig.patch(x=[-100.0, 100.0, 100.0, -100.0],
                               y=[0.0, 0.0, 100.0, 100.0],
                               fill_color=upper_color, fill_alpha=0.08,
                               line_alpha=0.0)
    lower_renderer = fig.patch(x=[-100.0, 100.0, 100.0, -100.0],
                               y=[0.0, 0.0, -100.0, -100.0],
                               fill_color=lower_color, fill_alpha=0.08,
                               line_alpha=0.0)
    angles = np.linspace(0.0, 2.0 * np.pi, 200)
    fig.line(np.cos(angles), np.sin(angles), line_color="gray", line_dash="dashed",
             line_width=1)
    source = ColumnDataSource(dict(re=[], im=[], index=[], frequency=[], growth=[],
                                   color=[], size=[], dir_x=[], dir_y=[],
                                   dir_angle=[], dir_color=[], dir_size=[]))
    eigen_renderer = fig.scatter(x="re", y="im", source=source, size="size",
                                 color="color", line_color="white")
    # small travel-direction triangles above the eigenvalue markers; the
    # application sets the angle (right = +x, left = -x) and hides
    # non-propagating modes via NaN coordinates
    fig.scatter(x="dir_x", y="dir_y", source=source, marker="triangle",
                angle="dir_angle", size="dir_size", color="dir_color",
                line_color="white")
    # eigenvalues are edited by dragging only: adding points is disabled, and the
    # application restores the source if a point is deleted via backspace
    fig.add_tools(PointDrawTool(renderers=[eigen_renderer], add=False, drag=True,
                                empty_value=0.0))
    # the hover is scoped to the scatter: the half-plane patches would otherwise
    # capture the hover everywhere with empty tooltip fields
    fig.add_tools(HoverTool(renderers=[eigen_renderer], tooltips=[
        ("mode", "@index"),
        ("lambda", "(@re{0.000}, @im{0.000})"),
        ("f", "@frequency{0.000} Hz"),
        ("growth rate σ", "@growth{0.000} 1/s"),
    ]))
    upper_label = Label(x=0.0, y=1.13, text="Im λ > 0", text_align="center",
                        text_font_size="11px", text_font_style="bold",
                        text_color=upper_color)
    lower_label = Label(x=0.0, y=-1.13, text="Im λ < 0", text_align="center",
                        text_font_size="11px", text_font_style="bold",
                        text_color=lower_color)
    fig.add_layout(upper_label)
    fig.add_layout(lower_label)
    fig.xaxis.axis_label = "Re(λ)"
    fig.yaxis.axis_label = "Im(λ)"
    return fig, source, upper_renderer, lower_renderer, upper_label, lower_label


def create_mode_dynamics_figure() -> tuple[figure, ColumnDataSource, ColumnDataSource,
                                           ColumnDataSource, GlyphRenderer]:
    """
    Create the complex-plane animation of the selected mode's eigenvalue powers.

    Shows the trajectory :math:`\\lambda^t` for :math:`t = 1 \\ldots t_{\\mathrm{now}}`
    of the selected mode: the trail spirals inward for decaying modes, outward for
    growing modes and orbits on a circle for neutral modes. A faded scatter marks
    all fitted eigenvalues for context. The trail color is set by the application
    to the stability color of the selected mode.

    :return: figure, the trail data source (columns ``x``, ``y``, and ``color``),
        the current point data source (columns ``x``, ``y``, and ``color``), the
        all-eigenvalues data source (columns ``x``, ``y``, ``index``, ``color``,
        and ``size``), and the trail line renderer (the application sets its
        glyph color)
    :rtype: tuple[figure, ColumnDataSource, ColumnDataSource, ColumnDataSource,
        GlyphRenderer]
    """
    fig = figure(width=620, height=560, title="mode dynamics: λᵗ in the complex plane",
                 match_aspect=True, x_range=(-1.6, 1.6), y_range=(-1.6, 1.6))
    angles = np.linspace(0.0, 2.0 * np.pi, 200)
    fig.line(np.cos(angles), np.sin(angles), line_color="gray", line_dash="dashed",
             line_width=1)
    all_source = ColumnDataSource(dict(x=[], y=[], index=[], color=[], size=[]))
    fig.scatter(x="x", y="y", source=all_source, size="size", color="color", alpha=0.7)
    trail_source = ColumnDataSource(dict(x=[], y=[], color=[]))
    trail_renderer = fig.line(x="x", y="y", source=trail_source, color="#9aa0a6",
                              line_width=2)
    fig.scatter(x="x", y="y", source=trail_source, size=4, color="color", alpha=0.6)
    point_source = ColumnDataSource(dict(x=[], y=[], color=[]))
    point_renderer = fig.scatter(x="x", y="y", source=point_source, size=12,
                                 color="color", line_color="white")
    fig.add_tools(HoverTool(tooltips=[("λᵗ", "(@x{0.000}, @y{0.000})")],
                            renderers=[point_renderer]))
    fig.xaxis.axis_label = "Re(λᵗ)"
    fig.yaxis.axis_label = "Im(λᵗ)"
    return fig, trail_source, point_source, all_source, trail_renderer


def create_spectrum_figure() -> tuple[figure, ColumnDataSource, list[Span]]:
    """
    Create the stem-style DMD spectrum of the mode amplitudes over frequency.

    Stems and markers are colored by stability (red growing, blue decaying, gray
    neutral) and the selected eigenvalue is emphasized; dashed vertical spans
    mark the Nyquist limits.

    :return: figure, the data source (columns ``frequency``, ``amplitude``,
        ``index``, ``growth``, ``color``, ``width``, and ``size``), and the two
        Nyquist spans (their locations are updated by the application)
    :rtype: tuple[figure, ColumnDataSource, list[Span]]
    """
    fig = figure(width=620, height=560, title="DMD spectrum: |b| over frequency",
                 x_range=(-1.0, 1.0))
    source = ColumnDataSource(dict(frequency=[], amplitude=[], index=[], growth=[],
                                   color=[], width=[], size=[]))
    fig.segment(x0="frequency", y0=0.0, x1="frequency", y1="amplitude", source=source,
                line_color="color", line_width="width")
    fig.scatter(x="frequency", y="amplitude", source=source, size="size",
                color="color", line_color="white")
    fig.add_tools(HoverTool(tooltips=[
        ("mode", "@index"),
        ("f", "@frequency{0.000} Hz"),
        ("|b|", "@amplitude{0.000}"),
        ("growth rate σ", "@growth{0.000} 1/s"),
    ]))
    spans = [Span(dimension="height", location=-1.0, line_color="gray",
                  line_dash="dashed", line_width=1),
             Span(dimension="height", location=1.0, line_color="gray",
                  line_dash="dashed", line_width=1)]
    for span in spans:
        fig.add_layout(span)
    fig.xaxis.axis_label = "f [Hz]"
    fig.yaxis.axis_label = "amplitude |b|"
    return fig, source, spans


def create_growth_direction_figure() -> tuple[figure, ColumnDataSource]:
    """
    Create the growth-direction plane of the fitted modes.

    Shows each mode as a marker over its growth rate
    :math:`\\sigma = \\Re(\\mu)` with
    :math:`\\mu = \\ln(\\lambda)/\\Delta t` (horizontal) and its phase velocity
    :math:`c = -2\\pi f / \\kappa` (vertical, :math:`\\kappa` = phase slope of
    the mode shape): the sign of ``c`` is the propagation direction, so modes
    above the zero line travel toward :math:`+x` and below toward :math:`-x`;
    conjugate partners form one physical wave and coincide. Markers carry
    per-point pair colors and the selection emphasis; dashed spans mark zero
    growth and zero propagation. The axis ranges are set by the application.

    :return: figure and the data source; the source expects the columns
        ``sigma``, ``velocity``, ``index``, ``color``, ``size``, ``frequency``,
        and ``direction``
    :rtype: tuple[figure, ColumnDataSource]
    """
    fig = figure(width=620, height=560,
                 title="growth rate σ over phase velocity c")
    source = ColumnDataSource(dict(sigma=[], velocity=[], index=[], color=[], size=[],
                                   frequency=[], direction=[]))
    scatter_renderer = fig.scatter(x="sigma", y="velocity", source=source,
                                   size="size", color="color", line_color="white")
    fig.add_tools(HoverTool(renderers=[scatter_renderer], tooltips=[
        ("mode", "@index"),
        ("σ", "@sigma{0.000} 1/s"),
        ("f", "@frequency{0.000} Hz"),
        ("c", "@velocity{0.000} units/s"),
        ("travels", "@direction"),
    ]))
    # dashed zero lines: vertical = neutral growth (σ = 0), horizontal = no
    # propagation (c = 0: standing wave or spatially constant shape)
    for dimension in ("height", "width"):
        fig.add_layout(Span(dimension=dimension, location=0.0, line_color="gray",
                            line_dash="dashed", line_width=1))
    fig.xaxis.axis_label = "σ [1/s]"
    fig.yaxis.axis_label = "c = −2πf/κ [units/s]"
    return fig, source


_MODE_FIGURE_SPECS = {
    "real": ("Re(φ) over space", "Re(φ)", "#1f77b4"),
    "imag": ("Im(φ) over space", "Im(φ)", "#ff7f0e"),
    "phase": ("arg(φ) over space", "arg(φ) [rad]", "#d62728"),
}


def create_mode_component_figure(quantity: str, x_range: Range1d | None = None,
                                 y_range: Range1d | None = None,
                                 height: int = 420,
                                 width: int = 1000) -> tuple[figure, ColumnDataSource]:
    """
    Create a single-quantity mode shape figure.

    The phase view renders the unwrapped phase as a continuous line and overlays
    the wrapped values as faded points to explain the 2π jumps. The imaginary
    and phase views draw a faded reference line of the fitted shape (column
    ``reference``) so edits like the quadrature morph are visible against the
    original.

    :param quantity: quantity to display; one of ``real``, ``imag``, or ``phase``
    :type quantity: str
    :param x_range: shared x-range for grouping several quantity figures with
        synced axes, defaults to None (auto range)
    :type x_range: Range1d | None, optional
    :param y_range: shared y-range (e.g. one constant ruler for the real and
        imaginary quadratures), defaults to None (auto range)
    :type y_range: Range1d | None, optional
    :param height: figure height in pixels, defaults to 420
    :type height: int
    :param width: figure width in pixels, defaults to 1000
    :type width: int
    :return: figure and the data source; the source expects the columns ``s``,
        ``value``, ``unwrapped``, and ``reference``
    :rtype: tuple[figure, ColumnDataSource]
    """
    assert quantity in _MODE_FIGURE_SPECS, (
        f"Unknown mode quantity {quantity!r}, must be one of: {sorted(_MODE_FIGURE_SPECS)}")
    title, y_label, color = _MODE_FIGURE_SPECS[quantity]
    options: dict = dict(width=width, height=height, title=title)
    if x_range is not None:
        # a shared range object syncs the axes of grouped quantity figures
        options["x_range"] = x_range
    if y_range is not None:
        options["y_range"] = y_range
    if quantity == "phase":
        fig = figure(**options, y_range=Range1d(-np.pi - 0.4, np.pi + 0.4))
    else:
        fig = figure(**options)
    source = ColumnDataSource(dict(s=[], value=[], unwrapped=[], reference=[]))
    if quantity == "phase":
        # the wrapped values explain where the jumps of the sawtooth come from
        fig.scatter(x="s", y="value", source=source, size=5, color=color, alpha=0.35,
                    legend_label="wrapped (±π)")
        fig.line(x="s", y="unwrapped", source=source, color=color, line_width=2,
                 legend_label="unwrapped")
        fig.line(x="s", y="reference", source=source, color=color, line_width=1.5,
                 line_dash="dotted", line_alpha=0.35,
                 legend_label="fitted (t = 1)")
        fig.legend.location = "top_left"
        fig.legend.click_policy = "hide"
    else:
        fig.line(x="s", y="value", source=source, color=color, line_width=2)
        if quantity == "imag":
            # faded fitted shape: the morph slider scales the solid curve
            # against this reference
            fig.line(x="s", y="reference", source=source, color=color,
                     line_width=1.5, line_dash="dotted", line_alpha=0.35)
    fig.xaxis.axis_label = "x"
    fig.yaxis.axis_label = y_label
    return fig, source


def create_quadrature_figure(x_range: Range1d | None = None,
                             y_range: Range1d | None = None,
                             height: int = 420,
                             width: int = 1000) -> tuple[figure, ColumnDataSource]:
    """
    Create the combined quadrature view of the selected mode shape.

    Draws the real and imaginary parts of :math:`\\phi` in one axes so the
    quarter-wavelength offset between the quadratures (their spatial phase
    relation) is directly visible; a faded dotted line shows the fitted
    imaginary part (column ``reference``) so the morph slider's scaling is
    visible against the original.

    :param x_range: shared x-range for grouping the shape figures with synced
        axes, defaults to None (auto range)
    :type x_range: Range1d | None, optional
    :param y_range: shared constant y-ruler of the quadrature view, defaults
        to None (auto range)
    :type y_range: Range1d | None, optional
    :param height: figure height in pixels, defaults to 420
    :type height: int
    :param width: figure width in pixels, defaults to 1000
    :type width: int
    :return: figure and the data source; the source expects the columns ``s``,
        ``real``, ``imag``, and ``reference``
    :rtype: tuple[figure, ColumnDataSource]
    """
    options: dict = dict(width=width, height=height,
                         title="Re(φ) and Im(φ) of the selected mode")
    if x_range is not None:
        options["x_range"] = x_range
    if y_range is not None:
        options["y_range"] = y_range
    fig = figure(**options)
    source = ColumnDataSource(dict(s=[], real=[], imag=[], reference=[]))
    fig.line(x="s", y="real", source=source, color="#1f77b4", line_width=2,
             legend_label="Re(φ)")
    fig.line(x="s", y="imag", source=source, color="#ff7f0e", line_width=2,
             legend_label="Im(φ)")
    fig.line(x="s", y="reference", source=source, color="#ff7f0e", line_width=1.5,
             line_dash="dotted", line_alpha=0.35,
             legend_label="fitted Im(φ) (t = 1)")
    # three-column legend, centered outside the frame above the plot
    fig.legend.ncols = 3
    fig.legend.orientation = "horizontal"
    fig.legend.location = "center"
    fig.add_layout(fig.legend[0], "above")
    fig.xaxis.axis_label = "x"
    fig.yaxis.axis_label = "φ quadratures"
    return fig, source


def _diverging_palette(steps: int = 256) -> list[str]:
    """
    Build a smooth diverging blue-white-red palette by interpolating the
    bokeh ``RdBu`` palette.

    :param steps: number of palette entries
    :type steps: int
    :return: ``steps`` hex colors from dark blue over white to dark red
    :rtype: list[str]
    """
    base = list(RdBu[11])
    anchors = np.array([[int(color[i:i + 2], 16) for i in (1, 3, 5)]
                        for color in base], dtype=np.float64)
    positions = np.linspace(0.0, 1.0, len(base))
    targets = np.linspace(0.0, 1.0, steps)
    channels = [np.interp(targets, positions, anchors[:, channel])
                for channel in range(3)]
    return [f"#{int(red):02x}{int(green):02x}{int(blue):02x}"
            for red, green, blue in zip(*[np.round(channel).astype(int)
                                          for channel in channels])]


_SPACE_TIME_PALETTE = _diverging_palette()


def create_space_time_figure(x_range: Range1d | None = None, height: int = 420,
                             width: int = 615) -> tuple[figure, ColumnDataSource,
                                                        LinearColorMapper]:
    """
    Create the space-time heatmap of the selected mode's contribution.

    Draws :math:`\\Re\\{\\phi b \\lambda^t\\}` over space (x, horizontal) and
    time (t, vertical, increasing upward) with a diverging color map and a
    color bar. The horizontal stripe spacing is the wavelength
    :math:`2\\pi/\\kappa` (a property of the shape), the vertical spacing the
    temporal period :math:`1/f` (a property of the eigenvalue), and the stripe
    slope the propagation speed — the decoupling of the two phases is visible
    as two independent spacings.

    .. note::
        The image glyph renders array row 0 at the top of the frame, so the
        application stores the time-flipped field to keep time increasing
        upward.

    :param x_range: shared x-range for grouping the shape figures with synced
        axes, defaults to None (auto range)
    :type x_range: Range1d | None, optional
    :param height: figure height in pixels, defaults to 420
    :type height: int
    :param width: figure width in pixels, defaults to 615
    :type width: int
    :return: figure, the data source (columns ``field`` and ``dh``; the app
        stores the time-flipped field and the vertical extent), and the color
        mapper (the app updates its symmetric limits)
    :rtype: tuple[figure, ColumnDataSource, LinearColorMapper]
    """
    options: dict = dict(width=width, height=height,
                         title="space–time map of the selected mode")
    if x_range is not None:
        options["x_range"] = x_range
    fig = figure(**options, y_range=Range1d(0.0, 3.0))
    mapper = LinearColorMapper(palette=_SPACE_TIME_PALETTE, low=-1.0, high=1.0)
    source = ColumnDataSource(dict(field=[], dh=[]))
    fig.image(image="field", x=0.0, y=0.0, dw=1.0, dh="dh", color_mapper=mapper,
              source=source)
    fig.add_layout(ColorBar(color_mapper=mapper), "right")
    fig.xaxis.axis_label = "x"
    fig.yaxis.axis_label = "t [s]"
    return fig, source, mapper


def create_direction_figure(x_range: Range1d | None = None, height: int = 420,
                            width: int = 1000) -> tuple[figure, ColumnDataSource, Arrow]:
    """
    Create the traveling-wave view of the selected mode with a direction arrow.

    The wave contribution :math:`\\Re\\{\\phi b \\lambda^t\\}` is animated with
    the global time cursor; the arrow shows the propagation direction and is
    positioned by the application. A faded reference line shows the
    contribution of the fitted shape (column ``reference``), so quadrature
    edits are visible against the original.

    :param x_range: shared x-range for grouping the shape figures with synced
        axes, defaults to None (auto range)
    :type x_range: Range1d | None, optional
    :param height: figure height in pixels, defaults to 420
    :type height: int
    :param width: figure width in pixels, defaults to 1000
    :type width: int
    :return: figure, the data source (columns ``s``, ``value``, and
        ``reference``), and the direction arrow annotation
    :rtype: tuple[figure, ColumnDataSource, Arrow]
    """
    options: dict = dict(width=width, height=height,
                         title="traveling wave of the selected mode")
    if x_range is not None:
        options["x_range"] = x_range
    # a plain Range1d: the application owns the envelope y-range (a DataRange1d
    # would recompute it client-side and ignore the application's writes)
    fig = figure(**options, y_range=Range1d(-1.0, 1.0))
    source = ColumnDataSource(dict(s=[], value=[], reference=[]))
    fig.line(x="s", y="reference", source=source, color="#8430ce", line_width=1.5,
             line_dash="dotted", line_alpha=0.35)
    fig.line(x="s", y="value", source=source, color="#8430ce", line_width=2)
    arrow = Arrow(x_start=0.35, y_start=0.0, x_end=0.65, y_end=0.0,
                  line_color="#8430ce", line_width=3,
                  end=NormalHead(size=16, fill_color="#8430ce", line_color="#8430ce"),
                  start=None)
    fig.add_layout(arrow)
    fig.xaxis.axis_label = "x"
    fig.yaxis.axis_label = "contribution"
    return fig, source, arrow


def create_signal_curve_figure() -> tuple[figure, ColumnDataSource]:
    """
    Create the 1D line view of the signal at the current animation time.

    The spatial axis is the fixed unit window :math:`x \\in [0, 1]`; the axis
    labels use LaTeX rendering (MathText in canvas text).

    :return: figure and the data source; the source expects the columns ``s``
        and ``value``
    :rtype: tuple[figure, ColumnDataSource]
    """
    # plain Range1d: the application owns the fixed y-range (a DataRange1d
    # would recompute it client-side and ignore the application's writes)
    fig = figure(width=500, height=380, title="signal at the current time",
                 y_range=Range1d(-1.0, 1.0))
    source = ColumnDataSource(dict(s=[], value=[]))
    line_renderer = fig.line(x="s", y="value", source=source, color="#1a73e8",
                             line_width=2)
    fig.xaxis.axis_label = "$$x$$"
    fig.yaxis.axis_label = "$$y(x)$$"
    return fig, source, line_renderer


def create_time_curves_figure() -> tuple[figure, ColumnDataSource, ColumnDataSource,
                                         GlyphRenderer]:
    """
    Create the time view showing the field along space at the current time.

    Shows the original field, the reconstruction, the sum over all checked
    eigenvalues, the selected eigenvalue's contribution (its color is set by the
    application to the pair color), and thin per-eigenvalue contribution lines
    in their pair colors for all checked eigenvalues.

    :return: figure, the main data source (columns ``s``, ``original``,
        ``reconstruction``, ``selected``, and ``selection``), the per-eigenvalue
        multi-line source (columns ``xs``, ``ys``, ``colors``, ``index``, and
        ``f``), and the renderer of the selected-eigenvalue line (the
        application sets its glyph color)
    :rtype: tuple[figure, ColumnDataSource, ColumnDataSource, GlyphRenderer]
    """
    # plain Range1d: the application owns the fixed y-range (a DataRange1d
    # would recompute it client-side and ignore the application's writes)
    fig = figure(width=1150, height=420, title="field at current time",
                 y_range=Range1d(-1.0, 1.0))
    source = ColumnDataSource(dict(s=[], original=[], reconstruction=[], selected=[],
                                   selection=[]))
    original_renderer = fig.line(x="s", y="original", source=source,
                                 legend_label="original", color="black", line_width=2)
    reconstruction_renderer = fig.line(x="s", y="reconstruction", source=source,
                                       legend_label="reconstruction", color="#5f6368",
                                       line_width=2, line_dash="dashed")
    selection_renderer = fig.line(x="s", y="selection", source=source,
                                  legend_label="sum of selected modes",
                                  color="#9aa0a6", line_width=2, line_dash="dashdot")
    selected_renderer = fig.line(x="s", y="selected", source=source,
                                 legend_label="selected eigenvalue", color="#9aa0a6",
                                 line_width=3)
    curve_renderers = {"original": original_renderer,
                       "reconstruction": reconstruction_renderer,
                       "selection": selection_renderer}
    pair_source = ColumnDataSource(dict(xs=[], ys=[], colors=[], index=[], f=[]))
    pair_renderer = fig.multi_line(xs="xs", ys="ys", source=pair_source,
                                   line_color="colors", line_width=1.5)
    fig.add_tools(HoverTool(renderers=[pair_renderer], line_policy="nearest",
                            tooltips=[("eigenvalue no.", "@index"),
                                      ("f", "@f{0.00} Hz")]))
    # four-column legend, centered outside the frame above the plot
    fig.legend.ncols = 4
    fig.legend.orientation = "horizontal"
    fig.legend.location = "center"
    fig.add_layout(fig.legend[0], "above")
    fig.legend.click_policy = "hide"
    fig.xaxis.axis_label = "x"
    fig.yaxis.axis_label = "y"
    return fig, source, pair_source, selected_renderer, curve_renderers


if __name__ == "__main__":
    pass
