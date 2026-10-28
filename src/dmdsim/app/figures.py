"""
    Implements the figure builders of the DMD playground application.
"""
import numpy as np

from bokeh.plotting import figure
from bokeh.models import (Span, Arrow, Range1d, HoverTool, NormalHead, GlyphRenderer,
                          PointDrawTool, ColumnDataSource)


def create_eigenvalue_figure() -> tuple[figure, ColumnDataSource]:
    """
    Create the eigenvalue plane with the unit circle and an empty mode scatter.

    :return: figure and the data source for the eigenvalue scatter; the source
        expects the columns ``re``, ``im``, ``index``, ``frequency``, and ``growth``
    :rtype: tuple[figure, ColumnDataSource]
    """
    fig = figure(width=620, height=560, title="eigenvalue plane", match_aspect=True,
                 x_range=(-1.6, 1.6), y_range=(-1.6, 1.6))
    angles = np.linspace(0.0, 2.0 * np.pi, 200)
    fig.line(np.cos(angles), np.sin(angles), line_color="gray", line_dash="dashed",
             line_width=1)
    source = ColumnDataSource(dict(re=[], im=[], index=[], frequency=[], growth=[],
                                   color=[], size=[]))
    eigen_renderer = fig.scatter(x="re", y="im", source=source, size="size",
                                 color="color", line_color="white")
    # eigenvalues are edited by dragging only: adding points is disabled, and the
    # application restores the source if a point is deleted via backspace
    fig.add_tools(PointDrawTool(renderers=[eigen_renderer], add=False, drag=True,
                                empty_value=0.0))
    fig.add_tools(HoverTool(tooltips=[
        ("mode", "@index"),
        ("lambda", "(@re{0.000}, @im{0.000})"),
        ("f", "@frequency{0.000} Hz"),
        ("growth rate σ", "@growth{0.000} 1/s"),
    ]))
    fig.xaxis.axis_label = "Re(λ)"
    fig.yaxis.axis_label = "Im(λ)"
    return fig, source


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
    fig.yaxis.axis_label = "|b|"
    return fig, source, spans


_MODE_FIGURE_SPECS = {
    "real": ("Re(φ) over space", "Re(φ)", "#1f77b4"),
    "imag": ("Im(φ) over space", "Im(φ)", "#ff7f0e"),
    "phase": ("arg(φ) over space", "arg(φ) [rad]", "#d62728"),
}


def create_mode_component_figure(quantity: str) -> tuple[figure, ColumnDataSource]:
    """
    Create a single-quantity mode shape figure.

    The phase view renders the unwrapped phase as a continuous line and overlays
    the wrapped values as faded points to explain the 2π jumps.

    :param quantity: quantity to display; one of ``real``, ``imag``, or ``phase``
    :type quantity: str
    :return: figure and the data source; the source expects the columns ``s``,
        ``value``, and ``unwrapped``
    :rtype: tuple[figure, ColumnDataSource]
    """
    assert quantity in _MODE_FIGURE_SPECS, (
        f"Unknown mode quantity {quantity!r}, must be one of: {sorted(_MODE_FIGURE_SPECS)}")
    title, y_label, color = _MODE_FIGURE_SPECS[quantity]
    if quantity == "phase":
        fig = figure(width=1000, height=420, title=title,
                     y_range=Range1d(-np.pi - 0.4, np.pi + 0.4))
    else:
        fig = figure(width=1000, height=420, title=title)
    source = ColumnDataSource(dict(s=[], value=[], unwrapped=[]))
    if quantity == "phase":
        # the wrapped values explain where the jumps of the sawtooth come from
        fig.scatter(x="s", y="value", source=source, size=5, color=color, alpha=0.35,
                    legend_label="wrapped (±π)")
        fig.line(x="s", y="unwrapped", source=source, color=color, line_width=2,
                 legend_label="unwrapped")
        fig.legend.location = "top_left"
        fig.legend.click_policy = "hide"
    else:
        fig.line(x="s", y="value", source=source, color=color, line_width=2)
    fig.xaxis.axis_label = "x"
    fig.yaxis.axis_label = y_label
    return fig, source


def create_direction_figure() -> tuple[figure, ColumnDataSource, Arrow]:
    """
    Create the traveling-wave view of the selected mode with a direction arrow.

    The wave contribution :math:`\\Re\\{\\phi b \\lambda^t\\}` is animated with
    the global time cursor; the arrow shows the propagation direction and is
    positioned by the application.

    :return: figure, the data source (columns ``s`` and ``value``), and the
        direction arrow annotation
    :rtype: tuple[figure, ColumnDataSource, Arrow]
    """
    fig = figure(width=1000, height=420, title="traveling wave of the selected mode")
    source = ColumnDataSource(dict(s=[], value=[]))
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
    fig = figure(width=500, height=380, title="signal at the current time")
    source = ColumnDataSource(dict(s=[], value=[]))
    fig.line(x="s", y="value", source=source, color="#1a73e8", line_width=2)
    fig.xaxis.axis_label = "$$x$$"
    fig.yaxis.axis_label = "$$y(x)$$"
    return fig, source


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
    fig = figure(width=1000, height=420, title="field at current time")
    source = ColumnDataSource(dict(s=[], original=[], reconstruction=[], selected=[],
                                   selection=[]))
    fig.line(x="s", y="original", source=source, legend_label="original", color="black",
             line_width=2)
    fig.line(x="s", y="reconstruction", source=source, legend_label="reconstruction",
             color="#5f6368", line_width=2, line_dash="dashed")
    fig.line(x="s", y="selection", source=source, legend_label="Σ selected",
             color="#9aa0a6", line_width=2, line_dash="dashdot")
    selected_renderer = fig.line(x="s", y="selected", source=source,
                                 legend_label="selected eigenvalue", color="#9aa0a6",
                                 line_width=3)
    pair_source = ColumnDataSource(dict(xs=[], ys=[], colors=[], index=[], f=[]))
    pair_renderer = fig.multi_line(xs="xs", ys="ys", source=pair_source,
                                   line_color="colors", line_width=1.5)
    fig.add_tools(HoverTool(renderers=[pair_renderer], line_policy="nearest",
                            tooltips=[("eigenvalue no.", "@index"),
                                      ("f", "@f{0.00} Hz")]))
    fig.legend.location = "top_left"
    fig.legend.click_policy = "hide"
    fig.xaxis.axis_label = "x"
    fig.yaxis.axis_label = "y"
    return fig, source, pair_source, selected_renderer


if __name__ == "__main__":
    pass
