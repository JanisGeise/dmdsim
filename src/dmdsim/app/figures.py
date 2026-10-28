"""
    Implements the figure builders of the DMD playground application.
"""
import numpy as np

from bokeh.palettes import RdBu
from bokeh.plotting import figure
from bokeh.models import ColorBar, ColumnDataSource, HoverTool, LinearColorMapper, Range1d, Span


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
    source = ColumnDataSource(dict(re=[], im=[], index=[], frequency=[], growth=[]))
    fig.scatter(x="re", y="im", source=source, size=9, color="#1f77b4", line_color="white")
    fig.add_tools(HoverTool(tooltips=[
        ("mode", "@index"),
        ("lambda", "(@re{0.000}, @im{0.000})"),
        ("f", "@frequency{0.000} Hz"),
        ("sigma", "@growth{0.000} 1/s"),
    ]))
    fig.xaxis.axis_label = "Re(λ)"
    fig.yaxis.axis_label = "Im(λ)"
    return fig, source


def create_spectrum_figure() -> tuple[figure, ColumnDataSource]:
    """
    Create the stem-style DMD spectrum of the mode amplitudes over frequency.

    :return: figure and the data source; the source expects the columns
        ``frequency``, ``amplitude``, ``index``, and ``growth``
    :rtype: tuple[figure, ColumnDataSource]
    """
    fig = figure(width=620, height=560, title="DMD spectrum: |b| over frequency",
                 x_range=(-1.0, 1.0))
    source = ColumnDataSource(dict(frequency=[], amplitude=[], index=[], growth=[]))
    fig.segment(x0="frequency", y0=0.0, x1="frequency", y1="amplitude", source=source,
                line_color="#1f77b4", line_width=2)
    fig.scatter(x="frequency", y="amplitude", source=source, size=9, color="#1f77b4",
                line_color="white")
    fig.add_tools(HoverTool(tooltips=[
        ("mode", "@index"),
        ("f", "@frequency{0.000} Hz"),
        ("|b|", "@amplitude{0.000}"),
        ("sigma", "@growth{0.000} 1/s"),
    ]))
    fig.xaxis.axis_label = "f [Hz]"
    fig.yaxis.axis_label = "|b|"
    return fig, source


_MODE_FIGURE_SPECS = {
    "real": ("Re(φ) over space", "Re(φ)", "#1f77b4"),
    "imag": ("Im(φ) over space", "Im(φ)", "#ff7f0e"),
    "phase": ("arg(φ) over space", "arg(φ) [rad]", "#d62728"),
    "direction": ("arg(φ) — propagation direction", "arg(φ) [rad]", "#d62728"),
}


def create_mode_component_figure(quantity: str) -> tuple[figure, ColumnDataSource]:
    """
    Create a single-quantity mode shape figure.

    :param quantity: quantity to display; one of ``real``, ``imag``, ``phase``,
        or ``direction`` (the direction view shows the phase with a focus on the
        propagation analysis)
    :type quantity: str
    :return: figure and the data source; the source expects the columns ``s``
        and ``value``
    :rtype: tuple[figure, ColumnDataSource]
    """
    assert quantity in _MODE_FIGURE_SPECS, (
        f"Unknown mode quantity {quantity!r}, must be one of: {sorted(_MODE_FIGURE_SPECS)}")
    title, y_label, color = _MODE_FIGURE_SPECS[quantity]
    if quantity in ("phase", "direction"):
        fig = figure(width=1000, height=420, title=title,
                     y_range=Range1d(-np.pi - 0.2, np.pi + 0.2))
    else:
        fig = figure(width=1000, height=420, title=title)
    source = ColumnDataSource(dict(s=[], value=[]))
    fig.line(x="s", y="value", source=source, color=color, line_width=2)
    fig.xaxis.axis_label = "s"
    fig.yaxis.axis_label = y_label
    return fig, source


def create_field_heatmap(show_cursor: bool = True,
                         title: str = "field x(s, t)") -> tuple[figure, ColumnDataSource,
                                                                LinearColorMapper, Span | None]:
    """
    Create the space-time heatmap of the field with an optional current-time cursor.

    :param show_cursor: draw the dashed time cursor span into the figure; disable
        for static previews of the signal, defaults to True
    :type show_cursor: bool
    :param title: figure title, defaults to ``"field x(s, t)"``
    :type title: str
    :return: figure, the image data source (columns ``image``, ``x``, ``y``, ``dw``,
        and ``dh``), the shared color mapper, and the time cursor span (``None``
        without cursor)
    :rtype: tuple[figure, ColumnDataSource, LinearColorMapper, Span | None]
    """
    mapper = LinearColorMapper(palette=RdBu[11], low=-1.0, high=1.0)
    fig = figure(width=500, height=380, title=title, x_range=(0.0, 1.0),
                 y_range=(0.0, 1.0))
    source = ColumnDataSource(dict(image=[], x=[], y=[], dw=[], dh=[]))
    fig.image(image="image", x="x", y="y", dw="dw", dh="dh", source=source,
              color_mapper=mapper)
    fig.add_layout(ColorBar(color_mapper=mapper), "right")
    span = None
    if show_cursor:
        span = Span(dimension="height", location=0.0, line_color="black", line_width=2,
                    line_dash="dashed")
        fig.add_layout(span)
    fig.xaxis.axis_label = "t [s]"
    fig.yaxis.axis_label = "s"
    return fig, source, mapper, span


def create_time_curves_figure() -> tuple[figure, ColumnDataSource]:
    """
    Create the time view showing the field along space at the current time.

    :return: figure and the data source; the source expects the columns ``s``,
        ``original``, ``reconstruction``, ``selected``, and ``selection``
    :rtype: tuple[figure, ColumnDataSource]
    """
    fig = figure(width=500, height=380, title="field at current time")
    source = ColumnDataSource(dict(s=[], original=[], reconstruction=[], selected=[],
                                   selection=[]))
    fig.line(x="s", y="original", source=source, legend_label="original", color="black",
             line_width=2)
    fig.line(x="s", y="reconstruction", source=source, legend_label="reconstruction",
             color="#1f77b4", line_width=2, line_dash="dashed")
    fig.line(x="s", y="selection", source=source, legend_label="Σ selected modes",
             color="#2ca02c", line_width=2)
    fig.line(x="s", y="selected", source=source, legend_label="selected mode",
             color="#d62728", line_width=1, line_dash="dotted")
    fig.legend.location = "top_left"
    fig.legend.click_policy = "hide"
    fig.xaxis.axis_label = "s"
    fig.yaxis.axis_label = "x"
    return fig, source


if __name__ == "__main__":
    pass
