"""
    Implements the figure builders of the DMD playground application.
"""
import numpy as np

from bokeh.palettes import RdBu
from bokeh.plotting import figure
from bokeh.models import ColorBar, ColumnDataSource, HoverTool, LinearAxis, LinearColorMapper, Range1d, Span


def create_eigenvalue_figure() -> tuple[figure, ColumnDataSource]:
    """
    Create the eigenvalue plane with the unit circle and an empty mode scatter.

    :return: figure and the data source for the eigenvalue scatter; the source
        expects the columns ``re``, ``im``, ``index``, ``frequency``, and ``growth``
    :rtype: tuple[figure, ColumnDataSource]
    """
    fig = figure(width=460, height=340, title="eigenvalue plane", match_aspect=True,
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
    fig = figure(width=460, height=340, title="DMD spectrum: |b| over frequency",
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


def create_mode_figure() -> tuple[figure, ColumnDataSource]:
    """
    Create the mode shape viewer with real, imaginary, magnitude, and phase curves.

    :return: figure and the data source; the source expects the columns ``s``,
        ``re``, ``im``, ``abs``, and ``phase``
    :rtype: tuple[figure, ColumnDataSource]
    """
    fig = figure(width=940, height=240, title="mode shape",
                 extra_y_ranges={"phase": Range1d(-np.pi - 0.2, np.pi + 0.2)})
    source = ColumnDataSource(dict(s=[], re=[], im=[], abs=[], phase=[]))
    fig.line(x="s", y="re", source=source, legend_label="Re(φ)", color="#1f77b4",
             line_width=2)
    fig.line(x="s", y="im", source=source, legend_label="Im(φ)", color="#ff7f0e",
             line_width=2)
    fig.line(x="s", y="abs", source=source, legend_label="|φ|", color="#2ca02c",
             line_width=2, line_dash="dashed")
    fig.line(x="s", y="phase", source=source, legend_label="arg(φ)", color="#d62728",
             line_width=2, y_range_name="phase")
    fig.add_layout(LinearAxis(y_range_name="phase", axis_label="arg(φ) [rad]"), "right")
    fig.legend.location = "top_left"
    fig.legend.click_policy = "hide"
    fig.xaxis.axis_label = "s"
    fig.yaxis.axis_label = "φ"
    return fig, source


def create_field_heatmap() -> tuple[figure, ColumnDataSource, LinearColorMapper, Span]:
    """
    Create the space-time heatmap of the field with the current-time cursor.

    :return: figure, the image data source (columns ``image``, ``x``, ``y``, ``dw``,
        and ``dh``), the shared color mapper, and the time cursor span
    :rtype: tuple[figure, ColumnDataSource, LinearColorMapper, Span]
    """
    mapper = LinearColorMapper(palette=RdBu[11], low=-1.0, high=1.0)
    fig = figure(width=460, height=280, title="field x(s, t)", x_range=(0.0, 1.0),
                 y_range=(0.0, 1.0))
    source = ColumnDataSource(dict(image=[], x=[], y=[], dw=[], dh=[]))
    fig.image(image="image", x="x", y="y", dw="dw", dh="dh", source=source,
              color_mapper=mapper)
    fig.add_layout(ColorBar(color_mapper=mapper), "right")
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
    fig = figure(width=460, height=280, title="field at current time")
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
