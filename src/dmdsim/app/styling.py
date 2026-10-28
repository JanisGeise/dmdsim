"""
    Implements the visual styling of the DMD playground application.
"""
from bokeh.models import Div, InlineStyleSheet
from bokeh.plotting import figure

# per-step accent colors: panel tint background and saturated accent color
_STEP_COLORS: dict[str, tuple[str, str]] = {
    "setup": ("#f0f5ff", "#1a73e8"),
    "eigenvalues": ("#eef7f0", "#188038"),
    "spectrum": ("#fffaef", "#b06000"),
    "modes": ("#f7f0fc", "#8430ce"),
    "reconstruction": ("#fdf1ef", "#c5221f"),
}

_TEXT_MUTED = "#5f6368"
_GRID = "#dadce0"

_TAB_CSS = """
    :host { background: #ffffff; }
    .bk-headers { padding: 8px 4px; display: flex; flex-direction: @DIRECTION;
                  align-items: @ALIGN; }
    .bk-tab { background: #f1f3f4; color: #5f6368; border: none;
              border-radius: 6px; margin: 4px 5px; padding: 9px 16px;
              font-weight: 500; font-size: 13px; text-align: left; }
    .bk-tab:hover { background: #e8eaed; color: #202124; }
    .bk-active { background: @ACCENT; color: #ffffff; }
"""


def step_colors(step: str) -> tuple[str, str]:
    """
    Return the tint and the accent color of a playground step.

    :param step: step key; one of ``setup``, ``eigenvalues``, ``spectrum``,
        ``modes``, or ``reconstruction``
    :type step: str
    :return: panel tint background and saturated accent color
    :rtype: tuple[str, str]
    """
    return _STEP_COLORS[step]


def style_figure(fig: figure, step: str) -> None:
    """
    Apply the step colors and a modern neutral style to a figure.

    The plot area stays white for data readability while the figure border area
    carries the panel tint; title, axes, grid, and legend are toned down to a
    muted gray with the accent color reserved for the title.

    :param fig: figure to style
    :type fig: figure
    :param step: step key selecting the panel colors
    :type step: str
    :return: None
    :rtype: None
    """
    tint, accent = step_colors(step)
    fig.background_fill_color = "#ffffff"
    fig.border_fill_color = tint
    fig.outline_line_color = _GRID
    fig.title.text_color = accent
    fig.title.text_font_style = "bold"
    for axis in (fig.xaxis, fig.yaxis):
        axis.axis_label_text_color = _TEXT_MUTED
        axis.major_label_text_color = _TEXT_MUTED
    for grid in (fig.xgrid, fig.ygrid):
        grid.grid_line_color = _GRID
        grid.grid_line_alpha = 0.7
    # only figures with glyph legends carry legend styling
    if len(fig.legend) > 0:
        fig.legend.background_fill_color = "#ffffff"
        fig.legend.border_line_color = _GRID
        fig.legend.label_text_color = _TEXT_MUTED


def header(text: str, step: str) -> Div:
    """
    Create a colored section header of a panel.

    :param text: header text (HTML, escape special characters)
    :type text: str
    :param step: step key selecting the panel colors
    :type step: str
    :return: styled header element
    :rtype: Div
    """
    tint, accent = step_colors(step)
    return Div(text=text, styles={
        "background": tint,
        "color": accent,
        "border-radius": "6px",
        "padding": "6px 10px",
        "font-weight": "600",
        "font-size": "13px",
    })


def hint(text: str, width: int | None = None) -> Div:
    """
    Create a muted inline explanation text block.

    :param text: explanation text (HTML, escape special characters)
    :type text: str
    :param width: fixed width of the block in pixels; unconstrained if ``None``,
        defaults to None
    :type width: int | None, optional
    :return: styled hint element
    :rtype: Div
    """
    div = Div(text=text, styles={
        "color": _TEXT_MUTED,
        "font-size": "12px",
        "line-height": "1.4",
        "padding": "2px 4px",
    })
    if width is not None:
        div.width = width
    return div


def info_box(text: str = "") -> Div:
    """
    Create a neutral info box for dynamic status or readout text.

    :param text: initial content of the box (HTML), defaults to ""
    :type text: str, optional
    :return: styled info box element
    :rtype: Div
    """
    return Div(text=text, styles={
        "background": "#f8f9fa",
        "border": "1px solid #dadce0",
        "border-radius": "6px",
        "padding": "6px 10px",
        "color": "#202124",
        "font-size": "12px",
    })


def tab_stylesheet(accent: str = "#1a73e8", vertical: bool = True) -> InlineStyleSheet:
    """
    Create the stylesheet for a tab strip styling.

    Vertical strips (the main sidebar) get uniformly wide, left-aligned buttons
    stacked from the top; horizontal strips (nested submenus) keep compact
    buttons. The active tab carries the given accent color.

    :param accent: background color of the active tab, defaults to the setup blue
    :type accent: str
    :param vertical: style a vertical (left location) tab strip instead of a
        horizontal one, defaults to True
    :type vertical: bool, optional
    :return: inline stylesheet for a ``Tabs`` widget
    :rtype: InlineStyleSheet
    """
    # placeholders instead of format() because the CSS is full of literal braces
    replacements = {
        "@DIRECTION": "column" if vertical else "row",
        "@ALIGN": "stretch" if vertical else "flex-start",
        "@ACCENT": accent,
    }
    css = _TAB_CSS
    for key, value in replacements.items():
        css = css.replace(key, value)
    return InlineStyleSheet(css=css)


if __name__ == "__main__":
    pass
