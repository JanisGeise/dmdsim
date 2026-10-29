"""
    Implements the visual styling of the DMD playground application.

    All colors are centralized here; the application supports a light and a dark
    theme (the dark theme follows the lecture website's dark variant). Figures
    are styled via `style_figure`, layout elements via `header`, `hint`, and
    `info_box`, and the tab strip and page via the returned stylesheets.
"""
from bokeh.models import Div
from bokeh.plotting import figure

_LIGHT = {
    "page_bg": "#ffffff", "text": "#202124", "muted": "#5f6368", "grid": "#dadce0",
    "plot_bg": "#ffffff", "box_bg": "#f8f9fa", "box_border": "#dadce0",
    "tab_bg": "#f1f3f4", "tab_text": "#5f6368", "tab_hover_bg": "#e8eaed",
    "tab_hover_text": "#202124",
    "line_original": "#202124", "line_reconstruction": "#5f6368",
    "line_selection": "#9aa0a6", "line_signal": "#1a73e8",
    "line_prediction": "#00897b", "line_error": "#c5221f",
    "tints": {"setup": "#f0f5ff", "eigenvalues": "#eef7f0", "spectrum": "#fffaef",
              "modes": "#f7f0fc", "reconstruction": "#fdf1ef",
              "prediction": "#e8f5f3"},
    "accents": {"setup": "#1a73e8", "eigenvalues": "#188038", "spectrum": "#b06000",
                "modes": "#8430ce", "reconstruction": "#c5221f",
                "prediction": "#00897b"},
    "pair_palette": ["#1a73e8", "#c5221f", "#188038", "#8430ce", "#b06000",
                     "#0b8043", "#a142f4", "#f4511e", "#039be5", "#c2185b"],
}

_DARK = {
    "page_bg": "#131313", "text": "#e8eaed", "muted": "#9aa0a6", "grid": "#3c4043",
    "plot_bg": "#1e1e1e", "box_bg": "#1e1e1e", "box_border": "#3c4043",
    "tab_bg": "#1e1e1e", "tab_text": "#9aa0a6", "tab_hover_bg": "#3c4043",
    "tab_hover_text": "#e8eaed",
    "line_original": "#e8eaed", "line_reconstruction": "#9aa0a6",
    "line_selection": "#5f6368", "line_signal": "#8ab4f8",
    "line_prediction": "#35d0ba", "line_error": "#f28b82",
    "tints": {"setup": "#1b2436", "eigenvalues": "#1b2a22", "spectrum": "#2b2618",
              "modes": "#271d33", "reconstruction": "#322020",
              "prediction": "#122b27"},
    "accents": {"setup": "#8ab4f8", "eigenvalues": "#81c995", "spectrum": "#fdd663",
                "modes": "#c58af9", "reconstruction": "#f28b82",
                "prediction": "#35d0ba"},
    "pair_palette": ["#8ab4f8", "#f28b82", "#81c995", "#c58af9", "#fdd663",
                     "#78d9ec", "#fcad03", "#de9e36", "#9ad0c2", "#e5a4c7"],
}

THEMES = {"light": _LIGHT, "dark": _DARK}

# categorical colors for conjugate eigenvalue pairs (one color per physical mode)
PAIR_PALETTE = _LIGHT["pair_palette"]
PAIR_PALETTE_DARK = _DARK["pair_palette"]

# half-plane shading of the eigenvalue plane: an eigenvalue above the real axis
# (Im(λ) > 0) has a positive frequency, below it a negative one; the labels
# "f > 0" / "f < 0" carry the same colors (subtle fills in both themes)
HALF_PLANE_COLORS = {"light": ("#f9ab00", "#1a73e8"),
                     "dark": ("#fdd663", "#8ab4f8")}

_TAB_CSS = """
    :host { background: @PAGE; }
    .bk-headers { padding: 8px 4px; display: flex; flex-direction: @DIRECTION;
                  align-items: @ALIGN; }
    .bk-tab { background: @TABBG; color: @TABTEXT; border: none;
              border-radius: 6px; margin: 4px 5px; padding: 9px 16px;
              font-weight: 500; font-size: 13px; text-align: left; }
    .bk-tab:hover { background: @HOVERBG; color: @HOVERTEXT; }
    .bk-active { background: @ACCENT; color: @PAGETEXT; }
"""

_PAGE_CSS = """
    body { background: @PAGE; color: @TEXT; }
    .bk-input, input.bk-input { background-color: @BOX !important; color: @TEXT
                               !important; border-color: @GRID !important; }
    .bk-btn { background-color: @BOX !important; color: @TEXT !important;
              border-color: @GRID !important; }
    .bk-title, .bk-slider-title { color: @MUTED !important; }
    .bk-description { color: @MUTED !important; }
    .bk-checkbox, input[type='checkbox'].bk-input { accent-color: @ACCENT
                                                    !important; }
    .bk-checkbox-group .bk-checkbox-label, .bk-radio-group .bk-radio-label {
        color: @TEXT !important; }
    .bk-menu { background-color: @BOX !important; color: @TEXT !important;
              border-color: @GRID !important; }
    .bk-menu > *:hover { background-color: @HOVER !important; color: @TEXT
                         !important; }
    .bk-tooltip > .bk-tooltip-content { background: @BOX !important; color: @TEXT
                                       !important; border: 1px solid @GRID
                                       !important; }
    .bk-toolbar-button { background-color: @BOX !important; color: @MUTED
                         !important; border-color: @GRID !important; }
    .bk-widget-box, .bk-widget { color: @TEXT !important; }
"""


def theme_colors(mode: str = "light") -> dict:
    """
    Return the color dictionary of a theme.

    :param mode: theme mode; ``"light"`` or ``"dark"``
    :type mode: str
    :return: mapping of color roles to hex colors (page_bg, text, muted, grid,
        plot_bg, box_bg, box_border, tab colors, line colors, per-step tints
        and accents)
    :rtype: dict
    """
    assert mode in THEMES, f"Unknown theme {mode!r}, must be one of: {sorted(THEMES)}"
    return THEMES[mode]


def step_colors(step: str, mode: str = "light") -> tuple[str, str]:
    """
    Return the tint and the accent color of a playground step.

    :param step: step key; one of ``setup``, ``eigenvalues``, ``spectrum``,
        ``modes``, or ``reconstruction``
    :type step: str
    :param mode: theme mode, defaults to ``"light"``
    :type mode: str, optional
    :return: panel tint background and saturated accent color
    :rtype: tuple[str, str]
    """
    theme = THEMES[mode]
    return theme["tints"][step], theme["accents"][step]


def halfplane_colors(mode: str = "light") -> tuple[str, str]:
    """
    Return the half-plane shading colors of the eigenvalue plane.

    The shading emphasizes the sign of the frequency: the upper half plane
    (``Im(λ) > 0``) belongs to positive frequencies and the lower half plane to
    negative ones.

    :param mode: theme mode, defaults to ``"light"``
    :type mode: str, optional
    :return: upper and lower half-plane colors as ``(upper, lower)``
    :rtype: tuple[str, str]
    """
    assert mode in HALF_PLANE_COLORS, (f"Unknown theme {mode!r}, must be one of: "
                                       f"{sorted(HALF_PLANE_COLORS)}")
    return HALF_PLANE_COLORS[mode]


def style_figure(fig, step: str, mode: str = "light") -> None:
    """
    Apply the step colors and the theme style to a figure.

    The plot area carries the theme plot background while the figure border
    area carries the panel tint; title, axes, grid, and legend follow the
    theme text and grid colors with the accent color reserved for the title.

    :param fig: figure to style
    :type fig: figure
    :param step: step key selecting the panel colors
    :type step: str
    :param mode: theme mode, defaults to ``"light"``
    :type mode: str, optional
    :return: None
    :rtype: None
    """
    theme = THEMES[mode]
    tint, accent = theme["tints"][step], theme["accents"][step]
    fig.background_fill_color = theme["plot_bg"]
    fig.border_fill_color = tint
    fig.outline_line_color = theme["grid"]
    fig.title.text_color = accent
    fig.title.text_font_style = "bold"
    for axis in (fig.xaxis, fig.yaxis):
        axis.axis_label_text_color = theme["muted"]
        axis.major_label_text_color = theme["muted"]
    for grid in (fig.xgrid, fig.ygrid):
        grid.grid_line_color = theme["grid"]
        grid.grid_line_alpha = 0.7
    if len(fig.legend) > 0:
        fig.legend.background_fill_color = theme["plot_bg"]
        fig.legend.border_line_color = theme["grid"]
        fig.legend.label_text_color = theme["text"]


def header(text: str, step: str, mode: str = "light") -> Div:
    """
    Create a colored section header of a panel.

    :param text: header text (HTML, escape special characters)
    :type text: str
    :param step: step key selecting the panel colors
    :type step: str
    :param mode: theme mode, defaults to ``"light"``
    :type mode: str, optional
    :return: styled header element
    :rtype: Div
    """
    return Div(text=text, styles=header_styles(step, mode))


def header_styles(step: str, mode: str = "light") -> dict:
    """
    Return the style dictionary of a section header.

    :param step: step key selecting the panel colors
    :type step: str
    :param mode: theme mode, defaults to ``"light"``
    :type mode: str, optional
    :return: inline styles for a header element
    :rtype: dict
    """
    tint, accent = step_colors(step, mode)
    return {"background": tint, "color": accent, "border-radius": "6px",
            "padding": "6px 10px", "font-weight": "600", "font-size": "13px"}


def hint(text: str, width: int | None = None, mode: str = "light") -> Div:
    """
    Create a muted inline explanation text block.

    :param text: explanation text (HTML, escape special characters)
    :type text: str
    :param width: fixed width of the block in pixels; unconstrained if ``None``,
        defaults to None
    :type width: int | None, optional
    :param mode: theme mode, defaults to ``"light"``
    :type mode: str, optional
    :return: styled hint element
    :rtype: Div
    """
    div = Div(text=text, styles=hint_styles(mode))
    if width is not None:
        div.width = width
    return div


def hint_styles(mode: str = "light") -> dict:
    """
    Return the style dictionary of a hint text block.

    :param mode: theme mode, defaults to ``"light"``
    :type mode: str, optional
    :return: inline styles for a hint element
    :rtype: dict
    """
    theme = THEMES[mode]
    return {"color": theme["muted"], "font-size": "12px", "line-height": "1.4",
            "padding": "2px 4px"}


def info_box(text: str = "", font_size: str = "12px", mode: str = "light") -> Div:
    """
    Create a neutral info box for dynamic status or readout text.

    :param text: initial content of the box (HTML), defaults to ""
    :type text: str, optional
    :param font_size: font size of the box text, defaults to ``"12px"``
    :type font_size: str, optional
    :param mode: theme mode, defaults to ``"light"``
    :type mode: str, optional
    :return: styled info box element
    :rtype: Div
    """
    return Div(text=text, styles=info_styles(mode, font_size))


def info_styles(mode: str = "light", font_size: str = "12px") -> dict:
    """
    Return the style dictionary of an info box.

    :param mode: theme mode, defaults to ``"light"``
    :type mode: str, optional
    :param font_size: font size of the box text, defaults to ``"12px"``
    :type font_size: str, optional
    :return: inline styles for an info box element
    :rtype: dict
    """
    theme = THEMES[mode]
    return {"background": theme["box_bg"], "border": f"1px solid {theme['box_border']}",
            "border-radius": "6px", "padding": "6px 10px", "color": theme["text"],
            "font-size": font_size}


def tab_css(mode: str = "light", accent: str | None = None,
            vertical: bool = True) -> str:
    """
    Build the CSS string for a tab strip styling.

    Vertical strips (the main sidebar) get uniformly wide, left-aligned buttons
    stacked from the top; horizontal strips (nested submenus) keep compact
    buttons. The active tab carries the given accent color.

    :param mode: theme mode, defaults to ``"light"``
    :type mode: str, optional
    :param accent: background color of the active tab; defaults to the setup
        accent of the theme
    :type accent: str | None, optional
    :param vertical: style a vertical (left location) tab strip instead of a
        horizontal one, defaults to True
    :type vertical: bool, optional
    :return: CSS string for a ``Tabs`` widget
    :rtype: str
    """
    theme = THEMES[mode]
    # placeholders instead of format() because the CSS is full of literal braces
    replacements = {
        "@DIRECTION": "column" if vertical else "row",
        "@ALIGN": "stretch" if vertical else "flex-start",
        "@ACCENT": accent or theme["accents"]["setup"],
        "@PAGETEXT": theme["page_bg"] if mode == "light" else "#131313",
        "@TABBG": theme["tab_bg"], "@TABTEXT": theme["tab_text"],
        "@HOVERBG": theme["tab_hover_bg"], "@HOVERTEXT": theme["tab_hover_text"],
    }
    css = _TAB_CSS
    for key, value in replacements.items():
        css = css.replace(key, value)
    return css


def page_css(mode: str = "light") -> str:
    """
    Build the page-wide CSS string of a theme.

    Styles the document background, the default text color, and the bokeh
    widgets (inputs, buttons, sliders, tooltips, menus) so they follow the
    theme; properties that compete with bokeh's own widget stylesheets use
    ``!important``.

    :param mode: theme mode, defaults to ``"light"``
    :type mode: str, optional
    :return: global CSS string for the whole page
    :rtype: str
    """
    theme = THEMES[mode]
    replacements = {
        "@PAGE": theme["page_bg"], "@TEXT": theme["text"], "@MUTED": theme["muted"],
        "@GRID": theme["grid"], "@BOX": theme["box_bg"],
        "@HOVER": theme["tab_hover_bg"],
        "@ACCENT": theme["accents"]["setup"],
    }
    css = _PAGE_CSS
    for key, value in replacements.items():
        css = css.replace(key, value)
    return css


if __name__ == "__main__":
    pass
