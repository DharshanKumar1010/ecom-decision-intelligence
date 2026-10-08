"""Design tokens: dark surfaces, one accent, fixed colours for the 5 actions and 4 quadrants.

Dark theme, one sans-serif font family for everything. Semantic colours exist only for the
five recommended actions and the four seller groups, are muted, and are used identically on
every page. Each carries a `color` for marks (>= 3:1 on both dark surfaces) and a lighter
`text` variant for words (>= 4.5:1); `tests/test_app.py` computes both.

The four quadrant colours were validated with the dataviz skill's validate_palette.js on the
real surface (#141A22), all pairs: lightness band, chroma, normal-vision floor and contrast
pass. The blue/violet pair of the first dark attempt was too close under red-green colour
blindness (delta E about 2), so the "Overlooked-Low-Quality" group is a rose-magenta instead:
blue vs rose is delta E >= 17 for normal vision, and green vs orange is about 22. The
remaining weak spot is rose vs green under deuteranopia (delta E about 7, the legal "needs a
second cue" band); the scatter prints each group's name in its corner and in the legend.
Promote shares Hidden Gem's green and Feature shares Star's blue on purpose: they are the
same idea.
"""

from __future__ import annotations

from dataclasses import dataclass

import plotly.graph_objects as go
import plotly.io as pio

PAGE = "#0C1015"
SURFACE = "#141A22"
TEXT = "#F2EFE9"
TEXT_MUTED = "#A8B0BA"
BORDER = "#2A3340"
GRID = "rgba(255, 255, 255, 0.12)"
ACCENT = "#D9B26F"  # gold: 9.6:1 on the page, readable for text and bars

# One sans-serif family everywhere. Streamlit's "sans serif" theme font (Source Sans, served
# from the Streamlit package, not a CDN) styles the app; charts use the same family with
# plain system sans-serif fallbacks. No serif font appears anywhere.
FONT = '"Source Sans", "Source Sans Pro", -apple-system, "Segoe UI", Arial, sans-serif'


@dataclass(frozen=True)
class Swatch:
    """A concept's colour: `color` for marks, `text` for words."""

    color: str
    text: str


QUADRANTS: dict[str, Swatch] = {
    "Star": Swatch("#3987e5", "#6aa8ee"),
    "Hidden Gem": Swatch("#1f9e74", "#3fc095"),
    "Overrated": Swatch("#d2702f", "#e69558"),
    "Overlooked-Low-Quality": Swatch("#cc65ab", "#de8fc3"),
}
ACTIONS: dict[str, Swatch] = {
    "Suspend": Swatch("#e66767", "#f08f8f"),
    "Warn": Swatch("#d9a82e", "#e6bd55"),
    "Keep": Swatch("#8b95a3", "#a8b0ba"),
    "Feature": Swatch("#3987e5", "#6aa8ee"),
    "Promote": Swatch("#1f9e74", "#3fc095"),
}
# Accent ramp for the funnel; dark ink labels stay >= 6:1 on every step.
FUNNEL_RAMP = ("#E8CB8F", "#D9B26F", "#C29F5C", "#AD8C4E")
FUNNEL_TEXT = PAGE
DIVERGING_SCALE = [[0.0, "#e66767"], [0.5, "#3A4452"], [1.0, "#3987e5"]]


def contrast_ratio(foreground: str, background: str) -> float:
    """WCAG 2 contrast ratio between two `#rrggbb` colours."""

    def luminance(hex_color: str) -> float:
        h = hex_color.lstrip("#")
        channels = [int(h[i : i + 2], 16) / 255 for i in (0, 2, 4)]
        lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]

    lighter, darker = sorted((luminance(foreground), luminance(background)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def _build_template() -> go.layout.Template:
    axis = {
        "linecolor": BORDER,
        "tickcolor": BORDER,
        "tickfont": {"color": TEXT_MUTED, "size": 12},
        "title": {"font": {"color": TEXT_MUTED, "size": 13}},
        "zeroline": False,
        "automargin": True,
    }
    return go.layout.Template(
        layout=go.Layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"family": FONT, "color": TEXT, "size": 13},
            colorway=[ACCENT, "#6aa8ee", "#8b95a3", "#1f9e74", "#d2702f"],
            margin={"l": 8, "r": 8, "t": 16, "b": 8},
            xaxis={**axis, "showgrid": False},
            yaxis={**axis, "showgrid": True, "gridcolor": GRID},
            legend={"orientation": "h", "yanchor": "top", "y": -0.18, "xanchor": "left",
                    "x": 0, "bgcolor": "rgba(0,0,0,0)", "font": {"color": TEXT_MUTED, "size": 12}},
            hoverlabel={"bgcolor": SURFACE, "bordercolor": BORDER, "font": {"color": TEXT}},
            coloraxis={"colorbar": {"outlinewidth": 0, "tickfont": {"color": TEXT_MUTED},
                                    "title": {"font": {"color": TEXT_MUTED}}}},
        )
    )


pio.templates["olist_dark"] = _build_template()
pio.templates.default = "olist_dark"
