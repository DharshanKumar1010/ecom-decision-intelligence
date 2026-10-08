"""Design tokens: one accent, greys, fixed colours for the 5 actions and 4 quadrants.

Light theme, one sans-serif font family for everything. Semantic colours exist
only for the five recommended actions and the four seller groups, are muted, and
are used identically on every page. Each carries a `color` for marks (>= 3:1 on
white) and a darker `text` variant for words (>= 4.5:1); `tests/test_app.py`
computes both. The quadrant colours were validated with the dataviz skill's
validate_palette.js on a white surface for the ring of groups that touch in the
Discovery scatter (Star, Hidden Gem, Overlooked-Low-Quality, Overrated). Promote
shares Hidden Gem's green and Feature shares Star's blue on purpose: they are the
same idea (an under-exposed good seller / an established good seller).
"""

from __future__ import annotations

from dataclasses import dataclass

import plotly.graph_objects as go
import plotly.io as pio

WHITE = "#FFFFFF"
SURFACE = "#F8F9FA"
TEXT = "#1A1A1A"
TEXT_MUTED = "#6B7280"
BORDER = "#E5E7EB"
GRID = "#EEF0F2"
ACCENT = "#1F4E79"

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
    "Star": Swatch("#2f6fb0", "#2a639f"),
    "Hidden Gem": Swatch("#2a8f6b", "#1f7a58"),
    "Overrated": Swatch("#c4682a", "#a9561b"),
    "Overlooked-Low-Quality": Swatch("#7a6fb8", "#685ca8"),
}
ACTIONS: dict[str, Swatch] = {
    "Suspend": Swatch("#c0504d", "#b04542"),
    "Warn": Swatch("#b8860b", "#8f6a08"),
    "Keep": Swatch("#6b7280", "#6b7280"),
    "Feature": Swatch("#2f6fb0", "#2a639f"),
    "Promote": Swatch("#2a8f6b", "#1f7a58"),
}
# Accent ramp (light to dark) for single-series charts such as the funnel.
FUNNEL_RAMP = ("#9db8d1", "#6f94b6", "#40729b", "#1F4E79")
DIVERGING_SCALE = [[0.0, "#c0504d"], [0.5, "#E5E7EB"], [1.0, "#2f6fb0"]]


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
            paper_bgcolor=WHITE,
            plot_bgcolor=WHITE,
            font={"family": FONT, "color": TEXT, "size": 13},
            colorway=[ACCENT, "#6f94b6", "#9ca3af", "#2a8f6b", "#c4682a"],
            margin={"l": 8, "r": 8, "t": 16, "b": 8},
            xaxis={**axis, "showgrid": False},
            yaxis={**axis, "showgrid": True, "gridcolor": GRID},
            legend={"orientation": "h", "yanchor": "top", "y": -0.18, "xanchor": "left",
                    "x": 0, "bgcolor": WHITE, "font": {"color": TEXT_MUTED, "size": 12}},
            hoverlabel={"bgcolor": WHITE, "bordercolor": BORDER, "font": {"color": TEXT}},
            coloraxis={"colorbar": {"outlinewidth": 0, "tickfont": {"color": TEXT_MUTED},
                                    "title": {"font": {"color": TEXT_MUTED}}}},
        )
    )


pio.templates["olist_light"] = _build_template()
pio.templates.default = "olist_light"
