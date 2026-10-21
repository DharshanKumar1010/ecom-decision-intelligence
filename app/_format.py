"""Number formatting shared by every page and table.

One place decides how numbers read: thousands separators for counts, a fixed number of
decimals for everything else, and never a negative zero (`-0.000`), which appears when a
tiny negative value is rounded for display. Presentation only; no business logic.
"""

from __future__ import annotations

import math


def clean(value: float, decimals: int) -> float:
    """Round to `decimals` places, turning a negative zero into a plain zero.

    Used for numbers that go into a table cell: after rounding, `-0.0` becomes `0.0`, so a
    renderer never shows "-0.000".
    """
    rounded = round(float(value), decimals)
    return rounded + 0.0  # -0.0 + 0.0 == 0.0


def fmt_int(value: float) -> str:
    """A whole number with thousands separators, e.g. `110,189`; an en dash if missing."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    return f"{int(round(float(value))):,}"


def fmt_num(value: float | None, decimals: int = 2, signed: bool = False) -> str:
    """A number with a fixed number of decimals and thousands separators.

    `None`/NaN give an en dash. `signed=True` adds an explicit `+` to positive values
    (for changes). A value that rounds to zero never shows a minus sign.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    shown = clean(value, decimals)
    return f"{shown:+,.{decimals}f}" if signed else f"{shown:,.{decimals}f}"


def fmt_pct(value: float | None, decimals: int = 1) -> str:
    """A share (0 to 1) as a percentage with fixed decimals, e.g. `7.91%`."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    return f"{clean(float(value) * 100, decimals):,.{decimals}f}%"
