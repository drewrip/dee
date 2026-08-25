"""Shared visual language for the dashboard and the static charts.

The categorical palette is the validated default from the dataviz reference
instance, in fixed slot order — hues are assigned by slot and never cycled, so
a series keeps its color when a filter changes the series count.

Both modes are selected, not flipped: the dark column is the same eight hues
re-stepped for the dark surface. The two sets were validated together
(``scripts/validate_palette.js``, all checks pass in both modes).
"""

from __future__ import annotations

# Categorical slots, in fixed assignment order.
LIGHT_SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
DARK_SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500",
               "#d55181", "#008300", "#9085e9", "#e66767"]

LIGHT = {
    "surface": "#fcfcfb",
    "surface_2": "#f4f4f2",
    "text_primary": "#0b0b0b",
    "text_secondary": "#52514e",
    "text_muted": "#78776f",
    "grid": "#e6e6e2",
    "border": "#dcdcd6",
}
DARK = {
    "surface": "#1a1a19",
    "surface_2": "#232322",
    "text_primary": "#ffffff",
    "text_secondary": "#c3c2b7",
    "text_muted": "#8f8e85",
    "grid": "#333331",
    "border": "#3a3a38",
}

# Reserved status colors. Never reused as a categorical series.
STATUS = {
    "good": "#008300",
    "warning": "#eda100",
    "critical": "#e34948",
}

# Variants get stable slots so a variant is the same color in every chart of
# the dashboard, whichever subset a given chart happens to show.
VARIANT_ORDER = ["unopt", "hmp", "hmp_pushdown", "omp", "full"]


def series_color(name: str, index: int, mode: str = "light") -> str:
    """Color for a named series, stable across charts."""
    palette = LIGHT_SERIES if mode == "light" else DARK_SERIES
    if name in VARIANT_ORDER:
        return palette[VARIANT_ORDER.index(name) % len(palette)]
    return palette[index % len(palette)]


def ordered(names: list[str]) -> list[str]:
    """Sort series so the ablation ladder reads in its natural order."""
    return sorted(names, key=lambda n: (VARIANT_ORDER.index(n) if n in VARIANT_ORDER else 99, n))
