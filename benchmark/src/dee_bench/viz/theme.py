"""Shared visual language for the dashboard and the static charts.

One palette, one set of roles, one ordering rule — so a variant is the same
colour in the interactive dashboard, in the png a slide pulls in, and in the
pdf a paper embeds.

The categorical palette is the validated default from the dataviz reference
instance, in fixed slot order: hues are assigned by slot and never cycled, so
a series keeps its colour when a filter changes the series count.

Both modes are selected, not flipped: the dark column is the same eight hues
re-stepped for the dark surface. The two sets were validated together, all
checks passing in both modes.
"""

from __future__ import annotations

import re

# Categorical slots, in fixed assignment order.
LIGHT_SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
DARK_SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500",
               "#d55181", "#008300", "#9085e9", "#e66767"]

LIGHT = {
    "surface": "#fcfcfb",
    "surface_2": "#f4f4f2",
    "surface_3": "#ebebe7",
    "text_primary": "#0b0b0b",
    "text_secondary": "#52514e",
    "text_muted": "#78776f",
    "grid": "#e6e6e2",
    "border": "#dcdcd6",
    "accent": "#2a78d6",
}
DARK = {
    "surface": "#1a1a19",
    "surface_2": "#232322",
    "surface_3": "#2c2c2a",
    "text_primary": "#ffffff",
    "text_secondary": "#c3c2b7",
    "text_muted": "#8f8e85",
    "grid": "#333331",
    "border": "#3a3a38",
    "accent": "#3987e5",
}

# Reserved status colours. Never reused as a categorical series.
STATUS = {
    "good": "#008300",
    "warning": "#eda100",
    "critical": "#e34948",
    "neutral": "#78776f",
}
STATUS_DARK = {
    "good": "#3aa03a",
    "warning": "#c98500",
    "critical": "#e66767",
    "neutral": "#8f8e85",
}

# A single-hue ramp for ordered magnitude (heatmaps, densities). Light steps
# first; the dark column starts darker so a cell reads against the dark surface.
SEQUENTIAL = ["#eef4fc", "#cfe0f6", "#a8c8ee", "#7aabe3", "#4a8bd6", "#2a78d6", "#1b558f"]
SEQUENTIAL_DARK = ["#1f2b3a", "#24405e", "#255884", "#2270ab", "#2a88cf", "#3f9de3", "#77bcf0"]

# What the optimizer's search did with a candidate. These are outcome states,
# not series, so they take reserved status hues rather than categorical slots.
OUTCOME_COLORS = {
    "baseline": "#78776f",
    "ok": "#2a78d6",
    "cancelled": "#eda100",
    "skipped": "#c9c8c2",
    "error": "#e34948",
}
OUTCOME_COLORS_DARK = {
    "baseline": "#8f8e85",
    "ok": "#3987e5",
    "cancelled": "#c98500",
    "skipped": "#4a4a47",
    "error": "#e66767",
}

# Variants get stable slots so a variant is the same colour in every chart of
# the dashboard, whichever subset a given chart happens to show. The baseline
# is always slot 0 and the first optimized rung always slot 1, which is what
# makes a two-series chart read the same way everywhere.
VARIANT_ORDER = ["unopt", "hmp", "hmp_pushdown", "omp",
                 "nodefusion", "parallelism", "pushdown", "full"]

# Names a config may give the same rung. An alias takes its target's slot, so
# a sweep that calls its baseline `base` and one that calls it `unopt` produce
# charts in the same colours.
VARIANT_ALIASES = {
    "base": "unopt",
    "baseline": "unopt",
    "none": "unopt",
    "nf": "nodefusion",
    "nf_rule": "nodefusion",
    "node_fusion": "nodefusion",
    "nf_adaptive": "hmp_pushdown",
    "par": "parallelism",
}

# The optimizations dee can run, in pipeline order, with what each one is for.
# The dashboard grows a page per pass it finds evidence of, and this is where
# that page's identity lives.
PASSES: dict[str, dict[str, str]] = {
    "hmp": {
        "label": "HMP",
        "full": "Heuristic materialization",
        "blurb": "Picks views worth materializing by measuring candidate DAGs against a "
                 "run budget.",
        "pass_name": "HMPPass",
    },
    "omp": {
        "label": "OMP",
        "full": "Optimal materialization",
        "blurb": "Enumerates materialization plans over the most central nodes and keeps "
                 "the best one it measured.",
        "pass_name": "OMPPass",
    },
    "nodefusion": {
        "label": "NodeFusion",
        "full": "Node fusion",
        "blurb": "Rolls a region of views up into one query, so shared work is computed "
                 "once inside a single plan.",
        "pass_name": "NodeFusionPass",
    },
    "parallelism": {
        "label": "Parallelism",
        "full": "Parallelism tuning",
        "blurb": "Walks a ladder of node-concurrency caps and keeps the rung that beat "
                 "its own control.",
        "pass_name": "ParallelismTuning",
    },
    "pushdown": {
        "label": "Pushdown",
        "full": "Predicate pushdown",
        "blurb": "A static rewrite: pushes filters into the queries that feed a "
                 "materialization. Spends no DAG runs.",
        "pass_name": "PushdownPass",
    },
}

# pass_stats.pass_name -> the key above.
PASS_BY_NAME = {meta["pass_name"]: key for key, meta in PASSES.items()}

_LABEL_SUFFIX = re.compile(r"\[.*\]$")


def base_name(name: str) -> str:
    """The series name with any swept-parameter label stripped.

    ``hmp[hmp_objective=makespan]`` and ``hmp[hmp_objective=query_time]`` are
    two cells of one variant, and colouring them from the same family is what
    makes a sweep chart readable: the eye groups the variant, and the legend
    separates the setting.
    """
    stripped = _LABEL_SUFFIX.sub("", str(name)).strip()
    return VARIANT_ALIASES.get(stripped, stripped)


def palette(mode: str = "light") -> list[str]:
    return LIGHT_SERIES if mode == "light" else DARK_SERIES


def surface(mode: str = "light") -> dict[str, str]:
    return LIGHT if mode == "light" else DARK


def status(mode: str = "light") -> dict[str, str]:
    return STATUS if mode == "light" else STATUS_DARK


def outcome_color(outcome: str, mode: str = "light") -> str:
    table = OUTCOME_COLORS if mode == "light" else OUTCOME_COLORS_DARK
    return table.get(str(outcome).split(" ")[0].lower(), table["skipped"])


def series_color(name: str, index: int, mode: str = "light") -> str:
    """Colour for a named series, stable across every chart."""
    colors = palette(mode)
    base = base_name(name)
    if base in VARIANT_ORDER:
        return colors[VARIANT_ORDER.index(base) % len(colors)]
    return colors[index % len(colors)]


def ramp(fraction: float, mode: str = "light") -> str:
    """A step of the sequential ramp for `fraction` in [0, 1]."""
    steps = SEQUENTIAL if mode == "light" else SEQUENTIAL_DARK
    if fraction != fraction:  # NaN
        return surface(mode)["surface_2"]
    i = int(max(0.0, min(1.0, fraction)) * (len(steps) - 1) + 0.5)
    return steps[i]


def ordered(names: list[str]) -> list[str]:
    """Sort series so the ablation ladder reads in its natural order."""
    def key(n: str):
        base = base_name(n)
        return (VARIANT_ORDER.index(base) if base in VARIANT_ORDER else 99, str(n))
    return sorted(names, key=key)


# Type stacks, shared by both renderers so the png matches the page.
FONT_STACK = 'ui-sans-serif, system-ui, -apple-system, "Segoe UI", Helvetica, sans-serif'
MONO_STACK = 'ui-monospace, SFMono-Regular, Menlo, "Cascadia Mono", monospace'
