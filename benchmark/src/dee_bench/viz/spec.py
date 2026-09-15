"""A renderer-neutral description of a chart, a table and a page.

The dashboard renders these with plotly (interactive) and the static exporter
renders the same objects with matplotlib (png/pdf). Keeping one description
means the downloadable chart is always the chart on screen, not a lookalike
that drifts — and a new chart kind is added once, here, rather than twice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Every kind both renderers understand.
#
#   line          x/y lines, markers when the series is short enough to want them
#   scatter       the same, for a dense timeseries (markers suppressed)
#   points        markers only — a correlation, not a trend
#   step          a value that holds until it changes (a DAG version, a rung)
#   area          a cumulative quantity, filled to the axis
#   bar           one series over categories
#   grouped_bar   several series side by side over categories
#   stacked_bar   several series summing to a whole over categories
#   hbar          one series over categories, horizontal (long labels, rankings)
#   stacked_hbar  parts of a whole, horizontal
#   heatmap       a matrix of magnitudes over two category axes
#   span          horizontal ranges on lanes — when things ran, and for how long
KINDS = ("line", "scatter", "points", "step", "area", "bar", "grouped_bar",
         "stacked_bar", "hbar", "stacked_hbar", "heatmap", "span")

BAR_KINDS = ("bar", "grouped_bar", "stacked_bar", "hbar", "stacked_hbar")
HORIZONTAL_KINDS = ("hbar", "stacked_hbar")
LINE_KINDS = ("line", "scatter", "points", "step", "area")


@dataclass
class Series:
    name: str
    x: list[Any]
    y: list[Any]
    # Per-point extras surfaced in tooltips, e.g. {"n": [3, 3], "combo": [...]}.
    # `span` charts carry their ends here, under "end".
    meta: dict[str, list[Any]] = field(default_factory=dict)
    # Symmetric (float) or asymmetric ((lo, hi)) error bars, in data units.
    error: list[Any] | None = None
    # Colour overrides. `color` fixes the whole series in both modes — used for
    # status hues, which mean the same thing on either surface. `colors` gives
    # a colour per point, for a chart whose marks encode an outcome.
    color: str | None = None
    colors: list[str] | None = None
    dash: str | None = None          # "dash" | "dot"
    kind: str | None = None          # per-series override (a line over bars)
    labels: list[str] | None = None  # text drawn on each mark
    fill: bool = False
    marker: str | None = None
    # Drawn but kept out of the legend — a control, a reference, a helper.
    legend: bool = True


@dataclass
class ChartSpec:
    id: str
    kind: str
    title: str
    series: list[Series] = field(default_factory=list)
    x_label: str = ""
    y_label: str = ""
    subtitle: str = ""
    x_type: str = "category"  # "category" | "linear" | "log"
    y_type: str = "linear"    # "linear" | "log"
    note: str = ""
    # A reference line, e.g. the unoptimized baseline at 1.0.
    hline: float | None = None
    hline_label: str = ""
    # Further reference lines, and event markers on the x axis — which is how a
    # run-by-run chart says "the search promoted its result here".
    hlines: list[tuple[float, str]] = field(default_factory=list)
    vlines: list[tuple[Any, str]] = field(default_factory=list)
    # Shaded y ranges (lo, hi, label), for a tolerance or a confidence band.
    bands: list[tuple[float, float, str]] = field(default_factory=list)
    # Free annotations at data coordinates.
    annotations: list[tuple[Any, Any, str]] = field(default_factory=list)
    # heatmap payload: matrix[row][col], with tick labels for both axes.
    matrix: list[list[float | None]] = field(default_factory=list)
    x_ticks: list[str] = field(default_factory=list)
    y_ticks: list[str] = field(default_factory=list)
    z_label: str = ""
    z_fmt: str = "{:.2f}"
    # What to print in each heatmap cell, when the number the colour encodes is
    # a code for something (a verdict, a state) rather than a quantity.
    z_text: list[list[str]] = field(default_factory=list)
    # Draw each mark's value beside it. Worth it on a handful of bars, noise on
    # a hundred.
    value_labels: bool = False
    value_fmt: str = "{:.2f}"
    height: int = 380
    legend_title: str = ""
    # Rendered under the chart, smaller than `note` — a caveat, a source.
    footnote: str = ""

    @property
    def is_bar(self) -> bool:
        return self.kind in BAR_KINDS

    @property
    def horizontal(self) -> bool:
        return self.kind in HORIZONTAL_KINDS

    @property
    def stacked(self) -> bool:
        return self.kind in ("stacked_bar", "stacked_hbar")

    @property
    def has_data(self) -> bool:
        if self.kind == "heatmap":
            return any(v is not None for row in self.matrix for v in row)
        if self.kind == "span":
            # A span's measurements are its ends, not its `y`, which only
            # names the lane a bar belongs to.
            return any(any(v is not None for v in s.meta.get("end", []))
                       for s in self.series)
        return any(any(v is not None for v in s.y) for s in self.series)


@dataclass
class Table:
    """A table view of a study's numbers.

    Every study carries at least one, because a table is the non-colour channel
    that makes a chart's content readable without seeing its hues.
    """

    key: str
    title: str
    columns: list[str]
    rows: list[list[Any]] = field(default_factory=list)
    note: str = ""
    # Columns to right-align, by index. Numbers, normally.
    numeric: tuple[int, ...] = ()

    @property
    def has_data(self) -> bool:
        return bool(self.rows)


@dataclass
class Kpi:
    """A headline number, shown as a tile above a page's charts."""

    label: str
    value: str
    hint: str = ""
    # "good" | "warning" | "critical" | "" — a reserved status hue, or none.
    tone: str = ""


@dataclass
class Study:
    """One page of the dashboard.

    `group` decides where it sits in the navigation: the seven numbered studies
    under "Studies", a page per enabled optimization under "Optimizations", and
    the run-by-run view of a continuous optimization under "Runs".
    """

    key: str
    number: int
    title: str
    question: str
    charts: list[ChartSpec] = field(default_factory=list)
    kpis: list[Kpi] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    # The older single-table form, still how the seven studies declare theirs.
    table_columns: list[str] = field(default_factory=list)
    table_rows: list[list[Any]] = field(default_factory=list)
    empty_reason: str = ""
    group: str = "studies"
    subtitle: str = ""

    def all_tables(self) -> list[Table]:
        tables = list(self.tables)
        if self.table_rows:
            tables.append(Table(key=f"{self.key}-data", title="Every measurement",
                                columns=self.table_columns, rows=self.table_rows))
        return tables

    @property
    def has_content(self) -> bool:
        return (any(c.has_data for c in self.charts)
                or any(t.has_data for t in self.all_tables())
                or bool(self.kpis))
