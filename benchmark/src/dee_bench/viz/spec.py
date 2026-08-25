"""A renderer-neutral description of a chart.

The dashboard renders these with plotly (interactive) and the static exporter
renders the same objects with matplotlib (png/pdf). Keeping one description
means the downloadable chart is always the chart on screen, not a lookalike
that drifts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Series:
    name: str
    x: list[Any]
    y: list[Any]
    # Per-point extras surfaced in tooltips, e.g. {"n": [3, 3], "ci": [...]}.
    meta: dict[str, list[Any]] = field(default_factory=dict)
    # Symmetric or (low, high) error bars, in data units.
    error: list[tuple[float, float] | None] | None = None


@dataclass
class ChartSpec:
    id: str
    kind: str  # "line" | "grouped_bar" | "bar" | "scatter"
    title: str
    series: list[Series]
    x_label: str = ""
    y_label: str = ""
    subtitle: str = ""
    x_type: str = "category"  # "category" | "linear" | "log"
    y_type: str = "linear"  # "linear" | "log"
    note: str = ""
    # A reference line, e.g. the unoptimized baseline at 1.0.
    hline: float | None = None
    hline_label: str = ""

    @property
    def has_data(self) -> bool:
        return any(s.y for s in self.series)


@dataclass
class Study:
    key: str
    number: int
    title: str
    question: str
    charts: list[ChartSpec] = field(default_factory=list)
    # Column headers and rows for the study's table view, which is how the
    # accessibility requirement for a non-color channel is met.
    table_columns: list[str] = field(default_factory=list)
    table_rows: list[list[Any]] = field(default_factory=list)
    empty_reason: str = ""

    @property
    def has_content(self) -> bool:
        return any(c.has_data for c in self.charts) or bool(self.table_rows)
