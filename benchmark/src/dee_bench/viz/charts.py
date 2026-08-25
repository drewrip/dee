"""Static png/pdf rendering of a :class:`ChartSpec` with matplotlib.

Every interactive chart in the dashboard links to the files produced here, so
these render the same spec rather than an approximation of it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .spec import ChartSpec  # noqa: E402
from .theme import LIGHT, series_color  # noqa: E402

# Mark specs from the dataviz reference: 2px lines, >=8px markers, hairline
# recessive grid, bars capped rather than filling their slot.
LINE_WIDTH = 2.0
MARKER_SIZE = 8.0
MAX_BAR_WIDTH = 0.24
# Above this many points a series is drawn as a bare line, without markers.
DENSE_POINTS = 25


def render(spec: ChartSpec, out_dir: Path, formats: set[str]) -> dict[str, str]:
    """Render `spec` to png/pdf. Returns {format: filename}."""
    formats = {f for f in formats if f in ("png", "pdf")}
    if not formats or not spec.has_data:
        return {}
    out_dir.mkdir(parents=True, exist_ok=True)

    is_bar = spec.kind not in ("line", "scatter")
    # A fixed 10in width crowds tick labels into an unreadable pile once a bar
    # chart has more than a handful of categories (e.g. a project x variant
    # cross product), so widen the figure to the label count instead.
    n_labels = len(dict.fromkeys(spec.series[0].x)) if is_bar and spec.series else 0
    width_in = max(10.0, min(22.0, 0.9 * n_labels + 2.0)) if is_bar else 10.0
    fig, ax = plt.subplots(figsize=(width_in, 5.2), dpi=140)
    fig.patch.set_facecolor(LIGHT["surface"])
    ax.set_facecolor(LIGHT["surface"])

    if not is_bar:
        _draw_line(ax, spec)
    else:
        _draw_bars(ax, spec, rotate=n_labels > 6)

    if spec.hline is not None:
        ax.axhline(spec.hline, color=LIGHT["text_muted"], linewidth=1.0,
                   linestyle="--", zorder=1)
        ax.annotate(spec.hline_label, xy=(0.995, spec.hline), xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", fontsize=8, color=LIGHT["text_muted"])

    ax.set_title(spec.title, fontsize=13, color=LIGHT["text_primary"],
                 pad=30 if spec.subtitle else 14, loc="left")
    if spec.subtitle:
        # Offset in points, not axes fraction, so the gap is constant and the
        # subtitle can never ride up into the title.
        ax.annotate(spec.subtitle, xy=(0, 1), xycoords="axes fraction",
                    xytext=(0, 8), textcoords="offset points",
                    fontsize=9, color=LIGHT["text_secondary"], va="bottom")
    ax.set_xlabel(spec.x_label, fontsize=9, color=LIGHT["text_secondary"])
    ax.set_ylabel(spec.y_label, fontsize=9, color=LIGHT["text_secondary"])

    ax.grid(True, axis="y", color=LIGHT["grid"], linewidth=1.0, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(LIGHT["border"])
    ax.tick_params(colors=LIGHT["text_secondary"], labelsize=8)

    # A legend is always present for two or more series, so identity never
    # rests on color alone; one series is already named by the title.
    if len(spec.series) > 1:
        ax.legend(frameon=False, fontsize=8, labelcolor=LIGHT["text_secondary"],
                  loc="upper left", bbox_to_anchor=(1.005, 1.0))
    fig.tight_layout()

    written = {}
    for fmt in sorted(formats):
        path = out_dir / f"{spec.id}.{fmt}"
        fig.savefig(path, format=fmt, facecolor=fig.get_facecolor(), bbox_inches="tight")
        written[fmt] = path.name
    plt.close(fig)
    return written


def _draw_line(ax, spec: ChartSpec) -> None:
    for i, s in enumerate(spec.series):
        pts = [(x, y) for x, y in zip(s.x, s.y) if y is not None]
        if not pts:
            continue
        xs, ys = zip(*pts)
        color = series_color(s.name, i)
        # A marker per point is signal on a handful of measurements and noise on
        # a 50ms-interval timeseries, where it also breaks the line up visually.
        dense = len(pts) > DENSE_POINTS
        ax.plot(xs, ys, label=s.name, color=color, linewidth=LINE_WIDTH,
                marker="" if dense else "o",
                markersize=MARKER_SIZE ** 0.5 * 2.2,
                # A surface-colored ring keeps markers legible where lines cross.
                markeredgecolor=LIGHT["surface"], markeredgewidth=2.0,
                solid_capstyle="round", solid_joinstyle="round", zorder=3)
    if spec.x_type == "log":
        ax.set_xscale("log")
    if spec.y_type == "log":
        ax.set_yscale("log")


def _draw_bars(ax, spec: ChartSpec, rotate: bool = False) -> None:
    labels = list(dict.fromkeys(spec.series[0].x)) if spec.series else []
    n = max(len(spec.series), 1)
    # Cap the bar width so the band keeps visible air rather than being filled.
    width = min(MAX_BAR_WIDTH, 0.8 / n)
    positions = range(len(labels))

    for i, s in enumerate(spec.series):
        offset = (i - (n - 1) / 2) * (width + 0.02)
        # None means "not measured", not zero -- a 0-height bar in a ratio
        # chart (speedup, relative resource use, payback runs) would read as a
        # real, favorable measurement instead of absent data.
        xs, ys = zip(*((p + offset, y) for p, y in zip(positions, s.y) if y is not None)) \
            if any(y is not None for y in s.y) else ((), ())
        ax.bar(xs, ys, width=width, label=s.name, color=series_color(s.name, i),
               zorder=3, linewidth=0)

    ax.set_xticks(list(positions))
    # A dozen multi-line category labels set flat overlap each other; angling
    # them past that count is the only way to keep every one readable.
    if rotate:
        ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    else:
        ax.set_xticklabels(labels, fontsize=8)
    # Without this, matplotlib shrinks the x range onto the bars themselves, so
    # a chart with one category renders as a single slab filling the panel.
    ax.set_xlim(-0.5, len(labels) - 0.5)
