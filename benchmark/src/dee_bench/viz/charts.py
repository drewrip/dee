"""Static png/pdf rendering of a :class:`ChartSpec` with matplotlib.

Every chart in the dashboard offers a PNG and a PDF of itself, and both come
from here — rendering the same spec the page rendered, rather than an
approximation of it that drifts.

The PNG is what lands on a slide, so it is rasterised with headroom to be
scaled up; the PDF is vector and stays sharp at any size, which is what a paper
wants. Both are drawn on the light surface: a downloaded asset goes into a
document with its own background, not into the dashboard's dark mode.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from .spec import BAR_KINDS, ChartSpec, Series  # noqa: E402
from .theme import LIGHT, STATUS, ramp, series_color  # noqa: E402

# Mark specs from the dataviz reference: 2px lines, >=8px markers, hairline
# recessive grid, bars capped rather than filling their slot.
LINE_WIDTH = 2.0
MARKER_SIZE = 6.5
# Bars are capped rather than filling their slot, so a band keeps visible air.
# A lone series can afford a wider bar than one of four sharing the slot.
MAX_BAR_WIDTH = 0.26
MAX_SINGLE_BAR_WIDTH = 0.46
# Above this many points a series is drawn as a bare line, without markers.
DENSE_POINTS = 25
# The raster is scaled up on a slide; the vector pdf ignores this.
PNG_DPI = 300

_DASHES = {"dash": (5, 3), "dot": (1.6, 2.4), None: None, "": None}

plt.rcParams.update({
    "font.size": 9,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "axes.facecolor": LIGHT["surface"],
    "figure.facecolor": LIGHT["surface"],
    "savefig.facecolor": LIGHT["surface"],
    "axes.edgecolor": LIGHT["border"],
    "axes.labelcolor": LIGHT["text_secondary"],
    "text.color": LIGHT["text_primary"],
    "xtick.color": LIGHT["text_secondary"],
    "ytick.color": LIGHT["text_secondary"],
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "grid.color": LIGHT["grid"],
    "axes.axisbelow": True,
    "legend.fontsize": 8,
    "figure.dpi": 110,
    "pdf.fonttype": 42,
})


def render(spec: ChartSpec, out_dir: Path, formats: set[str]) -> dict[str, str]:
    """Render `spec` to png/pdf. Returns {format: filename}."""
    formats = {f for f in formats if f in ("png", "pdf")}
    if not formats or not spec.has_data:
        return {}
    out_dir.mkdir(parents=True, exist_ok=True)

    width_in, height_in, rotate = _figure_shape(spec)
    fig, ax = plt.subplots(figsize=(width_in, height_in))

    if spec.kind == "heatmap":
        _draw_heatmap(fig, ax, spec)
    elif spec.kind == "span":
        _draw_spans(ax, spec)
    elif spec.horizontal:
        _draw_hbars(ax, spec)
    elif spec.is_bar:
        _draw_bars(ax, spec, rotate=rotate)
    else:
        _draw_lines(ax, spec)

    _reference_marks(ax, spec)
    _frame(ax, spec)
    _titles(ax, spec)
    _legend(ax, spec)

    fig.tight_layout()
    _caption(fig, ax, spec)

    written: dict[str, str] = {}
    for fmt in sorted(formats):
        path = out_dir / f"{spec.id}.{fmt}"
        fig.savefig(path, format=fmt, dpi=PNG_DPI if fmt == "png" else None,
                    facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.18)
        written[fmt] = path.name
    plt.close(fig)
    return written


# --------------------------------------------------------------------------
# shape
# --------------------------------------------------------------------------


def _categories(spec: ChartSpec) -> list:
    """The x categories of a bar chart, in first-seen order across all series."""
    seen: dict = {}
    for s in spec.series:
        for x in s.x:
            seen.setdefault(x, None)
    return list(seen)


def _figure_shape(spec: ChartSpec) -> tuple[float, float, bool]:
    """Width, height and whether x tick labels need angling.

    A fixed width crowds tick labels into an unreadable pile once a bar chart
    has more than a handful of categories (a project x variant cross product,
    say), so the figure widens with the label count instead — and only angles
    them when widening has run out of room.
    """
    height = max(3.4, spec.height / 72.0)
    if spec.kind == "heatmap":
        cols, rows = max(len(spec.x_ticks), 1), max(len(spec.y_ticks), 1)
        return (max(7.0, min(20.0, 0.62 * cols + 3.2)),
                max(3.2, min(16.0, 0.34 * rows + 1.9)), cols > 10)
    if spec.horizontal or spec.kind == "span":
        lanes = len(_lane_labels(spec)) or 1
        return 10.0, max(3.2, min(18.0, 0.30 * lanes + 1.7)), False
    if spec.is_bar:
        n = len(_categories(spec))
        width = max(9.0, min(22.0, 0.95 * n + 2.4))
        # Past this many categories no width keeps flat labels apart.
        return width, height, n > 7
    return 10.0, height, False


def _lane_labels(spec: ChartSpec) -> list[str]:
    if spec.kind == "span":
        seen: dict = {}
        for s in spec.series:
            for lane in s.meta.get("lane", []):
                seen.setdefault(str(lane), None)
        return list(seen)
    return [str(c) for c in _categories(spec)]


def _color(s: Series, i: int) -> str:
    return s.color or series_color(s.name, i)


# --------------------------------------------------------------------------
# marks
# --------------------------------------------------------------------------


def _draw_lines(ax, spec: ChartSpec) -> None:
    for i, s in enumerate(spec.series):
        pts = [(x, y) for x, y in zip(s.x, s.y) if y is not None]
        if not pts:
            continue
        xs, ys = zip(*pts)
        color = _color(s, i)
        kind = s.kind or spec.kind
        # A marker per point is signal on a handful of measurements and noise
        # on a 100ms-interval timeseries, where it also breaks up the line.
        dense = len(pts) > DENSE_POINTS
        marker = s.marker or ("" if (dense or kind == "scatter") else "o")
        label = s.name if s.legend else "_nolegend_"

        if kind == "points":
            colors = s.colors[:len(xs)] if s.colors else color
            ax.scatter(xs, ys, s=MARKER_SIZE ** 2, c=colors, label=label,
                       edgecolors=LIGHT["surface"], linewidths=1.2, zorder=3)
        elif kind == "step":
            ax.step(xs, ys, where="post", label=label, color=color,
                    linewidth=LINE_WIDTH, marker=marker, markersize=MARKER_SIZE,
                    markeredgecolor=LIGHT["surface"], markeredgewidth=1.6, zorder=3)
        else:
            dashes = _DASHES.get(s.dash)
            line, = ax.plot(
                xs, ys, label=label, color=color, linewidth=LINE_WIDTH, marker=marker,
                markersize=MARKER_SIZE,
                # A surface-coloured ring keeps markers legible where lines cross.
                markeredgecolor=LIGHT["surface"], markeredgewidth=1.6,
                solid_capstyle="round", solid_joinstyle="round", zorder=3,
            )
            if dashes:
                line.set_dashes(dashes)
            if kind == "area" or s.fill:
                ax.fill_between(xs, ys, color=color, alpha=0.13, linewidth=0, zorder=2)

        if s.error:
            lo, hi = _error_arms(s, ys)
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=color,
                        elinewidth=1.2, capsize=3, alpha=0.75, zorder=2)
        if s.labels:
            for x, y, text in zip(xs, ys, s.labels):
                if text:
                    ax.annotate(str(text), (x, y), textcoords="offset points",
                                xytext=(0, 8), ha="center", fontsize=7.5,
                                color=LIGHT["text_muted"])

    if spec.x_type == "log":
        ax.set_xscale("log")
    if spec.y_type == "log":
        ax.set_yscale("log")


def _error_arms(s: Series, ys) -> tuple[list[float], list[float]]:
    lo, hi = [], []
    for j, _ in enumerate(ys):
        e = s.error[j] if s.error and j < len(s.error) else None
        if e is None:
            lo.append(0.0), hi.append(0.0)
        elif isinstance(e, (tuple, list)):
            lo.append(float(e[0])), hi.append(float(e[1]))
        else:
            lo.append(float(e)), hi.append(float(e))
    return lo, hi


def _draw_bars(ax, spec: ChartSpec, rotate: bool = False) -> None:
    labels = _categories(spec)
    index = {label: i for i, label in enumerate(labels)}
    positions = range(len(labels))
    # A series that overrides the chart's kind is drawn over the bars — a
    # target, a prediction, a control — rather than beside them as another bar.
    bar_series = [s for s in spec.series if (s.kind or spec.kind) in BAR_KINDS]
    n = max(len(bar_series), 1)
    cap = MAX_SINGLE_BAR_WIDTH if n == 1 else MAX_BAR_WIDTH
    width = 0.72 if spec.stacked else min(cap, 0.78 / n)

    running = [0.0] * len(labels)
    for i, s in enumerate(spec.series):
        if (s.kind or spec.kind) not in BAR_KINDS:
            xs = [index[x] for x, y in zip(s.x, s.y) if x in index and y is not None]
            ys = [y for x, y in zip(s.x, s.y) if x in index and y is not None]
            ax.plot(xs, ys, color=s.color or STATUS["neutral"], linewidth=1.6,
                    marker="_", markersize=13, markeredgewidth=2.2,
                    linestyle="-" if (s.kind or spec.kind) == "line" else "none",
                    label=s.name if s.legend else "_nolegend_", zorder=5)
            continue

        slot = bar_series.index(s)
        offset = 0.0 if spec.stacked else (slot - (n - 1) / 2) * (width + 0.03)
        xs, ys, bottoms, colors = [], [], [], []
        for x, y in zip(s.x, s.y):
            if y is None or x not in index:
                # None means "not measured", not zero — a 0-height bar in a
                # ratio chart (speedup, relative resource use, payback runs)
                # would read as a real, favourable measurement.
                continue
            pos = index[x]
            xs.append(pos + offset)
            ys.append(y)
            bottoms.append(running[pos] if spec.stacked else 0.0)
            if spec.stacked:
                running[pos] += y
        if not xs:
            continue
        colors = s.colors[:len(xs)] if s.colors else _color(s, i)
        bars = ax.bar(xs, ys, width=width, bottom=bottoms, label=s.name if s.legend else "_nolegend_",
                      color=colors, zorder=3, linewidth=0)
        if s.error:
            lo, hi = _error_arms(s, ys)
            ax.errorbar(xs, [b + y for b, y in zip(bottoms, ys)], yerr=[lo, hi],
                        fmt="none", ecolor=LIGHT["text_muted"], elinewidth=1.1,
                        capsize=3, zorder=4)
        if spec.value_labels and len(xs) <= 40:
            ax.bar_label(bars, labels=[spec.value_fmt.format(v) for v in ys],
                         padding=2, fontsize=7.5, color=LIGHT["text_secondary"])

    ax.set_xticks(list(positions))
    ax.set_xticklabels([str(v) for v in labels], fontsize=8,
                       rotation=28 if rotate else 0, ha="right" if rotate else "center")
    # Without this, matplotlib shrinks the x range onto the bars themselves, so
    # a chart with one category renders as a single slab filling the panel.
    ax.set_xlim(-0.6, len(labels) - 0.4)
    if spec.y_type == "log":
        ax.set_yscale("log")


def _draw_hbars(ax, spec: ChartSpec) -> None:
    labels = _categories(spec)
    index = {label: i for i, label in enumerate(labels)}
    n = max(len(spec.series), 1)
    height = 0.66 if spec.stacked else min(0.74 / n, 0.46 if n == 1 else 0.3)

    running = [0.0] * len(labels)
    for i, s in enumerate(spec.series):
        ys, xs, lefts = [], [], []
        for x, y in zip(s.x, s.y):
            if y is None or x not in index:
                continue
            pos = index[x]
            offset = 0.0 if spec.stacked else (i - (n - 1) / 2) * (height + 0.03)
            ys.append(pos + offset)
            xs.append(y)
            lefts.append(running[pos] if spec.stacked else 0.0)
            if spec.stacked:
                running[pos] += y
        if not ys:
            continue
        colors = s.colors[:len(ys)] if s.colors else _color(s, i)
        bars = ax.barh(ys, xs, height=height, left=lefts,
                       label=s.name if s.legend else "_nolegend_",
                       color=colors, zorder=3, linewidth=0)
        if spec.value_labels and len(ys) <= 40:
            ax.bar_label(bars, labels=[spec.value_fmt.format(v) for v in xs],
                         padding=3, fontsize=7.5, color=LIGHT["text_secondary"])

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([str(v) for v in labels], fontsize=8)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    # Rankings read top-down: the biggest bar belongs at the top.
    ax.invert_yaxis()
    if spec.x_type == "log":
        ax.set_xscale("log")


def _draw_spans(ax, spec: ChartSpec) -> None:
    """Horizontal ranges on lanes — when each thing ran, and for how long."""
    lanes = _lane_labels(spec)
    index = {lane: i for i, lane in enumerate(lanes)}
    for i, s in enumerate(spec.series):
        ends = s.meta.get("end", [])
        lane_of = s.meta.get("lane", [])
        color = _color(s, i)
        drawn = False
        for j, start in enumerate(s.x):
            if j >= len(ends) or start is None or ends[j] is None:
                continue
            lane = index.get(str(lane_of[j])) if j < len(lane_of) else None
            if lane is None:
                continue
            ax.barh(lane, max(float(ends[j]) - float(start), 0.0), left=float(start),
                    height=0.58, color=color, zorder=3, linewidth=0,
                    label=s.name if (not drawn and s.legend) else "_nolegend_")
            drawn = True
    ax.set_yticks(range(len(lanes)))
    ax.set_yticklabels(lanes, fontsize=8)
    ax.set_ylim(-0.7, len(lanes) - 0.3)
    ax.invert_yaxis()


def _draw_heatmap(fig, ax, spec: ChartSpec) -> None:
    values = [v for row in spec.matrix for v in row if v is not None]
    lo, hi = (min(values), max(values)) if values else (0.0, 1.0)
    span = (hi - lo) or 1.0
    for r, row in enumerate(spec.matrix):
        for c, value in enumerate(row):
            if value is None:
                # Absent, not zero: a hatched cell says "not measured" where a
                # pale one would say "measured, and small".
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                           facecolor=LIGHT["surface_2"], hatch="////",
                                           edgecolor=LIGHT["border"], linewidth=0.5))
                continue
            fraction = (value - lo) / span
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                       facecolor=ramp(fraction), edgecolor=LIGHT["surface"],
                                       linewidth=1.2))
            if len(spec.matrix) * max(len(row), 1) <= 200:
                label = (spec.z_text[r][c] if r < len(spec.z_text)
                         and c < len(spec.z_text[r]) else spec.z_fmt.format(value))
                ax.text(c, r, label, ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if fraction > 0.62 else LIGHT["text_primary"])
    ax.set_xlim(-0.5, max(len(spec.x_ticks), 1) - 0.5)
    ax.set_ylim(max(len(spec.y_ticks), 1) - 0.5, -0.5)
    ax.set_xticks(range(len(spec.x_ticks)))
    ax.set_xticklabels(spec.x_ticks, fontsize=8,
                       rotation=30 if len(spec.x_ticks) > 8 else 0,
                       ha="right" if len(spec.x_ticks) > 8 else "center")
    ax.set_yticks(range(len(spec.y_ticks)))
    ax.set_yticklabels(spec.y_ticks, fontsize=8)
    ax.grid(False)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)


# --------------------------------------------------------------------------
# furniture
# --------------------------------------------------------------------------


def _reference_marks(ax, spec: ChartSpec) -> None:
    lines = list(spec.hlines)
    if spec.hline is not None:
        lines.append((spec.hline, spec.hline_label))
    for value, label in lines:
        ax.axhline(value, color=LIGHT["text_muted"], linewidth=1.0, linestyle="--", zorder=1)
        if label:
            ax.annotate(label, xy=(0.995, value), xycoords=("axes fraction", "data"),
                        ha="right", va="bottom", fontsize=7.5, color=LIGHT["text_muted"])
    for value, label in spec.vlines:
        ax.axvline(value, color=STATUS["warning"], linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
        if label:
            ax.annotate(label, xy=(value, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(4, -10), textcoords="offset points",
                        ha="left", va="top", fontsize=7.5, color=STATUS["warning"])
    for lo, hi, label in spec.bands:
        ax.axhspan(lo, hi, color=LIGHT["text_muted"], alpha=0.08, zorder=0, linewidth=0)
        if label:
            ax.annotate(label, xy=(0.005, hi), xycoords=("axes fraction", "data"),
                        xytext=(0, 3), textcoords="offset points",
                        fontsize=7.5, color=LIGHT["text_muted"])
    positions = {label: i for i, label in enumerate(_categories(spec))} \
        if spec.is_bar else {}
    for x, y, text in spec.annotations:
        # On a bar chart the x axis is positions, not the labels the caller
        # knows the categories by, so an annotation names its category and is
        # placed at that category's slot.
        at = positions.get(x, x)
        if isinstance(at, str) and positions:
            continue  # names a category this chart does not have
        ax.annotate(text, xy=(at, y), textcoords="offset points", xytext=(6, 6),
                    fontsize=7.5, color=LIGHT["text_secondary"])


def _frame(ax, spec: ChartSpec) -> None:
    if spec.kind != "heatmap":
        axis = "x" if (spec.horizontal or spec.kind == "span") else "y"
        ax.grid(True, axis=axis, color=LIGHT["grid"], linewidth=1.0, linestyle="-", zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(LIGHT["border"])
        ax.tick_params(colors=LIGHT["text_secondary"], labelsize=8, length=3, width=0.8)
    ax.set_xlabel(spec.x_label, fontsize=9, color=LIGHT["text_secondary"], labelpad=7)
    ax.set_ylabel(spec.y_label, fontsize=9, color=LIGHT["text_secondary"], labelpad=7)


def _titles(ax, spec: ChartSpec) -> None:
    ax.set_title(spec.title, fontsize=12.5, color=LIGHT["text_primary"],
                 fontweight="bold", pad=30 if spec.subtitle else 14, loc="left")
    if spec.subtitle:
        # Offset in points, not axes fraction, so the gap is constant and the
        # subtitle can never ride up into the title.
        ax.annotate(spec.subtitle, xy=(0, 1), xycoords="axes fraction",
                    xytext=(0, 9), textcoords="offset points",
                    fontsize=8.5, color=LIGHT["text_secondary"], va="bottom")


def _legend(ax, spec: ChartSpec) -> None:
    handles, labels = ax.get_legend_handles_labels()
    # A per-point colour encodes an outcome rather than a series, so that
    # series' legend entry has to be built from the encoding instead of from
    # its marks — which would otherwise show one arbitrary colour for all of
    # them.
    encoded: dict[str, str] = {}
    encoding_series = {s.name for s in spec.series if s.colors and s.meta.get("legend")}
    for s in spec.series:
        if s.name not in encoding_series:
            continue
        for label, color in zip(s.meta.get("legend", []), s.colors or []):
            encoded.setdefault(str(label), color)
    if encoded:
        keep = [(h, l) for h, l in zip(handles, labels) if l not in encoding_series]
        handles = [Patch(facecolor=c, edgecolor="none") for c in encoded.values()] \
            + [h for h, _ in keep]
        labels = list(encoded) + [l for _, l in keep]
    if len(labels) < 2:
        # One series is already named by the title; a legend would just repeat it.
        return
    ncol = 2 if len(labels) > 12 else 1
    ax.legend(handles, labels, frameon=False, fontsize=8, ncol=ncol,
              labelcolor=LIGHT["text_secondary"], title=spec.legend_title or None,
              title_fontsize=8, handlelength=1.4, handletextpad=0.6,
              loc="upper left", bbox_to_anchor=(1.008, 1.0))


def _caption(fig, ax, spec: ChartSpec) -> None:
    """Draw the note and footnote clear of everything the axes occupy.

    Measured from the drawn figure rather than guessed, because how much room
    the tick labels take depends on whether they were angled, and a guess puts
    the caption through the middle of them on exactly the charts that needed
    angling.
    """
    text = "  ".join(t for t in (spec.note, spec.footnote) if t)
    if not text:
        return
    fig.canvas.draw()
    box = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    wrapped = _wrap(text, 120)
    fig.text(box.x0, box.y0 - 0.035, wrapped, fontsize=7.8,
             color=LIGHT["text_muted"], va="top", ha="left", linespacing=1.45)


def _wrap(text: str, width: int) -> str:
    import textwrap

    return "\n".join(textwrap.wrap(text, width)) or text
