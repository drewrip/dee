"""Plotly figures for a :class:`ChartSpec`, in both colourways.

The page is theme-aware, and a chart has to move with it — but a chart whose
colours were computed in the browser could drift from the png beside it. So
each figure carries its light colourway inline and its dark one alongside, both
derived here from the same palette the static renderer uses, and the page only
ever swaps between the two.
"""

from __future__ import annotations

from typing import Any

from .spec import BAR_KINDS, ChartSpec, Series
from .theme import (DARK, LIGHT, SEQUENTIAL, SEQUENTIAL_DARK, STATUS, STATUS_DARK,
                    series_color)

_DASH = {"dash": "dash", "dot": "dot", None: "solid", "": "solid"}
# Status hues are the same idea on either surface, but not the same step.
_STATUS_PAIR = {STATUS[k]: STATUS_DARK[k] for k in STATUS}


def figure(spec: ChartSpec) -> dict[str, Any]:
    """A plotly figure dict, plus the per-trace overrides for dark mode."""
    if spec.kind == "heatmap":
        return _heatmap(spec)

    traces: list[dict[str, Any]] = []
    dark: list[dict[str, Any]] = []
    for i, s in enumerate(spec.series):
        kind = s.kind or spec.kind
        if kind == "span":
            trace, override = _span_trace(spec, s, i)
        elif kind in BAR_KINDS:
            trace, override = _bar_trace(spec, s, i)
        else:
            trace, override = _line_trace(spec, s, i, kind)
        if s.colors and s.meta.get("legend"):
            # One legend entry per colour the encoding uses, rather than one
            # for the whole series showing whichever colour came first.
            trace["showlegend"] = False
        traces.append(trace)
        dark.append(override)

    legend_traces, legend_dark = _encoding_legend(spec)
    traces += legend_traces
    dark += legend_dark

    layout = _layout(spec)
    if spec.horizontal or spec.kind == "span":
        # Lanes read top down, in the order the builder put them in: a ranking
        # with its largest bar at the top, a schedule with its first node
        # there. Left to plotly the first category lands at the bottom, and
        # dropping a None value would reorder the rest — so the order is
        # stated rather than inferred, and matches the png beside it.
        layout["yaxis"].update({
            "categoryorder": "array",
            "categoryarray": list(reversed(_lanes(spec))),
        })
    if legend_traces:
        layout["showlegend"] = True
    return {"data": traces, "layout": layout, "dark": dark}


def _lanes(spec: ChartSpec) -> list[str]:
    """Every lane the chart draws, in first-seen order across all series."""
    seen: dict[str, None] = {}
    for s in spec.series:
        values = s.meta.get("lane", []) if spec.kind == "span" else s.x
        for value in values:
            seen.setdefault(str(value), None)
    return list(seen)


def _encoding_legend(spec: ChartSpec) -> tuple[list[dict], list[dict]]:
    """Legend entries for a per-point colour encoding.

    Plotly gives a trace one legend swatch, so a series whose marks are
    coloured by outcome would advertise a single arbitrary colour. These carry
    no data: they exist to name the encoding.
    """
    seen: dict[str, str] = {}
    for s in spec.series:
        if not (s.colors and s.meta.get("legend")):
            continue
        for label, color in zip(s.meta["legend"], s.colors):
            seen.setdefault(str(label), color)
    traces, dark = [], []
    for label, color in seen.items():
        traces.append({
            "type": "scatter", "mode": "markers", "name": label,
            "x": [None], "y": [None], "showlegend": True,
            "marker": {"size": 9, "color": color},
            "hoverinfo": "skip",
        })
        dark.append({"marker": _STATUS_PAIR.get(color, color)})
    return traces, dark


# --------------------------------------------------------------------------


def _colors(s: Series, i: int) -> tuple[Any, Any]:
    """This series' light and dark colours, whatever form it declared them in."""
    if s.colors:
        return (list(s.colors),
                [_STATUS_PAIR.get(c, c) for c in s.colors])
    if s.color:
        return s.color, _STATUS_PAIR.get(s.color, s.color)
    return series_color(s.name, i, "light"), series_color(s.name, i, "dark")


def _hover(spec: ChartSpec, s: Series, keep: list[int]) -> tuple[str, list[list[Any]] | None]:
    """A hover template, plus the per-point extras it reads.

    `keep` is the indices that survived the None filter, so the extras stay
    aligned with the points actually drawn.
    """
    value = "%{x:.4g}" if spec.horizontal else "%{y:.4g}"
    other = "%{y}" if spec.horizontal else "%{x}"
    parts = [f"<b>{s.name}</b>", other, value]
    keys = [k for k in s.meta if k != "legend"]
    customdata = None
    if keys:
        customdata = [[_cell(s.meta[k], j) for k in keys] for j in keep]
        parts += [f"{k}: %{{customdata[{n}]}}" for n, k in enumerate(keys)]
    return "<br>".join(parts) + "<extra></extra>", customdata


def _cell(values: list[Any], index: int) -> Any:
    value = values[index] if index < len(values) else None
    return "-" if value is None else value


def _line_trace(spec: ChartSpec, s: Series, i: int, kind: str) -> tuple[dict, dict]:
    keep = [j for j, y in enumerate(s.y) if y is not None]
    xs = [s.x[j] for j in keep]
    ys = [s.y[j] for j in keep]
    light, dark = _colors(s, i)
    hover, customdata = _hover(spec, s, keep)

    # Markers per point are signal on a handful of measurements and noise on a
    # 100ms-interval timeseries, where they also break the line up visually.
    dense = len(xs) > 25
    if kind == "points":
        mode = "markers"
    elif kind == "scatter" or dense:
        mode = "lines"
    else:
        mode = "lines+markers"

    trace: dict[str, Any] = {
        "type": "scatter", "mode": mode, "name": s.name,
        "x": xs, "y": ys,
        "line": {"width": 2, "color": light if not isinstance(light, list) else None,
                 "dash": _DASH.get(s.dash, "solid"),
                 "shape": "hv" if kind == "step" else "linear"},
        "marker": {"size": 9, "color": light,
                   "line": {"width": 1.6, "color": LIGHT["surface"]}},
        "hovertemplate": hover,
        "showlegend": bool(s.legend),
    }
    if customdata:
        trace["customdata"] = customdata
    if kind == "area" or s.fill:
        trace["fill"] = "tozeroy"
        trace["fillcolor"] = _translucent(light)
    if s.error:
        lo, hi = _arms(s, keep)
        trace["error_y"] = {"type": "data", "array": hi, "arrayminus": lo,
                            "visible": True, "thickness": 1.2, "width": 4,
                            "color": light if not isinstance(light, list) else None}
    if s.labels:
        trace["mode"] = mode + "+text"
        trace["text"] = [s.labels[j] if j < len(s.labels) else "" for j in keep]
        trace["textposition"] = "top center"
    return trace, {"line": dark if not isinstance(dark, list) else None,
                   "marker": dark, "surface": DARK["surface"],
                   "fill": _translucent(dark) if trace.get("fill") else None}


def _bar_trace(spec: ChartSpec, s: Series, i: int) -> tuple[dict, dict]:
    keep = [j for j, y in enumerate(s.y) if y is not None]
    cats = [s.x[j] for j in keep]
    values = [s.y[j] for j in keep]
    light, dark = _colors(s, i)
    hover, customdata = _hover(spec, s, keep)

    trace: dict[str, Any] = {
        "type": "bar", "name": s.name,
        "orientation": "h" if spec.horizontal else "v",
        "marker": {"color": light, "line": {"width": 0}},
        "hovertemplate": hover,
        "showlegend": bool(s.legend),
    }
    if spec.horizontal:
        trace["x"], trace["y"] = values, cats
    else:
        trace["x"], trace["y"] = cats, values
    if customdata:
        trace["customdata"] = customdata
    if s.error and not spec.horizontal:
        lo, hi = _arms(s, keep)
        trace["error_y"] = {"type": "data", "array": hi, "arrayminus": lo,
                            "visible": True, "thickness": 1.2, "width": 4,
                            "color": LIGHT["text_muted"]}
    if spec.value_labels and len(values) <= 40:
        trace["text"] = [spec.value_fmt.format(v) for v in values]
        trace["textposition"] = "outside"
        trace["textfont"] = {"size": 10, "color": LIGHT["text_secondary"]}
        trace["cliponaxis"] = False
    return trace, {"marker": dark, "text": DARK["text_secondary"],
                   "error": DARK["text_muted"]}


def _span_trace(spec: ChartSpec, s: Series, i: int) -> tuple[dict, dict]:
    """A span is a horizontal bar that starts somewhere other than zero."""
    ends = s.meta.get("end", [])
    lanes = s.meta.get("lane", [])
    light, dark = _colors(s, i)
    starts, widths, rows = [], [], []
    for j, start in enumerate(s.x):
        if j >= len(ends) or start is None or ends[j] is None or j >= len(lanes):
            continue
        starts.append(float(start))
        widths.append(max(float(ends[j]) - float(start), 0.0))
        rows.append(str(lanes[j]))
    trace = {
        "type": "bar", "orientation": "h", "name": s.name,
        "x": widths, "y": rows, "base": starts,
        "marker": {"color": light, "line": {"width": 0}},
        "hovertemplate": f"<b>{s.name}</b><br>%{{y}}<br>%{{base:.4g}} → "
                         "%{base:.4g}<extra></extra>",
        "showlegend": bool(s.legend),
    }
    return trace, {"marker": dark}


def _arms(s: Series, keep: list[int]) -> tuple[list[float], list[float]]:
    lo, hi = [], []
    for j in keep:
        e = s.error[j] if s.error and j < len(s.error) else None
        if e is None:
            lo.append(0.0), hi.append(0.0)
        elif isinstance(e, (tuple, list)):
            lo.append(float(e[0])), hi.append(float(e[1]))
        else:
            lo.append(float(e)), hi.append(float(e))
    return lo, hi


def _translucent(color: Any) -> str | None:
    if not isinstance(color, str) or not color.startswith("#") or len(color) != 7:
        return None
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},0.13)"


def _scale(steps: list[str]) -> list[list[Any]]:
    last = len(steps) - 1
    return [[i / last, c] for i, c in enumerate(steps)]


def _heatmap(spec: ChartSpec) -> dict[str, Any]:
    trace = {
        "type": "heatmap",
        "z": spec.matrix,
        "x": spec.x_ticks, "y": spec.y_ticks,
        "colorscale": _scale(SEQUENTIAL),
        "hoverongaps": False,
        "xgap": 2, "ygap": 2,
        "colorbar": {"title": {"text": spec.z_label, "side": "right"},
                     "thickness": 10, "outlinewidth": 0, "len": 0.9},
        "hovertemplate": "%{y}<br>%{x}<br>%{z:.4g}<extra></extra>",
    }
    if len(spec.y_ticks) * max(len(spec.x_ticks), 1) <= 200:
        trace["text"] = spec.z_text or [
            [("" if v is None else spec.z_fmt.format(v)) for v in row]
            for row in spec.matrix
        ]
        trace["texttemplate"] = "%{text}"
        trace["textfont"] = {"size": 10}
    layout = _layout(spec)
    # Both axes are label sets, not numbers. Left as linear, plotly reads the
    # tick strings as coordinates it cannot place and draws an empty grid.
    layout["xaxis"].update({"type": "category", "showgrid": False})
    layout["yaxis"].update({"type": "category", "showgrid": False,
                            "autorange": "reversed"})
    layout["showlegend"] = False
    return {"data": [trace], "layout": layout,
            "dark": [{"colorscale": _scale(SEQUENTIAL_DARK)}]}


def _layout(spec: ChartSpec) -> dict[str, Any]:
    x_title = spec.x_label
    y_title = spec.y_label
    layout: dict[str, Any] = {
        "barmode": "stack" if spec.stacked else "group",
        "bargap": 0.34,
        "bargroupgap": 0.1,
        "margin": {"l": 72, "r": 20, "t": 14, "b": 64},
        "height": spec.height,
        # Crosshair on a trend, per-mark tooltip on anything discrete.
        "hovermode": "closest" if spec.kind in ("points", "heatmap", "span") or spec.is_bar
                     else "x unified",
        "showlegend": sum(1 for s in spec.series if s.legend) > 1,
        "legend": {"orientation": "h", "y": -0.2, "x": 0,
                   "title": {"text": spec.legend_title}},
        # A span's x axis carries time, not labels — the same way a horizontal
        # bar's does — so it types the same way even though its `y` is a lane.
        "xaxis": {"title": {"text": x_title},
                  "type": _axis(spec.x_type, spec.horizontal or spec.kind == "span"),
                  "showgrid": spec.horizontal or spec.kind == "span",
                  "zeroline": False, "automargin": True},
        "yaxis": {"title": {"text": y_title},
                  "type": "category" if (spec.horizontal or spec.kind == "span")
                          else spec.y_type,
                  "showgrid": not (spec.horizontal or spec.kind == "span"),
                  "zeroline": False, "automargin": True},
    }
    shapes, annotations = [], []
    references = list(spec.hlines) + ([(spec.hline, spec.hline_label)]
                                      if spec.hline is not None else [])
    for value, label in references:
        if value is None:
            continue
        shapes.append({"type": "line", "xref": "paper", "x0": 0, "x1": 1,
                       "yref": "y", "y0": value, "y1": value,
                       "line": {"dash": "dash", "width": 1}, "layer": "below"})
        if label:
            annotations.append({"xref": "paper", "x": 1, "y": value, "yref": "y",
                                "text": label, "showarrow": False, "xanchor": "right",
                                "yanchor": "bottom", "font": {"size": 10}})
    for value, label in spec.vlines:
        shapes.append({"type": "line", "yref": "paper", "y0": 0, "y1": 1,
                       "xref": "x", "x0": value, "x1": value,
                       "line": {"dash": "dot", "width": 1.4, "color": STATUS["warning"]},
                       "layer": "below", "name": "event"})
        if label:
            annotations.append({"xref": "x", "x": value, "yref": "paper", "y": 1,
                                "text": label, "showarrow": False, "xanchor": "left",
                                "yanchor": "top", "xshift": 5,
                                "font": {"size": 10, "color": STATUS["warning"]},
                                "name": "event"})
    for lo, hi, label in spec.bands:
        shapes.append({"type": "rect", "xref": "paper", "x0": 0, "x1": 1,
                       "yref": "y", "y0": lo, "y1": hi, "line": {"width": 0},
                       "fillcolor": "rgba(120,119,111,0.08)", "layer": "below"})
    for x, y, text in spec.annotations:
        annotations.append({"x": x, "y": y, "text": text, "showarrow": False,
                            "yshift": 12, "font": {"size": 10}})
    if shapes:
        layout["shapes"] = shapes
    if annotations:
        layout["annotations"] = annotations
    return layout


def _axis(x_type: str, horizontal: bool) -> str:
    if horizontal:
        return "linear"
    return {"category": "category", "linear": "linear", "log": "log"}.get(x_type, "category")
