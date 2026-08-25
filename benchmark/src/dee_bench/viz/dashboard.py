"""Build the interactive HTML dashboard.

Runs entirely from the parquet dataset, so it works on a partial run and can be
rebuilt at any time without re-running a benchmark:

    dee-bench viz <run_dir> [--only payback] [--format html,png,pdf]

The page is a single self-contained file (plotly inlined), theme-aware, with a
tab per study, a table view behind every chart, and a link from each chart to
the static png/pdf of the same spec.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..schema import render_markdown
from ..store import connect
from .charts import render as render_static
from .spec import ChartSpec, Study
from .studies import build_all
from .theme import DARK, DARK_SERIES, LIGHT, LIGHT_SERIES, series_color


def build(run_dir: Path, only: str | None = None, formats: set[str] | None = None) -> Path | None:
    """Build the dashboard. Returns the index path, or None if there is nothing yet."""
    formats = formats or {"html", "png", "pdf"}
    run_dir = Path(run_dir)
    if not (run_dir / "results").exists():
        return None

    con = connect(run_dir)
    studies = build_all(con)
    if only:
        studies = [s for s in studies if s.key == only]
        if not studies:
            raise ValueError(f"unknown study {only!r}; expected one of "
                             + ", ".join(s.key for s in build_all(con)))

    out_dir = run_dir / "dashboard"
    charts_dir = out_dir / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)

    static: dict[str, dict[str, str]] = {}
    for study in studies:
        for chart in study.charts:
            static[chart.id] = render_static(chart, charts_dir, formats)

    if "html" not in formats:
        return charts_dir

    meta = _run_meta(run_dir, con)
    index = out_dir / "index.html"
    index.write_text(_page(studies, static, meta, run_dir))
    (out_dir / "schemas.md").write_text(render_markdown())
    return index


def _run_meta(run_dir: Path, con) -> dict[str, Any]:
    meta: dict[str, Any] = {"name": run_dir.name}
    for fname in ("run.json", "provenance.json"):
        path = run_dir / fname
        if path.exists():
            try:
                meta.update(json.loads(path.read_text()))
            except json.JSONDecodeError:
                pass
    try:
        meta["cells_done"] = con.sql("SELECT count(DISTINCT cell_id) FROM runs").fetchone()[0]
        meta["measured_runs"] = con.sql(
            "SELECT count(*) FROM runs WHERE phase='measure'"
        ).fetchone()[0]
    except Exception:  # noqa: BLE001
        pass
    return meta


# --------------------------------------------------------------------------
# plotly figure construction
# --------------------------------------------------------------------------


def _figure(spec: ChartSpec) -> dict[str, Any]:
    """A plotly figure dict for `spec`, in both light and dark colorways."""
    traces = []
    for i, s in enumerate(spec.series):
        light = series_color(s.name, i, "light")
        dark = series_color(s.name, i, "dark")
        hover = "<b>%{fullData.name}</b><br>%{x}<br>%{y:.4g}<extra></extra>"
        if spec.kind in ("line", "scatter"):
            # Markers per point are noise on a dense timeseries; the hover
            # layer still exposes every sample.
            dense = len(s.x) > 25
            traces.append({
                "type": "scatter", "mode": "lines" if dense else "lines+markers",
                "name": s.name,
                "x": s.x, "y": s.y,
                "line": {"width": 2, "shape": "linear", "color": light},
                "marker": {"size": 9, "color": light,
                           "line": {"width": 2, "color": LIGHT["surface"]}},
                "hovertemplate": hover,
                "customdata": [dark] * max(len(s.x), 1),
            })
        else:
            traces.append({
                "type": "bar", "name": s.name, "x": s.x, "y": s.y,
                "marker": {"color": light, "line": {"width": 0}},
                "hovertemplate": hover,
                "customdata": [dark] * max(len(s.x), 1),
            })

    layout: dict[str, Any] = {
        "barmode": "group",
        "bargap": 0.35,
        "bargroupgap": 0.12,
        "margin": {"l": 64, "r": 24, "t": 12, "b": 72},
        "height": 380,
        # Crosshair on line charts, per-mark tooltip on bars.
        "hovermode": "x unified" if spec.kind in ("line", "scatter") else "closest",
        "showlegend": len(spec.series) > 1,
        "legend": {"orientation": "h", "y": -0.22, "x": 0},
        "xaxis": {"title": {"text": spec.x_label}, "type": _axis_type(spec.x_type),
                  "showgrid": False, "zeroline": False},
        "yaxis": {"title": {"text": spec.y_label}, "type": spec.y_type,
                  "gridwidth": 1, "zeroline": False},
    }
    if spec.hline is not None:
        layout["shapes"] = [{
            "type": "line", "xref": "paper", "x0": 0, "x1": 1,
            "yref": "y", "y0": spec.hline, "y1": spec.hline,
            "line": {"dash": "dash", "width": 1},
        }]
        layout["annotations"] = [{
            "xref": "paper", "x": 1, "y": spec.hline, "yref": "y",
            "text": spec.hline_label, "showarrow": False,
            "xanchor": "right", "yanchor": "bottom", "font": {"size": 10},
        }]
    return {"data": traces, "layout": layout}


def _axis_type(x_type: str) -> str:
    return {"category": "category", "linear": "linear", "log": "log"}[x_type]


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------


def _page(studies: list[Study], static: dict[str, dict[str, str]],
          meta: dict[str, Any], run_dir: Path) -> str:
    import plotly.offline

    plotly_js = plotly.offline.get_plotlyjs()

    tabs, panels, figures = [], [], {}
    for i, study in enumerate(studies):
        active = " active" if i == 0 else ""
        tabs.append(
            f'<button class="tab{active}" data-panel="{study.key}" role="tab" '
            f'aria-selected="{"true" if i == 0 else "false"}">'
            f'<span class="tab-num">{study.number or "-"}</span>{html.escape(study.title)}</button>'
        )
        panels.append(_panel(study, static, figures, first=(i == 0)))

    stats = [
        ("Cells with results", meta.get("cells_done", "-")),
        ("Measured runs", meta.get("measured_runs", "-")),
        ("Verbosity", meta.get("verbosity", "-")),
        ("Host", meta.get("host", "-")),
    ]
    stat_html = "".join(
        f'<div class="stat"><div class="stat-label">{html.escape(str(k))}</div>'
        f'<div class="stat-value">{html.escape(str(v))}</div></div>'
        for k, v in stats
    )

    provenance = []
    for label, key in (("dee", "dee_git_sha"), ("dag-bench", "dag_bench_git_sha")):
        sha = meta.get(key)
        if sha:
            provenance.append(f"{label} <code>{html.escape(sha[:12])}</code>")
    prov_html = " · ".join(provenance) or "provenance unavailable"

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>dee benchmark · {html.escape(str(meta.get("name", run_dir.name)))}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="shell">
  <header>
    <div class="eyebrow">dee benchmark</div>
    <h1>{html.escape(str(meta.get("name", run_dir.name)))}</h1>
    <p class="sub">Generated {generated} · {prov_html}</p>
    <div class="stats">{stat_html}</div>
  </header>
  <nav class="tabs" role="tablist">{"".join(tabs)}</nav>
  <main>{"".join(panels)}</main>
  <footer>
    <p>Results are parquet under <code>results/</code>. Query them directly:
    <code>duckdb -c "SELECT * FROM 'results/runs/**/*.parquet'"</code>.
    Column-by-column documentation is in <a href="schemas.md">schemas.md</a>.</p>
  </footer>
</div>
<script>{plotly_js}</script>
<script>
const FIGURES = {json.dumps(figures)};
const LIGHT_AXES = {json.dumps(LIGHT)};
const DARK_AXES = {json.dumps(DARK)};

function isDark() {{
  const stamped = document.documentElement.dataset.theme;
  if (stamped) return stamped === 'dark';
  return window.matchMedia('(prefers-color-scheme: dark)').matches;
}}

function themed(fig, dark) {{
  const t = dark ? DARK_AXES : LIGHT_AXES;
  const copy = JSON.parse(JSON.stringify(fig));
  copy.data.forEach(tr => {{
    // customdata carries this trace's dark-mode step of the same hue.
    const c = dark && tr.customdata ? tr.customdata[0] : null;
    if (c) {{
      if (tr.line) tr.line.color = c;
      if (tr.marker) tr.marker.color = c;
    }}
    if (tr.marker && tr.marker.line) tr.marker.line.color = t.surface;
  }});
  Object.assign(copy.layout, {{
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: {{ color: t.text_secondary, size: 11,
            family: 'ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif' }},
  }});
  ['xaxis','yaxis'].forEach(a => {{
    copy.layout[a] = Object.assign({{}}, copy.layout[a], {{
      gridcolor: t.grid, linecolor: t.border, tickfont: {{ color: t.text_secondary }},
    }});
  }});
  (copy.layout.shapes || []).forEach(s => s.line.color = t.text_muted);
  (copy.layout.annotations || []).forEach(a => a.font.color = t.text_muted);
  return copy;
}}

function drawAll() {{
  const dark = isDark();
  Object.entries(FIGURES).forEach(([id, fig]) => {{
    const el = document.getElementById('plot-' + id);
    if (!el) return;
    const f = themed(fig, dark);
    Plotly.react(el, f.data, f.layout, {{responsive: true, displaylogo: false,
      modeBarButtonsToRemove: ['lasso2d','select2d','autoScale2d']}});
  }});
}}

document.querySelectorAll('.tab').forEach(tab => {{
  tab.addEventListener('click', () => {{
    document.querySelectorAll('.tab').forEach(t => {{
      t.classList.remove('active'); t.setAttribute('aria-selected','false');
    }});
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    tab.classList.add('active'); tab.setAttribute('aria-selected','true');
    document.getElementById('panel-' + tab.dataset.panel).classList.add('active');
    // Plotly cannot size a chart inside a hidden panel, so resize on reveal.
    window.dispatchEvent(new Event('resize'));
  }});
}});

document.querySelectorAll('.table-toggle').forEach(btn => {{
  btn.addEventListener('click', () => {{
    const wrap = document.getElementById(btn.dataset.table);
    const open = wrap.classList.toggle('open');
    btn.textContent = open ? 'Hide table' : 'Show table';
  }});
}});

drawAll();
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', drawAll);
</script>
</body>
</html>
"""


def _panel(study: Study, static: dict[str, dict[str, str]], figures: dict,
           first: bool = False) -> str:
    active = " active" if first else ""
    body: list[str] = []

    body.append(f'<p class="question">{html.escape(study.question)}</p>')

    if not study.has_content:
        body.append(
            f'<div class="empty"><strong>Nothing to show yet.</strong>'
            f'<p>{html.escape(study.empty_reason or "No data for this study.")}</p></div>'
        )
    else:
        if study.empty_reason:
            body.append(f'<div class="notice">{html.escape(study.empty_reason)}</div>')
        for chart in study.charts:
            if not chart.has_data:
                continue
            figures[chart.id] = _figure(chart)
            links = static.get(chart.id, {})
            link_html = " ".join(
                f'<a class="dl" href="charts/{html.escape(fname)}" download>{fmt.upper()}</a>'
                for fmt, fname in sorted(links.items())
            )
            note = (f'<p class="note">{html.escape(chart.note)}</p>' if chart.note else "")
            body.append(f"""
<figure class="chart">
  <figcaption>
    <div>
      <h3>{html.escape(chart.title)}</h3>
      <p class="sub">{html.escape(chart.subtitle)}</p>
    </div>
    <div class="dls">{link_html}</div>
  </figcaption>
  <div class="plot" id="plot-{html.escape(chart.id)}"></div>
  {note}
</figure>""")

        if study.table_rows:
            table_id = f"table-{study.key}"
            head = "".join(f"<th>{html.escape(str(c))}</th>" for c in study.table_columns)
            rows = "".join(
                "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>"
                for row in study.table_rows
            )
            body.append(f"""
<div class="table-block">
  <button class="table-toggle" data-table="{table_id}">Show table</button>
  <div class="table-wrap" id="{table_id}">
    <table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>
  </div>
</div>""")

    return (f'<section class="panel{active}" id="panel-{study.key}" role="tabpanel">'
            f'<h2>{html.escape(study.title)}</h2>{"".join(body)}</section>')


_CSS = """
:root {
  --surface: #fcfcfb; --surface-2: #f4f4f2;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #78776f;
  --grid: #e6e6e2; --border: #dcdcd6; --accent: #2a78d6;
  color-scheme: light;
}
:root:not([data-theme="light"]) { }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --surface: #1a1a19; --surface-2: #232322;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #8f8e85;
    --grid: #333331; --border: #3a3a38; --accent: #3987e5;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --surface: #1a1a19; --surface-2: #232322;
  --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #8f8e85;
  --grid: #333331; --border: #3a3a38; --accent: #3987e5;
  color-scheme: dark;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--surface); color: var(--text-primary);
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 14px; line-height: 1.55;
}
.shell { max-width: 1180px; margin: 0 auto; padding: 32px 24px 64px; }
header { border-bottom: 1px solid var(--border); padding-bottom: 20px; }
.eyebrow {
  text-transform: uppercase; letter-spacing: .09em; font-size: 11px;
  font-weight: 600; color: var(--text-muted);
}
h1 { margin: 6px 0 4px; font-size: 30px; letter-spacing: -0.02em; }
h2 { font-size: 20px; margin: 0 0 4px; letter-spacing: -0.01em; }
h3 { font-size: 14px; margin: 0; font-weight: 600; }
.sub { color: var(--text-secondary); font-size: 12px; margin: 2px 0 0; }
.stats { display: flex; flex-wrap: wrap; gap: 28px; margin-top: 18px; }
.stat-label { font-size: 11px; color: var(--text-muted); text-transform: uppercase;
              letter-spacing: .06em; }
.stat-value { font-size: 20px; font-weight: 600; font-variant-numeric: tabular-nums; }
.tabs { display: flex; flex-wrap: wrap; gap: 4px; margin: 22px 0 26px;
        border-bottom: 1px solid var(--border); }
.tab {
  background: none; border: 0; border-bottom: 2px solid transparent;
  padding: 9px 12px; font: inherit; font-size: 13px; color: var(--text-secondary);
  cursor: pointer; display: inline-flex; align-items: center; gap: 7px;
}
.tab:hover { color: var(--text-primary); }
.tab.active { color: var(--text-primary); border-bottom-color: var(--accent); font-weight: 600; }
.tab-num {
  display: inline-flex; align-items: center; justify-content: center;
  width: 18px; height: 18px; border-radius: 999px; background: var(--surface-2);
  font-size: 10px; font-weight: 600; color: var(--text-muted);
}
.tab.active .tab-num { background: var(--accent); color: #fff; }
.panel { display: none; }
.panel.active { display: block; }
.question { color: var(--text-secondary); font-size: 14px; margin: 0 0 22px; max-width: 74ch; }
.chart { margin: 0 0 30px; border: 1px solid var(--border); border-radius: 10px;
         padding: 16px 16px 8px; background: var(--surface); }
figcaption { display: flex; justify-content: space-between; align-items: flex-start;
             gap: 16px; margin-bottom: 8px; }
.dls { display: flex; gap: 6px; flex-shrink: 0; }
.dl {
  font-size: 10px; font-weight: 600; letter-spacing: .05em; text-decoration: none;
  color: var(--text-secondary); border: 1px solid var(--border); border-radius: 5px;
  padding: 3px 7px;
}
.dl:hover { color: var(--text-primary); border-color: var(--text-muted); }
.plot { width: 100%; min-height: 380px; }
.note { font-size: 12px; color: var(--text-muted); margin: 4px 0 8px; max-width: 80ch; }
.empty, .notice {
  border: 1px dashed var(--border); border-radius: 10px; padding: 20px;
  color: var(--text-secondary); background: var(--surface-2);
}
.empty p, .notice { margin: 6px 0 0; font-size: 13px; }
.notice { margin-bottom: 20px; border-style: solid; }
.table-block { margin-top: 8px; }
.table-toggle {
  background: none; border: 1px solid var(--border); border-radius: 6px;
  padding: 5px 11px; font: inherit; font-size: 12px; color: var(--text-secondary);
  cursor: pointer;
}
.table-toggle:hover { color: var(--text-primary); }
.table-wrap { display: none; margin-top: 12px; overflow-x: auto;
              border: 1px solid var(--border); border-radius: 8px; }
.table-wrap.open { display: block; }
table { border-collapse: collapse; width: 100%; font-size: 12px;
        font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 7px 12px; border-bottom: 1px solid var(--grid);
         white-space: nowrap; }
th { background: var(--surface-2); font-weight: 600; color: var(--text-secondary);
     position: sticky; top: 0; }
tbody tr:last-child td { border-bottom: 0; }
footer { margin-top: 40px; padding-top: 18px; border-top: 1px solid var(--border);
         color: var(--text-muted); font-size: 12px; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px;
       background: var(--surface-2); padding: 1px 5px; border-radius: 4px; }
a { color: var(--accent); }
@media (max-width: 760px) {
  .shell { padding: 20px 14px 48px; }
  figcaption { flex-direction: column; }
}
"""
