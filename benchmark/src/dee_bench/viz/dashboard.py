"""Build the interactive HTML dashboard.

Runs entirely from the parquet dataset, so it works on a partial run and can be
rebuilt at any time without re-running a benchmark:

    dee-bench viz <run_dir> [--only payback] [--format html,png,pdf]

The page is a single self-contained file (plotly inlined), theme-aware, with a
page per study, a page per optimization the run enabled, a run-by-run page for
any cell optimized continuously, a table view behind every chart, and a PNG and
a PDF of every chart rendered beside it.
"""

from __future__ import annotations

import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..schema import render_markdown
from ..store import connect
from . import overview, passes
from .assets import CSS, JS
from .charts import render as render_static
from .continuous import build as build_continuous
from .figures import figure
from .spec import ChartSpec, Kpi, Study, Table
from .studies import build_all, label_all
from .theme import DARK, FONT_STACK, LIGHT, MONO_STACK

# Where each page group sits in the navigation, and what it is called there.
GROUPS = [
    ("overview", "Run"),
    ("studies", "The seven studies"),
    ("optimizations", "Optimizations"),
    ("runs", "Run by run"),
]


def build_pages(run_dir: Path, con=None) -> tuple[list[Study], dict[str, Any]]:
    """Every page this run's results support, in navigation order."""
    con = con or connect(run_dir)
    # Once, before anything is built: the optimization pages read `cells` too,
    # and a variant that names its swept setting on one page but not another
    # would be two different things under one name.
    label_all(con)

    meta = _run_meta(run_dir, con)
    pages = [overview.build(con, meta)]
    pages += build_all(con, labelled=True)
    pages += passes.build(con)
    continuous = build_continuous(con)
    if continuous:
        pages.append(continuous)
    return pages, meta


def build(run_dir: Path, only: str | None = None, formats: set[str] | None = None) -> Path | None:
    """Build the dashboard. Returns the index path, or None if there is nothing yet."""
    formats = set(formats or {"html", "png", "pdf"})
    run_dir = Path(run_dir)
    if not (run_dir / "results").exists():
        return None

    con = connect(run_dir)
    pages, meta = build_pages(run_dir, con)
    if only:
        keys = {p.key for p in pages}
        chosen = [p for p in pages if p.key == only or p.key == f"pass-{only}"]
        if not chosen:
            raise ValueError(f"unknown page {only!r}; expected one of "
                             + ", ".join(sorted(keys)))
        pages = chosen

    out_dir = run_dir / "dashboard"
    charts_dir = out_dir / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)

    # The page offers a PNG and a PDF of every chart, so building the page
    # builds them: a download button pointing at a file that was not rendered
    # is worse than no button.
    static_formats = formats | ({"png", "pdf"} if "html" in formats else set())
    static: dict[str, dict[str, str]] = {}
    for page in pages:
        for chart in page.charts:
            try:
                static[chart.id] = render_static(chart, charts_dir, static_formats)
            except Exception as e:  # noqa: BLE001
                # One chart matplotlib cannot draw is a missing download
                # button, not a lost dashboard.
                print(f"warning: could not render {chart.id}: {type(e).__name__}: {e}",
                      file=sys.stderr)
                static[chart.id] = {}

    if "html" not in formats:
        return charts_dir

    index = out_dir / "index.html"
    index.write_text(_page(pages, static, meta, run_dir))
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
# HTML
# --------------------------------------------------------------------------


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _page(pages: list[Study], static: dict[str, dict[str, str]],
          meta: dict[str, Any], run_dir: Path) -> str:
    import plotly.offline

    figures: dict[str, Any] = {}
    panels = [_panel(page, static, figures) for page in pages]
    nav = _nav(pages)
    first = pages[0].key if pages else "overview"

    name = _esc(meta.get("name", run_dir.name))
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    chips = [f"Generated {generated}"]
    for label, key in (("dee", "dee_git_sha"), ("dag-bench", "dag_bench_git_sha")):
        if meta.get(key):
            chips.append(f"{label} {str(meta[key])[:12]}")
    for label, key in (("verbosity", "verbosity"), ("host", "host")):
        if meta.get(key):
            chips.append(f"{label} {meta[key]}")
    chip_html = "".join(f'<span class="chip">{_esc(c)}</span>' for c in chips)

    script = (JS
              .replace("__FIGURES__", json.dumps(figures))
              .replace("__LIGHT__", json.dumps(LIGHT))
              .replace("__DARK__", json.dumps(DARK))
              .replace("__FONT_JSON__", json.dumps(FONT_STACK))
              .replace("__FIRST__", json.dumps(first)))
    css = CSS.replace("__FONT__", FONT_STACK).replace("__MONO__", MONO_STACK)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>dee benchmark · {name}</title>
<style>{css}</style>
</head>
<body>
<div class="app">
  <aside class="side">
    <div class="brand">dee · bench</div>
    <div class="run-name">{name}</div>
    <nav role="tablist" aria-label="Dashboard pages">{nav}</nav>
    <div class="nav-foot">
      <button class="ghost" id="theme-toggle" type="button">Theme: system</button>
      <p>Charts also written to <code>dashboard/charts/</code> as PNG and PDF.
      Column documentation in <a href="schemas.md">schemas.md</a>.</p>
    </div>
  </aside>
  <main>
    <div class="page-head">
      <div class="eyebrow">dee benchmark</div>
      <h1>{name}</h1>
      <div class="chips">{chip_html}</div>
    </div>
    {"".join(panels)}
    <footer>
      <p>Results are parquet under <code>results/</code>. Query them directly:
      <code>duckdb -c "SELECT * FROM 'results/runs/**/*.parquet'"</code>.</p>
    </footer>
  </main>
</div>
<script>{plotly.offline.get_plotlyjs()}</script>
<script>{script}</script>
</body>
</html>
"""


def _nav(pages: list[Study]) -> str:
    out: list[str] = []
    for group, title in GROUPS:
        members = [p for p in pages if p.group == group]
        if not members:
            continue
        if len(GROUPS) > 1:
            out.append(f'<div class="nav-group">{_esc(title)}</div>')
        for page in members:
            pill = (f'<span class="pill">{page.number}</span>' if page.number
                    else f'<span class="pill">{len([c for c in page.charts if c.has_data])}</span>')
            out.append(
                f'<button class="nav-item" role="tab" data-panel="{_esc(page.key)}" '
                f'aria-selected="false">{pill}<span>{_esc(page.title)}</span></button>'
            )
    return "".join(out)


def _panel(page: Study, static: dict[str, dict[str, str]], figures: dict) -> str:
    body: list[str] = [
        '<div class="page-head" style="border:0;padding:0;margin-bottom:14px">',
        f'<h1 style="font-size:23px;margin:0">{_esc(page.title)}</h1>',
    ]
    if page.subtitle:
        body.append(f'<p class="sub" style="font-size:13.5px">{_esc(page.subtitle)}</p>')
    body.append("</div>")

    if page.question:
        body.append(f'<p class="question">{_esc(page.question)}</p>')
    if page.kpis:
        body.append(_kpis(page.kpis))

    charts = [c for c in page.charts if c.has_data]
    tables = [t for t in page.all_tables() if t.has_data]

    if not charts and not tables and not page.kpis:
        body.append(
            f'<div class="empty"><strong>Nothing to show yet.</strong>'
            f'<p>{_esc(page.empty_reason or "No data for this page.")}</p></div>'
        )
    else:
        if page.empty_reason:
            body.append(f'<div class="notice">{_esc(page.empty_reason)}</div>')
        for chart in charts:
            try:
                figures[chart.id] = figure(chart)
            except Exception as e:  # noqa: BLE001
                body.append(f'<div class="notice">Could not render '
                            f'<strong>{_esc(chart.title)}</strong>: '
                            f'{_esc(type(e).__name__)}: {_esc(e)}</div>')
                continue
            body.append(_chart(chart, static.get(chart.id, {})))
        for i, table in enumerate(tables):
            body.append(_table(table, open_first=(i == 0 and not charts)))

    return (f'<section class="panel" id="panel-{_esc(page.key)}" role="tabpanel">'
            f'{"".join(body)}</section>')


def _kpis(kpis: list[Kpi]) -> str:
    tiles = []
    for kpi in kpis:
        hint = f'<div class="kpi-hint">{_esc(kpi.hint)}</div>' if kpi.hint else ""
        # A value that is a phrase rather than a figure needs a smaller size, or
        # it wraps across three lines and the tile stops reading as a number.
        size = " long" if len(kpi.value) > 12 else ""
        tiles.append(
            f'<div class="kpi {_esc(kpi.tone)}">'
            f'<div class="kpi-label">{_esc(kpi.label)}</div>'
            f'<div class="kpi-value{size}">{_esc(kpi.value)}</div>{hint}</div>'
        )
    return f'<div class="kpis">{"".join(tiles)}</div>'


def _chart(chart: ChartSpec, links: dict[str, str]) -> str:
    # PNG first: it is what goes on a slide, and the commoner want of the two.
    buttons = "".join(
        f'<a class="dl" href="charts/{_esc(links[fmt])}" download '
        f'title="Download this chart as {fmt.upper()}">{fmt.upper()}</a>'
        for fmt in ("png", "pdf") if fmt in links
    )
    downloads = (f'<div class="dls"><span class="dls-label">Download</span>{buttons}</div>'
                 if buttons else "")
    subtitle = f'<p class="sub">{_esc(chart.subtitle)}</p>' if chart.subtitle else ""
    note = f'<p class="note">{_esc(chart.note)}</p>' if chart.note else ""
    footnote = (f'<p class="note" style="font-size:11.5px">{_esc(chart.footnote)}</p>'
                if chart.footnote else "")
    return f"""
<figure class="chart">
  <figcaption>
    <div>
      <p class="cap-title">{_esc(chart.title)}</p>
      {subtitle}
    </div>
    {downloads}
  </figcaption>
  <div class="plot" id="plot-{_esc(chart.id)}" data-chart="{_esc(chart.id)}"
       style="min-height:{chart.height}px"></div>
  {note}{footnote}
</figure>"""


def _table(table: Table, open_first: bool = False) -> str:
    numeric = set(table.numeric)
    head = "".join(
        f'<th class="{"num" if i in numeric else ""}">{_esc(c)}</th>'
        for i, c in enumerate(table.columns)
    )
    rows = "".join(
        "<tr>" + "".join(
            f'<td class="{"num" if i in numeric else ""}">{_esc(v)}</td>'
            for i, v in enumerate(row)
        ) + "</tr>"
        for row in table.rows
    )
    note = f'<p class="table-note">{_esc(table.note)}</p>' if table.note else ""
    state = " open" if open_first else ""
    return f"""
<div class="table-block{state}">
  <button class="table-toggle" type="button" aria-expanded="{str(bool(open_first)).lower()}">
    <span class="caret">&#9656;</span>
    <span>{_esc(table.title)}</span>
    <span class="pill">{len(table.rows)}</span>
  </button>
  <div class="table-body">
    {note}
    <div class="table-wrap">
      <table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>
    </div>
  </div>
</div>"""
