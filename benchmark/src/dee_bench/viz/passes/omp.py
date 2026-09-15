"""OMP: which nodes centrality put in front of the search, and what it kept.

Where HMP ranks by measured cost, OMP ranks by *structure* — how central a node
is in the graph — and then enumerates materialization plans over the top few.
So the page's first question is whether the centrality that chose the
candidates agreed with the plan that won.
"""

from __future__ import annotations

from ..query import fmt_num, median, short_node
from ..spec import ChartSpec, Kpi, Series, Study, Table
from ..theme import STATUS
from ._common import (MAX_DETAILED_CELLS, iterations, page, pass_cells,
                      search_trace, settings_table)

SETTINGS = ["omp_top", "omp_node_centrality", "omp_exhaust", "omp_use_pushdown",
            "trial_resume", "trial_budget_eps", "trial_reuse"]


def build(con) -> Study | None:
    cells = pass_cells(con, "omp")
    if not cells:
        return None
    traces = iterations(con, "omp")

    study = page("omp", question=(
        "OMP ranks candidate nodes by centrality and enumerates plans over the top "
        "few. Did the ranking put the right nodes in front of the search, and what "
        "did enumerating them find?"
    ))
    study.kpis = _kpis(cells)
    study.charts.append(_improvement(cells))
    for cell in cells[:MAX_DETAILED_CELLS]:
        ranking = _ranking(cell)
        if ranking:
            study.charts.append(ranking)
    for cell in cells[:MAX_DETAILED_CELLS]:
        trace = traces.get(cell["cell_id"])
        if trace:
            study.charts.append(_trace_chart(cell, trace))

    study.tables = [t for t in (
        _summary_table(cells),
        settings_table(cells, SETTINGS, key="omp-settings"),
    ) if t]
    return study


def _kpis(cells: list[dict]) -> list[Kpi]:
    changes = [c["changes_applied"] for c in cells if c["changes_applied"] is not None]
    gains = [-(c["detail"].get("opt_change") or 0.0) for c in cells
             if c["detail"].get("opt_change") is not None]
    exhaustive = [c for c in cells if c["detail"].get("early_termination") is False]
    return [
        Kpi("Cells running OMP", str(len(cells))),
        Kpi("Materializations applied", str(sum(changes)),
            f"Median {median(changes):.0f} per cell." if changes else ""),
        Kpi("Best improvement found",
            f"{max(gains):.1%}" if gains else "-",
            "Against the plan OMP measured as its baseline, not against the "
            "unoptimized cell — see study 2 for that.",
            tone="good" if gains and max(gains) > 0 else "warning"),
        Kpi("Searched exhaustively", f"{len(exhaustive)}/{len(cells)}",
            "Cells that ran with early termination off, evaluating every plan fully."),
    ]


def _improvement(cells: list[dict]) -> ChartSpec:
    labels = [c["short"] for c in cells]
    return ChartSpec(
        id="omp-improvement", kind="grouped_bar",
        title="What enumeration found, against the plan it started from",
        subtitle="OMP's own baseline and best measurements, as it recorded them",
        y_label="Runtime (s)",
        series=[
            Series(name="baseline plan", x=labels,
                   y=[(c["detail"].get("baseline_value") or 0) / 1000.0 or None
                      for c in cells]),
            Series(name="best plan found", x=labels,
                   y=[(c["detail"].get("best_value") or 0) / 1000.0 or None
                      for c in cells]),
        ],
        value_labels=True, value_fmt="{:.2f}",
        note="These are the optimizer's own single measurements, taken while searching. "
             "They are not repeated, so treat them as what the search believed rather "
             "than as the study's runtime figures.",
    )


def _ranking(cell: dict) -> ChartSpec | None:
    ranked = cell["detail"].get("candidates_ranked") or []
    if not ranked:
        return None
    best = {short_node(n) for n in (cell["detail"].get("best_plan") or [])}
    nodes = [short_node(r.get("node_id")) for r in ranked][:18]
    scores = [r.get("score") for r in ranked][:18]
    centrality = cell["detail"].get("centrality") or "centrality"
    return ChartSpec(
        id=f"omp-ranking-{cell['slug']}", kind="hbar",
        title=f"Candidate ranking — {cell['label']}",
        subtitle=f"Nodes ordered by {centrality}; the highlighted ones are in the plan "
                 "OMP kept",
        x_label=f"{centrality} score",
        series=[Series(
            name="candidate", x=nodes, y=scores,
            colors=[STATUS["good"] if n in best else "#9db4cc" for n in nodes],
            meta={"legend": ["materialized" if n in best else "considered" for n in nodes]},
        )],
        value_labels=True, value_fmt="{:.0f}",
        note="Centrality decides what the search is allowed to look at; `omp_top` "
             "decides how far down this list it goes.",
    )


def _trace_chart(cell: dict, trace: list[dict]) -> ChartSpec:
    chart = search_trace(
        cell, trace, f"omp-trace-{cell['slug']}",
        f"Plans enumerated — {cell['label']}",
        "What each plan OMP executed cost, and what it measured",
        unit="plan",
    )
    chart.note += (" With early termination on, the search stops as soon as no "
                   "remaining plan can beat the incumbent, so a short trace is a "
                   "result rather than a truncation.")
    return chart


def _summary_table(cells: list[dict]) -> Table:
    table = Table(
        key="omp-summary", title="What OMP did, per cell",
        columns=["Project", "Backend", "SF", "Variant", "Centrality", "Pass time",
                 "DAG runs", "Candidates", "Applied", "Baseline", "Best", "Change",
                 "Plan kept"],
        numeric=(5, 6, 7, 8, 9, 10, 11),
    )
    for c in cells:
        d = c["detail"]
        change = d.get("opt_change")
        table.rows.append([
            c["project"], c["backend"], f"{float(c['sf']):g}", c["variant"],
            d.get("centrality") or "-",
            fmt_num((c["wall_ms"] or 0) / 1000.0, 2, "s"),
            c["dag_runs_used"], c["candidates_considered"], c["changes_applied"],
            fmt_num((d.get("baseline_value") or 0) / 1000.0 or None, 3, "s"),
            fmt_num((d.get("best_value") or 0) / 1000.0 or None, 3, "s"),
            f"{change:+.1%}" if change is not None else "-",
            ", ".join(short_node(n) for n in (d.get("best_plan") or [])) or "-",
        ])
    return table
