"""NodeFusion: what the rollup absorbed, and which CTEs it decided to spool.

Fusing a region of views into one query removes the round trips between them
and lets the backend plan the whole thing at once — but it also makes every
view's SQL run once per reader inside the fused plan, unless the CTE is
materialized. That trade is the pass's entire decision surface, and it is
recorded per node, so it can be shown per node.
"""

from __future__ import annotations

from ..query import fmt_num, median, short_node
from ..spec import ChartSpec, Kpi, Series, Study, Table
from ..theme import STATUS
from ._common import (MAX_DETAILED_CELLS, iterations, page, pass_cells,
                      search_trace, settings_table)

SETTINGS = ["nodefusion_adaptive_materialize_ctes", "nodefusion_objective",
            "nodefusion_search_budget", "nodefusion_max_runs", "nodefusion_top_share",
            "nodefusion_cost_model", "nodefusion_spool_cost_factor",
            "nodefusion_spool_seconds_per_byte", "nodefusion_materialize_ctes_override"]


def build(con) -> Study | None:
    cells = pass_cells(con, "nodefusion")
    if not cells:
        return None
    traces = iterations(con, "nodefusion")

    study = page("nodefusion", question=(
        "Rolling a region of views into one query removes the hops between them, but "
        "makes shared work run once per reader unless its CTE is spooled. How much did "
        "each rollup absorb, and which CTEs did it decide were worth spooling?"
    ))
    study.kpis = _kpis(cells)
    study.charts.append(_absorbed(cells))
    for cell in cells[:MAX_DETAILED_CELLS]:
        duplicates = _duplicate_work(cell)
        if duplicates:
            study.charts.append(duplicates)
    adaptive = _adaptive(cells)
    if adaptive:
        study.charts.append(adaptive)
    for cell in cells[:MAX_DETAILED_CELLS]:
        trace = traces.get(cell["cell_id"])
        if trace:
            study.charts.append(_trace_chart(cell, trace))

    study.tables = [t for t in (
        _summary_table(cells),
        settings_table(cells, SETTINGS, key="nodefusion-settings"),
        _cte_table(cells),
    ) if t]
    return study


def _kpis(cells: list[dict]) -> list[Kpi]:
    inlined = [c["detail"].get("views_inlined") or 0 for c in cells]
    spooled = [c["detail"].get("materialized_ctes") or 0 for c in cells]
    adaptive = [c for c in cells if c["detail"].get("adaptive")]
    duplicated = []
    for cell in cells:
        counts = [n for _, n in (cell["detail"].get("exec_counts") or [])]
        if counts:
            duplicated.append(sum(c - 1 for c in counts if c > 1))
    return [
        Kpi("Cells running NodeFusion", str(len(cells))),
        Kpi("Views absorbed", str(sum(inlined)),
            f"Median {median(inlined):.0f} per rollup." if inlined else ""),
        Kpi("CTEs spooled", str(sum(spooled)),
            "Materialized inside the fused query, so their readers share one copy "
            "instead of recomputing it."),
        Kpi("Re-executions avoided",
            str(sum(duplicated)) if duplicated else "-",
            "Extra executions the fused plan would have performed had nothing been "
            "spooled — the work the decision is about."),
        Kpi("Chosen by measurement", f"{len(adaptive)}/{len(cells)}",
            "Cells whose CTE set was picked by the adaptive search rather than by the "
            "rule of two or more readers."),
    ]


def _absorbed(cells: list[dict]) -> ChartSpec:
    labels = [c["short"] for c in cells]
    return ChartSpec(
        id="nodefusion-absorbed", kind="stacked_bar",
        title="What each rollup absorbed",
        subtitle="Nodes folded into the fused query, and how many of them were spooled "
                 "as materialized CTEs",
        y_label="Nodes",
        series=[
            Series(name="views inlined", x=labels,
                   y=[c["detail"].get("views_inlined") for c in cells]),
            Series(name="tables fused", x=labels,
                   y=[c["detail"].get("tables_fused") for c in cells]),
            Series(name="CTEs spooled", x=labels,
                   y=[c["detail"].get("materialized_ctes") for c in cells],
                   color=STATUS["good"]),
        ],
        note="A spooled CTE is also an inlined view; the third band counts how many of "
             "the absorbed nodes the pass decided to pay to materialize.",
        value_labels=True, value_fmt="{:.0f}",
    )


def _duplicate_work(cell: dict) -> ChartSpec | None:
    """How many times each absorbed node's SQL runs inside the fused query.

    This is the number the spooling decision is made against: a node whose SQL
    the fused plan would run three times is two executions of duplicate work,
    unless its CTE is materialized.
    """
    counts = cell["detail"].get("exec_counts") or []
    if not counts:
        return None
    spooled = {short_node(c.get("node")) for c in (cell["detail"].get("ctes") or [])
               if c.get("materialized")}
    ranked = sorted(((short_node(node), n) for node, n in counts),
                    key=lambda kv: (-kv[1], kv[0]))[:18]
    nodes = [n for n, _ in ranked]
    return ChartSpec(
        id=f"nodefusion-dup-{cell['slug']}", kind="hbar",
        title=f"How often each absorbed node would run — {cell['label']}",
        subtitle="Executions of the node's SQL inside the fused query, and which of "
                 "them the pass spooled",
        x_label="Executions inside the fused query",
        series=[Series(
            name="executions", x=nodes, y=[n for _, n in ranked],
            colors=[STATUS["good"] if n in spooled else "#9db4cc" for n in nodes],
            meta={"legend": ["spooled as a materialized CTE" if n in spooled
                             else "recomputed per reader" for n in nodes]},
        )],
        value_labels=True, value_fmt="{:.0f}",
        note="Spooling costs one write and a barrier its readers wait on; recomputing "
             "costs the node's own query, once per reader. Which is cheaper is what "
             "`nodefusion_spool_cost_factor` is asserting.",
    )


def _adaptive(cells: list[dict]) -> ChartSpec | None:
    adaptive = [c for c in cells if c["detail"].get("adaptive")]
    if not adaptive:
        return None
    labels = [c["short"] for c in adaptive]
    return ChartSpec(
        id="nodefusion-adaptive", kind="grouped_bar",
        title="What the adaptive search improved on",
        subtitle="The fusion under the default rule, against the CTE set the search "
                 "settled on",
        y_label="Runtime (s)",
        series=[
            Series(name="default rule", x=labels,
                   y=[(c["detail"].get("baseline_runtime_ms") or 0) / 1000.0 or None
                      for c in adaptive]),
            Series(name="chosen by search", x=labels,
                   y=[(c["detail"].get("final_runtime_ms") or 0) / 1000.0 or None
                      for c in adaptive]),
        ],
        value_labels=True,
        note="Unlike every other NodeFusion cell, an adaptive one spends DAG runs to "
             "decide — so read it on payback (study 3) and not on wall clock alone.",
    )


def _trace_chart(cell: dict, trace: list[dict]) -> ChartSpec:
    return search_trace(
        cell, trace, f"nodefusion-trace-{cell['slug']}",
        f"CTE sets tried — {cell['label']}",
        "What each candidate set the adaptive search executed cost, and what it "
        "measured",
        unit="candidate set",
    )


def _summary_table(cells: list[dict]) -> Table:
    table = Table(
        key="nodefusion-summary", title="What NodeFusion did, per cell",
        columns=["Project", "Backend", "SF", "Variant", "Adaptive", "Objective",
                 "Views inlined", "Tables fused", "CTEs spooled", "Columns",
                 "Pass time", "Outcome"],
        numeric=(6, 7, 8, 9, 10),
    )
    for c in cells:
        d = c["detail"]
        table.rows.append([
            c["project"], c["backend"], f"{float(c['sf']):g}", c["variant"],
            "yes" if d.get("adaptive") else "no",
            d.get("objective") or "-",
            d.get("views_inlined"), d.get("tables_fused"), d.get("materialized_ctes"),
            d.get("fused_columns"),
            fmt_num((c["wall_ms"] or 0) / 1000.0, 2, "s"),
            d.get("outcome") or "-",
        ])
    return table


def _cte_table(cells: list[dict]) -> Table | None:
    table = Table(
        key="nodefusion-ctes", title="Every CTE in the rollup, and why it is there",
        columns=["Cell", "Node", "Why it became a CTE", "Readers", "Executions",
                 "Spooled"],
        numeric=(3, 4),
        note="`Executions` counts how many times the fused plan would run this node's "
             "SQL if it were not spooled.",
    )
    for cell in cells:
        counts = {short_node(node): n for node, n in (cell["detail"].get("exec_counts") or [])}
        for cte in cell["detail"].get("ctes") or []:
            node = short_node(cte.get("node"))
            table.rows.append([
                cell["label"], node, cte.get("reason") or "-", cte.get("readers"),
                counts.get(node, "-"),
                "yes" if cte.get("materialized") else "no",
            ])
    return table if table.rows else None
