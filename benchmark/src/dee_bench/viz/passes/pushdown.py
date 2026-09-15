"""Pushdown: a static rewrite, so the only questions are how many and which.

Unlike every other pass, Pushdown spends no DAG runs — it is an analysis of the
SQL, not a search over measurements. So there is no trace, no budget and no
cost model to score; what there is, per node, is whether the filter could be
pushed and what happened if it could not.
"""

from __future__ import annotations

from collections import Counter

from ..query import fmt_num, short_node
from ..spec import ChartSpec, Kpi, Series, Study, Table
from ..theme import STATUS
from ._common import page, pass_cells


def build(con) -> Study | None:
    cells = pass_cells(con, "pushdown")
    if not cells:
        return None

    study = page("pushdown", question=(
        "Pushdown is a static rewrite that costs no DAG runs. How many rewrites did it "
        "find, and where it found none, why not?"
    ))
    study.kpis = _kpis(cells)
    study.charts.append(_rewrites(cells))
    outcomes = _outcomes(cells)
    if outcomes:
        study.charts.append(outcomes)
    study.tables = [t for t in (_summary_table(cells), _outcome_table(cells)) if t]
    return study


def _classify(outcome: str) -> str:
    """Group a per-node outcome string into what actually happened to it."""
    text = str(outcome or "").lower()
    if text.startswith("rewritten"):
        return "rewritten"
    if "no filter" in text or "nothing" in text or "none" in text:
        return "nothing to push"
    if "unsupported" in text or "cannot" in text or "refus" in text:
        return "could not be pushed"
    return "unchanged"


ORDER = ["rewritten", "nothing to push", "could not be pushed", "unchanged"]
TONE = {"rewritten": None, "nothing to push": STATUS["neutral"],
        "could not be pushed": STATUS["warning"], "unchanged": "#c9c8c2"}


def _kpis(cells: list[dict]) -> list[Kpi]:
    rewrites = sum(c["changes_applied"] or 0 for c in cells)
    considered = Counter()
    for cell in cells:
        for row in cell["detail"].get("outcomes") or []:
            considered[_classify(row.get("outcome"))] += 1
    total = sum(considered.values())
    return [
        Kpi("Cells running Pushdown", str(len(cells))),
        Kpi("Rewrites applied", str(rewrites)),
        Kpi("Nodes examined", str(total) if total else "-"),
        Kpi("DAG runs spent", "0",
            "A static analysis: it never executes the DAG, so it has nothing to repay.",
            tone="good"),
    ]


def _rewrites(cells: list[dict]) -> ChartSpec:
    labels = [c["short"] for c in cells]
    return ChartSpec(
        id="pushdown-rewrites", kind="grouped_bar",
        title="Rewrites applied per cell",
        subtitle="Queries the pass rewrote, and the materializations it had to push into",
        y_label="Count",
        series=[
            Series(name="queries rewritten", x=labels,
                   y=[c["detail"].get("rewrites_applied", c["changes_applied"])
                      for c in cells]),
            Series(name="materializations available", x=labels,
                   y=[c["detail"].get("temp_tables_count") for c in cells],
                   color=STATUS["neutral"]),
        ],
        value_labels=True, value_fmt="{:.0f}",
        note="Pushdown pushes filters into the queries feeding a materialization, so a "
             "cell with no materializations has nothing to push into — which is why it "
             "is normally paired with HMP or OMP rather than run alone.",
    )


def _outcomes(cells: list[dict]) -> ChartSpec | None:
    counts: dict[str, Counter] = {}
    for cell in cells:
        rows = cell["detail"].get("outcomes") or []
        if not rows:
            continue
        counter = counts.setdefault(cell["short"], Counter())
        for row in rows:
            counter[_classify(row.get("outcome"))] += 1
    if not counts:
        return None
    labels = list(counts)
    present = [k for k in ORDER if any(c.get(k) for c in counts.values())]
    return ChartSpec(
        id="pushdown-outcomes", kind="stacked_bar",
        title="What happened to each node the pass looked at",
        subtitle="Every node considered, grouped by outcome",
        y_label="Nodes",
        series=[Series(name=kind, x=labels, y=[counts[l].get(kind) for l in labels],
                       color=TONE[kind])
                for kind in present],
        note="`could not be pushed` is the interesting band: those are filters the "
             "rewrite recognised but the query shape refused.",
    )


def _summary_table(cells: list[dict]) -> Table:
    table = Table(
        key="pushdown-summary", title="What Pushdown did, per cell",
        columns=["Project", "Backend", "SF", "Variant", "Rewrites", "Materializations",
                 "Pass time"],
        numeric=(4, 5, 6),
    )
    for c in cells:
        table.rows.append([
            c["project"], c["backend"], f"{float(c['sf']):g}", c["variant"],
            c["detail"].get("rewrites_applied", c["changes_applied"]),
            c["detail"].get("temp_tables_count"),
            fmt_num((c["wall_ms"] or 0) / 1000.0, 3, "s"),
        ])
    return table


def _outcome_table(cells: list[dict]) -> Table | None:
    table = Table(
        key="pushdown-outcomes-table", title="Every node the pass examined",
        columns=["Cell", "Node", "Outcome", "What happened"],
    )
    for cell in cells:
        for row in cell["detail"].get("outcomes") or []:
            table.rows.append([
                cell["label"], short_node(row.get("node_id")),
                row.get("outcome") or "-", _classify(row.get("outcome")),
            ])
    return table if table.rows else None
