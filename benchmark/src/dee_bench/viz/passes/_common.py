"""Shared reads and furniture for the per-optimization pages."""

from __future__ import annotations

from typing import Any

from ..query import cell_label, dicts, loads, projection, tables
from ..spec import ChartSpec, Study, Table
from ..theme import PASSES

MAX_DETAILED_CELLS = 6


def page(key: str, question: str = "") -> Study:
    meta = PASSES[key]
    return Study(
        key=f"pass-{key}", number=0, group="optimizations",
        title=meta["label"], subtitle=meta["full"],
        question=question or meta["blurb"],
    )


def pass_cells(con, key: str) -> list[dict[str, Any]]:
    """Every cell that ran this pass, with its pass_stats row and parsed detail.

    A cell is matched on the recorded `pass_name` *or* on what its detail blob
    says it is, so a rename on either side leaves the page working.
    """
    if not {"cells", "pass_stats"} <= tables(con):
        return []
    name = PASSES[key]["pass_name"]
    kind = key.replace("_", "")
    cell_columns = projection(
        con, "cells",
        ["cell_id", "project", "backend", "sf", "variant", "dee_opt", "passes"], "c")
    pass_columns = projection(
        con, "pass_stats",
        ["pass_name", "pass_order", "wall_ms", "dag_runs_used", "changes_applied",
         "candidates_considered", "working_set_size", "detail"], "p")
    out = []
    for row in dicts(con, f"""
        SELECT {cell_columns}, {pass_columns}
        FROM pass_stats p JOIN cells c USING (cell_id)
        ORDER BY c.backend, c.project, c.sf, c.variant, p.pass_order
    """):
        detail = loads(row["detail"])
        if row["pass_name"] != name and str(detail.get("kind", "")).replace("_", "") != kind:
            continue
        row = dict(row)
        row["detail"] = detail
        row["opt"] = loads(row["dee_opt"])
        row["label"] = cell_label(row["project"], row["backend"], float(row["sf"]),
                                  row["variant"])
        row["short"] = cell_label(row["project"], row["backend"], float(row["sf"]),
                                  row["variant"], newline=True)
        row["slug"] = row["cell_id"][:12]
        out.append(row)
    return out


def iterations(con, key: str) -> dict[str, list[dict[str, Any]]]:
    """The pass's search trace, per cell, in iteration order."""
    if "pass_iterations" not in tables(con):
        return {}
    name = PASSES[key]["pass_name"]
    by_cell: dict[str, list[dict]] = {}
    for row in dicts(con, f"""
        SELECT * FROM pass_iterations
        WHERE pass_name = '{name}'
        ORDER BY cell_id, iteration
    """):
        by_cell.setdefault(row["cell_id"], []).append(row)
    return by_cell


def settings_table(cells: list[dict], keys: list[str], key: str = "settings") -> Table | None:
    """What each cell asked this pass for, limited to settings that vary.

    A page that printed every option would bury the one the sweep changed; a
    page that printed none could not tell two cells of the same variant apart.
    """
    values: dict[str, set] = {}
    for cell in cells:
        for name in keys:
            values.setdefault(name, set()).add(_hashable(cell["opt"].get(name)))
    varying = [name for name in keys if len(values.get(name, set())) > 1]
    if not varying:
        return None
    table = Table(key=key, title="Settings that vary between these cells",
                  columns=["Cell"] + varying)
    for cell in cells:
        table.rows.append([cell["label"]]
                          + [_show(cell["opt"].get(name)) for name in varying])
    return table


def _hashable(value: Any):
    return tuple(value) if isinstance(value, list) else value


def _show(value: Any) -> str:
    if value is None:
        return "default"
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value) or "-"
    return str(value)


def search_trace(cell: dict, trace: list[dict], chart_id: str, title: str,
                 subtitle: str, unit: str = "candidate") -> ChartSpec:
    """One chart for a search's whole trace: what it spent, and what it measured.

    Two numbers per iteration, and they are not the same number. `total_ms` is
    what the iteration cost end to end — the trial, the planning, and the run
    that finished under the incumbent — and is never a bound. `runtime_ms` is
    what the candidate *measured*, and where the search cancelled it for
    overrunning, that is the incumbent's budget rather than the candidate's
    runtime.

    Plotting only the second is what makes a trace look degenerate: every
    cancelled candidate lands exactly on the incumbent, because that is where
    it was stopped. The bar is therefore the cost and the tick is the
    measurement, which is the honest pair.
    """
    from ..query import short_node
    from ..spec import Series
    from ..theme import STATUS, outcome_color

    labels = [f"#{it['iteration']}" for it in trace]
    outcomes = [str(it["outcome"]) for it in trace]
    total = [(it.get("total_ms") or it.get("runtime_ms") or 0) / 1000.0 or None
             for it in trace]
    measured = [(it.get("runtime_ms") or 0) / 1000.0 or None for it in trace]
    baseline = next((m for it, m in zip(trace, measured)
                     if it["outcome"] == "baseline"), None)

    return ChartSpec(
        id=chart_id, kind="bar", title=title, subtitle=subtitle,
        x_label="", y_label="Wall time (s)",
        series=[
            Series(name=f"what the {unit} cost", x=labels, y=total,
                   colors=[outcome_color(o) for o in outcomes],
                   meta={
                       "legend": outcomes,
                       "outcome": outcomes,
                       "materialized": [", ".join(short_node(n) for n in (it["combo"] or []))
                                        or "(none)" for it in trace],
                       "predicted": [_seconds(it.get("predicted_makespan_ms"))
                                     for it in trace],
                   }),
            Series(name="runtime it measured", x=labels, y=measured,
                   kind="points", color=STATUS["neutral"]),
        ],
        hlines=[(baseline, "the DAG as authored")] if baseline else [],
        note=("The bar is what the iteration actually cost; the tick is what the "
              "candidate measured. Where the outcome is `cancelled` the tick is the "
              "incumbent's budget — the point at which the candidate was stopped — so "
              "it is a lower bound, and the bar above it is the price of finding that "
              "out."),
        height=360,
    )


def _seconds(ms) -> str:
    return "-" if not ms else f"{ms / 1000.0:.3f}s"
