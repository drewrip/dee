"""HMP: what the search spent its run budget on, and what that budget bought.

HMP picks views worth materializing by *measuring* candidate DAGs, so it has
two things no other pass has: a budget it can exhaust, and a cost model whose
predictions can be scored against what the candidate then actually did. Both
are what this page is about.
"""

from __future__ import annotations

from collections import Counter

from ..query import fmt_num, median, short_node
from ..spec import ChartSpec, Kpi, Series, Study, Table
from ..theme import STATUS
from ._common import (MAX_DETAILED_CELLS, iterations, page, pass_cells,
                      search_trace, settings_table)

SETTINGS = ["hmp_max_runs", "hmp_search_budget", "hmp_objective", "hmp_cost_method",
            "hmp_dup_cost_model", "hmp_top_cpu_time", "hmp_downstream_cost",
            "hmp_normalize_with_cardinality", "hmp_use_pushdown", "trial_resume",
            "trial_budget_eps", "trial_reuse"]


def build(con) -> Study | None:
    cells = pass_cells(con, "hmp")
    if not cells:
        return None
    traces = iterations(con, "hmp")

    study = page("hmp", question=(
        "HMP buys its answer with DAG runs. Which candidates did it spend them on, "
        "how close was the cost model that ranked them, and what did it end up "
        "materializing?"
    ))
    study.kpis = _kpis(cells, traces)
    study.charts.append(_search_cost(cells))
    for cell in cells[:MAX_DETAILED_CELLS]:
        trace = traces.get(cell["cell_id"])
        if trace:
            study.charts.append(_trace_chart(cell, trace))
    accuracy = _prediction_accuracy(cells, traces)
    if accuracy:
        study.charts.append(accuracy)
    cancelled = _cancellation(cells, traces)
    if cancelled:
        study.charts.append(cancelled)
    chosen = _chosen_nodes(cells)
    if chosen:
        study.charts.append(chosen)

    study.tables = [t for t in (
        _summary_table(cells),
        settings_table(cells, SETTINGS, key="hmp-settings"),
        _trace_table(cells, traces),
    ) if t]
    if not traces:
        study.empty_reason = (
            "No search trace was recorded, so the per-candidate charts are absent. "
            "`pass_iterations` is written at `standard` verbosity and above."
        )
    return study


# --------------------------------------------------------------------------


def _kpis(cells: list[dict], traces: dict) -> list[Kpi]:
    runs = [c["dag_runs_used"] for c in cells if c["dag_runs_used"] is not None]
    changes = [c["changes_applied"] for c in cells if c["changes_applied"] is not None]
    every = [it for trace in traces.values() for it in trace]
    cancelled = [it for it in every if it["outcome"] == "cancelled"]
    errors = _prediction_errors(traces)

    return [
        Kpi("Cells running HMP", str(len(cells)),
            "One row of `pass_stats` per cell whose variant enabled the pass."),
        Kpi("DAG runs spent", f"{sum(runs)}",
            f"Median {median(runs):.0f} per cell. The dominant term in optimization cost."
            if runs else "The dominant term in optimization cost."),
        Kpi("Materializations applied", f"{sum(changes)}",
            f"Median {median(changes):.0f} per cell." if changes else ""),
        Kpi("Candidates cut short",
            f"{len(cancelled)}/{len(every)}" if every else "-",
            "Candidates cancelled for overrunning the incumbent's budget, and finished "
            "under it instead.",
            tone="warning" if every and len(cancelled) > len(every) / 2 else ""),
        Kpi("Cost model error",
            f"{median(errors):.0%}" if errors else "-",
            "Median absolute error of the predicted makespan against what the candidate "
            "then measured.",
            tone=_error_tone(median(errors) if errors else None)),
    ]


def _error_tone(error: float | None) -> str:
    if error is None:
        return ""
    if error <= 0.15:
        return "good"
    return "warning" if error <= 0.4 else "critical"


def _prediction_errors(traces: dict) -> list[float]:
    out = []
    for trace in traces.values():
        for it in trace:
            predicted, measured = it.get("predicted_makespan_ms"), it.get("runtime_ms")
            if it["outcome"] == "ok" and predicted and measured:
                out.append(abs(predicted - measured) / measured)
    return out


# --------------------------------------------------------------------------
# charts
# --------------------------------------------------------------------------


def _search_cost(cells: list[dict]) -> ChartSpec:
    labels = [c["short"] for c in cells]
    return ChartSpec(
        id="hmp-cost", kind="grouped_bar",
        title="What each search spent, and what it found",
        subtitle="DAG runs the search executed, candidates it priced, and "
                 "materializations it kept",
        y_label="Count",
        series=[
            Series(name="DAG runs spent", x=labels,
                   y=[c["dag_runs_used"] for c in cells]),
            Series(name="candidates considered", x=labels,
                   y=[c["candidates_considered"] for c in cells]),
            Series(name="working set", x=labels,
                   y=[c["working_set_size"] for c in cells]),
            Series(name="materializations applied", x=labels,
                   y=[c["changes_applied"] for c in cells]),
        ],
        value_labels=True, value_fmt="{:.0f}",
        note="Candidates are priced with EXPLAIN, which is cheap; only a DAG run is "
             "expensive. `hmp_search_budget` bounds the first, `hmp_max_runs` the second.",
    )


def _trace_chart(cell: dict, trace: list[dict]) -> ChartSpec:
    return search_trace(
        cell, trace, f"hmp-trace-{cell['slug']}",
        f"The search, candidate by candidate — {cell['label']}",
        "What each iteration of the search cost, and what the candidate measured",
    )


def _prediction_accuracy(cells: list[dict], traces: dict) -> ChartSpec | None:
    """Predicted makespan against what the candidate then measured.

    The cost model is what decides which candidates are worth a DAG run at all,
    so how well it predicts is upstream of everything else the pass does.
    """
    by_cell = {c["cell_id"]: c for c in cells}
    series, values = [], []
    for i, (cell_id, trace) in enumerate(sorted(traces.items())):
        cell = by_cell.get(cell_id)
        if not cell:
            continue
        pts = [(it["predicted_makespan_ms"] / 1000.0, it["runtime_ms"] / 1000.0)
               for it in trace
               if it["outcome"] == "ok" and it.get("predicted_makespan_ms")
               and it.get("runtime_ms")]
        if not pts:
            continue
        xs, ys = zip(*pts)
        values.extend(xs), values.extend(ys)
        series.append(Series(name=cell["label"], x=list(xs), y=list(ys), kind="points"))
    if not series:
        return None

    lo, hi = min(values) * 0.9, max(values) * 1.1
    series.append(Series(name="a perfect prediction", x=[lo, hi], y=[lo, hi],
                         kind="line", dash="dash", color=STATUS["neutral"]))
    return ChartSpec(
        id="hmp-prediction", kind="points",
        title="Was the cost model right?",
        subtitle="What each candidate was predicted to take, against what it then took",
        x_label="Predicted makespan (s)", y_label="Measured runtime (s)",
        x_type="linear",
        series=series,
        note="Only candidates that ran to completion are scored: a cancelled one has no "
             "measurement to score against. Marks above the line ran slower than "
             "predicted, below it faster.",
        height=400,
    )


def _cancellation(cells: list[dict], traces: dict) -> ChartSpec | None:
    """What a cancelled candidate cost, against running it out to the end.

    Cancelling is only worth it if the trial, the resume planning and the
    resumed run together cost less than letting the candidate finish would
    have. That comparison is the whole argument for `trial_resume`, and it is
    measurable here.
    """
    by_cell = {c["cell_id"]: c for c in cells}
    labels, trial, overhead, resume, avoided = [], [], [], [], []
    for cell_id, trace in sorted(traces.items()):
        cell = by_cell.get(cell_id)
        if not cell:
            continue
        for it in trace:
            if it["outcome"] != "cancelled":
                continue
            labels.append(f"{cell['project']}\n#{it['iteration']}")
            trial.append((it.get("trial_ms") or 0) / 1000.0)
            overhead.append((it.get("resume_overhead_ms") or 0) / 1000.0)
            resume.append((it.get("resume_ms") or 0) / 1000.0)
            # What the run would have cost had it been allowed to finish is
            # unknown by construction; the incumbent it overran is the floor.
            avoided.append((it.get("total_ms") or 0) / 1000.0 or None)
    if not labels:
        return None
    return ChartSpec(
        id="hmp-cancellation", kind="stacked_bar",
        title="What cancelling a candidate cost",
        subtitle="Each cancelled candidate's trial, the resume planning, and the run "
                 "that finished under the incumbent",
        x_label="", y_label="Wall time (s)",
        series=[
            Series(name="trial before cancelling", x=labels, y=trial,
                   color=STATUS["warning"]),
            Series(name="resume planning", x=labels, y=overhead, color=STATUS["neutral"]),
            Series(name="finished under the incumbent", x=labels, y=resume),
        ],
        note="The stack is what the iteration actually cost end to end. It is worth "
             "paying only where letting the candidate finish would have cost more — "
             "which is what `trial_budget_eps` is tuning.",
        height=340,
    )


def _chosen_nodes(cells: list[dict]) -> ChartSpec | None:
    """Which views HMP kept, and which it only ever considered."""
    chosen: Counter = Counter()
    considered: Counter = Counter()
    for cell in cells:
        detail = cell["detail"]
        for node in detail.get("new_materializations") or []:
            chosen[short_node(node)] += 1
        for node in detail.get("working_set") or []:
            considered[short_node(node)] += 1
    if not considered and not chosen:
        return None
    order = [n for n, _ in sorted(considered.items() or chosen.items(),
                                  key=lambda kv: (-kv[1], kv[0]))][:18]
    return ChartSpec(
        id="hmp-nodes", kind="grouped_bar",
        title="Which views the search looked at, and which it kept",
        subtitle="Across every cell running HMP, counted in cells",
        x_label="", y_label="Cells",
        series=[
            Series(name="in the working set", x=order,
                   y=[considered.get(n) for n in order], color=STATUS["neutral"]),
            Series(name="materialized", x=order, y=[chosen.get(n) for n in order]),
        ],
        note="The working set is what `hmp_top_cpu_time` admitted to the search; a view "
             "in it that was never materialized was priced and rejected.",
        value_labels=True, value_fmt="{:.0f}",
    )


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------


def _summary_table(cells: list[dict]) -> Table:
    table = Table(
        key="hmp-summary", title="What HMP did, per cell",
        columns=["Project", "Backend", "SF", "Variant", "Pass time", "DAG runs",
                 "Working set", "Candidates", "Applied", "Baseline", "Final",
                 "Materialized"],
        numeric=(4, 5, 6, 7, 8, 9, 10),
    )
    for c in cells:
        detail = c["detail"]
        table.rows.append([
            c["project"], c["backend"], f"{float(c['sf']):g}", c["variant"],
            fmt_num((c["wall_ms"] or 0) / 1000.0, 2, "s"),
            c["dag_runs_used"], c["working_set_size"], c["candidates_considered"],
            c["changes_applied"],
            fmt_num((detail.get("baseline_runtime_ms") or 0) / 1000.0 or None, 3, "s"),
            fmt_num((detail.get("final_runtime_ms") or 0) / 1000.0 or None, 3, "s"),
            ", ".join(short_node(n) for n in (detail.get("new_materializations") or [])) or "-",
        ])
    return table


def _trace_table(cells: list[dict], traces: dict) -> Table | None:
    if not traces:
        return None
    by_cell = {c["cell_id"]: c for c in cells}
    table = Table(
        key="hmp-trace", title="Every candidate the search executed",
        columns=["Cell", "#", "Outcome", "Materialized", "Runtime", "Predicted",
                 "Node time", "Trial", "Resume", "Total"],
        numeric=(1, 4, 5, 6, 7, 8, 9),
        note="`Runtime` is a lower bound where the outcome is `cancelled`, and in the "
             "objective's own units. `Total` is what the iteration cost end to end, "
             "and never a bound.",
    )
    for cell_id, trace in sorted(traces.items()):
        cell = by_cell.get(cell_id)
        if not cell:
            continue
        for it in trace:
            table.rows.append([
                cell["label"], it["iteration"], it["outcome"],
                ", ".join(short_node(n) for n in (it["combo"] or [])) or "-",
                fmt_num((it["runtime_ms"] or 0) / 1000.0 or None, 3, "s"),
                fmt_num((it.get("predicted_makespan_ms") or 0) / 1000.0 or None, 3, "s"),
                fmt_num((it.get("node_time_ms") or 0) / 1000.0 or None, 3, "s"),
                fmt_num((it.get("trial_ms") or 0) / 1000.0 or None, 3, "s"),
                fmt_num((it.get("resume_ms") or 0) / 1000.0 or None, 3, "s"),
                fmt_num((it.get("total_ms") or 0) / 1000.0 or None, 3, "s"),
            ])
    return table
