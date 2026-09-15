"""Run by run: what a continuous optimization does while the DAG does its work.

A batch optimization is a thing that happens *before* the measurements, so a
single number describes it: what it cost, and what the result runs at. A
continuous optimization has no before. It is registered on the DAG and steps
around the runs the DAG was performing anyway, so its cost, its search and its
result are all spread across the same series of runs — and a chart of medians
hides exactly the thing worth seeing.

So this page plots the runs themselves, in order, for each continuous cell:
what each run took, which DAG version it executed, when the search promoted its
result, and how long the DAG had to keep running before the disturbance of
being optimized in place had paid for itself.
"""

from __future__ import annotations

from typing import Any

from .query import (cell_label, columns, dicts, fmt_ms, fmt_num, median,
                    projection, short_node, tables)
from .spec import ChartSpec, Kpi, Series, Study, Table
from .theme import LIGHT_SERIES, STATUS

# Past this many continuous cells the per-cell charts stop being a page and
# start being a wall, so the rest are left to the summary and the table.
MAX_DETAILED_CELLS = 4
# Nodes shown in the per-node comparisons, longest first. A DAG has dozens and
# a handful of them hold the time.
TOP_NODES = 14


def _cells(con) -> list[dict[str, Any]]:
    present = tables(con)
    if not {"cells", "runs"} <= present:
        return []
    if "optimization_mode" not in columns(con, "cells"):
        # Recorded before continuous optimization existed, so by construction
        # none of its cells was driven that way.
        return []
    cells = dicts(con, """
        SELECT cell_id, project, backend, sf, variant, repetitions, warmups,
               optimization_mode, passes, backend_config
        FROM cells
        WHERE optimization_mode = 'continuous'
        ORDER BY backend, project, sf, variant
    """)
    if not cells:
        return []

    runs = dicts(con, f"""
        SELECT {projection(con, "runs",
                           ["cell_id", "run_id", "phase", "rep_index", "dag_version",
                            "engine_wall_ms", "node_time_ms", "cpu_seconds", "delivery",
                            "trial_elapsed_ms", "resume_elapsed_ms", "status",
                            "started_at"])}
        FROM runs
        ORDER BY cell_id, started_at NULLS LAST, phase DESC, rep_index
    """)
    by_cell: dict[str, list[dict]] = {}
    for run in runs:
        by_cell.setdefault(run["cell_id"], []).append(run)

    registrations: dict[str, list[dict]] = {}
    if "optimizations" in present:
        for row in dicts(con, """
            SELECT cell_id, status, result_version, step_phase, optimization_type
            FROM optimizations
        """):
            registrations.setdefault(row["cell_id"], []).append(row)

    baselines = _baselines(con)

    out = []
    for cell in cells:
        cell = dict(cell)
        cell["runs"] = by_cell.get(cell["cell_id"], [])
        if not cell["runs"]:
            continue
        cell["registrations"] = registrations.get(cell["cell_id"], [])
        cell["baseline_s"] = baselines.get(
            (cell["project"], cell["backend"], float(cell["sf"]), cell["backend_config"])
        )
        _annotate(cell)
        out.append(cell)
    return out


def _baselines(con) -> dict[tuple, float]:
    """Median measured runtime of the unoptimized cell each continuous cell pairs with.

    The counterfactual a run-by-run chart is read against: what every one of
    these runs would have cost had nothing been optimizing the DAG underneath
    them.
    """
    found: dict[tuple, list[float]] = {}
    for row in dicts(con, """
        SELECT c.project, c.backend, c.sf, c.backend_config,
               r.engine_wall_ms / 1000.0 AS wall_s
        FROM runs r JOIN cells c USING (cell_id)
        WHERE r.phase = 'measure' AND r.status = 'ok'
          AND (c.passes IS NULL OR len(c.passes) = 0)
    """):
        key = (row["project"], row["backend"], float(row["sf"]), row["backend_config"])
        found.setdefault(key, []).append(float(row["wall_s"]))
    return {k: v for k, v in ((k, median(v)) for k, v in found.items()) if v}


def _annotate(cell: dict[str, Any]) -> None:
    """Number the runs and work out where — and whether — the search promoted.

    `dag_version` is the only thing in the data that separates the two halves
    of a continuous cell, and it is what `analyze` prices payback from, so the
    page reads the promotion the same way rather than inventing a second rule.
    """
    for i, run in enumerate(cell["runs"], start=1):
        run["ordinal"] = i
        run["wall_s"] = (run["engine_wall_ms"] or 0) / 1000.0 if run["engine_wall_ms"] else None

    converged = [r for r in cell["registrations"] if r["status"] == "converged"]
    cell["converged"] = bool(converged)
    promoted = next((r["result_version"] for r in converged if r["result_version"] is not None), None)

    versions = [r["dag_version"] for r in cell["runs"] if r["dag_version"] is not None]
    if promoted is None and versions:
        # No registration row said which version won — an older result, or a
        # cell still converging. The version rising is the same event.
        promoted = max(versions) if max(versions) > min(versions) else None
    cell["promoted_version"] = promoted

    at_promoted = [r for r in cell["runs"]
                   if promoted is not None and r["dag_version"] == promoted]
    cell["promotion_run"] = at_promoted[0]["ordinal"] if at_promoted else None
    cell["runs_at_result"] = len(at_promoted)

    measured = [r for r in cell["runs"] if r["phase"] == "measure" and r["wall_s"]]
    before = [r["wall_s"] for r in measured
              if cell["promotion_run"] is None or r["ordinal"] < cell["promotion_run"]]
    after = [r["wall_s"] for r in measured
             if cell["promotion_run"] is not None and r["ordinal"] >= cell["promotion_run"]]
    cell["converging_s"] = median(before)
    cell["settled_s"] = median(after)
    base = cell["baseline_s"]
    cell["settled_speedup"] = (base / cell["settled_s"]) if (base and cell["settled_s"]) else None
    cell["label"] = cell_label(cell["project"], cell["backend"], float(cell["sf"]),
                               cell["variant"])


def build(con) -> Study | None:
    """The run-by-run page, or None when no cell was optimized continuously."""
    cells = _cells(con)
    if not cells:
        return None

    study = Study(
        key="continuous", number=0, group="runs",
        title="Continuous optimization, run by run",
        subtitle="What each run did while the DAG was being optimized underneath it",
        question=(
            "A continuous optimization spends no runs of its own — it steps around the "
            "runs the DAG was performing anyway. So what did each of those runs actually "
            "execute, when did the search promote its result, and how long did the DAG "
            "have to keep running before being optimized in place paid for itself?"
        ),
    )
    study.kpis = _kpis(cells)
    if len(cells) > 1:
        study.charts.extend(_summary_charts(cells))
    detailed = cells[:MAX_DETAILED_CELLS]
    nodes = _node_executions(con, detailed)
    for cell in detailed:
        study.charts.extend(_cell_charts(cell))
        study.charts.extend(_node_charts(cell, nodes))
    study.tables = _tables(cells)
    if len(cells) > MAX_DETAILED_CELLS:
        study.empty_reason = (
            f"{len(cells)} cells ran continuously; the run-by-run charts show the first "
            f"{MAX_DETAILED_CELLS}. Every cell is in the tables below."
        )
    return study


# --------------------------------------------------------------------------
# headline
# --------------------------------------------------------------------------


def _kpis(cells: list[dict]) -> list[Kpi]:
    converged = [c for c in cells if c["converged"]]
    promotions = [c["promotion_run"] - 1 for c in converged if c["promotion_run"]]
    speedups = [c["settled_speedup"] for c in converged if c["settled_speedup"]]

    kpis = [
        Kpi("Cells converged", f"{len(converged)}/{len(cells)}",
            "A cell that ran out of runs before deciding is recorded as converging, "
            "and has no result to price.",
            tone="good" if len(converged) == len(cells) else "warning"),
        Kpi("Runs before promotion",
            f"{median(promotions):.0f}" if promotions else "-",
            "Median runs the search spent before it promoted a result."),
        Kpi("Settled speedup",
            f"{median(speedups):.2f}x" if speedups else "-",
            "Median runtime once promoted, against the unoptimized baseline.",
            tone=_tone(median(speedups) if speedups else None)),
        Kpi("Runs at the result",
            str(sum(c["runs_at_result"] for c in cells)),
            "Measured runs that executed the promoted version. Only these are runs "
            "of the optimized DAG."),
    ]
    return kpis


def _tone(speedup: float | None) -> str:
    if speedup is None:
        return ""
    if speedup >= 1.05:
        return "good"
    return "warning" if speedup >= 0.98 else "critical"


# --------------------------------------------------------------------------
# per-cell charts — the run-by-run view itself
# --------------------------------------------------------------------------


def _cell_charts(cell: dict) -> list[ChartSpec]:
    charts: list[ChartSpec] = []
    runs = cell["runs"]
    slug = cell["cell_id"][:12]
    promotion = cell["promotion_run"]
    vlines = ([(promotion, f"promoted v{cell['promoted_version']}")]
              if promotion and cell["promoted_version"] is not None else [])

    # 1. What each run took. The chart this page exists for.
    measured = [r for r in runs if r["phase"] == "measure"]
    warmups = [r for r in runs if r["phase"] != "measure"]

    def meta(subset: list[dict]) -> dict:
        return {
            "version": [r["dag_version"] for r in subset],
            "delivery": [r["delivery"] for r in subset],
            "status": [r["status"] for r in subset],
        }

    if promotion:
        # Split at the promotion rather than drawing one line through it: these
        # are runs of two different DAGs, and a single unbroken series says
        # they are runs of one.
        before = [r for r in measured if r["ordinal"] < promotion]
        after = [r for r in measured if r["ordinal"] >= promotion]
        # The last converging run is repeated at the head of the second series
        # so the line is continuous across the change it is marking.
        joined = (before[-1:] + after) if before else after
        series = [
            Series(name="while converging", x=[r["ordinal"] for r in before],
                   y=[r["wall_s"] for r in before], meta=meta(before),
                   color=STATUS["neutral"]),
            Series(name="at the promoted version", x=[r["ordinal"] for r in joined],
                   y=[r["wall_s"] for r in joined], meta=meta(joined)),
        ]
    else:
        series = [Series(name="measured run", x=[r["ordinal"] for r in measured],
                         y=[r["wall_s"] for r in measured], meta=meta(measured))]
    if warmups:
        series.append(Series(name="warmup", x=[r["ordinal"] for r in warmups],
                             y=[r["wall_s"] for r in warmups],
                             color=STATUS["neutral"], kind="points"))
    hlines = []
    if cell["baseline_s"]:
        hlines.append((cell["baseline_s"], "unoptimized baseline"))
    if cell["settled_s"] and promotion:
        hlines.append((cell["settled_s"], "median once promoted"))
    charts.append(ChartSpec(
        id=f"continuous-runs-{slug}", kind="line",
        title=f"Every run, in order — {cell['label']}",
        subtitle="Wall time of each execution while the optimization stepped around it",
        x_label="Run (in execution order)", y_label="Runtime (s)",
        x_type="linear", series=series, hlines=hlines, vlines=vlines,
        note=("Runs left of the marker executed the DAG as authored or a candidate the "
              "search was trying; runs from the marker on executed the promoted version. "
              "Only the latter are runs of the optimized DAG."),
        height=400,
    ))

    # 2. Which DAG version each run executed. The promotion, exactly.
    versions = [r for r in runs if r["dag_version"] is not None]
    if versions and len({r["dag_version"] for r in versions}) > 1:
        charts.append(ChartSpec(
            id=f"continuous-version-{slug}", kind="step",
            title=f"DAG version per run — {cell['label']}",
            subtitle="The version rises once, when the search promotes its result",
            x_label="Run (in execution order)", y_label="dee DAG version",
            x_type="linear",
            series=[Series(name="dag_version", x=[r["ordinal"] for r in versions],
                           y=[r["dag_version"] for r in versions])],
            vlines=vlines, height=260,
            note="A flat line to the end means the search had not promoted anything by "
                 "the cell's last run.",
        ))

    # 3. Has being optimized in place paid for itself yet?
    if cell["baseline_s"]:
        charts.append(_crossover(cell, slug))

    # 4. What a disturbed run spent on the search that disturbed it.
    resumed = [r for r in runs if (r["delivery"] or "direct") != "direct"]
    if resumed:
        charts.append(ChartSpec(
            id=f"continuous-delivery-{slug}", kind="stacked_bar",
            title=f"Runs the search cut short — {cell['label']}",
            subtitle="A cancelled candidate's trial, and what finishing under the "
                     "incumbent then took",
            x_label="Run", y_label="Wall time (s)",
            series=[
                Series(name="candidate trial",
                       x=[f"#{r['ordinal']}" for r in resumed],
                       y=[(r["trial_elapsed_ms"] or 0) / 1000.0 for r in resumed],
                       color=STATUS["warning"]),
                Series(name="finished under the incumbent",
                       x=[f"#{r['ordinal']}" for r in resumed],
                       y=[(r["resume_elapsed_ms"] or 0) / 1000.0 for r in resumed],
                       color=LIGHT_SERIES[0]),
            ],
            note="These runs measure neither DAG: the second half started from a warm, "
                 "half-built warehouse. They are the price the search charged the runs "
                 "it borrowed.",
            height=300,
        ))
    return charts


def _crossover(cell: dict, slug: str) -> ChartSpec:
    """Cumulative time against never having optimized at all.

    Payback, but in situ and without a model: the two lines are what the DAG
    actually spent, run by run, and what it would have spent had the
    optimization never been registered. Where they cross is where being
    optimized in place stopped costing.
    """
    runs = [r for r in cell["runs"] if r["phase"] == "measure" and r["wall_s"]]
    base = cell["baseline_s"]
    actual, counterfactual, total = [], [], 0.0
    for i, run in enumerate(runs, start=1):
        total += run["wall_s"]
        actual.append(total)
        counterfactual.append(base * i)

    # This chart counts measured runs; the run-by-run chart counts executions,
    # warmups included. Marking the promotion at the other chart's ordinal
    # would put it one run late for every cell that ran a warmup.
    promotion = cell["promotion_run"]
    marker = next((i for i, run in enumerate(runs, start=1)
                   if promotion and run["ordinal"] >= promotion), None)
    vlines = ([(marker, f"promoted v{cell['promoted_version']}")]
              if marker and cell["promoted_version"] is not None else [])
    # The crossing is where the actual line goes below the counterfactual and
    # *stays* there. Taking the first time it dips under would report a
    # crossing on run 1, where a single fast run before the search had even
    # started is enough to be momentarily ahead.
    behind = [i for i, (a, c) in enumerate(zip(actual, counterfactual))
              if a > c]
    crossing = None
    if actual and actual[-1] <= counterfactual[-1]:
        crossing = (max(behind) + 2) if behind else 1
    note = (f"The lines cross at run {crossing}, and stay crossed: from there on the "
            "DAG has spent less in total than it would have unoptimized."
            if crossing else
            "The lines have not crossed yet — over these runs, optimizing in place has "
            "cost more in total than it has saved.")
    return ChartSpec(
        id=f"continuous-crossover-{slug}", kind="line",
        title=f"Has it paid for itself yet? — {cell['label']}",
        subtitle="Total time spent so far, against never having optimized at all",
        x_label="Measured run", y_label="Cumulative runtime (s)",
        x_type="linear",
        series=[
            # Unfilled: the message is the gap between the two lines, and an
            # area under one of them competes with it for attention.
            Series(name="as it actually ran", x=list(range(1, len(actual) + 1)),
                   y=actual),
            Series(name="never optimized", x=list(range(1, len(counterfactual) + 1)),
                   y=counterfactual, dash="dash", color=STATUS["neutral"]),
        ],
        vlines=vlines, note=note, height=340,
    )


# --------------------------------------------------------------------------
# inside a run
# --------------------------------------------------------------------------


def _compared_runs(cell: dict) -> tuple[dict | None, dict | None]:
    """The last run before the promotion, and the last run after it.

    Two runs of the same cell that executed different DAGs — which is what
    makes them worth putting side by side. A cell that never promoted has only
    the first.
    """
    measured = [r for r in cell["runs"] if r["phase"] == "measure"]
    promotion = cell["promotion_run"]
    if not promotion:
        return (measured[-1] if measured else None), None
    before = [r for r in measured if r["ordinal"] < promotion]
    after = [r for r in measured if r["ordinal"] >= promotion]
    return (before[-1] if before else None), (after[-1] if after else None)


def _node_executions(con, cells: list[dict]) -> dict[str, list[dict]]:
    """Per-node timings for just the runs the page compares, keyed by run.

    Recorded at `detailed` verbosity and above; absent below it, which is a
    normal state rather than a failure.
    """
    if "node_executions" not in tables(con):
        return {}
    wanted = []
    for cell in cells:
        wanted += [r["run_id"] for r in _compared_runs(cell) if r]
    if not wanted:
        return {}
    quoted = ", ".join("'" + str(r).replace("'", "''") + "'" for r in wanted)
    by_run: dict[str, list[dict]] = {}
    for row in dicts(con, f"""
        SELECT run_id, node_id, materialization, duration_ms, started_at
        FROM node_executions
        WHERE run_id IN ({quoted})
        ORDER BY run_id, started_at NULLS LAST
    """):
        by_run.setdefault(row["run_id"], []).append(row)
    return by_run


def _node_charts(cell: dict, by_run: dict[str, list[dict]]) -> list[ChartSpec]:
    before, after = _compared_runs(cell)
    focus = after or before
    execs = by_run.get(focus["run_id"]) if focus else None
    if not execs:
        return []

    charts = [_schedule(cell, focus, execs)]
    if before and after and by_run.get(before["run_id"]):
        charts.append(_node_shift(cell, before, after, by_run))
    return charts


def _schedule(cell: dict, run: dict, execs: list[dict]) -> ChartSpec:
    """When each node ran inside one execution, and for how long.

    The per-run chart shows the DAG's wall clock; this shows where it went.
    Bars that overlap ran concurrently, and the longest unbroken chain across
    them is the path the optimizer has to shorten to move the number above.
    """
    origin = min((e["started_at"] for e in execs if e["started_at"]), default=None)
    starts, ends, lanes = [], [], []
    for e in execs:
        if e["started_at"] is None or e["duration_ms"] is None:
            continue
        offset = (e["started_at"] - origin).total_seconds() if origin else 0.0
        starts.append(offset)
        ends.append(offset + e["duration_ms"] / 1000.0)
        lanes.append(short_node(e["node_id"]))
    stage = ("the promoted version" if run["dag_version"] == cell["promoted_version"]
             else "the DAG as it was then")
    return ChartSpec(
        id=f"continuous-schedule-{cell['cell_id'][:12]}", kind="span",
        title=f"Inside run #{run['ordinal']} — {cell['label']}",
        subtitle=f"When each node ran, and for how long, executing {stage}",
        x_label="Seconds since the run started",
        series=[Series(name="node execution", x=starts, y=[None] * len(starts),
                       meta={"end": ends, "lane": lanes})],
        note="Bars that overlap in time ran concurrently. The run's wall clock is the "
             "longest chain through them, not the sum of the bars.",
        height=max(260, 22 * len(lanes) + 90),
    )


def _node_shift(cell: dict, before: dict, after: dict,
                by_run: dict[str, list[dict]]) -> ChartSpec:
    """Which nodes the promotion actually moved.

    A cell's runtime changing says the optimizer did something; this says what.
    """
    def durations(run_id: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for e in by_run.get(run_id, []):
            if e["duration_ms"] is not None:
                node = short_node(e["node_id"])
                out[node] = out.get(node, 0.0) + e["duration_ms"] / 1000.0
        return out

    was, now = durations(before["run_id"]), durations(after["run_id"])
    nodes = sorted(set(was) | set(now), key=lambda n: -max(was.get(n, 0), now.get(n, 0)))
    nodes = nodes[:TOP_NODES]
    return ChartSpec(
        id=f"continuous-nodes-{cell['cell_id'][:12]}", kind="hbar",
        title=f"What the promotion moved — {cell['label']}",
        subtitle=f"Node time in run #{before['ordinal']} against run #{after['ordinal']}, "
                 "longest first",
        x_label="Node time (s)",
        series=[
            Series(name=f"before promoting (run #{before['ordinal']})", x=nodes,
                   y=[was.get(n) for n in nodes], color=STATUS["neutral"]),
            Series(name=f"at the promoted version (run #{after['ordinal']})", x=nodes,
                   y=[now.get(n) for n in nodes]),
        ],
        note="A node present in one run and missing from the other was added or removed "
             "by the optimizer — a materialization it inserted, or a view it folded away.",
        height=max(280, 26 * len(nodes) + 90),
    )


# --------------------------------------------------------------------------
# across cells
# --------------------------------------------------------------------------


def _summary_charts(cells: list[dict]) -> list[ChartSpec]:
    labels = [cell_label(c["project"], c["backend"], float(c["sf"]), c["variant"],
                         newline=True) for c in cells]
    charts = [ChartSpec(
        id="continuous-convergence", kind="grouped_bar",
        title="Runs each search spent before promoting",
        subtitle="Measured runs that executed a candidate or the DAG as authored, "
                 "before the result was promoted",
        y_label="Runs before promotion",
        series=[Series(name="runs spent converging", x=labels,
                       y=[(c["promotion_run"] - 1) if c["promotion_run"] else None
                          for c in cells])],
        value_labels=True, value_fmt="{:.0f}",
        note="A cell with no bar had not promoted anything by its last run.",
    )]
    if any(c["settled_speedup"] for c in cells):
        charts.append(ChartSpec(
            id="continuous-settled", kind="grouped_bar",
            title="Runtime once the result was promoted",
            subtitle="Median runtime at the promoted version, against the unoptimized "
                     "baseline that shared its tuning",
            y_label="Speedup (x)",
            series=[
                Series(name="while converging", x=labels,
                       y=[(c["baseline_s"] / c["converging_s"])
                          if (c["baseline_s"] and c["converging_s"]) else None
                          for c in cells]),
                Series(name="at the promoted version", x=labels,
                       y=[c["settled_speedup"] for c in cells]),
            ],
            hline=1.0, hline_label="unoptimized",
            value_labels=True,
            note="The left bar includes the runs the search was experimenting on, which "
                 "is what the DAG's owner actually experienced while it converged.",
        ))
    return charts


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------


def _tables(cells: list[dict]) -> list[Table]:
    summary = Table(
        key="continuous-summary", title="Convergence, per cell",
        columns=["Project", "Backend", "SF", "Variant", "Optimizations", "Status",
                 "Promoted version", "Promoted at run", "Runs at result",
                 "While converging", "Once promoted", "Speedup"],
        numeric=(6, 7, 8, 9, 10, 11),
        note="`Promoted at run` counts in execution order, warmups included — the same "
             "ordering the run-by-run charts use.",
    )
    for c in cells:
        summary.rows.append([
            c["project"], c["backend"], f"{float(c['sf']):g}", c["variant"],
            ", ".join(c["passes"] or []) or "-",
            "converged" if c["converged"] else "converging",
            c["promoted_version"] if c["promoted_version"] is not None else "-",
            c["promotion_run"] or "-",
            c["runs_at_result"],
            fmt_num(c["converging_s"], 3, "s"),
            fmt_num(c["settled_s"], 3, "s"),
            fmt_num(c["settled_speedup"], 2, "x"),
        ])

    detail = Table(
        key="continuous-runs", title="Every run, in order",
        columns=["Cell", "Run", "Phase", "Version", "Runtime", "Node time",
                 "Delivery", "Trial", "Resume", "Status"],
        numeric=(1, 3, 4, 5, 7, 8),
        note="One row per execution, in the order they happened. `Delivery` is "
             "`resumed` where the search cancelled a candidate mid-run and the "
             "incumbent finished it.",
    )
    for c in cells:
        for run in c["runs"]:
            detail.rows.append([
                c["label"], run["ordinal"], run["phase"],
                run["dag_version"] if run["dag_version"] is not None else "-",
                fmt_ms(run["engine_wall_ms"], 3),
                fmt_ms(run["node_time_ms"], 3),
                run["delivery"] or "direct",
                fmt_ms(run["trial_elapsed_ms"], 3),
                fmt_ms(run["resume_elapsed_ms"], 3),
                run["status"],
            ])
    return [summary, detail]
