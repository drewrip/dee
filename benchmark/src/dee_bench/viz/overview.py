"""The front page: what this sweep is, how far it got, and where its time went.

Every other page answers a question about the optimizer. This one answers
questions about the *run* — which cells have results, which are still missing,
what the whole thing cost — because a dashboard built from a partial dataset
should say so on the way in rather than let a reader discover it in a chart
with a hole in it.
"""

from __future__ import annotations

from typing import Any

from .query import cell_label, dicts, median, projection, tables
from .spec import ChartSpec, Kpi, Series, Study, Table
from .theme import STATUS


def build(con, meta: dict[str, Any] | None = None) -> Study:
    meta = meta or {}
    study = Study(
        key="overview", number=0, group="overview",
        title="Overview",
        subtitle="What this sweep measured, and how far it got",
        question="",
    )
    present = tables(con)
    if "cells" not in present:
        study.empty_reason = "No cells have been written yet."
        return study

    cells = dicts(con, f"""
        SELECT {projection(con, "cells",
                           ["cell_id", "project", "backend", "sf", "variant", "passes",
                            "optimization_mode", "repeat_mode", "repetitions",
                            "warmups"])}
        FROM cells ORDER BY backend, project, sf, variant
    """)
    runs = dicts(con, f"""
        SELECT {projection(con, "runs", ["cell_id", "phase", "status",
                                         "engine_wall_ms"])}
        FROM runs
    """) if "runs" in present else []
    opts = dicts(con, "SELECT cell_id, opt_wall_ms, status FROM optimizations") \
        if "optimizations" in present else []

    study.kpis = _kpis(cells, runs, opts, meta)
    coverage = _coverage(cells, runs)
    if coverage:
        study.charts.append(coverage)
    spend = _spend(cells, runs, opts)
    if spend:
        study.charts.append(spend)
    study.tables = [_inventory(cells, runs, opts)]
    return study


def _measured(runs: list[dict]) -> list[dict]:
    return [r for r in runs if r["phase"] == "measure" and r["status"] == "ok"]


def _kpis(cells, runs, opts, meta) -> list[Kpi]:
    with_results = {r["cell_id"] for r in runs}
    failed = [r for r in runs if r["status"] not in ("ok", None)]
    wall_s = sum((r["engine_wall_ms"] or 0) for r in runs) / 1000.0
    opt_s = sum((o["opt_wall_ms"] or 0) for o in opts) / 1000.0
    modes = {c["optimization_mode"] for c in cells if c["optimization_mode"]}

    kpis = [
        Kpi("Cells", f"{len(with_results)}/{len(cells)}",
            "Cells with at least one recorded run, out of the expanded matrix.",
            tone="good" if len(with_results) == len(cells) else "warning"),
        Kpi("Measured runs", str(len(_measured(runs))),
            "Warmups are recorded but never aggregated."),
        Kpi("Projects × backends",
            f"{len({c['project'] for c in cells})} × {len({c['backend'] for c in cells})}",
            ", ".join(sorted({c["backend"] for c in cells}))),
        Kpi("Variants", str(len({c["variant"] for c in cells})),
            ", ".join(sorted({c["variant"] for c in cells})[:6])),
        Kpi("Time in the engine", _duration(wall_s),
            f"Plus {_duration(opt_s)} optimizing." if opt_s else
            "Summed over every recorded execution, warmups included."),
    ]
    if failed:
        kpis.append(Kpi("Failed runs", str(len(failed)),
                        "Runs recorded with a status other than ok.", tone="critical"))
    if len(modes) > 1 or "continuous" in modes:
        kpis.append(Kpi("Optimization mode", " + ".join(sorted(modes)),
                        "Batch optimizes up front; continuous steps around the measured "
                        "runs. Their costs are not comparable."))
    return kpis


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def _coverage(cells: list[dict], runs: list[dict]) -> ChartSpec | None:
    """The matrix itself: median runtime per group and variant, blanks and all.

    Reading the sweep's shape off a table of cell ids is work; reading it off a
    grid is a glance, and the cells that have not run yet are the empty
    squares rather than absent rows.
    """
    measured: dict[str, list[float]] = {}
    for run in _measured(runs):
        measured.setdefault(run["cell_id"], []).append((run["engine_wall_ms"] or 0) / 1000.0)

    groups: list[str] = []
    variants: list[str] = []
    values: dict[tuple[str, str], float] = {}
    for cell in cells:
        group = cell_label(cell["project"], cell["backend"], float(cell["sf"]))
        if group not in groups:
            groups.append(group)
        if cell["variant"] not in variants:
            variants.append(cell["variant"])
        value = median(measured.get(cell["cell_id"], []))
        if value is not None:
            values[(group, cell["variant"])] = value
    if len(groups) * len(variants) < 2 or not values:
        return None

    matrix = [[values.get((g, v)) for v in variants] for g in groups]
    return ChartSpec(
        id="overview-coverage", kind="heatmap",
        title="The matrix, and what has run",
        subtitle="Median measured runtime per cell; hatched squares have no measurement yet",
        x_label="Variant", y_label="",
        matrix=matrix, x_ticks=variants, y_ticks=groups, z_fmt="{:.2f}",
        z_label="Runtime (s)",
        note="Darker is slower. Compare within a row: across rows the DAGs and scale "
             "factors differ, so the colours are not one scale.",
    )


def _spend(cells: list[dict], runs: list[dict], opts: list[dict]) -> ChartSpec | None:
    """Where the sweep's own wall clock went, cell by cell."""
    by_cell: dict[str, dict[str, float]] = {}
    for run in runs:
        bucket = by_cell.setdefault(run["cell_id"], {"measure": 0.0, "warmup": 0.0,
                                                     "optimize": 0.0})
        key = "measure" if run["phase"] == "measure" else "warmup"
        bucket[key] += (run["engine_wall_ms"] or 0) / 1000.0
    for opt in opts:
        by_cell.setdefault(opt["cell_id"], {"measure": 0.0, "warmup": 0.0,
                                            "optimize": 0.0})["optimize"] += \
            (opt["opt_wall_ms"] or 0) / 1000.0
    if not by_cell:
        return None

    labelled = [(cell_label(c["project"], c["backend"], float(c["sf"]), c["variant"]),
                 by_cell.get(c["cell_id"]))
                for c in cells]
    labelled = [(label, spend) for label, spend in labelled if spend]
    if not labelled:
        return None
    labelled.sort(key=lambda item: -sum(item[1].values()))
    labelled = labelled[:24]
    labels = [label for label, _ in labelled]

    return ChartSpec(
        id="overview-spend", kind="stacked_hbar",
        title="Where the sweep's time went",
        subtitle="Wall clock per cell, split between optimizing and running",
        x_label="Wall time (s)",
        series=[
            Series(name="optimizing", x=labels,
                   y=[s["optimize"] for _, s in labelled], color=STATUS["warning"]),
            Series(name="warmups", x=labels,
                   y=[s["warmup"] for _, s in labelled], color=STATUS["neutral"]),
            Series(name="measured runs", x=labels,
                   y=[s["measure"] for _, s in labelled]),
        ],
        note="A continuous cell shows no optimizing time by construction: it spends the "
             "runs it was going to perform anyway, so its cost is inside the measured "
             "band.",
    )


def _inventory(cells: list[dict], runs: list[dict], opts: list[dict]) -> Table:
    counts: dict[str, dict[str, int]] = {}
    for run in runs:
        bucket = counts.setdefault(run["cell_id"], {"measure": 0, "warmup": 0, "failed": 0})
        if run["status"] not in ("ok", None):
            bucket["failed"] += 1
        elif run["phase"] == "measure":
            bucket["measure"] += 1
        else:
            bucket["warmup"] += 1
    opt_status = {o["cell_id"]: o["status"] for o in opts}

    table = Table(
        key="overview-cells", title="Every cell in the matrix",
        columns=["Project", "Backend", "SF", "Variant", "Passes", "Mode", "Repeat",
                 "Measured", "Warmups", "Failed", "Optimization", "Cell id"],
        numeric=(7, 8, 9),
        note="`Cell id` is the join key for every result table: "
             "`SELECT * FROM 'results/runs/cell_id=<id>/*.parquet'`.",
    )
    for cell in cells:
        got = counts.get(cell["cell_id"], {})
        table.rows.append([
            cell["project"], cell["backend"], f"{float(cell['sf']):g}", cell["variant"],
            ", ".join(cell["passes"] or []) or "none",
            cell["optimization_mode"] or "batch",
            cell["repeat_mode"] or "group",
            got.get("measure", 0), got.get("warmup", 0), got.get("failed", 0),
            opt_status.get(cell["cell_id"], "-"),
            cell["cell_id"][:12],
        ])
    return table
