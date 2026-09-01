"""Derive the payback table from raw results (study 3).

Kept separate from the runner so it can be recomputed from the parquet dataset
at any time — after a partial run, or with a changed definition — without
re-running a single benchmark.

Payback answers: *how many times must this DAG run before the time spent
optimizing it is repaid?*

    payback_runs = optimization_cost / savings_per_run

`savings_per_run` compares the optimized variant against the matching
unoptimized cell — same project, backend and scale factor. When a variant is no
faster than its baseline the cost is never repaid, and payback is null rather
than a large or negative number that would average into nonsense.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

from .store import ResultStore, connect

# Cell identity minus the variant: what a variant is compared against.
GROUP_KEYS = ("project", "backend", "sf")


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def _bootstrap_payback_ci(
    baseline: list[float], variant: list[float], cost_s: float,
    iterations: int = 2000, seed: int = 0,
) -> tuple[float | None, float | None]:
    """Bootstrap a 95% CI on payback_runs by resampling the repetitions.

    With a handful of repetitions per cell, a point estimate of payback hides
    how much of it is measurement noise; the interval makes that visible.
    Resamples where the variant is not faster contribute no finite payback, so
    the interval is reported over the finite ones only, and is None when too
    few remain to mean anything.
    """
    if not baseline or not variant or cost_s <= 0:
        return None, None
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(iterations):
        b = _median([rng.choice(baseline) for _ in baseline])
        v = _median([rng.choice(variant) for _ in variant])
        if b is None or v is None:
            continue
        savings = b - v
        if savings > 0:
            estimates.append(cost_s / savings)
    if len(estimates) < iterations * 0.5:
        # More than half the resamples showed no improvement, so any interval
        # would be describing a subset that isn't representative.
        return None, None
    estimates.sort()
    lo = estimates[int(0.025 * len(estimates))]
    hi = estimates[min(int(0.975 * len(estimates)), len(estimates) - 1)]
    return lo, hi


def compute_payback(run_dir: str | Path) -> list[dict[str, Any]]:
    """Compute payback rows from a run directory's parquet results."""
    con = connect(run_dir)
    tables = {r[0] for r in con.sql("SHOW TABLES").fetchall()}
    if not {"cells", "runs"} <= tables:
        return []

    # Per-repetition measurements, so medians and the bootstrap use the real
    # distribution rather than a pre-aggregated mean.
    measurements = con.sql("""
        SELECT c.cell_id, c.run_name, c.project, c.backend, c.sf, c.variant,
               r.engine_wall_ms / 1000.0 AS wall_s,
               r.cpu_seconds,
               r.dag_version
        FROM runs r JOIN cells c USING (cell_id)
        WHERE r.phase = 'measure' AND r.status = 'ok' AND r.engine_wall_ms IS NOT NULL
    """).fetchall()
    if not measurements:
        return []

    # A continuous optimization converges partway through its cell's runs, so
    # the cell's measurements are not all of the same DAG: the ones before it
    # promoted include its baseline and every candidate it tried. Comparing
    # those against an unoptimized baseline would answer a question nobody
    # asked -- "how fast is a DAG while being experimented on". The promoted
    # version is what separates them, and only runs at it are runs of the
    # optimized DAG.
    converged_version: dict[str, int] = {}
    if "optimizations" in tables:
        for cell_id, version in con.sql(
            "SELECT cell_id, result_version FROM optimizations "
            "WHERE status = 'converged' AND result_version IS NOT NULL"
        ).fetchall():
            converged_version[cell_id] = int(version)

    by_cell: dict[str, dict[str, Any]] = {}
    for cell_id, run_name, project, backend, sf, variant, wall_s, cpu_s, version in measurements:
        wanted = converged_version.get(cell_id)
        if wanted is not None and version is not None and int(version) != wanted:
            continue
        entry = by_cell.setdefault(cell_id, {
            "cell_id": cell_id, "run_name": run_name, "project": project,
            "backend": backend, "sf": float(sf), "variant": variant,
            "wall": [], "cpu": [],
        })
        entry["wall"].append(float(wall_s))
        if cpu_s is not None:
            entry["cpu"].append(float(cpu_s))

    costs: dict[str, dict[str, Any]] = {}
    if "optimizations" in tables:
        # 'ok' is a finished batch optimization; 'converged' is a continuous
        # one that reached an answer. A 'converging' cell ran out of runs
        # before deciding, and has no result to attribute a cost to.
        for cell_id, wall_ms, cpu_s in con.sql(
            "SELECT cell_id, opt_wall_ms, opt_cpu_seconds FROM optimizations "
            "WHERE status IN ('ok', 'converged')"
        ).fetchall():
            costs[cell_id] = {
                "wall_s": (wall_ms or 0) / 1000.0,
                "cpu_s": cpu_s,
            }

    # Baselines are the cells that ran no optimizer passes. Read from `cells`
    # rather than inferred from the absence of a cost row: a cell whose
    # optimization did not converge also has no cost, and treating it as a
    # baseline would make every other variant look faster than it should.
    baseline_ids: set[str] = set()
    for (cell_id,) in con.sql(
        "SELECT cell_id FROM cells WHERE passes IS NULL OR len(passes) = 0"
    ).fetchall():
        baseline_ids.add(cell_id)

    baselines: dict[tuple, dict[str, Any]] = {}
    for entry in by_cell.values():
        if entry["cell_id"] in baseline_ids:
            baselines[tuple(entry[k] for k in GROUP_KEYS)] = entry

    rows: list[dict[str, Any]] = []
    for entry in by_cell.values():
        if entry["cell_id"] in baseline_ids:
            continue  # this cell *is* a baseline
        cost = costs.get(entry["cell_id"])
        if cost is None:
            continue  # optimized, but it never reached a result to price
        base = baselines.get(tuple(entry[k] for k in GROUP_KEYS))
        if base is None:
            continue  # nothing to compare against, e.g. baseline cell failed

        base_wall = _median(base["wall"])
        var_wall = _median(entry["wall"])
        base_cpu = _median(base["cpu"])
        var_cpu = _median(entry["cpu"])

        savings_wall = (base_wall - var_wall) if (base_wall is not None and var_wall is not None) else None
        savings_cpu = (base_cpu - var_cpu) if (base_cpu is not None and var_cpu is not None) else None

        lo, hi = _bootstrap_payback_ci(base["wall"], entry["wall"], cost["wall_s"])
        rows.append({
            "run_name": entry["run_name"],
            "project": entry["project"],
            "backend": entry["backend"],
            "sf": entry["sf"],
            "variant": entry["variant"],
            "cell_id": entry["cell_id"],
            "baseline_cell_id": base["cell_id"],
            "opt_cost_wall_s": cost["wall_s"],
            "opt_cost_cpu_s": cost["cpu_s"],
            "baseline_wall_s": base_wall,
            "variant_wall_s": var_wall,
            "baseline_cpu_s": base_cpu,
            "variant_cpu_s": var_cpu,
            "savings_per_run_wall_s": savings_wall,
            "savings_per_run_cpu_s": savings_cpu,
            "speedup": (base_wall / var_wall) if (base_wall and var_wall) else None,
            "payback_runs_wall": _payback(cost["wall_s"], savings_wall),
            "payback_runs_cpu": _payback(cost["cpu_s"], savings_cpu),
            "payback_runs_wall_lo": lo,
            "payback_runs_wall_hi": hi,
            "n_baseline": len(base["wall"]),
            "n_variant": len(entry["wall"]),
        })

    rows.sort(key=lambda r: (r["backend"], r["project"], r["sf"], r["variant"]))
    return rows


def _payback(cost: float | None, savings: float | None) -> float | None:
    """Runs to repay `cost` at `savings` per run, or None if never repaid."""
    if cost is None or savings is None or savings <= 0:
        return None
    value = cost / savings
    return value if math.isfinite(value) else None


def analyze(run_dir: str | Path, verbosity=None) -> int:
    """Recompute every derived table. Returns rows written."""
    from .schema import Verbosity

    rows = compute_payback(run_dir)
    store = ResultStore(run_dir, verbosity or Verbosity.FULL)
    return store.write_derived("payback", rows)
