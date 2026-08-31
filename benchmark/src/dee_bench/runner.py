"""Execute one benchmark cell: optimize the DAG, then measure it.

A cell runs in two phases:

**optimize** (skipped for baseline variants)
    ``POST /v1/dags/{name}/optimize`` with the cell's pass list and options,
    sampled throughout. Produces a new DAG version and an ``OptimizeReport`` --
    the cost side of the payback analysis.

**measure**
    A trigger with the cell's ``warmups`` and ``repetitions``. The whole series
    runs inside one server against one already-warm connection pool, so
    ``engine_wall_ms`` measures the DAG rather than process startup and pool
    construction.

Both phases go over HTTP to a server the sweep started. The reports come back
as the same ``OptimizeReport`` and ``ProfileReport`` shapes dee has always
produced, so everything downstream of parsing is unchanged.

Every failure is contained: a cell that fails records its error and leaves the
sweep running, because a long matrix must not be lost to one bad project.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import DEE_OPT_BY_NAME, BenchConfig
from .infra.base import BackendContext
from .matrix import Cell
from .sampler import PhaseMetrics, PhaseSampler, samples_to_rows
from .store import ResultStore
from .server import ApiError, DeeClient, ServerError
from .workload import PreparedProject, WorkloadError, convert_dag, register


@dataclass
class CellResult:
    cell_id: str
    status: str = "done"
    error: str | None = None
    measured_runs: int = 0
    rows: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


def build_optimize_request(cell: Cell) -> dict[str, Any]:
    """Render this cell's optimizer settings as an optimize request body.

    Passes are always stated explicitly rather than relying on dee's defaults,
    so a variant's pass set is exactly its list and does not shift when those
    defaults change.
    """
    config: dict[str, Any] = {
        "run_hmp_pass": "hmp" in cell.passes,
        "run_omp_pass": "omp" in cell.passes,
        "run_pushdown_pass": "pushdown" in cell.passes,
    }
    for key, value in sorted(cell.dee_opt.items()):
        if value is None:
            continue
        spec = DEE_OPT_BY_NAME[key]
        config[spec.config_field] = spec.config_value(value)

    return {
        "config": config,
        "save_as_version": True,
        "explain": True,
    }


def build_run_request(cell: Cell, sample_interval_ms: int,
                      collect_plans: bool) -> dict[str, Any]:
    return {
        "warmups": cell.warmups,
        "repetitions": cell.repetitions,
        "cleanup_before": True,
        "collect_plans": collect_plans,
        "sample_interval_ms": sample_interval_ms,
    }


class CellRunner:
    """Runs a single cell and returns the rows it produced."""

    def __init__(self, cfg: BenchConfig, store: ResultStore, run_dir: Path,
                 client: DeeClient, server_pid: int | None = None, log=print):
        self.cfg = cfg
        self.store = store
        self.run_dir = run_dir
        self.client = client
        # What resource sampling attaches to. None when the sweep is attached
        # to a server it did not start, in which case process-level CPU and
        # memory are simply not recorded.
        self.server_pid = server_pid
        self.log = log
        # Set per cell by the sweep, because it depends on which backend the
        # cell runs against.
        self._cgroup = None

    def set_cgroup(self, cgroup) -> None:
        """Container cgroup to sample alongside the server, if any."""
        self._cgroup = cgroup

    def run(self, cell: Cell, prepared: PreparedProject, ctx: BackendContext) -> CellResult:
        result = CellResult(cell_id=cell.cell_id)
        artifacts = self.run_dir / "artifacts" / cell.cell_id
        artifacts.mkdir(parents=True, exist_ok=True)

        try:
            convert_dag(self.cfg.dee_bin, prepared)

            # One DAG name per cell. Cells differ by optimizer settings, and
            # sharing a name across them would interleave their histories and
            # make the optimizer's derived-from chain meaningless.
            dag_name = f"c{cell.cell_id}"
            target, submitted = register(self.client, prepared, dag_name)
            version = submitted["version"]
            self._record_graph(cell, dag_name, version, "unopt", result)

            dag_variant = "unopt"
            if not cell.is_baseline:
                version = self._optimize(cell, dag_name, artifacts, result)
                dag_variant = "optimized"
                self._record_graph(cell, dag_name, version, "optimized", result)

            self._measure(cell, dag_name, version, ctx, artifacts, dag_variant, result)
        except Exception as e:  # noqa: BLE001 - a cell failure must not end the sweep
            result.status = "failed"
            result.error = f"{type(e).__name__}: {e}"
        return result

    # -- optimize ----------------------------------------------------------

    def _optimize(self, cell: Cell, dag_name: str, artifacts: Path,
                  result: CellResult) -> int:
        body = build_optimize_request(cell)
        (artifacts / "optimize_request.json").write_text(json.dumps(body, indent=2))

        started = datetime.now(timezone.utc)
        with self._sample_phase() as sampler:
            accepted = self.client.optimize(dag_name, body, self.cfg.execution.timeout_s)
        metrics, wall_ms = sampler.result
        finished = datetime.now(timezone.utc)

        optimization_id = accepted["optimization_id"]
        detail = self.client.optimization(optimization_id)
        if detail["status"] != "succeeded":
            raise WorkloadError(
                f"optimize {detail['status']}: {detail.get('error') or 'no detail recorded'}"
            )

        report = self.client.optimization_report(optimization_id)
        (artifacts / "opt_report.json").write_text(json.dumps(report, indent=2))
        try:
            (artifacts / "explain.html").write_text(
                self.client.optimization_explain(optimization_id)
            )
        except (ApiError, ServerError):
            pass  # explain is a convenience, not a result

        opt_run_id = f"{cell.cell_id}-opt"
        result.rows.setdefault("optimizations", []).append({
            "cell_id": cell.cell_id,
            "started_at": started,
            "finished_at": finished,
            # dee's own measurement of the optimize phase, which excludes the
            # API round trip; `wall_ms` from the client is the outer bound.
            "opt_wall_ms": report.get("wall_ms", wall_ms),
            "opt_cpu_seconds": metrics.cpu_seconds,
            "opt_peak_rss_bytes": metrics.peak_rss_bytes,
            "dag_runs_used": report.get("dag_runs_used"),
            "baseline_runtime_ms": report.get("baseline_runtime_ms"),
            "final_runtime_ms": report.get("final_runtime_ms"),
            "total_changes_applied": sum(
                p.get("changes_applied", 0) for p in report.get("passes", [])
            ),
            "nodes_before": report.get("nodes_before"),
            "nodes_after": report.get("nodes_after"),
            "status": "ok",
            "error": None,
        })

        for p in report.get("passes", []):
            result.rows.setdefault("pass_stats", []).append({
                "cell_id": cell.cell_id,
                "pass_name": p.get("pass"),
                "pass_order": p.get("order"),
                "wall_ms": p.get("wall_ms"),
                "dag_runs_used": p.get("dag_runs_used"),
                "changes_applied": p.get("changes_applied"),
                "candidates_considered": p.get("candidates_considered"),
                "working_set_size": p.get("working_set_size"),
                "detail": json.dumps(p.get("detail")),
            })
            for it in p.get("iterations", []):
                agg = _aggregate_engine_samples(it.get("system_samples") or [])
                result.rows.setdefault("pass_iterations", []).append({
                    "cell_id": cell.cell_id,
                    "pass_name": p.get("pass"),
                    "iteration": it.get("iteration"),
                    "runtime_ms": it.get("runtime_ms"),
                    "combo": it.get("combo") or [],
                    "outcome": it.get("outcome"),
                    "cpu_seconds": agg.get("cpu_seconds"),
                    "peak_rss_bytes": agg.get("peak_rss_bytes"),
                })

        result.rows.setdefault("system_samples", []).extend(
            samples_to_rows(metrics, cell_id=cell.cell_id, run_id=opt_run_id, phase="optimize")
        )

        # The optimizer saved its rewrite as a new version; that is what gets
        # measured. When it changed nothing the content hash matches the input
        # and the source version comes back, which is correct.
        return detail.get("result_version") or detail["source_version"]

    # -- measure -----------------------------------------------------------

    def _measure(self, cell: Cell, dag_name: str, version: int, ctx: BackendContext,
                 artifacts: Path, dag_variant: str, result: CellResult) -> None:
        body = build_run_request(
            cell,
            self.cfg.execution.sample_interval_ms,
            collect_plans=self.store.records("plans"),
        )
        body["version"] = version
        (artifacts / "run_request.json").write_text(json.dumps(body, indent=2))

        with self._sample_phase() as sampler:
            triggered = self.client.trigger(dag_name, body, self.cfg.execution.timeout_s)
        metrics, wall_ms = sampler.result

        group_id = triggered["run_group_id"]
        group = self.client.run_group(group_id)
        if group["status"] != "succeeded":
            raise WorkloadError(
                f"run group {group['status']}: {group.get('error') or 'no detail recorded'}"
            )

        report = self.client.group_report(group_id)
        (artifacts / "run_report.json").write_text(json.dumps(report, indent=2))

        runs = report.get("runs", [])
        if not runs:
            raise WorkloadError("the run produced no measurements")

        # One sampler covers the whole phase, which contains every repetition.
        # Splitting its totals per repetition would be a fiction, so they are
        # apportioned by each repetition's share of engine time and the basis
        # is recorded in the schema.
        total_engine_ms = sum(r.get("duration_ms", 0) for r in runs) or 1

        # Pair the report's repetitions with their persisted run rows, so the
        # server's own run ids are what the parquet records.
        server_runs = group.get("runs", [])
        by_key = {(r["phase"], r["rep_index"]): r for r in server_runs}

        for r in runs:
            share = r.get("duration_ms", 0) / total_engine_ms
            key = (r.get("phase", "measure"), r.get("rep_index", 0))
            server_run = by_key.get(key)
            run_id = server_run["run_id"] if server_run else str(uuid.uuid4())
            node_execs = r.get("node_executions", [])

            result.rows.setdefault("runs", []).append({
                "cell_id": cell.cell_id,
                "run_id": run_id,
                "dag_variant": dag_variant,
                "phase": r.get("phase", "measure"),
                "rep_index": r.get("rep_index", 0),
                "started_at": _ts(r.get("run_started_at")),
                "finished_at": _ts(r.get("run_finished_at")),
                "engine_wall_ms": r.get("duration_ms"),
                "phase_wall_ms": int(wall_ms * share),
                "node_time_ms": r.get("time_executing_nodes_ms"),
                "node_count": len(r.get("graph", {}).get("nodes", [])),
                "rows_produced": sum((n.get("rows_produced") or 0) for n in node_execs) or None,
                "cpu_seconds": (metrics.cpu_seconds or 0) * share if metrics.cpu_seconds else None,
                "peak_rss_bytes": metrics.peak_rss_bytes,
                "peak_engine_mem_bytes": _peak(r.get("system_samples", []), "memory_bytes"),
                "read_bytes": int((metrics.read_bytes or 0) * share) if metrics.read_bytes else None,
                "written_bytes": int((metrics.written_bytes or 0) * share) if metrics.written_bytes else None,
                "db_size_bytes": _peak(r.get("system_samples", []), "disk_bytes"),
                "plan_time_basis": ctx.plan_time_basis,
                "status": "ok",
                "error": None,
            })
            if r.get("phase", "measure") == "measure":
                result.measured_runs += 1

            mats = {n["id"]: n.get("materialization") for n in r.get("graph", {}).get("nodes", [])}
            for n in node_execs:
                result.rows.setdefault("node_executions", []).append({
                    "cell_id": cell.cell_id,
                    "run_id": run_id,
                    "node_id": n.get("node_id"),
                    "materialization": mats.get(n.get("node_id")),
                    "started_at": _ts(n.get("start")),
                    "duration_ms": n.get("duration_ms"),
                    "rows_produced": n.get("rows_produced"),
                    "has_plan": bool(n.get("plan")),
                })
                if n.get("plan"):
                    result.rows.setdefault("plans", []).append({
                        "cell_id": cell.cell_id,
                        "run_id": run_id,
                        "node_id": n.get("node_id"),
                        "plan_format": f"{cell.backend}_json",
                        "plan_json": n["plan"],
                    })

            # dee's own connector samples, kept as supplementary engine-level
            # detail alongside the harness's external sampling.
            for sample in r.get("system_samples", []):
                result.rows.setdefault("system_samples", []).append({
                    "cell_id": cell.cell_id,
                    "run_id": run_id,
                    "phase": r.get("phase", "measure"),
                    "source": "engine_internal",
                    "elapsed_ms": sample.get("elapsed_ms"),
                    "timestamp": _ts(sample.get("timestamp")),
                    "cpu_seconds_cum": None,
                    "rss_bytes": None,
                    "engine_mem_bytes": sample.get("memory_bytes"),
                    "read_bytes": sample.get("read_bytes"),
                    "written_bytes": sample.get("written_bytes"),
                })

        measure_run_id = f"{cell.cell_id}-measure"
        result.rows.setdefault("system_samples", []).extend(
            samples_to_rows(metrics, cell_id=cell.cell_id, run_id=measure_run_id, phase="measure")
        )

    # -- helpers -----------------------------------------------------------

    @contextmanager
    def _sample_phase(self):
        """Sample the server for the duration of a phase.

        The sampler attaches to the long-lived server rather than to a
        short-lived child. CPU and IO stay correct because `PhaseSampler`
        baselines its counters at attach time and reports deltas. Peak RSS does
        not: it is absolute, so a previous cell's buffer pool is still resident.
        `schema.py` documents that qualification on the column.
        """
        sampler = PhaseSampler(self.cfg.execution.sample_interval_ms, cgroup=self._cgroup).start()
        if self.server_pid is not None:
            sampler.attach(self.server_pid)
        handle = _PhaseHandle()
        started = time.monotonic()
        try:
            yield handle
        finally:
            handle.result = (sampler.stop(), int((time.monotonic() - started) * 1000))

    def _record_graph(self, cell: Cell, dag_name: str, version: int, variant: str,
                      result: CellResult) -> None:
        if not self.store.records("dag_graph"):
            return
        # `out_degree` and `paths_to_sinks` come from the server, which derives
        # them with the same `dee::graph::Graph` code the optimizer uses. The
        # harness used to recompute `paths_to_sinks` itself with a subtly
        # different definition -- see the column's doc in schema.py.
        detail = self.client.dag_version(dag_name, version)
        full = self.store.verbosity.name == "FULL"
        for node in detail.get("nodes", []):
            result.rows.setdefault("dag_graph", []).append({
                "cell_id": cell.cell_id,
                "dag_variant": variant,
                "node_id": node.get("node_id"),
                "materialization": node.get("materialize"),
                "depends_on": node.get("depends_on") or [],
                "out_degree": node.get("out_degree"),
                "paths_to_sinks": node.get("paths_to_sinks"),
                "query_text": node.get("query_text") if full else None,
            })


class _PhaseHandle:
    """Carries a phase's sampler results out of the context manager."""

    result: tuple = (None, 0)


def _aggregate_engine_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Peak figures from dee's per-iteration connector samples."""
    if not samples:
        return {}
    mems = [s.get("memory_bytes") for s in samples if s.get("memory_bytes")]
    cpus = [s.get("cpu_percent") for s in samples if s.get("cpu_percent") is not None]
    elapsed = max((s.get("elapsed_ms") or 0) for s in samples) / 1000.0
    return {
        "peak_rss_bytes": max(mems) if mems else None,
        # dee samples CPU as a percentage, so this is a coarse mean-times-time
        # estimate, not the counter-derived figure the `runs` table carries.
        "cpu_seconds": (sum(cpus) / len(cpus) / 100.0 * elapsed) if cpus else None,
    }


def _peak(samples: list[dict[str, Any]], key: str) -> int | None:
    values = [s.get(key) for s in samples if s.get(key)]
    return max(values) if values else None


def _ts(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
