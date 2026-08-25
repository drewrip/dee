"""Execute one benchmark cell: optimize the DAG, then measure it.

A cell runs in two phases:

**optimize** (skipped for baseline variants)
    ``dee-cli opt`` with the cell's pass list and options, sampled throughout.
    Produces the optimized DAG and an ``OptimizeReport`` — the cost side of the
    payback analysis.

**measure**
    ``dee-cli run --warmups W --repeat N`` on the resulting DAG. Repetitions
    happen inside a single dee process, so ``engine_wall_ms`` measures the DAG
    rather than CLI startup, pool construction and cleanup.

Every failure is contained: a cell that fails records its error and leaves the
sweep running, because a long matrix must not be lost to one bad project.
"""

from __future__ import annotations

import json
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import DEE_OPT_BY_NAME, BenchConfig
from .infra.base import BackendContext
from .matrix import Cell
from .sampler import PhaseMetrics, PhaseSampler, samples_to_rows
from .store import ResultStore
from .workload import PreparedProject, WorkloadError, convert_dag


@dataclass
class CellResult:
    cell_id: str
    status: str = "done"
    error: str | None = None
    measured_runs: int = 0
    rows: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


def build_opt_command(dee_cli: Path, cell: Cell, prepared: PreparedProject,
                      out_dag: Path, report_json: Path, explain: Path | None) -> list[str]:
    """Render `dee-cli opt` for this cell.

    Passes are always specified with ``--enable``, never ``--disable``: enable
    starts from everything off, so the resulting pass set is exactly the
    variant's list and does not depend on dee's defaults changing.
    """
    cmd = [
        str(dee_cli), "opt",
        "-c", str(prepared.connections_json),
        "-t", prepared.target,
        "-o", str(out_dag),
        "--report-json", str(report_json),
        "--enable", ",".join(cell.passes),
    ]
    for key, value in sorted(cell.dee_opt.items()):
        spec = DEE_OPT_BY_NAME[key]
        if value is None:
            continue
        if spec.kind == "bool":
            # A plain flag is emitted when the option is True. A negated flag
            # (e.g. hmp_use_pushdown -> --hmp-no-pushdown) turns off behaviour
            # dee enables by default, so it is emitted when the option is False.
            if bool(value) == spec.negated:
                continue
            cmd.append(spec.flag)
        else:
            cmd += [spec.flag, str(value)]
    if explain is not None:
        cmd.append(f"--explain={explain}")
    cmd.append(str(prepared.dag_json))
    return cmd


def build_run_command(dee_cli: Path, cell: Cell, prepared: PreparedProject,
                      dag: Path, report_json: Path, sample_interval_ms: int) -> list[str]:
    return [
        str(dee_cli), "run",
        "-c", str(prepared.connections_json),
        "-t", prepared.target,
        "--repeat", str(cell.repetitions),
        "--warmups", str(cell.warmups),
        "--report-json", str(report_json),
        "--profile-interval-ms", str(sample_interval_ms),
        str(dag),
    ]


class CellRunner:
    """Runs a single cell and returns the rows it produced."""

    def __init__(self, cfg: BenchConfig, store: ResultStore, run_dir: Path, log=print):
        self.cfg = cfg
        self.store = store
        self.run_dir = run_dir
        self.log = log

    def run(self, cell: Cell, prepared: PreparedProject, ctx: BackendContext) -> CellResult:
        result = CellResult(cell_id=cell.cell_id)
        artifacts = self.run_dir / "artifacts" / cell.cell_id
        artifacts.mkdir(parents=True, exist_ok=True)

        try:
            convert_dag(self.cfg.dee_cli, prepared)
            self._record_graph(cell, prepared.dag_json, "unopt", result)

            dag_to_measure = prepared.dag_json
            dag_variant = "unopt"

            if not cell.is_baseline:
                dag_to_measure = self._optimize(cell, prepared, ctx, artifacts, result)
                dag_variant = "optimized"
                self._record_graph(cell, dag_to_measure, "optimized", result)

            self._measure(cell, prepared, ctx, artifacts, dag_to_measure, dag_variant, result)
        except Exception as e:  # noqa: BLE001 - a cell failure must not end the sweep
            result.status = "failed"
            result.error = f"{type(e).__name__}: {e}"
        return result

    # -- optimize ----------------------------------------------------------

    def _optimize(self, cell: Cell, prepared: PreparedProject, ctx: BackendContext,
                  artifacts: Path, result: CellResult) -> Path:
        out_dag = artifacts / "dag_opt.json"
        report_json = artifacts / "opt_report.json"
        explain = artifacts / "explain.html"
        cmd = build_opt_command(self.cfg.dee_cli, cell, prepared, out_dag, report_json, explain)
        (artifacts / "opt_command.txt").write_text(" ".join(cmd) + "\n")

        started = datetime.now(timezone.utc)
        metrics, proc, wall_ms = self._sampled(cmd, ctx, self.cfg.execution.timeout_s)
        finished = datetime.now(timezone.utc)
        (artifacts / "opt.log").write_text((proc.stdout or "") + (proc.stderr or ""))

        if proc.returncode != 0 or not report_json.exists():
            tail = "\n".join((proc.stderr or proc.stdout or "").strip().splitlines()[-20:])
            raise WorkloadError(f"dee-cli opt failed ({proc.returncode}):\n{tail}")

        report = json.loads(report_json.read_text())
        opt_run_id = f"{cell.cell_id}-opt"

        result.rows.setdefault("optimizations", []).append({
            "cell_id": cell.cell_id,
            "started_at": started,
            "finished_at": finished,
            # dee's own measurement of the optimize phase, which excludes CLI
            # startup; `wall_ms` from the subprocess is the outer bound.
            "opt_wall_ms": report.get("wall_ms", wall_ms),
            "opt_cpu_seconds": metrics.cpu_seconds,
            "opt_peak_rss_bytes": metrics.peak_rss_bytes,
            "dag_runs_used": report.get("dag_runs_used"),
            "baseline_runtime_ms": report.get("baseline_runtime_ms"),
            "final_runtime_ms": report.get("final_runtime_ms"),
            "total_changes_applied": sum(p.get("changes_applied", 0) for p in report.get("passes", [])),
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
        return out_dag

    # -- measure -----------------------------------------------------------

    def _measure(self, cell: Cell, prepared: PreparedProject, ctx: BackendContext,
                 artifacts: Path, dag: Path, dag_variant: str, result: CellResult) -> None:
        report_json = artifacts / "run_report.json"
        cmd = build_run_command(
            self.cfg.dee_cli, cell, prepared, dag, report_json,
            self.cfg.execution.sample_interval_ms,
        )
        (artifacts / "run_command.txt").write_text(" ".join(cmd) + "\n")

        metrics, proc, wall_ms = self._sampled(cmd, ctx, self.cfg.execution.timeout_s)
        (artifacts / "run.log").write_text((proc.stdout or "") + (proc.stderr or ""))

        if proc.returncode != 0 or not report_json.exists():
            tail = "\n".join((proc.stderr or proc.stdout or "").strip().splitlines()[-20:])
            raise WorkloadError(f"dee-cli run failed ({proc.returncode}):\n{tail}")

        runs = json.loads(report_json.read_text()).get("runs", [])
        if not runs:
            raise WorkloadError("dee-cli run produced no runs")

        # One sampler covers the whole subprocess, which contains every
        # repetition. Splitting its totals per repetition would be a fiction,
        # so they are apportioned by each repetition's share of engine time and
        # the basis is recorded in the schema.
        total_engine_ms = sum(r.get("duration_ms", 0) for r in runs) or 1

        for r in runs:
            share = r.get("duration_ms", 0) / total_engine_ms
            run_id = str(uuid.uuid4())
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
                "subprocess_wall_ms": int(wall_ms * share),
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
            for s in r.get("system_samples", []):
                result.rows.setdefault("system_samples", []).append({
                    "cell_id": cell.cell_id,
                    "run_id": run_id,
                    "phase": r.get("phase", "measure"),
                    "source": "engine_internal",
                    "elapsed_ms": s.get("elapsed_ms"),
                    "timestamp": _ts(s.get("timestamp")),
                    "cpu_seconds_cum": None,
                    "rss_bytes": None,
                    "engine_mem_bytes": s.get("memory_bytes"),
                    "read_bytes": s.get("read_bytes"),
                    "written_bytes": s.get("written_bytes"),
                })

        measure_run_id = f"{cell.cell_id}-measure"
        result.rows.setdefault("system_samples", []).extend(
            samples_to_rows(metrics, cell_id=cell.cell_id, run_id=measure_run_id, phase="measure")
        )

    # -- helpers -----------------------------------------------------------

    def _sampled(self, cmd: list[str], ctx: BackendContext, timeout: int
                 ) -> tuple[PhaseMetrics, subprocess.CompletedProcess, int]:
        """Run `cmd` with the external sampler attached for its lifetime."""
        sampler = PhaseSampler(self.cfg.execution.sample_interval_ms, cgroup=ctx.cgroup).start()
        t0 = time.monotonic()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        sampler.attach(proc.pid)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            returncode = -1
            stderr = (stderr or "") + f"\n[dee-bench] killed after {timeout}s timeout"
        wall_ms = int((time.monotonic() - t0) * 1000)
        metrics = sampler.stop()
        completed = subprocess.CompletedProcess(cmd, returncode, stdout, stderr)
        return metrics, completed, wall_ms

    def _record_graph(self, cell: Cell, dag_json: Path, variant: str, result: CellResult) -> None:
        if not self.store.records("dag_graph"):
            return
        dag = json.loads(dag_json.read_text())
        nodes = dag.get("nodes", [])
        children: dict[str, int] = {}
        for n in nodes:
            for dep in n.get("depends_on", []):
                children[dep] = children.get(dep, 0) + 1
        full = self.store.verbosity.name == "FULL"
        for n in nodes:
            result.rows.setdefault("dag_graph", []).append({
                "cell_id": cell.cell_id,
                "dag_variant": variant,
                "node_id": n.get("id"),
                "materialization": n.get("materialize"),
                "depends_on": n.get("depends_on") or [],
                "out_degree": children.get(n.get("id"), 0),
                "paths_to_sinks": _paths_to_sinks(n.get("id"), nodes),
                "query_text": n.get("query_text") if full else None,
            })


def _paths_to_sinks(node_id: str, nodes: list[dict[str, Any]]) -> int:
    """Distinct paths from `node_id` to a sink: how often its work is repeated."""
    children: dict[str, list[str]] = {}
    for n in nodes:
        for dep in n.get("depends_on", []):
            children.setdefault(dep, []).append(n["id"])

    memo: dict[str, int] = {}

    def count(nid: str) -> int:
        if nid in memo:
            return memo[nid]
        kids = children.get(nid, [])
        memo[nid] = 1 if not kids else sum(count(k) for k in kids)
        return memo[nid]

    try:
        return count(node_id)
    except RecursionError:
        return 0


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
