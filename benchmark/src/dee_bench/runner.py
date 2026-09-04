"""Execute one benchmark cell: optimize the DAG, then measure it.

A cell runs in two phases:

**optimize** (skipped for baseline variants)
    ``POST /v1/dags/{name}/optimize``, sampled throughout. Produces a new DAG
    version and an ``OptimizeReport`` -- the cost side of the payback analysis.

    The request carries no optimizer settings. A cell's pass list and options
    are submitted *with the DAG*, so the registry says what each DAG is for and
    the settings cannot drift from the thing they describe. dee echoes back the
    configuration it resolved to, and the runner checks it against the cell's
    own before trusting the result.

**measure**
    The cell's ``warmups`` and ``repetitions``, executed one of two ways
    depending on ``execution.repeat_mode``:

    ``group``
        One trigger carrying the whole series, which dee runs back to back
        inside a single driver against one already-warm engine. The tightest
        measurement of the DAG itself, and the default.

    ``queue``
        One queued run group per repetition, drained by the server strictly in
        sequence. Each repetition gets a fresh engine and its own group in
        dee's history, which is what a repetition looks like in production.
        Slightly slower and slightly noisier; the point is to be able to ask
        whether the shared engine is flattering the numbers.

    Either way the pool stays warm across repetitions -- it is cached by the
    server, not by the run -- so ``engine_wall_ms`` measures the DAG rather
    than process startup and pool construction.

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

from .config import DEE_OPT_BY_NAME, VALID_PASSES, BenchConfig
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
    # Things worth saying about a cell that did not fail -- a continuous
    # optimization that ran out of runs before converging, say.
    notes: list[str] = field(default_factory=list)


def build_optimizer_config(cell: Cell) -> dict[str, Any]:
    """Render this cell's optimizer settings as an `OptimizerConfig`.

    Passes are always stated explicitly rather than relying on dee's defaults,
    so a variant's pass set is exactly its list and does not shift when those
    defaults change.
    """
    # Derived from the pass list rather than written out, so a pass added to
    # dee is benchmarkable as soon as `VALID_PASSES` names it. Spelling the
    # three known ones here is what left `run_parallelism_pass` silently unset
    # once ParallelismTuning landed: the optimization then ran no passes at
    # all and produced no version to measure.
    config: dict[str, Any] = {
        f"run_{name}_pass": name in cell.passes for name in VALID_PASSES
    }
    for key, value in sorted(cell.dee_opt.items()):
        if value is None:
            continue
        spec = DEE_OPT_BY_NAME[key]
        config[spec.config_field] = spec.config_value(value)
    return config


def build_optimize_request() -> dict[str, Any]:
    """The optimize body. Deliberately carries no settings.

    The cell's settings were submitted with the DAG, and an optimization with
    no config of its own runs under them. Re-sending them here would work too,
    and would mean the stored settings were never actually exercised -- a
    benchmark of a feature should use the feature.
    """
    return {
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


def build_queue_request(cell: Cell, sample_interval_ms: int, collect_plans: bool,
                        *, count: int, warmups: int) -> dict[str, Any]:
    """One `POST /v1/dags/{name}/queue` body: `count` entries of one run each.

    ``repetitions`` is always 1 here. Under this mode the queue *is* the
    repetition, so putting repetitions inside an entry as well would nest one
    meaning of the word inside the other and make ``rep_index`` ambiguous.
    """
    return {
        "count": count,
        "warmups": warmups,
        "repetitions": 1,
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
            optimizer_config = build_optimizer_config(cell)
            (artifacts / "optimizer_config.json").write_text(
                json.dumps(optimizer_config, indent=2)
            )
            target, submitted = register(
                self.client, prepared, dag_name, optimizer_config
            )
            version = submitted["version"]
            self._record_graph(cell, dag_name, version, "unopt", result)

            dag_variant = "unopt"
            if cell.is_continuous:
                # Nothing is optimized up front. The optimization is attached
                # to the DAG and steps around the measured runs below, so the
                # version measured is whatever it has converged to by then --
                # read back after the fact rather than decided here.
                self._register(cell, dag_name, artifacts, result)
                dag_variant = "converging"
            elif not cell.is_baseline:
                version = self._optimize(cell, dag_name, artifacts, result)
                dag_variant = "optimized"
                self._record_graph(cell, dag_name, version, "optimized", result)

            self._measure(cell, dag_name, version, ctx, artifacts, dag_variant, result)

            if cell.is_continuous:
                self._collect_continuous(cell, dag_name, artifacts, result)
        except Exception as e:  # noqa: BLE001 - a cell failure must not end the sweep
            result.status = "failed"
            result.error = f"{type(e).__name__}: {e}"
        return result

    # -- continuous --------------------------------------------------------

    def _register(self, cell: Cell, dag_name: str, artifacts: Path,
                  result: CellResult) -> None:
        """Attach this cell's optimizations to the DAG.

        The settings were submitted with the DAG, so registration sends none of
        its own -- the same indirection `_optimize` relies on, and checked the
        same way against what the server echoes back.
        """
        registrations = []
        for optimization in cell.passes:
            accepted = self.client.register_optimization(
                dag_name, optimization, cell.variant.step_phase, None
            )
            _verify_settings(cell, accepted.get("config") or {})
            registrations.append(accepted)
        (artifacts / "registrations.json").write_text(json.dumps(registrations, indent=2))

        # A `once` optimization is not stepped around runs, so registering one
        # and then measuring would measure the unoptimized DAG while the result
        # claimed the optimization was applied.
        stepping = [r["name"] for r in registrations
                    if r.get("optimization_type") == "continuous"]
        if not stepping:
            raise WorkloadError(
                f"none of {list(cell.passes)} is a continuous optimization, so "
                "there is nothing for continuous mode to drive; run this variant "
                "under optimization_mode: batch"
            )

    def _collect_continuous(self, cell: Cell, dag_name: str, artifacts: Path,
                            result: CellResult) -> None:
        """Record what the registered optimizations converged to, if anything.

        The counterpart to `_optimize`'s rows. There is no `OptimizeReport` to
        read here -- nothing ran an optimization as a job -- so the same
        questions are answered from the registrations themselves: did it
        converge, on which version, and how many of the cell's runs did it
        spend getting there.
        """
        registrations = self.client.optimizations(dag_name)
        (artifacts / "registrations_final.json").write_text(
            json.dumps(registrations, indent=2)
        )

        for row in registrations:
            converged = not row.get("active", True)
            result.rows.setdefault("optimizations", []).append({
                "cell_id": cell.cell_id,
                "started_at": None,
                "finished_at": None,
                # A continuous optimization runs no jobs of its own: its cost
                # is the overhead its steps added to runs that were happening
                # anyway, which shows up in those runs' timings rather than as
                # a separate wall time. Recording zero here rather than null
                # says "measured, and it was none", which is the finding.
                "opt_wall_ms": 0,
                "opt_cpu_seconds": None,
                "opt_peak_rss_bytes": None,
                "dag_runs_used": 0,
                "baseline_runtime_ms": None,
                "final_runtime_ms": None,
                "total_changes_applied": None,
                "nodes_before": None,
                "nodes_after": None,
                "status": "converged" if converged else "converging",
                "error": None,
                "optimization_type": row.get("optimization_type"),
                "step_phase": row.get("step_phase"),
                "result_version": row.get("result_version"),
            })

        if any(r.get("active", True) for r in registrations):
            # Not a failure -- the search is real, it simply had fewer runs
            # than it needed. Recorded so a cell that did not converge is not
            # read as one that converged on the DAG as authored.
            result.notes.append(
                f"{cell.cell_id}: an optimization had not converged after "
                f"{cell.repetitions} run(s)"
            )

    # -- optimize ----------------------------------------------------------

    def _optimize(self, cell: Cell, dag_name: str, artifacts: Path,
                  result: CellResult) -> int:
        body = build_optimize_request()
        (artifacts / "optimize_request.json").write_text(json.dumps(body, indent=2))

        started = datetime.now(timezone.utc)
        with self._sample_phase() as sampler:
            accepted = self.client.optimize(dag_name, body, self.cfg.execution.timeout_s)
        metrics, wall_ms = sampler.result
        finished = datetime.now(timezone.utc)

        _verify_settings(cell, accepted.get("config") or {})
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
        with self._sample_phase() as sampler:
            group_ids = self._execute(cell, dag_name, version, artifacts)
        metrics, wall_ms = sampler.result

        executions = self._collect(group_ids, artifacts)
        if not executions:
            raise WorkloadError("the run produced no measurements")

        # One sampler covers the whole phase, which contains every repetition.
        # Splitting its totals per repetition would be a fiction, so they are
        # apportioned by each repetition's share of engine time and the basis
        # is recorded in the schema.
        total_engine_ms = sum(r.get("duration_ms", 0) for _, r, _ in executions) or 1

        # `rep_index` is per phase and per cell, not per run group. Under
        # `queue` each group holds one run and would otherwise report index 0
        # five times over, so the counter runs across groups instead.
        next_index = {"warmup": 0, "measure": 0}

        for group_id, r, server_run in executions:
            share = r.get("duration_ms", 0) / total_engine_ms
            phase = r.get("phase", "measure")
            rep_index = next_index.get(phase, 0)
            next_index[phase] = rep_index + 1
            run_id = server_run["run_id"] if server_run else str(uuid.uuid4())
            node_execs = r.get("node_executions", [])

            result.rows.setdefault("runs", []).append({
                "cell_id": cell.cell_id,
                "run_id": run_id,
                "run_group_id": group_id,
                "dag_variant": dag_variant,
                "dag_version": (server_run or {}).get("dag_version"),
                "phase": phase,
                "rep_index": rep_index,
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
                # Only a 'direct' delivery's engine_wall_ms is comparable to
                # another run's; see the column docs in schema.py.
                "delivery": (server_run or {}).get("delivery") or "direct",
                "trial_elapsed_ms": (server_run or {}).get("trial_elapsed_ms"),
                "resume_elapsed_ms": (server_run or {}).get("resume_elapsed_ms"),
                "read_bytes": int((metrics.read_bytes or 0) * share) if metrics.read_bytes else None,
                "written_bytes": int((metrics.written_bytes or 0) * share) if metrics.written_bytes else None,
                "db_size_bytes": _peak(r.get("system_samples", []), "disk_bytes"),
                "plan_time_basis": ctx.plan_time_basis,
                "status": "ok",
                "error": None,
            })
            if phase == "measure":
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
                    "phase": phase,
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

    def _execute(self, cell: Cell, dag_name: str, version: int,
                 artifacts: Path) -> list[str]:
        """Run the cell's repetitions. Returns their run group ids, in order."""
        if cell.repeat_mode == "queue":
            return self._execute_queued(cell, dag_name, version, artifacts)

        body = build_run_request(
            cell,
            self.cfg.execution.sample_interval_ms,
            collect_plans=self.store.records("plans"),
        )
        body["version"] = version
        (artifacts / "run_request.json").write_text(json.dumps(body, indent=2))

        triggered = self.client.trigger(dag_name, body, self.cfg.execution.timeout_s)
        return [triggered["run_group_id"]]

    def _execute_queued(self, cell: Cell, dag_name: str, version: int,
                        artifacts: Path) -> list[str]:
        """Put each repetition on the server's queue, one run group apiece.

        The version is pinned. A queue entry that names no version follows the
        DAG to whatever is current when its turn comes, which is the right
        default for watching a DAG adapt and exactly wrong here: a cell has
        already decided which version it measures.

        Two calls rather than one, because the warmups belong to the front of
        the queue only and every entry in one request shares a body. They cost
        nothing extra -- the entries run strictly in sequence either way.
        """
        timeout = self.cfg.execution.timeout_s
        collect_plans = self.store.records("plans")
        bodies: list[dict[str, Any]] = []

        if cell.warmups:
            bodies.append(build_queue_request(
                cell, self.cfg.execution.sample_interval_ms, collect_plans,
                count=1, warmups=cell.warmups,
            ))
        remaining = cell.repetitions - len(bodies)
        if remaining > 0:
            bodies.append(build_queue_request(
                cell, self.cfg.execution.sample_interval_ms, collect_plans,
                count=remaining, warmups=0,
            ))

        for body in bodies:
            body["version"] = version
        (artifacts / "queue_request.json").write_text(json.dumps(bodies, indent=2))

        group_ids: list[str] = []
        try:
            for body in bodies:
                accepted = self.client.enqueue(dag_name, body, timeout)
                group_ids.extend(e["run_group_id"] for e in accepted["entries"])
        except Exception:
            # Whatever went wrong, entries may still be waiting their turn.
            # Left alone they would run against the warehouse while the *next*
            # cell is being measured -- not a failure, just quietly wrong
            # numbers for a cell that looked fine.
            try:
                self.client.clear_queue(dag_name)
            except (ApiError, ServerError):
                pass
            raise

        return group_ids

    def _collect(self, group_ids: list[str],
                 artifacts: Path) -> list[tuple[str, dict[str, Any], dict[str, Any] | None]]:
        """Every execution across `group_ids`, as (group id, report, run row).

        The report carries the measurements and the run row carries the
        server's own run id, so the parquet records ids that join back to
        dee's history rather than ones the harness invented.
        """
        executions = []
        for index, group_id in enumerate(group_ids):
            group = self.client.run_group(group_id)
            if group["status"] != "succeeded":
                raise WorkloadError(
                    f"run group {group['status']}: "
                    f"{group.get('error') or 'no detail recorded'}"
                )

            report = self.client.group_report(group_id)
            name = "run_report.json" if len(group_ids) == 1 else f"run_report_{index:02d}.json"
            (artifacts / name).write_text(json.dumps(report, indent=2))

            by_key = {(r["phase"], r["rep_index"]): r for r in group.get("runs", [])}
            for r in report.get("runs", []):
                key = (r.get("phase", "measure"), r.get("rep_index", 0))
                executions.append((group_id, r, by_key.get(key)))
        return executions

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


def _verify_settings(cell: Cell, resolved: dict[str, Any]) -> None:
    """Check the optimization ran under the settings submitted with the DAG.

    The settings now reach the optimizer indirectly, so nothing about the
    request itself proves which ones were used. dee echoes the configuration it
    resolved to, and comparing it against the cell's own is what turns that
    indirection back into something checkable.

    The failure this exists for is silent: a dee that predates DAG-level
    settings ignores them on submit, then optimizes under its own defaults --
    a cell that looks like it benchmarked OMP but ran HMP as well.
    """
    expected = build_optimizer_config(cell)
    disagreed = {
        key: (value, resolved.get(key))
        for key, value in expected.items()
        if resolved.get(key) != value
    }
    if not disagreed:
        return
    detail = ", ".join(
        f"{key}: asked for {asked!r}, ran with {ran!r}"
        for key, (asked, ran) in sorted(disagreed.items())
    )
    raise WorkloadError(
        f"the optimizer did not run under this cell's settings ({detail}); "
        f"the dee server may predate per-DAG optimizer settings"
    )


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
