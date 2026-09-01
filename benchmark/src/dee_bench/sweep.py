"""Drive a whole benchmark run: prepare, execute every cell, tear down.

The scheduler groups cells by ``(backend, sf, project)`` so infrastructure and
data preparation are amortized. Preparing a project at a scale factor costs far
more than running one DAG, so every variant and repetition sharing a dataset
runs back to back against a single preparation.

Teardown always happens, including on interruption, and never touches results.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import time

import yaml
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import queue as q
from .config import HARNESS_VERSION, BenchConfig
from .infra import make_backend
from .infra.base import Backend, BackendContext
from .matrix import Cell, cells_to_rows, expand, summarize
from .runner import CellRunner
from .server import ApiError, DeeServer, ServerError
from .store import ResultStore
from .workload import prepare


def git_sha(path: Path) -> str | None:
    with suppress(Exception):
        proc = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    return None


def provenance(cfg: BenchConfig) -> dict[str, Any]:
    """Everything needed to interpret results months later."""
    mem_total = None
    with suppress(Exception):
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                mem_total = int(line.split()[1]) * 1024
                break
    return {
        "dee_git_sha": git_sha(cfg.dee_bin.parent.parent.parent),
        "dag_bench_git_sha": git_sha(cfg.dag_bench),
        "harness_version": HARNESS_VERSION,
        "host": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "mem_total_bytes": mem_total,
        "platform": platform.platform(),
    }


class Sweep:
    """Executes a config's cells into a run directory."""

    def __init__(self, cfg: BenchConfig, run_dir: Path | None = None,
                 fresh: bool = False, keep_infra: bool = False, log=print):
        self.cfg = cfg
        self.run_dir = Path(run_dir or cfg.output_dir)
        self.fresh = fresh
        self.keep_infra = keep_infra
        self.log = log
        self.store = ResultStore(self.run_dir, cfg.verbosity)
        self.queue = q.RunQueue(self.run_dir)
        self.cells: list[Cell] = expand(cfg)
        self._backends: dict[str, Backend] = {}

    # -- setup -------------------------------------------------------------

    def initialize(self) -> None:
        """Create the run directory and freeze everything needed to reproduce it."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "artifacts").mkdir(exist_ok=True)
        (self.run_dir / "scratch").mkdir(exist_ok=True)

        prov = provenance(self.cfg)
        (self.run_dir / "provenance.json").write_text(json.dumps(prov, indent=2))
        # A frozen copy, so results stay interpretable even if the original
        # config is later edited -- and so `resume` reproduces this exact run.
        # Paths are written resolved: the frozen copy lives at a different
        # depth from the original, so any relative path in it would point
        # somewhere else entirely.
        frozen = dict(self.cfg.raw)
        frozen["dag_bench"] = str(self.cfg.dag_bench)
        frozen["dee_bin"] = str(self.cfg.dee_bin)
        frozen["output_dir"] = str(self.run_dir)
        frozen["verbosity"] = self.cfg.verbosity.label()
        (self.run_dir / "config.yaml").write_text(yaml.safe_dump(frozen, sort_keys=False))

        self.queue.initialize(self.cells)
        # cells is the join key for every other table; write it up front so a
        # run that fails immediately is still self-describing.
        if not self.store.has("cells"):
            for row in cells_to_rows(self.cells, prov):
                self.store.write("cells", [row], cell_id=row["cell_id"])

    # -- execution ---------------------------------------------------------

    def run(self, retry_failed: bool = True) -> dict[str, int]:
        self.initialize()
        pending = self.queue.pending(self.cells, retry_failed=retry_failed)

        self.log(f"run directory: {self.run_dir}")
        self.log(summarize(self.cells))
        done_already = len(self.cells) - len(pending)
        if done_already:
            self.log(f"{done_already} cell(s) already complete; {len(pending)} to run")

        self.queue.write_pid(os.getpid())
        try:
            self._execute(pending)
        finally:
            self._teardown_all()
            with suppress(OSError):
                self.queue.pid_file.unlink()

        counts = self.queue.counts()
        self.log(
            f"finished: {counts.get(q.DONE, 0)} done, {counts.get(q.FAILED, 0)} failed, "
            f"{counts.get(q.PENDING, 0)} pending"
        )
        return counts

    def _execute(self, pending: list[Cell]) -> None:
        # One server for the whole sweep, not one per cell. Connection pools
        # then stay warm across cells, which is the same amortization the cell
        # ordering already assumes for infrastructure and data preparation.
        server = DeeServer(
            self.cfg.dee_bin,
            self.run_dir,
            bind=self.cfg.server.bind,
            url=self.cfg.server.url,
            startup_timeout_s=self.cfg.server.startup_timeout_s,
            timeout_s=self.cfg.execution.timeout_s,
        )
        with server as client:
            self.log(f"    dee server at {client.url} (metadata: {self.run_dir}/metadata.duckdb)")
            self._require_server_support(pending, client)
            self._execute_cells(pending, client, server.pid)

    @staticmethod
    def _require_server_support(pending: list[Cell], client) -> None:
        """Fail now, not on the first cell, if the server has no run queue.

        Probed by calling the endpoint, not by reading a version: dee keeps one
        schema rather than a migration chain, so nothing it reports about
        itself distinguishes a build with the queue from one without.

        Per-DAG optimizer settings are not checked here for the same reason --
        a server that ignores them on submit answers `GET /v1/dags/{name}` the
        same way. That one is caught per cell instead, by comparing the
        configuration dee says it resolved to against the cell's own.

        Only reachable when attached to a server the sweep did not build, but
        that is exactly the case where the binary can be older than the harness
        -- and a sweep that dies eight cells in has already wasted the
        expensive part.
        """
        if any(cell.repeat_mode == "queue" for cell in pending):
            try:
                client.queue()
            except (ApiError, ServerError) as e:
                raise ServerError(
                    "execution.repeat_mode is 'queue' but this dee server has no run "
                    f"queue ({e}); use a newer dee, or set repeat_mode: group"
                ) from None

        continuous = [c for c in pending if c.is_continuous]
        if not continuous:
            return
        try:
            available = {o["name"]: o for o in client.available_optimizations()}
        except (ApiError, ServerError) as e:
            raise ServerError(
                "optimization_mode is 'continuous' but this dee server cannot "
                f"register optimizations ({e}); use a newer dee, or set "
                "optimization_mode: batch"
            ) from None

        # A `once` optimization is not stepped around runs. Registering one and
        # measuring anyway would quietly measure the unoptimized DAG under a
        # variant name that says otherwise, so refuse the whole sweep here
        # rather than produce cells that look fine.
        for cell in continuous:
            steppable = [
                p for p in cell.passes
                if available.get(p, {}).get("optimization_type") == "continuous"
            ]
            if not steppable:
                raise ServerError(
                    f"variant '{cell.variant.name}' runs {list(cell.passes)}, none of "
                    "which is a continuous optimization, so continuous mode has "
                    "nothing to drive; run that variant under optimization_mode: batch"
                )

    def _execute_cells(self, pending: list[Cell], client, server_pid) -> None:
        prepared_key: tuple[str, str, float] | None = None
        prepared = None

        for i, cell in enumerate(pending, 1):
            started = time.monotonic()
            self.log(f"[{i}/{len(pending)}] {cell.describe()}  ({cell.cell_id})")
            self.queue.mark_running(cell.cell_id, cell.describe())
            try:
                backend, ctx = self._backend_for(cell.backend)

                key = (cell.backend, cell.project, cell.sf)
                if key != prepared_key:
                    self.log(f"    preparing {cell.project} at sf={cell.sf:g} for {cell.backend}")
                    prepared = prepare(
                        self.cfg.dag_bench, cell.project, cell.backend, cell.sf,
                        self.run_dir / "scratch",
                        backend_config=cell.backend_config,
                        postgres=ctx.postgres,
                        log=self.log,
                    )
                    backend.prepare_scale(cell.project, cell.sf, prepared)
                    prepared_key = key

                runner = CellRunner(
                    self.cfg, self.store, self.run_dir, client, server_pid, self.log
                )
                runner.set_cgroup(ctx.cgroup)
                result = runner.run(cell, prepared, ctx)
                self._persist(cell, result)

                elapsed = time.monotonic() - started
                if result.status == "done":
                    self.queue.mark_finished(cell.cell_id, q.DONE)
                    self.log(f"    ok in {elapsed:.1f}s ({result.measured_runs} measured runs)")
                else:
                    self.queue.mark_finished(cell.cell_id, q.FAILED, result.error)
                    self.log(f"    FAILED after {elapsed:.1f}s: {result.error}")
            except KeyboardInterrupt:
                self.queue.mark_finished(cell.cell_id, q.PENDING, "interrupted")
                raise
            except Exception as e:  # noqa: BLE001 - one bad cell must not end the sweep
                self.queue.mark_finished(cell.cell_id, q.FAILED, f"{type(e).__name__}: {e}")
                self.log(f"    FAILED: {type(e).__name__}: {e}")
                # A failed preparation must not be reused by the next cell.
                prepared_key = None

    def _persist(self, cell: Cell, result) -> None:
        """Write a cell's rows. Partial results from a failed cell are kept."""
        for table, rows in result.rows.items():
            if rows:
                self.store.write(table, rows, cell_id=cell.cell_id)

    def _backend_for(self, name: str) -> tuple[Backend, BackendContext]:
        if name not in self._backends:
            backend = make_backend(
                name, self.cfg.backends.get(name), dag_bench=self.cfg.dag_bench,
                fresh=self.fresh, keep=self.keep_infra, log=self.log,
            )
            self.log(f"  bringing up {backend.describe()}")
            ctx = backend.setup()
            self._backends[name] = backend
            backend._ctx = ctx  # type: ignore[attr-defined]
        backend = self._backends[name]
        return backend, backend._ctx  # type: ignore[attr-defined]

    def _teardown_all(self) -> None:
        for backend in self._backends.values():
            with suppress(Exception):
                backend.teardown()
        # Scratch holds copied warehouses, which are large and fully
        # regenerable. Results and artifacts are never touched.
        with suppress(OSError):
            shutil.rmtree(self.run_dir / "scratch", ignore_errors=True)


def write_run_meta(run_dir: Path, cfg: BenchConfig, total: int) -> None:
    (run_dir / "run.json").write_text(json.dumps({
        "name": cfg.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_cells": total,
        "verbosity": cfg.verbosity.label(),
        "config": str(cfg.source_path) if cfg.source_path else None,
    }, indent=2))
