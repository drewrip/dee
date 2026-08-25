"""Postgres backend: a managed docker container, or an instance you already run.

Bring-up is session scoped and teardown is registered with both ``atexit`` and
the termination signals, because a container left holding port 5432 after an
interrupted run blocks the next one and is easy to miss.

Data persistence is deliberate: the data directory lives in a named docker
volume keyed by scale factor, so re-running a scale factor skips the multi-minute
reload. ``--fresh`` drops it. Benchmark *results* are never touched by teardown.
"""

from __future__ import annotations

import atexit
import functools
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from ..sampler import CgroupReader, find_cgroup_path
from ..workload import (
    WorkloadError,
    ensure_scratch_repo_root,
    postgres_schema_for,
    run_cmd,
)
from .base import Backend, BackendContext

CONTAINER_LABEL = "dee-bench=1"
CONTAINER_NAME = "dee-bench-pg"


@functools.lru_cache(maxsize=1)
def container_runtime() -> str:
    """The container CLI to drive: docker if usable, else podman.

    Docker is preferred when it works, but on a host where the daemon socket
    is not accessible to this user it fails with a permission error rather
    than being absent, so presence on PATH is not enough -- the runtime is
    probed. Podman is command-line compatible for everything used here.
    """
    override = os.environ.get("DEE_BENCH_CONTAINER_RUNTIME")
    if override:
        return override
    for candidate in ("docker", "podman"):
        if not shutil.which(candidate):
            continue
        probe = subprocess.run([candidate, "info"], capture_output=True, text=True)
        if probe.returncode == 0:
            return candidate
    raise WorkloadError(
        "no usable container runtime found. Install docker or podman, or set "
        "`provider: external` under backends.postgres to use an instance you run yourself."
    )


def qualify_image(image: str) -> str:
    """Fully qualify a short image name when the runtime demands it.

    Podman refuses to guess a registry for a short name and, with no TTY to
    prompt on, fails outright. Docker silently assumes Docker Hub, so configs
    are conventionally written short; qualifying here keeps one config working
    on both runtimes.
    """
    if container_runtime() != "podman":
        return image
    first = image.split("/")[0]
    already_qualified = "/" in image and ("." in first or ":" in first or first == "localhost")
    return image if already_qualified else f"docker.io/library/{image}"


class PostgresBackend(Backend):
    name = "postgres"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        dag_bench: Path | None = None,
        fresh: bool = False,
        keep: bool = False,
        log=print,
    ):
        self.config = dict(config or {})
        self.dag_bench = dag_bench
        self.fresh = fresh
        self.keep = keep
        self.log = log
        # "container" is the runtime-neutral name; "docker" and "podman" are
        # accepted as aliases since configs in the wild name the runtime.
        provider = str(self.config.get("provider", "container"))
        self.provider = "external" if provider == "external" else "container"
        self.container_id: str | None = None
        self._torn_down = False
        self._loaded: set[tuple[str, float]] = set()
        self._prev_handlers: dict[int, Any] = {}

        self.conn = {
            "host": str(self.config.get("host", "127.0.0.1")),
            "port": int(self.config.get("port", 5432)),
            "user": str(self.config.get("user", "runner")),
            "password": str(self.config.get("password", "password")),
            "dbname": str(self.config.get("dbname", "benchmark")),
        }

    # -- lifecycle ---------------------------------------------------------

    def setup(self) -> BackendContext:
        if self.provider == "external":
            self.log(f"  postgres: using external instance at {self.conn['host']}:{self.conn['port']}")
            self._wait_ready()
        else:
            self._start_container()
        self._install_teardown_hooks()
        return BackendContext(
            name=self.name,
            postgres=dict(self.conn),
            cgroup=self._cgroup(),
            # Postgres reports `Actual Total Time` per plan node, which is
            # wall time (x loops), not CPU time as DuckDB's `cpu_time` is.
            # The optimizer's cost ranking therefore means something subtly
            # different here, and analyses must not silently mix the two.
            plan_time_basis="wall_time",
        )

    def _start_container(self) -> None:
        self._remove_existing()
        volume = f"dee-bench-pg-{self.config.get('volume_suffix', 'data')}"
        if self.fresh:
            subprocess.run([container_runtime(), "volume", "rm", "-f", volume], capture_output=True)

        image = qualify_image(str(self.config.get("image", "postgres:18")))
        cmd = [
            container_runtime(), "run", "-d",
            "--name", CONTAINER_NAME,
            "--label", CONTAINER_LABEL,
            "-e", f"POSTGRES_USER={self.conn['user']}",
            "-e", f"POSTGRES_PASSWORD={self.conn['password']}",
            "-e", f"POSTGRES_DB={self.conn['dbname']}",
            "-p", f"{self.conn['port']}:5432",
            # Mounted at the parent, not .../data: postgres:18 moved its data
            # directory into a version-named subdirectory and refuses to start
            # against a volume mounted directly at the old path. The parent
            # mount persists the data for both old and new layouts.
            "-v", f"{volume}:/var/lib/postgresql",
        ]
        if self.config.get("cpus"):
            cmd += ["--cpus", str(self.config["cpus"])]
        if self.config.get("memory"):
            cmd += ["--memory", str(self.config["memory"]), "--memory-swap", str(self.config["memory"])]
        cmd.append(image)
        # Server tuning is passed as postgres arguments so it needs no
        # config-file mount and stays visible in `docker inspect`.
        for key, value in (self.config.get("settings") or {}).items():
            cmd += ["-c", f"{key}={value}"]

        self.log(f"  postgres: starting {image} (volume {volume})")
        proc = run_cmd(cmd)
        self.container_id = proc.stdout.strip()
        self._wait_ready()
        self._bootstrap()

    def _remove_existing(self) -> None:
        subprocess.run([container_runtime(), "rm", "-f", CONTAINER_NAME], capture_output=True)

    def _wait_ready(self, timeout: int = 180) -> None:
        """Block until the server accepts connections.

        A container that has already exited will never become ready, so that
        is detected and reported with the server's own logs rather than
        waiting out the full timeout and reporting a probe error that says
        nothing about the actual cause.
        """
        deadline = time.time() + timeout
        last = ""
        while time.time() < deadline:
            if self.provider != "external" and not self._container_running():
                raise WorkloadError(
                    "postgres container exited before becoming ready:\n"
                    + self._container_logs()
                )
            if self.provider != "external" and self.container_id:
                probe = subprocess.run(
                    [container_runtime(), "exec", CONTAINER_NAME, "pg_isready",
                     "-U", self.conn["user"], "-d", self.conn["dbname"]],
                    capture_output=True, text=True,
                )
            else:
                probe = subprocess.run(
                    ["pg_isready", "-h", self.conn["host"], "-p", str(self.conn["port"]),
                     "-U", self.conn["user"], "-d", self.conn["dbname"]],
                    capture_output=True, text=True,
                )
            if probe.returncode == 0:
                # pg_isready can succeed while the entrypoint is still
                # restarting during first-time initdb, so require it twice.
                time.sleep(1.0)
                return
            last = (probe.stdout or probe.stderr or "").strip()
            time.sleep(1.0)
        raise WorkloadError(f"postgres did not become ready within {timeout}s: {last}")

    def _container_running(self) -> bool:
        probe = subprocess.run(
            [container_runtime(), "inspect", CONTAINER_NAME, "--format", "{{.State.Running}}"],
            capture_output=True, text=True,
        )
        return probe.returncode == 0 and probe.stdout.strip() == "true"

    def _container_logs(self, lines: int = 30) -> str:
        proc = subprocess.run(
            [container_runtime(), "logs", "--tail", str(lines), CONTAINER_NAME],
            capture_output=True, text=True,
        )
        return ((proc.stdout or "") + (proc.stderr or "")).strip() or "(no container logs)"

    def _bootstrap(self) -> None:
        """Apply dag-bench's postgres bootstrap SQL, if the checkout has it."""
        if self.dag_bench is None:
            return
        sql_path = self.dag_bench / "utils" / "bootstrap_postgres.sql"
        if not sql_path.exists():
            return
        self._psql(sql_path.read_text())

    def _psql(self, sql: str) -> None:
        env = {"PGPASSWORD": self.conn["password"]}
        run_cmd(
            ["psql", "-h", self.conn["host"], "-p", str(self.conn["port"]),
             "-U", self.conn["user"], "-d", self.conn["dbname"], "-v", "ON_ERROR_STOP=1",
             "-c", sql],
            env=env, timeout=600,
        )

    def _cgroup(self) -> CgroupReader | None:
        if not self.container_id:
            return None
        path = find_cgroup_path(self.container_id)
        if path is None:
            self.log(
                "  postgres: cgroup counters unavailable; container CPU/memory will not be sampled"
            )
            return None
        return CgroupReader(path)

    # -- data --------------------------------------------------------------

    def prepare_scale(self, project: str, sf: float, prepared) -> None:
        """Load a project's source tables into postgres, once per (project, sf)."""
        key = (project, sf)
        if key in self._loaded:
            return
        if self.dag_bench is None:
            raise WorkloadError("postgres data loading needs the dag-bench checkout")

        ensure_scratch_repo_root(prepared.project_dir.parent, self.dag_bench)
        schema = postgres_schema_for(prepared.project_dir)
        self.log(f"    loading {project} sf={sf:g} into postgres schema {schema}")
        # dag-bench's own loader: duckdb -> CSV -> psql \copy. Reused rather
        # than reimplemented so the harness and dag-bench cannot drift.
        env = {
            "PGPASSWORD": self.conn["password"],
            "PGHOST": self.conn["host"],
            "PGPORT": str(self.conn["port"]),
        }
        run_cmd(
            ["python3", str(self.dag_bench / "utils" / "postgres_bench_utils.py"),
             "--project-dir", str(prepared.project_dir),
             "--duckdb-path", str(prepared.warehouse)],
            cwd=self.dag_bench, env=env, timeout=7200,
        )
        self._loaded.add(key)

    # -- teardown ----------------------------------------------------------

    def _install_teardown_hooks(self) -> None:
        """Guarantee teardown on normal exit and on interruption.

        An interrupted run that leaves the container running holds port 5432
        and silently breaks the next run, so SIGINT/SIGTERM are handled as well
        as normal exit.
        """
        atexit.register(self.teardown)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            try:
                self._prev_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, self._on_signal)
            except (ValueError, OSError):
                # Not on the main thread; atexit still covers normal exit.
                pass

    def _on_signal(self, signum, frame) -> None:
        self.teardown()
        previous = self._prev_handlers.get(signum)
        if callable(previous):
            previous(signum, frame)
            return
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    def teardown(self) -> None:
        if self._torn_down or self.provider == "external":
            self._torn_down = True
            return
        self._torn_down = True
        if self.keep:
            self.log(f"  postgres: leaving {CONTAINER_NAME} running (--keep-infra)")
            return
        if self.container_id or _container_exists():
            self.log("  postgres: removing container")
            subprocess.run([container_runtime(), "rm", "-f", CONTAINER_NAME], capture_output=True)

    def describe(self) -> str:
        if self.provider == "external":
            return f"postgres (external {self.conn['host']}:{self.conn['port']})"
        return f"postgres ({container_runtime()} {self.config.get('image', 'postgres:18')})"


def _container_exists() -> bool:
    proc = subprocess.run(
        [container_runtime(), "ps", "-aq", "--filter", f"name=^{CONTAINER_NAME}$"],
        capture_output=True, text=True,
    )
    return bool(proc.stdout.strip())


def stray_containers() -> list[dict[str, str]]:
    """Every container this harness labelled, for `dee-bench doctor`."""
    proc = subprocess.run(
        [container_runtime(), "ps", "-a", "--filter", f"label={CONTAINER_LABEL}",
         "--format", "{{json .}}"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return []
    return [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]


def remove_stray_containers() -> int:
    strays = stray_containers()
    for c in strays:
        subprocess.run([container_runtime(), "rm", "-f", c["ID"]], capture_output=True)
    return len(strays)
