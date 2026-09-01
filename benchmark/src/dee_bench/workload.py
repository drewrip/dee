"""Turn a dag-bench project at a scale factor into something dee can run.

The pipeline, per (project, scale factor, backend)::

    dbgen -p N -s SF        ->  data/warehouse.duckdb   [cached per (project, sf)]
      (postgres) load_postgres.py -> server-side schema
    dbt deps && dbt compile ->  target/manifest.json
    dee-cli convert         ->  dag.json

Two details drive the design:

* dag-bench already caches generated warehouses at
  ``dbgen/.cache/p{N}_sf{SF}.duckdb``. Generating data is by far the most
  expensive step, so the harness populates and reuses that cache rather than
  regenerating per cell.
* Each cell gets its *own copy* of the warehouse. The previous harness
  symlinked the shared file, so every run mutated the cached dataset that
  later runs depended on.
"""

from __future__ import annotations

import functools
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class WorkloadError(Exception):
    """A project could not be prepared."""


@dataclass
class PreparedProject:
    """A project laid out on disk and ready for `dee-cli`."""

    project: str
    backend: str
    sf: float
    project_dir: Path
    # The dag-bench source the project was copied from. Kept because the
    # connection is rewritten per cell (see `apply_backend_config`) and the
    # postgres profile it falls back on lives in the original checkout.
    src_dir: Path
    dag_json: Path
    connections_json: Path
    target: str
    warehouse: Path | None


def run_cmd(
    cmd: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess:
    """Run a command, raising WorkloadError with its stderr on failure."""
    merged = {**os.environ, **(env or {})}
    try:
        proc = subprocess.run(
            cmd, cwd=cwd, env=merged, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as e:
        raise WorkloadError(f"timed out after {timeout}s: {' '.join(cmd)}") from e
    except FileNotFoundError as e:
        raise WorkloadError(f"command not found: {cmd[0]}") from e
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-25:]
        raise WorkloadError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n" + "\n".join(tail)
        )
    return proc


@functools.lru_cache(maxsize=1)
def dbt_executable() -> str:
    """The dbt to invoke.

    Prefer the one installed alongside this harness rather than whatever `dbt`
    PATH resolves to. dbt-core pins a narrow range of Python versions, and a
    system-wide dbt on a newer interpreter fails deep inside its own
    deserialization with an error that has nothing to do with the benchmark.
    Using our own environment's dbt makes the harness self-contained.
    """
    override = os.environ.get("DEE_BENCH_DBT")
    if override:
        return override
    local = Path(sys.executable).parent / "dbt"
    if local.exists():
        return str(local)
    found = shutil.which("dbt")
    if found:
        return found
    raise WorkloadError(
        "dbt not found. Install it into this harness's environment "
        "(`uv pip install -e .` in benchmark/), or set DEE_BENCH_DBT to a working dbt."
    )


def project_number(project: str) -> int:
    """`p03_ecommerce` -> 3, matching dbgen's `-p` argument."""
    m = re.match(r"p(\d+)", project)
    if not m:
        raise WorkloadError(f"cannot derive a dbgen project number from {project!r}")
    return int(m.group(1))


def discover_projects(dag_bench: Path) -> list[str]:
    """Projects with an actual dbt project file.

    dag-bench's ``projects.yaml`` also lists ``gym``, ``tpch`` and ``tpcds``,
    whose directories exist but hold only gitignored build artifacts — no
    ``dbt_project.yml``. Globbing avoids inheriting those dead entries.
    """
    root = dag_bench / "projects"
    return sorted(p.parent.name for p in root.glob("*/dbt_project.yml"))


# --------------------------------------------------------------------------
# data generation
# --------------------------------------------------------------------------


def cache_path(dag_bench: Path, project: str, sf: float) -> Path:
    """Where dag-bench caches this (project, sf) warehouse.

    Mirrors `dbgen/project_runner.py`: the scale factor's decimal point
    becomes an underscore.
    """
    sf_str = str(sf).replace(".", "_")
    return dag_bench / "dbgen" / ".cache" / f"p{project_number(project)}_sf{sf_str}.duckdb"


def dbgen_binary(dag_bench: Path) -> Path:
    """The dbgen binary, building it if it hasn't been built yet."""
    env_path = os.environ.get("DBGEN")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    built = dag_bench / "dbgen" / "target" / "release" / "dbgen"
    if not built.exists():
        run_cmd(["cargo", "build", "--release"], cwd=dag_bench / "dbgen", timeout=1800)
    if not built.exists():
        raise WorkloadError(f"dbgen binary not found at {built} after building")
    return built


def ensure_data(dag_bench: Path, project: str, sf: float, log=print) -> Path:
    """Ensure a warehouse exists for (project, sf), generating it if needed.

    Returns the path to the cached warehouse, which callers should copy rather
    than use in place.
    """
    cached = cache_path(dag_bench, project, sf)
    if cached.exists():
        return cached

    cached.parent.mkdir(parents=True, exist_ok=True)
    binary = dbgen_binary(dag_bench)
    log(f"    generating data: {project} at sf={sf:g} (first time for this scale factor)")
    tmp = cached.with_suffix(".duckdb.tmp")
    if tmp.exists():
        tmp.unlink()
    run_cmd(
        [str(binary), "-p", str(project_number(project)), "-s", str(sf), "-o", str(tmp)],
        cwd=dag_bench,
        timeout=7200,
    )
    # Only publish into the cache once generation fully succeeded, so an
    # interrupted generation never leaves a truncated warehouse that later
    # runs would silently trust.
    tmp.replace(cached)
    return cached


# --------------------------------------------------------------------------
# project preparation
# --------------------------------------------------------------------------


def prepare(
    dag_bench: Path,
    project: str,
    backend: str,
    sf: float,
    scratch_dir: Path,
    backend_config: dict[str, Any] | None = None,
    postgres: dict[str, Any] | None = None,
    log=print,
) -> PreparedProject:
    """Lay out `project` at `sf` under `scratch_dir` and convert it to a dee DAG."""
    backend_config = backend_config or {}
    src = dag_bench / "projects" / project
    if not (src / "dbt_project.yml").exists():
        raise WorkloadError(f"{src} is not a dbt project (no dbt_project.yml)")

    dest = scratch_dir / project
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # target/ and logs/ are stale build output from whoever last used the
    # checkout; dbt regenerates them and copying them just wastes time.
    shutil.copytree(
        src, dest, ignore=shutil.ignore_patterns("target", "logs", "data", ".user.yml")
    )

    warehouse = None
    if backend == "duckdb" or postgres is not None:
        cached = ensure_data(dag_bench, project, sf, log=log)
        (dest / "data").mkdir(exist_ok=True)
        warehouse = dest / "data" / "warehouse.duckdb"
        # A copy, not a symlink: runs materialize tables into this file, and
        # must not mutate the shared cache other cells depend on.
        shutil.copy2(cached, warehouse)

    dbt_target = "dev" if backend == "duckdb" else "postgres"
    env = {"DBT_TARGET": dbt_target}

    # The copied project's profiles.yml must describe the server the harness
    # actually brought up, not the one hardcoded in the dag-bench checkout.
    # Both `dbt compile` and dag-bench's own loader read connection details
    # from this file and ignore PG* environment variables, so rewriting it is
    # the only way a non-default host or port reaches them.
    if backend == "postgres" and postgres:
        rewrite_postgres_profile(dest, postgres)

    if (dest / "packages.yml").exists():
        run_cmd([dbt_executable(), "deps"], cwd=dest, env=env, timeout=600)
    run_cmd([dbt_executable(), "compile", "--target", dbt_target],
            cwd=dest, env=env, timeout=1800)

    manifest = dest / "target" / "manifest.json"
    if not manifest.exists():
        raise WorkloadError(f"dbt compile produced no manifest at {manifest}")

    connections_json, target = write_connections(
        dest, src, backend, backend_config, warehouse, postgres
    )
    return PreparedProject(
        project=project,
        backend=backend,
        sf=sf,
        project_dir=dest,
        src_dir=src,
        dag_json=dest / "dag.json",
        connections_json=connections_json,
        target=target,
        warehouse=warehouse,
    )


def rewrite_postgres_profile(project_dir: Path, conn: dict[str, Any]) -> None:
    """Point a copied project's `postgres` output at `conn`.

    Only the connection fields are touched; `schema` and `threads` are left as
    dag-bench set them, since those describe the workload rather than where it
    runs.
    """
    path = project_dir / "profiles.yml"
    if not path.exists():
        raise WorkloadError(f"no profiles.yml in {project_dir}")
    profiles = yaml.safe_load(path.read_text()) or {}
    changed = False
    for name, block in profiles.items():
        if name == "config" or not isinstance(block, dict):
            continue
        output = (block.get("outputs") or {}).get("postgres")
        if not isinstance(output, dict):
            continue
        output["host"] = conn.get("host", output.get("host"))
        output["port"] = int(conn.get("port", output.get("port", 5432)))
        output["user"] = conn.get("user", output.get("user"))
        output["password"] = conn.get("password", output.get("password"))
        output["dbname"] = conn.get("dbname", conn.get("database", output.get("dbname")))
        changed = True
    if not changed:
        raise WorkloadError(f"{path} has no 'postgres' output to point at the benchmark server")
    path.write_text(yaml.safe_dump(profiles, sort_keys=False))


def ensure_scratch_repo_root(scratch_dir: Path, dag_bench: Path) -> None:
    """Make the scratch tree resolve like a dag-bench checkout.

    dag-bench's postgres loader locates its own `utils/bootstrap_postgres.sql`
    by walking up from the project directory to the nearest `.git`. Our project
    copies live under the benchmark run directory, so that walk would find
    whatever repository happens to contain it and look for the file in the
    wrong place.

    Rather than reimplement the loader -- which would then drift from
    dag-bench's own -- give the scratch root the two things that walk needs: a
    `.git` marker (`.exists()` is all it tests) and a copy of `utils/`.
    """
    scratch_dir.mkdir(parents=True, exist_ok=True)
    marker = scratch_dir / ".git"
    if not marker.exists():
        marker.write_text(
            "Not a repository. This marker makes dag-bench's postgres loader\n"
            "resolve its utils/ directory here rather than in an enclosing repo.\n"
        )
    src_utils = dag_bench / "utils"
    dest_utils = scratch_dir / "utils"
    if src_utils.is_dir() and not dest_utils.exists():
        shutil.copytree(src_utils, dest_utils, ignore=shutil.ignore_patterns("__pycache__"))


def convert_dag(dee_bin: Path, prepared: PreparedProject) -> Path:
    """Convert the compiled dbt manifest into a dee DAG file."""
    run_cmd(
        [
            str(dee_bin),
            "convert",
            "--format",
            "dbt",
            "-o",
            str(prepared.dag_json),
            str(prepared.project_dir / "target" / "manifest.json"),
        ],
        timeout=600,
    )
    assert_dialect_matches(prepared.dag_json, prepared.backend)
    return prepared.dag_json


def assert_dialect_matches(dag_json: Path, backend: str) -> None:
    """Guard against running a DAG compiled for one backend against another.

    `dee-cli` picks its connector from connections.json, entirely independently
    of the DAG's own recorded dialect, so a DuckDB-compiled DAG would happily
    be handed to Postgres and fail deep inside the optimizer with a confusing
    SQL error. Catch it here instead.
    """
    dialect = (json.loads(dag_json.read_text()).get("metadata") or {}).get("sql_dialect")
    expected = {"duckdb": "duckdb", "postgres": "postgres"}[backend]
    if dialect and dialect.lower() != expected:
        raise WorkloadError(
            f"{dag_json} was compiled for sql_dialect={dialect!r} but the target backend is "
            f"{backend!r}. The dbt target and the benchmark backend have diverged."
        )


def write_connections(
    dest: Path,
    src: Path,
    backend: str,
    backend_config: dict[str, Any],
    warehouse: Path | None,
    postgres: dict[str, Any] | None,
) -> tuple[Path, str]:
    """Write the dee connections.json for this prepared project."""
    if backend == "duckdb":
        cfg: dict[str, Any] = {
            "type": "duckdb",
            "database": str(warehouse),
            "num_connections": int(backend_config.get("num_connections", 16)),
        }
        if backend_config.get("threads"):
            cfg["threads"] = int(backend_config["threads"])
        if backend_config.get("max_memory"):
            cfg["max_memory"] = str(backend_config["max_memory"])
        target = "dev"
    else:
        profile = postgres or read_profile_output(src, "postgres")
        cfg = {
            "type": "postgres",
            "host": profile.get("host", "127.0.0.1"),
            "port": int(profile.get("port", 5432)),
            "user": profile.get("user", "runner"),
            "password": profile.get("password", "password"),
            "database": profile.get("dbname", profile.get("database", "benchmark")),
            "num_connections": int(backend_config.get("num_connections", 16)),
        }
        target = "postgres"

    path = dest / "connections.json"
    path.write_text(json.dumps({target: cfg}, indent=2))
    return path, target


def apply_backend_config(
    prepared: PreparedProject,
    backend_config: dict[str, Any],
    postgres: dict[str, Any] | None = None,
) -> None:
    """Point a prepared project's connection at `backend_config`.

    Cells sharing one preparation can still differ in their backend tuning --
    two DuckDB memory ceilings over the same compiled project are two cells,
    not one. Only the connection differs between them, and `register` upserts
    it per cell, so rewriting the file is enough and the expensive part of the
    preparation is not repeated.
    """
    write_connections(
        prepared.project_dir, prepared.src_dir, prepared.backend,
        backend_config, prepared.warehouse, postgres,
    )


def read_profile_output(project_dir: Path, output: str) -> dict[str, Any]:
    """Read one output block out of a dag-bench project's profiles.yml."""
    profiles_path = project_dir / "profiles.yml"
    if not profiles_path.exists():
        raise WorkloadError(f"no profiles.yml in {project_dir}")
    profiles = yaml.safe_load(profiles_path.read_text()) or {}
    for name, block in profiles.items():
        if name == "config" or not isinstance(block, dict):
            continue
        outputs = block.get("outputs") or {}
        if output in outputs:
            return outputs[output]
    raise WorkloadError(f"{profiles_path} has no '{output}' output")


def postgres_schema_for(project_dir: Path) -> str:
    """The postgres schema a project's sources live in."""
    return str(read_profile_output(project_dir, "postgres").get("schema") or "public")


def connection_name(prepared: PreparedProject) -> str:
    """A connection name unique to this prepared project.

    Unique per (project, backend, scale factor) because each preparation builds
    its own warehouse. Reusing one name across cells would leave the server's
    cached pool holding the previous cell's database file open.
    """
    sf = f"{prepared.sf}".replace(".", "_")
    return f"{prepared.project}_{prepared.backend}_sf{sf}"


def register(client, prepared: PreparedProject, dag_name: str,
             optimizer_config: dict | None = None) -> tuple[str, dict]:
    """Register this project's connection and DAG with the server.

    Returns the connection name and the submit result. The connection is always
    upserted: replacing the config changes its hash, which is what evicts a
    pool still pointing at the previous cell's warehouse.

    `optimizer_config` is stored on the DAG, so the cell's optimizer settings
    are a property of the thing being benchmarked rather than an argument
    repeated on every request that touches it.
    """
    import json

    target = connection_name(prepared)
    config = json.loads(prepared.connections_json.read_text())[prepared.target]
    client.register_connection(target, config)

    definition = json.loads(prepared.dag_json.read_text())
    submitted = client.submit_dag(dag_name, definition, target, optimizer_config)
    return target, submitted
