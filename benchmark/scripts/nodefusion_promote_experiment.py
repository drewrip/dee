"""Price NodeFusion's alternative emission across projects, without building it.

    .venv/bin/python scripts/nodefusion_promote_experiment.py \
        results/p02-p09-prepare/metadata.duckdb --reps 3 --warmup 1

Three shapes per project, executed by one mini-driver against one PostgreSQL
server, so the only thing that differs between them is the DAG:

  unopt     the DAG as dbt authored it
  fused     what NodeFusion emits today: one `dee_fused` rollup node whose
            query is a WITH chain, read back by `kind`
  promote   the proposal: the models NodeFusion *itself* chose to share, each
            materialized as its own table node instead of as an
            `AS MATERIALIZED` CTE inside that one query

The shared set is not recomputed here. It is read off the rollup SQL the pass
emitted: a CTE marked `AS MATERIALIZED` is, by the pass's rule, a model with two
or more readers inside the rollup. Those CTEs are named `n_<model>`, which maps
back to the view of that name in the unoptimized DAG. Taking the set from the
output rather than reimplementing the rule is what keeps this an experiment
about *emission* and not about a second, subtly different analysis.

A promoted node keeps its name, so every downstream view and table resolves to
the table without being rewritten -- which is why this can be mocked at all,
and why implementing it for real would be contained.

The driver mirrors how dee executes: nodes run concurrently as soon as their
dependencies are met, over a fixed connection pool, inside one warm server.
Absolute times are not comparable to `runs.engine_wall_ms` -- the driver has
none of dee's per-node overhead -- but the shapes are comparable to each other.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import duckdb
import psycopg2

SHAPES = ("unopt", "fused", "promote")


def bare(node_id: str) -> str:
    return node_id.replace('"', "").split(".")[-1]


# ---------------------------------------------------------------- DAG loading
def load_projects(meta: str) -> dict[str, dict]:
    """Per project: the unoptimized DAG, the fused DAG, and the shared set.

    The `nf_rule` cell's DAG carries both versions -- version 1 as submitted,
    version 2 as NodeFusion rewrote it -- so both shapes come from one DAG and
    cannot drift apart.
    """
    con = duckdb.connect(meta, read_only=True)
    dags = con.execute("""
        select d.name, n.version, n.node_id, n.materialize, n.query_text, n.depends_on
        from dag_version_nodes n join dags d using (dag_id)
        order by d.name, n.version
    """).fetchall()

    by_dag: dict[tuple[str, int], dict] = defaultdict(dict)
    for name, version, node_id, mat, query, deps in dags:
        by_dag[(name, version)][bare(node_id)] = {
            "id": node_id,
            "name": bare(node_id),
            "materialize": mat,
            "query": query,
            "deps": [bare(d) for d in (deps or [])],
        }

    # Only DAGs that were actually rewritten have a version 2; that is the
    # NodeFusion cell, and the one this experiment is about.
    out: dict[str, dict] = {}
    for (name, version), nodes in sorted(by_dag.items()):
        if version != 2:
            continue
        unopt = by_dag.get((name, 1))
        if not unopt:
            continue
        fused_node = next((n for n in nodes.values()
                           if n["name"].startswith("dee_fused")), None)
        if fused_node is None:
            continue
        schema = fused_node["id"].replace('"', "").split(".")[-2]
        project = re.sub(r"^synth_(multi|one)_", "", schema)
        shared = shared_models(fused_node["query"], unopt)
        out[project] = {
            "schema": schema, "dag": name,
            "unopt": unopt, "fused": nodes, "shared": shared,
        }
    return out


def shared_models(fused_sql: str, unopt: dict) -> list[str]:
    """The models the pass marked `AS MATERIALIZED` inside the rollup."""
    names = re.findall(r'\b(n_[A-Za-z_0-9]+) AS MATERIALIZED \(', fused_sql)
    out = []
    for n in names:
        model = n[2:]           # strip the rollup's `n_` prefix
        if model in unopt and unopt[model]["materialize"] == "view":
            out.append(model)
        else:
            print(f"    !! `{n}` does not map to a view of the DAG; skipped")
    return out


def make_promote(unopt: dict, shared: list[str]) -> dict:
    out = {k: dict(v) for k, v in unopt.items()}
    for name in shared:
        out[name]["materialize"] = "table"
    return out


# ---------------------------------------------------------------- execution
def ddl(node: dict) -> str:
    kind = "VIEW" if node["materialize"] == "view" else "TABLE"
    return f'CREATE {kind} {node["id"]} AS ({node["query"]})'


def reset(cur, shapes: list[dict]) -> None:
    """Drop everything any shape creates, so a repetition starts from sources."""
    for nodes in shapes:
        for n in nodes.values():
            for kind in ("VIEW", "TABLE"):
                try:
                    cur.execute(f'DROP {kind} IF EXISTS {n["id"]} CASCADE')
                except psycopg2.Error:
                    cur.connection.rollback()


def run_shape(dsn: str, nodes: dict, pool: int) -> tuple[float, dict]:
    pending = {k: set(v["deps"]) & set(nodes) for k, v in nodes.items()}
    done: set[str] = set()
    timings: dict[str, float] = {}
    conns = [psycopg2.connect(dsn) for _ in range(pool)]
    for c in conns:
        c.autocommit = True
    free = list(conns)
    start = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=pool) as ex:
            running: dict = {}

            def launch(name: str):
                conn = free.pop()

                def work():
                    t0 = time.monotonic()
                    with conn.cursor() as cur:
                        cur.execute(ddl(nodes[name]))
                    return name, conn, time.monotonic() - t0
                return ex.submit(work)

            while len(done) < len(nodes):
                ready = [n for n, d in pending.items()
                         if n not in done and n not in running and not (d - done)]
                while ready and free:
                    n = ready.pop(0)
                    running[n] = launch(n)
                if not running:
                    raise RuntimeError("deadlock: nothing runnable, nothing in flight")
                for fut in as_completed(list(running.values())):
                    name, conn, secs = fut.result()
                    timings[name] = secs * 1000.0
                    free.append(conn)
                    done.add(name)
                    running.pop(name)
                    break
        return (time.monotonic() - start) * 1000.0, timings
    finally:
        for c in conns:
            c.close()


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("metadata")
    ap.add_argument("--dsn", default="host=127.0.0.1 port=55433 user=runner "
                                     "password=password dbname=benchmark")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--pool", type=int, default=16)
    ap.add_argument("--projects", default="")
    ap.add_argument("--out", default="promote_experiment.json")
    args = ap.parse_args()

    projects = load_projects(args.metadata)
    if args.projects:
        want = set(args.projects.split(","))
        projects = {k: v for k, v in projects.items() if k in want}
    if not projects:
        sys.exit("no projects with a NodeFusion-rewritten DAG in that metadata db")

    admin = psycopg2.connect(args.dsn)
    admin.autocommit = True
    results: dict[str, dict] = {}

    for project in sorted(projects):
        p = projects[project]
        shapes = {"unopt": p["unopt"], "fused": p["fused"],
                  "promote": make_promote(p["unopt"], p["shared"])}
        print(f"\n{project}  ({len(p['unopt'])} nodes, "
              f"shares {len(p['shared'])}: {', '.join(p['shared']) or 'none'})",
              flush=True)
        if not p["shared"]:
            print("  no materialized CTEs -- the pass shared nothing here", flush=True)
        results[project] = {"shared": p["shared"], "schema": p["schema"], "shapes": {}}

        for shape in SHAPES:
            samples, last, failed = [], {}, None
            for i in range(args.warmup + args.reps):
                with admin.cursor() as cur:
                    reset(cur, list(shapes.values()))
                try:
                    ms, timings = run_shape(args.dsn, shapes[shape], args.pool)
                except Exception as exc:                      # noqa: BLE001
                    failed = f"{type(exc).__name__}: {exc}"
                    print(f"  {shape:<8} FAILED  {failed[:120]}", flush=True)
                    break
                phase = "warmup" if i < args.warmup else "measure"
                print(f"  {shape:<8} {phase:<7} {ms / 1000:7.1f}s", flush=True)
                if phase == "measure":
                    samples.append(ms)
                    last = timings
            results[project]["shapes"][shape] = {
                "samples": samples, "nodes": last, "error": failed,
            }
        with admin.cursor() as cur:
            reset(cur, list(shapes.values()))

    print("\n" + "=" * 78)
    print(f"{'project':<16} {'shared':>7} {'unopt':>9} {'fused':>9} {'promote':>9} "
          f"{'fused x':>8} {'promote x':>10}")
    for project in sorted(results):
        r = results[project]
        med = {}
        for s in SHAPES:
            smp = r["shapes"][s]["samples"]
            med[s] = statistics.median(smp) if smp else None
        cell = lambda s: f"{med[s] / 1000:8.1f}s" if med[s] else "        -"  # noqa: E731
        sp = lambda s: (f"{med['unopt'] / med[s]:9.2f}x"                      # noqa: E731
                        if med[s] and med["unopt"] else "         -")
        print(f"{project:<16} {len(r['shared']):>7} {cell('unopt')} {cell('fused')} "
              f"{cell('promote')} {sp('fused')} {sp('promote')}")

    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
