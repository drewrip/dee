"""Measure the makespan of three DAG shapes against one PostgreSQL server.

A stand-in for the NodeFusion emission change, so its value can be priced
before it is built. Three shapes, executed by the same mini-driver so the only
difference between them is the DAG:

  unopt   the DAG as dbt authored it -- 7 tables over 17 views, the views
          re-executed once per table that reads them
  fused   what NodeFusion emits today -- one `dee_fused` rollup node whose
          query is a WITH chain, read back by `kind`
  nodes   the proposed emission -- the same models NodeFusion chose to share,
          but each materialized as its own table node rather than as an
          `AS MATERIALIZED` CTE inside one query. Downstream views are
          untouched: a table replaces the view under the same name, so every
          reader resolves to it without being rewritten.

The driver mirrors how dee executes: nodes run concurrently as soon as their
dependencies are met, over a fixed connection pool, inside one warm server.
Makespan is the wall clock of the whole graph, which is what `runs.engine_wall_ms`
records in the harness.

  python shape_bench.py <dags.json> --reps 3 --warmup 1
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import psycopg2

SCHEMA = '"benchmark"."synth_multi_p05_hr"'
# The models NodeFusion's own rule picked to share on this DAG: the CTEs it
# emitted `AS MATERIALIZED`, which are exactly the ones with two or more
# readers inside the rollup. The proposed shape promotes these same three.
SHARED = ("stg_employees", "current_salary", "employee_profile")


def bare(node_id: str) -> str:
    return node_id.replace('"', "").split(".")[-1]


def load(path: str):
    rows = json.load(open(path))
    shapes: dict[str, dict] = {}
    for r in rows:
        dep = r["depends_on"]
        if isinstance(dep, str):
            dep = json.loads(dep)
        shapes.setdefault(r["shape"], {})[bare(r["node_id"])] = {
            "name": bare(r["node_id"]),
            "materialize": r["materialize"],
            "query": r["query_text"],
            "deps": [bare(d) for d in (dep or [])],
        }
    return shapes


def make_nodes_shape(unopt: dict) -> dict:
    """The proposed emission: promote the shared views to table nodes.

    Nothing else changes. The promoted node keeps its name, so the views and
    tables that read it need no rewriting -- which is the point of doing it
    this way rather than by pasting the models into one query.
    """
    out = {k: dict(v) for k, v in unopt.items()}
    for name in SHARED:
        if name not in out:
            sys.exit(f"{name} is not a node of the unoptimized DAG")
        if out[name]["materialize"] != "view":
            sys.exit(f"{name} is already materialized as {out[name]['materialize']}")
        out[name]["materialize"] = "table"
    return out


def reset(cur, nodes: dict) -> None:
    """Drop everything the DAG creates, so a repetition starts from sources.

    CASCADE rather than a reverse topological order: a shape that turns a view
    into a table changes what depends on what, and the teardown should not have
    to know that.
    """
    for n in nodes.values():
        kind = "VIEW" if n["materialize"] == "view" else "TABLE"
        other = "TABLE" if kind == "VIEW" else "VIEW"
        for k in (kind, other):
            try:
                cur.execute(f'DROP {k} IF EXISTS {SCHEMA}."{n["name"]}" CASCADE')
            except psycopg2.Error:
                cur.connection.rollback()


def ddl(node: dict) -> str:
    kind = "VIEW" if node["materialize"] == "view" else "TABLE"
    return f'CREATE {kind} {SCHEMA}."{node["name"]}" AS ({node["query"]})'


def run_shape(dsn: str, nodes: dict, pool: int) -> tuple[float, dict]:
    """Execute the graph, nodes concurrently once their dependencies are done."""
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
            running = {}

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
                    sys.exit("deadlock: no runnable node and nothing in flight")
                for fut in as_completed(list(running.values())):
                    name, conn, secs = fut.result()
                    timings[name] = secs * 1000.0
                    free.append(conn)
                    done.add(name)
                    running.pop(name)
                    break
        makespan = (time.monotonic() - start) * 1000.0
    finally:
        for c in conns:
            c.close()
    return makespan, timings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dags")
    ap.add_argument("--dsn", default="host=127.0.0.1 port=55442 user=runner "
                                     "password=password dbname=benchmark")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--pool", type=int, default=16)
    args = ap.parse_args()

    shapes = load(args.dags)
    shapes["nodes"] = make_nodes_shape(shapes["unopt"])

    admin = psycopg2.connect(args.dsn)
    admin.autocommit = True

    results = {}
    for shape in ("unopt", "fused", "nodes"):
        nodes = shapes[shape]
        samples, last = [], {}
        for i in range(args.warmup + args.reps):
            with admin.cursor() as cur:
                for s in shapes.values():
                    reset(cur, s)
            ms, timings = run_shape(args.dsn, nodes, args.pool)
            phase = "warmup" if i < args.warmup else "measure"
            print(f"  {shape:<6} {phase:<7} {ms / 1000:7.1f}s", flush=True)
            if phase == "measure":
                samples.append(ms)
                last = timings
        results[shape] = (samples, last)

    print("\n" + "=" * 62)
    print(f"{'shape':<8} {'median':>9} {'min':>9} {'max':>9} {'node time':>11} {'overlap':>8}")
    base = statistics.median(results["unopt"][0])
    for shape in ("unopt", "fused", "nodes"):
        s, t = results[shape]
        med = statistics.median(s)
        work = sum(t.values())
        print(f"{shape:<8} {med / 1000:8.1f}s {min(s) / 1000:8.1f}s {max(s) / 1000:8.1f}s "
              f"{work / 1000:10.1f}s {work / med:7.2f}x")
    print()
    for shape in ("fused", "nodes"):
        med = statistics.median(results[shape][0])
        print(f"  {shape:<6} speedup vs unopt: {base / med:.2f}x")

    print("\ntop nodes, last measured run")
    for shape in ("unopt", "fused", "nodes"):
        t = results[shape][1]
        top = sorted(t.items(), key=lambda kv: -kv[1])[:6]
        print(f"  {shape}: " + ", ".join(f"{k} {v / 1000:.1f}s" for k, v in top))

    json.dump({k: {"samples": v[0], "nodes": v[1]} for k, v in results.items()},
              open("shape_bench_results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
