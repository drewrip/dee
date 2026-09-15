"""Why NodeFusion's cut in query work does not become wall clock on PostgreSQL.

    .venv/bin/python scripts/nodefusion_pg_profile.py \
        results/p05-nodefusion-pg-w12-profile

Reads the `plans` table a `full`-verbosity sweep records -- PostgreSQL's
EXPLAIN (ANALYZE, VERBOSE, BUFFERS, FORMAT JSON) for every node -- and prints
what the timing run can only imply.

The timing run measures two DAGs that do very different amounts of work in
almost the same wall clock. The explanation has to be parallelism, and a plan
is where parallelism is visible:

  1. concurrency      summed node time against makespan, per variant. The
                      unoptimized DAG overlaps its duplicated work across
                      concurrently executing nodes; the rollup has one node to
                      overlap and so cannot.
  2. gathers          Workers Planned against Workers Launched, so a grant that
                      was refused is distinguished from one that was never
                      asked for.
  3. parallel-restricted
                      how much of each node's time sits under a Gather at all.
                      A CTE Scan is not parallel-safe, and the rollup is built
                      out of `AS MATERIALIZED` CTEs, so this is the number the
                      whole study turns on.

Times printed here come from the instrumented run and are inflated by the
instrumentation; they are read as proportions, never as speeds.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import duckdb

DEFAULT = Path(__file__).resolve().parents[1] / "results" / "p05-nodefusion-pg-w12-profile"
RUN = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT
if not (RUN / "results").is_dir():
    sys.exit(f"{RUN} is not a dee-bench run directory (no results/ inside)")

con = duckdb.connect()
for tbl in ("cells", "runs", "node_executions", "plans"):
    d = RUN / "results" / tbl
    if not list(d.glob("**/*.parquet")):
        sys.exit(f"{RUN.name} has no `{tbl}` -- was it run at full verbosity?")
    con.execute(f"create view {tbl} as select * from read_parquet("
                f"'{d}/**/*.parquet', hive_partitioning=1, union_by_name=1)")


def q(sql):
    return con.execute(sql).fetchall()


def short(node_id: str) -> str:
    return node_id.replace('"', "").split(".")[-1]


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


# ------------------------------------------------------------------ 1
rule("1. Work against wall clock")
print("  summed node time / makespan is how many nodes were executing at once,")
print("  averaged over the run. It is the DAG's own parallelism.\n")
print(f"  {'variant':<10} {'makespan':>10} {'node time':>11} {'overlap':>9}  nodes>1s")
for variant, wall, work in q("""
    select c.variant, median(r.engine_wall_ms), median(r.node_time_ms)
    from runs r join cells c using (cell_id)
    where r.phase = 'measure' and r.status = 'ok'
    group by 1 order by 1
"""):
    big = q(f"""
        select count(*) from (
          select n.node_id from node_executions n join cells c using (cell_id)
          where c.variant = '{variant}' group by 1 having median(n.duration_ms) > 1000)
    """)[0][0]
    print(f"  {variant:<10} {wall / 1000:9.1f}s {work / 1000:10.1f}s "
          f"{work / wall:8.2f}x  {big:>8}")

# ------------------------------------------------------------------ plans
# Only the plans of nodes that actually executed. A View is created with
# EXPLAIN (VERBOSE) and no ANALYZE -- there is nothing to analyze, since a view
# materializes nothing -- so its gathers report `Workers Launched: 0` for the
# trivial reason that the plan never ran. Counting those as refused workers
# reads the DAG's twenty-odd views as a starved server.
PLANS = q("""
    select c.variant, p.node_id, p.plan_json, n.materialization, n.duration_ms
    from plans p
    join cells c using (cell_id)
    join node_executions n on n.run_id = p.run_id and n.node_id = p.node_id
    where p.plan_format = 'postgres_json'
      and n.materialization in ('table', 'temp_table')
""")


def walk(plan, depth=0):
    yield depth, plan
    for child in plan.get("Plans", []) or []:
        yield from walk(child, depth + 1)


def root_of(plan_json):
    try:
        doc = json.loads(plan_json)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(doc, list):
        return doc[0].get("Plan") if doc else None
    return doc.get("Plan")


def self_time(node) -> float:
    """A node's own time: its total, less the children it waited on.

    `Actual Total Time` is inclusive and per-loop, so a child's contribution is
    its total times its loop count, and the remainder is the work this operator
    did itself. Parallel-aware nodes report per-worker averages, which is what
    makes the leader's share and a worker's share comparable.
    """
    total = node.get("Actual Total Time", 0.0) * max(node.get("Actual Loops", 1), 1)
    for child in node.get("Plans", []) or []:
        total -= child.get("Actual Total Time", 0.0) * max(child.get("Actual Loops", 1), 1)
    return max(total, 0.0)


# ------------------------------------------------------------------ 2
rule("2. Did the twelve-worker grant reach the queries?")
print("  Workers Planned is what the planner asked for under the grant;")
print("  Workers Launched is what the server-wide pool handed over.\n")
print("  The grant is only a ceiling. compute_parallel_worker() in")
print("  optimizer/path/allpaths.c sizes a gather from the relation: one")
print("  worker at min_parallel_table_scan_size (8MB), and one more each")
print("  time the threshold triples. Twelve workers needs a ~1.35TB table,")
print("  so on anything this size the cap is never the binding constraint.\n")
gathers = defaultdict(list)
for variant, node_id, plan_json, _mat, _ms in PLANS:
    root = root_of(plan_json)
    if root is None:
        continue
    for _, node in walk(root):
        if "Gather" in str(node.get("Node Type", "")):
            gathers[variant].append((short(node_id),
                                     node.get("Workers Planned", 0),
                                     node.get("Workers Launched", 0)))
if not gathers:
    print("  no Gather nodes at all -- nothing in this DAG went parallel")
for variant in sorted(gathers):
    items = gathers[variant]
    most = max(x[1] for x in items)
    starved = [x for x in items if x[2] < x[1]]
    print(f"  {variant:<10} {len(items):>2} gather(s) in executed nodes · "
          f"most workers the planner ever asked for: {most} of 12 granted")
    print(f"  {'':<10} {len(starved):>2} got fewer than planned "
          f"({sum(x[2] for x in starved)} launched against "
          f"{sum(x[1] for x in starved)} asked) -- the pool is shared across "
          f"concurrently executing nodes")

# ------------------------------------------------------------------ 3
rule("3. What each executed node spends in operators that cannot parallelize")
print("  A CTE Scan reads a tuplestore the leader filled; it is not")
print("  parallel-safe, so its time is serial however many workers the")
print("  server would have allowed. `AS MATERIALIZED` is what creates one,")
print("  and the rollup emits it on every CTE with two or more readers.\n")
print(f"  {'variant':<9} {'node':<24} {'plan ms':>9} {'CTE-scan ms':>12} "
      f"{'serial':>7} {'gathers':>8} {'max wrk':>8}")
rows = []
for variant, node_id, plan_json, _mat, _ms in PLANS:
    root = root_of(plan_json)
    if root is None:
        continue
    total = root.get("Actual Total Time", 0.0)
    if total < 500:
        continue
    cte_ms = 0.0
    gathers_here = 0
    max_workers = 0
    for _, node in walk(root):
        typ = str(node.get("Node Type", ""))
        if typ == "CTE Scan":
            cte_ms += self_time(node)
        if "Gather" in typ:
            gathers_here += 1
            max_workers = max(max_workers, node.get("Workers Launched", 0))
    rows.append((variant, short(node_id), total, cte_ms,
                 cte_ms / total if total else 0.0, gathers_here, max_workers))

for variant, name, total, cte_ms, frac, ngather, maxw in sorted(
        rows, key=lambda r: (r[0], -r[2])):
    print(f"  {variant:<9} {name:<24} {total / 1000:8.1f}s {cte_ms / 1000:11.1f}s "
          f"{frac * 100:6.0f}% {ngather:>8} {maxw:>8}")

# ------------------------------------------------------------------ 4
rule("4. The rollup's own hot operators")
print("  Where the fused node's time actually goes, and whether that operator")
print("  had workers. `loops` above 1 is a CTE or inner side re-executed.\n")
fused = [(v, n, p) for v, n, p, _m, _d in PLANS if short(n).startswith("dee_fused")]
if not fused:
    print("  no fused node in this run")
for variant, node_id, plan_json in fused:
    root = root_of(plan_json)
    if root is None:
        continue
    ops = []
    for _, node in walk(root):
        ops.append((self_time(node), node.get("Node Type", "?"),
                    node.get("Actual Loops", 1),
                    node.get("Workers Launched", None),
                    node.get("Relation Name") or node.get("CTE Name") or ""))
    ops.sort(reverse=True)
    print(f"  {short(node_id)} ({variant}):")
    print(f"    {'self ms':>9} {'operator':<28} {'loops':>6} {'workers':>8}  on")
    for t, typ, loops, workers, rel in ops[:14]:
        if t < 100:
            break
        w = "-" if workers in (None, 0) else str(workers)
        print(f"    {t / 1000:8.1f}s {typ:<28} {loops:>6} {w:>8}  {rel}")

print()
