"""The candidate rule: a View whose SQL runs for more than one stored node
should be stored itself.

    .venv/bin/python scripts/shared_view_rule.py <run_dir> [<run_dir> ...]

A dee DAG executes one relation per node. A View is created as a view, so it
materializes nothing and its SQL is re-executed inside every stored build that
reads it -- once per Table, not once per DAG. That is the duplication
NodeFusion removes by pasting the shared models into one rollup query, and the
duplication this rule would remove instead by giving each shared model a node
of its own.

Counting consumers correctly matters more than it looks:

  * A view's SQL runs once per *stored* node that reads it, and a path that
    passes through another stored node does not count -- past that node the
    reader is scanning a table, not re-running the view.
  * So the consumers of V are the stored nodes reachable from V by a path whose
    intermediate nodes are all views.
  * Two paths to the *same* stored node still mean two executions, which is why
    paths are counted alongside consumers; a diamond under one Table duplicates
    work that a consumer count alone would call unique.

`promote` is the rule as stated: two or more stored consumers. `paths` is
reported next to it so a DAG where the two disagree is visible rather than
silently rounded off.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import duckdb

STORED = ("table", "temp_table")


def bare(node_id: str) -> str:
    return node_id.replace('"', "").split(".")[-1]


def load_dags(run_dirs: list[Path]) -> dict[str, dict]:
    """One unoptimized DAG per project, from the `dag_graph` the harness records."""
    con = duckdb.connect()
    globs, cell_globs = [], []
    for d in run_dirs:
        if list((d / "results" / "dag_graph").glob("**/*.parquet")):
            globs.append(f"'{d / 'results' / 'dag_graph'}/**/*.parquet'")
            cell_globs.append(f"'{d / 'results' / 'cells'}/**/*.parquet'")
    if not globs:
        sys.exit("no dag_graph parquet in the given run directories")
    con.execute(f"create view g as select * from read_parquet([{','.join(globs)}], "
                "hive_partitioning=1, union_by_name=1)")
    con.execute(f"create view c as select * from read_parquet([{','.join(cell_globs)}], "
                "hive_partitioning=1, union_by_name=1)")
    rows = con.execute("""
        select distinct c.project, g.node_id, g.materialization, g.depends_on
        from g join c using (cell_id)
        where g.dag_variant = 'unopt'
    """).fetchall()

    dags: dict[str, dict] = defaultdict(dict)
    for project, node_id, mat, deps in rows:
        dags[project][bare(node_id)] = {
            "materialization": mat,
            "deps": [bare(d) for d in (deps or [])],
        }
    return dags


def analyze(nodes: dict) -> dict:
    """Per view: which stored nodes re-execute it, and over how many paths."""
    children = defaultdict(list)
    for name, n in nodes.items():
        for d in n["deps"]:
            if d in nodes:
                children[d].append(name)

    def is_view(n: str) -> bool:
        return nodes[n]["materialization"] == "view"

    out = {}
    for name, n in nodes.items():
        if not is_view(name):
            continue
        consumers: set[str] = set()
        paths = 0
        # Walk downstream. A stored node terminates the walk and counts once;
        # a view continues it, because a view's reader re-runs the view's SQL.
        stack = [(c, ) for c in children[name]]
        seen_edges = 0
        while stack:
            (cur,) = stack.pop()
            seen_edges += 1
            if seen_edges > 100000:
                break
            if is_view(cur):
                stack.extend((c,) for c in children[cur])
            else:
                consumers.add(cur)
                paths += 1
        out[name] = {
            "consumers": sorted(consumers),
            "n_consumers": len(consumers),
            "paths": paths,
            "promote": len(consumers) >= 2,
        }
    return out


def main() -> None:
    run_dirs = [Path(a).resolve() for a in sys.argv[1:]]
    if not run_dirs:
        sys.exit(__doc__)
    dags = load_dags(run_dirs)

    print(f"{'project':<16} {'nodes':>6} {'views':>6} {'stored':>7} "
          f"{'promote':>8} {'dup paths saved':>16}")
    detail = {}
    for project in sorted(dags):
        nodes = dags[project]
        res = analyze(nodes)
        promoted = [v for v, r in res.items() if r["promote"]]
        # Each promoted view currently runs `paths` times and would run once.
        saved = sum(res[v]["paths"] - 1 for v in promoted)
        n_stored = sum(1 for n in nodes.values() if n["materialization"] in STORED)
        print(f"{project:<16} {len(nodes):>6} {len(res):>6} {n_stored:>7} "
              f"{len(promoted):>8} {saved:>16}")
        detail[project] = (res, promoted)

    for project, (res, promoted) in detail.items():
        if not promoted:
            continue
        print(f"\n{project}: {len(promoted)} view(s) the rule would store")
        for v in sorted(promoted, key=lambda v: -res[v]["paths"]):
            r = res[v]
            note = "" if r["paths"] == r["n_consumers"] else \
                f"  (!! {r['paths']} paths to {r['n_consumers']} consumers)"
            print(f"    {v:<28} runs {r['paths']}x for "
                  f"{r['n_consumers']} stored consumer(s){note}")


if __name__ == "__main__":
    main()
