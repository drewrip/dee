#!/usr/bin/env python3
"""Static analysis of a dbt DAG: what gets recomputed, and how often.

Reads target/manifest.json (and target/run_results.json if present) and
reports, per model:

  mat      declared materialization
  expand   how many times this model's SQL is expanded into a query the
           database actually plans.  A table is built once (1).  A view is
           re-expanded once per materialized consumer, transitively, counting
           every distinct path -- which is the number of times its work is
           really done.
  refs     direct upstream models
  fanout   direct downstream models
  exec_s   execution_time from the last run_results.json, if available

Costs nothing: no queries, no build.  Run it before spending anything.

Usage:
    analyze.py [project_dir] [--json] [--top N]
"""

import argparse
import json
import os
import sys
from collections import defaultdict


def load(project_dir, name):
    path = os.path.join(project_dir, "target", name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def build_graph(manifest):
    """Return (models, children) over model nodes only."""
    models = {}
    for uid, node in manifest["nodes"].items():
        if node["resource_type"] != "model":
            continue
        models[uid] = {
            "name": node["name"],
            "path": node.get("original_file_path"),
            "mat": (node.get("config") or {}).get("materialized", "view"),
            "parents": [],
            "sources": [],
        }
    for uid, deps in manifest.get("parent_map", {}).items():
        if uid not in models:
            continue
        for p in deps:
            if p in models:
                models[uid]["parents"].append(p)
            elif p.startswith("source."):
                models[uid]["sources"].append(p)
    children = defaultdict(list)
    for uid, m in models.items():
        for p in m["parents"]:
            children[p].append(uid)
    return models, children


def expansion_counts(models, children):
    """How many times each model's SQL is planned by the database.

    A model whose relation is persisted (table/incremental/materialized view)
    is computed once.  An ephemeral or view model is inlined into each
    consumer's query, so its count is the sum of its consumers' counts --
    every distinct path through views is a separate recomputation.
    """
    PERSISTED = {"table", "incremental", "materialized_view", "snapshot"}
    memo = {}

    def count(uid, stack=()):
        if uid in memo:
            return memo[uid]
        if uid in stack:  # cycle guard; dbt forbids these, but be safe
            return 1
        node = models[uid]
        kids = children.get(uid, [])
        if node["mat"] in PERSISTED:
            n = 1
        elif not kids:
            # A leaf view is still created (cheap) but never expanded by a
            # consumer.  Someone queries it eventually; count it once.
            n = 1
        else:
            n = sum(count(k, stack + (uid,)) for k in kids)
        memo[uid] = n
        return n

    return {uid: count(uid) for uid in models}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project_dir", nargs="?", default=".")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--top", type=int, default=0, help="only the N highest-expansion models")
    args = ap.parse_args()

    manifest = load(args.project_dir, "manifest.json")
    if manifest is None:
        sys.exit(
            "no target/manifest.json -- run `dbt parse` in the project first "
            "(parsing is free, it sends no queries)"
        )
    results = load(args.project_dir, "run_results.json")
    timings = {}
    if results:
        for r in results.get("results", []):
            timings[r["unique_id"]] = r.get("execution_time")

    models, children = build_graph(manifest)
    expand = expansion_counts(models, children)

    rows = []
    for uid, m in models.items():
        rows.append(
            {
                "name": m["name"],
                "unique_id": uid,
                "materialized": m["mat"],
                "expansions": expand[uid],
                "parents": [models[p]["name"] for p in m["parents"]],
                "sources": [s.split(".")[-1] for s in m["sources"]],
                "children": [models[c]["name"] for c in children.get(uid, [])],
                "exec_s": timings.get(uid),
                "path": m["path"],
            }
        )
    rows.sort(key=lambda r: (-r["expansions"], r["name"]))
    # Totals are over the whole DAG, not over whatever --top shows.
    total_expand = sum(r["expansions"] for r in rows)
    shown = rows[: args.top] if args.top else rows

    if args.json:
        json.dump(shown, sys.stdout, indent=2)
        print()
        return

    print(f"{len(models)} models, {total_expand} total SQL expansions"
          + (f" (showing top {len(shown)})" if len(shown) < len(rows) else "") + "\n")
    print(f"{'model':<34} {'mat':<12} {'expand':>6} {'exec_s':>8}  upstream")
    print("-" * 100)
    for r in shown:
        exec_s = f"{r['exec_s']:.3f}" if r["exec_s"] is not None else "-"
        ups = ", ".join(r["parents"] + r["sources"]) or "-"
        print(
            f"{r['name']:<34} {r['materialized']:<12} {r['expansions']:>6} "
            f"{exec_s:>8}  {ups[:40]}"
        )
    print()
    hot = [r for r in rows if r["expansions"] > 1 and r["materialized"] == "view"]
    if hot:
        print("views recomputed more than once (materialization candidates,")
        print("highest expansion first -- but confirm the subtree is actually")
        print("expensive before spending a trial on it):")
        for r in hot[:10]:
            print(f"  {r['name']:<34} x{r['expansions']}")


if __name__ == "__main__":
    main()
