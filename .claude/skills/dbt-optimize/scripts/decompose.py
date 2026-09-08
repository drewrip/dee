#!/usr/bin/env python3
"""Decompose a dbt run's wall clock. Costs nothing: no queries, no builds.

Everything here comes from artefacts a run you already paid for left behind -
the debug log in `logs/dbt.log` and, optionally, a bench.py measurement. It
answers the question that decides whether optimizing this project is worth
any budget at all:

    of the wall clock, how much is the DAG, and how much of the DAG is
    reducible?

It reports:

  * per-node database time and statement count, from the log
  * the node execution window vs. total wall clock -- the difference is dbt's
    own startup, parse, and teardown, which no SQL change can touch
  * achieved concurrency against the configured thread count
  * the share of database time held by the top few nodes
  * a headroom verdict, and whether to spend anything further

Run it immediately after the baseline, BEFORE the all-tables probe. It is
free, and it is allowed to end the exercise.

    decompose.py --project P --measurement .opt/measurements/baseline.json
    decompose.py --project P --time-parse --dbt /path/to/dbt
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime

TS = re.compile(r"(\d\d:\d\d:\d\d\.\d+) \[")
THREAD = re.compile(r"\[(Thread-\d+[^\]]*|MainThread)\]")
NODE = re.compile(r"On ((?:model|test|snapshot|seed)\.[\w.]+):")
STATUS = re.compile(r"SQL status: .* in ([0-9.]+) seconds")


INVOCATION = re.compile(r"Running with dbt=")


def last_invocation(path):
    """The most recent `dbt` invocation that actually executed SQL.

    dbt appends to logs/dbt.log, so a log that has seen several commands holds
    all of them. bench.py truncates before each run; a bare `dbt run` does not,
    and parsing the accumulated file silently sums every invocation together -
    which looks like a DAG many times more expensive than it is. Later
    invocations are also often a `parse`, `show`, or `compile` that sent no
    statements at all, so walk backwards to the last one that did.
    """
    with open(path, errors="replace") as f:
        lines = f.readlines()
    starts = [i for i, l in enumerate(lines) if INVOCATION.search(l)]
    if not starts:
        return lines
    bounds = list(zip(starts, starts[1:] + [len(lines)]))
    for a, b in reversed(bounds):
        chunk = lines[a:b]
        if any(STATUS.search(l) for l in chunk):
            return chunk
    return lines[starts[-1]:]


def parse_log(path):
    """Per-node db time, statement count, and first/last timestamp."""
    cur, db, cnt, span = {}, defaultdict(float), defaultdict(int), {}
    for line in last_invocation(path):
        th_m = THREAD.search(line)
        th = th_m.group(1) if th_m else "?"
        n_m = NODE.search(line)
        if n_m:
            cur[th] = n_m.group(1)
            t_m = TS.search(line)
            if t_m:
                ts = datetime.strptime(t_m.group(1), "%H:%M:%S.%f")
                sp = span.setdefault(n_m.group(1), [ts, ts])
                sp[0], sp[1] = min(sp[0], ts), max(sp[1], ts)
        s_m = STATUS.search(line)
        if s_m and th in cur:
            db[cur[th]] += float(s_m.group(1))
            cnt[cur[th]] += 1
    return db, cnt, span


def threads_of(project):
    """Configured thread count, from the manifest's resolved profile."""
    try:
        with open(os.path.join(project, "target", "manifest.json")) as f:
            meta = json.load(f)["metadata"]
        return meta.get("threads")
    except Exception:
        return None


def time_parse(dbt, project, n=2):
    """Wall time of `dbt parse` - dbt's fixed cost, sending zero queries."""
    best = None
    for _ in range(n):
        t0 = time.perf_counter()
        p = subprocess.run([dbt, "parse"], cwd=project, capture_output=True, text=True)
        el = time.perf_counter() - t0
        if p.returncode == 0:
            best = el if best is None else min(best, el)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    ap.add_argument("--measurement", help="a bench.py JSON, for the wall-clock comparison. "
                    "If a sibling <label>.log exists (bench.py archives one per "
                    "variant) it is decomposed instead of logs/dbt.log, so the "
                    "breakdown always describes the run the numbers came from.")
    ap.add_argument("--log", help="explicit debug log to decompose")
    ap.add_argument("--time-parse", action="store_true",
                    help="also time `dbt parse` (free: sends no queries)")
    ap.add_argument("--dbt")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    project = os.path.abspath(args.project)
    log = args.log
    if not log and args.measurement:
        cand = os.path.splitext(os.path.abspath(args.measurement))[0] + ".log"
        if os.path.exists(cand):
            log = cand
    if not log:
        log = os.path.join(project, "logs", "dbt.log")
        if args.measurement:
            print(f"note: decomposing {log}, which is whatever ran last -- it may not "
                  f"be the\n      variant {os.path.basename(args.measurement)} measured.\n")
    if not os.path.exists(log):
        sys.exit(f"no {log} -- run the DAG once first (bench.py leaves one behind)")

    db, cnt, span = parse_log(log)
    if not db:
        sys.exit("no SQL statements found in the log -- is it from a `run`?")

    total_db = sum(db.values())
    total_stmt = sum(cnt.values())
    t0 = min(s[0] for s in span.values())
    t1 = max(s[1] for s in span.values())
    window = (t1 - t0).total_seconds()
    busy = sum((s[1] - s[0]).total_seconds() for s in span.values())

    print(f"{'node':<38}{'db_s':>8}{'stmts':>7}")
    print("-" * 55)
    ranked = sorted(db.items(), key=lambda x: -x[1])
    for k, v in ranked[: args.top]:
        print(f"{k.split('.')[-1]:<38}{v:8.3f}{cnt[k]:7d}")
    rest = ranked[args.top:]
    if rest:
        print(f"{f'... {len(rest)} more nodes':<38}"
              f"{sum(v for _, v in rest):8.3f}{sum(cnt[k] for k, _ in rest):7d}")
    print("-" * 55)
    print(f"{'TOTAL':<38}{total_db:8.3f}{total_stmt:7d}\n")

    top3 = sum(v for _, v in ranked[:3])
    conc = max(busy / window, 1.0)
    nthreads = threads_of(project)
    print(f"node execution window     {window:6.2f}s")
    print(f"sum of per-node spans     {busy:6.2f}s"
          f"   -> concurrency {busy/window:.2f}x"
          + (f" (threads configured: {nthreads})" if nthreads else ""))
    print(f"database time             {total_db:6.2f}s   in {total_stmt} statements")
    print(f"top 3 nodes               {top3:6.2f}s   "
          f"= {100*top3/total_db:.0f}% of database time")

    wall = None
    if args.measurement:
        with open(args.measurement) as f:
            m = json.load(f)
        wall = m["wall_s"]
        print(f"wall clock                {wall:6.2f}s   ({m['label']}, n={m['runs']})")
        print(f"  outside node execution  {wall-window:6.2f}s   "
              f"= {100*(wall-window)/wall:.0f}% of wall - dbt startup, parse, teardown")

    if args.time_parse:
        dbt = args.dbt or shutil.which("dbt")
        if not dbt:
            print("\n(--time-parse needs --dbt or dbt on PATH)")
        else:
            p = time_parse(dbt, project)
            if p:
                print(f"`dbt parse` alone         {p:6.2f}s   fixed cost, zero queries sent")

    print("\n" + "=" * 55)
    print("HEADROOM")
    print("=" * 55)
    # The ceiling on wall clock is the elapsed window the DAG occupies, NOT the
    # summed database time. Statements run concurrently, so the sum can exceed
    # wall clock outright (it does whenever concurrency > 1) and treating it as
    # a ceiling overstates the opportunity by exactly that factor.
    print(f"Total database work is {total_db:.2f}s, but it runs"
          f" {busy/window:.1f}x concurrent, so the")
    print(f"elapsed window the DAG occupies -- {window:.2f}s -- is the real ceiling:")
    print(f"making every model instant saves at most that.")
    if wall:
        share = 100 * window / wall
        print(f"That is {share:.0f}% of wall clock. Halving the DAG moves wall clock"
              f" by {50*window/wall:.0f}%.")
        # Convert the top-3 database work into elapsed time the same way the
        # rest of the DAG is measured -- divided by achieved concurrency, and
        # capped by the window, since no change can save more elapsed time than
        # the DAG occupies.
        top3_elapsed = min(top3 / conc, window)
        realistic = 0.3 * top3_elapsed
        print(f"\nThe top 3 nodes are {top3:.2f}s of database work ="
              f" ~{top3_elapsed:.2f}s elapsed at {conc:.1f}x concurrency.")
        print(f"Making them free -- which no real change does -- saves at most that,"
              f" {100*top3_elapsed/wall:.0f}% of wall.")
        print(f"A realistic 30% cut of them is {realistic:.2f}s"
              f" ({100*realistic/wall:.0f}% of wall).")
        if realistic > 0:
            print(f"At {wall:.2f}s per run, one trial run costs {wall:.2f}s, so such a"
                  f" change needs")
            print(f"{wall/realistic:.0f} production runs per trial run spent to break even.")
        if share < 40:
            print("\nVERDICT: dbt's own fixed overhead dominates this DAG. Materialization")
            print("and SQL work have little to move, and per-node overhead means adding")
            print("nodes actively costs you. Report this and stop, or change the")
            print("conditions - a bigger scale factor, or the other adapter - before")
            print("spending a budget here.")
        elif conc >= 1.8:
            print(f"\nVERDICT: this DAG already runs {conc:.1f}x concurrent. The work that"
                  f" expansion")
            print("counts call duplicated is being done in parallel on threads that would")
            print("otherwise idle, so it is close to free in elapsed time -- and")
            print("materializing it replaces free parallel work with a serialization")
            print("barrier every consumer waits on. Measured across five projects, that")
            print("trade lost wall clock every time concurrency was above ~2, even where")
            print("it removed 38% of the database work. Treat materialization as")
            print("unpromising here; target the hot nodes' SQL instead. (If you are")
            print("optimizing database cost rather than latency, the trade may still be")
            print("worth it -- decide which signal you are buying.)")
        elif top3 / total_db > 0.6:
            print(f"\nVERDICT: {100*top3/total_db:.0f}% of database time is in 3 nodes. Read those three")
            print("models before spending anything: the whole win, if there is one, is in")
            print("them, and a broad materialization search will not find it.")
        else:
            print("\nVERDICT: database time is a real share of wall clock and spread across")
            print("nodes. A materialization search is worth its probe run.")
    print()


if __name__ == "__main__":
    main()
