#!/usr/bin/env python3
"""Measure a dbt DAG: wall clock, queries sent, and database time.

One invocation = one measurement point.  It runs the DAG `--runs` times
(after `--warmups` unrecorded runs) and reports the median of each signal,
plus the spread, so you can tell a real improvement from noise.

Both optimization signals come out of here:

  wall_s     end-to-end runtime of the dbt invocation           (signal 1)
  queries    statements dbt sent to the database                (signal 2)
  db_s       time the database spent executing them             (signal 2)

`queries`/`db_s` are parsed from the debug log dbt already writes to
logs/dbt.log ("SQL status: OK in N seconds"), so they cost nothing extra.

Per-model execution times are taken from target/run_results.json and saved
with the result, but read them knowing what they do not say: a `view` model
costs ~0ms to create, and the work its SQL describes is paid inside whichever
downstream model finally materializes.  Timings attribute cost to the node
that persists, not to the node that causes it.

Usage:
    bench.py --project ../projects/p01_iot --label baseline --runs 3 --warmups 1
    bench.py --project ... --label cand_a --runs 3 --out .opt/measurements
    bench.py --compare .opt/measurements/baseline.json .opt/measurements/cand_a.json
"""

import argparse
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time

SQL_STATUS = re.compile(r"SQL status: .* in ([0-9.]+) seconds")


def dbt_bin(explicit):
    if explicit:
        return explicit
    for cand in (
        os.path.join(os.path.dirname(sys.executable), "dbt"),
        shutil.which("dbt"),
    ):
        if cand and os.path.exists(cand):
            return cand
    sys.exit("no dbt on PATH -- pass --dbt")


def parse_log(log_path):
    """(query_count, db_seconds) from a dbt debug log."""
    if not os.path.exists(log_path):
        return None, None
    n, total = 0, 0.0
    with open(log_path, errors="replace") as f:
        for line in f:
            m = SQL_STATUS.search(line)
            if m:
                n += 1
                total += float(m.group(1))
    return n, total


def one_run(project, cmd, dbt, target, extra):
    log_path = os.path.join(project, "logs", "dbt.log")
    if os.path.exists(log_path):
        os.remove(log_path)  # so the parse below covers exactly this run
    argv = [dbt] + cmd.split() + list(extra)
    if target:
        argv += ["--target", target]
    t0 = time.perf_counter()
    proc = subprocess.run(argv, cwd=project, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        tail = (proc.stdout or "")[-3000:] + (proc.stderr or "")[-2000:]
        raise SystemExit(f"dbt failed (exit {proc.returncode}):\n{tail}")
    queries, db_s = parse_log(log_path)
    nodes = {}
    rr = os.path.join(project, "target", "run_results.json")
    if os.path.exists(rr):
        with open(rr) as f:
            data = json.load(f)
        for r in data.get("results", []):
            nodes[r["unique_id"]] = r.get("execution_time")
    return {"wall_s": wall, "queries": queries, "db_s": db_s, "nodes": nodes}


def summarize(label, runs, spent=None):
    def med(key):
        vals = [r[key] for r in runs if r.get(key) is not None]
        return statistics.median(vals) if vals else None

    walls = [r["wall_s"] for r in runs]
    return {
        "label": label,
        "runs": len(runs),
        "wall_s": med("wall_s"),
        "wall_min": min(walls),
        "wall_max": max(walls),
        "wall_stdev": statistics.stdev(walls) if len(walls) > 1 else 0.0,
        "queries": med("queries"),
        "db_s": med("db_s"),
        # What this measurement point cost to take -- warmups included, because
        # they are runs you paid the database for too. This is the "B" in the
        # payback calculation, and it is the whole reason it is recorded.
        "spent": spent or {},
        "raw": runs,
    }


def ledger_path(out_dir):
    return os.path.join(os.path.dirname(os.path.abspath(out_dir)), "ledger.json")


def ledger_append(out_dir, entry):
    """Every measurement point, appended. The running total of what the search
    has cost so far is what makes payback computable at any moment."""
    path = ledger_path(out_dir)
    log = []
    if os.path.exists(path):
        try:
            with open(path) as f:
                log = json.load(f)
        except Exception:
            log = []
    log.append(entry)
    with open(path, "w") as f:
        json.dump(log, f, indent=2)
    return path, log


def ledger_totals(out_dir):
    path = ledger_path(out_dir)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        log = json.load(f)
    return {
        "points": len(log),
        "runs": sum(e.get("runs", 0) for e in log),
        "wall_s": sum(e.get("wall_s", 0.0) for e in log),
        "queries": sum(e.get("queries", 0) for e in log),
    }


def payback(spend_s, saving_s):
    """Runs of the DAG before a change that saves `saving_s` each run has
    repaid the `spend_s` seconds spent finding it. Nothing else is needed to
    compute this -- not the schedule, not the run frequency. Those decide
    whether the answer is acceptable, not what the answer is."""
    if saving_s is None or saving_s <= 0:
        return None
    return spend_s / saving_s


def fmt(s):
    q = "-" if s["queries"] is None else f"{s['queries']:.0f}"
    d = "-" if s["db_s"] is None else f"{s['db_s']:.2f}s"
    return (
        f"{s['label']:<20} wall {s['wall_s']:.2f}s "
        f"(min {s['wall_min']:.2f} max {s['wall_max']:.2f} sd {s['wall_stdev']:.2f}, "
        f"n={s['runs']})  queries {q}  db {d}"
    )


def report_payback(a, b, out_dir):
    """What this change has to be worth, given what the search has cost.

    saving  = baseline runtime - candidate runtime, per production run
    spend   = every second of every measurement run taken so far, warmups
              included, read from the ledger
    payback = spend / saving, in runs of the DAG

    All three are measured quantities. How often the DAG actually runs is not
    an input here -- it is what you compare the answer against afterwards.
    """
    saving = a["wall_s"] - b["wall_s"]
    tot = ledger_totals(out_dir)
    this = (a.get("spent") or {}).get("wall_s", 0.0) + (b.get("spent") or {}).get("wall_s", 0.0)
    spend = tot["wall_s"] if tot else this
    scope = (f"{tot['runs']} runs across {tot['points']} measurement points"
             if tot else "these two measurement points")
    print(f"\nsearch cost so far  {spend:.1f}s  ({scope})")
    if saving <= 0:
        print("payback             never - this candidate is not faster")
        return
    pb = payback(spend, saving)
    pb_this = payback(this, saving)
    print(f"saving              {saving:.2f}s per run")
    print(f"payback             {pb:.0f} runs of the DAG to repay the whole search")
    print(f"                    {pb_this:.0f} runs to repay just these two measurements")
    if tot and tot["queries"]:
        print(f"statements spent    {tot['queries']:,} while searching")


def compare(a_path, b_path):
    a = json.load(open(a_path))
    b = json.load(open(b_path))
    print(fmt(a))
    print(fmt(b))
    dw = b["wall_s"] - a["wall_s"]  # candidate minus baseline; negative is faster
    # Test the difference against the standard error OF THE DIFFERENCE, not
    # against one sample's spread. sd describes how much a single run varies;
    # what is uncertain here is the gap between two estimates, and that
    # shrinks as sqrt(n). Comparing to 2*max(sd) is roughly sqrt(n) too
    # conservative and buries small real effects as "noise".
    na, nb = max(a["runs"], 1), max(b["runs"], 1)
    se = math.sqrt(a["wall_stdev"] ** 2 / na + b["wall_stdev"] ** 2 / nb)
    pct = 100 * dw / a["wall_s"] if a["wall_s"] else 0
    direction = "faster" if dw < 0 else "slower"
    if se == 0:
        verdict = "REAL" if dw else "NO CHANGE"
        detail = "zero variance observed"
    elif abs(dw) > 2 * se:
        verdict, detail = "REAL", f"2*SE = {2*se:.3f}s, n={na}/{nb}"
    else:
        need = math.ceil(((2 * math.sqrt(a["wall_stdev"] ** 2 + b["wall_stdev"] ** 2))
                          / abs(dw)) ** 2) if dw else 0
        verdict = "WITHIN NOISE"
        detail = (f"2*SE = {2*se:.3f}s, n={na}/{nb}; "
                  f"~{need} runs per side would resolve an effect this size")
    print(
        f"\nwall  {dw:+.2f}s ({abs(pct):.1f}% {direction})   "
        f"{verdict} ({detail})"
    )
    if a["queries"] and b["queries"]:
        print(f"query {b['queries'] - a['queries']:+.0f} statements")
    if a["db_s"] and b["db_s"]:
        print(f"db    {b['db_s'] - a['db_s']:+.2f}s")
    # per-model movers, on the shared node set
    shared = set(a["raw"][0]["nodes"]) & set(b["raw"][0]["nodes"])
    deltas = []
    for uid in shared:
        av = statistics.median([r["nodes"][uid] for r in a["raw"] if uid in r["nodes"]])
        bv = statistics.median([r["nodes"][uid] for r in b["raw"] if uid in r["nodes"]])
        deltas.append((bv - av, uid.split(".")[-1], av, bv))
    deltas.sort()
    movers = [d for d in deltas if abs(d[0]) > 0.05]
    if movers:
        show = movers[:6] if len(movers) <= 12 else movers[:6] + movers[-6:]
        print("\nper-model movers (run_results execution_time):")
        for d, name, av, bv in show:
            print(f"  {name:<34} {av:7.3f}s -> {bv:7.3f}s  {d:+.3f}s")
    report_payback(a, b, os.path.dirname(os.path.abspath(a_path)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    ap.add_argument("--label", help="name for this measurement point")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument(
        "--command",
        default="run --full-refresh",
        help="dbt subcommand and flags (default: 'run --full-refresh'). "
        "Use the SAME command for every measurement point you intend to compare.",
    )
    ap.add_argument("--target", help="dbt target (e.g. postgres)")
    ap.add_argument("--dbt", help="path to the dbt executable")
    ap.add_argument("--out", default=".opt/measurements")
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"))
    ap.add_argument("extra", nargs="*", help="extra args passed through to dbt")
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare)
        return
    if not args.label:
        sys.exit("--label is required")

    dbt = dbt_bin(args.dbt)
    project = os.path.abspath(args.project)
    print(f"# {args.label}: {args.warmups} warmup + {args.runs} recorded "
          f"`dbt {args.command}` in {project}", file=sys.stderr)

    spent = {"runs": 0, "wall_s": 0.0, "queries": 0}

    def account(r):
        spent["runs"] += 1
        spent["wall_s"] += r["wall_s"]
        spent["queries"] += r["queries"] or 0

    for i in range(args.warmups):
        account(one_run(project, args.command, dbt, args.target, args.extra))
        print(f"  warmup {i+1} done", file=sys.stderr)
    runs = []
    for i in range(args.runs):
        r = one_run(project, args.command, dbt, args.target, args.extra)
        account(r)
        runs.append(r)
        print(f"  run {i+1}: {r['wall_s']:.2f}s  {r['queries']} queries", file=sys.stderr)

    summary = summarize(args.label, runs, spent)
    out_dir = os.path.join(project, args.out) if not os.path.isabs(args.out) else args.out
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{args.label}.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    # Archive this variant's debug log next to its measurement, so decompose.py
    # can be pointed at the pair and cannot silently decompose a different
    # variant's run than the one it is reporting numbers for.
    src_log = os.path.join(project, "logs", "dbt.log")
    if os.path.exists(src_log):
        shutil.copyfile(src_log, os.path.join(out_dir, f"{args.label}.log"))
    _, log = ledger_append(out_dir, {"label": args.label, **spent})
    print(fmt(summary))
    tot = ledger_totals(out_dir)
    print(f"cost of this point: {spent['runs']} runs, {spent['wall_s']:.1f}s, "
          f"{spent['queries']:,} statements")
    print(f"search total so far: {tot['runs']} runs, {tot['wall_s']:.1f}s, "
          f"{tot['queries']:,} statements, over {tot['points']} points")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
