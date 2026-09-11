"""Score every candidate VIEW-costing model against measured ground truth.

Two truths, because they turn out not to be the same question:

  gross   measured cost of computing the View once
  delta   measured change in DAG makespan from materializing it -- the decision
          the ranking is actually used to make

Calibration is fitted leave-one-DAG-out on TABLE nodes only. A candidate View
promoted to a TABLE is the evaluation label and never enters a fit.
"""

import json
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from view_costing_models import (  # noqa: E402
    fit_rates, parse, penalty, pg_total_cost, priced, query_wall_s,
)

HERE = os.environ.get("BENCH_DIR", os.path.dirname(os.path.abspath(__file__)))
BACKENDS = ("duckdb", "postgres")


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def spearman(a, b):
    n = len(a)
    if n < 3:
        return float("nan")
    def rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return num / (da * db) if da and db else float("nan")


def load(backend):
    """Per-DAG records: calibration samples, candidates, and both truths."""
    raw = json.load(open(os.path.join(HERE, f"final_{backend}.json")))
    dags = []
    for r in raw:
        cands = set(r["candidates"])
        base_ms = med(r["baseline_makespan_ms"])

        # Calibration samples: TABLE nodes of the baseline run, which are never
        # candidates, plus each candidate's own build -- the latter tagged so it
        # can be held out of every fit that scores it.
        samples = []
        for nid, mode in r["materialize"].items():
            if mode != "table" or nid in cands:
                continue
            text = r["baseline_plans"].get(nid)
            ops = [o for root in parse(backend, text) for o in root.walk()]
            if not ops:
                continue
            root = parse(backend, text)[0]
            wall = query_wall_s(backend, text)
            node_s = r["baseline_node_ms"].get(nid, 0) / 1000.0
            rows = root.rows_act
            if backend == "postgres":
                write_s = (node_s - wall) if wall else None
            else:
                # DuckDB's plan root is the CREATE_TABLE_AS operator, whose own
                # cardinality is 1 (the count it returns); the rows written are
                # its input's, exactly as `rows_written_from_plan` reads them.
                rows = root.children[0].rows_act if root.children else None
                total_cpu = root.subtree_time()
                write_s = (wall * (root.time_s or 0.0) / total_cpu
                           if wall and total_cpu > 0 else None)
            samples.append({"node": nid, "ops": ops, "rows": rows,
                            "write_s": max(write_s, 0.0) if write_s is not None else None})

        est = {}
        for key in ("leafset", "signature", "leafset_dup", "signature_dup",
                    "learned_cost", "learned_cost_dup", "dup_attribution"):
            per = {}
            for rep in r.get(key, []):
                for x in rep:
                    per.setdefault(x["node"], []).append(x)
            est[key] = {k: (med([x["cost_s"] for x in v]), med([x["cardinality"] for x in v]))
                        for k, v in per.items()}

        cands_out = []
        for t in r["truth"]:
            nid = t["node"]
            vops = [o for root in parse(backend, r["baseline_plans"].get(nid)) for o in root.walk()]
            vroots = parse(backend, r["baseline_plans"].get(nid))
            gross = (med(t["plan_compute_s"]) if backend == "duckdb" else med(t["plan_total_s"]))
            cands_out.append({
                "dag": r["dag"], "node": nid, "backend": backend,
                "gross": gross,
                "delta": base_ms - med(t["makespan_ms"]),
                "delta_sd": (st.pstdev(r["baseline_makespan_ms"]) + st.pstdev(t["makespan_ms"])) / 2,
                "rows_true": t["rows"],
                "rows_planner": (vroots[0].rows_est if vroots else None),
                "rows_attributed": est["leafset"].get(nid, (None, None))[1],
                "width": (vroots[0].width if vroots else None),
                "view_ops": vops,
                "pg_cost": pg_total_cost(vops),
                **{k: est[k].get(nid, (None, None))[0] for k in est},
            })
        dags.append({"dag": r["dag"], "samples": samples, "candidates": cands_out})
    return dags


def evaluate(backend):
    dags = load(backend)
    rows = []
    for i, d in enumerate(dags):
        # Leave-one-DAG-out: nothing from the DAG being scored enters the fit.
        train = [s for j, o in enumerate(dags) if j != i for s in o["samples"]]
        cal = fit_rates(train)
        for c in d["candidates"]:
            rows_best = (c["rows_planner"] if backend == "postgres" else c["rows_attributed"])
            rows_best = rows_best if rows_best else (c["rows_attributed"] or c["rows_planner"])
            pen = penalty(rows_best, c["width"], cal)
            pr = priced(c["view_ops"], cal)
            c["priced"] = pr
            c["penalty"] = pen
            c["rows_est"] = rows_best
            c["penalty_only"] = (-rows_best) if rows_best else None
            # Cost per row -- what `--hmp-normalize-with-cardinality` already
            # computes. A penalty-aware ranking dee can express today.
            c["dup_per_row"] = (c["leafset_dup"] / rows_best
                                if c["leafset_dup"] is not None and rows_best else None)
            c["priced_per_row"] = (pr / rows_best) if pr is not None and rows_best else None
            rows.append(c)
    return rows


MODELS = [
    ("leafset (default today)", "leafset"),
    ("signature", "signature"),
    ("leafset, downstream", "leafset_dup"),
    ("signature, downstream", "signature_dup"),
    ("learned cost", "learned_cost"),
    ("learned cost, downstream", "learned_cost_dup"),
    # Already a duplicate cost: the CTE regions of every consumer's plan, less
    # one standalone build. No "downstream" variant, because that is all it is.
    ("dup attribution (measured CTE regions)", "dup_attribution"),
    ("priced (own plan, calibrated)", "priced"),
    ("pg Total Cost (uncalibrated)", "pg_cost"),
    ("penalty only (-rows)", "penalty_only"),
    ("downstream / rows", "dup_per_row"),
    ("priced / rows", "priced_per_row"),
    ("net = downstream - L x rows (LODO)", "net_lodo"),
]


def report(rows, backend):
    rs = [r for r in rows if r["backend"] == backend]
    solid = [r for r in rs if r["delta_sd"] > 0 and abs(r["delta"]) > 2 * r["delta_sd"]]
    print(f"\n{'=' * 92}\n{backend.upper()}   {len(rs)} candidate Views · "
          f"{len(solid)} with a makespan effect above 2 sigma\n{'=' * 92}")
    print(f"{'model':32s} {'miss':>5s} {'rho vs GROSS':>13s} {'rho vs DELTA':>13s} "
          f"{'top-1 delta':>12s}")
    print("-" * 92)
    by_dag = {}
    for r in rs:
        by_dag.setdefault(r["dag"], []).append(r)
    for label, key in MODELS:
        have = [r for r in rs if r.get(key) is not None]
        if not have:
            print(f"{label:32s} {'n/a':>5s}")
            continue
        g = spearman([r[key] for r in have], [r["gross"] for r in have])
        hs = [r for r in solid if r.get(key) is not None]
        dl = spearman([r[key] for r in hs], [r["delta"] for r in hs]) if len(hs) >= 3 else float("nan")
        hits = tot = 0
        for cs in by_dag.values():
            cs2 = [c for c in cs if c["delta_sd"] > 0]
            if len(cs2) < 2:
                continue
            tot += 1
            pick = max(cs2, key=lambda r: (r[key] if r.get(key) is not None else -math.inf))
            if abs(pick["delta"] - max(c["delta"] for c in cs2)) < 1e-9:
                hits += 1
        print(f"{label:32s} {len(rs) - len(have):>5d} {g:>13.3f} {dl:>13.3f} "
              f"{str(hits) + '/' + str(tot):>12s}")


LAMBDAS = [10.0 ** e for e in range(-10, 0)]


def add_lodo_blend(rows, backend):
    """`downstream - lambda x rows`, with lambda chosen leave-one-DAG-out.

    Picking the best lambda on the same candidates it is scored against would
    report a number no future DAG could reproduce. Here each DAG is scored with
    the lambda that ranked the *other* DAGs best, so the figure is the one a
    fresh DAG would actually get.
    """
    rs = [r for r in rows if r["backend"] == backend]
    usable = [r for r in rs if r.get("leafset_dup") is not None and r.get("rows_est")
              and r["delta_sd"] > 0 and abs(r["delta"]) > 2 * r["delta_sd"]]
    for r in rs:
        train = [t for t in usable if t["dag"] != r["dag"]]
        if len(train) < 3 or r.get("leafset_dup") is None or not r.get("rows_est"):
            r["net_lodo"] = None
            continue
        tgt = [t["delta"] for t in train]
        best = max(LAMBDAS, key=lambda lam: (
            spearman([t["leafset_dup"] - lam * float(t["rows_est"]) for t in train], tgt)
            if not math.isnan(spearman([t["leafset_dup"] - lam * float(t["rows_est"])
                                        for t in train], tgt)) else -2))
        r["net_lodo"] = r["leafset_dup"] - best * float(r["rows_est"])
        r["lambda"] = best


def blend(rows, backend):
    """Does `gross - lambda x rows` beat either term alone, and how touchy is it?

    The write rate is swept rather than fitted: every TABLE node in this
    catalog writes between 3 and 136 rows, so `write_s` there is ~1 ms of fixed
    overhead carrying no per-row signal, while the candidates write 1.2M-12M
    rows. The population cannot calibrate the constant. What it can answer is
    whether the *signal* is there at all.
    """
    rs = [r for r in rows if r["backend"] == backend
          and r["delta_sd"] > 0 and abs(r["delta"]) > 2 * r["delta_sd"]
          and r.get("leafset_dup") is not None and r.get("rows_est")]
    tgt = [r["delta"] for r in rs]
    gross = [r["leafset_dup"] for r in rs]
    size = [float(r["rows_est"]) for r in rs]
    print(f"\n  blend  score = downstream - lambda x rows      (n = {len(rs)})")
    print(f"    {'lambda (s/row)':>16s} {'rho vs delta':>13s}")
    best = (float("-inf"), None)
    for e in range(-10, 0):
        lam = 10.0 ** e
        rho = spearman([g - lam * s for g, s in zip(gross, size)], tgt)
        if rho > best[0]:
            best = (rho, lam)
        print(f"    {lam:>16.0e} {rho:>13.3f}")
    print(f"    best rho {best[0]:.3f} at lambda {best[1]:.0e};  "
          f"gross alone {spearman(gross, tgt):.3f}, "
          f"rows alone {spearman([-s for s in size], tgt):.3f}")


if __name__ == "__main__":
    allrows = []
    for b in BACKENDS:
        allrows += evaluate(b)
    for b in BACKENDS:
        add_lodo_blend(allrows, b)
    for b in BACKENDS:
        report(allrows, b)
        blend(allrows, b)
    slim = [{k: v for k, v in r.items() if k != "view_ops"} for r in allrows]
    json.dump(slim, open(os.path.join(HERE, "evaluated.json"), "w"), indent=1, default=str)
    print(f"\nwrote {os.path.join(HERE, 'evaluated.json')}")
