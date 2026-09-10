"""Score the two VIEW-costing methods against measured ground truth."""
import json, math, os, statistics as st
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))

def median(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None

def load(backend):
    rows = json.load(open(os.path.join(HERE, f"final_{backend}.json")))
    out = []
    for r in rows:
        base_ms = median(r["baseline_makespan_ms"])
        truth = {t["node"]: t for t in r["truth"]}
        # Estimates, medianed over the baseline reps. A candidate a method
        # produced no row for at all is a miss, recorded as None.
        def est(key):
            per = {}
            for rep in r[key]:
                for x in rep:
                    per.setdefault(x["node"], []).append(x["cost_s"])
            return per
        L, S = est("leafset"), est("signature")
        for c in r["candidates"]:
            t = truth[c]
            # DuckDB's plan root is the CREATE_TABLE_AS write operator, so the
            # compute is everything below it. Postgres reports no write
            # operator at all: its plan total already is the compute.
            if backend == "duckdb":
                gt = median(t["plan_compute_s"])
            else:
                gt = median(t["plan_total_s"])
            out.append({
                "backend": backend, "dag": r["dag"], "doc": r["doc"], "node": c,
                "n_cand": len(r["candidates"]),
                "truth_s": gt,
                "truth_build_ms": median(t["build_ms"]),
                "rows": t["rows"],
                "baseline_ms": base_ms,
                "variant_ms": median(t["makespan_ms"]),
                "saving_ms": base_ms - median(t["makespan_ms"]),
                "leafset": median(L.get(c)) if c in L else None,
                "signature": median(S.get(c)) if c in S else None,
                "leafset_reps": L.get(c, []),
                "signature_reps": S.get(c, []),
                "matched": next((x["matched"] for rep in r["leafset"] for x in rep
                                 if x["node"] == c), []),
            })
    return out

EPS = 1e-6

def log_err(est, truth):
    if est is None or truth is None: return None
    return math.log10(max(est, EPS) / max(truth, EPS))

def spearman(a, b):
    n = len(a)
    if n < 2: return None
    def rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        rk = [0.0]*n; i = 0
        while i < n:
            j = i
            while j+1 < n and v[order[j+1]] == v[order[i]]: j += 1
            avg = (i+j)/2 + 1
            for k in range(i, j+1): rk[order[k]] = avg
            i = j+1
        return rk
    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra)/n, sum(rb)/n
    num = sum((x-ma)*(y-mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x-ma)**2 for x in ra)); db = math.sqrt(sum((y-mb)**2 for y in rb))
    return num/(da*db) if da and db else None

def pair_stats(rows, method):
    """Per-DAG pairwise ordering accuracy: for every pair of candidates in the
    same DAG whose true costs differ by >20%, did the method order them right?"""
    ok = bad = tie = 0
    by_dag = {}
    for r in rows: by_dag.setdefault((r["backend"], r["dag"]), []).append(r)
    detail = []
    for k, rs in by_dag.items():
        for a, b in combinations(rs, 2):
            ta, tb = a["truth_s"], b["truth_s"]
            if max(ta, tb) / max(min(ta, tb), EPS) < 1.2:
                continue
            ea, eb = a[method], b[method]
            if ea is None or eb is None:
                bad += 1; detail.append((k, a["node"], b["node"], "missing")); continue
            if ea == eb:
                tie += 1; detail.append((k, a["node"], b["node"], "tied")); continue
            right = (ea > eb) == (ta > tb)
            if right: ok += 1
            else: bad += 1; detail.append((k, a["node"], b["node"], "inverted"))
    return ok, tie, bad, detail

def dump_table(rows):
    print(f"\n{'='*118}\nPER-VIEW DETAIL\n{'='*118}")
    hdr = (f"{'backend':9s} {'dag':28s} {'view':15s} {'true_s':>8s} {'leafset':>9s} "
           f"{'sig':>9s} {'L err':>7s} {'S err':>7s} {'save_ms':>8s}")
    print(hdr); print('-'*len(hdr))
    for r in rows:
        le, se = log_err(r["leafset"], r["truth_s"]), log_err(r["signature"], r["truth_s"])
        f = lambda v: f"{v:9.4f}" if v is not None else f"{'--':>9s}"
        g = lambda v: f"{v:+7.2f}" if v is not None else f"{'--':>7s}"
        print(f"{r['backend']:9s} {r['dag']:28s} {r['node']:15s} {r['truth_s']:8.4f} "
              f"{f(r['leafset'])} {f(r['signature'])} {g(le)} {g(se)} {r['saving_ms']:8.0f}")

TRIVIAL = 0.005  # seconds: below this a View's cost is not measurable here

def summarize(rows, backend, out):
    rs = [r for r in rows if r["backend"] == backend]
    big = [r for r in rs if r["truth_s"] >= TRIVIAL]
    small = [r for r in rs if r["truth_s"] < TRIVIAL]
    print(f"\n{'='*78}\n{backend.upper()}: {len(rs)} candidate Views over "
          f"{len({r['dag'] for r in rs})} DAGs "
          f"({len(big)} measurable, {len(small)} sub-5ms)\n{'='*78}")
    for m in ("leafset", "signature"):
        errs = [(r, log_err(r[m], r["truth_s"])) for r in big]
        got = [e for _, e in errs if e is not None]
        miss = sum(1 for r in rs if r[m] is None)
        bias = st.median(got)
        spread = st.median([abs(e - bias) for e in got])
        within2 = sum(1 for e in got if abs(e - bias) < math.log10(2)) / len(got)
        worst = max(errs, key=lambda x: abs((x[1] or 0) - bias))
        ok, tie, bad, detail = pair_stats(rs, m)
        by_dag = {}
        for r in rs: by_dag.setdefault(r["dag"], []).append(r)
        hits = tot = 0
        for d, cs in by_dag.items():
            if len(cs) < 2: continue
            tot += 1
            pick = max(cs, key=lambda r: (r[m] if r[m] is not None else -1))
            best = max(cs, key=lambda r: r["truth_s"])
            if abs(pick["truth_s"] - best["truth_s"]) < 1e-12: hits += 1
        sp = spearman([r[m] if r[m] is not None else -1 for r in big],
                      [r["truth_s"] for r in big])
        # trivial Views: does the method report them as the cheapest thing around?
        triv_ok = sum(1 for r in small
                      if r[m] is not None and r[m] <= min(
                          (x[m] for x in rs if x[m] is not None and x["dag"] == r["dag"]),
                          default=0) + 1e-12)
        rec = {
            "backend": backend, "method": m, "n": len(rs), "n_measurable": len(big),
            "missing": miss, "bias_dex": bias, "bias_x": 10**bias,
            "spread_dex": spread, "spread_x": 10**spread, "within2": within2,
            "spearman": sp, "pair_ok": ok, "pair_tie": tie, "pair_wrong": bad,
            "top1": hits, "top1_of": tot, "trivial_ok": triv_ok, "trivial_n": len(small),
            "worst": {"dag": worst[0]["dag"], "node": worst[0]["node"],
                      "err_dex": worst[1], "true_s": worst[0]["truth_s"],
                      "est_s": worst[0][m]},
            "pair_detail": [{"dag": k[1], "a": a, "b": b, "why": w} for k, a, b, w in detail],
        }
        out.append(rec)
        print(f"\n  {m}:")
        print(f"    produced no cost at all for              : {miss}/{len(rs)} Views")
        print(f"    calibration bias  median log10(est/true) : {bias:+.2f}  (est = {10**bias:.2f}x truth)")
        print(f"    spread after de-biasing (MAD)            : {spread:.2f} dex -> typical {10**spread:.1f}x")
        print(f"    within 2x of the de-biased fit           : {within2*100:.0f}%")
        print(f"    Spearman rho vs truth (measurable Views) : {sp:+.3f}")
        print(f"    within-DAG pair ordering ok/tied/wrong    : {ok}/{tie}/{bad}")
        print(f"    per-DAG top-1 correct                    : {hits}/{tot}")
        print(f"    trivial Views ranked cheapest            : {triv_ok}/{len(small)}")
        print(f"    worst single View                        : {worst[0]['dag']}/"
              f"{worst[0]['node']} est {worst[0][m]} vs true {worst[0]['truth_s']:.4f}"
              f" ({worst[1]:+.2f} dex)")
        for d in rec["pair_detail"]:
            print(f"       - {d['dag']}: {d['a']} vs {d['b']}: {d['why']}")

if __name__ == "__main__":
    rows = load("duckdb") + load("postgres")
    json.dump(rows, open(os.path.join(HERE, "scored.json"), "w"), indent=1)
    dump_table(rows)
    summary = []
    for b in ("duckdb", "postgres"):
        summarize(rows, b, summary)
    json.dump(summary, open(os.path.join(HERE, "summary.json"), "w"), indent=1)
