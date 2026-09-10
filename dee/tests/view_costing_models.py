"""Candidate VIEW-costing models, scored offline against measured ground truth.

Prototyped here rather than in Rust so that a model can be discarded cheaply.
Only a model that earns its place gets ported into `dee/src/opt/`.

Every model answers the same question the HMP ranking asks -- *which of these
Views is worth materializing* -- and is scored against two different truths:

  gross   the measured cost of computing the View once (what `hmp_cost_method`
          claims to estimate today)
  delta   the measured change in DAG makespan from materializing it (what the
          ranking is actually used to decide)

Run via `view_costing_analyze.py`.
"""

import json
import math
import statistics as st

EPS = 1e-9


# ---------------------------------------------------------------------------
# Plan parsing
#
# A deliberately small mirror of `dee/src/plan.rs`: enough of each backend's
# plan to price it, including the parallel-worker divisor, so that a model
# prototyped here can be ported without the numbers moving.
# ---------------------------------------------------------------------------

class Op:
    __slots__ = ("op", "rows_est", "rows_act", "time_s", "relation", "width", "cost", "children")

    def __init__(self, op, rows_est=None, rows_act=None, time_s=None,
                 relation=None, width=None, cost=None, children=None):
        self.op = op
        self.rows_est = rows_est
        self.rows_act = rows_act
        self.time_s = time_s
        self.relation = relation
        self.width = width
        self.cost = cost
        self.children = children or []

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()

    def subtree_time(self):
        return sum(n.time_s or 0.0 for n in self.walk())


def _pg(node, procs=1.0):
    loops = node.get("Actual Loops") or 1.0
    t = node.get("Actual Total Time")
    incl = (t * loops / procs / 1000.0) if t is not None else None
    kids = node.get("Plans") or []
    # Below a Gather, `Actual Loops` counts parallel processes, not sequential
    # passes; the Gather's immediate child names that count exactly.
    child_procs = procs
    if node.get("Node Type") in ("Gather", "Gather Merge") and kids:
        child_procs = (kids[0].get("Actual Loops") or procs) or procs
    children = [_pg(k, child_procs) for k in kids]
    excl = None
    if incl is not None:
        inner = sum(
            ((k.get("Actual Total Time") or 0.0) * (k.get("Actual Loops") or 1.0)
             / child_procs / 1000.0)
            for k in kids if k.get("Actual Total Time") is not None
        )
        excl = max(incl - inner, 0.0)
    ar = node.get("Actual Rows")
    return Op(
        op=node.get("Node Type", "?"),
        rows_est=node.get("Plan Rows"),
        rows_act=(ar * loops) if ar is not None else None,
        time_s=excl,
        relation=node.get("Relation Name"),
        width=node.get("Plan Width"),
        cost=node.get("Total Cost"),
        children=children,
    )


def _duck(node):
    ei = node.get("extra_info") or {}
    est = ei.get("Estimated Cardinality")
    try:
        est = float(est) if est is not None else None
    except (TypeError, ValueError):
        est = None
    name = node.get("operator_name") or node.get("name")
    children = [_duck(k) for k in (node.get("children") or [])]
    # DuckDB's profiling root is an unnamed query wrapper; splice it out.
    if not name:
        return children
    return Op(
        op=name,
        rows_est=est,
        rows_act=node.get("operator_cardinality"),
        time_s=node.get("operator_timing"),
        relation=(ei.get("Table") or "").rsplit(".", 1)[-1].lower() or None,
        width=None,
        cost=None,
        children=[c for k in children for c in (k if isinstance(k, list) else [k])],
    )


def parse(backend, text):
    """Roots of a plan, or [] when the text is not a plan this backend emits."""
    if not text:
        return []
    try:
        doc = json.loads(text)
    except (ValueError, TypeError):
        return []
    if backend == "postgres":
        return [_pg(w["Plan"]) for w in doc if "Plan" in w]
    out = _duck(doc if isinstance(doc, dict) else {"children": doc, "name": None})
    return out if isinstance(out, list) else [out]


def query_wall_s(backend, text):
    """The wall-clock seconds the backend says the whole query took."""
    try:
        doc = json.loads(text)
    except (ValueError, TypeError):
        return None
    if backend == "postgres":
        d = doc[0] if isinstance(doc, list) else doc
        et, pt = d.get("Execution Time"), d.get("Planning Time")
        return ((et or 0.0) + (pt or 0.0)) / 1000.0 or None
    return doc.get("latency") if isinstance(doc, dict) else None


# ---------------------------------------------------------------------------
# Operator classes
#
# Three, not fifteen. Twelve DAGs buys a handful of free parameters, and
# per-operator-name rates would be rank-deficient anyway: scans never occur
# without the joins and aggregates above them, so the fit would be stable in
# prediction and arbitrary in its coefficients.
# ---------------------------------------------------------------------------

SCAN = ("SEQ_SCAN", "Seq Scan", "Index Scan", "Index Only Scan", "Bitmap Heap Scan",
        "TABLE_SCAN", "PARQUET_SCAN")
BLOCKING = ("HASH_JOIN", "HASH_GROUP_BY", "PERFECT_HASH_GROUP_BY", "UNGROUPED_AGGREGATE",
            "ORDER_BY", "WINDOW", "Hash Join", "Hash", "Aggregate", "HashAggregate",
            "GroupAggregate", "Sort", "WindowAgg", "Merge Join", "Materialize", "Memoize")


def op_class(op):
    if op in SCAN:
        return "scan"
    if op in BLOCKING:
        return "blocking"
    return "stream"


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def median_ratio(pairs):
    """rate = median(y / x) over positive x.

    A median ratio rather than least squares: these costs span orders of
    magnitude, so a squared-error fit in seconds is decided entirely by the one
    biggest node, and the objective here is a ranking. Three medians also need
    no linear algebra and are explainable out loud -- "this engine does about N
    rows a second of hash aggregate on this box".
    """
    rs = [y / x for x, y in pairs if x > EPS and y is not None and y >= 0]
    return st.median(rs) if rs else None


def fit_rates(samples):
    """Per-operator-class seconds-per-row, plus seconds-per-row written.

    `samples` are TABLE nodes only -- never a candidate View, which is the
    evaluation label. Each contributes its own executed plan's operators.
    """
    per_class = {}
    for s in samples:
        for o in s["ops"]:
            if o.time_s is None:
                continue
            rows = o.rows_act if o.rows_act is not None else o.rows_est
            if not rows or rows <= 0:
                continue
            per_class.setdefault(op_class(o.op), []).append((rows, o.time_s))
    rates = {k: median_ratio(v) for k, v in per_class.items()}
    write = median_ratio([(s["rows"], s["write_s"]) for s in samples
                          if s.get("rows") and s.get("write_s") is not None])
    return {"rates": rates, "write": write}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def priced(view_ops, cal):
    """Price the View's OWN plan: no matching against a consumer at all."""
    rates = cal["rates"]
    if not rates:
        return None
    total = 0.0
    seen = False
    for o in view_ops:
        rows = o.rows_est
        if not rows or rows <= 0:
            continue
        r = rates.get(op_class(o.op))
        if r is None:
            continue
        total += r * rows
        seen = True
    return total if seen else None


def pg_total_cost(view_ops):
    """Postgres already has a cost model; use its number, uncalibrated.

    Ordering is scale-invariant, so no conversion to seconds is needed to rank.
    """
    roots = [o for o in view_ops if o.cost is not None]
    return max((o.cost for o in roots), default=None)


def penalty(rows, width, cal):
    """What materializing costs: writing the rows out, and reading them back."""
    if rows is None or cal.get("write") is None:
        return None
    return rows * cal["write"] * (1.0 if width is None else max(width, 1) / 8.0)
