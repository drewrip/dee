"""
estimate_view_runtime.py
========================
Estimates the standalone runtime of a DuckDB VIEW by locating the subtree
in a parent query's EXPLAIN ANALYZE JSON that corresponds to the view's
EXPLAIN plan, then reading the actual cpu_time accumulated there.

Background
----------
DuckDB's EXPLAIN ANALYZE JSON uses two timing fields per node:

  operator_timing   – wall-clock time spent in THIS operator only (seconds).
                      For leaf scans this equals cpu_time.

  cpu_time          – cumulative time for this operator AND all its descendants.
                      Invariant:  cpu_time(node) == operator_timing(node)
                                  + sum(cpu_time(child) for child in children)

  operator_cardinality – actual rows output by this operator.

So the view's estimated standalone runtime is simply the `cpu_time` of the
root node of the view's subtree inside the query plan.  That single number
already includes every scan, join, aggregation, and projection inside V.

Matching strategy
-----------------
DuckDB inlines views into the calling query before optimisation, so there is
no single "view node" in the plan.  However, when a view (or any large CTE)
is referenced more than once, DuckDB materialises it as a CTE and emits a
dedicated `CTE` operator whose first child is the CTE body (= the view).

The script therefore:
  1. Collects the leaf table names referenced in V's EXPLAIN plan.
  2. Walks Q's EXPLAIN ANALYZE tree looking for subtrees whose leaf tables
     are a superset of V's leaf tables and whose operator structure matches
     V's plan structurally.
  3. Among all candidates, picks the one whose leaf-table set is the
     smallest superset of V's tables (closest match).
  4. Reports that subtree's `cpu_time` as the estimated view runtime.

If a `CTE` operator is found whose body covers exactly V's tables, it is
preferred outright, because DuckDB's CTE body IS the materialised view
computation.

Usage
-----
  python estimate_view_runtime.py \
      --query  path/to/final_business_report_EXPLAIN_ANALYZE.json \
      --view   path/to/product_performance_EXPLAIN.json

Or import and call estimate_view_runtime() directly.
"""

import json
import argparse
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class PlanNode:
    """Unified representation of a single operator node in a DuckDB plan."""
    operator_name: str
    operator_timing: float         # seconds, this operator only (ANALYZE only)
    cpu_time: float                # seconds, cumulative subtree  (ANALYZE only)
    operator_cardinality: int      # actual rows out (ANALYZE) or estimated (EXPLAIN)
    extra_info: dict
    children: list["PlanNode"] = field(default_factory=list)

    @property
    def table(self) -> str:
        """Return the scanned table name for SEQ_SCAN / INDEX_SCAN nodes."""
        if isinstance(self.extra_info, dict):
            return self.extra_info.get("Table", "")
        return ""

    @property
    def is_cte(self) -> bool:
        return self.operator_name == "CTE"

    @property
    def is_cte_scan(self) -> bool:
        return self.operator_name == "CTE_SCAN"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_node_analyze(raw: dict) -> PlanNode:
    """Parse a node from a DuckDB EXPLAIN ANALYZE JSON (has timing fields)."""
    children = [_parse_node_analyze(c) for c in raw.get("children", [])]
    return PlanNode(
        operator_name=raw.get("operator_name", raw.get("operator_type", "UNKNOWN")),
        operator_timing=float(raw.get("operator_timing", 0.0)),
        cpu_time=float(raw.get("cpu_time", 0.0)),
        operator_cardinality=int(raw.get("operator_cardinality", 0)),
        extra_info=raw.get("extra_info", {}),
        children=children,
    )


def _parse_node_explain(raw: dict) -> PlanNode:
    """Parse a node from a plain DuckDB EXPLAIN JSON (no timing fields)."""
    children = [_parse_node_explain(c) for c in raw.get("children", [])]
    return PlanNode(
        operator_name=raw.get("name", raw.get("operator_type", "UNKNOWN")),
        operator_timing=0.0,
        cpu_time=0.0,
        operator_cardinality=0,
        extra_info=raw.get("extra_info", {}),
        children=children,
    )


def load_query_plan(path: str) -> PlanNode:
    """
    Load the EXPLAIN ANALYZE JSON for query Q.

    DuckDB emits a top-level dict with metadata + a 'children' list whose
    first element is the root operator.
    """
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "children" in raw:
        return _parse_node_analyze(raw["children"][0])
    raise ValueError(f"Unexpected EXPLAIN ANALYZE format in {path!r}")


def load_view_plan(path: str) -> PlanNode:
    """
    Load the plain EXPLAIN JSON for view V.

    DuckDB emits a JSON array with one element (the root operator).
    """
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, list) and len(raw) >= 1:
        return _parse_node_explain(raw[0])
    raise ValueError(f"Unexpected EXPLAIN format in {path!r}")


# ---------------------------------------------------------------------------
# Tree utilities
# ---------------------------------------------------------------------------

def leaf_tables(node: PlanNode) -> frozenset:
    """Return the set of base table names reachable from this node."""
    tables: set[str] = set()
    if node.table:
        tables.add(node.table)
    for child in node.children:
        tables |= leaf_tables(child)
    return frozenset(tables)


def all_nodes(node: PlanNode):
    """Yield every node in the subtree (pre-order)."""
    yield node
    for child in node.children:
        yield from all_nodes(child)


def structural_match_score(view_node: PlanNode, query_node: PlanNode) -> float:
    """
    Heuristic similarity score between a view subtree (from plain EXPLAIN)
    and a candidate subtree from the EXPLAIN ANALYZE plan.

    Returns a value in [0, 1]: 1.0 = perfect structural match.
    Matching is based on operator names and leaf-table sets; the plain EXPLAIN
    plan lacks cardinality / timing so we cannot compare those.
    """
    v_ops = [n.operator_name for n in all_nodes(view_node)]
    q_ops = [n.operator_name for n in all_nodes(query_node)]

    # Count how many operator types overlap (as multisets)
    from collections import Counter
    v_cnt = Counter(v_ops)
    q_cnt = Counter(q_ops)
    overlap = sum((v_cnt & q_cnt).values())
    total = max(sum(v_cnt.values()), 1)
    return overlap / total


# ---------------------------------------------------------------------------
# Core matching logic
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    subtree_root: PlanNode
    match_path: list[int]        # child-index path from plan root to this node
    leaf_tables_in_subtree: frozenset
    structural_score: float
    estimated_runtime_s: float
    matched_via_cte: bool        # True if we matched via a CTE body


def find_view_subtree(query_root: PlanNode, view_root: PlanNode) -> MatchResult:
    """
    Locate the subtree in the EXPLAIN ANALYZE plan that best corresponds to
    the view's plan and return timing information for it.

    Strategy (in priority order):
      1. CTE body match: if a CTE node exists whose first child covers exactly
         the same leaf tables as V, use that child's cpu_time directly.
      2. Structural subtree match: find the subtree with the smallest leaf-
         table superset of V's tables and the highest structural similarity.
    """
    v_tables = leaf_tables(view_root)

    # --- Pass 1: look for a CTE whose body matches V's leaf tables exactly ---
    cte_match = _find_cte_match(query_root, v_tables, path=[])
    if cte_match is not None:
        return cte_match

    # --- Pass 2: structural subtree search ---
    return _find_structural_match(query_root, view_root, v_tables, path=[])


def _find_cte_match(
    node: PlanNode,
    v_tables: frozenset,
    path: list[int],
) -> MatchResult | None:
    """
    DFS search for a CTE operator whose first child (the CTE body) covers
    exactly the same tables as V.  Returns the first (deepest) such match.
    """
    if node.is_cte and node.children:
        cte_body = node.children[0]
        body_tables = leaf_tables(cte_body)
        if body_tables == v_tables:
            return MatchResult(
                subtree_root=cte_body,
                match_path=path + [0],
                leaf_tables_in_subtree=body_tables,
                structural_score=1.0,   # table sets are identical
                estimated_runtime_s=cte_body.cpu_time,
                matched_via_cte=True,
            )

    for i, child in enumerate(node.children):
        result = _find_cte_match(child, v_tables, path + [i])
        if result is not None:
            return result
    return None


def _find_structural_match(
    query_root: PlanNode,
    view_root: PlanNode,
    v_tables: frozenset,
    path: list[int],
) -> MatchResult:
    """
    Walk every subtree of the query plan.  Collect all subtrees that are a
    superset of V's leaf tables.  Among those, pick the one with:
      - smallest number of *extra* tables (tightest cover), then
      - highest structural operator-overlap score as a tiebreaker.
    """

    @dataclass
    class Candidate:
        node: PlanNode
        path: list[int]
        tables: frozenset
        score: float

    candidates: list[Candidate] = []

    def _walk(node: PlanNode, p: list[int]):
        t = leaf_tables(node)
        if v_tables.issubset(t):
            score = structural_match_score(view_root, node)
            candidates.append(Candidate(node, list(p), t, score))
        for i, child in enumerate(node.children):
            _walk(child, p + [i])

    _walk(query_root, [])

    if not candidates:
        raise RuntimeError(
            f"Could not find any subtree in the query plan that covers all "
            f"view tables: {sorted(v_tables)}"
        )

    # Sort: fewest extra tables first, then highest structural score
    candidates.sort(key=lambda c: (len(c.tables - v_tables), -c.score))
    best = candidates[0]

    return MatchResult(
        subtree_root=best.node,
        match_path=best.path,
        leaf_tables_in_subtree=best.tables,
        structural_score=best.score,
        estimated_runtime_s=best.node.cpu_time,
        matched_via_cte=False,
    )


# ---------------------------------------------------------------------------
# Timing breakdown
# ---------------------------------------------------------------------------

def build_timing_breakdown(node: PlanNode, depth: int = 0) -> list[dict]:
    """
    Recursively collect per-operator timing from the matched subtree.
    Returns a list of dicts ready for display.
    """
    rows = []
    indent = "  " * depth
    rows.append({
        "indent": indent,
        "operator": node.operator_name,
        "table": node.table or "",
        "operator_timing_s": node.operator_timing,
        "cumulative_cpu_s": node.cpu_time,
        "cardinality": node.operator_cardinality,
    })
    for child in node.children:
        rows.extend(build_timing_breakdown(child, depth + 1))
    return rows


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def estimate_view_runtime(query_plan_path: str, view_plan_path: str) -> dict:
    """
    Estimate the standalone runtime of a DuckDB view.

    Parameters
    ----------
    query_plan_path : str
        Path to the JSON output of EXPLAIN ANALYZE on the query Q.
    view_plan_path : str
        Path to the JSON output of EXPLAIN on view V.

    Returns
    -------
    dict with keys:
      estimated_runtime_s        – estimated wall-clock seconds for V alone
      matched_via_cte            – True if a CTE body was used (exact match)
      structural_score           – operator-overlap similarity [0,1]
      view_leaf_tables           – tables referenced by V
      matched_subtree_tables     – tables in the matched Q subtree
      extra_tables               – tables in Q's subtree not in V
      operator_breakdown         – per-operator timing list for the subtree
    """
    query_root = load_query_plan(query_plan_path)
    view_root  = load_view_plan(view_plan_path)

    v_tables = leaf_tables(view_root)
    result   = find_view_subtree(query_root, view_root)
    breakdown = build_timing_breakdown(result.subtree_root)

    return {
        "estimated_runtime_s":    result.estimated_runtime_s,
        "matched_via_cte":        result.matched_via_cte,
        "structural_score":       result.structural_score,
        "view_leaf_tables":       sorted(v_tables),
        "matched_subtree_tables": sorted(result.leaf_tables_in_subtree),
        "extra_tables":           sorted(result.leaf_tables_in_subtree - v_tables),
        "operator_breakdown":     breakdown,
    }


def print_report(r: dict) -> None:
    """Print a human-readable summary of the estimation result."""
    sep = "─" * 70

    print(sep)
    print("  VIEW RUNTIME ESTIMATION REPORT")
    print(sep)

    print(f"\n  Estimated standalone runtime : {r['estimated_runtime_s']:.6f} s"
          f"  ({r['estimated_runtime_s'] * 1000:.2f} ms)")
    print(f"  Match method                 : "
          f"{'CTE body (exact table match)' if r['matched_via_cte'] else 'Structural subtree search'}")
    print(f"  Structural similarity score  : {r['structural_score']:.3f}  (1.0 = perfect)")

    print(f"\n  View leaf tables  ({len(r['view_leaf_tables'])}):")
    for t in r["view_leaf_tables"]:
        print(f"    • {t}")

    if r["extra_tables"]:
        print(f"\n  ⚠  Extra tables in matched subtree (not in view):")
        for t in r["extra_tables"]:
            print(f"    • {t}")
        print("     These inflate the estimate; the true view cost may be lower.")
    else:
        print("\n  ✓  Matched subtree covers exactly the view's tables.")

    print(f"\n  Operator breakdown (matched subtree):")
    print(f"  {'Operator':<30} {'Table':<40} {'op_timing (s)':>14} {'cpu_time (s)':>13} {'rows_out':>10}")
    print(f"  {'-'*30} {'-'*40} {'-'*14} {'-'*13} {'-'*10}")
    for row in r["operator_breakdown"]:
        label = row["indent"] + row["operator"]
        print(f"  {label:<30} {row['table']:<40} "
              f"{row['operator_timing_s']:>14.6f} "
              f"{row['cumulative_cpu_s']:>13.6f} "
              f"{row['cardinality']:>10,}")

    print(f"\n{sep}")
    print("  INTERPRETATION")
    print(sep)
    print("""
  • estimated_runtime_s is the cpu_time of the root node of the view's
    matched subtree inside the EXPLAIN ANALYZE plan.  cpu_time is the
    DuckDB cumulative wall-clock cost for the entire subtree, so it
    already includes every scan, join, aggregation, and projection in V.

  • This is the cost as observed inside Q, with Q's predicate pushdowns
    applied.  If Q pushed filters into V, the standalone cost of V could
    be higher (V would scan/join more rows without those filters).

  • operator_timing is the self-cost of each individual operator (time
    not attributable to its children).  Summing all operator_timing values
    in the breakdown equals the root's cpu_time exactly, because:
        cpu_time(node) = operator_timing(node) + Σ cpu_time(children)
    """)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimate standalone VIEW runtime from a parent query's EXPLAIN ANALYZE."
    )
    parser.add_argument(
        "--query", required=True,
        help="Path to the EXPLAIN ANALYZE JSON for the query Q."
    )
    parser.add_argument(
        "--view", required=True,
        help="Path to the EXPLAIN JSON for view V."
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit raw JSON result instead of the formatted report."
    )
    args = parser.parse_args()

    result = estimate_view_runtime(args.query, args.view)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print_report(result)
