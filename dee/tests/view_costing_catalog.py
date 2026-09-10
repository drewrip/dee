"""Generate the micro-DAG catalog `view_costing_bench.rs` reads.

Twelve minimal DAGs, each with at least one VIEW that is a branch point -- the
only kind HMP will consider -- chosen to cover the ordinary case and the shapes
where a costing method can be expected to break: chained VIEWs over one leaf
set, siblings that read the same relations, a self-join, a UNION, a VIEW whose
consumers push a predicate inside it, and three orders of magnitude between two
candidates.

    SF=5 OUT=catalog_duckdb.json python3 view_costing_catalog.py
    SF=1 OUT=catalog_pg.json     python3 view_costing_catalog.py

The scale differs per backend so runs land in the same time band on both.
"""
import json, os

_sf = float(os.environ.get("SF", "1"))
N_ORDERS   = int(1_200_000 * _sf)
N_CUST     = int(  200_000 * _sf)
N_PROD     = int(   40_000 * _sf)
N_EVENTS   = int(2_400_000 * _sf)
N_REGION   =         8

DUCK_SETUP = [
 "DROP TABLE IF EXISTS orders", "DROP TABLE IF EXISTS customers",
 "DROP TABLE IF EXISTS products", "DROP TABLE IF EXISTS events",
 "DROP TABLE IF EXISTS regions",
 f"""CREATE TABLE customers AS
     SELECT i AS customer_id,
            'cust_' || i AS cust_name,
            (i * 2654435761) % {N_REGION} AS region_id,
            CASE WHEN i % 11 = 0 THEN 'ent' WHEN i % 3 = 0 THEN 'mid' ELSE 'smb' END AS segment,
            (i % 97) * 13.5 AS credit
     FROM range({N_CUST}) t(i)""",
 f"""CREATE TABLE products AS
     SELECT i AS product_id,
            'prod_' || i AS prod_name,
            i % 40 AS category_id,
            (((i * 7919) % 5000) / 10.0)::DOUBLE AS list_price
     FROM range({N_PROD}) t(i)""",
 f"""CREATE TABLE regions AS
     SELECT i AS region_id, 'region_' || i AS region_name, (i % 3) AS tier
     FROM range({N_REGION}) t(i)""",
 f"""CREATE TABLE orders AS
     SELECT i AS order_id,
            (i * 2654435761) % {N_CUST} AS customer_id,
            (i * 40503) % {N_PROD} AS product_id,
            TIMESTAMP '2023-01-01 00:00:00' + INTERVAL (i % 900) DAY AS order_ts,
            1 + (i % 9) AS qty,
            (((i * 7919) % 100000) / 100.0)::DOUBLE AS amount,
            CASE WHEN i % 5 = 0 THEN 'A' WHEN i % 5 = 1 THEN 'B'
                 WHEN i % 5 = 2 THEN 'C' WHEN i % 5 = 3 THEN 'D' ELSE 'E' END AS status
     FROM range({N_ORDERS}) t(i)""",
 f"""CREATE TABLE events AS
     SELECT i AS event_id,
            (i * 2654435761) % {N_CUST} AS customer_id,
            i % 17 AS event_type,
            TIMESTAMP '2023-01-01 00:00:00' + INTERVAL (i % 900) DAY AS event_ts,
            (((i * 104729) % 1000) / 10.0)::DOUBLE AS weight
     FROM range({N_EVENTS}) t(i)""",
 "ANALYZE",
]

PG_SETUP = [
 "DROP TABLE IF EXISTS orders CASCADE", "DROP TABLE IF EXISTS customers CASCADE",
 "DROP TABLE IF EXISTS products CASCADE", "DROP TABLE IF EXISTS events CASCADE",
 "DROP TABLE IF EXISTS regions CASCADE",
 f"""CREATE TABLE customers AS
     SELECT i AS customer_id,
            'cust_' || i AS cust_name,
            ((i::bigint * 2654435761) % {N_REGION})::int AS region_id,
            CASE WHEN i % 11 = 0 THEN 'ent' WHEN i % 3 = 0 THEN 'mid' ELSE 'smb' END AS segment,
            (i % 97) * 13.5 AS credit
     FROM generate_series(0, {N_CUST}-1) AS t(i)""",
 f"""CREATE TABLE products AS
     SELECT i AS product_id, 'prod_' || i AS prod_name,
            i % 40 AS category_id,
            (((i::bigint * 7919) % 5000) / 10.0)::float8 AS list_price
     FROM generate_series(0, {N_PROD}-1) AS t(i)""",
 f"""CREATE TABLE regions AS
     SELECT i AS region_id, 'region_' || i AS region_name, (i % 3) AS tier
     FROM generate_series(0, {N_REGION}-1) AS t(i)""",
 f"""CREATE TABLE orders AS
     SELECT i AS order_id,
            ((i::bigint * 2654435761) % {N_CUST})::int AS customer_id,
            ((i::bigint * 40503) % {N_PROD})::int AS product_id,
            TIMESTAMP '2023-01-01 00:00:00' + ((i % 900) || ' day')::interval AS order_ts,
            1 + (i % 9) AS qty,
            (((i::bigint * 7919) % 100000) / 100.0)::float8 AS amount,
            CASE WHEN i % 5 = 0 THEN 'A' WHEN i % 5 = 1 THEN 'B'
                 WHEN i % 5 = 2 THEN 'C' WHEN i % 5 = 3 THEN 'D' ELSE 'E' END AS status
     FROM generate_series(0, {N_ORDERS}-1) AS t(i)""",
 f"""CREATE TABLE events AS
     SELECT i AS event_id,
            ((i::bigint * 2654435761) % {N_CUST})::int AS customer_id,
            i % 17 AS event_type,
            TIMESTAMP '2023-01-01 00:00:00' + ((i % 900) || ' day')::interval AS event_ts,
            (((i::bigint * 104729) % 1000) / 10.0)::float8 AS weight
     FROM generate_series(0, {N_EVENTS}-1) AS t(i)""",
 "ANALYZE",
]

def n(id, sql, deps, mat="view"):
    return {"id": id, "query_text": " ".join(sql.split()), "depends_on": deps, "materialize": mat}

dags = []

# --- d01: the ordinary case. One view, two table consumers. -----------------
dags.append({
 "name": "d01_fanout_join",
 "doc": "The ordinary case: one VIEW over a 3-way join, consumed by two tables.",
 "nodes": [
  n("v_enriched", """
     SELECT o.order_id, o.customer_id, o.product_id, o.amount, o.qty, o.status,
            c.segment, c.region_id, p.category_id, p.list_price,
            o.amount - p.list_price * o.qty AS margin
     FROM orders o
     JOIN customers c ON o.customer_id = c.customer_id
     JOIN products  p ON o.product_id  = p.product_id""", []),
  n("t_by_region", """
     SELECT region_id, segment, count(*) AS n, sum(amount) AS revenue, avg(margin) AS avg_margin
     FROM v_enriched GROUP BY region_id, segment""", ["v_enriched"], "table"),
  n("t_by_product", """
     SELECT category_id, status, count(*) AS n, sum(qty) AS units, max(margin) AS best_margin
     FROM v_enriched GROUP BY category_id, status""", ["v_enriched"], "table"),
 ]})

# --- d02: chained views over the same leaf set (the 5.3 case). --------------
dags.append({
 "name": "d02_chain_window_agg",
 "doc": "v_win (window) feeds v_agg (group by); both read exactly the same three "
        "relations, so leaf sets alone cannot separate them.",
 "nodes": [
  n("v_win", """
     SELECT o.order_id, o.customer_id, o.amount, c.segment, c.region_id, p.category_id,
            row_number() OVER (PARTITION BY c.region_id ORDER BY o.amount DESC) AS rk,
            sum(o.amount) OVER (PARTITION BY p.category_id) AS cat_total
     FROM orders o
     JOIN customers c ON o.customer_id = c.customer_id
     JOIN products  p ON o.product_id  = p.product_id""", []),
  n("v_agg", """
     SELECT region_id, category_id, count(*) AS n, sum(amount) AS revenue, max(rk) AS max_rk
     FROM v_win GROUP BY region_id, category_id""", ["v_win"]),
  n("t_top", """SELECT * FROM v_win WHERE rk <= 5""", ["v_win"], "table"),
  n("t_agg_a", """SELECT region_id, sum(revenue) AS revenue FROM v_agg GROUP BY region_id""",
    ["v_agg"], "table"),
  n("t_agg_b", """SELECT category_id, sum(n) AS n FROM v_agg GROUP BY category_id""",
    ["v_agg"], "table"),
 ]})

# --- d03: a heavy view and a trivial one, on disjoint relations. ------------
dags.append({
 "name": "d03_skew_siblings",
 "doc": "One expensive VIEW over events, one trivial VIEW over regions. Both are "
        "branch points; a costing method that cannot tell them apart is useless.",
 "nodes": [
  n("v_heavy", """
     SELECT e.customer_id, e.event_type, count(*) AS n, sum(e.weight) AS w,
            avg(e.weight) AS aw
     FROM events e GROUP BY e.customer_id, e.event_type""", []),
  n("v_tiny", """SELECT region_id, tier, region_name FROM regions WHERE tier >= 0""", []),
  n("t_a", """
     SELECT r.tier, count(*) AS n, sum(h.w) AS w
     FROM v_heavy h JOIN customers c ON h.customer_id = c.customer_id
     JOIN v_tiny r ON c.region_id = r.region_id GROUP BY r.tier""",
    ["v_heavy", "v_tiny"], "table"),
  n("t_b", """
     SELECT r.region_name, h.event_type, sum(h.n) AS n
     FROM v_heavy h JOIN customers c ON h.customer_id = c.customer_id
     JOIN v_tiny r ON c.region_id = r.region_id GROUP BY r.region_name, h.event_type""",
    ["v_heavy", "v_tiny"], "table"),
 ]})

# --- d04: diamond, nested branch points. ------------------------------------
dags.append({
 "name": "d04_diamond",
 "doc": "v_base branches into v_left and v_right, which recombine. Three nested "
        "branch points, so an attribution that double-counts shows up immediately.",
 "nodes": [
  n("v_base", """
     SELECT o.order_id, o.customer_id, o.amount, o.qty, o.status, c.segment, c.region_id
     FROM orders o JOIN customers c ON o.customer_id = c.customer_id""", []),
  n("v_left", """
     SELECT customer_id, segment, sum(amount) AS spend, count(*) AS n
     FROM v_base GROUP BY customer_id, segment""", ["v_base"]),
  n("v_right", """
     SELECT region_id, status, sum(qty) AS units, avg(amount) AS avg_amount
     FROM v_base GROUP BY region_id, status""", ["v_base"]),
  n("t1", """
     SELECT l.segment, sum(l.spend) AS spend, max(r.units) AS units
     FROM v_left l JOIN v_right r ON r.region_id = l.customer_id % 8 GROUP BY l.segment""",
    ["v_left", "v_right"], "table"),
  n("t2", """SELECT segment, count(*) AS n, sum(spend) AS spend FROM v_left GROUP BY segment""",
    ["v_left"], "table"),
  n("t3", """SELECT region_id, sum(units) AS units FROM v_right GROUP BY region_id""",
    ["v_right"], "table"),
 ]})

# --- d05: sibling views with identical leaf sets, no DAG order between them --
dags.append({
 "name": "d05_shared_leafset_siblings",
 "doc": "Two sibling VIEWs read exactly the same two relations and neither depends "
        "on the other, so DAG-order containment cannot disambiguate them.",
 "nodes": [
  n("v_sum", """
     SELECT c.customer_id, c.segment, sum(o.amount) AS spend, count(*) AS n
     FROM orders o JOIN customers c ON o.customer_id = c.customer_id
     GROUP BY c.customer_id, c.segment""", []),
  n("v_rank", """
     SELECT o.order_id, c.region_id, o.amount,
            rank() OVER (PARTITION BY c.region_id ORDER BY o.amount) AS rk
     FROM orders o JOIN customers c ON o.customer_id = c.customer_id""", []),
  n("t_a", """
     SELECT s.segment, count(*) AS n, sum(s.spend) AS spend
     FROM v_sum s GROUP BY s.segment""", ["v_sum"], "table"),
  n("t_b", """
     SELECT r.region_id, count(*) AS n, sum(s.spend) AS spend
     FROM v_rank r JOIN v_sum s ON s.customer_id = r.order_id
     WHERE r.rk <= 50 GROUP BY r.region_id""", ["v_rank", "v_sum"], "table"),
  n("t_c", """SELECT region_id, max(rk) AS max_rk FROM v_rank GROUP BY region_id""",
    ["v_rank"], "table"),
 ]})

# --- d06: leaf set of size one -----------------------------------------------
dags.append({
 "name": "d06_single_relation",
 "doc": "Both VIEWs read a single relation, so every scan of it trivially clears "
        "the coverage floor and leaf sets carry almost no information.",
 "nodes": [
  n("v_ord_agg", """
     SELECT status, count(*) AS n, sum(amount) AS revenue, avg(qty) AS avg_qty
     FROM orders GROUP BY status""", []),
  n("v_ord_big", """
     SELECT order_id, customer_id, product_id, amount, qty, status,
            amount / (qty + 1) AS unit_amount
     FROM orders WHERE amount > 10""", []),
  n("t_a", """
     SELECT b.status, count(*) AS n, sum(b.unit_amount) AS ua, max(a.revenue) AS rev
     FROM v_ord_big b JOIN v_ord_agg a ON a.status = b.status GROUP BY b.status""",
    ["v_ord_big", "v_ord_agg"], "table"),
  n("t_b", """
     SELECT a.status, a.revenue, count(b.order_id) AS n
     FROM v_ord_agg a JOIN v_ord_big b ON a.status = b.status
     WHERE b.qty > 4 GROUP BY a.status, a.revenue""",
    ["v_ord_agg", "v_ord_big"], "table"),
 ]})

# --- d07: self join. Leaf set {orders} but the work is a join of two copies. -
dags.append({
 "name": "d07_self_join",
 "doc": "A self-join: the VIEW's leaf set is one relation but its region is a join "
        "of two scans of it, next to a cheap VIEW over the same relation.",
 "nodes": [
  n("v_selfjoin", """
     SELECT o1.status, o2.status AS status2, count(*) AS n, sum(o1.amount) AS amt
     FROM orders o1 JOIN orders o2 ON o1.product_id = o2.product_id
                                  AND o1.order_id < o2.order_id
     WHERE o1.qty = 9 AND o2.qty = 9
     GROUP BY o1.status, o2.status""", []),
  n("v_cheap", """SELECT status, count(*) AS n FROM orders GROUP BY status""", []),
  n("t_a", """
     SELECT s.status, s.n, c.n AS cn FROM v_selfjoin s JOIN v_cheap c ON c.status = s.status""",
    ["v_selfjoin", "v_cheap"], "table"),
  n("t_b", """
     SELECT s.status2, sum(s.amt) AS amt, max(c.n) AS cn
     FROM v_selfjoin s JOIN v_cheap c ON c.status = s.status2 GROUP BY s.status2""",
    ["v_selfjoin", "v_cheap"], "table"),
 ]})

# --- d08: wide fan-out --------------------------------------------------------
dags.append({
 "name": "d08_wide_fanout",
 "doc": "One VIEW inlined into four different consumers, each of which optimizes it "
        "differently -- so the same VIEW is costed four times from four plans.",
 "nodes": [
  n("v_enriched", """
     SELECT o.order_id, o.customer_id, o.amount, o.qty, o.status, c.segment, c.region_id,
            p.category_id, o.amount * 1.07 AS gross
     FROM orders o JOIN customers c ON o.customer_id = c.customer_id
     JOIN products p ON o.product_id = p.product_id""", []),
  n("t1", "SELECT region_id, sum(gross) AS g FROM v_enriched GROUP BY region_id",
    ["v_enriched"], "table"),
  n("t2", "SELECT segment, count(*) AS n FROM v_enriched GROUP BY segment",
    ["v_enriched"], "table"),
  n("t3", "SELECT category_id, avg(amount) AS a FROM v_enriched WHERE qty > 5 GROUP BY category_id",
    ["v_enriched"], "table"),
  n("t4", "SELECT status, sum(qty) AS q FROM v_enriched GROUP BY status",
    ["v_enriched"], "table"),
 ]})

# --- d09: union of disjoint branches ------------------------------------------
dags.append({
 "name": "d09_union_branches",
 "doc": "A VIEW whose body is a UNION ALL of two disjoint branches: its region is "
        "not a single subtree of any consumer's plan.",
 "nodes": [
  n("v_union", """
     SELECT customer_id, 'order' AS kind, amount AS val FROM orders WHERE qty > 3
     UNION ALL
     SELECT customer_id, 'event' AS kind, weight AS val FROM events WHERE event_type < 9""", []),
  n("v_side", """
     SELECT customer_id, count(*) AS n FROM orders GROUP BY customer_id""", []),
  n("t_a", """
     SELECT u.kind, count(*) AS n, sum(u.val) AS v
     FROM v_union u JOIN v_side s ON s.customer_id = u.customer_id GROUP BY u.kind""",
    ["v_union", "v_side"], "table"),
  n("t_b", """
     SELECT s.n, count(*) AS c FROM v_side s JOIN v_union u ON u.customer_id = s.customer_id
     WHERE u.val > 50 GROUP BY s.n""", ["v_side", "v_union"], "table"),
 ]})

# --- d10: three orders of magnitude between two candidates --------------------
dags.append({
 "name": "d10_micro_vs_macro",
 "doc": "A sub-millisecond VIEW next to a multi-second one: does the method keep "
        "them apart, or does noise swamp the small one?",
 "nodes": [
  n("v_micro", "SELECT tier, count(*) AS n FROM regions GROUP BY tier", []),
  n("v_macro", """
     SELECT e.customer_id, count(*) AS n, sum(e.weight) AS w,
            max(e.event_ts) AS last_ts, avg(e.weight * e.event_type) AS ax
     FROM events e JOIN customers c ON e.customer_id = c.customer_id
     WHERE c.segment <> 'zzz' GROUP BY e.customer_id""", []),
  n("t_a", """
     SELECT m.tier, count(*) AS n, sum(v.w) AS w
     FROM v_macro v JOIN customers c ON c.customer_id = v.customer_id
     JOIN v_micro m ON m.tier = c.region_id % 3 GROUP BY m.tier""",
    ["v_macro", "v_micro"], "table"),
  n("t_b", """
     SELECT m.n AS micro_n, count(*) AS n FROM v_micro m CROSS JOIN v_macro v
     WHERE v.n > 20 GROUP BY m.n""", ["v_micro", "v_macro"], "table"),
 ]})

# --- d11: operator-signature collision ---------------------------------------
dags.append({
 "name": "d11_signature_collision",
 "doc": "Two VIEWs whose scans the planner estimates identically, so "
        "(operator name, estimated cardinality) is the same key for both.",
 "nodes": [
  n("v_cheap_slice", """
     SELECT order_id, customer_id, amount FROM orders WHERE status = 'A'""", []),
  n("v_costly_slice", """
     SELECT o.order_id, o.customer_id, o.amount, e.w
     FROM orders o JOIN (SELECT customer_id, sum(weight) AS w FROM events
                         GROUP BY customer_id) e ON e.customer_id = o.customer_id
     WHERE o.status = 'B'""", []),
  n("t_a", """
     SELECT count(*) AS n, sum(c.amount) AS a, sum(k.w) AS w
     FROM v_cheap_slice c JOIN v_costly_slice k ON k.customer_id = c.customer_id""",
    ["v_cheap_slice", "v_costly_slice"], "table"),
  n("t_b", """
     SELECT c.customer_id, count(*) AS n
     FROM v_cheap_slice c JOIN v_costly_slice k ON k.order_id = c.order_id
     GROUP BY c.customer_id""", ["v_cheap_slice", "v_costly_slice"], "table"),
 ]})

# --- d12: consumers push a selective predicate into the view ------------------
dags.append({
 "name": "d12_pushed_predicate",
 "doc": "Every consumer filters the VIEW selectively, so the optimizer pushes the "
        "predicate inside it and the region in the plan is far cheaper than "
        "building the VIEW standalone would be.",
 "nodes": [
  n("v_wide", """
     SELECT o.order_id, o.customer_id, o.product_id, o.amount, o.qty, o.status,
            c.segment, c.region_id, p.category_id, p.list_price
     FROM orders o JOIN customers c ON o.customer_id = c.customer_id
     JOIN products p ON o.product_id = p.product_id""", []),
  n("t_a", """SELECT region_id, sum(amount) AS a FROM v_wide WHERE status = 'A' AND qty = 9
              GROUP BY region_id""", ["v_wide"], "table"),
  n("t_b", """SELECT category_id, count(*) AS n FROM v_wide WHERE status = 'B' AND qty = 1
              GROUP BY category_id""", ["v_wide"], "table"),
 ]})

catalog = {
 "sources": ["orders", "customers", "products", "events", "regions"],
 "setup": {"duckdb": DUCK_SETUP, "postgres": PG_SETUP},
 "dags": dags,
}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get("OUT", "catalog.json"))
json.dump(catalog, open(out, "w"), indent=1)
print(out, len(dags), "dags")
