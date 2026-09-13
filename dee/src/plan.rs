//! A backend-neutral query plan.
//!
//! The HMP pass ranks materialization candidates by tracing the cost of
//! individual operators back to the Views that produce them. That needs
//! per-operator cost and cardinality, which every engine reports in its own
//! shape: DuckDB in its profiling JSON, Postgres in `EXPLAIN (FORMAT JSON)`.
//! [`PlanNode`] is the shape the optimizer works in, and each connector
//! converts its own format into it via [`Connector::parse_plan`].
//!
//! # What the timings mean
//!
//! DuckDB's `operator_timing` is **CPU time**; Postgres's `Actual Total Time`
//! is **wall time**, inclusive of children and averaged over loops. Both are
//! normalized here to *exclusive* seconds, but they remain different physical
//! quantities. HMP's ranking is therefore a CPU-time ranking on DuckDB and a
//! wall-time ranking on Postgres. That is a real difference in what the
//! optimizer optimizes for, so it is recorded alongside results as
//! `runs.plan_time_basis` rather than being quietly averaged together.

use serde::{Deserialize, Serialize};

/// What a backend's per-operator plan timings physically measure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimeBasis {
    /// CPU time summed across threads (DuckDB).
    CpuTime,
    /// Wall-clock time (Postgres).
    WallTime,
}

impl TimeBasis {
    pub fn as_str(&self) -> &'static str {
        match self {
            TimeBasis::CpuTime => "cpu_time",
            TimeBasis::WallTime => "wall_time",
        }
    }
}

/// One operator in a query plan, normalized across backends.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanNode {
    /// Operator name, e.g. `HASH_JOIN` or `Seq Scan`.
    pub operator: String,
    /// Time attributable to this operator alone, with children subtracted.
    /// `None` when the plan was not executed (a plain EXPLAIN).
    pub exclusive_time_s: Option<f64>,
    /// Rows this operator actually emitted, when the plan was executed.
    pub cardinality: Option<u64>,
    /// Rows the planner estimated this operator would emit.
    pub estimated_cardinality: Option<f64>,
    /// Bytes one output tuple of this operator occupies.
    ///
    /// Postgres reports it directly as `Plan Width`; DuckDB reports the whole
    /// `result_set_size` and this is that divided by the cardinality. Both are
    /// the operator's *output* schema, which is what makes
    /// `cardinality * row_width_bytes` the bytes the operator produced --- the
    /// denominator the learned cost model divides a timing by.
    #[serde(default)]
    pub row_width_bytes: Option<f64>,
    /// The aggregate functions this operator computes, lowercased, sorted and
    /// deduplicated. Empty for everything that is not an aggregate.
    ///
    /// A `HASH_GROUP_BY` computing `count(*)` and one computing
    /// `string_agg(...)` cost wildly different amounts per output byte, so the
    /// learned model keys them apart --- see [`PlanNode::cost_key`].
    #[serde(default)]
    pub aggregates: Vec<String>,
    /// The base relation this operator scans, normalized by
    /// [`normalize_relation`]. `None` for everything that is not a real scan
    /// -- including the pseudo-scans that read an intermediate rather than a
    /// relation (see [`is_pseudo_scan`]).
    #[serde(default)]
    pub relation: Option<String>,
    /// The name of the CTE whose *definition* this subtree is, when the engine
    /// reported one.
    ///
    /// Set only on the root of a CTE body, never on the operators inside it and
    /// never on the scans that read the CTE back. Both backends say which part
    /// of a plan computes a CTE, and both say it differently --- DuckDB hangs
    /// the body off a `CTE` operator carrying `CTE Name`, Postgres marks the
    /// body's own root with `Subplan Name: "CTE <n>"` --- so each parser
    /// normalizes its own spelling into this one field and
    /// [`PlanNode::find_subplan`] can then be written once.
    ///
    /// This is what lets a caller ask an EXPLAIN "which of these operators are
    /// the CTE I put there", which is how duplicate-computation attribution
    /// isolates an inlined View inside its consumer's plan. See
    /// [`crate::opt::dup`].
    #[serde(default)]
    pub subplan: Option<String>,
    /// Rows this operator read, before any filter it applied.
    ///
    /// The learned cost model prices an operator by what it *consumed*, and
    /// for a leaf scan the plan's output cardinality is the wrong number: a
    /// scan that reads 3.2M rows and emits a thousand did the work of the
    /// 3.2M. Only a scan reports this; everything else derives its input from
    /// its children. `None` where the backend does not say.
    #[serde(default)]
    pub rows_scanned: Option<f64>,
    /// Wall clock for the whole statement, set on the root node only.
    ///
    /// Postgres reports it as `Execution Time`, a sibling of `Plan`, and the
    /// gap between it and the root's inclusive time is what writing the result
    /// cost --- Postgres has no write operator of its own to time. DuckDB does
    /// (`CREATE_TABLE_AS`), so it leaves this `None`.
    #[serde(default)]
    pub total_execution_time_s: Option<f64>,
    pub children: Vec<PlanNode>,
}

impl PlanNode {
    /// The identity used to match the same operator across different plans.
    ///
    /// Estimated rather than actual cardinality, because the same logical
    /// operator must key identically whether it appears in an executed plan
    /// or an un-executed one.
    pub fn signature(&self) -> OpKey {
        OpKey {
            name: self.operator.clone(),
            cardinality: self
                .estimated_cardinality
                .map(|c| c.to_string())
                .unwrap_or_else(|| "0".to_string()),
        }
    }

    /// The operator type the learned cost model keys a `SecondsPerByte`
    /// constant by.
    ///
    /// The bare operator name, except for aggregates, which carry the
    /// functions they compute: `HASH_GROUP_BY[count_star,sum]`. The time an
    /// aggregate spends per output byte is a property of the aggregate
    /// function far more than of the grouping strategy, so folding
    /// `count(*)` and `string_agg` into one constant learns the average of two
    /// unrelated numbers.
    pub fn cost_key(&self) -> String {
        if self.aggregates.is_empty() {
            return self.operator.to_ascii_uppercase();
        }
        format!(
            "{}[{}]",
            self.operator.to_ascii_uppercase(),
            self.aggregates.join(",")
        )
    }

    /// Bytes this operator actually emitted: its measured row count times the
    /// width of one of its output tuples.
    ///
    /// `None` unless the plan was executed *and* the backend reported a width,
    /// because a learned constant may only be fitted to something measured.
    pub fn output_bytes(&self) -> Option<f64> {
        let rows = self.cardinality? as f64;
        let width = self.row_width_bytes?;
        Some(rows * width)
    }

    /// Bytes this operator consumed.
    ///
    /// Its own scan count where it has one, otherwise the sum of what its
    /// children emitted. A leaf with neither falls back to its own output: a
    /// scan that says nothing about what it read is at least known to have
    /// produced what it produced, and pricing it at nothing would make the
    /// operators we know least about look cheapest.
    ///
    /// This is the denominator the learned cost model fits against. Output
    /// bytes were the wrong one --- an aggregate over 3.2M rows emitting twenty
    /// of them is not cheap, and dividing its seconds by those twenty rows'
    /// bytes produced a constant orders of magnitude off anything it could
    /// then be applied to.
    pub fn input_bytes(&self) -> Option<f64> {
        if let (Some(rows), Some(width)) = (self.rows_scanned, self.row_width_bytes)
            && rows > 0.0
        {
            return Some(rows * width);
        }
        let from_children: f64 = self.children.iter().filter_map(Self::output_bytes).sum();
        if from_children > 0.0 {
            return Some(from_children);
        }
        self.output_bytes()
    }

    /// The rows this operator is expected to emit: the measured count where the
    /// plan was executed, the planner's estimate where it was not.
    pub fn rows(&self) -> Option<f64> {
        self.cardinality
            .map(|c| c as f64)
            .or(self.estimated_cardinality)
    }

    /// Total exclusive time per unique operator, plus how often each appears.
    pub fn collect_operator_stats(
        &self,
        timing: &mut std::collections::HashMap<OpKey, f64>,
        occurrences: &mut std::collections::HashMap<OpKey, usize>,
    ) {
        let sig = self.signature();
        if let Some(t) = self.exclusive_time_s {
            *timing.entry(sig.clone()).or_insert(0.0) += t;
        }
        *occurrences.entry(sig).or_insert(0) += 1;
        for child in &self.children {
            child.collect_operator_stats(timing, occurrences);
        }
    }

    /// Every `(operator, cost)` pair, one entry per occurrence.
    ///
    /// Unlike [`collect_operator_stats`](Self::collect_operator_stats) this
    /// does not de-duplicate, because totalling the real cost of duplicated
    /// downstream work depends on counting every occurrence.
    pub fn collect_operators(&self, out: &mut Vec<(OpKey, f64)>) {
        if let Some(t) = self.exclusive_time_s {
            out.push((self.signature(), t));
        }
        for child in &self.children {
            child.collect_operators(out);
        }
    }

    /// Whether this subtree contains an operator matching `key`.
    pub fn contains(&self, key: &OpKey) -> bool {
        &self.signature() == key || self.children.iter().any(|c| c.contains(key))
    }

    /// See [`is_aggregate_boundary`].
    pub fn is_aggregate_boundary(&self) -> bool {
        is_aggregate_boundary(&self.operator)
    }

    /// The subtree that computes the CTE named `name`, searched for anywhere
    /// in this plan.
    ///
    /// Case-insensitive: the name goes into the SQL as written and comes back
    /// out of the plan folded however the engine folds identifiers.
    pub fn find_subplan(&self, name: &str) -> Option<&PlanNode> {
        if self
            .subplan
            .as_deref()
            .is_some_and(|s| s.eq_ignore_ascii_case(name))
        {
            return Some(self);
        }
        self.children.iter().find_map(|c| c.find_subplan(name))
    }
}

/// The subtree of `roots` that computes the CTE named `name`.
pub fn find_subplan<'a>(roots: &'a [PlanNode], name: &str) -> Option<&'a PlanNode> {
    roots.iter().find_map(|r| r.find_subplan(name))
}

/// Strip a plan's catalog/schema qualification and quoting down to the bare
/// relation name, lowercased.
///
/// The plan prints `warehouse.main.shipments`; the DAG knows `shipments`. Leaf
/// sets are compared by size, so a spelling mismatch does not fail loudly -- it
/// produces an empty attribution and a candidate list ordered by nothing at
/// all. Both sides normalize through here.
pub fn normalize_relation(name: &str) -> String {
    name.rsplit('.')
        .next()
        .unwrap_or(name)
        .trim_matches('"')
        .to_lowercase()
}

/// Operators that read an *intermediate* -- a CTE, a subquery, a function
/// result -- rather than a stored relation.
///
/// Treating one as its own leaf means the CTE's consumer no longer looks like
/// it reads the underlying tables, every leaf-set match above it fails, and the
/// whole chain is attributed one level too low. This is not hypothetical: a CTE
/// is exactly how an engine represents the duplication being measured.
pub fn is_pseudo_scan(operator: &str) -> bool {
    matches!(
        operator.to_ascii_uppercase().as_str(),
        // DuckDB
        "CTE_SCAN"
            | "DELIM_SCAN"
            | "CHUNK_SCAN"
            | "COLUMN_DATA_SCAN"
            | "RECURSIVE_CTE_SCAN"
            // Postgres
            | "CTE SCAN"
            | "SUBQUERY SCAN"
            | "WORKTABLE SCAN"
            | "FUNCTION SCAN"
            | "VALUES SCAN"
            | "NAMED TUPLESTORE SCAN"
            | "RESULT"
    )
}

/// The write path a plan with no named write operator is fitted under.
///
/// Postgres has no write operator to name --- `EXPLAIN ANALYZE` of a
/// `CREATE TABLE AS` shows the `SELECT` and nothing else --- and an engine that
/// persists every relation the same way has only one path anyway. One name
/// keeps those engines in the same keyed structure as the ones with several.
pub const SINGLE_WRITE_PATH: &str = "WRITE";

/// The write path a plan was written through, read off an executed plan.
///
/// The name of the write operator the engine actually used, which is what the
/// constant for that path is fitted under. [`SINGLE_WRITE_PATH`] for an engine
/// that names no write operator.
///
/// This is the *learning* side. Pricing a View that has never been written has
/// no write operator to read, and asks the engine instead ---
/// [`crate::connectors::Connector::write_path_for`].
pub fn observed_write_path(roots: &[PlanNode]) -> String {
    fn find(node: &PlanNode) -> Option<String> {
        if is_write_operator(&node.operator) {
            return Some(node.operator.to_ascii_uppercase());
        }
        node.children.iter().find_map(find)
    }
    roots
        .iter()
        .find_map(find)
        .unwrap_or_else(|| SINGLE_WRITE_PATH.to_string())
}

/// Operators that write a relation rather than compute one.
///
/// Their cost is proportional to what they *wrote*, and the rows they emit are
/// a one-row count of it: DuckDB gives `CREATE_TABLE_AS` an
/// `operator_cardinality` of 1 and a `result_set_size` of 8 bytes however large
/// the table. Seconds divided by those 8 bytes is not a seconds-per-byte
/// constant for anything --- measured on p03 it came out at 1.6e-2, seven orders
/// of magnitude above a hash join's 5.6e-10, and dragged the whole model's mean
/// with it. So a write is kept out of the per-operator constants, and no plan
/// they price contains one: a VIEW's plan is a `SELECT`.
///
/// What is wrong there is the operator's *output size*, not its *timing*.
/// DuckDB reports `operator_timing` on the write like any other operator, and
/// dividing that by the payload it wrote --- rows written times the width of the
/// operator below --- is a real constant. That is what
/// [`crate::opt::learned::LearnedCostModel::observe_write`] fits, and it is why
/// this predicate exists in two places: to exclude a write from the compute
/// constants, and to find it for the write constant.
///
/// The same fact `rows_written_from_plan` is built on, applied to cost.
pub fn is_write_operator(operator: &str) -> bool {
    matches!(
        operator.to_ascii_uppercase().as_str(),
        // DuckDB
        "CREATE_TABLE_AS"
            | "BATCH_CREATE_TABLE_AS"
            | "INSERT"
            | "BATCH_INSERT"
            | "UPDATE"
            | "DELETE"
            | "COPY_TO_FILE"
            | "BATCH_COPY_TO_FILE"
            // Postgres
            | "MODIFYTABLE"
            | "INSERT ON CONFLICT"
    )
}

/// Whether an operator collapses cardinality, and so ends the region a view
/// with a top-level `GROUP BY` occupies.
///
/// `WINDOW` and `ORDER_BY` are deliberately absent. They were in this set at
/// first and it was a bug: a windowed view emits one row per input row, so
/// treating it as an aggregate made the matcher skip past its very expensive
/// region.
pub fn is_aggregate_boundary(operator: &str) -> bool {
    matches!(
        operator.to_ascii_uppercase().as_str(),
        // DuckDB
        "HASH_GROUP_BY" | "PERFECT_HASH_GROUP_BY" | "UNGROUPED_AGGREGATE" | "GROUP_BY" | "DISTINCT"
        // Postgres
            | "AGGREGATE"
            | "HASHAGGREGATE"
            | "GROUPAGGREGATE"
            | "MIXEDAGGREGATE"
            | "GROUP"
            | "UNIQUE"
            | "SETOP"
    )
}

/// The aggregate functions named in `exprs`, lowercased, sorted, deduplicated.
///
/// Both backends describe an aggregate operator by the expressions it
/// computes --- DuckDB in `extra_info["Aggregates"]` (`"sum(#1)"`), Postgres in
/// the aggregate node's `Output` (`"count(*)"`) --- so the function is the head
/// of a call inside a string, and is read back out the same way for both.
///
/// `group_keys` are dropped: Postgres lists the grouping columns in `Output`
/// alongside the aggregates, and a group key that happens to be an expression
/// (`date_trunc('day', ts)`) would otherwise be mistaken for an aggregate and
/// split one operator type into several.
pub fn aggregate_functions(exprs: &[String], group_keys: &[String]) -> Vec<String> {
    let mut out: Vec<String> = exprs
        .iter()
        .filter(|e| !group_keys.iter().any(|g| g == *e))
        .filter_map(|e| function_head(e))
        .collect();
    out.sort();
    out.dedup();
    out
}

/// The name of the first function call in `expr`, lowercased, or `None` if it
/// contains none.
///
/// A deliberately syntactic reading rather than a parse: the strings come from
/// a plan's own rendering of an expression and are not SQL the dialects agree
/// on, so anything that does not look like `name(` is simply not a call.
fn function_head(expr: &str) -> Option<String> {
    let mut start: Option<usize> = None;
    for (i, c) in expr.char_indices() {
        if c.is_alphanumeric() || c == '_' {
            if start.is_none() {
                start = Some(i);
            }
        } else if c == '(' {
            if let Some(s) = start {
                let name = &expr[s..i];
                // `#1(` is not a call, and neither is `2(`.
                if !name.starts_with(|c: char| c.is_ascii_digit()) {
                    return Some(name.to_lowercase());
                }
            }
            start = None;
        } else {
            start = None;
        }
    }
    None
}

/// A plan operator's identity: its name and its estimated output size.
#[derive(Hash, Eq, PartialEq, Clone, Debug)]
pub struct OpKey {
    pub name: String,
    pub cardinality: String,
}

// ---------------------------------------------------------------------------
// DuckDB
// ---------------------------------------------------------------------------

#[derive(Deserialize, Debug)]
struct DuckDBPlan {
    operator_name: Option<String>,
    #[serde(alias = "name")]
    name: Option<String>,
    #[serde(default)]
    operator_timing: Option<f64>,
    #[serde(default)]
    operator_cardinality: Option<u64>,
    /// Bytes this operator emitted in total, reported only by profiling
    /// output. A plain `EXPLAIN (FORMAT JSON)` carries no such field, which is
    /// why the learned model has to fall back to a learned width there.
    #[serde(default)]
    result_set_size: Option<f64>,
    /// Rows this operator read before filtering, reported by DuckDB profiling
    /// on scans. Absent from `EXPLAIN (FORMAT JSON)` and from older DuckDB, so
    /// `extra_info` is checked as well.
    #[serde(default)]
    operator_rows_scanned: Option<f64>,
    #[serde(default)]
    extra_info: std::collections::HashMap<String, serde_json::Value>,
    #[serde(default)]
    children: Vec<DuckDBPlan>,
}

impl DuckDBPlan {
    /// Convert to plan nodes, splicing unnamed wrappers out of the tree.
    ///
    /// DuckDB's profiling output is rooted at a query-level object that has no
    /// `operator_name` of its own -- the real plan hangs off its `children`.
    /// An unnamed node therefore contributes its children in its own place
    /// rather than nothing, otherwise the entire plan would be discarded and
    /// the optimizer would see no operators at all.
    fn into_plan_nodes(self) -> Vec<PlanNode> {
        let name = self.operator_name.clone().or_else(|| self.name.clone());
        let children: Vec<PlanNode> = self
            .children
            .into_iter()
            .flat_map(DuckDBPlan::into_plan_nodes)
            .collect();
        let Some(operator) = name else {
            return children;
        };
        // DuckDB puts a materialized CTE's body under a `CTE` operator that
        // names it, with the body first and the query that reads it second.
        // Tagging the body's own root -- rather than the `CTE` node, which also
        // covers the consumer -- is what makes the tag mean "these operators
        // are the CTE and nothing else".
        let mut children = children;
        if operator.eq_ignore_ascii_case("CTE")
            && let Some(cte_name) = self
                .extra_info
                .get("CTE Name")
                .and_then(|v| v.as_str())
                .filter(|n| !n.is_empty())
            && let Some(body) = children.first_mut()
        {
            body.subplan = Some(cte_name.to_string());
        }
        let estimated = self
            .extra_info
            .get("Estimated Cardinality")
            .and_then(json_to_f64);
        // A scan is identified by the presence of a `Table` key rather than by
        // an operator-name allowlist: the operator is spelled `SEQ_SCAN` or
        // `TABLE_SCAN` depending on the DuckDB version, and `READ_PARQUET` /
        // `READ_CSV` for external data.
        let relation = if is_pseudo_scan(&operator) {
            None
        } else {
            self.extra_info
                .get("Table")
                .and_then(|v| v.as_str())
                .map(normalize_relation)
        };
        // DuckDB reports the operator's whole output size rather than a per-row
        // width, so the width is that divided by the rows it came from. Guarded
        // on a nonzero cardinality: an operator that emitted nothing has no
        // observable tuple width, and 0/0 would record one of zero bytes.
        let row_width_bytes = match (self.result_set_size, self.operator_cardinality) {
            (Some(bytes), Some(rows)) if rows > 0 => Some(bytes / rows as f64),
            _ => None,
        };
        let aggregates = self
            .extra_info
            .get("Aggregates")
            .and_then(|v| v.as_array())
            .map(|items| {
                let exprs: Vec<String> = items
                    .iter()
                    .filter_map(|i| i.as_str().map(str::to_string))
                    .collect();
                aggregate_functions(&exprs, &[])
            })
            .unwrap_or_default();
        let rows_scanned = self
            .operator_rows_scanned
            .or_else(|| {
                self.extra_info
                    .get("Rows Scanned")
                    .or_else(|| self.extra_info.get("rows_scanned"))
                    .and_then(json_to_f64)
            })
            .filter(|r| *r > 0.0);
        vec![PlanNode {
            operator,
            // DuckDB's operator_timing is already this operator's own time.
            exclusive_time_s: self.operator_timing,
            cardinality: self.operator_cardinality,
            estimated_cardinality: estimated,
            row_width_bytes,
            aggregates,
            relation,
            subplan: None,
            rows_scanned,
            // DuckDB times its write operator directly; nothing to derive.
            total_execution_time_s: None,
            children,
        }]
    }
}

fn json_to_f64(v: &serde_json::Value) -> Option<f64> {
    v.as_f64()
        .or_else(|| v.as_i64().map(|i| i as f64))
        .or_else(|| v.as_str().and_then(|s| s.parse::<f64>().ok()))
}

/// Parse DuckDB profiling or `EXPLAIN (FORMAT JSON)` output.
///
/// Handles both shapes DuckDB emits: profiling output is a single object,
/// while `EXPLAIN (FORMAT JSON)` is an array of roots.
pub fn parse_duckdb_plan(json: &str) -> Option<Vec<PlanNode>> {
    if let Ok(roots) = serde_json::from_str::<Vec<DuckDBPlan>>(json) {
        return Some(
            roots
                .into_iter()
                .flat_map(DuckDBPlan::into_plan_nodes)
                .collect(),
        );
    }
    let root = serde_json::from_str::<DuckDBPlan>(json).ok()?;
    Some(root.into_plan_nodes())
}

// ---------------------------------------------------------------------------
// Postgres
// ---------------------------------------------------------------------------

#[derive(Deserialize, Debug)]
struct PgPlanWrapper {
    #[serde(rename = "Plan")]
    plan: PgPlan,
    /// Wall clock for the whole statement, a sibling of `Plan` under
    /// `EXPLAIN ANALYZE`. What it has beyond the root node's inclusive time is
    /// the part of the run no operator accounted for --- on a `CREATE TABLE AS`
    /// that is the write.
    #[serde(rename = "Execution Time", default)]
    execution_time_ms: Option<f64>,
}

#[derive(Deserialize, Debug)]
struct PgPlan {
    #[serde(rename = "Node Type")]
    node_type: String,
    #[serde(rename = "Actual Total Time", default)]
    actual_total_time: Option<f64>,
    #[serde(rename = "Actual Rows", default)]
    actual_rows: Option<f64>,
    #[serde(rename = "Actual Loops", default)]
    actual_loops: Option<f64>,
    #[serde(rename = "Plan Rows", default)]
    plan_rows: Option<f64>,
    /// The planner's estimate of one output row's width in bytes. Present with
    /// or without ANALYZE, which is what lets a View's un-executed plan be
    /// costed on Postgres without a learned width standing in.
    #[serde(rename = "Plan Width", default)]
    plan_width: Option<f64>,
    /// The expressions this node emits, present under `EXPLAIN (VERBOSE)`.
    /// Read only on aggregate nodes, to name the aggregate functions.
    #[serde(rename = "Output", default)]
    output: Vec<String>,
    #[serde(rename = "Group Key", default)]
    group_key: Vec<String>,
    /// Emitted by every real scan node -- `Seq Scan`, `Index Scan`, `Index Only
    /// Scan`, `Bitmap Heap Scan` -- and already unqualified. The pseudo-scans
    /// do not carry it at all, but they are filtered anyway so the rule is one
    /// rule on both backends.
    #[serde(rename = "Relation Name", default)]
    relation_name: Option<String>,
    /// `"CTE <name>"` on the root of a CTE's body, `"InitPlan N"` /
    /// `"SubPlan N"` elsewhere. Only the CTE spelling is read.
    #[serde(rename = "Subplan Name", default)]
    subplan_name: Option<String>,
    /// Rows a scan read and then discarded. Added back to `Actual Rows` it
    /// gives what the scan actually consumed --- see [`PlanNode::rows_scanned`].
    #[serde(rename = "Rows Removed by Filter", default)]
    rows_removed_by_filter: Option<f64>,
    #[serde(rename = "Plans", default)]
    plans: Vec<PgPlan>,
}

impl PgPlan {
    /// Whether this node fans its subtree out across parallel workers.
    fn is_gather(&self) -> bool {
        matches!(self.node_type.as_str(), "Gather" | "Gather Merge")
    }

    /// How many processes executed this node's children.
    ///
    /// Below a Gather, Postgres runs the subplan once per participating
    /// process and reports each node's `Actual Loops` as that count. The
    /// Gather's immediate child is the root of exactly one such subplan, so
    /// its loop count *is* the number of processes -- which means nothing has
    /// to be assumed about whether the leader joined in.
    fn child_procs(&self, procs: f64) -> f64 {
        if !self.is_gather() {
            return procs;
        }
        self.plans
            .first()
            .and_then(|c| c.actual_loops)
            .filter(|loops| *loops > 0.0)
            .unwrap_or(procs)
    }

    /// Inclusive **wall** time for this node across its loops, in seconds.
    ///
    /// `Actual Total Time` is an average per loop. Multiplying by `Actual
    /// Loops` is right when the loops are sequential re-executions -- the
    /// inner side of a nested loop -- and wrong when they are parallel
    /// workers, which ran at the same time. Left uncorrected it turns a
    /// parallel subtree's wall time into a CPU-like sum while the Gather above
    /// it still reports plain wall, so the subtraction in `into_plan_node`
    /// goes negative and clamps the Gather to zero -- silently attributing
    /// none of the run to the operator that gathered it. `procs` divides that
    /// parallel dimension back out, leaving genuine nested-loop repetition
    /// multiplied as before.
    fn inclusive_time_s(&self, procs: f64) -> Option<f64> {
        let per_loop_ms = self.actual_total_time?;
        Some(per_loop_ms * self.actual_loops.unwrap_or(1.0) / procs / 1000.0)
    }

    fn into_plan_node(self, procs: f64) -> PlanNode {
        // `Actual Total Time` includes every child, so a node's own cost is
        // what remains after subtracting them. Without this, parents would be
        // charged for their children's work and the cost of a shared subplan
        // would be counted many times over.
        let inclusive = self.inclusive_time_s(procs);
        let child_procs = self.child_procs(procs);
        let children_total: f64 = self
            .plans
            .iter()
            .filter_map(|c| c.inclusive_time_s(child_procs))
            .sum();
        let exclusive = inclusive.map(|t| (t - children_total).max(0.0));

        let loops = self.actual_loops.unwrap_or(1.0);
        let relation = if is_pseudo_scan(&self.node_type) {
            None
        } else {
            self.relation_name.as_deref().map(normalize_relation)
        };
        // Only on an aggregate node: elsewhere `Output` is full of ordinary
        // function calls that say nothing about what the operator costs.
        let aggregates = if is_aggregate_boundary(&self.node_type) {
            aggregate_functions(&self.output, &self.group_key)
        } else {
            Vec::new()
        };
        // Postgres names a CTE's body on the body itself, as `CTE <name>`.
        // The other `Subplan Name` spellings (`InitPlan 1`, `SubPlan 2`) are
        // not CTEs and must not answer to one.
        let subplan = self
            .subplan_name
            .as_deref()
            .and_then(|s| s.strip_prefix("CTE "))
            .map(|n| n.trim().trim_matches('"').to_string())
            .filter(|n| !n.is_empty());
        // What a scan consumed is what it kept plus what it threw away. Only
        // recorded where the node discarded something: a node with no filter
        // has no separate input to report, and its children speak for it.
        let rows_scanned = self.rows_removed_by_filter.filter(|r| *r > 0.0).map(|removed| {
            (self.actual_rows.unwrap_or(0.0) + removed) * loops
        });
        PlanNode {
            operator: self.node_type,
            exclusive_time_s: exclusive,
            cardinality: self.actual_rows.map(|r| (r * loops).round() as u64),
            estimated_cardinality: self.plan_rows,
            row_width_bytes: self.plan_width,
            aggregates,
            relation,
            subplan,
            rows_scanned,
            // Filled in by `parse_postgres_plan` on the root, where the
            // statement-level timing lives.
            total_execution_time_s: None,
            children: self
                .plans
                .into_iter()
                .map(|c| c.into_plan_node(child_procs))
                .collect(),
        }
    }
}

/// Parse Postgres `EXPLAIN (FORMAT JSON)` output, with or without ANALYZE.
pub fn parse_postgres_plan(json: &str) -> Option<Vec<PlanNode>> {
    let wrappers: Vec<PgPlanWrapper> = serde_json::from_str(json).ok()?;
    Some(
        wrappers
            .into_iter()
            .map(|w| {
                let total = w.execution_time_ms;
                let mut root = w.plan.into_plan_node(1.0);
                // Root only. It is a property of the statement, not of an
                // operator, and a reader that found it deeper would be
                // reading the same number twice.
                root.total_execution_time_s = total.map(|ms| ms / 1000.0);
                root
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bare(operator: &str, rows: u64, width: f64, children: Vec<PlanNode>) -> PlanNode {
        PlanNode {
            operator: operator.into(),
            exclusive_time_s: None,
            cardinality: Some(rows),
            estimated_cardinality: Some(rows as f64),
            row_width_bytes: Some(width),
            aggregates: Vec::new(),
            relation: None,
            subplan: None,
            rows_scanned: None,
            total_execution_time_s: None,
            children,
        }
    }

    /// An interior operator consumed whatever its children emitted --- which
    /// for an aggregate is nothing like what it emits itself.
    #[test]
    fn input_bytes_sums_what_the_children_emitted() {
        let agg = bare("HASH_GROUP_BY", 20, 8.0, vec![bare("SEQ_SCAN", 1000, 24.0, vec![])]);
        assert_eq!(agg.input_bytes(), Some(24000.0));
        assert_eq!(agg.output_bytes(), Some(160.0), "and it is not this");
    }

    /// A scan that says what it read is priced on that, not on what survived
    /// its filter.
    #[test]
    fn input_bytes_prefers_a_scans_own_count() {
        let mut scan = bare("SEQ_SCAN", 10, 24.0, vec![]);
        scan.rows_scanned = Some(1000.0);
        assert_eq!(scan.input_bytes(), Some(24000.0));
    }

    /// A leaf that reports neither is still known to have produced something.
    /// Pricing it at nothing would make the operators least is known about
    /// look cheapest of all.
    #[test]
    fn a_leaf_with_no_scan_count_falls_back_to_its_own_output() {
        assert_eq!(bare("SEQ_SCAN", 100, 24.0, vec![]).input_bytes(), Some(2400.0));
    }

    /// `Execution Time` is a property of the statement. A reader that found it
    /// on an inner node would be counting the same seconds twice.
    #[test]
    fn the_postgres_execution_time_lands_on_the_root_and_nowhere_else() {
        let json = r#"[{"Execution Time": 150.0, "Plan": {
            "Node Type": "Aggregate", "Actual Total Time": 100.0,
            "Actual Rows": 1, "Actual Loops": 1, "Plan Rows": 1, "Plan Width": 8,
            "Plans": [{"Node Type": "Seq Scan", "Actual Total Time": 60.0,
                       "Actual Rows": 1000, "Actual Loops": 1, "Plan Rows": 1000,
                       "Plan Width": 24, "Plans": []}]}}]"#;
        let plans = parse_postgres_plan(json).expect("the fixture parses");
        assert_eq!(plans[0].total_execution_time_s, Some(0.150));
        assert_eq!(plans[0].children[0].total_execution_time_s, None);
    }

    /// A scan's input is what it kept plus what its filter threw away.
    #[test]
    fn a_postgres_filter_reveals_what_the_scan_actually_read() {
        let json = r#"[{"Plan": {"Node Type": "Seq Scan", "Actual Total Time": 60.0,
            "Actual Rows": 10, "Rows Removed by Filter": 990, "Actual Loops": 1,
            "Plan Rows": 10, "Plan Width": 24, "Plans": []}}]"#;
        let plans = parse_postgres_plan(json).expect("the fixture parses");
        assert_eq!(plans[0].rows_scanned, Some(1000.0));
        assert_eq!(plans[0].input_bytes(), Some(24000.0));
    }


    const PG_ANALYZE: &str = r#"[{"Plan": {
        "Node Type": "Aggregate",
        "Actual Total Time": 100.0, "Actual Rows": 1, "Actual Loops": 1, "Plan Rows": 1,
        "Plans": [
          {"Node Type": "Hash Join",
           "Actual Total Time": 70.0, "Actual Rows": 500, "Actual Loops": 1, "Plan Rows": 480,
           "Plans": [
             {"Node Type": "Seq Scan", "Actual Total Time": 20.0, "Actual Rows": 1000,
              "Actual Loops": 1, "Plan Rows": 1000, "Plans": []}
           ]}
        ]}}]"#;

    #[test]
    fn postgres_names_the_relation_a_real_scan_reads() {
        let json = r#"[{"Plan": {"Node Type": "Seq Scan", "Relation Name": "Shipments",
            "Actual Total Time": 5.0, "Actual Rows": 10, "Actual Loops": 1,
            "Plan Rows": 10, "Plans": []}}]"#;
        let plans = parse_postgres_plan(json).unwrap();
        assert_eq!(plans[0].relation.as_deref(), Some("shipments"));
    }

    #[test]
    fn postgres_index_scans_are_leaves_too() {
        // Ranking a view by the tables it reads must not depend on which access
        // method the planner picked for them.
        for node_type in ["Index Scan", "Index Only Scan", "Bitmap Heap Scan"] {
            let json = format!(
                r#"[{{"Plan": {{"Node Type": "{node_type}", "Relation Name": "orders",
                    "Actual Total Time": 1.0, "Actual Rows": 1, "Actual Loops": 1,
                    "Plan Rows": 1, "Plans": []}}}}]"#
            );
            let plans = parse_postgres_plan(&json).unwrap();
            assert_eq!(
                plans[0].relation.as_deref(),
                Some("orders"),
                "{node_type} did not yield a leaf"
            );
        }
    }

    #[test]
    fn a_pseudo_scan_is_never_a_leaf() {
        // A CTE Scan reads an intermediate. Treating it as its own relation
        // makes every leaf-set match above it fail and attributes the whole
        // chain one level too low.
        let json = r#"[{"Plan": {"Node Type": "CTE Scan", "Relation Name": "cte_1",
            "Actual Total Time": 5.0, "Actual Rows": 10, "Actual Loops": 1,
            "Plan Rows": 10, "Plans": []}}]"#;
        let plans = parse_postgres_plan(json).unwrap();
        assert_eq!(plans[0].relation, None);

        let duck = r#"{"operator_name": "CTE_SCAN", "operator_timing": 0.1,
            "extra_info": {"Table": "cte_1"}, "children": []}"#;
        let plans = parse_duckdb_plan(duck).unwrap();
        assert_eq!(plans[0].relation, None);
    }

    #[test]
    fn duckdb_names_the_relation_and_strips_its_qualification() {
        // The plan prints `warehouse.main.shipments`; the manifest knows
        // `shipments`. Leaf sets are compared by size, so a mismatch is silent.
        let json = r#"{"operator_name": "SEQ_SCAN", "operator_timing": 0.5,
            "extra_info": {"Table": "warehouse.main.Shipments"}, "children": []}"#;
        let plans = parse_duckdb_plan(json).unwrap();
        assert_eq!(plans[0].relation.as_deref(), Some("shipments"));
    }

    #[test]
    fn duckdb_finds_scans_by_their_table_key_not_their_operator_name() {
        // The operator is SEQ_SCAN or TABLE_SCAN depending on version, and
        // READ_PARQUET / READ_CSV for external data.
        for op in ["SEQ_SCAN", "TABLE_SCAN", "READ_PARQUET"] {
            let json = format!(
                r#"{{"operator_name": "{op}", "operator_timing": 0.5,
                    "extra_info": {{"Table": "orders"}}, "children": []}}"#
            );
            let plans = parse_duckdb_plan(&json).unwrap();
            assert_eq!(plans[0].relation.as_deref(), Some("orders"), "{op}");
        }
    }

    #[test]
    fn an_operator_with_no_table_has_no_relation() {
        let json = r#"{"operator_name": "HASH_JOIN", "operator_timing": 0.5,
            "extra_info": {"Estimated Cardinality": 10}, "children": []}"#;
        let plans = parse_duckdb_plan(json).unwrap();
        assert_eq!(plans[0].relation, None);
    }

    #[test]
    fn window_and_order_by_are_not_aggregate_boundaries() {
        // They were in this set at first and it was a bug: a windowed view
        // emits one row per input row, so treating it as an aggregate made the
        // matcher skip past its very expensive region.
        assert!(!is_aggregate_boundary("WINDOW"));
        assert!(!is_aggregate_boundary("ORDER_BY"));
        assert!(!is_aggregate_boundary("Sort"));
        for op in [
            "HASH_GROUP_BY",
            "PERFECT_HASH_GROUP_BY",
            "UNGROUPED_AGGREGATE",
            "GROUP_BY",
            "DISTINCT",
            "Aggregate",
            "HashAggregate",
            "GroupAggregate",
            "MixedAggregate",
            "Group",
            "Unique",
            "SetOp",
        ] {
            assert!(is_aggregate_boundary(op), "{op} should collapse cardinality");
        }
    }

    #[test]
    fn postgres_time_is_made_exclusive() {
        let plans = parse_postgres_plan(PG_ANALYZE).unwrap();
        let agg = &plans[0];
        // 100ms inclusive minus the 70ms child = 30ms of its own work.
        assert!((agg.exclusive_time_s.unwrap() - 0.030).abs() < 1e-9);
        let join = &agg.children[0];
        assert!((join.exclusive_time_s.unwrap() - 0.050).abs() < 1e-9);
        let scan = &join.children[0];
        assert!((scan.exclusive_time_s.unwrap() - 0.020).abs() < 1e-9);
    }

    #[test]
    fn postgres_exclusive_times_sum_to_the_root_inclusive_time() {
        let plans = parse_postgres_plan(PG_ANALYZE).unwrap();
        let mut ops = Vec::new();
        plans[0].collect_operators(&mut ops);
        let total: f64 = ops.iter().map(|(_, t)| t).sum();
        assert!((total - 0.100).abs() < 1e-9);
    }

    #[test]
    fn postgres_scales_rows_and_time_by_loop_count() {
        let json = r#"[{"Plan": {"Node Type": "Nested Loop", "Actual Total Time": 5.0,
            "Actual Rows": 10, "Actual Loops": 4, "Plan Rows": 40, "Plans": []}}]"#;
        let plans = parse_postgres_plan(json).unwrap();
        // Postgres reports per-loop averages, so a node run 4 times did 4x the
        // work. Sequential loops keep being multiplied -- only the parallel
        // dimension is divided back out, see the Gather test below.
        assert_eq!(plans[0].cardinality, Some(40));
        assert!((plans[0].exclusive_time_s.unwrap() - 0.020).abs() < 1e-9);
    }

    #[test]
    fn postgres_parallel_workers_do_not_multiply_wall_time() {
        // Shape taken from a real `EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)`:
        // three workers launched, so everything under the Gather Merge reports
        // `Actual Loops: 4` -- four processes that ran at the same time, not
        // four sequential passes.
        let json = r#"[{"Plan": {
            "Node Type": "Aggregate", "Actual Total Time": 116.591,
            "Actual Rows": 3, "Actual Loops": 1, "Plan Rows": 3, "Plans": [
            {"Node Type": "Gather Merge", "Actual Total Time": 116.585,
             "Actual Rows": 12, "Actual Loops": 1, "Plan Rows": 9,
             "Workers Launched": 3, "Plans": [
            {"Node Type": "Sort", "Actual Total Time": 107.178,
             "Actual Rows": 3, "Actual Loops": 4, "Plan Rows": 3, "Plans": [
            {"Node Type": "Seq Scan", "Actual Total Time": 14.09,
             "Actual Rows": 300000, "Actual Loops": 4, "Plan Rows": 387097,
             "Relation Name": "orders", "Plans": []}]}]}]}}]"#;
        let plans = parse_postgres_plan(json).unwrap();
        let gather = &plans[0].children[0];
        let sort = &gather.children[0];

        // Before the parallel divisor, Sort's inclusive time came out as
        // 107.178 * 4 = 428ms against the Gather's 116ms, so the subtraction
        // went negative and `.max(0.0)` silently charged the Gather nothing.
        assert!(
            gather.exclusive_time_s.unwrap() > 0.0,
            "the Gather was clamped to zero again"
        );
        assert!((gather.exclusive_time_s.unwrap() - 0.009407).abs() < 1e-6);
        assert!((sort.exclusive_time_s.unwrap() - 0.093088).abs() < 1e-6);

        // Wall time still nests: no node may exceed the root that contains it.
        let root = plans[0].exclusive_time_s.unwrap()
            + gather.exclusive_time_s.unwrap()
            + sort.exclusive_time_s.unwrap()
            + sort.children[0].exclusive_time_s.unwrap();
        assert!((root - 0.116591).abs() < 1e-6, "exclusive times lost {root}");

        // Rows are unaffected: every worker really did emit its share.
        assert_eq!(sort.children[0].cardinality, Some(1_200_000));
    }

    #[test]
    fn postgres_plain_explain_has_no_timings() {
        let json = r#"[{"Plan": {"Node Type": "Seq Scan", "Plan Rows": 1000, "Plans": []}}]"#;
        let plans = parse_postgres_plan(json).unwrap();
        assert_eq!(plans[0].exclusive_time_s, None);
        assert_eq!(plans[0].estimated_cardinality, Some(1000.0));
    }

    #[test]
    fn duckdb_accepts_both_the_object_and_array_shapes() {
        let obj = r#"{"operator_name":"PROJECTION","operator_timing":0.5,
            "operator_cardinality":10,"extra_info":{"Estimated Cardinality":12},"children":[]}"#;
        let arr = format!("[{obj}]");
        for text in [obj.to_string(), arr] {
            let plans = parse_duckdb_plan(&text).unwrap();
            assert_eq!(plans[0].operator, "PROJECTION");
            assert_eq!(plans[0].exclusive_time_s, Some(0.5));
            assert_eq!(plans[0].estimated_cardinality, Some(12.0));
        }
    }

    #[test]
    fn duckdb_profiling_root_is_spliced_out_not_discarded() {
        // Shape of real `enable_profiling='json'` output: the root is a
        // query-level object with no operator_name, and the plan hangs off its
        // children. Discarding it would leave the optimizer with no operators,
        // an empty candidate ranking, and nothing to do.
        let json = r#"{
            "query_name": "CREATE TABLE x AS ...", "cpu_time": 2.8, "rows_returned": 1,
            "children": [
              {"operator_name":"BATCH_CREATE_TABLE_AS","operator_timing":0.001,
               "operator_cardinality":1,"extra_info":{"Estimated Cardinality":1},
               "children":[
                 {"operator_name":"HASH_GROUP_BY","operator_timing":1.5,
                  "operator_cardinality":541,"extra_info":{"Estimated Cardinality":600},
                  "children":[]}]}]}"#;
        let plans = parse_duckdb_plan(json).unwrap();
        assert_eq!(plans.len(), 1, "the unnamed root should be replaced by its children");
        assert_eq!(plans[0].operator, "BATCH_CREATE_TABLE_AS");

        // The operators below it must still be reachable for cost tracing.
        let mut ops = Vec::new();
        plans[0].collect_operators(&mut ops);
        let names: Vec<&str> = ops.iter().map(|(k, _)| k.name.as_str()).collect();
        assert!(names.contains(&"HASH_GROUP_BY"), "got {names:?}");
    }

    #[test]
    fn duckdb_unnamed_intermediate_node_keeps_its_subtree() {
        let json = r#"{"children":[{"children":[
            {"operator_name":"SEQ_SCAN","operator_timing":0.4,
             "extra_info":{"Estimated Cardinality":10},"children":[]}]}]}"#;
        let plans = parse_duckdb_plan(json).unwrap();
        assert_eq!(plans.len(), 1);
        assert_eq!(plans[0].operator, "SEQ_SCAN");
    }

    #[test]
    fn duckdb_estimated_cardinality_may_be_a_string() {
        let json = r#"{"operator_name":"SEQ_SCAN","operator_timing":0.1,
            "extra_info":{"Estimated Cardinality":"2048"},"children":[]}"#;
        let plans = parse_duckdb_plan(json).unwrap();
        assert_eq!(plans[0].estimated_cardinality, Some(2048.0));
    }

    #[test]
    fn signature_matches_the_same_operator_across_plans() {
        let a = parse_duckdb_plan(
            r#"{"operator_name":"HASH_JOIN","extra_info":{"Estimated Cardinality":100},"children":[]}"#,
        ).unwrap();
        let b = parse_duckdb_plan(
            r#"{"operator_name":"HASH_JOIN","operator_timing":9.0,
                "extra_info":{"Estimated Cardinality":100},"children":[]}"#,
        ).unwrap();
        assert_eq!(a[0].signature(), b[0].signature());
        assert!(b[0].contains(&a[0].signature()));
    }
}
