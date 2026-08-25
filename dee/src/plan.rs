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
        let estimated = self
            .extra_info
            .get("Estimated Cardinality")
            .and_then(json_to_f64);
        vec![PlanNode {
            operator,
            // DuckDB's operator_timing is already this operator's own time.
            exclusive_time_s: self.operator_timing,
            cardinality: self.operator_cardinality,
            estimated_cardinality: estimated,
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
    #[serde(rename = "Plans", default)]
    plans: Vec<PgPlan>,
}

impl PgPlan {
    /// Inclusive time for this node across all its loops, in seconds.
    fn inclusive_time_s(&self) -> Option<f64> {
        let per_loop_ms = self.actual_total_time?;
        Some(per_loop_ms * self.actual_loops.unwrap_or(1.0) / 1000.0)
    }

    fn into_plan_node(self) -> PlanNode {
        // `Actual Total Time` includes every child, so a node's own cost is
        // what remains after subtracting them. Without this, parents would be
        // charged for their children's work and the cost of a shared subplan
        // would be counted many times over.
        let inclusive = self.inclusive_time_s();
        let children_total: f64 = self
            .plans
            .iter()
            .filter_map(|c| c.inclusive_time_s())
            .sum();
        let exclusive = inclusive.map(|t| (t - children_total).max(0.0));

        let loops = self.actual_loops.unwrap_or(1.0);
        PlanNode {
            operator: self.node_type,
            exclusive_time_s: exclusive,
            cardinality: self.actual_rows.map(|r| (r * loops).round() as u64),
            estimated_cardinality: self.plan_rows,
            children: self.plans.into_iter().map(PgPlan::into_plan_node).collect(),
        }
    }
}

/// Parse Postgres `EXPLAIN (FORMAT JSON)` output, with or without ANALYZE.
pub fn parse_postgres_plan(json: &str) -> Option<Vec<PlanNode>> {
    let wrappers: Vec<PgPlanWrapper> = serde_json::from_str(json).ok()?;
    Some(wrappers.into_iter().map(|w| w.plan.into_plan_node()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // Postgres reports per-loop averages, so a node run 4 times did 4x the work.
        assert_eq!(plans[0].cardinality, Some(40));
        assert!((plans[0].exclusive_time_s.unwrap() - 0.020).abs() < 1e-9);
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
