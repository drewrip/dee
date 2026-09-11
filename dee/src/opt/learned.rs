//! A cost model fitted to what the engine actually did.
//!
//! Every `CREATE TABLE ... AS` dee runs comes back with an EXPLAIN ANALYZE
//! plan, and every operator in that plan carries three things: how long it
//! took, how many tuples it emitted, and how wide one of those tuples is (see
//! [`PlanNode::row_width_bytes`]). Rows times width is the operator's output in
//! bytes, and seconds divided by bytes is a **seconds-per-byte** constant for
//! that operator type --- a small number that says what this engine, on this
//! machine, charges to push a byte through a hash join.
//!
//! Those constants accumulate across runs. Costing a plan is then arithmetic on
//! the plan alone: for each operator, its output bytes times the constant for
//! its type, summed. Nothing about it needs the plan to have been executed,
//! which is the point --- a VIEW's plan never is.
//!
//! # What "operator type" means
//!
//! The operator's name, except for aggregates, which also carry the functions
//! they compute ([`PlanNode::cost_key`]). `count(*)` and `string_agg` share a
//! `HASH_GROUP_BY` and cost nothing like each other per output byte, and one
//! constant fitted to both is a constant fitted to neither.
//!
//! # Aggregating repeat observations
//!
//! An operator type is seen many times, and each sighting gives its own
//! seconds-per-byte. They are combined **weighted by bytes**: total seconds
//! over total bytes, so an observation counts in proportion to the work it
//! actually represents.
//!
//! The unweighted mean of the ratios was tried first and is kept alongside as
//! [`OperatorSamples::unweighted_seconds_per_byte`], because the difference
//! between them is not academic. On p03_ecommerce a single `HASH_GROUP_BY`
//! whose output was small enough that fixed per-call overhead dwarfed anything
//! proportional to bytes measured 1.6e-4 s/byte --- a hundred thousand times a
//! hash join's 6.2e-10. Counted once, it set the constant for its whole
//! operator type, and multiplied by a 6.9M-row view it priced that view at
//! 13,199 seconds on a DAG that runs in three. Weighting by bytes puts that
//! same sighting where it belongs: it moved a few hundred bytes, so it moves
//! the constant by a few hundred bytes' worth.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::plan::{PlanNode, is_write_operator};

/// Everything observed about one operator type.
///
/// Sums rather than a running mean: sums merge associatively, which is what
/// lets a model loaded from the metadata store be combined with one fitted to
/// the run that just finished without keeping either's observations around.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct OperatorSamples {
    /// Sum over observations of `seconds / bytes`.
    pub seconds_per_byte_sum: f64,
    /// Observations behind `seconds_per_byte_sum`.
    pub seconds_per_byte_n: u64,
    /// Sum over observations of one output tuple's width in bytes.
    pub bytes_per_tuple_sum: f64,
    /// Observations behind `bytes_per_tuple_sum`. Larger than
    /// `seconds_per_byte_n`, because a width is observable on an un-timed plan.
    pub bytes_per_tuple_n: u64,
    /// Total measured seconds and total output bytes across observations. Not
    /// read by the mean below; kept so a byte-weighted constant can be fitted
    /// from stored state without re-collecting plans.
    pub seconds_total: f64,
    pub bytes_total: f64,
}

impl OperatorSamples {
    /// Total seconds over total bytes: the byte-weighted constant.
    ///
    /// This is what prices a plan. A sighting counts for as much as the work it
    /// did, so an operator called once on a handful of bytes cannot outvote a
    /// scan of a million rows.
    pub fn seconds_per_byte(&self) -> Option<f64> {
        (self.bytes_total > 0.0).then(|| self.seconds_total / self.bytes_total)
    }

    /// The plain mean of the per-observation ratios, every sighting counting
    /// once. Not used to price anything; kept because it is the obvious
    /// alternative and the two are worth comparing on a real DAG.
    pub fn unweighted_seconds_per_byte(&self) -> Option<f64> {
        (self.seconds_per_byte_n > 0)
            .then(|| self.seconds_per_byte_sum / self.seconds_per_byte_n as f64)
    }

    /// The mean observed width of one output tuple.
    pub fn bytes_per_tuple(&self) -> Option<f64> {
        (self.bytes_per_tuple_n > 0)
            .then(|| self.bytes_per_tuple_sum / self.bytes_per_tuple_n as f64)
    }

    pub fn merge(&mut self, other: &OperatorSamples) {
        self.seconds_per_byte_sum += other.seconds_per_byte_sum;
        self.seconds_per_byte_n += other.seconds_per_byte_n;
        self.bytes_per_tuple_sum += other.bytes_per_tuple_sum;
        self.bytes_per_tuple_n += other.bytes_per_tuple_n;
        self.seconds_total += other.seconds_total;
        self.bytes_total += other.bytes_total;
    }
}

/// Seconds-per-byte constants, one per operator type, fitted to executed plans.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LearnedCostModel {
    ops: HashMap<String, OperatorSamples>,
}

impl LearnedCostModel {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    pub fn operators(&self) -> impl Iterator<Item = (&String, &OperatorSamples)> {
        self.ops.iter()
    }

    /// Fold another model's observations into this one.
    pub fn merge(&mut self, other: &LearnedCostModel) {
        for (key, samples) in &other.ops {
            self.ops.entry(key.clone()).or_default().merge(samples);
        }
    }

    /// Record what every operator of an executed plan revealed.
    ///
    /// Safe to call on an un-executed plan: a timing is required before a
    /// seconds-per-byte observation is recorded, so a plain EXPLAIN contributes
    /// only widths --- which are just as real there as anywhere.
    pub fn observe(&mut self, roots: &[PlanNode]) {
        for root in roots {
            self.observe_node(root);
        }
    }

    fn observe_node(&mut self, node: &PlanNode) {
        // A write operator's output is a one-row count of what it wrote, not
        // what it worked on, so a constant fitted to it describes nothing ---
        // see `is_write_operator`.
        if !is_write_operator(&node.operator)
            && let Some(width) = node.row_width_bytes.filter(|w| *w > 0.0)
        {
            let entry = self.ops.entry(node.cost_key()).or_default();
            entry.bytes_per_tuple_sum += width;
            entry.bytes_per_tuple_n += 1;

            // Both halves of the ratio have to be measured, not estimated: an
            // operator's cost is fitted to what it did, and `cardinality` is
            // `None` on a plan that was never executed.
            if let (Some(seconds), Some(bytes)) = (node.exclusive_time_s, node.output_bytes())
                && bytes > 0.0
            {
                entry.seconds_per_byte_sum += seconds / bytes;
                entry.seconds_per_byte_n += 1;
                entry.seconds_total += seconds;
                entry.bytes_total += bytes;
            }
        }
        for child in &node.children {
            self.observe_node(child);
        }
    }

    /// The constant for `key`, falling back to the model's overall mean for an
    /// operator type never seen before.
    ///
    /// Falling back rather than returning zero: an unseen operator priced at
    /// zero is not "unknown", it is "free", and a plan full of unseen operators
    /// would come out cheapest of all --- exactly backwards.
    pub fn seconds_per_byte(&self, key: &str) -> Option<f64> {
        self.ops
            .get(key)
            .and_then(OperatorSamples::seconds_per_byte)
            .or_else(|| self.mean_seconds_per_byte())
    }

    /// The byte-weighted seconds-per-byte over every operator type: the whole
    /// model's total seconds over its total bytes.
    ///
    /// Weighted for the same reason the per-operator constant is, and more
    /// urgently --- this is the fallback an operator type nobody has ever seen
    /// gets priced at, so one wild observation anywhere would otherwise set the
    /// price of everything unknown.
    pub fn mean_seconds_per_byte(&self) -> Option<f64> {
        let (seconds, bytes) = self
            .ops
            .values()
            .fold((0.0, 0.0), |(s, b), o| (s + o.seconds_total, b + o.bytes_total));
        (bytes > 0.0).then(|| seconds / bytes)
    }

    /// The mean output tuple width over every observation of every operator
    /// type.
    pub fn mean_bytes_per_tuple(&self) -> Option<f64> {
        let (sum, n) = self.ops.values().fold((0.0, 0u64), |(s, n), o| {
            (s + o.bytes_per_tuple_sum, n + o.bytes_per_tuple_n)
        });
        (n > 0).then(|| sum / n as f64)
    }

    /// How wide one of `node`'s output tuples is: what the plan says, or what
    /// this operator type was measured at, or the model's overall mean.
    ///
    /// The fallbacks exist for DuckDB, whose plain `EXPLAIN (FORMAT JSON)` --- the
    /// only plan a VIEW ever has --- reports no width at all. Postgres prints
    /// `Plan Width` with or without ANALYZE and takes the first branch.
    fn width_of(&self, node: &PlanNode) -> Option<f64> {
        node.row_width_bytes
            .filter(|w| *w > 0.0)
            .or_else(|| {
                self.ops
                    .get(&node.cost_key())
                    .and_then(OperatorSamples::bytes_per_tuple)
            })
            .or_else(|| self.mean_bytes_per_tuple())
    }

    /// What this plan should cost: output bytes times seconds-per-byte, summed
    /// over every operator.
    ///
    /// `None` when the model has learned nothing at all, or when no operator in
    /// the plan carried enough information to be priced --- neither is a cost of
    /// zero, and reporting one would rank a plan nothing is known about above
    /// every plan something is.
    pub fn cost(&self, roots: &[PlanNode]) -> Option<f64> {
        if self.is_empty() {
            return None;
        }
        let mut total = 0.0;
        let mut priced = 0usize;
        for root in roots {
            self.cost_node(root, &mut total, &mut priced);
        }
        (priced > 0).then_some(total)
    }

    fn cost_node(&self, node: &PlanNode, total: &mut f64, priced: &mut usize) {
        if !is_write_operator(&node.operator)
            && let (Some(rows), Some(width), Some(spb)) = (
            node.rows(),
            self.width_of(node),
            self.seconds_per_byte(&node.cost_key()),
        ) {
            *total += rows * width * spb;
            *priced += 1;
        }
        for child in &node.children {
            self.cost_node(child, total, priced);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::{parse_duckdb_plan, parse_postgres_plan};

    /// A profiling plan: two operators, known bytes, known times.
    fn duckdb_profile() -> String {
        // 1000 rows x 24 bytes = 24000 bytes in 0.002s -> 8.33e-8 s/byte.
        // 100 rows x 10 bytes = 1000 bytes in 0.001s   -> 1e-6   s/byte.
        r#"{"operator_name":"HASH_GROUP_BY","operator_timing":0.001,
            "operator_cardinality":100,"result_set_size":1000,
            "extra_info":{"Aggregates":["sum(#1)"],"Estimated Cardinality":100},
            "children":[
              {"operator_name":"SEQ_SCAN","operator_timing":0.002,
               "operator_cardinality":1000,"result_set_size":24000,
               "extra_info":{"Table":"orders","Estimated Cardinality":1000},
               "children":[]}]}"#
            .to_string()
    }

    #[test]
    fn an_operators_constant_is_its_seconds_divided_by_its_output_bytes() {
        let plan = parse_duckdb_plan(&duckdb_profile()).unwrap();
        let mut model = LearnedCostModel::new();
        model.observe(&plan);

        assert!((model.seconds_per_byte("SEQ_SCAN").unwrap() - 0.002 / 24000.0).abs() < 1e-15);
        assert!(
            (model.seconds_per_byte("HASH_GROUP_BY[sum]").unwrap() - 0.001 / 1000.0).abs() < 1e-15
        );
    }

    #[test]
    fn an_aggregates_type_carries_the_function_it_computes() {
        let plan = parse_duckdb_plan(&duckdb_profile()).unwrap();
        let mut model = LearnedCostModel::new();
        model.observe(&plan);
        let keys: Vec<&String> = model.ops.keys().collect();
        assert!(keys.iter().any(|k| k.as_str() == "HASH_GROUP_BY[sum]"));
        // The bare name is *not* a key: two aggregates that differ only in
        // their function must not share a constant.
        assert!(!keys.iter().any(|k| k.as_str() == "HASH_GROUP_BY"));
    }

    #[test]
    fn repeat_observations_of_an_operator_are_combined() {
        let mut model = LearnedCostModel::new();
        // 100 bytes in 1s, then 100 bytes in 3s: 0.01 and 0.03 s/byte. Equal
        // byte counts, so weighting changes nothing and both means agree.
        for seconds in [1.0, 3.0] {
            let json = format!(
                r#"{{"operator_name":"FILTER","operator_timing":{seconds},
                     "operator_cardinality":10,"result_set_size":100,
                     "extra_info":{{}},"children":[]}}"#
            );
            model.observe(&parse_duckdb_plan(&json).unwrap());
        }
        assert!((model.seconds_per_byte("FILTER").unwrap() - 0.02).abs() < 1e-12);
    }

    #[test]
    fn a_tiny_observation_cannot_outvote_a_large_one() {
        // The p03 failure in miniature. One call on 100 bytes that took 1s --
        // fixed overhead, nothing to do with bytes -- against a million bytes in
        // 1s. Counted once each the constant would be ~5e-3 s/byte, set almost
        // entirely by the operator that moved a thousandth of the bytes.
        let mut model = LearnedCostModel::new();
        for (seconds, rows, bytes) in [(1.0, 10u64, 100u64), (1.0, 100_000u64, 1_000_000u64)] {
            let json = format!(
                r#"{{"operator_name":"HASH_GROUP_BY","operator_timing":{seconds},
                     "operator_cardinality":{rows},"result_set_size":{bytes},
                     "extra_info":{{"Aggregates":["sum(#1)"]}},"children":[]}}"#
            );
            model.observe(&parse_duckdb_plan(&json).unwrap());
        }

        let key = "HASH_GROUP_BY[sum]";
        // Weighted: 2 seconds over 1,000,100 bytes.
        let weighted = model.seconds_per_byte(key).unwrap();
        assert!((weighted - 2.0 / 1_000_100.0).abs() < 1e-15, "was {weighted}");
        // Unweighted: (0.01 + 0.000001) / 2, ~2500x larger, and the reason the
        // p03 ranking put 13,199 seconds on a three-second DAG.
        let unweighted = model
            .ops
            .get(key)
            .unwrap()
            .unweighted_seconds_per_byte()
            .unwrap();
        assert!(unweighted > weighted * 2000.0, "unweighted was {unweighted}");
    }

    #[test]
    fn merging_two_models_is_the_same_as_observing_both_runs() {
        let plan = parse_duckdb_plan(&duckdb_profile()).unwrap();
        let mut both = LearnedCostModel::new();
        both.observe(&plan);
        both.observe(&plan);

        let mut a = LearnedCostModel::new();
        a.observe(&plan);
        let mut b = LearnedCostModel::new();
        b.observe(&plan);
        a.merge(&b);

        assert_eq!(a, both);
    }

    #[test]
    fn costing_a_plan_prices_every_operators_output_bytes() {
        let mut model = LearnedCostModel::new();
        model.observe(&parse_duckdb_plan(&duckdb_profile()).unwrap());

        // A VIEW's plan: same operators, estimates only, no widths and no
        // timings. Widths come from what the same operator types were measured
        // at, so the cost is the profiled cost reproduced from estimates.
        let view = r#"[{"name":"HASH_GROUP_BY","extra_info":
            {"Aggregates":["sum(#1)"],"Estimated Cardinality":100},"children":[
              {"name":"SEQ_SCAN","extra_info":
                {"Table":"orders","Estimated Cardinality":1000},"children":[]}]}]"#;
        let cost = model.cost(&parse_duckdb_plan(view).unwrap()).unwrap();
        assert!((cost - (0.001 + 0.002)).abs() < 1e-12, "cost was {cost}");
    }

    #[test]
    fn an_operator_type_never_seen_costs_the_average_rather_than_nothing() {
        let mut model = LearnedCostModel::new();
        model.observe(&parse_duckdb_plan(&duckdb_profile()).unwrap());
        let unseen = r#"[{"name":"PIVOT","extra_info":{"Estimated Cardinality":1000},
                          "children":[]}]"#;
        let cost = model.cost(&parse_duckdb_plan(unseen).unwrap()).unwrap();
        assert!(cost > 0.0, "an unpriced operator must not be free");
    }

    #[test]
    fn a_write_operator_teaches_the_model_nothing() {
        // DuckDB gives CREATE_TABLE_AS one row of 8 bytes however large the
        // table it wrote, so its seconds-per-byte is enormous and meaningless.
        // Left in, it sets the model's mean and so prices every operator type
        // the model has never seen.
        let json = r#"{"operator_name":"CREATE_TABLE_AS","operator_timing":0.26,
             "operator_cardinality":1,"result_set_size":8,"extra_info":{},
             "children":[
               {"operator_name":"SEQ_SCAN","operator_timing":0.002,
                "operator_cardinality":1000,"result_set_size":24000,
                "extra_info":{"Table":"orders","Estimated Cardinality":1000},
                "children":[]}]}"#;
        let mut model = LearnedCostModel::new();
        model.observe(&parse_duckdb_plan(json).unwrap());

        assert!(model.ops.get("CREATE_TABLE_AS").is_none());
        // The mean is the scan's constant alone, not 0.0325 s/byte.
        assert!((model.mean_seconds_per_byte().unwrap() - 0.002 / 24000.0).abs() < 1e-15);
    }

    #[test]
    fn a_model_that_has_learned_nothing_declines_to_cost() {
        let model = LearnedCostModel::new();
        assert!(model.cost(&parse_duckdb_plan(&duckdb_profile()).unwrap()).is_none());
    }

    #[test]
    fn postgres_widths_come_from_the_plan_itself() {
        // `Plan Width` is reported with and without ANALYZE, so nothing has to
        // be learned before a Postgres VIEW plan can be priced.
        let analyzed = r#"[{"Plan":{"Node Type":"Aggregate","Strategy":"Hashed",
            "Actual Total Time":2.0,"Actual Rows":100,"Actual Loops":1,
            "Plan Rows":100,"Plan Width":40,
            "Output":["id","sum(f)","count(*)"],"Group Key":["t.id"],
            "Plans":[{"Node Type":"Seq Scan","Relation Name":"t",
                      "Actual Total Time":1.0,"Actual Rows":1000,"Actual Loops":1,
                      "Plan Rows":1000,"Plan Width":36,"Output":["id","f"]}]}}]"#;
        let mut model = LearnedCostModel::new();
        model.observe(&parse_postgres_plan(analyzed).unwrap());

        // 100 rows x 40 bytes, in (2.0 - 1.0) ms.
        let agg = model.seconds_per_byte("AGGREGATE[count,sum]").unwrap();
        assert!((agg - 0.001 / 4000.0).abs() < 1e-15);
        // 1000 rows x 36 bytes in 1.0 ms.
        let scan = model.seconds_per_byte("SEQ SCAN").unwrap();
        assert!((scan - 0.001 / 36000.0).abs() < 1e-15);
    }

    #[test]
    fn a_postgres_group_key_expression_is_not_mistaken_for_an_aggregate() {
        let plan = r#"[{"Plan":{"Node Type":"Aggregate","Plan Rows":10,"Plan Width":8,
            "Output":["date_trunc('day', ts)","count(*)"],
            "Group Key":["date_trunc('day', ts)"]}}]"#;
        let parsed = parse_postgres_plan(plan).unwrap();
        assert_eq!(parsed[0].cost_key(), "AGGREGATE[count]");
    }
}
