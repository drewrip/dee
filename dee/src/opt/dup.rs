//! Duplicate-computation attribution: what a VIEW costs by *not* being a table.
//!
//! Every other costing method in HMP answers "what does this View cost to
//! compute?" and then guesses at the duplication from the graph --- the learned
//! method multiplies by the number of consumers, the leaf-set method subtracts
//! the largest attribution from the sum. Both are estimates of the quantity the
//! materialization hypothesis is actually about, which is:
//!
//! > If this View were built once instead of inlined into each of its
//! > consumers, how much computation would simply stop happening?
//!
//! This module measures that quantity by asking the engine directly.
//!
//! # The method
//!
//! For a candidate View `V`:
//!
//! 1. **Contract the graph minor around `V`.** Every View between `V` and a
//!    materialized consumer is text, not a relation, so it is inlined into that
//!    consumer ([`contract_intermediate_views`]). What is left is the graph the
//!    engine really runs: `V` with an edge to each consumer that will pay for
//!    it.
//! 2. **Inline `V` into each consumer as a materialized CTE.** The consumer's
//!    references to `V` are repointed at a CTE whose body is `V`'s own query,
//!    marked `MATERIALIZED`. The keyword buys one thing, and it is the thing
//!    this needs: the engine plans `V` as a single shared computation with a
//!    name on it, instead of folding a copy of it into each reference, so the
//!    plan has a region that *is* `V` and can be pointed at.
//! 3. **EXPLAIN each consumer.** Both backends say which part of a plan
//!    computes a CTE, and [`PlanNode::subplan`] normalizes the two spellings,
//!    so the subtree that *is* `V` inside consumer `i`'s plan can be picked out
//!    of the plan by name.
//! 4. **Cost each subtree.** By whatever [`SubtreeCostMethod`] is configured;
//!    [`SubtreeCostMethod::LearnedCost`] by default, which prices a plan in
//!    seconds using per-operator constants fitted to plans the engine really
//!    executed (see [`crate::opt::learned`]).
//! 5. `dup(V) = c₁ + ... + cₙ - cost(V)`, where `cost(V)` is `V`'s own EXPLAIN
//!    plan costed the same way. Every consumer computes `V`; only one of those
//!    computations would survive materializing it, so the rest is the
//!    duplicate.
//!
//! # What `MATERIALIZED` does not do
//!
//! It is not a full optimizer barrier, and expecting one would be a mistake
//! worth writing down. DuckDB still pushes a consumer's predicates and
//! projections into the CTE's body: `... FROM v WHERE g = 3` plans the CTE with
//! `g = 3` on its scan, over a seventh of the rows. So the copies are *not*
//! each a standalone build of `V`.
//!
//! That is the right answer, not a leak. The question being asked of consumer
//! `i` is "what does computing `V` cost you today", and today `V` is a view
//! whose body is folded into `i`'s query with exactly those predicates pushed
//! into it. `cᵢ` is what `i` really pays. `cost(V)` --- the standalone plan --- is
//! what one shared build would really cost. Their difference is therefore the
//! honest saving, and it can be **negative**: a set of consumers that each read
//! a narrow slice of a wide View would collectively do *less* work than
//! building the whole thing once, and this reports that rather than hiding it.
//!
//! # Why this beats multiplying by the consumer count
//!
//! The copies are not identical, and the differences are the interesting part.
//! One consumer filters. Another projects two columns of thirty, and the
//! engine prunes the rest out of the CTE's body. A third sits behind a join the
//! planner reorders. The engine's own plan for each consumer accounts for all
//! of that; `n × cost(V)` accounts for none of it.
//!
//! # What it costs
//!
//! One EXPLAIN per consumer per candidate, against the live engine. An EXPLAIN
//! plans a query without running it, so this is planner time and nothing else;
//! it is still the only costing method here that talks to the database at all,
//! which is why it is not the default.

use std::collections::HashSet;

use log::{debug, warn};
use serde::{Deserialize, Serialize};

use crate::{
    connectors::Connector,
    dag::Dag,
    opt::{
        common::{
            bare_table_name, contract_intermediate_views, dialect_for_db, rewrite_node_refs,
            wrap_in_materialized_cte,
        },
        learned::LearnedCostModel,
    },
    plan::{PlanNode, find_subplan, is_write_operator},
};

/// How a plan subtree is turned into a number.
///
/// The attribution above decides *which* operators belong to a View; this
/// decides what they are worth. The two are independent, so the second is a
/// setting: a run can keep the same attribution and ask what it looks like
/// priced in seconds, in rows, or in operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubtreeCostMethod {
    /// Seconds, from per-operator seconds-per-byte constants fitted to executed
    /// plans ([`crate::opt::learned`]).
    ///
    /// The default, and the only one denominated in the same unit as the
    /// runtimes the search is trying to reduce.
    #[default]
    LearnedCost,
    /// Rows the plan is estimated to move: every operator's output cardinality,
    /// summed.
    ///
    /// Engine-independent and always available --- a plan that has learned
    /// nothing still has estimates --- but it prices a row of forty columns the
    /// same as a row of one, and a hash join the same as a projection.
    Cardinality,
    /// Operators in the subtree, counted.
    ///
    /// A control rather than a cost model: it knows nothing about data volume
    /// at all, so any advantage the other two show over it is the part of the
    /// ranking that came from costing rather than from attribution.
    Operators,
}

impl SubtreeCostMethod {
    pub fn as_str(&self) -> &'static str {
        match self {
            SubtreeCostMethod::LearnedCost => "learned_cost",
            SubtreeCostMethod::Cardinality => "cardinality",
            SubtreeCostMethod::Operators => "operators",
        }
    }

    /// The coster this method names, borrowing `model` if it needs one.
    pub fn coster<'a>(&self, model: &'a LearnedCostModel) -> Box<dyn SubtreeCost + 'a> {
        match self {
            SubtreeCostMethod::LearnedCost => Box::new(LearnedSubtreeCost(model)),
            SubtreeCostMethod::Cardinality => Box::new(CardinalitySubtreeCost),
            SubtreeCostMethod::Operators => Box::new(OperatorCountSubtreeCost),
        }
    }
}

impl std::str::FromStr for SubtreeCostMethod {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "learned_cost" | "learnedcost" => Ok(SubtreeCostMethod::LearnedCost),
            "cardinality" | "rows" => Ok(SubtreeCostMethod::Cardinality),
            "operators" | "operator_count" => Ok(SubtreeCostMethod::Operators),
            other => Err(format!(
                "unknown subtree cost method '{other}'; expected learned_cost, cardinality \
                 or operators"
            )),
        }
    }
}

/// What one region of a query plan is worth.
///
/// A trait rather than an enum match at the call site because the interesting
/// direction of this work is adding cost models, and a new one should be a new
/// implementation rather than a new arm in the middle of the attribution.
///
/// `Send + Sync` because the attribution holds one across the EXPLAIN of every
/// consumer, and those are awaits inside a pass the server drives from a
/// multi-threaded runtime.
pub trait SubtreeCost: Send + Sync {
    /// `None` when this model cannot price these operators at all --- which is
    /// not a cost of zero, and must not be read as one.
    fn cost(&self, roots: &[PlanNode]) -> Option<f64>;
}

/// Seconds, via the learned seconds-per-byte constants.
pub struct LearnedSubtreeCost<'a>(pub &'a LearnedCostModel);

impl SubtreeCost for LearnedSubtreeCost<'_> {
    fn cost(&self, roots: &[PlanNode]) -> Option<f64> {
        self.0.cost(roots)
    }
}

/// Estimated output rows, summed over every operator.
pub struct CardinalitySubtreeCost;

impl SubtreeCost for CardinalitySubtreeCost {
    fn cost(&self, roots: &[PlanNode]) -> Option<f64> {
        fn walk(node: &PlanNode, total: &mut f64, priced: &mut usize) {
            if !is_write_operator(&node.operator)
                && let Some(rows) = node.rows()
            {
                *total += rows;
                *priced += 1;
            }
            for child in &node.children {
                walk(child, total, priced);
            }
        }
        let (mut total, mut priced) = (0.0, 0usize);
        for root in roots {
            walk(root, &mut total, &mut priced);
        }
        (priced > 0).then_some(total)
    }
}

/// Operators, counted.
pub struct OperatorCountSubtreeCost;

impl SubtreeCost for OperatorCountSubtreeCost {
    fn cost(&self, roots: &[PlanNode]) -> Option<f64> {
        fn walk(node: &PlanNode) -> usize {
            usize::from(!is_write_operator(&node.operator))
                + node.children.iter().map(walk).sum::<usize>()
        }
        let n: usize = roots.iter().map(walk).sum();
        (n > 0).then_some(n as f64)
    }
}

/// What one View's duplication came to, and the numbers behind it.
///
/// The components are kept rather than just the difference because the
/// difference alone cannot be sanity-checked: a `duplicate` of zero means
/// "one consumer" when there is one entry in `per_consumer` and "the engine
/// planned every copy away" when there are four.
#[derive(Debug, Clone)]
pub struct DuplicateCost {
    pub view: String,
    /// What computing the View once costs, from its own EXPLAIN plan.
    pub own: f64,
    /// `(consumer node, what the View's CTE region of its plan costs)`, sorted
    /// by consumer for a stable report.
    pub per_consumer: Vec<(String, f64)>,
    /// `sum(per_consumer) - own`: the computation that would stop happening if
    /// the View were built once.
    ///
    /// Can come out negative, and that is a finding rather than a bug: it says
    /// the engine plans the View more cheaply inside its consumers --- pushing
    /// their predicates and projections into it --- than it would standing
    /// alone, so materializing it would *add* work.
    pub duplicate: f64,
}

/// The CTE name a View is inlined under.
///
/// Prefixed so it cannot collide with a CTE the consumer already defines, and
/// derived from the View so two candidates in one query are still distinct.
fn cte_name(view_id: &str) -> String {
    format!("dee_dup_{}", bare_table_name(view_id))
}

/// The consumer query with `view` inlined into it as a materialized CTE, one
/// per materialized consumer on `view`'s frontier.
///
/// `dag` must already be the contracted minor around `view` --- every consumer
/// here refers to `view` directly.
fn consumer_queries(dag: &Dag, view: &str, frontier: &[String]) -> Vec<(String, String)> {
    let dialect = dialect_for_db(&dag.db);
    let cte = cte_name(view);
    let Some(body) = dag.nodes.get(view.to_string()).map(|v| v.query_text.clone()) else {
        return Vec::new();
    };

    let mut out = Vec::new();
    for consumer in frontier {
        let Some(node) = dag.nodes.get(consumer.clone()) else {
            continue;
        };
        // Repoint the consumer's references at the CTE. On the AST, like every
        // other rewrite here: a textual one would rename the View's name
        // wherever it appears as a string literal or a column alias.
        let mapping = std::collections::HashMap::from([(view.to_string(), cte.clone())]);
        let Some(repointed) = rewrite_node_refs(&node.query_text, &mapping, dialect) else {
            warn!(
                "dup: could not repoint '{consumer}' at a CTE for '{view}'; \
                 skipping this consumer"
            );
            continue;
        };
        match wrap_in_materialized_cte(&repointed, &cte, &body, dialect) {
            Some(sql) => out.push((consumer.clone(), sql)),
            None => warn!(
                "dup: could not wrap '{view}' into '{consumer}' as a CTE; \
                 skipping this consumer"
            ),
        }
    }
    out
}

/// Attribute `view`'s duplicate computation, by planning each of its consumers
/// with it inlined as a materialized CTE.
///
/// `view_plan` is the View's own EXPLAIN plan, already collected by the run
/// being analyzed --- it is the `cost(V)` term, and re-EXPLAINing to get it
/// would only ask the engine something it has already answered.
///
/// Returns `None` when nothing could be measured: the View has no materialized
/// consumer, no consumer's plan could be obtained, or the cost model declined
/// to price what came back. The caller must treat that as "unknown" and fall
/// back, never as a duplication of zero.
pub async fn duplicate_cost<C>(
    conn: &C,
    dag: &Dag,
    view: &str,
    view_plan: &[PlanNode],
    coster: &dyn SubtreeCost,
) -> Option<DuplicateCost>
where
    C: Connector + Send + Sync,
{
    let frontier: HashSet<String> = dag.nodes.frontier_materializes(view);
    if frontier.is_empty() {
        return None;
    }
    let mut frontier: Vec<String> = frontier.into_iter().collect();
    frontier.sort();

    // The graph minor: everything between the View and its materialized
    // consumers folded into the consumers, on a copy, because this is a
    // question about the DAG and not a change to it.
    let mut minor = dag.clone();
    if let Err(e) = contract_intermediate_views(&mut minor, view, &frontier.iter().cloned().collect())
    {
        warn!("dup: could not contract the graph minor around '{view}': {e}");
        return None;
    }

    let own = coster.cost(view_plan)?;

    let mut per_consumer: Vec<(String, f64)> = Vec::new();
    for (consumer, sql) in consumer_queries(&minor, view, &frontier) {
        let plan_json = match conn.explain(&sql).await {
            Ok(Some(json)) => json,
            Ok(None) => {
                debug!("dup: this connector cannot EXPLAIN, so '{view}' cannot be attributed");
                return None;
            }
            Err(e) => {
                warn!("dup: EXPLAIN of '{consumer}' with '{view}' inlined failed: {e}");
                continue;
            }
        };
        let Some(plans) = conn.parse_plan(&plan_json) else {
            warn!("dup: could not parse the plan of '{consumer}' with '{view}' inlined");
            continue;
        };
        // No region for the CTE means the engine did not keep it as one ---
        // an older server that ignores the hint, or a consumer whose reference
        // to the View the rewrite did not reach. Either way this consumer's
        // copy is not measurable, and charging it zero would understate the
        // duplication rather than admit to not knowing.
        let Some(region) = find_subplan(&plans, &cte_name(view)) else {
            warn!(
                "dup: the plan of '{consumer}' has no region for '{view}'s materialized CTE; \
                 skipping this consumer"
            );
            continue;
        };
        let Some(cost) = coster.cost(std::slice::from_ref(region)) else {
            debug!("dup: the cost model declined to price '{view}' inside '{consumer}'");
            continue;
        };
        per_consumer.push((consumer, cost));
    }

    if per_consumer.is_empty() {
        return None;
    }

    let total: f64 = per_consumer.iter().map(|(_, c)| c).sum();
    let duplicate = total - own;
    debug!(
        "dup({view}) = {total:.4} - {own:.4} = {duplicate:.4} over {} consumer(s)",
        per_consumer.len()
    );
    Some(DuplicateCost {
        view: view.to_string(),
        own,
        per_consumer,
        duplicate,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dag::{MaterializeMode, TransformNode},
        graph::Graph,
        plan::parse_duckdb_plan,
    };
    use std::collections::HashMap;

    fn dag_of(nodes: Vec<TransformNode>) -> Dag {
        let mut map = HashMap::new();
        for n in nodes {
            map.insert(n.id.clone(), n);
        }
        Dag {
            db: "duckdb".into(),
            nodes: Graph::new(map),
            sources: vec![],
            max_parallelism: None,
        }
    }

    fn node(id: &str, mode: MaterializeMode, deps: &[&str], query: &str) -> TransformNode {
        TransformNode {
            id: id.to_string(),
            query_text: query.to_string(),
            materialize: mode,
            depends_on: deps.iter().map(|s| s.to_string()).collect(),
            schema: None,
        }
    }

    /// A View read by two tables: both consumer queries must come back with the
    /// View's body inside a materialized CTE and their own reference repointed
    /// at it.
    #[test]
    fn each_consumer_gets_the_view_as_a_materialized_cte() {
        let dag = dag_of(vec![
            node(
                "v",
                MaterializeMode::View,
                &["raw"],
                "SELECT id, amt FROM raw WHERE amt > 0",
            ),
            node("m1", MaterializeMode::Table, &["v"], "SELECT sum(amt) AS s FROM v"),
            node("m2", MaterializeMode::Table, &["v"], "SELECT count(*) AS n FROM v"),
        ]);
        let queries = consumer_queries(&dag, "v", &["m1".into(), "m2".into()]);
        assert_eq!(queries.len(), 2);
        for (consumer, sql) in &queries {
            let upper = sql.to_uppercase();
            assert!(
                upper.contains("MATERIALIZED"),
                "{consumer} lost the optimizer barrier, so the region would be \
                 the View fused into its consumer: {sql}"
            );
            assert!(
                sql.contains("dee_dup_v"),
                "{consumer} was not repointed at the CTE: {sql}"
            );
            assert!(
                sql.contains("amt > 0"),
                "{consumer} did not get the View's body: {sql}"
            );
        }
    }

    /// The View may sit behind other Views. Those are text, not relations, so
    /// the minor has to contract them before the consumer can name the View.
    #[test]
    fn an_intermediate_view_is_contracted_before_the_cte_goes_in() {
        let mut dag = dag_of(vec![
            node("v", MaterializeMode::View, &["raw"], "SELECT id, amt FROM raw"),
            node(
                "mid",
                MaterializeMode::View,
                &["v"],
                "SELECT id, amt FROM v WHERE amt > 10",
            ),
            node(
                "m1",
                MaterializeMode::Table,
                &["mid"],
                "SELECT sum(amt) AS s FROM mid",
            ),
        ]);
        let frontier: HashSet<String> = dag.nodes.frontier_materializes("v");
        assert_eq!(frontier, HashSet::from(["m1".to_string()]));
        contract_intermediate_views(&mut dag, "v", &frontier).unwrap();

        let queries = consumer_queries(&dag, "v", &["m1".into()]);
        assert_eq!(queries.len(), 1);
        let sql = &queries[0].1;
        assert!(sql.contains("dee_dup_v"), "the CTE did not reach m1: {sql}");
        assert!(
            sql.contains("amt > 10"),
            "the intermediate view's predicate was lost: {sql}"
        );
    }

    /// A consumer with a `WITH` clause of its own must keep it.
    #[test]
    fn a_consumers_existing_ctes_survive() {
        let dag = dag_of(vec![
            node("v", MaterializeMode::View, &["raw"], "SELECT id FROM raw"),
            node(
                "m1",
                MaterializeMode::Table,
                &["v"],
                "WITH mine AS (SELECT 1 AS x) SELECT count(*) AS n FROM v, mine",
            ),
        ]);
        let sql = &consumer_queries(&dag, "v", &["m1".into()])[0].1;
        assert!(sql.contains("mine"), "the consumer's own CTE was dropped: {sql}");
        assert!(sql.contains("dee_dup_v"), "the View's CTE was not added: {sql}");
    }

    /// The subtree is found by the name it was given, and covers the CTE's body
    /// only -- not the consumer that reads it back.
    #[test]
    fn the_cte_region_is_the_body_and_not_the_query_that_reads_it() {
        // The shape DuckDB returns: a CTE node over [body, main query].
        let json = r#"[{"name":"CTE","extra_info":{"CTE Name":"dee_dup_v",
              "Estimated Cardinality":1000},"children":[
            {"name":"HASH_GROUP_BY","extra_info":{"Aggregates":["sum(#1)"],
              "Estimated Cardinality":7},"children":[
              {"name":"SEQ_SCAN","extra_info":{"Table":"t",
                "Estimated Cardinality":1000},"children":[]}]},
            {"name":"HASH_JOIN","extra_info":{"Estimated Cardinality":1000},"children":[
              {"name":"SEQ_SCAN","extra_info":{"Table":"t",
                "Estimated Cardinality":1000},"children":[]},
              {"name":"CTE_SCAN","extra_info":{"Estimated Cardinality":7},
                "children":[]}]}]}]"#;
        let plans = parse_duckdb_plan(json).unwrap();
        let region = find_subplan(&plans, "dee_dup_v").expect("the CTE region");
        assert_eq!(region.operator, "HASH_GROUP_BY");
        // Three operators in the whole plan belong to the consumer; the region
        // must contain only the two that are the View.
        assert_eq!(
            OperatorCountSubtreeCost.cost(std::slice::from_ref(region)),
            Some(2.0)
        );
    }

    #[test]
    fn postgres_names_a_cte_body_the_same_way_duckdb_does() {
        let json = r#"[{"Plan":{"Node Type":"Hash Join","Plan Rows":10,"Plan Width":8,
            "Plans":[
              {"Node Type":"Aggregate","Subplan Name":"CTE dee_dup_v",
               "Parent Relationship":"InitPlan","Plan Rows":7,"Plan Width":12,
               "Plans":[{"Node Type":"Seq Scan","Relation Name":"t",
                         "Plan Rows":1000,"Plan Width":12}]},
              {"Node Type":"CTE Scan","CTE Name":"dee_dup_v","Plan Rows":7,
               "Plan Width":12}]}}]"#;
        let plans = crate::plan::parse_postgres_plan(json).unwrap();
        let region = find_subplan(&plans, "dee_dup_v").expect("the CTE region");
        assert_eq!(region.operator, "Aggregate");
        // The `CTE Scan` that reads the CTE back is not part of it.
        assert_eq!(
            OperatorCountSubtreeCost.cost(std::slice::from_ref(region)),
            Some(2.0)
        );
    }

    /// `Subplan Name` also spells InitPlans and SubPlans, which are not CTEs.
    #[test]
    fn an_initplan_is_not_mistaken_for_a_cte() {
        let json = r#"[{"Plan":{"Node Type":"Result","Plan Rows":1,"Plan Width":8,
            "Plans":[{"Node Type":"Aggregate","Subplan Name":"InitPlan 1 (returns $0)",
                      "Plan Rows":1,"Plan Width":8}]}}]"#;
        let plans = crate::plan::parse_postgres_plan(json).unwrap();
        assert!(find_subplan(&plans, "InitPlan 1").is_none());
        assert!(plans[0].children[0].subplan.is_none());
    }

    #[test]
    fn cardinality_sums_every_operators_estimate() {
        let json = r#"[{"name":"HASH_GROUP_BY","extra_info":{"Estimated Cardinality":7},
            "children":[{"name":"SEQ_SCAN","extra_info":{"Table":"t",
              "Estimated Cardinality":1000},"children":[]}]}]"#;
        let plans = parse_duckdb_plan(json).unwrap();
        assert_eq!(CardinalitySubtreeCost.cost(&plans), Some(1007.0));
    }

    #[test]
    fn a_cost_model_that_can_price_nothing_declines_rather_than_saying_zero() {
        let json = r#"[{"name":"SEQ_SCAN","extra_info":{"Table":"t"},"children":[]}]"#;
        let plans = parse_duckdb_plan(json).unwrap();
        // No cardinality anywhere: rows-based costing has nothing to add up.
        assert_eq!(CardinalitySubtreeCost.cost(&plans), None);
        // An empty learned model prices nothing either.
        let empty = LearnedCostModel::new();
        assert_eq!(LearnedSubtreeCost(&empty).cost(&plans), None);
    }

    // ---------------------------------------------------------------------
    // Against a live engine. The attribution's whole claim is that the engine
    // will tell you what a View costs inside its consumers, so the parts that
    // matter most -- that the rewritten SQL is valid, that DuckDB honours the
    // barrier, that the region comes back where the plan says it is -- cannot
    // be tested against a fabricated plan.
    // ---------------------------------------------------------------------

    async fn engine_with_raw() -> std::sync::Arc<crate::connectors::duckdb::DuckDBConnection> {
        use crate::connectors::duckdb::{DuckDBConfig, DuckDBConnection};
        let conn = DuckDBConnection::new(DuckDBConfig::new_from_path(":memory:".to_string()))
            .await
            .unwrap();
        conn.execute(
            "CREATE TABLE raw AS SELECT i AS id, i % 7 AS g, i * 1.5 AS amt              FROM range(1000) t(i)"
                .to_string(),
        )
        .await
        .unwrap();
        conn
    }

    /// Two consumers, each of which computes the View. The duplicate is what
    /// the second one costs, because only the first would survive
    /// materializing it.
    #[tokio::test]
    async fn two_consumers_are_charged_one_duplicate_copy() {
        let conn = engine_with_raw().await;
        let view_sql = "SELECT g, sum(amt) AS total FROM raw GROUP BY g";
        let dag = dag_of(vec![
            node("v", MaterializeMode::View, &["raw"], view_sql),
            node("m1", MaterializeMode::Table, &["v"], "SELECT count(*) AS n FROM v"),
            node(
                "m2",
                MaterializeMode::Table,
                &["v"],
                "SELECT max(total) AS biggest FROM v",
            ),
        ]);

        let view_plan = conn
            .parse_plan(&conn.explain(view_sql).await.unwrap().unwrap())
            .unwrap();
        let attributed = duplicate_cost(
            conn.as_ref(),
            &dag,
            "v",
            &view_plan,
            &CardinalitySubtreeCost,
        )
        .await
        .expect("the engine planned both consumers");

        assert_eq!(attributed.per_consumer.len(), 2, "{attributed:?}");
        assert!(attributed.own > 0.0, "{attributed:?}");
        // Two consumers that each aggregate the whole View: neither can push
        // anything into it worth speaking of, so the duplication is one build,
        // give or take the column pruning one of them affords. Not exactly one
        // build -- see the module docs on what `MATERIALIZED` does not do.
        let ratio = attributed.duplicate / attributed.own;
        assert!(
            (0.9..1.1).contains(&ratio),
            "two full copies should duplicate about one build, ratio was {ratio}: \
             {attributed:?}"
        );
    }

    /// A View read only once has nothing to deduplicate, and must not be made
    /// to look as though it does.
    #[tokio::test]
    async fn a_single_consumer_duplicates_nothing() {
        let conn = engine_with_raw().await;
        let view_sql = "SELECT g, sum(amt) AS total FROM raw GROUP BY g";
        let dag = dag_of(vec![
            node("v", MaterializeMode::View, &["raw"], view_sql),
            node("m1", MaterializeMode::Table, &["v"], "SELECT count(*) AS n FROM v"),
        ]);
        let view_plan = conn
            .parse_plan(&conn.explain(view_sql).await.unwrap().unwrap())
            .unwrap();
        let attributed =
            duplicate_cost(conn.as_ref(), &dag, "v", &view_plan, &CardinalitySubtreeCost)
                .await
                .expect("the engine planned the one consumer");

        assert_eq!(attributed.per_consumer.len(), 1);
        assert!(
            attributed.duplicate.abs() < 1e-9,
            "one consumer, one build, nothing duplicated: {attributed:?}"
        );
    }

    /// A consumer that reads the View twice pays for it twice, and that is the
    /// case the graph alone cannot see: out-degree counts the edge once.
    #[tokio::test]
    async fn a_self_join_inside_one_consumer_is_still_two_copies() {
        let conn = engine_with_raw().await;
        let view_sql = "SELECT g, sum(amt) AS total FROM raw GROUP BY g";
        let dag = dag_of(vec![
            node("v", MaterializeMode::View, &["raw"], view_sql),
            node(
                "m1",
                MaterializeMode::Table,
                &["v"],
                "SELECT a.g, a.total FROM v a JOIN v b ON b.g = a.g + 1",
            ),
            node("m2", MaterializeMode::Table, &["v"], "SELECT count(*) AS n FROM v"),
        ]);
        let view_plan = conn
            .parse_plan(&conn.explain(view_sql).await.unwrap().unwrap())
            .unwrap();
        let attributed =
            duplicate_cost(conn.as_ref(), &dag, "v", &view_plan, &CardinalitySubtreeCost)
                .await
                .expect("the engine planned both consumers");

        // The self-join reads the CTE twice, but a materialized CTE is
        // *computed* once however often it is read -- that is what
        // materializing it means -- so the duplication is still one copy per
        // consumer beyond the first.
        assert_eq!(attributed.per_consumer.len(), 2, "{attributed:?}");
        assert!(attributed.duplicate > 0.0, "{attributed:?}");
    }

    /// The View sits two Views deep from the only thing that materializes it.
    /// Without contracting the minor there is no consumer that even names it.
    #[tokio::test]
    async fn the_minor_is_contracted_against_a_live_engine_too() {
        let conn = engine_with_raw().await;
        let view_sql = "SELECT g, sum(amt) AS total FROM raw GROUP BY g";
        let dag = dag_of(vec![
            node("v", MaterializeMode::View, &["raw"], view_sql),
            node(
                "mid",
                MaterializeMode::View,
                &["v"],
                "SELECT g, total FROM v WHERE total > 100",
            ),
            node(
                "m1",
                MaterializeMode::Table,
                &["mid"],
                "SELECT count(*) AS n FROM mid",
            ),
            node(
                "m2",
                MaterializeMode::Table,
                &["mid"],
                "SELECT sum(total) AS s FROM mid",
            ),
        ]);
        let view_plan = conn
            .parse_plan(&conn.explain(view_sql).await.unwrap().unwrap())
            .unwrap();
        let attributed =
            duplicate_cost(conn.as_ref(), &dag, "v", &view_plan, &CardinalitySubtreeCost)
                .await
                .expect("the minor put v in reach of both consumers");

        assert_eq!(
            attributed.per_consumer.iter().map(|(c, _)| c.as_str()).collect::<Vec<_>>(),
            vec!["m1", "m2"]
        );
        assert!(attributed.duplicate > 0.0, "{attributed:?}");
    }

    /// A consumer that reads a slice of the View is charged for the slice.
    ///
    /// `MATERIALIZED` is not a full barrier --- DuckDB pushes `g = 3` into the
    /// CTE's own scan --- and this is the test that says so, because the
    /// alternative reading is that something leaked. It has not: what this
    /// consumer pays for the View today *is* the filtered computation, and
    /// charging it a full build would overstate what materializing the View
    /// could save by a factor of seven.
    #[tokio::test]
    async fn a_consumer_that_reads_a_slice_of_the_view_is_charged_for_the_slice() {
        let conn = engine_with_raw().await;
        let view_sql = "SELECT g, sum(amt) AS total FROM raw GROUP BY g";
        let dag = dag_of(vec![
            node("v", MaterializeMode::View, &["raw"], view_sql),
            node("m1", MaterializeMode::Table, &["v"], "SELECT count(*) AS n FROM v"),
            node(
                "m2",
                MaterializeMode::Table,
                &["v"],
                "SELECT total FROM v WHERE g = 3",
            ),
        ]);
        let view_plan = conn
            .parse_plan(&conn.explain(view_sql).await.unwrap().unwrap())
            .unwrap();
        let attributed =
            duplicate_cost(conn.as_ref(), &dag, "v", &view_plan, &CardinalitySubtreeCost)
                .await
                .unwrap();

        // per_consumer is sorted by consumer id: m1 reads all of it, m2 one
        // group of seven.
        let [(_, all), (_, slice)] = attributed.per_consumer.as_slice() else {
            panic!("expected both consumers: {attributed:?}");
        };
        assert!(
            *slice < *all / 2.0,
            "the filtering consumer was charged for the whole View: {attributed:?}"
        );
    }

    /// And when *every* consumer reads a slice, materializing the View costs
    /// more than it saves. The number comes out negative, and must survive as
    /// one: clamped to zero it would read as "no duplication here", which is
    /// a different and much less useful statement than "do not do this".
    #[tokio::test]
    async fn a_view_every_consumer_filters_is_worth_less_than_it_costs() {
        let conn = engine_with_raw().await;
        let view_sql = "SELECT g, sum(amt) AS total FROM raw GROUP BY g";
        let dag = dag_of(vec![
            node("v", MaterializeMode::View, &["raw"], view_sql),
            node(
                "m1",
                MaterializeMode::Table,
                &["v"],
                "SELECT total FROM v WHERE g = 3",
            ),
            node(
                "m2",
                MaterializeMode::Table,
                &["v"],
                "SELECT total FROM v WHERE g = 5",
            ),
        ]);
        let view_plan = conn
            .parse_plan(&conn.explain(view_sql).await.unwrap().unwrap())
            .unwrap();
        let attributed =
            duplicate_cost(conn.as_ref(), &dag, "v", &view_plan, &CardinalitySubtreeCost)
                .await
                .unwrap();

        assert!(
            attributed.duplicate < 0.0,
            "two consumers reading a seventh each cannot justify building all \
             seven: {attributed:?}"
        );
    }

    #[test]
    fn a_subtree_cost_method_round_trips_through_its_name() {
        for method in [
            SubtreeCostMethod::LearnedCost,
            SubtreeCostMethod::Cardinality,
            SubtreeCostMethod::Operators,
        ] {
            assert_eq!(method.as_str().parse::<SubtreeCostMethod>(), Ok(method));
        }
    }
}
