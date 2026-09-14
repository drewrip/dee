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

use std::collections::{HashMap, HashSet};

use log::{debug, warn};
use serde::{Deserialize, Serialize};

use crate::{
    connectors::Connector,
    dag::Dag,
    opt::{
        common::{
            contract_intermediate_views_set, dialect_for_db, rewrite_node_refs,
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

    /// What writing this region's `rows` out as a table costs, on top of
    /// computing it, through the write path named by `path`.
    ///
    /// `path` is how the engine says it would persist this region ---
    /// [`Connector::write_path_for`](crate::connectors::Connector::write_path_for)
    /// --- because an engine can have several at very different cost per byte,
    /// and the region itself is a View that has never been written, so its own
    /// plan names no sink.
    ///
    /// Defaults to `None`: most models price plan operators, and a write
    /// operator's output is a one-row count of what it wrote rather than the
    /// payload, so there is nothing in the plan to fit. `None` means *this
    /// model does not know*, not that writing is free.
    fn write_cost(&self, _path: &str, _roots: &[PlanNode], _rows: f64) -> Option<f64> {
        None
    }

    /// What spooling this region's `rows` into a `MATERIALIZED` CTE costs.
    ///
    /// A CTE spool is a sink of the same shape as a write at a different rate,
    /// so the default is the write cost scaled by `factor` --- see
    /// [`LearnedCostModel::spool_cost`] for why the ratio is a caller's
    /// estimate rather than something fitted. A model that cannot price a
    /// write cannot price a spool either, and says `None` rather than zero.
    fn spool_cost(
        &self,
        path: &str,
        roots: &[PlanNode],
        rows: f64,
        factor: f64,
    ) -> Option<f64> {
        if factor <= 0.0 {
            return Some(0.0);
        }
        Some(self.write_cost(path, roots, rows)? * factor)
    }
}

/// Seconds, via the learned seconds-per-byte constants.
pub struct LearnedSubtreeCost<'a>(pub &'a LearnedCostModel);

impl SubtreeCost for LearnedSubtreeCost<'_> {
    fn cost(&self, roots: &[PlanNode]) -> Option<f64> {
        self.0.cost(roots)
    }

    fn write_cost(&self, path: &str, roots: &[PlanNode], rows: f64) -> Option<f64> {
        self.0.write_cost(path, roots, rows)
    }

    fn spool_cost(&self, path: &str, roots: &[PlanNode], rows: f64, factor: f64) -> Option<f64> {
        self.0.spool_cost(path, roots, rows, factor)
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

/// What building one member of a set costs.
///
/// `compute` and `write` are kept apart because only `compute` is measured the
/// same way the consumer side is, and only `compute` may enter
/// [`DuplicateCost::duplicate`] --- mixing a modelled write into that
/// subtraction would change what the number has always meant.
#[derive(Debug, Clone, PartialEq)]
pub struct MemberBuild {
    pub view: String,
    /// The member's own region in the build probe. Sums to
    /// [`DuplicateCost::build_once`].
    pub compute: f64,
    /// Writing the result out as a table, from the cost model's write constant.
    ///
    /// `None` when the model has no write constant to apply, which is not a
    /// write that is free. Materializing a multi-million-row View is mostly
    /// the write: measured on `p05_hr`, building `stg_employees` takes 2.0s
    /// against 0.17s of modelled compute. A schedule built on the compute
    /// alone is not merely imprecise, it is wrong by an order of magnitude and
    /// wrong in one direction, so callers that need a duration must refuse the
    /// estimate rather than substitute a zero.
    pub write: Option<f64>,
}

/// What a *set* of Views duplicates between them, and the numbers behind it.
///
/// The components are kept rather than just the difference because the
/// difference alone cannot be sanity-checked: a `duplicate` of zero means
/// "one consumer" when there is one entry in `per_consumer` and "the engine
/// planned every copy away" when there are four.
#[derive(Debug, Clone)]
pub struct DuplicateCost {
    /// The members priced, consumer-most first.
    pub views: Vec<String>,
    /// What building every member exactly once costs, with each member reading
    /// the materialized form of its in-set ancestors rather than recomputing
    /// them.
    ///
    /// Not the sum of the members' standalone plans. On a chain -- and HMP's
    /// candidates are routinely all on one -- an upstream member's body is
    /// nested inside a downstream member's, so summing standalone plans counts
    /// the shared work once per member. See [`build_once_cost`].
    pub build_once: f64,
    /// Per-member breakdown of `build_once`, in `views` order.
    ///
    /// The sum of the `compute` terms is exactly `build_once`. Kept because a
    /// set on one chain builds its members *in sequence*, and a schedule needs
    /// the terms rather than the total.
    pub build_per_member: Vec<MemberBuild>,
    /// `(consumer node, what that consumer's plan spends on every member it can
    /// see)`, sorted by consumer for a stable report.
    pub per_consumer: Vec<(String, f64)>,
    /// `(consumer node, what that consumer's *whole* plan costs)`, same order as
    /// `per_consumer`.
    ///
    /// Subtracting `per_consumer` leaves the work that survives materializing
    /// the set --- what the consumer would still do once its members are tables
    /// it merely scans.
    pub consumer_total: Vec<(String, f64)>,
    /// `sum(per_consumer) - build_once`: the computation that would stop
    /// happening if every member were built once.
    ///
    /// Can come out negative, and that is a finding rather than a bug: it says
    /// the engine plans the members more cheaply inside their consumers ---
    /// pushing their predicates and projections into them --- than it would
    /// standing alone, so materializing them would *add* work.
    pub duplicate: f64,
    /// What materializing every member would cost to *write*, summed.
    ///
    /// Deliberately outside [`Self::duplicate`], which stays what it has always
    /// been: a difference between two quantities measured the same way, with no
    /// modelled term in it. The write belongs to the decision rather than to
    /// that subtraction --- materializing a View removes `duplicate` of repeated
    /// work and adds this much new work, so the net change in total query time
    /// is `duplicate - write_total`, and that is what a query-time ranking
    /// should sort on.
    ///
    /// `None` unless every member could be priced. A partial sum would
    /// understate the charge and silently favour the very candidates whose
    /// writes could not be priced, which are the ones most likely to be large.
    pub write_total: Option<f64>,
}

/// The CTE name a View is inlined under.
///
/// Prefixed so it cannot collide with a CTE the consumer already defines, and
/// derived from the *fully qualified* node id so that two members of one set
/// cannot collide with each other. The bare name is not enough: a set holding
/// `"wh"."main"."orders"` and `"wh"."staging"."orders"` would emit two CTEs
/// under one alias, repoint both members' references at it, and then attribute
/// whichever region `find_subplan` reached first to both -- silently, with a
/// plausible-looking number.
fn cte_name(view_id: &str) -> String {
    let mut out = String::from("dee_dup");
    for segment in view_id.split('.') {
        let segment = segment.trim_matches('"');
        if segment.is_empty() {
            continue;
        }
        out.push('_');
        out.push_str(segment);
    }
    out
}

/// The consumer query with every member of `visible` inlined into it as its own
/// materialized CTE, intra-set references repointed at CTE names.
///
/// `dag` must already be the contracted minor around the set --- every member
/// here is referenced directly by the consumer or by another member's body.
///
/// `visible` must be ordered consumer-most first. That is not cosmetic:
/// [`wrap_in_materialized_cte`] *prepends*, so wrapping in this order leaves the
/// CTEs defined producer-first, which is the only order in which a CTE that
/// references another one is legal SQL.
fn consumer_query(
    dag: &Dag,
    consumer: &str,
    visible: &[String],
    bodies: &HashMap<String, String>,
    dialect: polyglot_sql::dialects::DialectType,
) -> Option<String> {
    let node = dag.nodes.get(consumer.to_string())?;

    // One rewrite for every member the consumer names directly.
    let mapping: HashMap<String, String> = visible
        .iter()
        .map(|v| (v.clone(), cte_name(v)))
        .collect();
    let mut sql = rewrite_node_refs(&node.query_text, &mapping, dialect).or_else(|| {
        warn!("dup: could not repoint '{consumer}' at the candidate CTEs; skipping it");
        None
    })?;

    for v in visible {
        let body = bodies.get(v)?;
        sql = wrap_in_materialized_cte(&sql, &cte_name(v), body, dialect).or_else(|| {
            warn!("dup: could not wrap '{v}' into '{consumer}' as a CTE; skipping it");
            None
        })?;
    }
    Some(sql)
}

/// Each member's body as it stands in the contracted minor, with references to
/// its in-set ancestors repointed at their CTE names.
///
/// This is what stops a downstream member from being charged for an upstream
/// one: its body reads `dee_dup_<ancestor>` instead of containing the
/// ancestor's computation, so the two plan regions are disjoint.
fn member_bodies(
    dag: &Dag,
    set: &HashSet<String>,
    dialect: polyglot_sql::dialects::DialectType,
) -> Option<HashMap<String, String>> {
    let mapping: HashMap<String, String> =
        set.iter().map(|v| (v.clone(), cte_name(v))).collect();
    let mut out = HashMap::new();
    for v in set {
        let body = dag.nodes.get(v.clone())?.query_text.clone();
        // A member with no in-set ancestor comes back unchanged, which is what
        // `rewrite_node_refs` does when nothing matches.
        let rewritten = rewrite_node_refs(&body, &mapping, dialect).or_else(|| {
            warn!("dup: could not repoint '{v}'s body at the candidate CTEs");
            None
        })?;
        out.insert(v.clone(), rewritten);
    }
    Some(out)
}

/// What building every member once costs: one probe query, one EXPLAIN.
///
/// The probe defines every member as a materialized CTE and then reads each one
/// exactly once. Reading each one is the part that cannot be skipped --- an
/// unreferenced `MATERIALIZED` CTE is pruned outright, and a probe that only
/// read the last member would come back missing its ancestors' regions, which
/// would quietly inflate every `duplicate` on a chain.
///
/// The `count(*)` wrappers sit outside the tagged region (the tag is the CTE's
/// own root), so they cost nothing here.
async fn build_once_cost<C>(
    conn: &C,
    order: &[String],
    bodies: &HashMap<String, String>,
    coster: &dyn SubtreeCost,
    dialect: polyglot_sql::dialects::DialectType,
) -> Option<Vec<MemberBuild>>
where
    C: Connector + Send + Sync,
{
    let probes: Vec<String> = order
        .iter()
        .enumerate()
        .map(|(i, v)| format!("(SELECT count(*) FROM {}) AS c{i}", cte_name(v)))
        .collect();
    let mut sql = format!("SELECT {}", probes.join(", "));
    for v in order {
        let body = bodies.get(v)?;
        sql = wrap_in_materialized_cte(&sql, &cte_name(v), body, dialect)?;
    }

    let plan_json = match conn.explain(&sql).await {
        Ok(Some(json)) => json,
        Ok(None) => return None,
        Err(e) => {
            warn!("dup: EXPLAIN of the build-once probe failed: {e}");
            return None;
        }
    };
    let plans = conn.parse_plan(&plan_json)?;

    let mut builds = Vec::with_capacity(order.len());
    for v in order {
        let region = find_subplan(&plans, &cte_name(v))?;
        let compute = coster.cost(std::slice::from_ref(region))?;
        // The region's own root cardinality is how many rows the member would
        // write. A model that does not price writes contributes nothing here
        // rather than refusing the whole measurement: `compute` is the part
        // `duplicate` is built from, and it is already in hand.
        // Which sink the engine would give this member, asked of the engine:
        // it decides during planning and names the sink when the statement it
        // is handed is the write rather than the `SELECT` under it, which costs
        // one more EXPLAIN and is exact. The probe plan above cannot answer --
        // it defines each member as a materialized CTE, and a CTE is tagged
        // `CTE`, not a write -- so this is its own call.
        //
        // No fallback. An engine that will not name the path leaves the write
        // unpriced, which is not a write of zero: `makespan::estimate` refuses
        // a build it cannot price rather than charging it as free, and that is
        // the behaviour wanted here. Guessing the path from the shape of the
        // `SELECT` was tried and removed -- a guess that reads like a
        // measurement downstream is worse than no answer.
        let write = match conn.write_path_for(bodies.get(v)?).await {
            Ok(Some(path)) => region
                .rows()
                .and_then(|rows| coster.write_cost(&path, std::slice::from_ref(region), rows)),
            Ok(None) => None,
            Err(e) => {
                warn!("dup: could not ask {} which write path it would use: {e}", v);
                None
            }
        };
        builds.push(MemberBuild {
            view: v.clone(),
            compute,
            write,
        });
    }
    Some(builds)
}

/// Attribute the duplicate computation of a *set* of Views, by planning each of
/// their consumers with every member inlined as its own materialized CTE.
///
/// Returns `None` when nothing could be measured: no member has a materialized
/// consumer, no consumer's plan could be obtained, or the cost model declined to
/// price what came back. The caller must treat that as "unknown" and fall back,
/// never as a duplication of zero.
pub async fn duplicate_cost_set<C>(
    conn: &C,
    dag: &Dag,
    views: &[String],
    coster: &dyn SubtreeCost,
) -> Option<DuplicateCost>
where
    C: Connector + Send + Sync,
{
    if views.is_empty() {
        return None;
    }
    let dialect = dialect_for_db(&dag.db);

    // Consumer-most first, node id as the tiebreak: the order the CTEs have to
    // be wrapped in, and a total order so the report is stable.
    let heights = dag.nodes.heights();
    let mut order: Vec<String> = {
        let mut seen = HashSet::new();
        views.iter().filter(|v| seen.insert((*v).clone())).cloned().collect()
    };
    order.sort_by(|a, b| {
        heights
            .get(a)
            .copied()
            .unwrap_or(0)
            .cmp(&heights.get(b).copied().unwrap_or(0))
            .then_with(|| a.cmp(b))
    });
    let set: HashSet<String> = order.iter().cloned().collect();

    // Two members whose CTE names collide would be measured as one. The name is
    // built from the qualified id precisely so this cannot happen; checked
    // anyway, because the failure is silent rather than loud.
    let mut names = HashSet::new();
    for v in &order {
        if !names.insert(cte_name(v)) {
            warn!("dup: two members of {order:?} share a CTE name; refusing to guess");
            return None;
        }
    }

    // Which consumers pay for which members. `c` is in a member's frontier
    // exactly when that member's body is inlined into the query the engine runs
    // for `c`, which is the definition of "c pays for it".
    let mut visible: HashMap<String, Vec<String>> = HashMap::new();
    for v in &order {
        for c in dag.nodes.frontier_materializes(v) {
            visible.entry(c).or_default().push(v.clone());
        }
    }
    if visible.is_empty() {
        return None;
    }
    // Keep each consumer's member list in the global wrap order.
    for members in visible.values_mut() {
        members.sort_by_key(|m| order.iter().position(|o| o == m).unwrap_or(usize::MAX));
    }

    // The graph minor around the whole set, on a copy, because this is a
    // question about the DAG and not a change to it.
    let mut minor = dag.clone();
    let frontier: HashSet<String> = visible.keys().cloned().collect();
    if let Err(e) = contract_intermediate_views_set(&mut minor, &set, &frontier) {
        warn!("dup: could not contract the graph minor around {order:?}: {e}");
        return None;
    }

    let bodies = member_bodies(&minor, &set, dialect)?;
    let build_per_member = build_once_cost(conn, &order, &bodies, coster, dialect).await?;
    let build_once: f64 = build_per_member.iter().map(|b| b.compute).sum();

    let mut consumers: Vec<String> = frontier.into_iter().collect();
    consumers.sort();

    let mut per_consumer: Vec<(String, f64)> = Vec::new();
    let mut consumer_total: Vec<(String, f64)> = Vec::new();
    for consumer in consumers {
        let members = match visible.get(&consumer) {
            Some(m) => m.clone(),
            None => continue,
        };
        let Some(sql) = consumer_query(&minor, &consumer, &members, &bodies, dialect) else {
            continue;
        };
        let plan_json = match conn.explain(&sql).await {
            Ok(Some(json)) => json,
            Ok(None) => {
                debug!("dup: this connector cannot EXPLAIN, so {order:?} cannot be attributed");
                return None;
            }
            Err(e) => {
                warn!("dup: EXPLAIN of '{consumer}' with {order:?} inlined failed: {e}");
                continue;
            }
        };
        let Some(plans) = conn.parse_plan(&plan_json) else {
            warn!("dup: could not parse the plan of '{consumer}' with {order:?} inlined");
            continue;
        };

        // Every member this consumer can see has to be found, or the number
        // would be a partial sum masquerading as a total. No region means the
        // engine did not keep the CTE as one; charging the consumer for the rest
        // would understate what it pays rather than admit to not knowing.
        let mut member_regions = 0.0;
        let mut complete = true;
        for v in &members {
            let Some(region) = find_subplan(&plans, &cte_name(v)) else {
                warn!(
                    "dup: the plan of '{consumer}' has no region for '{v}'s materialized \
                     CTE; skipping this consumer"
                );
                complete = false;
                break;
            };
            let Some(cost) = coster.cost(std::slice::from_ref(region)) else {
                debug!("dup: the cost model declined to price '{v}' inside '{consumer}'");
                complete = false;
                break;
            };
            member_regions += cost;
        }
        if complete {
            // The whole plan, not just the member regions: what is left after
            // subtracting them is the work this consumer keeps doing once the
            // members are tables, which is what a schedule needs. Priced from
            // the plan already in hand, so it costs no extra EXPLAIN. A model
            // that declines the whole plan falls back to the member regions,
            // leaving a residual of zero rather than dropping the consumer.
            let whole = coster.cost(&plans).unwrap_or(member_regions);
            per_consumer.push((consumer.clone(), member_regions));
            consumer_total.push((consumer, whole.max(member_regions)));
        }
    }

    if per_consumer.is_empty() {
        return None;
    }

    let total: f64 = per_consumer.iter().map(|(_, c)| c).sum();
    let duplicate = total - build_once;
    let write_total = build_per_member
        .iter()
        .map(|b| b.write)
        .sum::<Option<f64>>();
    debug!(
        "dup({order:?}) = {total:.4} - {build_once:.4} = {duplicate:.4} over {} consumer(s)",
        per_consumer.len()
    );
    Some(DuplicateCost {
        views: order,
        build_once,
        build_per_member,
        per_consumer,
        consumer_total,
        duplicate,
        write_total,
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

    /// The single-member spelling of `consumer_query`, for the tests that
    /// predate set costing and are kept as the `k = 1` regression suite.
    fn one_consumer_query(dag: &Dag, view: &str, consumer: &str) -> String {
        let dialect = dialect_for_db(&dag.db);
        let set = HashSet::from([view.to_string()]);
        let bodies = member_bodies(dag, &set, dialect).expect("body should rewrite");
        consumer_query(dag, consumer, &[view.to_string()], &bodies, dialect)
            .expect("consumer should rewrite")
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
        let queries: Vec<(String, String)> = ["m1", "m2"]
            .iter()
            .map(|c| (c.to_string(), one_consumer_query(&dag, "v", c)))
            .collect();
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
        contract_intermediate_views_set(&mut dag, &HashSet::from(["v".to_string()]), &frontier).unwrap();

        let sql = &one_consumer_query(&dag, "v", "m1");
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
        let sql = &one_consumer_query(&dag, "v", "m1");
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

    /// The write is summed for the ranking but stays out of `duplicate`, which
    /// remains a difference between two quantities measured the same way.
    #[tokio::test]
    async fn the_write_is_summed_beside_duplicate_not_inside_it() {
        let conn = engine_with_raw().await;
        let dag = dag_of(vec![
            node("v", MaterializeMode::View, &["raw"], "SELECT g, sum(amt) AS total FROM raw GROUP BY g"),
            node("m1", MaterializeMode::Table, &["v"], "SELECT count(*) AS n FROM v"),
            node("m2", MaterializeMode::Table, &["v"], "SELECT max(total) AS biggest FROM v"),
        ]);
        // `CardinalitySubtreeCost` prices no writes at all, so the total is
        // unknown rather than zero -- and `duplicate` is unaffected either way.
        let d = duplicate_cost_set(conn.as_ref(), &dag, &["v".to_string()], &CardinalitySubtreeCost)
            .await
            .expect("priced");
        assert!(
            d.write_total.is_none(),
            "a model that prices no writes reported a write total"
        );
        assert!(d.duplicate > 0.0, "duplicate still comes from compute alone");
        assert_eq!(d.build_per_member.len(), 1);
        assert!(d.build_per_member[0].write.is_none());
    }

    /// A set is only charged a write total when every member has one. A partial
    /// sum would understate the charge and favour exactly the members whose
    /// writes could not be priced.
    #[test]
    fn a_partly_priced_set_reports_no_write_total() {
        let priced = |w: Option<f64>| MemberBuild {
            view: "v".to_string(),
            compute: 1.0,
            write: w,
        };
        let all: Option<f64> = [priced(Some(1.0)), priced(Some(2.0))]
            .iter()
            .map(|b| b.write)
            .sum();
        assert_eq!(all, Some(3.0));
        let partial: Option<f64> = [priced(Some(1.0)), priced(None)]
            .iter()
            .map(|b| b.write)
            .sum();
        assert_eq!(partial, None, "a partial sum was reported as a total");
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

        let attributed = duplicate_cost_set(
            conn.as_ref(),
            &dag,
            &["v".to_string()],
            &CardinalitySubtreeCost,
        )
        .await
        .expect("the engine planned both consumers");

        assert_eq!(attributed.per_consumer.len(), 2, "{attributed:?}");
        assert!(attributed.build_once > 0.0, "{attributed:?}");
        // Two consumers that each aggregate the whole View: neither can push
        // anything into it worth speaking of, so the duplication is one build,
        // give or take the column pruning one of them affords. Not exactly one
        // build -- see the module docs on what `MATERIALIZED` does not do.
        let ratio = attributed.duplicate / attributed.build_once;
        assert!(
            (0.9..1.1).contains(&ratio),
            "two full copies should duplicate about one build, ratio was {ratio}: \
             {attributed:?}"
        );
    }

    // -----------------------------------------------------------------------
    // Set-level attribution
    // -----------------------------------------------------------------------

    /// The p05 shape: `salary -> profile`, `profile` read by three tables and
    /// `salary` by one more. Both are multi-consumer Views on one chain.
    fn nested_dag() -> Dag {
        dag_of(vec![
            node("salary", MaterializeMode::View, &["raw"], "SELECT id, amt FROM raw WHERE amt > 0"),
            node(
                "profile",
                MaterializeMode::View,
                &["salary"],
                "SELECT id, amt FROM salary WHERE amt > 10",
            ),
            node("t1", MaterializeMode::Table, &["profile"], "SELECT sum(amt) AS s FROM profile"),
            node("t2", MaterializeMode::Table, &["profile"], "SELECT count(*) AS n FROM profile"),
            node("t3", MaterializeMode::Table, &["profile"], "SELECT max(amt) AS x FROM profile"),
            node("t4", MaterializeMode::Table, &["salary"], "SELECT min(amt) AS m FROM salary"),
        ])
    }

    async fn realize(conn: &crate::connectors::duckdb::DuckDBConnection, dag: &Dag, views: &[&str]) {
        for v in views {
            let sql = dag.nodes.get(v.to_string()).unwrap().query_text.clone();
            conn.execute(format!("CREATE VIEW {v} AS {sql}")).await.unwrap();
        }
    }

    /// The claim the set version exists to make. Both members are on one chain,
    /// so `profile`'s body contains `salary`'s; pricing them independently and
    /// adding counts the shared scan twice.
    #[tokio::test]
    async fn a_nested_pair_is_not_charged_twice() {
        let conn = engine_with_raw().await;
        let dag = nested_dag();
        realize(&conn, &dag, &["salary", "profile"]).await;

        async fn one(
            conn: &crate::connectors::duckdb::DuckDBConnection,
            dag: &Dag,
            v: &str,
        ) -> f64 {
            duplicate_cost_set(conn, dag, &[v.to_string()], &CardinalitySubtreeCost)
                .await
                .expect("singleton should price")
                .duplicate
        }
        let salary = one(conn.as_ref(), &dag, "salary").await;
        let profile = one(conn.as_ref(), &dag, "profile").await;

        let pair = duplicate_cost_set(
            conn.as_ref(),
            &dag,
            &["salary".to_string(), "profile".to_string()],
            &CardinalitySubtreeCost,
        )
        .await
        .expect("the pair should price");

        assert!(
            pair.duplicate < salary + profile,
            "the members overlap, so the set must be worth less than their sum: \
             pair={}, salary={salary}, profile={profile}",
            pair.duplicate
        );
        assert_eq!(
            pair.views,
            vec!["profile".to_string(), "salary".to_string()],
            "members come back consumer-most first"
        );
    }

    /// The discount above has to come from overlap, not from the set machinery
    /// shaving numbers in general.
    #[tokio::test]
    async fn an_independent_pair_is_about_additive() {
        let conn = engine_with_raw().await;
        let dag = dag_of(vec![
            node("lo", MaterializeMode::View, &["raw"], "SELECT id, amt FROM raw WHERE amt < 10"),
            node("hi", MaterializeMode::View, &["raw"], "SELECT id, amt FROM raw WHERE amt > 40"),
            node("a1", MaterializeMode::Table, &["lo"], "SELECT sum(amt) AS s FROM lo"),
            node("a2", MaterializeMode::Table, &["lo"], "SELECT count(*) AS n FROM lo"),
            node("a3", MaterializeMode::Table, &["lo"], "SELECT max(amt) AS x FROM lo"),
            node("b1", MaterializeMode::Table, &["hi"], "SELECT sum(amt) AS s FROM hi"),
            node("b2", MaterializeMode::Table, &["hi"], "SELECT count(*) AS n FROM hi"),
            node("b3", MaterializeMode::Table, &["hi"], "SELECT max(amt) AS x FROM hi"),
        ]);
        realize(&conn, &dag, &["lo", "hi"]).await;

        async fn single(
            conn: &crate::connectors::duckdb::DuckDBConnection,
            dag: &Dag,
            v: &str,
        ) -> f64 {
            duplicate_cost_set(conn, dag, &[v.to_string()], &CardinalitySubtreeCost)
                .await
                .expect("singleton should price")
                .duplicate
        }
        let sum = single(conn.as_ref(), &dag, "lo").await
            + single(conn.as_ref(), &dag, "hi").await;
        let pair = duplicate_cost_set(
            conn.as_ref(),
            &dag,
            &["lo".to_string(), "hi".to_string()],
            &CardinalitySubtreeCost,
        )
        .await
        .expect("the pair should price")
        .duplicate;

        let ratio = pair / sum;
        assert!(
            (0.95..1.05).contains(&ratio),
            "nothing is shared between these two, so the set should be about the \
             sum: pair={pair}, sum={sum}, ratio={ratio}"
        );
    }

    /// Every member the consumer can see gets its own region, and the regions
    /// are disjoint -- which is what makes summing them legitimate.
    #[tokio::test]
    async fn every_member_gets_its_own_disjoint_region() {
        let conn = engine_with_raw().await;
        let dag = nested_dag();
        realize(&conn, &dag, &["salary", "profile"]).await;
        let dialect = dialect_for_db(&dag.db);

        let set = HashSet::from(["salary".to_string(), "profile".to_string()]);
        let mut minor = dag.clone();
        let frontier = HashSet::from(["t1".to_string()]);
        contract_intermediate_views_set(&mut minor, &set, &frontier).unwrap();
        let bodies = member_bodies(&minor, &set, dialect).unwrap();
        let sql = consumer_query(
            &minor,
            "t1",
            &["profile".to_string(), "salary".to_string()],
            &bodies,
            dialect,
        )
        .unwrap();

        let plans = conn
            .parse_plan(&conn.explain(&sql).await.unwrap().unwrap())
            .unwrap();
        let salary = find_subplan(&plans, &cte_name("salary")).expect("salary region");
        let profile = find_subplan(&plans, &cte_name("profile")).expect("profile region");

        // Disjoint: neither region contains the other.
        assert!(
            salary.find_subplan(&cte_name("profile")).is_none(),
            "salary's region swallowed profile's, so summing would double-count"
        );
        assert!(
            profile.find_subplan(&cte_name("salary")).is_none(),
            "profile's region swallowed salary's, so summing would double-count"
        );
        // And the downstream member reads the CTE rather than the base table.
        let mut ops = Vec::new();
        fn collect(n: &PlanNode, out: &mut Vec<String>) {
            out.push(n.operator.to_uppercase());
            for c in &n.children {
                collect(c, out);
            }
        }
        collect(profile, &mut ops);
        assert!(
            ops.iter().any(|o| o.contains("CTE_SCAN")),
            "profile's region should read salary's CTE, not recompute it: {ops:?}"
        );
    }

    /// The CTE order is implicit in `wrap_in_materialized_cte` prepending, and
    /// the engine rejects a forward reference -- so pin it.
    #[tokio::test]
    async fn ctes_are_emitted_in_dependency_order() {
        let conn = engine_with_raw().await;
        let dag = nested_dag();
        realize(&conn, &dag, &["salary", "profile"]).await;
        let dialect = dialect_for_db(&dag.db);

        let set = HashSet::from(["salary".to_string(), "profile".to_string()]);
        let bodies = member_bodies(&dag, &set, dialect).unwrap();
        let sql = consumer_query(
            &dag,
            "t1",
            &["profile".to_string(), "salary".to_string()],
            &bodies,
            dialect,
        )
        .unwrap();

        let salary_at = sql.find(&cte_name("salary")).expect("salary CTE");
        let profile_at = sql.find(&cte_name("profile")).expect("profile CTE");
        assert!(
            salary_at < profile_at,
            "the ancestor's CTE must be defined first or this is a forward \
             reference: {sql}"
        );
        assert!(
            conn.explain(&sql).await.unwrap().is_some(),
            "the engine should accept the generated SQL: {sql}"
        );
    }

    /// An unreferenced MATERIALIZED CTE is pruned outright, so a probe that read
    /// only its last member would silently lose every other member's build cost
    /// and inflate the duplication by exactly that much.
    ///
    /// Two members that share nothing make this checkable: built once each,
    /// their costs must add, and a pruned member would show up as a shortfall.
    #[tokio::test]
    async fn the_build_once_probe_prices_every_member() {
        let conn = engine_with_raw().await;
        let lo = "SELECT id, amt FROM raw WHERE amt < 10";
        let hi = "SELECT id, amt FROM raw WHERE amt > 40";
        let dialect = dialect_for_db("duckdb");
        let bodies = HashMap::from([
            ("lo".to_string(), lo.to_string()),
            ("hi".to_string(), hi.to_string()),
        ]);

        let probe = |members: Vec<String>| {
            let bodies = bodies.clone();
            let conn = conn.clone();
            async move {
                build_once_cost(
                    conn.as_ref(),
                    &members,
                    &bodies,
                    &CardinalitySubtreeCost,
                    dialect,
                )
                .await
                .expect("the probe should price")
                .iter()
                .map(|b| b.compute)
                .sum::<f64>()
            }
        };

        let both = probe(vec!["lo".to_string(), "hi".to_string()]).await;
        let lo_only = probe(vec!["lo".to_string()]).await;
        let hi_only = probe(vec!["hi".to_string()]).await;

        assert!(
            (both - (lo_only + hi_only)).abs() < 1e-6,
            "both members must be priced, so an independent pair's build cost \
             adds: both={both}, lo={lo_only}, hi={hi_only}"
        );
    }

    /// The `count(*)` the probe wraps each member in must not enter its region.
    #[tokio::test]
    async fn the_probe_region_excludes_the_counting_wrapper() {
        let conn = engine_with_raw().await;
        let view_sql = "SELECT g, sum(amt) AS total FROM raw GROUP BY g";
        let dag = dag_of(vec![node("v", MaterializeMode::View, &["raw"], view_sql)]);
        let dialect = dialect_for_db(&dag.db);
        let bodies = HashMap::from([("v".to_string(), view_sql.to_string())]);

        let probed = build_once_cost(
            conn.as_ref(),
            &["v".to_string()],
            &bodies,
            &OperatorCountSubtreeCost,
            dialect,
        )
        .await
        .expect("the probe should price")
        .iter()
        .map(|b| b.compute)
        .sum::<f64>();

        let standalone = OperatorCountSubtreeCost
            .cost(
                &conn
                    .parse_plan(&conn.explain(view_sql).await.unwrap().unwrap())
                    .unwrap(),
            )
            .unwrap();

        assert_eq!(
            probed, standalone,
            "the probe region should be the view's own plan and nothing else"
        );
    }

    /// The per-member breakdown is the same measurement `build_once` always
    /// was, just not yet added up.
    #[tokio::test]
    async fn build_per_member_sums_to_build_once() {
        let conn = engine_with_raw().await;
        let dag = nested_dag();
        realize(&conn, &dag, &["salary", "profile"]).await;

        let d = duplicate_cost_set(
            conn.as_ref(),
            &dag,
            &["salary".to_string(), "profile".to_string()],
            &CardinalitySubtreeCost,
        )
        .await
        .expect("the nested pair should price");

        assert_eq!(d.build_per_member.len(), d.views.len());
        let summed: f64 = d.build_per_member.iter().map(|b| b.compute).sum();
        assert!(
            (summed - d.build_once).abs() < 1e-6,
            "build_once must stay exactly the sum of the compute terms: \
             {summed} vs {}",
            d.build_once
        );
    }

    /// The residual a schedule is built from --- the whole plan minus the member
    /// regions --- must never come out negative.
    #[tokio::test]
    async fn the_consumer_total_covers_its_member_regions() {
        let conn = engine_with_raw().await;
        let dag = nested_dag();
        realize(&conn, &dag, &["salary", "profile"]).await;

        let d = duplicate_cost_set(
            conn.as_ref(),
            &dag,
            &["salary".to_string(), "profile".to_string()],
            &CardinalitySubtreeCost,
        )
        .await
        .expect("the nested pair should price");

        assert_eq!(d.consumer_total.len(), d.per_consumer.len());
        for ((c1, whole), (c2, regions)) in d.consumer_total.iter().zip(&d.per_consumer) {
            assert_eq!(c1, c2, "the two lists must stay in step");
            assert!(
                whole >= regions,
                "'{c1}' spends {regions} on its members out of a {whole} plan"
            );
        }
    }

    /// Two members whose bare names collide must not be measured as one.
    #[test]
    fn members_in_different_schemas_get_distinct_cte_names() {
        assert_ne!(
            cte_name("\"wh\".\"main\".\"orders\""),
            cte_name("\"wh\".\"staging\".\"orders\""),
            "a bare-name CTE would make these one region and misattribute both"
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
        let attributed =
            duplicate_cost_set(conn.as_ref(), &dag, &["v".to_string()], &CardinalitySubtreeCost)
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
        let attributed =
            duplicate_cost_set(conn.as_ref(), &dag, &["v".to_string()], &CardinalitySubtreeCost)
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
        let attributed =
            duplicate_cost_set(conn.as_ref(), &dag, &["v".to_string()], &CardinalitySubtreeCost)
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
        let attributed =
            duplicate_cost_set(conn.as_ref(), &dag, &["v".to_string()], &CardinalitySubtreeCost)
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
        let attributed =
            duplicate_cost_set(conn.as_ref(), &dag, &["v".to_string()], &CardinalitySubtreeCost)
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
