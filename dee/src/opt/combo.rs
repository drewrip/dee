//! Costing and enumeration of *combinations* of Views to materialize.
//!
//! HMP's search used to decide what order to try candidates in from singleton
//! scores alone: a combination was never priced before it was executed, so the
//! pass found out what a pair was worth by spending a whole DAG run on it. With
//! a run budget in the single digits that is most of the budget gone before the
//! search has learned anything.
//!
//! This module prices combinations up front, by asking
//! [`duplicate_cost_set`](crate::opt::dup::duplicate_cost_set) what each one
//! removes, so the run budget is spent on the candidates most likely to pay for
//! themselves.
//!
//! # Why a set is not the sum of its members
//!
//! Costs do not add, and on real DAGs they do not come close. HMP's candidates
//! are Views with more than one consumer, and those tend to sit on a single
//! chain: `stg_employees -> current_salary -> employee_profile -> ...`. Each
//! member's body is textually nested inside the next, so pricing them
//! independently and adding counts the same base scans once per member. A
//! five-node chain then scores about five times a singleton for work that is
//! largely the *same work*, and the search spends its entire budget on the
//! largest sets it can build.
//!
//! That is not a bias to correct with a normalizing factor. It is a wrong
//! measurement, and [`crate::opt::dup`] fixes it by measuring the set as a set:
//! one graph minor for the whole candidate, every member inlined into its
//! consumers as its own materialized CTE, so a downstream member's plan region
//! reads its ancestor's CTE instead of recomputing it. Overlap stops being paid
//! for twice because the engine's own plan stops doing it twice.
//!
//! This module is therefore only the *search*: which combinations to price, in
//! what order, and when to stop. The arithmetic lives in `dup`.

use std::collections::{BinaryHeap, HashMap, HashSet};

use log::debug;

use crate::{
    connectors::Connector,
    dag::Dag,
    executor::ExecStats,
    opt::dup::{SubtreeCost, duplicate_cost_set},
    opt::hmp::HmpObjective,
    opt::makespan,
};

/// A candidate combination and what materializing all of it is estimated to
/// remove.
#[derive(Debug, Clone, PartialEq)]
pub struct CostedCombo {
    /// Members, in ascending-height (consumer-most-first) order.
    pub combo: Vec<String>,
    /// Duplicate computation this combination removes, measured as a set.
    pub cost: f64,
    /// Predicted wall clock of the DAG with this combination materialized, in
    /// seconds.
    ///
    /// `None` when there was no baseline run to calibrate against. A second
    /// axis rather than a correction to `cost`: duplicated work is a sum and
    /// makespan is a longest path, and no reweighting of the one produces the
    /// other. See [`crate::opt::makespan`].
    pub makespan_s: Option<f64>,
    /// How many members lie on one dependency chain, and so have to build one
    /// after another.
    pub stages: usize,
}

/// Everything one combination was measured at.
#[derive(Debug, Clone, Copy)]
struct Scored {
    cost: f64,
    makespan_s: Option<f64>,
    stages: usize,
}

/// All combinations of `items` of size `k`, in the order `items` is given.
pub(crate) fn combinations(items: &[String], k: usize) -> Vec<Vec<String>> {
    fn helper(
        items: &[String],
        start: usize,
        k: usize,
        combo: &mut Vec<String>,
        out: &mut Vec<Vec<String>>,
    ) {
        if combo.len() == k {
            out.push(combo.clone());
            return;
        }
        for i in start..items.len() {
            combo.push(items[i].clone());
            helper(items, i + 1, k, combo, out);
            combo.pop();
        }
    }

    let mut out = Vec::new();
    if k == 0 || k > items.len() {
        return out;
    }
    let mut combo = Vec::with_capacity(k);
    helper(items, 0, k, &mut combo, &mut out);
    out
}

/// Order `candidates` consumer-most first: ascending height, node id as the
/// tiebreak so the order is total and stable across runs.
///
/// This is both the order a chain prices its members in and the index order the
/// enumeration extends sets by -- deliberately the same order, so that
/// extending a set always appends to its chain. See the module docs.
pub fn canonical_order(dag: &Dag, candidates: &[String]) -> Vec<String> {
    let heights = dag.nodes.heights();
    let mut ordered = candidates.to_vec();
    ordered.sort_by(|a, b| {
        let ha = heights.get(a).copied().unwrap_or(0);
        let hb = heights.get(b).copied().unwrap_or(0);
        ha.cmp(&hb).then_with(|| a.cmp(b))
    });
    ordered
}

/// Cost up to `budget` combinations of `candidates` and return them sorted by
/// the duplicate computation each removes, descending.
///
/// `candidates` must already be in [`canonical_order`]. `singletons` seeds the
/// search's priority queue.
///
/// When the number of multi-node combinations fits inside `budget` they are all
/// costed. Otherwise the search is lazy and best-first: it always expands the
/// combination with the highest *estimated* cost, where the estimate is the
/// measured cost of the set plus the singleton cost of the node being added.
///
/// That estimate is optimistic, not admissible, and the code should not pretend
/// otherwise -- adding a member usually only reduces what the others remove, but
/// nothing guarantees it. This is a good expansion heuristic rather than an A*
/// guarantee, and it costs coverage when it is wrong, never correctness:
/// everything returned is sorted by its own exact measured cost.
///
/// Budget counts combinations costed, and charges only sizes of two or more --
/// the singletons were priced by the ranking pass that produced `singletons`.
///
/// # One objective, chosen by the caller
///
/// Each combination is scored on both the duplicate computation it removes and
/// the wall clock it is predicted to take, and those disagree: on `p05_hr`
/// every candidate that cut total query time raised makespan. `objective` says
/// which of the two the resulting order is sorted on --- see [`HmpObjective`].
///
/// A Pareto frontier over both was tried and taken back out. Computing it from
/// the measured numbers of the eleven candidates p05 actually trialled yields a
/// *single* point: one combination was best on both axes at once and dominated
/// the other ten outright. A frontier that emits one candidate and then falls
/// back to a sort is a sort.
///
/// Both scores are still recorded on every [`CostedCombo`], whichever one the
/// order used. They are what the report scores a prediction against.
///
/// `baseline` supplies the measured per-node durations the makespan prediction
/// is anchored to. Without it nothing is ranked on that axis, and an objective
/// of [`HmpObjective::Makespan`] has nothing to sort by --- see `order_by`.
// One more than clippy's threshold. Every argument is a distinct input to the
// search and bundling them into a struct would only move the list.
#[allow(clippy::too_many_arguments)]
pub async fn search_combos<C>(
    conn: &C,
    base: &Dag,
    candidates: &[String],
    singletons: &HashMap<String, f64>,
    coster: &dyn SubtreeCost,
    budget: usize,
    baseline: Option<&ExecStats>,
    objective: HmpObjective,
) -> Vec<CostedCombo>
where
    C: Connector + Send + Sync,
{
    let n = candidates.len();
    if n == 0 {
        return Vec::new();
    }

    let mut memo: HashMap<Vec<usize>, Option<Scored>> = HashMap::new();
    let mut costed: Vec<(Vec<usize>, Scored)> = Vec::new();
    let mut priced_anything = false;
    let mut spent = 0usize;

    // One call per combination, whatever its size: `duplicate_cost_set` prices
    // the whole set against one graph minor.
    #[allow(clippy::too_many_arguments)]
    async fn cost_of<C>(
        conn: &C,
        base: &Dag,
        idx: &[usize],
        candidates: &[String],
        coster: &dyn SubtreeCost,
        baseline: Option<&ExecStats>,
        memo: &mut HashMap<Vec<usize>, Option<Scored>>,
    ) -> Option<Scored>
    where
        C: Connector + Send + Sync,
    {
        if let Some(hit) = memo.get(idx) {
            return *hit;
        }
        let members: Vec<String> = idx.iter().map(|&i| candidates[i].clone()).collect();
        // One measurement, three readings: `duplicate` is the repeated work this
        // set removes, `write_total` is the new work materializing it adds, and
        // the per-member and per-consumer terms behind them are what the
        // schedule is built from. No extra EXPLAIN is issued for the others.
        let priced = duplicate_cost_set(conn, base, &members, coster)
            .await
            .map(|d| {
                let est = baseline.and_then(|stats| makespan::estimate(base, &d, stats));
                Scored {
                    // Net work removed, not gross. Materializing a View stops
                    // its consumers recomputing it and starts paying to write
                    // it out, and a ranking that counts only the first half
                    // over-values exactly the candidates that are expensive to
                    // write -- which are the large ones. Measured on `p05_hr`:
                    // the set HMP promoted removed 1897ms of work net, while
                    // the four tables it materialized cost 1425ms to write, so
                    // the gross figure overstated the result by 43%.
                    //
                    // An unpriced write charges nothing, which is the behaviour
                    // this replaced. It is reached only when the model has no
                    // write constant at all, since `write_rate_for` will
                    // otherwise substitute another path's and say so.
                    cost: d.duplicate - d.write_total.unwrap_or(0.0),
                    makespan_s: est.as_ref().map(|e| e.makespan_s),
                    stages: est.as_ref().map(|e| e.stages).unwrap_or(0),
                }
            });
        memo.insert(idx.to_vec(), priced);
        priced
    }

    // Singletons are free: they are what `singletons` was measured from. They
    // are still re-costed here so that every entry in the result was produced by
    // the same method -- the ranking that supplied `singletons` may have used a
    // different cost method entirely, and mixing units would make the sort
    // meaningless.
    for i in 0..n {
        if let Some(scored) =
            cost_of(conn, base, &[i], candidates, coster, baseline, &mut memo).await
        {
            priced_anything = true;
            costed.push((vec![i], scored));
        }
    }

    // `2^n - 1 - n` multi-node combinations, saturating rather than shifting
    // past the width of the type.
    let multi_total = if n >= 63 {
        usize::MAX
    } else {
        (1usize << n) - 1 - n
    };

    if multi_total <= budget {
        // Small enough to price exhaustively; no need to guess an order.
        for k in 2..=n {
            for combo in combinations(candidates, k) {
                let idx: Vec<usize> = combo
                    .iter()
                    .filter_map(|c| candidates.iter().position(|x| x == c))
                    .collect();
                if let Some(scored) =
                    cost_of(conn, base, &idx, candidates, coster, baseline, &mut memo).await
                {
                    priced_anything = true;
                    costed.push((idx, scored));
                }
                spent += 1;
            }
        }
    } else {
        let mut heap: BinaryHeap<Entry> = BinaryHeap::new();
        for i in 0..n {
            let seed = singletons.get(&candidates[i]).copied().unwrap_or(0.0);
            for (j, cand) in candidates.iter().enumerate().skip(i + 1) {
                let add = singletons.get(cand).copied().unwrap_or(0.0);
                heap.push(Entry::new(vec![i, j], seed + add));
            }
        }

        let mut seen: HashSet<Vec<usize>> = HashSet::new();
        while spent < budget {
            let Some(entry) = heap.pop() else { break };
            if !seen.insert(entry.idx.clone()) {
                continue;
            }
            let priced =
                cost_of(conn, base, &entry.idx, candidates, coster, baseline, &mut memo).await;
            spent += 1;
            let Some(scored) = priced else { continue };
            priced_anything = true;
            costed.push((entry.idx.clone(), scored));
            let cost = scored.cost;

            // Extend only by a node of higher index. Every set is then
            // generated exactly once, by its unique increasing enumeration.
            let last = *entry.idx.last().expect("combinations are never empty");
            for (j, cand) in candidates.iter().enumerate().skip(last + 1) {
                let add = singletons.get(cand).copied().unwrap_or(0.0);
                let mut next = entry.idx.clone();
                next.push(j);
                heap.push(Entry::new(next, cost + add));
            }
        }
    }

    if !priced_anything {
        // Nothing could be measured -- the connector cannot EXPLAIN, or no
        // plan was priceable. Returning an empty list here would read as
        // "nothing worth doing" and stop the pass dead; the caller has to be
        // able to tell that apart and fall back.
        debug!("combo: nothing could be priced; returning no candidates");
        return Vec::new();
    }

    let kept: Vec<CostedCombo> = costed
        .into_iter()
        // A combination that removes nothing is not worth a DAG run. Mirrors
        // the same filter the singleton ranking applies.
        .filter(|(_, s)| s.cost > 0.0)
        .map(|(idx, s)| CostedCombo {
            combo: idx.iter().map(|&i| candidates[i].clone()).collect(),
            cost: s.cost,
            makespan_s: s.makespan_s,
            stages: s.stages,
        })
        .collect();

    let out = order_by(kept, objective);
    debug!(
        "combo: costed {} combination(s) beyond the singletons, kept {}",
        spent,
        out.len()
    );
    out
}

/// Sort the priced combinations by whichever objective the caller picked.
///
/// The two keys are not interchangeable and neither is a proxy for the other:
/// duplicate computation is a sum over operators, predicted makespan is a
/// longest path, and a set's chain depth --- the thing that drives its
/// makespan --- is invisible to a sum. On p05 the candidates separated perfectly
/// by chain depth (three stages 4772-5506 ms, four stages 6676-7186 ms, no
/// overlap) while set size explained almost nothing.
fn order_by(kept: Vec<CostedCombo>, objective: HmpObjective) -> Vec<CostedCombo> {
    let by_cost = |a: &CostedCombo, b: &CostedCombo| {
        b.cost
            .partial_cmp(&a.cost)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.combo.cmp(&b.combo))
    };
    let mut out = kept;
    match objective {
        HmpObjective::QueryTime => out.sort_by(by_cost),
        // Cheapest predicted wall clock first. A combination with no prediction
        // sorts after every one that has a prediction, ordered among its peers
        // by work removed: it is unranked on this axis, and an absence must not
        // read as a zero and take the front of the queue.
        HmpObjective::Makespan => out.sort_by(|a, b| match (a.makespan_s, b.makespan_s) {
            (Some(x), Some(y)) => x
                .partial_cmp(&y)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| by_cost(a, b)),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => by_cost(a, b),
        }),
    }
    out
}

/// A priority-queue entry ordered by estimated cost, descending.
///
/// `f64` is not `Ord`, and the estimate only ever needs to order entries, so it
/// is held as a scaled integer rather than dragging in a total-order float
/// wrapper for one comparison.
#[derive(Debug, PartialEq, Eq)]
struct Entry {
    key: i64,
    idx: Vec<usize>,
}

impl Entry {
    fn new(idx: Vec<usize>, estimate: f64) -> Self {
        let scaled = (estimate * 1e6).clamp(i64::MIN as f64, i64::MAX as f64);
        Self {
            key: scaled as i64,
            idx,
        }
    }
}

impl Ord for Entry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // `BinaryHeap` is a max-heap and the highest estimate is what should pop
        // first, so this is a plain comparison. The index tiebreak keeps the
        // order total, which keeps the search deterministic.
        self.key
            .cmp(&other.key)
            .then_with(|| other.idx.cmp(&self.idx))
    }
}

impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        connectors::duckdb::{DuckDBConfig, DuckDBConnection},
        dag::{MaterializeMode, TransformNode},
        graph::Graph,
        opt::dup::CardinalitySubtreeCost,
    };
    use std::sync::Arc;

    async fn in_memory_conn() -> Arc<DuckDBConnection> {
        let config = DuckDBConfig::new_from_path(":memory:".to_string());
        DuckDBConnection::new(config).await.unwrap()
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

    /// The p05 shape that motivated set costing: a chain of multi-consumer
    /// Views, each nested inside the next.
    fn nested_dag() -> Dag {
        dag_of(vec![
            node("salary", MaterializeMode::View, &[], "SELECT id, amt FROM raw WHERE amt > 0"),
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

    async fn realize(conn: &DuckDBConnection, dag: &Dag, views: &[&str]) {
        conn.execute(
            "CREATE TABLE raw AS SELECT range AS id, range % 50 AS amt FROM range(200)".to_string(),
        )
        .await
        .unwrap();
        for v in views {
            let sql = dag.nodes.get(v.to_string()).unwrap().query_text.clone();
            conn.execute(format!("CREATE VIEW {v} AS {sql}")).await.unwrap();
        }
    }

    #[test]
    fn canonical_order_is_consumer_most_first() {
        let dag = nested_dag();
        let order = canonical_order(&dag, &["salary".to_string(), "profile".to_string()]);
        assert_eq!(
            order,
            vec!["profile".to_string(), "salary".to_string()],
            "the descendant must come first, so the prepending CTE wrap leaves \
             the ancestor defined first"
        );
    }

    /// Ties are broken by id, so two runs over the same DAG agree.
    #[test]
    fn canonical_order_is_deterministic_under_ties() {
        let dag = dag_of(vec![
            node("b", MaterializeMode::View, &[], "SELECT 1"),
            node("a", MaterializeMode::View, &[], "SELECT 1"),
            node("c", MaterializeMode::View, &[], "SELECT 1"),
        ]);
        let ids = ["c".to_string(), "a".to_string(), "b".to_string()];
        assert_eq!(canonical_order(&dag, &ids), vec!["a", "b", "c"]);
    }

    fn combo_of(name: &str, cost: f64, makespan_s: Option<f64>) -> CostedCombo {
        CostedCombo {
            combo: vec![name.to_string()],
            cost,
            makespan_s,
            stages: 1,
        }
    }

    /// Under `query_time` the order is the duplicate-computation ranking, and
    /// the makespan predictions are along for the ride.
    #[test]
    fn query_time_orders_by_duplicate_computation_descending() {
        let out = order_by(
            vec![
                combo_of("small", 1.0, Some(1.0)),
                combo_of("big", 9.0, Some(9.0)),
                combo_of("mid", 5.0, Some(5.0)),
            ],
            HmpObjective::QueryTime,
        );
        let names: Vec<&str> = out.iter().map(|c| c.combo[0].as_str()).collect();
        assert_eq!(names, vec!["big", "mid", "small"]);
    }

    /// Under `makespan` the same three candidates come back in the opposite
    /// order --- which is the whole reason the objective is a setting and not a
    /// weighting.
    #[test]
    fn makespan_orders_by_predicted_wall_clock_ascending() {
        let out = order_by(
            vec![
                combo_of("small", 1.0, Some(1.0)),
                combo_of("big", 9.0, Some(9.0)),
                combo_of("mid", 5.0, Some(5.0)),
            ],
            HmpObjective::Makespan,
        );
        let names: Vec<&str> = out.iter().map(|c| c.combo[0].as_str()).collect();
        assert_eq!(names, vec!["small", "mid", "big"]);
    }

    /// An unpriced second axis is an absence, not a zero. Sorted as a zero it
    /// would take the front of the queue and spend the trial budget on the
    /// candidates least is known about.
    #[test]
    fn a_candidate_with_no_prediction_sorts_last_under_makespan() {
        let out = order_by(
            vec![
                combo_of("unpredicted", 99.0, None),
                combo_of("slow", 10.0, Some(9.0)),
                combo_of("fast", 1.0, Some(1.0)),
            ],
            HmpObjective::Makespan,
        );
        let names: Vec<&str> = out.iter().map(|c| c.combo[0].as_str()).collect();
        assert_eq!(names, vec!["fast", "slow", "unpredicted"]);
        assert_eq!(names.len(), 3, "nothing is dropped for being unpredicted");
    }

    /// With nothing predicted at all, `makespan` has no axis to sort on and
    /// falls back to work removed rather than to insertion order.
    #[test]
    fn with_no_predictions_makespan_falls_back_to_duplicate_computation() {
        let out = order_by(
            vec![
                combo_of("small", 1.0, None),
                combo_of("big", 9.0, None),
                combo_of("mid", 5.0, None),
            ],
            HmpObjective::Makespan,
        );
        let names: Vec<&str> = out.iter().map(|c| c.combo[0].as_str()).collect();
        assert_eq!(names, vec!["big", "mid", "small"]);
    }

    /// The regression the whole change exists to prevent: a nested pair must
    /// not outrank a singleton merely by being a superset of it.
    #[tokio::test]
    async fn a_nested_pair_does_not_outrank_its_members_by_being_a_superset() {
        let conn = in_memory_conn().await;
        let dag = nested_dag();
        realize(&conn, &dag, &["salary", "profile"]).await;
        let order = canonical_order(&dag, &["salary".to_string(), "profile".to_string()]);
        let singletons =
            HashMap::from([("salary".to_string(), 1.0), ("profile".to_string(), 1.0)]);

        let out = search_combos(
            &*conn,
            &dag,
            &order,
            &singletons,
            &CardinalitySubtreeCost,
            32,
            None,
            HmpObjective::QueryTime,
        )
        .await;

        let cost_of = |members: &[&str]| -> Option<f64> {
            out.iter()
                .find(|c| {
                    c.combo.len() == members.len()
                        && members.iter().all(|m| c.combo.iter().any(|x| x == m))
                })
                .map(|c| c.cost)
        };
        let pair = cost_of(&["salary", "profile"]).expect("the pair should have priced");
        let salary = cost_of(&["salary"]).expect("salary should have priced");
        let profile = cost_of(&["profile"]).expect("profile should have priced");

        assert!(
            pair < salary + profile,
            "the pair overlaps, so it must be worth less than the sum of its \
             members: pair={pair}, salary={salary}, profile={profile}"
        );
        // Descending by cost, so the ordering the search hands HMP is sound.
        for w in out.windows(2) {
            assert!(w[0].cost >= w[1].cost, "not sorted descending: {out:?}");
        }
    }

    /// Below the budget every multi-node combination is priced.
    #[tokio::test]
    async fn exhaustive_below_budget() {
        let conn = in_memory_conn().await;
        let dag = nested_dag();
        realize(&conn, &dag, &["salary", "profile"]).await;
        let order = canonical_order(&dag, &["salary".to_string(), "profile".to_string()]);
        let singletons =
            HashMap::from([("salary".to_string(), 1.0), ("profile".to_string(), 1.0)]);

        let out = search_combos(
            &*conn,
            &dag,
            &order,
            &singletons,
            &CardinalitySubtreeCost,
            32,
            None,
            HmpObjective::QueryTime,
        )
        .await;

        assert!(
            out.iter().any(|c| c.combo.len() == 2),
            "the only pair should have been costed: {out:?}"
        );
    }

    /// Above the budget the search is lazy, and must stop at it.
    #[tokio::test]
    async fn best_first_above_budget_stops_at_the_budget() {
        let conn = in_memory_conn().await;

        // Six independent Views, each read three times: 2^6 - 1 - 6 = 57
        // multi-node combinations, well above a budget of 4.
        let mut nodes = Vec::new();
        let mut views = Vec::new();
        for i in 0..6 {
            let v = format!("v{i}");
            nodes.push(node(
                &v,
                MaterializeMode::View,
                &[],
                &format!("SELECT id, amt FROM raw WHERE amt > {i}"),
            ));
            for suffix in ["a", "b", "c"] {
                nodes.push(node(
                    &format!("t{i}{suffix}"),
                    MaterializeMode::Table,
                    &[&v],
                    &format!("SELECT count(*) AS n_{suffix} FROM {v}"),
                ));
            }
            views.push(v);
        }
        let dag = dag_of(nodes);
        let refs: Vec<&str> = views.iter().map(|v| v.as_str()).collect();
        realize(&conn, &dag, &refs).await;
        let order = canonical_order(&dag, &views);
        let singletons: HashMap<String, f64> = views
            .iter()
            .enumerate()
            .map(|(i, v)| (v.clone(), (10 - i) as f64))
            .collect();

        let out =
            search_combos(&*conn, &dag, &order, &singletons, &CardinalitySubtreeCost, 4, None, HmpObjective::QueryTime).await;

        let multi = out.iter().filter(|c| c.combo.len() >= 2).count();
        assert!(
            multi <= 4,
            "the budget is 4 multi-node combinations, priced {multi}: {out:?}"
        );
        assert!(
            out.iter().any(|c| c.combo.len() == 1),
            "singletons are costed outside the budget and must still be offered"
        );
    }

    /// The index-extension rule is what makes the enumeration cheap; if it ever
    /// generates a set twice the budget is being spent on duplicates.
    #[test]
    fn every_subset_is_generated_at_most_once() {
        let items: Vec<String> = (0..5).map(|i| format!("n{i}")).collect();
        let mut seen = HashSet::new();
        let mut total = 0;
        for k in 1..=items.len() {
            for combo in combinations(&items, k) {
                assert!(seen.insert(combo.clone()), "generated {combo:?} twice");
                total += 1;
            }
        }
        assert_eq!(total, (1 << items.len()) - 1, "should be every non-empty subset");
    }

    /// When nothing can be priced the caller must be able to tell that apart
    /// from "nothing is worth doing", so it can fall back rather than stop.
    #[tokio::test]
    async fn nothing_priceable_returns_nothing_rather_than_zeroes() {
        let conn = in_memory_conn().await;
        // Views that were never created in the database: every EXPLAIN fails.
        let dag = nested_dag();
        let order = canonical_order(&dag, &["salary".to_string(), "profile".to_string()]);

        let out = search_combos(
            &*conn,
            &dag,
            &order,
            &HashMap::new(),
            &CardinalitySubtreeCost,
            32,
            None,
            HmpObjective::QueryTime,
        )
        .await;

        assert!(
            out.is_empty(),
            "unpriceable candidates must not be returned as zero-cost ones: {out:?}"
        );
    }
}
