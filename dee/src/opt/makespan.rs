//! How long the DAG would *take*, as opposed to how much work it would do.
//!
//! Every other cost in this crate is a sum over plan operators: a total-work
//! number. Makespan is not a sum, it is a **longest path** --- the DAG finishes
//! when its slowest chain finishes, and everything off that chain is free. The
//! two come apart exactly where it matters to HMP. Materializing a View inserts
//! a node that has to *finish* before its consumers can start, so a set of
//! candidates strung along one dependency chain serializes into as many stages
//! as it has links, while the same number of candidates spread across
//! independent branches adds one stage to each.
//!
//! Measured on `p05_hr`: the baseline critical path is 1799 ms against a
//! measured makespan of 1801 ms, so on that DAG makespan is entirely
//! path-bound. Its eleven trialled candidates separate perfectly by how many
//! members lie on one chain --- three stages ran 4772-5506 ms, four ran
//! 6676-7186 ms, with no overlap --- while set *size* barely orders them at
//! all. A scalar of duplicated work cannot see that; only a walk over the graph
//! can.
//!
//! # Where the numbers come from
//!
//! Nothing here runs a query or asks for a plan. [`estimate`] is fed the
//! [`DuplicateCost`] that [`crate::opt::dup::duplicate_cost_set`] already
//! measured, plus the baseline run's per-node timings, and combines them:
//!
//! * a **member** becomes a node in its own right, priced at what the build
//!   probe said it costs to compute plus what the model says it costs to write;
//! * a **consumer** keeps its measured duration, scaled down by the fraction of
//!   its plan that the members accounted for --- that work becomes a scan of a
//!   table instead;
//! * every **other** node keeps its measured duration, because materializing
//!   somewhere else does not change what it does.
//!
//! Only the *fraction* comes from the cost model. Magnitudes are anchored to
//! the measured run wherever a measurement exists, and the one place they
//! cannot be --- a member, which has never run as a node --- is converted
//! through a calibration ratio fitted to the consumers, where both a measured
//! and an estimated number are available. That keeps a biased cost model from
//! translating into a biased makespan: a model that reads twice as slow as the
//! machine cancels out of the ratio.
//!
//! # What it does not model
//!
//! Concurrency limits. This is a pure critical path, as though the executor
//! could run every ready node at once. On `p05_hr` the observed peak was six
//! concurrent nodes and the path and the makespan agreed to 2 ms, so the cap
//! never bound; a wide DAG against a low `max_parallelism` would need list
//! scheduling instead, and would come out optimistic here.

use std::collections::{HashMap, HashSet};

use crate::dag::Dag;
use crate::executor::ExecStats;
use crate::opt::dup::DuplicateCost;

/// A predicted makespan and the shape behind it.
#[derive(Debug, Clone, PartialEq)]
pub struct MakespanEstimate {
    /// Predicted wall clock for the whole DAG, in seconds.
    pub makespan_s: f64,
    /// The same walk over the untouched DAG, for comparison.
    ///
    /// Carried so the prediction can be read against the run that actually
    /// happened: this number is built from measured durations alone, so a gap
    /// between it and the recorded baseline is a fault in the *walk*, and a gap
    /// between `makespan_s` and a trial is a fault in the *costing*. Keeping
    /// them apart is the difference between a model that can be debugged and
    /// one that can only be believed.
    pub baseline_s: f64,
    /// The longest chain of set members under ancestry: how many builds have to
    /// happen one after another before any consumer can start.
    pub stages: usize,
    /// The nodes on the predicted critical path, source-most first.
    pub critical_path: Vec<String>,
}

/// The DAG's own critical path from the run it already had, in seconds.
///
/// No estimate anywhere in it: every node is priced at what it measured. This
/// is the control the predictions are read against.
pub fn measured_baseline(dag: &Dag, baseline: &ExecStats) -> f64 {
    let measured: HashMap<String, f64> = baseline
        .node_stats
        .iter()
        .map(|(id, s)| (id.clone(), s.duration.num_milliseconds() as f64 / 1000.0))
        .collect();
    critical_path(dag, &measured).0
}

/// The longest weighted path through `dag`, and the nodes on it.
///
/// A node missing from `cost_of` contributes zero rather than aborting the
/// walk: Views cost nothing to create, and a DAG is mostly Views.
pub fn critical_path(dag: &Dag, cost_of: &HashMap<String, f64>) -> (f64, Vec<String>) {
    let order = dag.nodes.topological_sort();
    // Finish time of each node, and which predecessor it waited on.
    let mut finish: HashMap<String, f64> = HashMap::with_capacity(order.len());
    let mut waited_on: HashMap<String, Option<String>> = HashMap::with_capacity(order.len());

    for id in &order {
        let deps = dag
            .nodes
            .get(id.clone())
            .map(|n| n.depends_on.clone())
            .unwrap_or_default();
        let mut start = 0.0;
        let mut blocker: Option<String> = None;
        for dep in &deps {
            if let Some(f) = finish.get(dep)
                && *f > start
            {
                start = *f;
                blocker = Some(dep.clone());
            }
        }
        finish.insert(id.clone(), start + cost_of.get(id).copied().unwrap_or(0.0));
        waited_on.insert(id.clone(), blocker);
    }

    let Some((last, total)) = finish
        .iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(k, v)| (k.clone(), *v))
    else {
        return (0.0, Vec::new());
    };

    let mut path = vec![last.clone()];
    let mut cur = last;
    while let Some(Some(prev)) = waited_on.get(&cur).cloned() {
        path.push(prev.clone());
        cur = prev;
    }
    path.reverse();
    (total, path)
}

/// The longest chain of `set` members under ancestry in `dag`.
///
/// One stage per link: members on one chain build in sequence, members on
/// independent branches build at the same time.
pub fn stages(dag: &Dag, set: &HashSet<String>) -> usize {
    let mut depth: HashMap<String, usize> = HashMap::new();
    let mut best = 0usize;
    for id in dag.nodes.topological_sort() {
        let deps = dag
            .nodes
            .get(id.clone())
            .map(|n| n.depends_on.clone())
            .unwrap_or_default();
        let inherited = deps.iter().filter_map(|d| depth.get(d)).copied().max().unwrap_or(0);
        let d = if set.contains(&id) { inherited + 1 } else { inherited };
        best = best.max(d);
        depth.insert(id, d);
    }
    best
}

/// Predict the makespan of `dag` with every member of `dup` materialized.
///
/// `baseline` is the run the DAG has already had, which supplies a measured
/// duration for every node the change does not touch. Returns `None` when the
/// consumers carry no priced work to calibrate against --- there is then no
/// honest way to convert the members' estimated build cost into the units the
/// measured nodes are in, and a guess would be worse than an absence.
pub fn estimate(dag: &Dag, dup: &DuplicateCost, baseline: &ExecStats) -> Option<MakespanEstimate> {
    let measured: HashMap<String, f64> = baseline
        .node_stats
        .iter()
        .map(|(id, s)| (id.clone(), s.duration.num_milliseconds() as f64 / 1000.0))
        .collect();

    // Measured seconds per estimated second, fitted on the consumers --- the
    // only nodes with both. A cost model that reads uniformly fast or slow
    // divides out here instead of landing in the answer.
    let (mut measured_sum, mut estimated_sum) = (0.0, 0.0);
    for (consumer, whole) in &dup.consumer_total {
        if let Some(m) = measured.get(consumer)
            && *whole > 0.0
        {
            measured_sum += m;
            estimated_sum += whole;
        }
    }
    if estimated_sum <= 0.0 {
        return None;
    }
    let calibration = measured_sum / estimated_sum;

    let members: HashSet<String> = dup.views.iter().cloned().collect();
    let regions: HashMap<&str, f64> = dup
        .per_consumer
        .iter()
        .map(|(c, v)| (c.as_str(), *v))
        .collect();

    let mut cost_of = measured.clone();

    // A member is a node now: it computes its body once and writes it out.
    // Both halves are required. Without a write constant the build term loses
    // the part that dominates it --- on `p05_hr`, modelled compute accounts for
    // 0.17s of a 2.01s build --- and an estimate short by that much does not
    // rank candidates, it ranks noise. Refuse rather than round the missing
    // term to zero; the caller's fallback is the single-objective order, which
    // is at least measured end to end.
    for build in &dup.build_per_member {
        let write = build.write?;
        cost_of.insert(build.view.clone(), (build.compute + write) * calibration);
    }

    // A consumer keeps doing everything its plan did except the part the
    // members accounted for, which becomes a scan of a table it no longer has
    // to compute. Expressed as a fraction of its own measured duration so the
    // magnitude stays a measurement.
    for (consumer, whole) in &dup.consumer_total {
        let Some(m) = measured.get(consumer) else {
            continue;
        };
        if *whole <= 0.0 {
            continue;
        }
        let removed = regions.get(consumer.as_str()).copied().unwrap_or(0.0);
        let kept = ((whole - removed) / whole).clamp(0.0, 1.0);
        cost_of.insert(consumer.clone(), m * kept);
    }

    log::debug!(
        "makespan: calibration={calibration:.6} (measured {measured_sum:.4}s over estimated \
         {estimated_sum:.4}s across {} consumer(s)); members {:?}",
        dup.consumer_total.len(),
        dup.build_per_member
            .iter()
            .map(|b| (
                b.view.clone(),
                b.compute,
                b.write,
                (b.compute + b.write.unwrap_or(0.0)) * calibration,
                b.write.is_some()
            ))
            .collect::<Vec<_>>()
    );

    let (makespan_s, path) = critical_path(dag, &cost_of);
    let (baseline_s, _) = critical_path(dag, &measured);

    Some(MakespanEstimate {
        makespan_s,
        baseline_s,
        stages: stages(dag, &members),
        critical_path: path,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::{MaterializeMode, TransformNode};
    use crate::executor::NodeStats;
    use crate::graph::Graph;
    use crate::opt::dup::MemberBuild;
    use chrono::{TimeDelta, Utc};

    fn node(id: &str, deps: &[&str]) -> TransformNode {
        TransformNode {
            id: id.to_string(),
            query_text: "SELECT 1".to_string(),
            materialize: MaterializeMode::View,
            depends_on: deps.iter().map(|d| d.to_string()).collect(),
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

    /// An `ExecStats` carrying nothing but the durations named.
    fn stats_of(durations: &[(&str, i64)]) -> ExecStats {
        let now = Utc::now();
        ExecStats {
            start: now,
            finish: now,
            duration: TimeDelta::milliseconds(0),
            node_stats: durations
                .iter()
                .map(|(id, ms)| {
                    (
                        id.to_string(),
                        NodeStats {
                            start: now,
                            finish: now,
                            duration: TimeDelta::milliseconds(*ms),
                            plan: None,
                            rows_produced: None,
                        },
                    )
                })
                .collect(),
            system_samples: Vec::new(),
        }
    }

    fn costs(pairs: &[(&str, f64)]) -> HashMap<String, f64> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    /// The walk must follow the slowest chain, not the longest one. A DAG that
    /// counted nodes would take the other branch here.
    #[test]
    fn the_critical_path_is_the_longest_weighted_chain() {
        let dag = dag_of(vec![
            node("src", &[]),
            node("cheap_a", &["src"]),
            node("cheap_b", &["cheap_a"]),
            node("cheap_c", &["cheap_b"]),
            node("dear", &["src"]),
            node("sink", &["cheap_c", "dear"]),
        ]);
        let (total, path) = critical_path(
            &dag,
            &costs(&[
                ("src", 1.0),
                ("cheap_a", 1.0),
                ("cheap_b", 1.0),
                ("cheap_c", 1.0),
                ("dear", 90.0),
                ("sink", 2.0),
            ]),
        );
        assert!((total - 93.0).abs() < 1e-9, "got {total}");
        assert_eq!(path, vec!["src", "dear", "sink"]);
    }

    /// A node nobody priced is free, not fatal: a DAG is mostly Views, and
    /// creating a View does no work.
    #[test]
    fn a_missing_node_cost_contributes_zero_rather_than_aborting() {
        let dag = dag_of(vec![node("a", &[]), node("v", &["a"]), node("b", &["v"])]);
        let (total, path) = critical_path(&dag, &costs(&[("a", 3.0), ("b", 4.0)]));
        assert!((total - 7.0).abs() < 1e-9, "got {total}");
        assert_eq!(path, vec!["a", "v", "b"]);
    }

    /// The whole reason a second axis exists. Two sets of the same size, priced
    /// identically per member --- one strung along a chain, one spread across
    /// branches. The chain has to predict the longer wall clock.
    #[test]
    fn stacking_members_on_one_chain_costs_more_than_spreading_them() {
        // chain: src -> c1 -> c2 -> sink;  branches: src -> {b1, b2} -> sink
        let dag = dag_of(vec![
            node("src", &[]),
            node("c1", &["src"]),
            node("c2", &["c1"]),
            node("b1", &["src"]),
            node("b2", &["src"]),
            node("sink", &["c2", "b1", "b2"]),
        ]);

        let member_cost = 5.0;
        let chain = costs(&[("c1", member_cost), ("c2", member_cost), ("sink", 1.0)]);
        let spread = costs(&[("b1", member_cost), ("b2", member_cost), ("sink", 1.0)]);

        let (chained, _) = critical_path(&dag, &chain);
        let (spread_total, _) = critical_path(&dag, &spread);
        assert!(
            chained > spread_total,
            "two members on one chain serialize; two on branches do not: \
             chain={chained}, spread={spread_total}"
        );

        assert_eq!(stages(&dag, &HashSet::from(["c1".into(), "c2".into()])), 2);
        assert_eq!(stages(&dag, &HashSet::from(["b1".into(), "b2".into()])), 1);
    }

    /// Materializing on one branch cannot predict a makespan below what an
    /// untouched branch already takes.
    #[test]
    fn an_unaffected_branch_still_sets_the_floor() {
        let dag = dag_of(vec![
            node("src", &[]),
            node("v", &["src"]),
            node("touched", &["v"]),
            node("untouched", &["src"]),
        ]);
        let baseline = stats_of(&[("src", 10), ("touched", 100), ("untouched", 900)]);
        let dup = DuplicateCost {
            views: vec!["v".to_string()],
            build_once: 1.0,
            build_per_member: vec![MemberBuild {
                view: "v".to_string(),
                compute: 1.0,
                write: Some(0.0),
            }],
            // The consumer's whole plan is 2.0, of which the member is 1.5 ---
            // so materializing removes three quarters of its work.
            per_consumer: vec![("touched".to_string(), 1.5)],
            consumer_total: vec![("touched".to_string(), 2.0)],
            duplicate: 0.5,
            write_total: None,
        };

        let est = estimate(&dag, &dup, &baseline).expect("the consumer calibrates it");
        assert!(
            est.makespan_s >= 0.910,
            "the untouched 900ms branch behind a 10ms source is a floor: got {}",
            est.makespan_s
        );
        assert!(
            (est.baseline_s - 0.910).abs() < 1e-9,
            "the measured-only walk is src + untouched: got {}",
            est.baseline_s
        );
    }

    /// Materializing a large View is mostly the write. A cost model that
    /// cannot price one is missing the dominant term of every build, and the
    /// answer has to be an absence rather than a number an order of magnitude
    /// low.
    ///
    /// Measured on `p05_hr`: `stg_employees` models at 0.17s of compute and
    /// takes 2.01s to build.
    #[test]
    fn a_build_with_no_write_constant_is_refused_rather_than_under_priced() {
        let dag = dag_of(vec![node("src", &[]), node("v", &["src"]), node("c", &["v"])]);
        let baseline = stats_of(&[("src", 10), ("c", 100)]);
        let mut dup = DuplicateCost {
            views: vec!["v".to_string()],
            build_once: 1.0,
            build_per_member: vec![MemberBuild {
                view: "v".to_string(),
                compute: 1.0,
                write: None,
            }],
            per_consumer: vec![("c".to_string(), 0.5)],
            consumer_total: vec![("c".to_string(), 2.0)],
            duplicate: 0.5,
            write_total: None,
        };
        assert!(
            estimate(&dag, &dup, &baseline).is_none(),
            "an unpriced write must not be silently charged as zero"
        );

        dup.build_per_member[0].write = Some(0.25);
        assert!(
            estimate(&dag, &dup, &baseline).is_some(),
            "with a write constant the same set estimates fine"
        );
    }

    /// Consumers are where the model is calibrated, so a set with none of them
    /// priced has no honest conversion into measured seconds."""
    #[test]
    fn nothing_to_calibrate_against_is_an_absence_not_a_guess() {
        let dag = dag_of(vec![node("v", &[]), node("c", &["v"])]);
        let dup = DuplicateCost {
            views: vec!["v".to_string()],
            build_once: 1.0,
            build_per_member: vec![MemberBuild {
                view: "v".to_string(),
                compute: 1.0,
                write: Some(0.0),
            }],
            per_consumer: vec![("c".to_string(), 1.0)],
            consumer_total: vec![("c".to_string(), 0.0)],
            duplicate: 1.0,
            write_total: None,
        };
        assert!(estimate(&dag, &dup, &stats_of(&[("c", 5)])).is_none());
    }
}
