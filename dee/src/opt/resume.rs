//! Finishing a run whose trial was cancelled, under the previous best DAG.
//!
//! Cancelling a trial is right for the *experiment* and wrong for the consumer
//! waiting on the tables. So a cancelled trial is followed by a **resume**:
//! build only what never finished, under the incumbent configuration -- the
//! trial's is the one just rejected -- on the warehouse the trial half-filled.
//!
//! The hard part is deciding what "already finished" means, because the trial
//! DAG and the incumbent are not the same node set.
//! [`make_temp`](crate::opt::common::make_temp) inserts landing-pad nodes and
//! inlines intermediate views, and the pushdown pass rewrites query text. A
//! relation the trial left on disk is only reusable if the incumbent's node of
//! that name would have produced exactly the same thing, and only if everything
//! it was derived from is reusable too. Get either half wrong and the resume
//! silently delivers a relation built from stale inputs -- which is worse than
//! rebuilding the DAG, because nothing about it looks like a failure.

use std::collections::{HashMap, HashSet};

use crate::{
    connectors::Connector,
    dag::{Dag, MaterializeMode, TransformNode},
};

/// What a node is, canonically: everything about it that decides what its
/// relation contains.
///
/// Two nodes with the same signature in two different DAGs produce the same
/// relation, which is exactly the question a resume has to answer.
pub fn node_signature(node: &TransformNode) -> String {
    let mut deps: Vec<&str> = node.depends_on.iter().map(String::as_str).collect();
    deps.sort_unstable();
    format!(
        "{}::{}::{}::[{}]",
        node.id,
        node.materialize.as_str(),
        node.query_text,
        deps.join(",")
    )
}

/// What the resume must drop and what it may keep.
#[derive(Debug, Clone, Default)]
pub struct ResumePlan {
    /// Nodes of `incumbent` whose relations the trial already built correctly.
    /// The resume skips these.
    pub reusable: HashSet<String>,
    /// Relations to drop before the resume runs.
    pub to_drop: Vec<String>,
}

impl ResumePlan {
    /// Whether the trial left nothing worth keeping, so the resume is an
    /// ordinary full run.
    pub fn is_empty(&self) -> bool {
        self.reusable.is_empty()
    }
}

/// Decide what a resume under `incumbent` may reuse from a cancelled run of
/// `trial`, and what it must drop first.
///
/// `completed` is what the executor reported finishing -- the only evidence a
/// relation is whole. Two filters are applied to it, in order:
///
/// 1. **Identity.** The incumbent must have a node with the same
///    [`node_signature`]. A landing pad the incumbent does not have, or a node
///    whose query the pushdown pass rewrote, is not the same relation even
///    though it has the same name.
/// 2. **Dependency closure.** A node whose upstream will be rebuilt must be
///    rebuilt too. Without this the resume produces a relation derived from
///    whatever the trial happened to leave behind.
pub fn plan(trial: &Dag, incumbent: &Dag, completed: &HashSet<String>) -> ResumePlan {
    let incumbent_sigs: HashMap<&str, String> = incumbent
        .nodes
        .nodes()
        .map(|n| (n.id.as_str(), node_signature(n)))
        .collect();

    let mut reusable: HashSet<String> = HashSet::new();
    for node in trial.nodes.nodes() {
        if !completed.contains(&node.id) {
            continue;
        }
        if incumbent_sigs.get(node.id.as_str()) == Some(&node_signature(node)) {
            reusable.insert(node.id.clone());
        }
    }

    // Dependency closure, over the incumbent: walking it in topological order
    // means every dependency is resolved before the node that reads it, so one
    // pass is enough however deep the chain.
    for id in incumbent.nodes.topological_sort() {
        if !reusable.contains(&id) {
            continue;
        }
        let Some(node) = incumbent.nodes.get(id.clone()) else {
            reusable.remove(&id);
            continue;
        };
        if node.depends_on.iter().any(|dep| !reusable.contains(dep)) {
            reusable.remove(&id);
        }
    }

    // Everything the trial may have written that the resume will not reuse: the
    // nodes still in flight when it was killed, the landing pads the incumbent
    // does not have, and every incumbent node about to be rebuilt. The last is
    // not optional -- Postgres creates a table with a bare `CREATE TABLE ... AS`
    // and refuses one that already exists.
    let mut to_drop: Vec<String> = trial
        .nodes
        .nodes()
        .map(|n| n.id.clone())
        .chain(incumbent.nodes.nodes().map(|n| n.id.clone()))
        .filter(|id| !reusable.contains(id))
        .collect();
    to_drop.sort_unstable();
    to_drop.dedup();

    ResumePlan { reusable, to_drop }
}

/// Drop every relation named in `ids`, in all three materialization modes.
///
/// Best-effort, exactly like [`Executor::cleanup`](crate::executor::Executor::cleanup):
/// a name that does not exist is the common case, not a failure.
pub async fn drop_relations<C>(conn: &C, ids: &[String]) -> usize
where
    C: Connector + Send + Sync,
{
    let mut dropped = 0;
    for id in ids {
        for mode in [
            MaterializeMode::View,
            MaterializeMode::Table,
            MaterializeMode::TempTable,
        ] {
            dropped += conn.drop_relation(mode, id.clone()).await.unwrap_or(0);
        }
    }
    dropped
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use std::collections::HashSet as Set;

    fn node(id: &str, sql: &str, mode: MaterializeMode, deps: &[&str]) -> TransformNode {
        TransformNode {
            id: id.to_string(),
            query_text: sql.to_string(),
            materialize: mode,
            depends_on: deps.iter().map(|s| s.to_string()).collect(),
            schema: None,
        }
    }

    fn dag(nodes: Vec<TransformNode>) -> Dag {
        let mut g = Graph::new(HashMap::new());
        for n in nodes {
            g.add_node_unchecked(n);
        }
        Dag {
            db: "duckdb".into(),
            nodes: g,
            sources: Vec::new(),
            max_parallelism: None,
        }
    }

    fn base() -> Dag {
        dag(vec![
            node("a", "SELECT 1", MaterializeMode::Table, &[]),
            node("b", "SELECT * FROM a", MaterializeMode::View, &["a"]),
            node("c", "SELECT * FROM b", MaterializeMode::Table, &["b"]),
        ])
    }

    fn completed(ids: &[&str]) -> Set<String> {
        ids.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn identical_nodes_that_finished_are_reused() {
        let plan = plan(&base(), &base(), &completed(&["a", "b"]));
        assert_eq!(plan.reusable, completed(&["a", "b"]));
        // `c` never ran, so it is rebuilt -- and dropped first in case the kill
        // landed inside it.
        assert!(plan.to_drop.contains(&"c".to_string()));
        assert!(!plan.to_drop.contains(&"a".to_string()));
    }

    #[test]
    fn a_node_the_incumbent_defines_differently_is_not_reusable() {
        // The pushdown pass rewrote `a`'s query in the trial. The relation on
        // disk has the right name and the wrong contents.
        let trial = dag(vec![
            node("a", "SELECT 1 WHERE true", MaterializeMode::Table, &[]),
            node("b", "SELECT * FROM a", MaterializeMode::View, &["a"]),
        ]);
        let incumbent = dag(vec![
            node("a", "SELECT 1", MaterializeMode::Table, &[]),
            node("b", "SELECT * FROM a", MaterializeMode::View, &["a"]),
        ]);
        let plan = plan(&trial, &incumbent, &completed(&["a", "b"]));
        assert!(!plan.reusable.contains("a"));
        assert!(plan.to_drop.contains(&"a".to_string()));
    }

    #[test]
    fn a_node_whose_upstream_is_rebuilt_is_rebuilt_too() {
        // `b` completed and is defined identically, but it reads `a`, which is
        // not reusable. Keeping `b` would deliver a relation derived from
        // whatever the trial left behind -- the failure this closure exists to
        // prevent, and the one nothing downstream would notice.
        let trial = dag(vec![
            node("a", "SELECT 2", MaterializeMode::Table, &[]),
            node("b", "SELECT * FROM a", MaterializeMode::Table, &["a"]),
        ]);
        let incumbent = dag(vec![
            node("a", "SELECT 1", MaterializeMode::Table, &[]),
            node("b", "SELECT * FROM a", MaterializeMode::Table, &["a"]),
        ]);
        let plan = plan(&trial, &incumbent, &completed(&["a", "b"]));
        assert!(plan.reusable.is_empty());
    }

    #[test]
    fn the_closure_runs_the_whole_length_of_a_chain() {
        // a -> b -> c -> d, with only `a` differing. Nothing survives.
        let chain = |first: &str| {
            dag(vec![
                node("a", first, MaterializeMode::Table, &[]),
                node("b", "SELECT * FROM a", MaterializeMode::Table, &["a"]),
                node("c", "SELECT * FROM b", MaterializeMode::Table, &["b"]),
                node("d", "SELECT * FROM c", MaterializeMode::Table, &["c"]),
            ])
        };
        let plan = plan(
            &chain("SELECT 2"),
            &chain("SELECT 1"),
            &completed(&["a", "b", "c", "d"]),
        );
        assert!(plan.reusable.is_empty(), "a stale input reached a consumer");
    }

    #[test]
    fn a_landing_pad_the_incumbent_does_not_have_is_dropped() {
        // A trial materializes `b` behind a landing pad. The incumbent has no
        // such node, so the pad is a relation nothing will read and a name the
        // next candidate could collide with.
        let trial = dag(vec![
            node("a", "SELECT 1", MaterializeMode::Table, &[]),
            node("lp_b", "SELECT * FROM a", MaterializeMode::TempTable, &["a"]),
        ]);
        let plan = plan(&trial, &base(), &completed(&["a", "lp_b"]));
        assert!(!plan.reusable.contains("lp_b"));
        assert!(plan.to_drop.contains(&"lp_b".to_string()));
        assert!(plan.reusable.contains("a"));
    }

    #[test]
    fn a_materialized_view_rewrites_its_consumers_out_of_reuse() {
        // The shape that decides how much a resume actually saves, and the one
        // worth being explicit about: `make_temp` does not only *add* a landing
        // pad, it repoints every consumer at it. Those consumers then have a
        // different definition from the incumbent's, so relations the trial
        // finished building are rebuilt even though their contents would have
        // been identical -- the pad is `SELECT * FROM <view>`.
        //
        // That is deliberate. Reusing them would mean deciding that two
        // different queries produce the same relation, which is a semantic
        // judgement this comparison does not make. But it means the saving is
        // smallest exactly where a trial is most expensive: the more consumers
        // a candidate view has, the more the trial rewrites, and the more the
        // resume has to redo.
        use crate::opt::common::make_temp;
        let mut nodes = vec![node(
            "joined",
            "select * from orders join lineitem using (l_orderkey)",
            MaterializeMode::View,
            &[],
        )];
        nodes.extend((1..=6).map(|i| {
            node(
                &format!("s{i}"),
                "select o_custkey, sum(v) from joined group by o_custkey",
                MaterializeMode::Table,
                &["joined"],
            )
        }));
        let incumbent = dag(nodes);

        let mut trial = incumbent.clone();
        make_temp(&mut trial, "joined").unwrap();
        assert_eq!(trial.nodes.num_nodes(), 8, "the pad is an extra node");
        let s1 = trial.nodes.get("s1".to_string()).unwrap();
        assert!(
            s1.query_text.contains("lp_joined") && s1.depends_on.contains("lp_joined"),
            "the fixture no longer exercises a repointed consumer: {}",
            s1.query_text
        );

        // The kill lands after four nodes: the view, its pad, and two consumers.
        let p = plan(
            &trial,
            &incumbent,
            &completed(&["joined", "lp_joined", "s1", "s2"]),
        );
        assert_eq!(
            p.reusable,
            completed(&["joined"]),
            "only the node both DAGs define identically is reusable"
        );
        // The pad is not the incumbent's to keep, and the two consumers built
        // from it must be dropped before they are rebuilt.
        for id in ["lp_joined", "s1", "s2"] {
            assert!(p.to_drop.contains(&id.to_string()), "{id} was left behind");
        }
    }

    #[test]
    fn a_node_that_never_reported_is_never_reused() {
        // The executor's completed set is the only evidence a relation is
        // whole. A node that was in flight when the kill landed may have left a
        // partial relation, and it is not distinguishable from a finished one
        // by looking at the warehouse.
        let plan = plan(&base(), &base(), &completed(&["a"]));
        assert_eq!(plan.reusable, completed(&["a"]));
        assert!(plan.to_drop.contains(&"b".to_string()));
    }
}
