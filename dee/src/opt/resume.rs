//! Finishing a run whose trial was cancelled, under the previous best DAG.
//!
//! Cancelling a trial is right for the *experiment* and wrong for the consumer
//! waiting on the tables. So a cancelled trial is followed by a **resume**:
//! finish the run under the incumbent configuration -- the trial's is the one
//! just rejected -- on the warehouse the trial half-filled.
//!
//! # What the resume is allowed to assume
//!
//! Every DAG dee produces is equivalent to the one it was derived from: for
//! every node of the baseline there is a node of the candidate holding the same
//! tuples. Topology and node count may differ; the relations they have in common
//! may not. Optimization only ever transforms a DAG in correctness-preserving
//! ways -- that is the invariant every pass is written to hold, and it is what
//! makes a resume cheap.
//!
//! Call the incumbent `B` and the cancelled candidate `C`. Two consequences:
//!
//! * **A node in both that `C` finished holds the tuples `B` wants**, whatever
//!   its definition looks like. Comparing query text would be asking a question
//!   the invariant has already answered, and answering it wrongly: `make_temp`
//!   repoints every consumer of a promoted view, so the nodes whose text differs
//!   most are exactly the ones whose contents are identical.
//! * **A landing pad `C` built is still worth having even though `B` has no such
//!   node.** It holds its backing view's tuples, so `B`'s remaining nodes can
//!   read it instead of recomputing that view once per consumer. The work `C`
//!   spent on it is not wasted by the cancellation; it accelerates what is left.
//!   Once the pad has been consumed the resume reverts to `B`'s plan and drops
//!   it.
//!
//! # Where the invariant stops
//!
//! The pushdown pass narrows a `TempTable`'s projection *and* filter, so a
//! narrowed pad holds only what `C`'s consumers needed of the view -- not the
//! view. That does not break the invariant (the pad is not a node of `B`), but
//! it does make the pad useless to `B`'s consumers, which may want a pruned
//! column or row. So a pad is only usable while it is still *trivial*, and that
//! is checked here rather than trusted: nothing in this module needs to know how
//! pushdown works, only whether the pad in front of it still says
//! `SELECT * FROM <view>`.

use std::collections::{HashMap, HashSet};

use polyglot_sql::dialects::DialectType;

use crate::{
    connectors::Connector,
    dag::{Dag, MaterializeMode, TransformNode},
    opt::common::{landing_pad_name, make_temp},
};

/// What a node is, canonically: id, materialization, query and dependencies.
///
/// No longer used to decide reuse -- the invariant above answers that -- but
/// still the canonical form [`dag_signature`](crate::opt::hmp) is built from,
/// which is how a search recognises two combinations that reduce to the same
/// DAG.
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

/// How much of a cancelled run a resume is allowed to keep.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReusePolicy {
    /// Keep only relations whose node is defined identically in both DAGs, and
    /// only when everything upstream is too. What dee did before it relied on
    /// the equivalence invariant; kept as a way to turn the new behaviour off
    /// without a rebuild, because its failure mode would be silent.
    Strict,
    /// Keep every relation the cancelled run finished that the incumbent also
    /// has, and put its leftover landing pads to work finishing the rest.
    #[default]
    Equivalent,
}

impl ReusePolicy {
    pub fn as_str(&self) -> &'static str {
        match self {
            ReusePolicy::Strict => "strict",
            ReusePolicy::Equivalent => "equivalent",
        }
    }
}

impl std::str::FromStr for ReusePolicy {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "strict" => Ok(ReusePolicy::Strict),
            "equivalent" => Ok(ReusePolicy::Equivalent),
            other => Err(format!(
                "unknown trial reuse policy '{other}'; expected strict or equivalent"
            )),
        }
    }
}

/// What the resume must build, what it may skip, and what it must drop first.
#[derive(Debug, Clone, Default)]
pub struct ResumePlan {
    /// Nodes the resume does not need to build: their relations are already in
    /// the warehouse and hold what the incumbent wants.
    pub reusable: HashSet<String>,
    /// Relations to drop before the resume runs.
    pub to_drop: Vec<String>,
    /// Landing pads left by the cancelled run that the resume will read from,
    /// as `(backing view, pad)`. The resume's DAG is the incumbent with these
    /// pads wired back in; they are dropped once it finishes.
    pub pads: Vec<(String, String)>,
}

impl ResumePlan {
    /// Whether the trial left nothing worth keeping, so the resume is an
    /// ordinary full run.
    pub fn is_empty(&self) -> bool {
        self.reusable.is_empty()
    }
}

/// Decide what a resume under `incumbent` may keep from a cancelled run of
/// `trial`.
///
/// `completed` is what the executor reported finishing -- the only evidence
/// that a relation is whole. A node that was still in flight when the kill
/// landed may have left a partial relation, and nothing about the warehouse
/// distinguishes it from a finished one.
pub fn plan(
    trial: &Dag,
    incumbent: &Dag,
    completed: &HashSet<String>,
    policy: ReusePolicy,
) -> ResumePlan {
    match policy {
        ReusePolicy::Strict => plan_strict(trial, incumbent, completed),
        ReusePolicy::Equivalent => plan_equivalent(trial, incumbent, completed),
    }
}

/// The equivalence-invariant policy: node identity, plus the cancelled run's
/// pads.
fn plan_equivalent(trial: &Dag, incumbent: &Dag, completed: &HashSet<String>) -> ResumePlan {
    let mut reusable: HashSet<String> = HashSet::new();
    for node in incumbent.nodes.nodes() {
        if !completed.contains(&node.id) {
            continue;
        }
        // A view is not a stored relation, and re-creating one is DDL with no
        // data movement -- so reuse buys nothing measurable, while rebuilding
        // guarantees the definitions delivered are the incumbent's own.
        if node.materialize == MaterializeMode::View {
            continue;
        }
        // The physical object has to be the right kind. `CREATE OR REPLACE
        // VIEW` over a table fails, and a table standing in for a view would
        // deliver a snapshot where the DAG promised a live query.
        let Some(built) = trial.nodes.get(node.id.clone()) else {
            continue;
        };
        if built.materialize != node.materialize {
            continue;
        }
        reusable.insert(node.id.clone());
    }

    let pads = usable_pads(trial, incumbent, completed);
    for (_, pad) in &pads {
        reusable.insert(pad.clone());
    }

    ResumePlan {
        to_drop: to_drop(trial, incumbent, &reusable),
        reusable,
        pads,
    }
}

/// Landing pads the cancelled run finished that the resume can read from.
///
/// A pad qualifies only while it is still *trivial* -- built, absent from the
/// incumbent, named for a view the incumbent has, depending on exactly that
/// view, and still saying `SELECT * FROM <view>`. A pad the pushdown pass
/// narrowed fails that test and is dropped rather than used, which is the whole
/// reason this check exists: a narrowed pad holds a subset of its view.
fn usable_pads(
    trial: &Dag,
    incumbent: &Dag,
    completed: &HashSet<String>,
) -> Vec<(String, String)> {
    let dialect = crate::opt::common::dialect_for_db(&incumbent.db);
    let mut pads: Vec<(String, String)> = incumbent
        .nodes
        .nodes()
        .filter(|view| view.materialize == MaterializeMode::View)
        .filter_map(|view| {
            let pad_id = landing_pad_name(&view.id);
            if incumbent.nodes.get(pad_id.clone()).is_some() {
                // Not a leftover: the incumbent promotes this view itself, so
                // the pad is an ordinary node and was handled above.
                return None;
            }
            if !completed.contains(&pad_id) {
                return None;
            }
            let pad = trial.nodes.get(pad_id.clone())?;
            if pad.materialize != MaterializeMode::TempTable {
                return None;
            }
            if pad.depends_on.len() != 1 || !pad.depends_on.contains(&view.id) {
                return None;
            }
            if !is_trivial_pad(&pad.query_text, &view.id, dialect) {
                return None;
            }
            Some((view.id.clone(), pad_id))
        })
        .collect();
    // Deterministic, and consumers before producers so a pad nested inside
    // another is wired in first.
    pads.sort();
    pads
}

/// Whether `sql` is exactly "read this view, whole".
///
/// Compared through the parser rather than as text so that whitespace and
/// quoting cannot make a trivial pad look narrowed, and -- far more importantly
/// -- so that a narrowed one cannot look trivial.
fn is_trivial_pad(sql: &str, view_id: &str, dialect: DialectType) -> bool {
    let normalize = |s: &str| {
        polyglot_sql::parse_one(s, dialect)
            .ok()
            .and_then(|e| polyglot_sql::generate(&e, dialect).ok())
    };
    match (normalize(sql), normalize(&format!("SELECT * FROM {view_id}"))) {
        (Some(a), Some(b)) => a == b,
        // A pad whose text will not parse cannot be shown to be trivial, so it
        // is not one.
        _ => false,
    }
}

/// The DAG the resume actually executes: the incumbent, with the cancelled
/// run's usable pads wired back in.
///
/// This is `make_temp` again -- the same rebase that created the pads in the
/// first place -- so the consumers read the pad instead of recomputing the
/// view. The pads themselves are already built and are in `reusable`, so
/// nothing re-materializes them.
pub fn resume_dag(incumbent: &Dag, pads: &[(String, String)]) -> Dag {
    let mut dag = incumbent.clone();
    for (view, _) in pads {
        if let Err(e) = make_temp(&mut dag, view) {
            // The pad is then simply not used: `reusable` still names it, but
            // nothing reads it, and it is dropped with the rest afterwards. A
            // slower resume, never a wrong one.
            log::debug!("resume: could not wire in the pad for '{view}': {e}");
        }
    }
    dag
}

/// The strict policy: identical definitions plus a dependency closure.
fn plan_strict(trial: &Dag, incumbent: &Dag, completed: &HashSet<String>) -> ResumePlan {
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

    // Dependency closure, over the incumbent: a node whose upstream will be
    // rebuilt must be rebuilt too.
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

    ResumePlan {
        to_drop: to_drop(trial, incumbent, &reusable),
        reusable,
        pads: Vec::new(),
    }
}

/// Everything that must be dropped before the resume runs: whatever the
/// cancelled run may have written that is not being kept, and every incumbent
/// node about to be rebuilt.
///
/// The second half is not optional. Postgres creates a table with a bare
/// `CREATE TABLE ... AS` and refuses one that already exists, so a half-written
/// relation left by the kill would fail the resume outright.
fn to_drop(trial: &Dag, incumbent: &Dag, reusable: &HashSet<String>) -> Vec<String> {
    let mut to_drop: Vec<String> = trial
        .nodes
        .nodes()
        .map(|n| n.id.clone())
        .chain(incumbent.nodes.nodes().map(|n| n.id.clone()))
        .filter(|id| !reusable.contains(id))
        .collect();
    to_drop.sort_unstable();
    to_drop.dedup();
    to_drop
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

    fn completed(ids: &[&str]) -> Set<String> {
        ids.iter().map(|s| s.to_string()).collect()
    }

    /// One branch-point view feeding several tables -- the shape a
    /// materialization trial is built from, and the one where the old rule gave
    /// most of the cancelled run away.
    fn base(consumers: usize) -> Dag {
        let mut nodes = vec![node(
            "joined",
            "SELECT * FROM orders JOIN lineitem USING (k)",
            MaterializeMode::View,
            &[],
        )];
        nodes.extend((1..=consumers).map(|i| {
            node(
                &format!("s{i}"),
                "SELECT k, count(*) FROM joined GROUP BY k",
                MaterializeMode::Table,
                &["joined"],
            )
        }));
        dag(nodes)
    }

    // -----------------------------------------------------------------
    // Equivalent: the default
    // -----------------------------------------------------------------

    #[test]
    fn a_repointed_consumer_is_reused() {
        // The whole point. `make_temp` repoints every consumer at the pad, so
        // their definitions differ from the incumbent's -- and their contents
        // do not. Under the equivalence invariant that difference is not a
        // reason to rebuild a relation that is already correct.
        let incumbent = base(6);
        let mut trial = incumbent.clone();
        crate::opt::common::make_temp(&mut trial, "joined").unwrap();

        let done = completed(&["joined", "lp_joined", "s1", "s2"]);
        let p = plan(&trial, &incumbent, &done, ReusePolicy::Equivalent);

        assert!(p.reusable.contains("s1"), "a finished consumer was rebuilt");
        assert!(p.reusable.contains("s2"));
        // The view is rebuilt: DDL only, and it keeps the delivered definition
        // the incumbent's.
        assert!(!p.reusable.contains("joined"));
        // And the same run under the old rule keeps almost nothing, which is
        // what this change is about.
        let strict = plan(&trial, &incumbent, &done, ReusePolicy::Strict);
        assert!(
            strict.reusable.len() < p.reusable.len(),
            "the equivalent policy did not actually keep more"
        );
    }

    #[test]
    fn a_leftover_pad_is_kept_and_wired_into_the_resume() {
        // The cancelled run paid to materialize the view. That relation still
        // holds the view's tuples, so the nodes still to build should read it
        // rather than recomputing the view once each.
        let incumbent = base(6);
        let mut trial = incumbent.clone();
        crate::opt::common::make_temp(&mut trial, "joined").unwrap();

        let p = plan(
            &trial,
            &incumbent,
            &completed(&["joined", "lp_joined", "s1"]),
            ReusePolicy::Equivalent,
        );
        assert_eq!(
            p.pads,
            vec![("joined".to_string(), "lp_joined".to_string())]
        );
        assert!(p.reusable.contains("lp_joined"), "the pad would be rebuilt");
        assert!(
            !p.to_drop.contains(&"lp_joined".to_string()),
            "the pad would be dropped before it could be used"
        );

        let r = resume_dag(&incumbent, &p.pads);
        let s2 = r.nodes.get("s2".to_string()).unwrap();
        assert!(
            s2.query_text.contains("lp_joined") && s2.depends_on.contains("lp_joined"),
            "a node still to build does not read the pad: {}",
            s2.query_text
        );
    }

    #[test]
    fn a_narrowed_pad_is_not_used() {
        // Pushdown narrows a pad's projection and filter to what *that* run's
        // consumers needed, so it holds a subset of the view. Using it to build
        // the incumbent's nodes would silently drop rows or columns they want.
        let incumbent = base(2);
        let mut trial = incumbent.clone();
        crate::opt::common::make_temp(&mut trial, "joined").unwrap();
        trial.nodes.get_mut("lp_joined".to_string()).unwrap().query_text =
            "SELECT k FROM (SELECT * FROM joined) AS joined WHERE k > 0".to_string();

        let p = plan(
            &trial,
            &incumbent,
            &completed(&["joined", "lp_joined", "s1"]),
            ReusePolicy::Equivalent,
        );
        assert!(p.pads.is_empty(), "a narrowed pad was treated as the view");
        assert!(p.to_drop.contains(&"lp_joined".to_string()));
        // The consumer built from it is still reusable: its own contents are
        // right, whatever it happened to read.
        assert!(p.reusable.contains("s1"));
    }

    #[test]
    fn a_pad_is_only_trivial_up_to_formatting() {
        let d = DialectType::DuckDB;
        assert!(is_trivial_pad("select   *   from   joined", "joined", d));
        assert!(!is_trivial_pad("SELECT k FROM joined", "joined", d));
        assert!(!is_trivial_pad("SELECT * FROM joined WHERE k > 0", "joined", d));
        assert!(!is_trivial_pad("SELECT * FROM other", "joined", d));
        // Unparseable cannot be shown to be trivial, so it is not.
        assert!(!is_trivial_pad("NOT SQL (((", "joined", d));
    }

    #[test]
    fn a_node_that_never_reported_is_never_reused() {
        // The executor's completed set is the only evidence a relation is
        // whole. A node in flight when the kill landed may have left a partial
        // one, and nothing about the warehouse distinguishes the two.
        let incumbent = base(3);
        let p = plan(
            &incumbent.clone(),
            &incumbent,
            &completed(&["s1"]),
            ReusePolicy::Equivalent,
        );
        assert_eq!(p.reusable, completed(&["s1"]));
        for id in ["s2", "s3"] {
            assert!(p.to_drop.contains(&id.to_string()));
        }
    }

    #[test]
    fn a_mode_mismatch_is_never_reused() {
        // Same name, different kind of object: the relation on disk is not what
        // the incumbent asks for, and `CREATE OR REPLACE VIEW` over a table
        // fails outright.
        let incumbent = dag(vec![node("a", "SELECT 1", MaterializeMode::Table, &[])]);
        let trial = dag(vec![node("a", "SELECT 1", MaterializeMode::TempTable, &[])]);
        let p = plan(&trial, &incumbent, &completed(&["a"]), ReusePolicy::Equivalent);
        assert!(p.reusable.is_empty());
    }

    #[test]
    fn a_node_the_incumbent_does_not_have_is_dropped() {
        let incumbent = dag(vec![node("a", "SELECT 1", MaterializeMode::Table, &[])]);
        let trial = dag(vec![
            node("a", "SELECT 1", MaterializeMode::Table, &[]),
            node("scratch", "SELECT 2", MaterializeMode::Table, &[]),
        ]);
        let p = plan(
            &trial,
            &incumbent,
            &completed(&["a", "scratch"]),
            ReusePolicy::Equivalent,
        );
        assert!(p.reusable.contains("a"));
        assert!(p.to_drop.contains(&"scratch".to_string()));
    }

    // -----------------------------------------------------------------
    // Strict: the escape hatch, unchanged
    // -----------------------------------------------------------------

    #[test]
    fn strict_keeps_only_identical_definitions() {
        let incumbent = base(6);
        let mut trial = incumbent.clone();
        crate::opt::common::make_temp(&mut trial, "joined").unwrap();
        let p = plan(
            &trial,
            &incumbent,
            &completed(&["joined", "lp_joined", "s1", "s2"]),
            ReusePolicy::Strict,
        );
        // Only the untouched view matches, and it is a view -- strict keeps it
        // because strict is the old behaviour verbatim.
        assert_eq!(p.reusable, completed(&["joined"]));
        assert!(p.pads.is_empty());
    }

    #[test]
    fn strict_rebuilds_a_node_whose_upstream_is_rebuilt() {
        let mk = |first: &str| {
            dag(vec![
                node("a", first, MaterializeMode::Table, &[]),
                node("b", "SELECT * FROM a", MaterializeMode::Table, &["a"]),
            ])
        };
        let p = plan(
            &mk("SELECT 2"),
            &mk("SELECT 1"),
            &completed(&["a", "b"]),
            ReusePolicy::Strict,
        );
        assert!(p.reusable.is_empty(), "a stale input reached a consumer");
    }
}
