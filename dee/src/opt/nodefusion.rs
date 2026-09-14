//! NodeFusion -- collapse the DAG's Table nodes into one fused query.
//!
//! A dee DAG executes as one relation per node: every View is created as a
//! view and every Table as its own `CREATE TABLE ... AS`. Work shared between
//! two Tables is therefore planned -- and usually scanned -- once per Table
//! that reaches it. That duplication is what HMP and OMP spend measured DAG
//! runs deciding how to materialize away.
//!
//! This pass attacks the same duplication structurally, and for free. It
//! builds one new node `f` that reproduces the whole upstream DAG as a `WITH`
//! chain and emits every Table's rows in a single `UNION ALL`, so the engine
//! sees one query and can share a CTE across every consumer inside it. Each
//! original Table node becomes a cheap projection out of `f`:
//!
//! ```sql
//! -- node f (TempTable, depends on nothing)
//! WITH n_v1 AS (<v1 sql>),
//!      n_t0 AS MATERIALIZED (<t0 sql, refs to v1 rewritten to n_v1>),
//!      n_v2 AS (<v2 sql, refs to t0 rewritten to n_t0>),
//!      n_t1 AS (<t1 sql, refs to v2 rewritten to n_v2>)
//! SELECT 0 AS kind, a AS "k0_a", b AS "k0_b", CAST(NULL AS VARCHAR) AS "k1_c" FROM n_t0
//! UNION ALL
//! SELECT 1 AS kind, CAST(NULL AS INTEGER) AS "k0_a", CAST(NULL AS VARCHAR) AS "k0_b", c AS "k1_c" FROM n_t1
//!
//! -- node t0                              -- node t1
//! SELECT "k0_a" AS a, "k0_b" AS b         SELECT "k1_c" AS c
//! FROM <f> WHERE kind = 0                 FROM <f> WHERE kind = 1
//! ```
//!
//! The branches of a `UNION ALL` must agree on their row shape and the Tables
//! do not, so the fused relation's schema is a `kind` discriminator followed
//! by every Table's columns concatenated in `kind` order, each branch filling
//! the columns that are not its own with NULL.
//!
//! Like Pushdown, this is a pure rewrite: it measures nothing, runs the DAG
//! zero times, and decides everything from the DAG in front of it.
//!
//! # What is fused, and what is not
//!
//! Every `Table` node is fused, *including* a Table that feeds another Table.
//! An intermediate Table gets a CTE as well as a `kind` branch, so it is
//! computed once inside `f`, read from that CTE by whatever is downstream of
//! it, and still delivered as its own relation. Its CTE is materialized by
//! default, because it is read by both its own branch and whatever is
//! downstream of it; a Table that feeds nothing is read once and is not. That is also what keeps the
//! graph acyclic: everything upstream of any Table ends up inside `f`, so `f`
//! itself depends on nothing but the warehouse's own source tables.
//!
//! `TempTable` nodes are not fused. They are the landing pads HMP and OMP
//! create, and their whole purpose is to be a materialization barrier the
//! search placed deliberately; folding one into a CTE would silently undo the
//! decision that put it there. A TempTable anywhere upstream of a Table makes
//! the DAG unfusable rather than partly fused, because a partial fusion would
//! have to leave that Table out and the point is to share work across all of
//! them.
//!
//! # What fusion changes
//!
//! The rows a Table holds, its columns and their types are all preserved
//! exactly -- that is the contract, and the tests check it against a real
//! engine. What is not preserved is the *order* those rows are stored in: a
//! Table whose query ends in `ORDER BY` gets that ordering applied inside its
//! CTE and then loses it passing through the `UNION ALL`. A SQL table is an
//! unordered relation and a consumer that cares must order for itself, so this
//! breaks nothing that was guaranteed -- but a DAG that was quietly relying on
//! `CREATE TABLE AS ... ORDER BY` laying rows down in order will notice.
//!
//! View nodes are left in the graph and are still created as views. A view is
//! a definition rather than a computation, so the duplication costs nothing,
//! and keeping them is what lets a sink view -- or any consumer that was not
//! fused -- go on binding to the name it was written against.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use log::{debug, warn};
use polyglot_sql::dialects::DialectType;
use serde::{Deserialize, Serialize};

use crate::{
    connectors::Connector,
    dag::{Dag, MaterializeMode, TransformNode},
    executor::{ExecStats, Executor},
    opt::{
        Optimization, OptimizerConfig, OptimizerError,
        combo::{ComboCoster, ComboScore, CostedCombo, Objective, canonical_order},
        common::{
            bare_table_name, default_spool_factor, dialect_for_db, fused_node_name,
            rewrite_node_refs, supports_materialized_hint,
        },
        dup::{SubtreeCost, SubtreeCostMethod},
        explain::{render_card_grid, render_ranked_table},
        learned::LearnedCostModel,
        report::{IterationStat, NodeFusionDetail, PassDetail, PassOutcome},
        step::{
            BudgetMetric, OptimizationType, RegisterContext, StepContext, StepOutcome, StepPhase,
        },
        store::{OptStore, Registration},
    },
};

/// The bare name of the fused node, before the schema prefix it inherits from
/// the nodes it stands in front of.
const FUSED_BASE: &str = "dee_fused";

/// The discriminator column every branch of the fused `UNION ALL` leads with.
const KIND_COLUMN: &str = "kind";

// ---------------------------------------------------------------------------
// The pass
// ---------------------------------------------------------------------------

pub struct NodeFusionPass {
    /// Emit inlined *View* CTEs `AS MATERIALIZED`. Never applies to an inlined
    /// Table's CTE, which is materialized on its own account.
    materialize_ctes: bool,
    /// Materialize an inlined *View* CTE that more than one Table reads.
    ///
    /// Naive on purpose: it counts how many Table nodes reach the View through
    /// the fused graph and materializes it above one, without pricing what the
    /// View costs or what sharing it would save. That is the whole point of
    /// having it -- a floor to measure a cost-based rule against.
    naive_materialize_ctes: bool,
    /// When set, the exact set of node IDs whose CTEs are materialized --
    /// overriding both `materialize_ctes` and the Table default, so this is
    /// also how an intermediate Table's CTE is made plain.
    materialize_override: Option<Vec<String>>,
    /// Choose the materialized set by measurement instead of by rule -- the
    /// `adaptive` variant. See [`NodeFusionState`].
    adaptive: bool,
    objective: Objective,
    search_budget: usize,
    max_runs: usize,
    top_share: f64,
    cost_model: SubtreeCostMethod,
    /// An explicit spool rate, in seconds per byte, or `None` to scale the
    /// learned write constant by `spool_factor`.
    spool_seconds_per_byte: Option<f64>,
    /// The spool rate as a fraction of the write constant, or `None` for the
    /// per-dialect default.
    spool_factor: Option<f64>,
    /// The cost constants the adaptive search prices regions with, read from
    /// the store HMP publishes them to.
    ///
    /// Behind a `Mutex` rather than owned because the model is read across the
    /// awaits of every EXPLAIN the search issues, and a `MutexGuard` cannot be
    /// held across one -- the same snapshot-then-build dance `ranking_dup`
    /// does.
    ///
    /// The connector and the engine are deliberately *not* held beside it:
    /// [`StepContext`] hands both to every step, so storing them would be a
    /// second copy of something already in scope, and it would make the pass
    /// generic over two parameters that appear nowhere in its own signatures.
    learned: Arc<Mutex<LearnedCostModel>>,
    step_phase: StepPhase,
    explain_data: Option<ExplainData>,
    /// What the adaptive search decided, kept for `explain()`. `None` when the
    /// pass ran as a plain rewrite.
    search_data: Option<SearchExplain>,
}

/// What the adaptive search has decided so far, for the explain page.
#[derive(Debug, Clone)]
struct SearchExplain {
    objective: &'static str,
    phase: String,
    baseline_ms: i64,
    best_ms: i64,
    best_node_time_ms: i64,
    best_set: Option<Vec<String>>,
    default_set: Vec<String>,
    working_set: Vec<String>,
    candidates: Vec<NodeFusionCandidate>,
    runs_used: usize,
    iterations: usize,
    /// Why nothing was tried, when nothing was.
    no_candidates_because: Option<String>,
}

struct ExplainData {
    outcome: String,
    fused_id: Option<String>,
    /// The `WITH` chain, in emission order.
    ctes: Vec<ExplainCte>,
    /// `(kind, node id, column count)` in kind order.
    tables: Vec<(usize, String, usize)>,
    fused_columns: usize,
}

struct ExplainCte {
    node_id: String,
    role: &'static str,
    cte_name: String,
    readers: usize,
    materialized: bool,
    decidable: bool,
}

impl NodeFusionPass {
    pub fn new(
        materialize_ctes: bool,
        naive_materialize_ctes: bool,
        materialize_override: Option<Vec<String>>,
    ) -> Self {
        Self {
            materialize_ctes,
            naive_materialize_ctes,
            materialize_override,
            adaptive: false,
            objective: Objective::default(),
            search_budget: 32,
            max_runs: 1,
            top_share: 0.5,
            cost_model: SubtreeCostMethod::default(),
            spool_seconds_per_byte: None,
            spool_factor: None,
            learned: Arc::new(Mutex::new(LearnedCostModel::new())),
            step_phase: StepPhase::Before,
            explain_data: None,
            search_data: None,
        }
    }

    pub fn from_config(config: &OptimizerConfig) -> Self {
        let mut pass = Self::new(
            config.nodefusion_materialize_ctes,
            config.nodefusion_naive_materialize_ctes,
            config.nodefusion_materialize_ctes_override.clone(),
        );
        pass.adaptive = config.nodefusion_adaptive_materialize_ctes;
        pass.objective = config.nodefusion_objective;
        // At least one, matching HMP: a budget of zero would price nothing and
        // read downstream as "no candidate is worth a run".
        pass.search_budget = config.nodefusion_search_budget.max(1);
        pass.max_runs = config.nodefusion_max_runs.max(1);
        pass.top_share = if config.nodefusion_top_share > 0.0 && config.nodefusion_top_share <= 1.0 {
            config.nodefusion_top_share
        } else {
            0.5
        };
        pass.cost_model = config.nodefusion_cost_model;
        pass.spool_seconds_per_byte = config.nodefusion_spool_seconds_per_byte;
        pass.spool_factor = config.nodefusion_spool_cost_factor;
        pass
    }

    /// The contradiction [`OptimizerConfig::validate`] should already have
    /// caught, checked again here.
    ///
    /// The backstop, not the primary guard: a `dags.optimizer_config` row
    /// written before the rule existed decodes fine and would otherwise be
    /// obeyed. Refusing is the only safe answer -- picking one of the two
    /// would mean the pass quietly did something the config did not ask for.
    fn check_config(&self) -> Result<(), OptimizerError> {
        if !self.adaptive {
            return Ok(());
        }
        if self.naive_materialize_ctes {
            return Err(OptimizerError::Config(
                "nodefusion: the adaptive search and the naive reader-count rule are \
                 incompatible -- the naive rule is the floor the search exists to beat, \
                 and applying it to the search's own baseline would compare every \
                 candidate against the wrong control"
                    .into(),
            ));
        }
        if self.materialize_override.is_some() {
            return Err(OptimizerError::Config(
                "nodefusion: the adaptive search and an explicit materialize-CTE override \
                 are incompatible -- the override pins the exact set the search exists to \
                 find"
                    .into(),
            ));
        }
        if self.materialize_ctes {
            return Err(OptimizerError::Config(
                "nodefusion: the adaptive search and nodefusion_materialize_ctes are \
                 incompatible -- the global switch materializes every View CTE regardless \
                 of what the search decides"
                    .into(),
            ));
        }
        Ok(())
    }
}

/// Why a DAG cannot be fused.
///
/// Not an error: an unfusable DAG is a normal thing to hand this pass, and the
/// honest response is to leave it exactly as it was and say why. An
/// [`OptimizerError`] is reserved for a DAG that *is* fusable and whose
/// rewrite then failed, which is a bug rather than a shape.
#[derive(Debug, Clone)]
struct NotFusable(String);

impl std::fmt::Display for NotFusable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// ---------------------------------------------------------------------------
// Planning
// ---------------------------------------------------------------------------

/// One CTE in the fused query's `WITH` chain.
#[derive(Debug, Clone)]
struct CtePlan {
    node_id: String,
    cte_name: String,
    materialized: bool,
    /// A Table's CTE also becomes a branch of the `UNION ALL`; a View's does not.
    is_table: bool,
    /// The other CTEs this one reads. The `WITH` chain is a DAG of its own, and
    /// this is its edge set -- what the adaptive search walks to find the path a
    /// materialized CTE puts a barrier on. See [`barrier_chain`].
    reads: Vec<String>,
    /// How many CTEs and `UNION ALL` branches read this one.
    ///
    /// More than one is what makes its `MATERIALIZED` flag *decidable*: a CTE
    /// read once is unfolded once whatever the hint says, so there is nothing
    /// to decide and nothing to measure.
    readers: usize,
    /// Whether this CTE's flag is the search's to choose.
    decidable: bool,
}

/// One Table node's branch of the fused `UNION ALL`, and the projection that
/// reads it back out.
#[derive(Debug, Clone)]
struct TablePlan {
    node_id: String,
    kind: usize,
    cte_name: String,
    /// `(column name in the node's own schema, its name in the fused
    /// relation)`, in the node's schema order.
    columns: Vec<(String, String)>,
}

#[derive(Debug, Clone)]
struct FusionPlan {
    fused_id: String,
    /// The `WITH` chain, in a topological order of the fused nodes.
    ctes: Vec<CtePlan>,
    /// The Tables, in `kind` order.
    tables: Vec<TablePlan>,
    /// Every column of the fused relation after `kind`, in `kind` order.
    fused_columns: Vec<String>,
}

impl FusionPlan {
    /// The CTEs whose `MATERIALIZED` flag the adaptive search gets to choose,
    /// in the chain's own order.
    ///
    /// A CTE read more than once inside the fused query, whether it came from a
    /// View or from an intermediate Table. Both directions are in play: a
    /// shared View is a promotion from the plain default, and an intermediate
    /// Table -- materialized by default because it was authored as a barrier --
    /// is a demotion. That is a deliberate widening of what the rules can
    /// express, and the reason the default is only a default: a barrier the
    /// author put there to stop a *table* being recomputed is not obviously the
    /// right barrier inside a single query, and the search is what settles it.
    fn decidable(&self) -> Vec<String> {
        self.ctes
            .iter()
            .filter(|c| c.decidable)
            .map(|c| c.node_id.clone())
            .collect()
    }

    /// The set currently marked `AS MATERIALIZED`.
    fn materialized_set(&self) -> HashSet<String> {
        self.ctes
            .iter()
            .filter(|c| c.materialized)
            .map(|c| c.node_id.clone())
            .collect()
    }
}

/// What decides whether one CTE is emitted `AS MATERIALIZED`.
///
/// Split out of the pass so the adaptive search can plan the same DAG under a
/// set it is pricing without building a second pass to hold the setting. The
/// two arms are the two ways the question gets answered: by the configured
/// rules, or by naming the set outright.
#[derive(Debug, Clone, Copy)]
enum MaterializeRule<'a> {
    /// The configured rules: the Table default, plus the global and naive
    /// switches, unless an override names the set.
    Configured {
        all_views: bool,
        naive: bool,
        over: Option<&'a [String]>,
    },
    /// Exactly this set among the decidable CTEs; an undecidable CTE keeps the
    /// default it would have had.
    ///
    /// Every CTE whose default is `MATERIALIZED` is decidable -- that is only
    /// ever an intermediate Table, which by definition has more than one reader
    /// -- so in practice this is the exact set full stop. Spelled as "among the
    /// decidable" anyway, because relying on that coincidence would make the
    /// search silently wrong the day a new default is added.
    Exact(&'a HashSet<String>),
}

impl NodeFusionPass {
    /// Decide what to fuse under the pass's own configuration.
    fn plan(&self, dag: &Dag) -> Result<FusionPlan, NotFusable> {
        plan_fusion(
            dag,
            MaterializeRule::Configured {
                all_views: self.materialize_ctes,
                naive: self.naive_materialize_ctes,
                over: self.materialize_override.as_deref(),
            },
        )
    }

    /// Whether `node_id`'s CTE is emitted `AS MATERIALIZED` under this pass's
    /// configuration. See [`MaterializeRule::wants`], where the rule lives.
    ///
    /// Test-only: the rewrite itself goes through `MaterializeRule` directly.
    /// Kept because the tests that pin the rule's behaviour ask the question at
    /// the level a reader thinks about it -- of a configured pass, not of a rule
    /// value assembled by hand.
    #[cfg(test)]
    fn wants_materialized(
        &self,
        node_id: &str,
        intermediate_table: bool,
        shared_view: bool,
    ) -> bool {
        MaterializeRule::Configured {
            all_views: self.materialize_ctes,
            naive: self.naive_materialize_ctes,
            over: self.materialize_override.as_deref(),
        }
        .wants(node_id, intermediate_table, shared_view)
    }
}

impl MaterializeRule<'_> {
    /// Whether `node_id`'s CTE is emitted `AS MATERIALIZED`.
    ///
    /// Under [`MaterializeRule::Configured`] with no override, an
    /// *intermediate* Table's CTE is materialized -- a Table that feeds another
    /// fused node -- and everything else follows the global default. That Table
    /// was authored as a materialization barrier, and a plain CTE read by both
    /// its own `UNION ALL` branch and whatever is downstream of it is free to
    /// unfold a copy into each, which is the duplication this pass exists to
    /// remove. A Table that feeds nothing is read exactly once, by its own
    /// branch, so materializing it would buy nothing and is left to the global
    /// default like any other CTE.
    ///
    /// `naive` adds the Views more than one Table reaches -- `shared_view`. It
    /// is the narrower of the two global switches and is subsumed by
    /// `all_views`, which turns on every View regardless.
    ///
    /// An override names the exact set instead, so it is also the way to make
    /// an intermediate Table's CTE plain. A name matches either as the full
    /// node ID or as its bare table name, so `"wh"."main"."orders"` can be
    /// asked for as `orders`.
    ///
    /// [`MaterializeRule::Exact`] is the search's arm and is the set full stop.
    /// It needs no default branch for an undecidable CTE because every default
    /// that says `MATERIALIZED` is an intermediate Table, and an intermediate
    /// Table always has more than one reader and so is always decidable --- so
    /// an undecidable CTE's default is plain, and "not in the set" is already
    /// that.
    fn wants(&self, node_id: &str, intermediate_table: bool, shared_view: bool) -> bool {
        match self {
            MaterializeRule::Exact(set) => set.contains(node_id),
            MaterializeRule::Configured {
                over: Some(list), ..
            } => list
                .iter()
                .any(|n| n == node_id || bare_table_name(n) == bare_table_name(node_id)),
            MaterializeRule::Configured {
                all_views, naive, ..
            } => intermediate_table || *all_views || (*naive && shared_view),
        }
    }
}

/// Decide what to fuse, or why this DAG cannot be.
///
/// A free function rather than a method because the adaptive search plans the
/// same DAG dozens of times under different materialization sets, and it has no
/// business constructing a pass to do it.
fn plan_fusion(dag: &Dag, rule: MaterializeRule<'_>) -> Result<FusionPlan, NotFusable> {
    {
        let dialect = dialect_for_db(&dag.db);
        let topo = dag.nodes.topological_sort();
        if topo.len() < dag.nodes.num_nodes() {
            return Err(NotFusable(
                "the graph does not topologically sort, so there is no order to emit CTEs in"
                    .into(),
            ));
        }

        // Sorted by ID, not by topological position. A `kind` is a label
        // rather than an ordering, and the fused query's text has to be a
        // function of the DAG alone: `Graph::topological_sort` breaks ties
        // between independent nodes in `HashMap` iteration order, which varies
        // per process, so kinds taken from it would renumber run to run. dee's
        // DAGs are content-addressed, so a rewrite that is not byte-stable
        // mints a new version every time it runs over the same definition.
        let mut tables: Vec<String> = dag
            .nodes
            .nodes()
            .filter(|n| n.materialize == MaterializeMode::Table)
            .map(|n| n.id.clone())
            .collect();
        tables.sort();
        // One Table fused with itself is strictly worse than not fusing: the
        // fused node computes exactly what the node did, and the node then
        // copies it back out.
        if tables.len() < 2 {
            return Err(NotFusable(format!(
                "{} Table node(s); fusion needs at least two to share anything",
                tables.len()
            )));
        }

        // Everything the Tables read, transitively -- the Tables included,
        // since each one's own query becomes a CTE too.
        let closure = upstream_closure(dag, &tables);
        for id in &closure {
            let Some(node) = dag.nodes.get(id.clone()) else {
                return Err(NotFusable(format!("node '{id}' is not in the graph")));
            };
            if node.materialize == MaterializeMode::TempTable {
                return Err(NotFusable(format!(
                    "'{id}' is a TempTable upstream of a Table. A materialization search put it \
                     there on purpose, and folding it into a CTE would undo that decision"
                )));
            }
        }

        // A name for the fused node in the same catalog/schema as the nodes
        // that will read it, and not one already taken.
        let fused_id = {
            let sibling = &tables[0];
            let mut candidate = fused_node_name(sibling, FUSED_BASE);
            let mut n = 2;
            while dag.nodes.get(candidate.clone()).is_some() {
                candidate = fused_node_name(sibling, &format!("{FUSED_BASE}_{n}"));
                n += 1;
                if n > 64 {
                    return Err(NotFusable(
                        "could not find a free node ID for the fused node".into(),
                    ));
                }
            }
            candidate
        };

        // CTE names. Node IDs are qualified identifiers and a CTE name is a
        // single one, so the bare names are what is available -- and two
        // schemas may spell the same bare name, hence the dedupe.
        let mut taken: HashSet<String> = HashSet::new();
        let mut cte_names: HashMap<String, String> = HashMap::new();
        for id in &closure {
            let base = format!("n_{}", bare_table_name(id));
            let mut name = base.clone();
            let mut n = 2;
            while !taken.insert(name.clone()) {
                name = format!("{base}_{n}");
                n += 1;
            }
            cte_names.insert(id.clone(), name);
        }

        // The fused relation's columns: every Table's schema, concatenated in
        // kind order, each prefixed by its kind so two Tables spelling the
        // same column name stay distinct.
        let mut fused_columns: Vec<String> = Vec::new();
        let mut fused_taken: HashSet<String> = HashSet::new();
        let mut table_plans: Vec<TablePlan> = Vec::new();
        for (kind, id) in tables.iter().enumerate() {
            let node = dag.nodes.get(id.clone()).expect("checked above");
            let Some(schema) = node.schema.as_ref() else {
                return Err(NotFusable(format!(
                    "'{id}' has no resolved schema; call resolve_schemas before NodeFusion"
                )));
            };
            let fields = schema.flattened_fields();
            if fields.is_empty() {
                return Err(NotFusable(format!("'{id}' resolved to a schema with no columns")));
            }
            let mut columns = Vec::with_capacity(fields.len());
            for field in fields {
                let base = format!("k{kind}_{}", field.name());
                let mut name = base.clone();
                let mut n = 2;
                while !fused_taken.insert(name.clone()) {
                    name = format!("{base}_{n}");
                    n += 1;
                }
                fused_columns.push(name.clone());
                columns.push((field.name().clone(), name));
            }
            table_plans.push(TablePlan {
                node_id: id.clone(),
                kind,
                cte_name: cte_names[id].clone(),
                columns,
            });
        }

        // Which fused nodes another fused node reads. A Table in here is an
        // *intermediate* Table: something downstream of it was fused too, so
        // its CTE is read by more than its own `UNION ALL` branch.
        let mut feeds_another: HashSet<&String> = HashSet::new();
        for id in &closure {
            if let Some(node) = dag.nodes.get(id.clone()) {
                for dep in &node.depends_on {
                    if let Some(dep) = closure.get(dep) {
                        feeds_another.insert(dep);
                    }
                }
            }
        }

        // How many Tables reach each fused node. The naive materialization
        // rule reads off this: a View more than one Table reaches is a View
        // whose work the fused query would otherwise do more than once.
        let readers = table_readers(dag, &tables, &closure);

        // The `WITH` chain is emitted in a topological order, so a CTE is
        // always defined before the CTE that reads it -- and in a *stable*
        // one, so the same DAG always produces the same text.
        let table_set: HashSet<&String> = tables.iter().collect();
        let hint = supports_materialized_hint(dialect);
        // How many CTEs and branches read each one. A Table reads itself for
        // its own `UNION ALL` branch, which `table_readers` already counts, so
        // this is the number of places the fused query would have to compute it
        // if its CTE were unfolded rather than shared.
        let ctes: Vec<CtePlan> = stable_topological_order(dag, &closure)
            .iter()
            .map(|id| {
                let is_table = table_set.contains(id);
                let intermediate_table = is_table && feeds_another.contains(id);
                let reader_count = readers.get(id).copied().unwrap_or(0);
                let shared_view = !is_table && reader_count > 1;
                let reads: Vec<String> = dag
                    .nodes
                    .get(id.clone())
                    .map(|n| {
                        let mut deps: Vec<String> = n
                            .depends_on
                            .iter()
                            .filter(|d| closure.contains(*d))
                            .cloned()
                            .collect();
                        deps.sort();
                        deps
                    })
                    .unwrap_or_default();
                CtePlan {
                    node_id: id.clone(),
                    cte_name: cte_names[id].clone(),
                    materialized: hint
                        && rule.wants(id, intermediate_table, shared_view),
                    is_table,
                    reads,
                    readers: reader_count,
                    // Nothing to decide where the dialect ignores the hint, and
                    // nothing to decide for a CTE read once: it is computed
                    // exactly once either way, so both settings produce the same
                    // work and the search would be spending runs on a coin flip.
                    decidable: hint && reader_count > 1,
                }
            })
            .collect();

        Ok(FusionPlan {
            fused_id,
            ctes,
            tables: table_plans,
            fused_columns,
        })
    }
}

impl NodeFusionPass {


    /// Rewrite `dag` in place under this pass's configuration. Returns what
    /// happened, for the report.
    pub fn rewrite(&mut self, dag: &mut Dag) -> Result<PassOutcome, OptimizerError> {
        // Cloned rather than borrowed: the rule holds a slice of
        // `materialize_override`, and `rewrite_under` needs `&mut self` for the
        // explain data it fills in.
        let over = self.materialize_override.clone();
        let rule = MaterializeRule::Configured {
            all_views: self.materialize_ctes,
            naive: self.naive_materialize_ctes,
            over: over.as_deref(),
        };
        self.rewrite_under(dag, rule)
    }

    /// The CTEs the adaptive search would be allowed to decide about on `dag`:
    /// those read more than once inside the fused query.
    ///
    /// Public so tooling can enumerate the configurations the search can reach
    /// without reimplementing the reader count -- in particular the
    /// `nodefusion_validate` example, which checks every one of them against a
    /// real warehouse. Empty when the DAG cannot be fused at all.
    pub fn decidable_ctes(&self, dag: &Dag) -> Vec<String> {
        self.plan(dag).map(|p| p.decidable()).unwrap_or_default()
    }

    /// Rewrite `dag` in place with exactly `set` materialized -- what the
    /// adaptive search installs, for a trial and for the winner.
    ///
    /// Public for the same reason as [`Self::decidable_ctes`]: this is the code
    /// path a trial and a promotion both go through, so it is the one a
    /// semantic-equivalence check has to exercise. It is *not* the same arm as
    /// `nodefusion_materialize_ctes_override`, which also matches bare table
    /// names; checking the override instead would leave this path unchecked.
    pub fn rewrite_with(
        &mut self,
        dag: &mut Dag,
        set: &HashSet<String>,
    ) -> Result<PassOutcome, OptimizerError> {
        self.rewrite_under(dag, MaterializeRule::Exact(set))
    }

    fn rewrite_under(
        &mut self,
        dag: &mut Dag,
        rule: MaterializeRule<'_>,
    ) -> Result<PassOutcome, OptimizerError> {
        let plan = match plan_fusion(dag, rule) {
            Ok(plan) => plan,
            Err(why) => return Ok(self.not_fused(why)),
        };

        let dialect = dialect_for_db(&dag.db);
        let FusedSql {
            sql: fused_sql,
            verbatim_bodies,
        } = match build_fused_sql(dag, &plan, dialect) {
            Ok(built) => built,
            Err(why) => return Ok(self.not_fused(why)),
        };

        // Parse what we assembled before handing it to the executor, so a
        // malformed fused query is a failed optimization rather than a DAG
        // that only breaks at run time. This can only be asked when every body
        // was rewritten -- each of those round-tripped through the parser, so a
        // failure here is the frame. A body carried through verbatim may be
        // one the parser cannot read at all (that is why it was carried
        // through), and it would fail this check while saying nothing about
        // the frame. The engine is the authority on those either way.
        if verbatim_bodies == 0 {
            polyglot_sql::parse_one(&fused_sql, dialect).map_err(|e| {
                OptimizerError::Exec(format!(
                    "nodefusion built a fused query that does not parse ({e}); \
                     refusing to install it"
                ))
            })?;
        } else {
            debug!(
                "nodefusion: {verbatim_bodies} CTE bod(ies) are used as authored, so the \
                 assembled query was not parse-checked"
            );
        }

        // The fused node reads nothing but the warehouse's own source tables:
        // everything upstream of a Table is inside it now.
        dag.nodes
            .add_node(TransformNode {
                id: plan.fused_id.clone(),
                query_text: fused_sql,
                materialize: MaterializeMode::TempTable,
                depends_on: HashSet::new(),
                schema: None,
            })
            .map_err(|e| OptimizerError::Exec(format!("nodefusion: adding the fused node: {e}")))?;

        for table in &plan.tables {
            let projection = table
                .columns
                .iter()
                .map(|(original, fused)| format!("\"{fused}\" AS \"{original}\""))
                .collect::<Vec<_>>()
                .join(", ");
            let node = dag
                .nodes
                .get_mut(table.node_id.clone())
                .expect("planned from this graph");
            node.query_text = format!(
                "SELECT {projection} FROM {} WHERE {KIND_COLUMN} = {}",
                plan.fused_id, table.kind
            );
            // The node's own schema is unchanged -- the projection puts every
            // column back under its original name, in its original order -- so
            // it is deliberately left alone for whatever reads it next.
            node.depends_on = HashSet::from([plan.fused_id.clone()]);
        }

        dag.nodes.check().map_err(|e| {
            OptimizerError::Exec(format!("nodefusion produced an inconsistent graph: {e}"))
        })?;

        let views_inlined = plan.ctes.iter().filter(|c| !c.is_table).count();
        let materialized = plan.ctes.iter().filter(|c| c.materialized).count();
        let outcome = format!(
            "fused {} Table node(s) and {views_inlined} inlined View(s) into '{}'",
            plan.tables.len(),
            plan.fused_id
        );
        debug!("nodefusion: {outcome}");

        self.explain_data = Some(ExplainData {
            outcome: outcome.clone(),
            fused_id: Some(plan.fused_id.clone()),
            ctes: plan
                .ctes
                .iter()
                .map(|c| ExplainCte {
                    node_id: c.node_id.clone(),
                    role: if c.is_table { "Table" } else { "View" },
                    cte_name: c.cte_name.clone(),
                    readers: c.readers,
                    materialized: c.materialized,
                    decidable: c.decidable,
                })
                .collect(),
            tables: plan
                .tables
                .iter()
                .map(|t| (t.kind, t.node_id.clone(), t.columns.len()))
                .collect(),
            fused_columns: plan.fused_columns.len(),
        });

        let mut record = PassOutcome::empty().with_detail(PassDetail::NodeFusion(
            NodeFusionDetail {
                fused_node: Some(plan.fused_id.clone()),
                tables_fused: plan.tables.len(),
                views_inlined,
                materialized_ctes: materialized,
                fused_columns: plan.fused_columns.len(),
                outcome,
                adaptive: self.adaptive,
                objective: self.adaptive.then(|| self.objective.as_str().to_string()),
                baseline_runtime_ms: self
                    .search_data
                    .as_ref()
                    .map(|s| s.baseline_ms)
                    .filter(|ms| *ms > 0),
                final_runtime_ms: self
                    .search_data
                    .as_ref()
                    .map(|s| s.best_ms)
                    .filter(|ms| *ms != i64::MAX),
                candidates_costed: self
                    .search_data
                    .as_ref()
                    .map(|s| s.candidates.len())
                    .unwrap_or(0),
            },
        ));
        // One change: the DAG became one fused node. The Tables rewritten to
        // read it are that change's consequence, not separate decisions.
        record.changes_applied = 1;
        record.candidates_considered = 1;
        record.working_set_size = plan.ctes.len() as u32;
        Ok(record)
    }

    /// Record that this DAG was left alone, and why. The DAG itself is
    /// untouched -- everything up to here only read it.
    fn not_fused(&mut self, why: NotFusable) -> PassOutcome {
        debug!("nodefusion: not fusing this DAG -- {why}");
        let outcome = format!("not fused: {why}");
        self.explain_data = Some(ExplainData {
            outcome: outcome.clone(),
            fused_id: None,
            ctes: Vec::new(),
            tables: Vec::new(),
            fused_columns: 0,
        });
        PassOutcome::empty().with_detail(PassDetail::NodeFusion(NodeFusionDetail {
            fused_node: None,
            tables_fused: 0,
            views_inlined: 0,
            materialized_ctes: 0,
            fused_columns: 0,
            outcome,
            adaptive: self.adaptive,
            objective: self.adaptive.then(|| self.objective.as_str().to_string()),
            baseline_runtime_ms: None,
            final_runtime_ms: None,
            candidates_costed: 0,
        }))
    }

    /// What the adaptive search did, or nothing at all when the pass ran as a
    /// plain rewrite.
    ///
    /// Deliberately shows the *predicted* and the *measured* numbers side by
    /// side. A gap between the two is the interesting quantity here: the spool
    /// term the ranking leans on is modelled rather than fitted (see
    /// [`crate::opt::common::default_spool_factor`]), so a candidate whose
    /// measured order disagrees with its predicted order is evidence about the
    /// constant, and hiding one of the two numbers would throw that away.
    fn search_panel(&self) -> String {
        let Some(s) = &self.search_data else {
            return String::new();
        };

        let change = if s.baseline_ms > 0 && s.best_ms != i64::MAX {
            format!(
                "{:+.1}%",
                (s.best_ms - s.baseline_ms) as f64 / s.baseline_ms as f64 * 100.0
            )
        } else {
            "-".to_string()
        };
        let winner = match &s.best_set {
            Some(set) => describe_set(set),
            None => "the default rule".to_string(),
        };
        let cards = render_card_grid(&[
            ("Objective", s.objective.to_string()),
            ("Phase", s.phase.clone()),
            (
                "Baseline",
                if s.baseline_ms > 0 {
                    format!("{} ms", s.baseline_ms)
                } else {
                    "-".into()
                },
            ),
            (
                "Best",
                if s.best_ms == i64::MAX {
                    "-".into()
                } else {
                    format!("{} ms", s.best_ms)
                },
            ),
            ("Change", change),
            (
                "Best query time",
                if s.best_node_time_ms == i64::MAX {
                    "-".into()
                } else {
                    format!("{} ms", s.best_node_time_ms)
                },
            ),
            ("DAG runs", s.runs_used.to_string()),
            ("Iterations", s.iterations.to_string()),
            ("Candidates priced", s.candidates.len().to_string()),
            ("Working set", s.working_set.len().to_string()),
            ("Winner", winner),
            ("Default rule", describe_set(&s.default_set)),
        ]);

        let rows: Vec<Vec<String>> = s
            .candidates
            .iter()
            .enumerate()
            .map(|(i, c)| {
                vec![
                    (i + 1).to_string(),
                    describe_set(&c.set),
                    format!("{:.4}", c.cost),
                    c.makespan_s
                        .map(|m| format!("{m:.4}"))
                        .unwrap_or_else(|| "-".into()),
                    c.stages.to_string(),
                ]
            })
            .collect();
        let table = render_ranked_table(
            &["#", "Materialized set", "Work removed", "Path", "Stages"],
            &rows,
        );

        // Why it found nothing, when it found nothing. Without this a search
        // that converged on its baseline is indistinguishable from one that
        // never had a signal to search on.
        let why = match (&s.no_candidates_because, s.candidates.is_empty()) {
            (Some(reason), true) => format!(
                r#"<div class="subtle"><b>No candidate was tried.</b> {reason}.</div>"#
            ),
            _ => String::new(),
        };

        format!(
            r#"<div class="panel">
          <h2>The adaptive search</h2>
          {why}
          <div class="subtle">Candidate sets of CTEs to mark <code>AS MATERIALIZED</code>, priced by EXPLAINing the fused query with each set applied and ordered by the objective. <b>Work removed</b> is relative to the default rule, higher is better; <b>Path</b> is the longest chain through the WITH clause with that set materialized, lower is better. The two disagree because materializing removes repeated computation and inserts a barrier, which is why the objective picks both the order candidates are trialled in and the test each has to pass. The spool term behind both is modelled rather than measured.</div>
          {cards}
          {table}
        </div>"#
        )
    }

    fn explain_html(&self) -> String {
        let Some(data) = &self.explain_data else {
            return r#"<div class="panel"><p class="subtle">NodeFusionPass did not run.</p></div>"#
                .to_string();
        };

        let cards = render_card_grid(&[
            ("Outcome", data.outcome.clone()),
            (
                "Fused node",
                data.fused_id.clone().unwrap_or_else(|| "-".into()),
            ),
            ("CTEs", data.ctes.len().to_string()),
            (
                "Materialized CTEs",
                data.ctes.iter().filter(|c| c.materialized).count().to_string(),
            ),
            ("Fused columns", data.fused_columns.to_string()),
        ]);

        let cte_rows: Vec<Vec<String>> = data
            .ctes
            .iter()
            .enumerate()
            .map(|(i, cte)| {
                vec![
                    (i + 1).to_string(),
                    cte.node_id.clone(),
                    cte.role.to_string(),
                    cte.cte_name.clone(),
                    cte.readers.to_string(),
                    if cte.materialized { "MATERIALIZED" } else { "plain" }.to_string(),
                    // Whose decision it was. A CTE read once is computed once
                    // whichever way the hint goes, so there is nothing there for
                    // a search to decide and nothing for it to measure.
                    if cte.decidable { "search" } else { "fixed" }.to_string(),
                ]
            })
            .collect();
        let cte_table = render_ranked_table(
            &["#", "Node", "Role", "CTE", "Readers", "Hint", "Decided by"],
            &cte_rows,
        );

        let search_panel = self.search_panel();

        let kind_rows: Vec<Vec<String>> = data
            .tables
            .iter()
            .map(|(kind, node_id, cols)| {
                vec![kind.to_string(), node_id.clone(), cols.to_string()]
            })
            .collect();
        let kind_table = render_ranked_table(&["kind", "Table", "Columns"], &kind_rows);

        format!(
            r#"<div class="section-stack">
        {cards}
        {search_panel}
        <div class="panel">
          <h2>The WITH chain</h2>
          <div class="subtle">Emitted in the graph's own topological order, so every CTE is defined before the CTE that reads it. A Table's CTE is materialized by default -- it was authored as a materialization barrier, and a plain CTE read by several downstream CTEs may be unfolded into each of them, which is the duplication this pass removes.</div>
          {cte_table}
        </div>
        <div class="panel">
          <h2>The UNION ALL</h2>
          <div class="subtle">One branch per Table, discriminated by <code>kind</code>. The fused relation's columns are these schemas concatenated in kind order; each branch fills the columns that are not its own with NULL, and each Table reads its own back out under their original names.</div>
          {kind_table}
        </div>
      </div>"#
        )
    }
}

// ---------------------------------------------------------------------------
// The adaptive search's persisted state
// ---------------------------------------------------------------------------

const STATE_TABLE: &str = "opt_nodefusion_state";
const TRIALS_TABLE: &str = "opt_nodefusion_trials";
/// Seconds-per-byte constants, keyed by engine.
///
/// NodeFusion's own copy rather than a read of HMP's, because a pass may not
/// reach outside its own `opt_<name>_` namespace -- `OptStore` enforces it, and
/// the test that pins it is deliberate. The duplication is the price of that
/// isolation, and it is cheap: both passes fit the same constants from the same
/// executed plans, so a NodeFusion-only pipeline learns everything it needs from
/// its own baseline run rather than depending on HMP having been there first.
const LEARNED_TABLE: &str = "opt_nodefusion_learned_cost";

/// Where the adaptive search is, as persisted between steps.
///
/// The same shape [`crate::opt::hmp`] keeps, and for the same reason: a step
/// ends when the DAG runs, the next one may not happen for hours, and it may
/// happen in another process after a restart. Everything the search needs to
/// pick up where it left off has to survive in the metadata database.
///
/// `#[serde(default)]` on the struct as a whole so a row written by an older
/// build decodes rather than hard-failing --- removing a field is already
/// tolerated, adding one is not without this, and the failure mode is a live
/// search stranded by a deploy.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
struct NodeFusionState {
    /// `"baseline"` -- waiting for the first measurement, which is of the fused
    /// DAG under the default rule. `"searching"` -- working through candidate
    /// sets. `"converged"` -- finished; the winner has been installed.
    phase: String,
    baseline_ms: i64,
    best_ms: i64,
    best_node_time_ms: i64,
    /// The best set measured so far, or `None` while the default rule is still
    /// the incumbent.
    ///
    /// `Option` rather than a bare `Vec` because an empty set is a real
    /// configuration here --- "every CTE plain", which is what demoting the
    /// intermediate-Table default looks like --- and a bare empty `Vec` could
    /// not be told apart from "nothing has won yet". HMP does not have this
    /// problem: its empty combo *is* its baseline.
    best_set: Option<Vec<String>>,
    /// What the default rule materialized, recorded so the report can say what
    /// the search was measured against.
    default_set: Vec<String>,
    /// The ranked candidate CTEs the search will explore.
    working_set: Vec<String>,
    /// Each candidate CTE's score from the baseline, seeding the enumeration's
    /// priority queue.
    baseline_scores: HashMap<String, f64>,
    runs_used: usize,
    iterations: Vec<IterationStat>,
    /// Signatures of trial DAGs already measured, so two sets that fuse to the
    /// same text are not paid for twice.
    tried_sigs: Vec<String>,
    candidates: Vec<NodeFusionCandidate>,
    cursor: usize,
    /// Whether `candidates` has been enumerated yet.
    ///
    /// Not the same question as `candidates.is_empty()`: a state row persisted
    /// before the candidate list existed decodes with an empty one, and without
    /// this flag a live search would read that as "exhausted" and converge on
    /// its next step.
    candidates_built: bool,
    /// Why the search found nothing to try, when it found nothing.
    ///
    /// Persisted rather than logged because the run that discovers it and the
    /// report somebody reads are not the same moment, and "converged on the
    /// default rule" is the same outcome for four quite different causes.
    #[serde(default)]
    no_candidates_because: Option<String>,
    in_flight: Option<NodeFusionInFlight>,
}

impl Default for NodeFusionState {
    fn default() -> Self {
        Self {
            phase: "baseline".to_string(),
            baseline_ms: 0,
            best_ms: i64::MAX,
            best_node_time_ms: i64::MAX,
            best_set: None,
            default_set: Vec::new(),
            working_set: Vec::new(),
            baseline_scores: HashMap::new(),
            runs_used: 0,
            iterations: Vec::new(),
            tried_sigs: Vec::new(),
            candidates: Vec::new(),
            cursor: 0,
            candidates_built: false,
            no_candidates_because: None,
            in_flight: None,
        }
    }
}

/// One priced set of CTEs: what materializing exactly it is estimated to cost.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct NodeFusionCandidate {
    /// The decidable CTEs whose hint differs from the default rule's, which is
    /// what the search actually chose. See [`flip`].
    #[serde(default)]
    flips: Vec<String>,
    /// The CTEs to mark `AS MATERIALIZED` -- the configuration `flips` reaches,
    /// and what gets installed.
    set: Vec<String>,
    /// Work removed relative to the default rule. Higher is better.
    cost: f64,
    /// Predicted longest path through the `WITH` chain, in the cost model's
    /// units, or `None` when it could not be predicted.
    #[serde(default)]
    makespan_s: Option<f64>,
    /// Members lying on one chain, and so spooling in sequence.
    #[serde(default)]
    stages: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NodeFusionInFlight {
    set: Vec<String>,
    sig: String,
    #[serde(default)]
    makespan_s: Option<f64>,
}

/// What one round of ranking produced, and why it produced nothing when it did.
///
/// The `reason` is the point of the type. Every path out of `build_candidates`
/// that yields no candidates used to be a bare empty vector, and a search that
/// converges on its baseline looks identical whether the default rule really is
/// optimal, the engine ignores the hint, or the cost model has no constants
/// yet. Those call for completely different responses and only one of them is
/// good news.
#[derive(Debug, Default)]
struct SearchSetup {
    scores: HashMap<String, f64>,
    working_set: Vec<String>,
    candidates: Vec<NodeFusionCandidate>,
    /// Why there are no candidates, when there are none.
    reason: Option<String>,
}

impl SearchSetup {
    fn nothing(reason: &str) -> Self {
        debug!("nodefusion: not searching -- {reason}");
        Self {
            reason: Some(reason.to_string()),
            ..Default::default()
        }
    }
}

/// A candidate set, for a log line or a report label.
fn describe_set(set: &[String]) -> String {
    if set.is_empty() {
        "every CTE plain".to_string()
    } else {
        set.iter().map(|s| bare_table_name(s)).collect::<Vec<_>>().join(", ")
    }
}

// ---------------------------------------------------------------------------
// The adaptive search
// ---------------------------------------------------------------------------

/// What one CTE costs on its own, measured once and reused for every candidate
/// set that mentions it.
///
/// Measured *standalone* -- the node's own query, EXPLAINed by itself -- rather
/// than read out of the fused plan. Two reasons. A CTE's region inside the
/// fused plan is only findable when the engine kept it as a region, which is
/// exactly what a plain CTE does not do: it is unfolded into its readers and
/// there is nothing left to name. And a standalone measurement is the same
/// number whichever candidate set is being priced, so it can be taken `|closure|`
/// times at setup instead of once per set per member.
#[derive(Debug, Clone)]
struct CteProfile {
    /// What computing it once costs, in the cost model's units.
    compute: f64,
    /// What spooling it into a materialized CTE costs, in the same units.
    ///
    /// `None` when the model could not price it --- which is not a spool of
    /// zero, and is why a candidate that materializes this CTE comes back
    /// unranked on the makespan axis rather than cheap.
    spool: Option<f64>,
}

/// Prices a set of CTEs by what the fused query costs with exactly that set
/// marked `AS MATERIALIZED`.
///
/// # Why this does not go through [`crate::opt::dup`]
///
/// HMP prices a candidate set by building a graph minor and inlining each
/// member into its consumers as a materialized CTE --- an elaborate way to
/// simulate, in several queries, what one query would do. NodeFusion has
/// already *built* that query. So the honest measurement is to assemble the
/// fused SQL with the candidate set materialized and ask the engine to plan it,
/// which is both more faithful and less code: the plan says whether the engine
/// actually unfolded a plain CTE, where the simulation could only assume it.
///
/// One EXPLAIN per candidate set, plus one per CTE at setup. No DAG runs: the
/// runs are what the search spends on the ranked candidates afterwards.
struct FusedComboCoster<'a, C> {
    conn: &'a C,
    dag: &'a Dag,
    dialect: DialectType,
    coster: &'a dyn SubtreeCost,
    /// Per-CTE standalone cost, keyed by node id.
    profiles: &'a HashMap<String, CteProfile>,
    /// Which CTEs each CTE reads, and how many readers each has --- the shape
    /// of the `WITH` chain, taken from the plan once.
    chain: &'a [CtePlan],
    /// What the default rule materializes: the configuration every candidate is
    /// a departure from.
    default_set: &'a HashSet<String>,
    /// What the fused query costs under that rule.
    ///
    /// The search's zero. Every candidate's `cost` is this minus its own, so a
    /// positive cost means "removes work relative to the rule NodeFusion would
    /// have applied on its own" --- which is the comparison the naive floor is
    /// also measured on, and the sign convention [`ComboScore`] requires.
    baseline_work: f64,
}

/// The configuration reached by flipping `flips` in `default_set`.
///
/// # Why candidates are flips rather than sets
///
/// A candidate has to be able to say two different things -- "materialize this
/// View, which the rule leaves plain" and "leave this intermediate Table plain,
/// which the rule materializes" -- and an absolute set says them both at once
/// and neither clearly. Priced as an absolute set, the singleton `{stg}` does
/// not mean "is `stg` worth materializing"; it means "is materializing `stg`
/// and *demoting everything the rule turned on* better than the rule", which is
/// two decisions measured as one. A ranking built from those scores cannot tell
/// which half it is ranking.
///
/// As a flip set, the singleton `{stg}` is exactly "flip `stg`" and the
/// singleton `{base}` is exactly "flip `base`", whichever direction each flip
/// happens to go. The empty flip set is then the default rule itself, which
/// makes it the baseline rather than a candidate -- restoring the property HMP
/// relies on, where the empty combo *is* the control and pricing it would spend
/// a DAG run proving the DAG is itself.
fn flip(default_set: &HashSet<String>, flips: &[String]) -> HashSet<String> {
    let mut out = default_set.clone();
    for id in flips {
        if !out.remove(id) {
            out.insert(id.clone());
        }
    }
    out
}

impl<C> FusedComboCoster<'_, C>
where
    C: Connector + Send + Sync,
{
    /// What the fused query costs with exactly `set` materialized: the compute
    /// the engine's plan implies, and the modelled spool on top.
    ///
    /// Kept apart because only the first is a measurement, and only the first
    /// can tell whether the engine responded to the hint at all --- see
    /// [`Self::responds_to_the_hint`].
    async fn parts_of(&self, set: &HashSet<String>) -> Option<(f64, f64)> {
        let compute = self.compute_of(set).await?;
        Some((compute, self.spool_total(set).unwrap_or(0.0)))
    }

    /// What the fused query costs with exactly `set` materialized.
    async fn work_of(&self, set: &HashSet<String>) -> Option<f64> {
        let (compute, spool) = self.parts_of(set).await?;
        Some(compute + spool)
    }

    /// Whether this engine plans the *decidable* CTEs differently when they are
    /// hinted.
    ///
    /// # Why this check has to exist
    ///
    /// The search's whole signal is that materializing a CTE changes what the
    /// engine does. On DuckDB, for the CTEs the search is allowed to decide
    /// about, it does not --- and the reason is precise enough to be worth
    /// stating exactly, because a looser version of it is wrong.
    ///
    /// DuckDB already materializes a CTE that is referenced more than once. A
    /// decidable CTE is, by definition, one referenced more than once. So for
    /// every CTE the search may flip, the hint asks for what DuckDB was going
    /// to do anyway and the plan does not move: measured on this module's
    /// fixture, identical operators, identical counts, identical cost to the
    /// last digit.
    ///
    /// What is *not* true is the broader claim that the hint never changes a
    /// DuckDB plan. Materializing the singly-referenced CTEs does change it ---
    /// on the same fixture, four `CTE` nodes instead of one. Those are the CTEs
    /// the search deliberately leaves alone, because a CTE read once is
    /// computed once either way and flipping it buys nothing. Comparing over
    /// all the CTEs rather than the decidable ones would therefore see a
    /// difference, conclude the engine responds, and send the search off to
    /// rank a signal it cannot act on.
    ///
    /// The honest response to no signal is to say so. Modelling the
    /// duplication analytically instead --- charging a plain CTE once per
    /// reader --- would predict large wins on exactly the engine that is
    /// already avoiding them, which is worse than finding nothing: it is
    /// finding the wrong thing confidently.
    ///
    /// PostgreSQL is the backend this search is for. There a CTE is inlined by
    /// default since 12 and `MATERIALIZED` genuinely changes the plan, which is
    /// the difference the whole `nodefusion_*_materialize_ctes` family exists
    /// to exploit. That is documented Postgres behaviour rather than something
    /// verified here; a run against Postgres is what would confirm it.
    async fn responds_to_the_hint(&self, decidable: &HashSet<String>) -> bool {
        let none = HashSet::new();
        match (
            self.compute_of(&none).await,
            self.compute_of(decidable).await,
        ) {
            (Some(a), Some(b)) => a != b,
            // Unmeasurable is not the same as unresponsive. Let the search
            // proceed and decline per candidate, which is where the fallback
            // for that already lives.
            _ => true,
        }
    }

    /// The compute the engine's plan implies for `set`.
    async fn compute_of(&self, set: &HashSet<String>) -> Option<f64> {
        let plan = plan_fusion(self.dag, MaterializeRule::Exact(set)).ok()?;
        let FusedSql { sql, .. } = build_fused_sql(self.dag, &plan, self.dialect).ok()?;
        let raw = match self.conn.explain(&sql).await {
            Ok(Some(raw)) => raw,
            Ok(None) => return None,
            Err(e) => {
                debug!("nodefusion: the engine would not EXPLAIN a candidate fusion: {e}");
                return None;
            }
        };
        let roots = self.conn.parse_plan(&raw)?;
        self.coster.cost(&roots)
    }
}

impl<C> FusedComboCoster<'_, C>
where
    C: Connector + Send + Sync,
{
    /// The spool half, documented where it is charged.
    ///
    /// The engine's plan prices the compute; the spool is modelled on top,
    /// because a CTE materialization does not separate out in either backend's
    /// plan the way a write does.
    ///
    /// An unpriced spool charges nothing on the work axis and takes the
    /// candidate out of the makespan ranking entirely --- exactly the split HMP
    /// makes over the write term, for the same reason. The work axis is a
    /// difference between two quantities the engine measured the same way, and
    /// it stays rankable without the modelled correction; the path axis is
    /// *made of* the barrier the spool sizes, so without it there is no path to
    /// predict. Refusing both would be stricter than HMP and would mean the
    /// search declines to rank anything the moment one CTE's payload width is
    /// unavailable --- which, on a DAG whose default set is a single aggregate,
    /// is every candidate.
    ///
    /// All or nothing across the set on purpose. A partial sum would understate
    /// the charge and so silently favour the candidates whose spools could not
    /// be priced --- which are the large ones, the ones the charge exists to
    /// hold back. Same rule as
    /// [`crate::opt::dup::DuplicateCost::write_total`].
    ///
    /// All or nothing on purpose. A partial sum would understate the charge and
    /// so silently favour the candidates whose spools could not be priced ---
    /// which are the large ones, the ones the charge exists to hold back. Same
    /// rule as [`crate::opt::dup::DuplicateCost::write_total`].
    fn spool_total(&self, set: &HashSet<String>) -> Option<f64> {
        let mut total = 0.0;
        for id in set {
            total += self.profiles.get(id)?.spool?;
        }
        Some(total)
    }
}

#[async_trait]
impl<C> ComboCoster for FusedComboCoster<'_, C>
where
    C: Connector + Send + Sync,
{
    /// `members` is a set of *flips* against the default rule. See [`flip`].
    async fn price(&self, members: &[String]) -> Option<ComboScore> {
        let set = flip(self.default_set, members);
        let work = self.work_of(&set).await?;
        Some(ComboScore {
            // Work removed relative to the default rule. Higher is better, which
            // is what `order_by` sorts `QueryTime` on.
            cost: self.baseline_work - work,
            // Work plus the waiting the barriers add to it. See
            // [`barrier_chain`] for why it is a sum of the two rather than the
            // chain alone.
            makespan_s: barrier_chain(self.chain, self.profiles, &set).map(|c| work + c),
            stages: makespan_stages(self.chain, &set),
        })
    }
}

/// The longest chain of barriers in the `WITH` chain with `set` materialized.
///
/// This is the *waiting* materializing adds, not a wall-clock prediction on its
/// own --- and the difference matters, because the two rank differently.
/// Barriers only ever add to this number, so a ranking on the chain alone would
/// prefer the configuration with the fewest materializations every time,
/// whatever it cost in repeated work. That is not an objective, it is a
/// constant opinion.
///
/// So the makespan axis is `work + chain`: what the query does, plus what it
/// waits for. Both terms are in the cost model's own units and the axis is only
/// ever used to *order* candidates, so no conversion to seconds is needed --- and
/// none would be honest, since nothing here measures how much of the work
/// overlaps.
///
/// A consequence worth stating rather than hiding: inside one fused query the
/// two objectives are much closer together than they are for HMP. HMP's
/// makespan is a path through separately-executed nodes, where a materialized
/// View can add a whole stage while removing work elsewhere; here there is one
/// node, and the only thing separating wall clock from total work is how much
/// the barriers serialize. On a DAG whose candidates all have the same barrier
/// depth the two orders will coincide, and that is the model being right rather
/// than the setting being ignored.
///
/// # Why makespan is a different question inside one query
///
/// After fusion the DAG is one large node and a handful of cheap projections
/// off it, so [`crate::opt::makespan::estimate`] --- which prices DAG-level
/// table builds and the stages they serialize into --- has nothing left to
/// measure. But the `WITH` chain is a DAG in its own right, and materializing a
/// CTE is the same move one level down: it removes the repeated computation and
/// inserts a barrier that has to finish before its readers start.
///
/// So the walk is over CTEs rather than nodes, and the two kinds of CTE
/// contribute differently:
///
/// * a **materialized** CTE is a barrier. Its compute and its spool land on the
///   path once, and its readers start after it finishes.
/// * a **plain** CTE is pipelined into each reader, so it adds nothing of its
///   own to the path --- its work is already inside whichever reader's cost the
///   path runs through, and the reader does not wait for it.
///
/// That is the whole of the disagreement between the two objectives here.
/// Materializing cuts total work (one computation instead of `readers` of them)
/// and can lengthen the path (a barrier where there was a pipeline), which is
/// why ranking on one and accepting on the other finds improvements it then
/// refuses --- the failure [`Objective`] exists to prevent.
///
/// `None` when any materialized member could not be priced. Not a path of zero:
/// see [`CteProfile::spool`].
///
/// Uses [`crate::opt::makespan::critical_path`] rather than a walk of its own,
/// over a `Dag` synthesized from the chain. The weights are what differ between
/// the two searches; the longest-path argument is the same one.
fn barrier_chain(
    chain: &[CtePlan],
    profiles: &HashMap<String, CteProfile>,
    set: &HashSet<String>,
) -> Option<f64> {
    let mut cost_of: HashMap<String, f64> = HashMap::with_capacity(chain.len());
    for cte in chain {
        let profile = profiles.get(&cte.node_id)?;
        let weight = if set.contains(&cte.node_id) {
            // On the path once, compute plus spool.
            profile.compute + profile.spool?
        } else {
            // Pipelined into its readers: charged to them, not to the path
            // through it. The readers' own weights already contain it, because
            // each reader's standalone plan has the unfolded body inside it.
            0.0
        };
        cost_of.insert(cte.node_id.clone(), weight);
    }
    // A plain CTE contributes nothing but still carries the edge, so the walk
    // has to see it: dropping it would disconnect a materialized ancestor from
    // the materialized descendant that waits on it.
    let synthetic = chain_as_dag(chain);
    Some(crate::opt::makespan::critical_path(&synthetic, &cost_of).0)
}

/// How many members of `set` lie on one chain, and so spool one after another.
fn makespan_stages(chain: &[CtePlan], set: &HashSet<String>) -> usize {
    crate::opt::makespan::stages(&chain_as_dag(chain), set)
}

/// The `WITH` chain as a [`Dag`], so the makespan walk can be the shared one.
///
/// Only the edges matter --- `critical_path` and `stages` read `depends_on` and
/// nothing else --- so the query text is a placeholder and the materialize mode
/// is irrelevant. Building a real `Dag` rather than duplicating a topological
/// walk here is the point: two longest-path implementations would be two things
/// to keep agreeing.
fn chain_as_dag(chain: &[CtePlan]) -> Dag {
    let mut nodes: HashMap<String, TransformNode> = HashMap::with_capacity(chain.len());
    for cte in chain {
        nodes.insert(
            cte.node_id.clone(),
            TransformNode {
                id: cte.node_id.clone(),
                query_text: String::new(),
                materialize: MaterializeMode::View,
                depends_on: cte.reads.iter().cloned().collect(),
                schema: None,
            },
        );
    }
    Dag {
        db: "duckdb".to_string(),
        nodes: crate::graph::Graph::new(nodes),
        sources: Vec::new(),
        max_parallelism: None,
    }
}


// ---------------------------------------------------------------------------
// The adaptive search: ranking, pricing, and the step loop
// ---------------------------------------------------------------------------

impl NodeFusionPass {
    // -- state ---------------------------------------------------------------

    async fn load_state(
        &self,
        store: &dyn OptStore,
        dag_id: &str,
    ) -> Result<Option<NodeFusionState>, OptimizerError> {
        let rows = match store
            .query(
                &format!("SELECT state FROM {STATE_TABLE} WHERE dag_id = ?"),
                &[serde_json::json!(dag_id)],
            )
            .await
        {
            Ok(rows) => rows,
            // Not registered, or deregistered while a run was in flight.
            Err(e) if crate::opt::store::is_missing_table(&e) => return Ok(None),
            Err(e) => return Err(e.into()),
        };
        let Some(row) = rows.first() else {
            return Ok(None);
        };
        let raw = row.get("state").and_then(|v| v.as_str()).unwrap_or("");
        serde_json::from_str(raw)
            .map(Some)
            .map_err(|e| OptimizerError::Store(crate::opt::OptStoreError::Decode(e.to_string())))
    }

    async fn save_state(
        &self,
        store: &dyn OptStore,
        dag_id: &str,
        state: &NodeFusionState,
    ) -> Result<(), OptimizerError> {
        let encoded = serde_json::to_string(state)
            .map_err(|e| OptimizerError::Store(crate::opt::OptStoreError::Decode(e.to_string())))?;
        store
            .execute(
                &format!("DELETE FROM {STATE_TABLE} WHERE dag_id = ?"),
                &[serde_json::json!(dag_id)],
            )
            .await?;
        store
            .execute(
                &format!(
                    "INSERT INTO {STATE_TABLE} (dag_id, state, updated_at) \
                     VALUES (?, ?, now())"
                ),
                &[serde_json::json!(dag_id), serde_json::json!(encoded)],
            )
            .await?;
        Ok(())
    }

    async fn record_trial(
        &self,
        store: &dyn OptStore,
        dag_id: &str,
        run_id: &str,
        state: &NodeFusionState,
        runtime_ms: i64,
        improved: bool,
    ) -> Result<(), OptimizerError> {
        let set = state
            .in_flight
            .as_ref()
            .map(|f| f.set.join(","))
            .unwrap_or_default();
        store
            .execute(
                &format!(
                    "INSERT INTO {TRIALS_TABLE} \
                     (dag_id, run_id, iteration, cte_set, runtime_ms, improved, recorded_at) \
                     VALUES (?, ?, ?, ?, ?, ?, now())"
                ),
                &[
                    serde_json::json!(dag_id),
                    serde_json::json!(run_id),
                    serde_json::json!(state.iterations.len() as i64),
                    serde_json::json!(set),
                    serde_json::json!(runtime_ms),
                    serde_json::json!(improved),
                ],
            )
            .await?;
        Ok(())
    }

    // -- pricing -------------------------------------------------------------

    /// The spool rate to charge, as a fraction of the learned write constant.
    ///
    /// An explicit `nodefusion_spool_seconds_per_byte` is expressed *against*
    /// the write constant rather than applied directly, because the write
    /// constant is the per-byte rate the model already knows how to apply to a
    /// plan region --- it has the payload width, which is not otherwise
    /// reachable from here. The arithmetic is exact: `(rate / write) * write`
    /// is `rate`.
    ///
    /// When there is no write constant to express it against, the explicit rate
    /// cannot be applied and the configured factor stands in, with a warning.
    /// Silently ignoring a number somebody set is the one thing not to do.
    fn spool_factor_for(
        &self,
        model: &LearnedCostModel,
        path: &str,
        dialect: DialectType,
    ) -> f64 {
        if let Some(rate) = self.spool_seconds_per_byte {
            match model.write_rate_for(path).map(|(r, _)| r) {
                Some(write) if write > 0.0 => return rate / write,
                _ => warn!(
                    "nodefusion: nodefusion_spool_seconds_per_byte is set to {rate} but the \
                     model has no write constant on '{path}' to express it against; falling \
                     back to the spool factor"
                ),
            }
        }
        self.spool_factor
            .unwrap_or_else(|| default_spool_factor(dialect))
    }

    /// What each CTE costs on its own, from one EXPLAIN of the fused query with
    /// *every* CTE materialized.
    ///
    /// Materializing all of them is what makes the regions disjoint: each CTE's
    /// body then reads its ancestors by CTE name instead of containing their
    /// computation, so `find_subplan` returns that CTE's work and nothing
    /// else. The same trick [`crate::opt::dup::build_once_cost`] uses, and for
    /// the same reason --- a chain priced by summing standalone plans counts the
    /// shared base scans once per member.
    ///
    /// A CTE whose region is missing from the plan gets no profile, and every
    /// candidate set that mentions it then comes back unpriced. That is the
    /// intended behaviour: the alternative is charging it zero, which would rank
    /// it as the cheapest thing in the chain.
    async fn profile_ctes<C>(
        &self,
        conn: &C,
        dag: &Dag,
        plan: &FusionPlan,
        coster: &dyn SubtreeCost,
        model: &LearnedCostModel,
        dialect: DialectType,
    ) -> Option<HashMap<String, CteProfile>>
    where
        C: Connector + Send + Sync,
    {
        let all: HashSet<String> = plan.ctes.iter().map(|c| c.node_id.clone()).collect();
        let probe = plan_fusion(dag, MaterializeRule::Exact(&all)).ok()?;
        let FusedSql { sql, .. } = build_fused_sql(dag, &probe, dialect).ok()?;
        let raw = match conn.explain(&sql).await {
            Ok(Some(raw)) => raw,
            Ok(None) => {
                debug!("nodefusion: the engine returned no plan for the profiling probe");
                return None;
            }
            Err(e) => {
                warn!("nodefusion: could not EXPLAIN the profiling probe: {e}");
                return None;
            }
        };
        let roots = conn.parse_plan(&raw)?;

        let mut profiles: HashMap<String, CteProfile> = HashMap::new();
        for cte in &probe.ctes {
            let Some(region) = crate::plan::find_subplan(&roots, &cte.cte_name) else {
                debug!(
                    "nodefusion: '{}' has no region in the probe plan; sets naming it will be \
                     unpriced",
                    cte.node_id
                );
                continue;
            };
            let Some(compute) = coster.cost(std::slice::from_ref(region)) else {
                continue;
            };
            // Which sink the engine would give this CTE, asked of the engine
            // rather than guessed: the probe plan cannot answer, because a CTE
            // is tagged `CTE` and not a write. No fallback -- an engine that
            // will not name the path leaves the spool unpriced, which is not a
            // spool of zero.
            let body = dag
                .nodes
                .get(cte.node_id.clone())
                .map(|n| n.query_text.clone())
                .unwrap_or_default();
            let spool = match conn.write_path_for(&body).await {
                Ok(Some(path)) => {
                    let factor = self.spool_factor_for(model, &path, dialect);
                    region.rows().and_then(|rows| {
                        coster.spool_cost(&path, std::slice::from_ref(region), rows, factor)
                    })
                }
                Ok(None) => None,
                Err(e) => {
                    warn!(
                        "nodefusion: could not ask {} which write path it would use: {e}",
                        cte.node_id
                    );
                    None
                }
            };
            profiles.insert(cte.node_id.clone(), CteProfile { compute, spool });
        }
        (!profiles.is_empty()).then_some(profiles)
    }

    /// Rank the decidable CTEs, then price sets of them.
    ///
    /// Returns `(ranking, candidates)`. The ranking scores each CTE on its own
    /// through exactly the measurement the sets use --- never a different unit,
    /// which is the mistake that makes a working-set prefix meaningless.
    async fn build_candidates<C>(
        &self,
        conn: &C,
        dag: &Dag,
        model: &LearnedCostModel,
    ) -> SearchSetup
    where
        C: Connector + Send + Sync,
    {
        let dialect = dialect_for_db(&dag.db);
        let Ok(plan) = self.plan(dag) else {
            return SearchSetup::nothing("this DAG cannot be fused, so there is no WITH chain to search");
        };
        let decidable = plan.decidable();
        if decidable.is_empty() {
            return SearchSetup::nothing(
                "no CTE in the fused query has more than one reader, so every one of them is \
                 computed once whichever way the hint goes",
            );
        }

        let coster = self.cost_model.coster(model);
        let Some(profiles) = self
            .profile_ctes(conn, dag, &plan, coster.as_ref(), model, dialect)
            .await
        else {
            return SearchSetup::nothing(
                "no CTE could be priced -- the engine answered no EXPLAIN, or the cost model \
                 has no constants fitted for this backend yet",
            );
        };

        let default_set = plan.materialized_set();
        let fused = FusedComboCoster {
            conn,
            dag,
            dialect,
            coster: coster.as_ref(),
            profiles: &profiles,
            chain: &plan.ctes,
            default_set: &default_set,
            // The zero the candidates are read against: what the default rule
            // costs. Taken before anything is ranked, because every score below
            // is a difference from it.
            baseline_work: 0.0,
        };
        let Some(baseline_work) = fused.work_of(&default_set).await else {
            return SearchSetup::nothing(
                "the default fusion could not be priced, so there is nothing to measure \
                 candidates against",
            );
        };
        let fused = FusedComboCoster {
            baseline_work,
            ..fused
        };

        // Before spending anything: does this engine plan the two extremes
        // differently at all? On DuckDB it does not, and a search that ranked
        // regardless would be ranking rounding error. See
        // `FusedComboCoster::responds_to_the_hint`.
        let decidable_set: HashSet<String> = decidable.iter().cloned().collect();
        if !fused.responds_to_the_hint(&decidable_set).await {
            return SearchSetup::nothing(
                "this engine plans the fused query identically whether the CTEs the search \
                 may decide about are MATERIALIZED or not, so there is nothing for a \
                 cost-based rule to find. DuckDB is such an engine: it already materializes \
                 a CTE referenced more than once, which is exactly the set of CTEs that are \
                 decidable, so the hint asks for what it was going to do anyway",
            );
        }

        // Singletons, in exactly the units the sets are priced in.
        let mut scores: Vec<(String, f64)> = Vec::new();
        for id in &decidable {
            if let Some(s) = fused.price(std::slice::from_ref(id)).await
                && s.cost > 0.0
            {
                scores.push((id.clone(), s.cost));
            }
        }
        scores.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        let singletons: HashMap<String, f64> = scores.iter().cloned().collect();
        let working_set = crate::opt::combo::working_set_from(&scores, self.top_share);
        if working_set.is_empty() {
            // Every decidable CTE is already as good as it gets under the
            // default rule. Not a failure -- a finding, and the honest answer is
            // to keep the default rather than spend runs proving it.
            return SearchSetup {
                scores: singletons,
                working_set: Vec::new(),
                candidates: Vec::new(),
                reason: Some(
                    "no single flip of the default rule removes any work, so there is no \
                     ranking to search from"
                        .to_string(),
                ),
            };
        }

        let ordered = canonical_order(dag, &working_set);
        let costed: Vec<CostedCombo> = crate::opt::combo::search_combos_with(
            &fused,
            &ordered,
            &singletons,
            self.search_budget,
            self.objective,
        )
        .await;

        let candidates: Vec<NodeFusionCandidate> = costed
            .into_iter()
            .map(|c| {
                // Both, because they answer different questions. `flips` is what
                // the search decided; `set` is what gets installed, and is the
                // only one a reader of the report can check against the emitted
                // SQL.
                let mut set: Vec<String> = flip(&default_set, &c.combo).into_iter().collect();
                set.sort();
                NodeFusionCandidate {
                    flips: c.combo,
                    set,
                    cost: c.cost,
                    makespan_s: c.makespan_s,
                    stages: c.stages,
                }
            })
            .collect();
        debug!(
            "nodefusion: {} candidate CTE set(s) priced from a working set of {}",
            candidates.len(),
            working_set.len()
        );
        let reason = candidates
            .is_empty()
            .then(|| "no candidate set removes work, once its spool is charged".to_string());
        SearchSetup {
            scores: singletons,
            working_set,
            candidates,
            reason,
        }
    }

    // -- the loop ------------------------------------------------------------

    /// Which measure this search's trial budget caps.
    ///
    /// The objective's own, so "already lost" means the same thing to the
    /// cancellation as it does to the accept test.
    fn budget_metric(&self) -> BudgetMetric {
        match self.objective {
            Objective::Makespan => BudgetMetric::WallClock,
            Objective::QueryTime => BudgetMetric::NodeTime,
        }
    }

    /// Fuse `dag` under `set`, or under the default rule when `set` is `None`.
    fn apply(&mut self, dag: &mut Dag, set: Option<&[String]>) -> Result<PassOutcome, OptimizerError> {
        match set {
            None => self.rewrite(dag),
            Some(members) => {
                let exact: HashSet<String> = members.iter().cloned().collect();
                self.rewrite_with(dag, &exact)
            }
        }
    }

    async fn step_before_adaptive<C, E>(
        &mut self,
        ctx: &mut StepContext<'_, C, E>,
    ) -> Result<StepOutcome, OptimizerError>
    where
        C: Connector + Send + Sync + 'static,
        E: Executor<C> + Send + Sync,
    {
        let Some(mut state) = self.load_state(ctx.store, ctx.dag_id).await? else {
            // Not registered, or registered and never stepped.
            return Ok(StepOutcome::Idle);
        };

        match state.phase.as_str() {
            // Unlike HMP, the baseline is *not* the DAG as it stands. HMP's
            // baseline is the authored DAG and its `Before` step does nothing;
            // ours has to be the fused DAG under the default rule, or every
            // candidate would be compared against an unfused control and would
            // "win" for reasons that have nothing to do with which CTEs are
            // materialized.
            "baseline" => {
                let record = self.apply(ctx.dag, None)?;
                Ok(StepOutcome::Trial {
                    label: "fusion under the default rule (baseline)".to_string(),
                    budget_ms: None,
                    budget_metric: self.budget_metric(),
                    fallback: None,
                    reuse: Default::default(),
                    record: Box::new(record),
                })
            }
            "converged" => Ok(StepOutcome::Idle),
            "searching" => {
                if let Some(in_flight) = state.in_flight.clone() {
                    // A trial was proposed and never reported on -- a run that
                    // never happened, or one whose `After` step was missed.
                    // Propose it again rather than moving on, so the candidate
                    // is measured rather than silently skipped.
                    let record = self.apply(ctx.dag, Some(&in_flight.set))?;
                    return Ok(StepOutcome::Trial {
                        label: describe_set(&in_flight.set),
                        budget_ms: None,
                        budget_metric: self.budget_metric(),
                        fallback: None,
                        reuse: Default::default(),
                        record: Box::new(record),
                    });
                }
                if !state.candidates_built {
                    // `ctx.dag` is the committed definition here -- unfused,
                    // which is what a candidate fusion has to be planned from.
                    let model = self.snapshot_model(ctx.store, ctx.conn.as_ref()).await;
                    let setup = self
                        .build_candidates(ctx.conn.clone().as_ref(), ctx.dag, &model)
                        .await;
                    state.baseline_scores = setup.scores;
                    state.working_set = setup.working_set;
                    state.candidates = setup.candidates;
                    state.no_candidates_because = setup.reason;
                    state.candidates_built = true;
                    self.save_state(ctx.store, ctx.dag_id, &state).await?;
                }
                // One run for the baseline plus `max_runs` for candidates.
                if state.runs_used >= self.max_runs + 1 {
                    return self.converge(ctx, state).await;
                }
                while state.cursor < state.candidates.len() {
                    let candidate = state.candidates[state.cursor].clone();
                    state.cursor += 1;
                    let mut trial = ctx.dag.clone();
                    if self.apply(&mut trial, Some(&candidate.set)).is_err() {
                        continue;
                    }
                    let sig = crate::opt::hmp::dag_signature(&trial);
                    if state.tried_sigs.contains(&sig) {
                        continue;
                    }
                    state.tried_sigs.push(sig.clone());
                    state.in_flight = Some(NodeFusionInFlight {
                        set: candidate.set.clone(),
                        sig,
                        makespan_s: candidate.makespan_s,
                    });
                    let record = self.apply(ctx.dag, Some(&candidate.set))?;
                    self.save_state(ctx.store, ctx.dag_id, &state).await?;
                    return Ok(StepOutcome::Trial {
                        label: describe_set(&candidate.set),
                        budget_ms: None,
                        budget_metric: self.budget_metric(),
                        fallback: None,
                        reuse: Default::default(),
                        record: Box::new(record),
                    });
                }
                self.converge(ctx, state).await
            }
            other => {
                warn!("nodefusion: unknown search phase '{other}'; doing nothing");
                Ok(StepOutcome::Idle)
            }
        }
    }

    /// Install the winner and stop.
    ///
    /// The winner is applied to the *committed* DAG rather than to whatever the
    /// last trial left behind, which is what keeps a rejected candidate from
    /// being persisted because it happened to run last.
    async fn converge<C, E>(
        &mut self,
        ctx: &mut StepContext<'_, C, E>,
        mut state: NodeFusionState,
    ) -> Result<StepOutcome, OptimizerError>
    where
        C: Connector + Send + Sync + 'static,
        E: Executor<C> + Send + Sync,
    {
        state.phase = "converged".to_string();
        let best = state.best_set.clone();
        let record = self.apply(ctx.dag, best.as_deref())?;
        debug!(
            "nodefusion: converged on {}",
            best.as_deref().map(describe_set).unwrap_or_else(|| "the default rule".into())
        );
        self.remember_search(&state);
        self.save_state(ctx.store, ctx.dag_id, &state).await?;
        Ok(StepOutcome::Promote {
            record: Box::new(record),
        })
    }

    async fn step_after_adaptive<C, E>(
        &mut self,
        ctx: &mut StepContext<'_, C, E>,
    ) -> Result<StepOutcome, OptimizerError>
    where
        C: Connector + Send + Sync + 'static,
        E: Executor<C> + Send + Sync,
    {
        let Some(mut state) = self.load_state(ctx.store, ctx.dag_id).await? else {
            return Ok(StepOutcome::Idle);
        };
        let Some(run) = ctx.run.clone() else {
            return Ok(StepOutcome::Idle);
        };
        if !run.is_measured() {
            // A warmup's timing is deliberately discarded; comparing candidates
            // against one would compare cold-cache cost.
            return Ok(StepOutcome::Idle);
        }
        let Some(stats) = ctx.stats().cloned() else {
            return self.reject_censored(ctx, state, &run.run_id).await;
        };
        let runtime_ms = stats.duration.num_milliseconds();
        let node_time_ms = stats.node_time_ms();

        if state.phase == "baseline" {
            state.baseline_ms = runtime_ms;
            state.best_ms = runtime_ms;
            state.best_node_time_ms = node_time_ms;
            state.runs_used = 1;
            state.phase = "searching".to_string();
            if let Ok(plan) = self.plan(ctx.dag) {
                state.default_set = {
                    let mut s: Vec<String> = plan.materialized_set().into_iter().collect();
                    s.sort();
                    s
                };
            }

            // The candidates are *not* priced here, unlike HMP, which prices
            // them on its baseline's `After` step. `ctx.dag` on an `After` step
            // is the fused trial, and pricing candidate fusions of an
            // already-fused DAG is meaningless -- fusing it again is refused
            // outright. The next `Before` step sees the committed, unfused
            // definition, which is the DAG the candidates are fusions of, so
            // that is where they get built. `candidates_built` is what carries
            // the "still to do" across the gap.
            // Fit the constants to the baseline's own plans before anything is
            // priced. The fused node is a TempTable, so this run carried an
            // EXPLAIN ANALYZE of the whole fused query -- every operator the
            // candidates will be priced on, measured on this machine.
            self.learn_from(ctx.conn.as_ref(), ctx.dag, &stats);
            self.publish_learned(ctx.store, ctx.conn.as_ref()).await;
            state.iterations.push(
                IterationStat::new(1, runtime_ms)
                    .with_outcome("ok")
                    .with_run_cost(ctx),
            );
            self.record_trial(ctx.store, ctx.dag_id, &run.run_id, &state, runtime_ms, false)
                .await?;
            self.save_state(ctx.store, ctx.dag_id, &state).await?;
            self.remember_search(&state);
            return Ok(StepOutcome::Idle);
        }

        let Some(in_flight) = state.in_flight.take() else {
            // A run this search did not cause is not its business.
            return Ok(StepOutcome::Idle);
        };
        state.runs_used += 1;

        // Promote on the measure the search was ordering by. Ranking on one and
        // accepting on the other is how a search finds improvements and then
        // refuses every one of them.
        let (improved, measure, was, now) = match self.objective {
            Objective::Makespan => (
                runtime_ms < state.best_ms,
                "makespan",
                state.best_ms,
                runtime_ms,
            ),
            Objective::QueryTime => (
                node_time_ms < state.best_node_time_ms,
                "query time",
                state.best_node_time_ms,
                node_time_ms,
            ),
        };
        state.iterations.push(
            IterationStat::new(state.iterations.len() + 1, runtime_ms)
                .with_combo(in_flight.set.clone())
                .with_predicted_makespan(in_flight.makespan_s)
                .with_outcome("ok")
                .with_run_cost(ctx),
        );
        if improved {
            debug!(
                "nodefusion: {} improved {measure}: {was}ms -> {now}ms",
                describe_set(&in_flight.set)
            );
            state.best_set = Some(in_flight.set.clone());
            state.best_ms = runtime_ms;
            state.best_node_time_ms = node_time_ms;
        } else {
            debug!(
                "nodefusion: {} did not improve {measure} ({was}ms -> {now}ms)",
                describe_set(&in_flight.set)
            );
        }

        // A trial is another executed plan. The search does not re-rank between
        // rounds, so this does not change what is tried next -- it is what keeps
        // the constants improving for the runs after this one.
        self.learn_from(ctx.conn.as_ref(), ctx.dag, &stats);
        self.publish_learned(ctx.store, ctx.conn.as_ref()).await;

        self.record_trial(
            ctx.store,
            ctx.dag_id,
            &run.run_id,
            &state,
            runtime_ms,
            improved,
        )
        .await?;
        self.save_state(ctx.store, ctx.dag_id, &state).await?;
        self.remember_search(&state);
        Ok(StepOutcome::Idle)
    }

    /// File a trial that produced no usable measurement and move the search on.
    async fn reject_censored<C, E>(
        &mut self,
        ctx: &mut StepContext<'_, C, E>,
        mut state: NodeFusionState,
        run_id: &str,
    ) -> Result<StepOutcome, OptimizerError>
    where
        C: Connector + Send + Sync + 'static,
        E: Executor<C> + Send + Sync,
    {
        let Some(in_flight) = state.in_flight.take() else {
            return Ok(StepOutcome::Idle);
        };
        if state.phase == "baseline" {
            // The baseline is the control; recording a censored one would leave
            // every later trial compared against a number no run produced.
            return Ok(StepOutcome::Idle);
        }
        state.runs_used += 1;
        debug!(
            "nodefusion: {} produced no usable measurement; rejecting as censored",
            describe_set(&in_flight.set)
        );
        state.iterations.push(
            IterationStat::new(state.iterations.len() + 1, i64::MAX)
                .with_combo(in_flight.set.clone())
                .with_predicted_makespan(in_flight.makespan_s)
                .with_outcome("cancelled")
                .with_run_cost(ctx),
        );
        if !state.tried_sigs.contains(&in_flight.sig) {
            state.tried_sigs.push(in_flight.sig.clone());
        }
        self.record_trial(ctx.store, ctx.dag_id, run_id, &state, i64::MAX, false)
            .await?;
        self.save_state(ctx.store, ctx.dag_id, &state).await?;
        Ok(StepOutcome::Idle)
    }

    /// Keep what the search has decided so far, so `explain()` can render it
    /// after the fact.
    fn remember_search(&mut self, state: &NodeFusionState) {
        self.search_data = Some(SearchExplain {
            objective: self.objective.as_str(),
            phase: state.phase.clone(),
            baseline_ms: state.baseline_ms,
            best_ms: state.best_ms,
            best_node_time_ms: state.best_node_time_ms,
            best_set: state.best_set.clone(),
            default_set: state.default_set.clone(),
            working_set: state.working_set.clone(),
            candidates: state.candidates.clone(),
            runs_used: state.runs_used,
            iterations: state.iterations.len(),
            no_candidates_because: state.no_candidates_because.clone(),
        });
    }

    /// Whether this configuration prices anything with the seconds-per-byte
    /// constants, and so has a reason to read and write them.
    fn uses_learned_constants(&self) -> bool {
        self.adaptive && self.cost_model == SubtreeCostMethod::LearnedCost
    }

    /// A stable identity for the engine the constants belong to.
    ///
    /// Seconds per byte is a property of an engine on a machine. Pooling two
    /// engines' samples under one key gives a constant that describes neither.
    async fn backend_key<C: Connector + Send + Sync>(&self, conn: &C) -> Option<String> {
        match conn.cost_backend_key().await {
            Ok(Some(key)) => Some(key),
            Ok(None) => {
                warn!(
                    "nodefusion: this connector does not identify its backend, so the learned \
                     cost constants will not be persisted; they would be pooled with every \
                     other engine on this store"
                );
                None
            }
            Err(e) => {
                warn!("nodefusion: could not identify the backend to key cost constants by: {e}");
                None
            }
        }
    }

    /// Fit the cost constants to what the run just executed.
    ///
    /// Without this the adaptive search prices nothing on a pipeline that has
    /// never run HMP: `SubtreeCostMethod::LearnedCost` declines rather than
    /// inventing a constant, every candidate comes back unpriced, and the search
    /// converges on its baseline having measured one run for nothing. It would
    /// look exactly like "the default rule is already optimal".
    ///
    /// The baseline run is the natural place to learn from and the fused node is
    /// the natural thing to learn from: it is a `TempTable`, so the run carries
    /// an EXPLAIN ANALYZE of the whole fused query, which is every operator the
    /// candidates will be priced on.
    fn learn_from<C: Connector + Send + Sync>(&self, conn: &C, dag: &Dag, stats: &ExecStats) {
        let Ok(mut model) = self.learned.lock() else {
            return;
        };
        for node in dag.nodes.nodes() {
            if !matches!(
                node.materialize,
                MaterializeMode::Table | MaterializeMode::TempTable
            ) {
                continue;
            }
            if let Some(node_stat) = stats.node_stats.get(&node.id)
                && let Some(plan_str) = &node_stat.plan
                && let Some(plans) = conn.parse_plan(plan_str)
            {
                model.observe(&plans);
                // What the write itself cost. Nothing else observes it: a write
                // operator's own output is a one-row count of the table, so its
                // bytes have to come from the rows the run reported writing.
                // This is also the constant the spool charge is a fraction of.
                if let Some(rows) = node_stat.rows_produced {
                    model.observe_write(&node.id, &plans, rows as f64);
                }
            }
        }
    }

    async fn load_learned(&self, store: &dyn OptStore, backend: &str) -> LearnedCostModel {
        let rows = match store
            .query(
                &format!("SELECT model FROM {LEARNED_TABLE} WHERE backend = ?"),
                &[serde_json::json!(backend)],
            )
            .await
        {
            Ok(rows) => rows,
            Err(_) => return LearnedCostModel::new(),
        };
        rows.first()
            .and_then(|r| r.get("model"))
            .and_then(|v| v.as_str())
            .and_then(|raw| serde_json::from_str(raw).ok())
            .unwrap_or_else(LearnedCostModel::new)
    }

    /// Persist what this run learned, so the next one starts from it.
    ///
    /// Necessary rather than an optimization: the model is fitted on the
    /// baseline's `After` step and spent on the next `Before` step, and under
    /// the server those two may be different processes. An in-memory model would
    /// be empty exactly when the search needs it.
    async fn publish_learned<C: Connector + Send + Sync>(&self, store: &dyn OptStore, conn: &C) {
        if !self.uses_learned_constants() {
            return;
        }
        let snapshot = match self.learned.lock() {
            Ok(model) => model.clone(),
            Err(_) => return,
        };
        if snapshot.is_empty() {
            return;
        }
        let Some(backend) = self.backend_key(conn).await else {
            return;
        };
        let Ok(encoded) = serde_json::to_string(&snapshot) else {
            return;
        };
        let write = async {
            store
                .execute(
                    &format!("DELETE FROM {LEARNED_TABLE} WHERE backend = ?"),
                    &[serde_json::json!(backend)],
                )
                .await?;
            store
                .execute(
                    &format!(
                        "INSERT INTO {LEARNED_TABLE} (backend, model, updated_at) \
                         VALUES (?, ?, now())"
                    ),
                    &[serde_json::json!(backend), serde_json::json!(encoded)],
                )
                .await
        };
        if let Err(e) = write.await {
            warn!("nodefusion: could not persist the learned cost model: {e}");
        }
    }

    /// The constants to price this round's candidates with: what earlier runs
    /// fitted, merged with what this process has learned since.
    ///
    /// An empty model is not a failure in itself --- `Cardinality` and
    /// `Operators` do not use it at all --- but under `LearnedCost` it means
    /// every candidate comes back unpriced, and `build_candidates` says so
    /// rather than reporting no candidates worth having.
    async fn snapshot_model<C: Connector + Send + Sync>(
        &self,
        store: &dyn OptStore,
        conn: &C,
    ) -> LearnedCostModel {
        let mut model = match self.learned.lock() {
            Ok(m) => m.clone(),
            Err(_) => LearnedCostModel::new(),
        };
        if self.uses_learned_constants()
            && let Some(backend) = self.backend_key(conn).await
        {
            model.merge(&self.load_learned(store, &backend).await);
        }
        model
    }
}

/// Every node `tables` read, transitively, plus `tables` themselves.
fn upstream_closure(dag: &Dag, tables: &[String]) -> HashSet<String> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut stack: Vec<String> = tables.to_vec();
    while let Some(id) = stack.pop() {
        if !seen.insert(id.clone()) {
            continue;
        }
        if let Some(node) = dag.nodes.get(id) {
            // A dependency that is not a node is one of the warehouse's own
            // source tables, which the fused query reads by name like any
            // other query does.
            for dep in &node.depends_on {
                if dag.nodes.get(dep.clone()).is_some() {
                    stack.push(dep.clone());
                }
            }
        }
    }
    seen
}

/// A topological order of `closure` that depends only on the graph, never on
/// how a `HashMap` happened to iterate.
///
/// [`Graph::topological_sort`] peels off source nodes in hash order, so two
/// independent nodes come back in either order and the fused query's CTE chain
/// would be spelled differently run to run. Taking the lexicographically
/// smallest ready node at each step instead makes the whole rewrite a function
/// of the DAG, which is what dee's content-addressed versioning needs: a
/// version is minted when the definition changes, not when the optimizer is
/// re-run over an unchanged one.
///
/// A node whose dependencies cannot all be emitted -- which a cycle would
/// cause, and [`NodeFusionPass::plan`] rejects before this is reached -- is
/// left out rather than emitted out of order.
fn stable_topological_order(dag: &Dag, closure: &HashSet<String>) -> Vec<String> {
    let mut remaining: Vec<String> = closure.iter().cloned().collect();
    remaining.sort();
    let mut emitted: HashSet<String> = HashSet::new();
    let mut order: Vec<String> = Vec::with_capacity(remaining.len());

    while !remaining.is_empty() {
        let ready: Option<usize> = remaining.iter().position(|id| {
            dag.nodes.get(id.clone()).is_some_and(|node| {
                node.depends_on
                    .iter()
                    .all(|dep| !closure.contains(dep) || emitted.contains(dep))
            })
        });
        match ready {
            Some(i) => {
                let id = remaining.remove(i);
                emitted.insert(id.clone());
                order.push(id);
            }
            None => break,
        }
    }
    order
}

/// How many of `tables` reach each node of `closure`, following dependencies
/// transitively.
///
/// "Reach" is deliberately plain: the walk does not stop at an intermediate
/// Table, so a View above one is counted for every Table downstream of it even
/// though that Table's own CTE, being materialized, already shares the View
/// once. Modelling that is what a cost-based rule would do; this is the naive
/// floor it gets measured against.
///
/// A Table reaches itself, which is why the naive rule is only ever consulted
/// for Views -- every Table would otherwise have at least one reader and the
/// rule would say nothing.
fn table_readers(
    dag: &Dag,
    tables: &[String],
    closure: &HashSet<String>,
) -> HashMap<String, usize> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for table in tables {
        let mut seen: HashSet<&String> = HashSet::new();
        let mut stack: Vec<&String> = vec![table];
        while let Some(id) = stack.pop() {
            if !seen.insert(id) {
                continue;
            }
            *counts.entry(id.clone()).or_insert(0) += 1;
            if let Some(node) = dag.nodes.get(id.clone()) {
                for dep in &node.depends_on {
                    if let Some(dep) = closure.get(dep) {
                        stack.push(dep);
                    }
                }
            }
        }
    }
    counts
}

/// Assemble the fused node's query text.
///
/// Each CTE body is the node's own query with its references to other fused
/// nodes rewritten to their CTE names. That rewrite happens on the parsed AST
/// ([`rewrite_node_refs`]), and a body that cannot be rewritten there fails the
/// pass rather than falling back to text substitution -- the same rule
/// [`make_temp`](crate::opt::common::make_temp) follows, and for the same
/// reason: a substring match inside a string literal would hand the engine a
/// query that means something else.
///
/// The frame around those bodies -- the `WITH` chain, the branches, their
/// projections -- is assembled as text. Nothing in it comes from user SQL: the
/// identifiers are generated here and the literals are the integers this pass
/// chose, so there is no substring to match wrongly. The result is parsed
/// before it is installed.
fn build_fused_sql(
    dag: &Dag,
    plan: &FusionPlan,
    dialect: DialectType,
) -> Result<FusedSql, NotFusable> {
    // Only fused nodes are rewritten; a reference to anything else is a
    // reference to a real relation and must survive untouched.
    let mapping: HashMap<String, String> = plan
        .ctes
        .iter()
        .map(|c| (c.node_id.clone(), c.cte_name.clone()))
        .collect();

    // Bodies used exactly as authored, never round-tripped through the parser.
    // Counted because the assembled query can only be parse-checked when there
    // are none: a body the parser could not read makes the whole query
    // unreadable to it too, and that says nothing about the frame.
    let mut verbatim = 0usize;
    let mut cte_sql: Vec<String> = Vec::with_capacity(plan.ctes.len());
    for cte in &plan.ctes {
        let node = dag
            .nodes
            .get(cte.node_id.clone())
            .ok_or_else(|| NotFusable(format!("'{}' vanished from the graph", cte.node_id)))?;
        // A body that names no fused node needs no rewriting, and is used
        // exactly as authored. That is not only cheaper than a parse and
        // regenerate that would change nothing -- it is what lets a DAG fuse
        // when a leaf staging view uses syntax the parser does not cover. Most
        // nodes with no fused dependencies are exactly those.
        let names_a_fused_node = node.depends_on.iter().any(|dep| mapping.contains_key(dep));
        if !names_a_fused_node {
            verbatim += 1;
        }
        let body = if names_a_fused_node {
            rewrite_node_refs(&node.query_text, &mapping, dialect).ok_or_else(|| {
                // Refusing to fall back to text substitution, which would
                // silently change what the query means. Not an error: a node
                // the parser cannot read is a fact about this DAG, the same
                // kind of fact as a TempTable in the closure, and the honest
                // response is to leave the DAG alone and name the node.
                NotFusable(format!(
                    "'{}' reads a node that would become a CTE, but its query cannot be \
                     rewritten at the AST level -- the parser does not cover it, and \
                     substituting the reference textually could change what the query means",
                    cte.node_id
                ))
            })?
        } else {
            node.query_text.clone()
        };
        let hint = if cte.materialized { " MATERIALIZED" } else { "" };
        let name = &cte.cte_name;
        cte_sql.push(format!("{name} AS{hint} ({body})"));
    }

    // One branch per Table: its own columns read from its CTE, every other
    // Table's columns filled with a NULL of the right type.
    let mut branches: Vec<String> = Vec::with_capacity(plan.tables.len());
    for table in &plan.tables {
        let own: HashMap<&str, &str> = table
            .columns
            .iter()
            .map(|(original, fused)| (fused.as_str(), original.as_str()))
            .collect();
        let mut projection: Vec<String> = vec![format!("{} AS {KIND_COLUMN}", table.kind)];
        for fused in &plan.fused_columns {
            match own.get(fused.as_str()) {
                Some(original) => projection.push(format!("\"{original}\" AS \"{fused}\"")),
                // An untyped NULL, deliberately. Every fused column belongs to
                // exactly one Table and is selected from that Table's CTE in
                // exactly one branch, so set-operation type resolution gives
                // the column that branch's type -- which is the type the
                // unfused node produced, exactly.
                //
                // Casting instead is what this used to do, and it was worse:
                // the only type available to cast to is the one read back off
                // the node's *Arrow* schema, and that round trip is lossy.
                // DuckDB's HUGEINT arrives as Decimal128(38, 0) and would have
                // gone back as DECIMAL(38,0), quietly changing the delivered
                // table's column type.
                None => projection.push(format!("NULL AS \"{fused}\"")),
            }
        }
        branches.push(format!(
            "SELECT {} FROM {}",
            projection.join(", "),
            table.cte_name
        ));
    }

    Ok(FusedSql {
        sql: format!(
            "WITH {}\n{}",
            cte_sql.join(",\n     "),
            branches.join("\nUNION ALL\n")
        ),
        verbatim_bodies: verbatim,
    })
}

/// The assembled fused query, and how many of its CTE bodies were used exactly
/// as authored rather than rewritten.
struct FusedSql {
    sql: String,
    verbatim_bodies: usize,
}

// ---------------------------------------------------------------------------
// The Optimization interface
// ---------------------------------------------------------------------------

#[async_trait]
impl<C, E> Optimization<C, E> for NodeFusionPass
where
    C: Connector + Send + 'static + Sync,
    E: Executor<C> + Send + Sync,
{
    fn name(&self) -> &'static str {
        "nodefusion"
    }

    /// `Once` as a rule-driven rewrite; `Continuous` when it is a search.
    ///
    /// The one place in dee where this is a setting rather than a constant, and
    /// it is a real difference rather than a technicality: as a rewrite the pass
    /// decides everything from the DAG in front of it and spends zero DAG runs,
    /// which is most of what makes it cheap. The adaptive variant gives that up
    /// -- it has to measure a baseline and then one candidate per run, exactly
    /// like HMP, because which CTEs are worth materializing is not a fact about
    /// the graph. A benchmark comparing the two has to account for the runs, not
    /// just the wall clock.
    fn optimization_type(&self) -> OptimizationType {
        if self.adaptive {
            OptimizationType::Continuous
        } else {
            OptimizationType::Once
        }
    }

    /// `Before` as a rewrite. The adaptive search needs both sides -- one to
    /// propose a candidate fusion, one to read what it measured.
    fn step_phase(&self) -> StepPhase {
        if self.adaptive {
            StepPhase::Both
        } else {
            self.step_phase
        }
    }

    fn set_step_phase(&mut self, phase: StepPhase) {
        self.step_phase = phase;
    }

    /// Nothing to set up as a rewrite -- it keeps no state between steps. The
    /// adaptive search keeps all of its state between steps, so it does.
    async fn register(
        &self,
        ctx: &RegisterContext<'_>,
    ) -> Result<Option<Registration>, OptimizerError> {
        if !self.adaptive {
            return Ok(None);
        }
        self.check_config()?;
        ctx.store
            .execute(
                &format!(
                    "CREATE TABLE IF NOT EXISTS {STATE_TABLE} (
                         dag_id     VARCHAR PRIMARY KEY,
                         state      VARCHAR NOT NULL,
                         updated_at TIMESTAMPTZ NOT NULL
                     )"
                ),
                &[],
            )
            .await?;
        ctx.store
            .execute(
                &format!(
                    "CREATE TABLE IF NOT EXISTS {TRIALS_TABLE} (
                         dag_id      VARCHAR NOT NULL,
                         run_id      VARCHAR,
                         iteration   INTEGER NOT NULL,
                         cte_set     VARCHAR,
                         runtime_ms  BIGINT,
                         improved    BOOLEAN,
                         recorded_at TIMESTAMPTZ NOT NULL
                     )"
                ),
                &[],
            )
            .await?;
        // Registering is idempotent -- a server restart re-registers what a DAG
        // already had -- so an existing search is left where it was rather than
        // restarted from its baseline.
        if self.load_state(ctx.store, ctx.dag_id).await?.is_none() {
            self.save_state(ctx.store, ctx.dag_id, &NodeFusionState::default())
                .await?;
        }
        ctx.store
            .execute(
                &format!(
                    "CREATE TABLE IF NOT EXISTS {LEARNED_TABLE} (
                         backend    VARCHAR,
                         model      VARCHAR NOT NULL,
                         updated_at TIMESTAMPTZ NOT NULL
                     )"
                ),
                &[],
            )
            .await?;
        Ok(Some(Registration::new([
            STATE_TABLE,
            TRIALS_TABLE,
            LEARNED_TABLE,
        ])))
    }

    async fn deregister(
        &self,
        ctx: &RegisterContext<'_>,
    ) -> Result<Option<Registration>, OptimizerError> {
        if !self.adaptive {
            return Ok(None);
        }
        // Only this DAG's rows: the tables are shared by every DAG the search is
        // registered on, and dropping them would take the others with it. They
        // go when the last registration does.
        for table in [STATE_TABLE, TRIALS_TABLE] {
            ctx.store
                .execute(
                    &format!("DELETE FROM {table} WHERE dag_id = ?"),
                    &[serde_json::json!(ctx.dag_id)],
                )
                .await?;
        }
        let remaining = ctx
            .store
            .query(&format!("SELECT count(*) AS n FROM {STATE_TABLE}"), &[])
            .await?;
        let empty = remaining
            .first()
            .and_then(|r| r.get("n"))
            .and_then(|v| v.as_i64())
            .map(|n| n == 0)
            .unwrap_or(false);
        if empty {
            // The constants have no `dag_id` to delete by -- they describe an
            // engine, not any one DAG -- so they go when the last registration
            // does, with everything else.
            for table in [STATE_TABLE, TRIALS_TABLE, LEARNED_TABLE] {
                ctx.store
                    .execute(&format!("DROP TABLE IF EXISTS {table}"), &[])
                    .await?;
            }
        }
        Ok(Some(Registration::new([
            STATE_TABLE,
            TRIALS_TABLE,
            LEARNED_TABLE,
        ])))
    }

    async fn step(
        &mut self,
        ctx: &mut StepContext<'_, C, E>,
    ) -> Result<StepOutcome, OptimizerError> {
        // Checked here as well as at the config boundaries, because a
        // `dags.optimizer_config` row written before the rule existed decodes
        // fine and would otherwise be obeyed.
        self.check_config()?;
        if !self.adaptive {
            let record = self.rewrite(ctx.dag)?;
            return Ok(StepOutcome::Rewrote {
                record: Box::new(record),
            });
        }
        match ctx.side {
            StepPhase::Before => self.step_before_adaptive(ctx).await,
            StepPhase::After => self.step_after_adaptive(ctx).await,
            StepPhase::Both => Ok(StepOutcome::Idle),
        }
    }

    fn explain(&self) -> Option<(String, String)> {
        Some(("NodeFusionPass".to_string(), self.explain_html()))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectors::duckdb::{DuckDBConfig, DuckDBConnection};
    use crate::dag::SourceNode;
    use crate::executor::{Executor, SimpleEngine};
    use crate::plan::PlanNode;
    use crate::opt::{Optimizer, report::OptimizeReport, store::MemoryStoreFactory};
    use crate::graph::Graph;
    use std::sync::Arc;

    // ------------------------------------------------------------------
    // Helpers -- a real in-memory DuckDB and the real SimpleEngine, so these
    // tests exercise the same path a run does. A fused query that only looks
    // right is not evidence of anything; one the engine executes to the same
    // rows is.
    // ------------------------------------------------------------------

    async fn in_memory_conn() -> Arc<DuckDBConnection> {
        let config = DuckDBConfig::new_from_path(":memory:".to_string());
        DuckDBConnection::new(config).await.unwrap()
    }

    fn node(id: &str, query: &str, mode: MaterializeMode, deps: &[&str]) -> TransformNode {
        TransformNode {
            id: id.to_string(),
            query_text: query.to_string(),
            materialize: mode,
            depends_on: deps.iter().map(|s| s.to_string()).collect::<HashSet<_>>(),
            schema: None,
        }
    }

    fn make_dag(nodes: Vec<TransformNode>) -> Dag {
        let mut graph = Graph::new(HashMap::new());
        for n in nodes {
            graph.add_node(n).unwrap();
        }
        Dag {
            db: "DuckDB".to_string(),
            nodes: graph,
            sources: vec![SourceNode {
                name: "orders".to_string(),
                schema: Arc::new(duckdb::arrow::datatypes::Schema::empty()),
            }],
            max_parallelism: None,
        }
    }

    async fn setup_orders(conn: &DuckDBConnection) {
        conn.execute(
            "CREATE TABLE orders AS SELECT \
                range AS order_id, \
                CASE WHEN range % 2 = 0 THEN 'US' ELSE 'EU' END AS region, \
                range * 1.5 AS amount \
             FROM range(20)"
                .to_string(),
        )
        .await
        .unwrap();
    }

    /// The DAG the happy-path tests use:
    ///
    ///   orders (a real table)
    ///     └─ stg (View) ──┬──► by_region (Table)
    ///                     └──► totals    (Table)
    ///
    /// `stg` is read by both Tables, which is exactly the duplication fusion
    /// is meant to collapse into one shared CTE.
    fn two_table_dag() -> Dag {
        make_dag(vec![
            node(
                "stg",
                "SELECT order_id, region, amount FROM orders WHERE amount > 0",
                MaterializeMode::View,
                &[],
            ),
            node(
                "by_region",
                "SELECT region, count(*) AS n, sum(amount) AS total FROM stg GROUP BY region",
                MaterializeMode::Table,
                &["stg"],
            ),
            node(
                "totals",
                "SELECT count(*) AS orders, sum(amount) AS amount_sum FROM stg",
                MaterializeMode::Table,
                &["stg"],
            ),
        ])
    }

    /// An order-independent fingerprint of a relation: its row count and the
    /// sum of the hashes of its rows, so a rewrite that reorders rows is not
    /// mistaken for one that changes them.
    fn fingerprint(conn: &DuckDBConnection, relation: &str) -> (i64, i64) {
        let c = conn.pool.get().unwrap();
        let n: i64 = c
            .query_row(&format!("SELECT count(*) FROM {relation}"), [], |r| r.get(0))
            .unwrap();
        let h: i64 = c
            .query_row(
                &format!(
                    "SELECT coalesce((sum(hash(t)::HUGEINT) % 1000000007)::BIGINT, 0) \
                     FROM {relation} AS t"
                ),
                [],
                |r| r.get(0),
            )
            .unwrap();
        (n, h)
    }

    /// The relation's columns as `name type`, in order.
    ///
    /// The types matter as much as the rows: a fused branch that filled a
    /// column with a *cast* NULL rather than an untyped one used to turn
    /// DuckDB's HUGEINT into DECIMAL(38,0), because the only type available to
    /// cast to is the one read back off the node's Arrow schema and that round
    /// trip is lossy. The rows were identical; the delivered table was not.
    fn columns(conn: &DuckDBConnection, relation: &str) -> Vec<String> {
        let c = conn.pool.get().unwrap();
        let mut stmt = c
            .prepare(&format!(
                "SELECT column_name, column_type FROM (DESCRIBE SELECT * FROM {relation})"
            ))
            .unwrap();
        let rows = stmt
            .query_map([], |r| {
                Ok(format!("{} {}", r.get::<_, String>(0)?, r.get::<_, String>(1)?))
            })
            .unwrap();
        rows.map(|r| r.unwrap()).collect()
    }

    /// Run `dag`, fingerprint each of `relations`, then drop everything it made.
    async fn run_and_fingerprint(
        conn: &Arc<DuckDBConnection>,
        dag: &Dag,
        relations: &[&str],
    ) -> Vec<(Vec<String>, (i64, i64))> {
        let engine = SimpleEngine::new(Arc::clone(conn)).unwrap();
        engine.run(dag).await.expect("the DAG should run");
        let out = relations
            .iter()
            .map(|r| (columns(conn, r), fingerprint(conn, r)))
            .collect();
        engine.cleanup(dag).await.unwrap();
        out
    }

    async fn resolved(conn: &Arc<DuckDBConnection>, dag: &mut Dag) {
        let engine = SimpleEngine::new(Arc::clone(conn)).unwrap();
        engine.resolve_schemas(dag).await.expect("schemas resolve");
    }

    // ------------------------------------------------------------------
    // End to end: the fused DAG produces the same relations
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn test_fused_dag_produces_the_same_tables() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;

        let baseline_dag = two_table_dag();
        let baseline = run_and_fingerprint(&conn, &baseline_dag, &["by_region", "totals"]).await;

        let mut dag = two_table_dag();
        resolved(&conn, &mut dag).await;
        let mut pass = NodeFusionPass::new(false, false, None);
        pass.rewrite(&mut dag).expect("the DAG should fuse");

        let fused = run_and_fingerprint(&conn, &dag, &["by_region", "totals"]).await;

        assert_eq!(
            baseline, fused,
            "fusion must not change a Table's columns or its rows"
        );
    }

    #[tokio::test]
    async fn test_the_rewritten_dag_has_the_shape_fusion_promises() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;
        let mut dag = two_table_dag();
        resolved(&conn, &mut dag).await;

        NodeFusionPass::new(false, false, None)
            .rewrite(&mut dag)
            .expect("the DAG should fuse");

        let fused = dag.nodes.get("dee_fused".to_string()).expect("fused node");
        assert_eq!(fused.materialize, MaterializeMode::TempTable);
        assert!(
            fused.depends_on.is_empty(),
            "everything upstream of a Table is inside the fused node, so it depends on nothing"
        );
        assert!(fused.query_text.contains("UNION ALL"));

        for table in ["by_region", "totals"] {
            let node = dag.nodes.get(table.to_string()).unwrap();
            assert_eq!(node.materialize, MaterializeMode::Table);
            assert_eq!(
                node.depends_on,
                HashSet::from(["dee_fused".to_string()]),
                "a fused Table reads the fused node and nothing else"
            );
            assert!(node.query_text.contains("WHERE kind = "));
        }

        // The View stays in the graph, and stays a View.
        let stg = dag.nodes.get("stg".to_string()).expect("the View is kept");
        assert_eq!(stg.materialize, MaterializeMode::View);
    }

    #[tokio::test]
    async fn test_fusion_preserves_column_types() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;

        // `sum(order_id)` is a HUGEINT, which reaches an Arrow schema as
        // Decimal128(38, 0) and cannot be told apart from a real
        // DECIMAL(38,0) on the way back. The branches fill a foreign column
        // with an untyped NULL precisely so that round trip never happens.
        let dag_of = || {
            make_dag(vec![
                node(
                    "stg",
                    "SELECT order_id, region, amount FROM orders",
                    MaterializeMode::View,
                    &[],
                ),
                node(
                    "wide",
                    "SELECT sum(order_id) AS id_sum, region, \
                     count(*) > 0 AS any_rows, max(amount) AS biggest \
                     FROM stg GROUP BY region",
                    MaterializeMode::Table,
                    &["stg"],
                ),
                node(
                    "narrow",
                    "SELECT count(*) AS n FROM stg",
                    MaterializeMode::Table,
                    &["stg"],
                ),
            ])
        };

        let baseline = run_and_fingerprint(&conn, &dag_of(), &["wide", "narrow"]).await;

        let mut dag = dag_of();
        resolved(&conn, &mut dag).await;
        NodeFusionPass::new(false, false, None).rewrite(&mut dag).unwrap();
        let fused = run_and_fingerprint(&conn, &dag, &["wide", "narrow"]).await;

        assert_eq!(
            baseline, fused,
            "a fused Table must keep its column types, not only its rows"
        );
        assert!(
            baseline[0].0.iter().any(|c| c.contains("HUGEINT")),
            "the fixture must actually exercise HUGEINT -- got {:?}",
            baseline[0].0
        );
    }

    // ------------------------------------------------------------------
    // A Table above a Table
    // ------------------------------------------------------------------

    /// DAG layout:
    ///
    ///   orders ─► stg (View) ─► base (Table) ─► enriched (View) ─► rollup (Table)
    ///
    /// `base` feeds `rollup` through a View, so it is inlined as a CTE *and*
    /// still delivered as its own relation.
    fn layered_dag() -> Dag {
        make_dag(vec![
            node(
                "stg",
                "SELECT order_id, region, amount FROM orders",
                MaterializeMode::View,
                &[],
            ),
            node(
                "base",
                "SELECT region, sum(amount) AS total FROM stg GROUP BY region",
                MaterializeMode::Table,
                &["stg"],
            ),
            node(
                "enriched",
                "SELECT region, total, total * 2 AS doubled FROM base",
                MaterializeMode::View,
                &["base"],
            ),
            node(
                "rollup",
                "SELECT sum(doubled) AS all_doubled FROM enriched",
                MaterializeMode::Table,
                &["enriched"],
            ),
        ])
    }

    #[tokio::test]
    async fn test_a_table_feeding_a_table_is_inlined_and_still_delivered() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;

        let baseline_dag = layered_dag();
        let baseline = run_and_fingerprint(&conn, &baseline_dag, &["base", "rollup"]).await;

        let mut dag = layered_dag();
        resolved(&conn, &mut dag).await;
        let mut pass = NodeFusionPass::new(false, false, None);
        pass.rewrite(&mut dag).expect("the DAG should fuse");

        let fused_sql = dag
            .nodes
            .get("dee_fused".to_string())
            .unwrap()
            .query_text
            .clone();
        assert!(
            fused_sql.contains("n_base AS MATERIALIZED ("),
            "an inlined Table's CTE is materialized by default -- got:\n{fused_sql}"
        );
        assert!(
            fused_sql.contains("n_stg AS ("),
            "an inlined View's CTE is plain by default -- got:\n{fused_sql}"
        );
        assert!(
            dag.nodes
                .get("base".to_string())
                .unwrap()
                .query_text
                .contains("WHERE kind = "),
            "the intermediate Table still gets a kind of its own"
        );

        let fused = run_and_fingerprint(&conn, &dag, &["base", "rollup"]).await;
        assert_eq!(baseline, fused);
    }

    #[tokio::test]
    async fn test_the_rewrite_is_a_function_of_the_dag_alone() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;

        // The same logical DAG, built by inserting its nodes in two different
        // orders. `Graph` is a `HashMap`, so that is enough to change how it
        // iterates -- and the fused query must not notice. dee's DAGs are
        // content-addressed: a rewrite that is not byte-stable mints a new
        // version every time it is re-run over an unchanged definition.
        let build = |reverse: bool| {
            let mut nodes = vec![
                node(
                    "stg",
                    "SELECT order_id, region, amount FROM orders",
                    MaterializeMode::View,
                    &[],
                ),
                node(
                    "by_region",
                    "SELECT region, count(*) AS n FROM stg GROUP BY region",
                    MaterializeMode::Table,
                    &["stg"],
                ),
                node(
                    "totals",
                    "SELECT count(*) AS n FROM stg",
                    MaterializeMode::Table,
                    &["stg"],
                ),
                node(
                    "biggest",
                    "SELECT max(amount) AS m FROM stg",
                    MaterializeMode::Table,
                    &["stg"],
                ),
            ];
            if reverse {
                nodes.reverse();
            }
            // Inserted in the given order and only checked afterwards:
            // `add_node` refuses a node whose dependency is not in the graph
            // yet, which is the very ordering this test needs to vary.
            let mut graph = Graph::new(HashMap::new());
            for n in nodes {
                graph.add_node_unchecked(n);
            }
            graph.check().unwrap();
            Dag {
                db: "DuckDB".to_string(),
                nodes: graph,
                sources: vec![SourceNode {
                    name: "orders".to_string(),
                    schema: Arc::new(duckdb::arrow::datatypes::Schema::empty()),
                }],
                max_parallelism: None,
            }
        };

        async fn fuse(conn: &Arc<DuckDBConnection>, mut dag: Dag) -> Vec<(String, String)> {
            resolved(conn, &mut dag).await;
            NodeFusionPass::new(false, false, None)
                .rewrite(&mut dag)
                .unwrap();
            let mut texts: Vec<(String, String)> = dag
                .nodes
                .nodes()
                .map(|n| (n.id.clone(), n.query_text.clone()))
                .collect();
            texts.sort();
            texts
        }

        assert_eq!(
            fuse(&conn, build(false)).await,
            fuse(&conn, build(true)).await,
            "the fused query and every rewritten Table must be byte-identical"
        );
    }

    #[tokio::test]
    async fn test_kinds_are_assigned_in_sorted_node_id_order() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;
        let mut dag = layered_dag();
        resolved(&conn, &mut dag).await;

        let plan = NodeFusionPass::new(false, false, None).plan(&dag).unwrap();
        let by_kind: Vec<&str> = plan.tables.iter().map(|t| t.node_id.as_str()).collect();
        assert_eq!(
            by_kind,
            vec!["base", "rollup"],
            "a kind is a label, not an ordering, so it follows the one total order \
             the DAG has: its node IDs"
        );
        assert_eq!(plan.tables[0].kind, 0);
        assert_eq!(plan.tables[1].kind, 1);
    }

    #[tokio::test]
    async fn test_ctes_are_emitted_in_a_topological_order() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;
        let mut dag = layered_dag();
        resolved(&conn, &mut dag).await;

        let plan = NodeFusionPass::new(false, false, None).plan(&dag).unwrap();
        let order: Vec<&str> = plan.ctes.iter().map(|c| c.node_id.as_str()).collect();
        assert_eq!(order, vec!["stg", "base", "enriched", "rollup"]);
    }

    // ------------------------------------------------------------------
    // The materialization knobs
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn test_the_global_default_materializes_views_and_not_only_tables() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;
        let mut dag = layered_dag();
        resolved(&conn, &mut dag).await;

        let plan = NodeFusionPass::new(true, false, None).plan(&dag).unwrap();
        let materialized: HashSet<&str> = plan
            .ctes
            .iter()
            .filter(|c| c.materialized)
            .map(|c| c.node_id.as_str())
            .collect();
        assert_eq!(
            materialized,
            HashSet::from(["stg", "base", "enriched", "rollup"]),
            "the global default turns on everything the intermediate-Table rule did not"
        );
    }

    #[tokio::test]
    async fn test_only_a_table_that_feeds_another_is_materialized_by_default() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;

        // `base` feeds `rollup`, so its CTE is read by its own UNION ALL
        // branch and by `enriched` -- two readers, and a plain CTE may be
        // unfolded into each. `rollup` feeds nothing and is read once.
        let mut layered = layered_dag();
        resolved(&conn, &mut layered).await;
        let plan = NodeFusionPass::new(false, false, None).plan(&layered).unwrap();
        let materialized: Vec<&str> = plan
            .ctes
            .iter()
            .filter(|c| c.materialized)
            .map(|c| c.node_id.as_str())
            .collect();
        assert_eq!(materialized, vec!["base"]);

        // And where no Table feeds another, nothing is materialized at all.
        let mut flat = two_table_dag();
        resolved(&conn, &mut flat).await;
        let plan = NodeFusionPass::new(false, false, None).plan(&flat).unwrap();
        assert!(
            plan.ctes.iter().all(|c| !c.materialized),
            "a Table read only by its own branch gains nothing from MATERIALIZED"
        );
    }

    #[tokio::test]
    async fn test_an_override_is_the_exact_set_and_can_demote_a_table() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;
        let mut dag = layered_dag();
        resolved(&conn, &mut dag).await;

        // `base` is a Table and would be materialized by default; naming only
        // `stg` takes it back off.
        let plan = NodeFusionPass::new(true, false, Some(vec!["stg".to_string()]))
            .plan(&dag)
            .unwrap();
        let materialized: Vec<&str> = plan
            .ctes
            .iter()
            .filter(|c| c.materialized)
            .map(|c| c.node_id.as_str())
            .collect();
        assert_eq!(materialized, vec!["stg"]);
    }

    /// DAG layout, built so the naive rule has both answers to give:
    ///
    ///   orders ─► shared (View) ─┬─► mid (View) ─► tbl_a (Table)
    ///                            └────────────────► tbl_b (Table)
    ///          └─► lonely (View) ──────────────────► tbl_b (Table)
    ///
    /// `shared` is reached by both Tables; `mid` by only `tbl_a` and `lonely`
    /// by only `tbl_b`.
    ///
    fn shared_view_dag() -> Dag {
        make_dag(vec![
            node(
                "shared",
                "SELECT order_id, region, amount FROM orders",
                MaterializeMode::View,
                &[],
            ),
            node(
                "mid",
                "SELECT region, amount * 2 AS doubled FROM shared",
                MaterializeMode::View,
                &["shared"],
            ),
            node(
                "lonely",
                "SELECT order_id, amount FROM orders WHERE amount > 1",
                MaterializeMode::View,
                &[],
            ),
            node(
                "tbl_a",
                "SELECT region, sum(doubled) AS d FROM mid GROUP BY region",
                MaterializeMode::Table,
                &["mid"],
            ),
            node(
                "tbl_b",
                "SELECT count(*) AS n, (SELECT count(*) FROM lonely) AS m FROM shared",
                MaterializeMode::Table,
                &["shared", "lonely"],
            ),
        ])
    }

    #[tokio::test]
    async fn test_the_naive_rule_materializes_only_views_more_than_one_table_reads() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;
        let mut dag = shared_view_dag();
        resolved(&conn, &mut dag).await;

        let plan = NodeFusionPass::new(false, true, None).plan(&dag).unwrap();
        let materialized: HashSet<&str> = plan
            .ctes
            .iter()
            .filter(|c| c.materialized)
            .map(|c| c.node_id.as_str())
            .collect();
        assert_eq!(
            materialized,
            HashSet::from(["shared"]),
            "only the View both Tables reach; `mid` and `lonely` have one reader each"
        );

        // Off, it materializes nothing here: no Table feeds another.
        let plan = NodeFusionPass::new(false, false, None).plan(&dag).unwrap();
        assert!(plan.ctes.iter().all(|c| !c.materialized));
    }

    #[tokio::test]
    async fn test_the_naive_rule_counts_readers_through_intermediate_views() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;
        // In `layered_dag`, `stg` is read by `base` and -- through `base` and
        // `enriched` -- by `rollup`. Reachability is transitive and does not
        // stop at the intermediate Table, which is the naive part.
        let mut dag = layered_dag();
        resolved(&conn, &mut dag).await;

        let plan = NodeFusionPass::new(false, true, None).plan(&dag).unwrap();
        let materialized: HashSet<&str> = plan
            .ctes
            .iter()
            .filter(|c| c.materialized)
            .map(|c| c.node_id.as_str())
            .collect();
        assert_eq!(
            materialized,
            HashSet::from(["stg", "base"]),
            "`stg` by the naive rule, `base` because it feeds another Table; \
             `enriched` is reached by `rollup` alone"
        );
    }

    #[tokio::test]
    async fn test_the_naive_rule_does_not_change_what_the_dag_produces() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;

        let baseline = run_and_fingerprint(&conn, &shared_view_dag(), &["tbl_a", "tbl_b"]).await;

        let mut dag = shared_view_dag();
        resolved(&conn, &mut dag).await;
        NodeFusionPass::new(false, true, None).rewrite(&mut dag).unwrap();
        assert!(
            dag.nodes
                .get("dee_fused".to_string())
                .unwrap()
                .query_text
                .contains("n_shared AS MATERIALIZED ("),
            "the shared View should carry the hint"
        );

        let fused = run_and_fingerprint(&conn, &dag, &["tbl_a", "tbl_b"]).await;
        assert_eq!(baseline, fused, "a materialization hint changes plans, not rows");
    }

    #[test]
    fn test_the_global_switch_subsumes_the_naive_one() {
        let all = NodeFusionPass::new(true, false, None);
        let naive = NodeFusionPass::new(false, true, None);
        // A View no second Table reaches: on under the global switch, off
        // under the naive rule.
        assert!(all.wants_materialized("v", false, false));
        assert!(!naive.wants_materialized("v", false, false));
        // And an override still names the exact set, ignoring both.
        let overridden = NodeFusionPass::new(true, true, Some(vec!["other".into()]));
        assert!(!overridden.wants_materialized("v", true, true));
    }

    #[test]
    fn test_an_override_matches_a_bare_name_against_a_qualified_id() {
        let pass = NodeFusionPass::new(false, false, Some(vec!["orders".to_string()]));
        assert!(pass.wants_materialized("\"wh\".\"main\".\"orders\"", false, false));
        assert!(!pass.wants_materialized("\"wh\".\"main\".\"other\"", true, true));
    }

    // ------------------------------------------------------------------
    // The DAGs that are left alone
    // ------------------------------------------------------------------

    async fn assert_not_fused(dag: &mut Dag, expect: &str) {
        let before: Vec<(String, String)> = dag
            .nodes
            .nodes()
            .map(|n| (n.id.clone(), n.query_text.clone()))
            .collect();
        let n_before = dag.nodes.num_nodes();

        let mut pass = NodeFusionPass::new(false, false, None);
        let record = pass.rewrite(dag).expect("an unfusable DAG is not an error");

        let PassDetail::NodeFusion(detail) = &record.detail else {
            panic!("expected a NodeFusion detail");
        };
        assert!(
            detail.outcome.contains(expect),
            "expected an outcome mentioning '{expect}', got '{}'",
            detail.outcome
        );
        assert_eq!(detail.tables_fused, 0);
        assert_eq!(record.changes_applied, 0);
        assert_eq!(dag.nodes.num_nodes(), n_before, "no node was added");
        let after: Vec<(String, String)> = dag
            .nodes
            .nodes()
            .map(|n| (n.id.clone(), n.query_text.clone()))
            .collect();
        let mut before = before;
        let mut after = after;
        before.sort();
        after.sort();
        assert_eq!(before, after, "an unfusable DAG must be left exactly as it was");
    }

    #[tokio::test]
    async fn test_a_single_table_is_not_worth_fusing() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;
        let mut dag = make_dag(vec![
            node("stg", "SELECT * FROM orders", MaterializeMode::View, &[]),
            node(
                "only",
                "SELECT count(*) AS n FROM stg",
                MaterializeMode::Table,
                &["stg"],
            ),
        ]);
        resolved(&conn, &mut dag).await;
        assert_not_fused(&mut dag, "at least two").await;
    }

    #[tokio::test]
    async fn test_a_temp_table_upstream_of_a_table_blocks_fusion() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;
        let mut dag = make_dag(vec![
            node("pad", "SELECT * FROM orders", MaterializeMode::TempTable, &[]),
            node(
                "count_pad",
                "SELECT count(*) AS n FROM pad",
                MaterializeMode::Table,
                &["pad"],
            ),
            node(
                "sum_pad",
                "SELECT sum(amount) AS s FROM pad",
                MaterializeMode::Table,
                &["pad"],
            ),
        ]);
        resolved(&conn, &mut dag).await;
        assert_not_fused(&mut dag, "TempTable").await;
    }

    #[tokio::test]
    async fn test_a_node_the_parser_cannot_read_blocks_fusion_only_if_it_reads_a_cte() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;

        // `extract('year' from d)` is DuckDB-valid and polyglot-sql cannot
        // parse it. Here it sits in a leaf staging view that names no fused
        // node, so nothing needs rewriting and it is carried through exactly
        // as authored.
        let mut dag = make_dag(vec![
            node(
                "stg",
                "SELECT order_id, region, amount, \
                 extract('year' from DATE '2024-03-01') AS yr FROM orders",
                MaterializeMode::View,
                &[],
            ),
            node(
                "by_region",
                "SELECT region, count(*) AS n FROM stg GROUP BY region",
                MaterializeMode::Table,
                &["stg"],
            ),
            node(
                "totals",
                "SELECT count(*) AS n, max(yr) AS yr FROM stg",
                MaterializeMode::Table,
                &["stg"],
            ),
        ]);
        resolved(&conn, &mut dag).await;
        NodeFusionPass::new(false, false, None)
            .rewrite(&mut dag)
            .expect("an unparseable leaf is carried through verbatim");
        let fused = dag.nodes.get("dee_fused".to_string()).unwrap();
        assert!(
            fused.query_text.contains("extract('year'"),
            "the body should be used as authored, not regenerated -- got:\n{}",
            fused.query_text
        );
        let engine = SimpleEngine::new(Arc::clone(&conn)).unwrap();
        engine.run(&dag).await.expect("and the fused DAG still runs");
        engine.cleanup(&dag).await.unwrap();

        // The same syntax in a node that *does* read a CTE cannot be rewritten,
        // and blocks fusion rather than being substituted textually.
        let mut dag = make_dag(vec![
            node(
                "stg",
                "SELECT order_id, region, amount FROM orders",
                MaterializeMode::View,
                &[],
            ),
            node(
                "by_region",
                "SELECT region, count(*) AS n, \
                 extract('year' from DATE '2024-03-01') AS yr FROM stg GROUP BY region",
                MaterializeMode::Table,
                &["stg"],
            ),
            node(
                "totals",
                "SELECT count(*) AS n FROM stg",
                MaterializeMode::Table,
                &["stg"],
            ),
        ]);
        resolved(&conn, &mut dag).await;
        assert_not_fused(&mut dag, "cannot be rewritten at the AST level").await;
    }

    #[tokio::test]
    async fn test_an_unresolved_schema_blocks_fusion() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;
        // Deliberately not resolved: the fused relation's columns come from
        // these schemas, and guessing them would be worse than not fusing.
        let mut dag = two_table_dag();
        assert_not_fused(&mut dag, "no resolved schema").await;
    }

    #[tokio::test]
    async fn test_fusing_an_already_fused_dag_is_refused() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;
        let mut dag = two_table_dag();
        resolved(&conn, &mut dag).await;
        NodeFusionPass::new(false, false, None).rewrite(&mut dag).unwrap();

        // The fused node is a TempTable and every Table now reads it, so the
        // second pass sees a TempTable upstream of a Table and declines. It
        // must decline rather than fuse again: a second fusion would wrap the
        // projections in another UNION ALL and gain nothing.
        assert_not_fused(&mut dag, "TempTable").await;
    }

    // ------------------------------------------------------------------
    // Naming
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn test_the_fused_node_lands_in_the_same_schema_as_its_tables() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;
        conn.execute("CREATE SCHEMA wh".to_string()).await.unwrap();

        let mut dag = make_dag(vec![
            node(
                "\"wh\".\"stg\"",
                "SELECT order_id, region, amount FROM orders",
                MaterializeMode::View,
                &[],
            ),
            node(
                "\"wh\".\"count_stg\"",
                "SELECT count(*) AS n FROM \"wh\".\"stg\"",
                MaterializeMode::Table,
                &["\"wh\".\"stg\""],
            ),
            node(
                "\"wh\".\"sum_stg\"",
                "SELECT sum(amount) AS s FROM \"wh\".\"stg\"",
                MaterializeMode::Table,
                &["\"wh\".\"stg\""],
            ),
        ]);
        resolved(&conn, &mut dag).await;

        NodeFusionPass::new(false, false, None).rewrite(&mut dag).unwrap();
        assert!(
            dag.nodes.get("\"wh\".\"dee_fused\"".to_string()).is_some(),
            "the fused node inherits the schema prefix of the Tables reading it"
        );

        let engine = SimpleEngine::new(Arc::clone(&conn)).unwrap();
        engine.run(&dag).await.expect("the fused DAG runs");
        assert_eq!(fingerprint(&conn, "\"wh\".\"count_stg\"").0, 1);
        engine.cleanup(&dag).await.unwrap();
    }

    // ------------------------------------------------------------------
    // The adaptive variant
    // ------------------------------------------------------------------

    /// NodeFusion's adaptive search and nothing else.
    ///
    /// `with_all_disabled` first, because HMP and OMP are on by default and
    /// they are not neutral here: both materialize Views into `TempTable`
    /// landing pads, and a TempTable upstream of a Table makes the DAG
    /// unfusable by design -- the pass refuses to fold away a barrier a
    /// materialization search placed deliberately. Leaving them on does not
    /// make the test harder, it makes it measure a DAG that never gets fused.
    fn adaptive_config() -> OptimizerConfig {
        OptimizerConfig::default()
            .with_all_disabled()
            .with_nodefusion_pass()
            .with_nodefusion_adaptive_materialize_ctes(true)
    }

    #[test]
    fn adaptive_and_naive_together_is_a_config_error() {
        // The naive reader-count rule is the floor the search exists to beat.
        // Applying it to the search's own baseline would make every candidate a
        // comparison against the wrong control -- so this is refused rather than
        // resolved in favour of whichever the pass happens to read first.
        let config = adaptive_config().with_nodefusion_naive_materialize_ctes(true);
        let problem = config.validate().expect_err("the pair must be refused");
        assert!(
            problem.contains("nodefusion_naive_materialize_ctes"),
            "the message has to name both settings; got {problem}"
        );

        // And again at the pass, which is the backstop for a config row stored
        // before the rule existed: that row still decodes, and obeying it
        // silently is the failure this guards.
        let pass = NodeFusionPass::from_config(&config);
        let err = pass.check_config().expect_err("the pass must refuse too");
        assert!(matches!(err, OptimizerError::Config(_)), "got {err:?}");
    }

    #[test]
    fn adaptive_refuses_the_settings_that_would_leave_it_nothing_to_decide() {
        // An override pins the exact set the search exists to find, and the
        // global switch turns every View on regardless of what it decides.
        // Either way the runs would be spent re-deriving a fixed answer.
        for config in [
            adaptive_config()
                .with_nodefusion_materialize_ctes_override(Some(vec!["stg".into()])),
            adaptive_config().with_nodefusion_materialize_ctes(true),
        ] {
            assert!(
                config.validate().is_err(),
                "a search with nothing to decide must be refused"
            );
        }
        // The plain adaptive config is fine, or the assertions above prove
        // nothing.
        adaptive_config().validate().expect("adaptive alone is valid");
    }

    #[test]
    fn adaptive_without_the_pass_enabled_is_refused() {
        // Nothing would read the setting. Silently ignoring it looks exactly
        // like a search that ran and found nothing.
        let config = OptimizerConfig::default().with_nodefusion_adaptive_materialize_ctes(true);
        assert!(config.validate().is_err());
    }

    #[test]
    fn the_naive_rule_alone_is_still_valid() {
        // The floor has to remain reachable, or there is nothing to measure
        // the search against.
        OptimizerConfig::default()
            .with_nodefusion_pass()
            .with_nodefusion_naive_materialize_ctes(true)
            .validate()
            .expect("naive on its own is the control cell");
    }

    #[test]
    fn adaptive_is_a_search_and_a_rule_driven_fusion_is_not() {
        // The one place in dee where `optimization_type` is a setting. It is
        // also the honest accounting: adaptive gives up NodeFusion's zero-run
        // property, and anything scheduling on this has to see that.
        fn kind(config: &OptimizerConfig) -> OptimizationType {
            let pass = NodeFusionPass::from_config(config);
            Optimization::<DuckDBConnection, SimpleEngine<DuckDBConnection>>::optimization_type(
                &pass,
            )
        }
        assert_eq!(
            kind(&OptimizerConfig::default().with_nodefusion_pass()),
            OptimizationType::Once
        );
        assert_eq!(kind(&adaptive_config()), OptimizationType::Continuous);
    }

    #[tokio::test]
    async fn only_a_cte_with_more_than_one_reader_is_the_searchs_to_decide() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;
        let mut dag = two_table_dag();
        resolved(&conn, &mut dag).await;

        let plan = NodeFusionPass::new(false, false, None).plan(&dag).unwrap();
        let decidable = plan.decidable();

        // `stg` is read by both Tables, so materializing it is a real question.
        assert_eq!(
            decidable,
            vec!["stg".to_string()],
            "only the shared View is decidable; got {decidable:?}"
        );
        // The two leaf Tables are read exactly once each -- by their own
        // `UNION ALL` branch -- so they are computed once whichever way the hint
        // goes. A search that trialled them would be spending DAG runs on a
        // coin flip.
        for leaf in ["by_region", "totals"] {
            let cte = plan.ctes.iter().find(|c| c.node_id == leaf).unwrap();
            assert_eq!(cte.readers, 1);
            assert!(!cte.decidable, "{leaf} is read once and has nothing to decide");
        }
    }

    #[tokio::test]
    async fn an_intermediate_tables_cte_is_the_searchs_to_demote() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;
        let mut dag = layered_dag();
        resolved(&conn, &mut dag).await;

        let plan = NodeFusionPass::new(false, false, None).plan(&dag).unwrap();
        let base = plan.ctes.iter().find(|c| c.node_id == "base").unwrap();

        // Materialized by default -- it was authored as a barrier -- but
        // decidable, which is the widening that pricing Tables as well as Views
        // buys. Whether a barrier that made sense between two *tables* is the
        // right barrier inside a single query is exactly what the search is for.
        assert!(base.materialized, "the default still materializes it");
        assert!(base.decidable, "and the search still gets to disagree");
        assert!(base.readers > 1);

        // And the search can actually express the demotion: the empty set is a
        // configuration, not an absence.
        let mut plain = NodeFusionPass::new(false, false, None);
        let empty: HashSet<String> = HashSet::new();
        let mut demoted = dag.clone();
        plain.rewrite_with(&mut demoted, &empty).unwrap();
        let sql = demoted
            .nodes
            .get("dee_fused".to_string())
            .unwrap()
            .query_text
            .clone();
        assert!(
            !sql.contains("MATERIALIZED"),
            "the empty set means every CTE plain; got {sql}"
        );
    }

    #[tokio::test]
    async fn a_demoted_intermediate_table_still_delivers_the_same_rows() {
        // The search may move the hint around; it may not change what the DAG
        // produces. Same standard the naive rule is held to.
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;

        let baseline = run_and_fingerprint(&conn, &layered_dag(), &["base", "rollup"]).await;

        let mut dag = layered_dag();
        resolved(&conn, &mut dag).await;
        let empty: HashSet<String> = HashSet::new();
        NodeFusionPass::new(false, false, None)
            .rewrite_with(&mut dag, &empty)
            .unwrap();
        let fused = run_and_fingerprint(&conn, &dag, &["base", "rollup"]).await;

        assert_eq!(
            baseline, fused,
            "demoting the intermediate Table's CTE changes the plan, never the rows"
        );
    }

    // ------------------------------------------------------------------
    // The two objectives, over the WITH chain
    // ------------------------------------------------------------------

    /// A three-CTE chain: `a` feeds `b` feeds `c`.
    fn chain_of_three() -> Vec<CtePlan> {
        let cte = |id: &str, reads: &[&str]| CtePlan {
            node_id: id.to_string(),
            cte_name: format!("n_{id}"),
            materialized: false,
            is_table: false,
            reads: reads.iter().map(|s| s.to_string()).collect(),
            readers: 2,
            decidable: true,
        };
        vec![cte("a", &[]), cte("b", &["a"]), cte("c", &["b"])]
    }

    fn profiles_of(entries: &[(&str, f64, Option<f64>)]) -> HashMap<String, CteProfile> {
        entries
            .iter()
            .map(|(id, compute, spool)| {
                (
                    id.to_string(),
                    CteProfile {
                        compute: *compute,
                        spool: *spool,
                    },
                )
            })
            .collect()
    }

    #[test]
    fn a_plain_cte_is_pipelined_and_a_materialized_one_is_a_barrier() {
        let chain = chain_of_three();
        let profiles = profiles_of(&[
            ("a", 1.0, Some(0.5)),
            ("b", 2.0, Some(0.5)),
            ("c", 4.0, Some(0.5)),
        ]);

        // Nothing materialized: nothing is a barrier, so the path through the
        // chain is empty -- every CTE's work is inside whichever reader unfolded
        // it, and no reader waits.
        let none = barrier_chain(&chain, &profiles, &HashSet::new()).unwrap();
        assert_eq!(none, 0.0);

        // One barrier in the middle: its compute plus its spool, once.
        let mid: HashSet<String> = ["b".to_string()].into_iter().collect();
        assert_eq!(barrier_chain(&chain, &profiles, &mid).unwrap(), 2.5);

        // Two barriers on one chain serialize: `a` must finish before `b` can.
        let both: HashSet<String> = ["a".to_string(), "b".to_string()].into_iter().collect();
        assert_eq!(
            barrier_chain(&chain, &profiles, &both).unwrap(),
            4.0,
            "a barrier above a barrier waits for it -- that is the cost a sum \
             of removed work cannot see"
        );
    }

    #[test]
    fn members_on_one_chain_spool_in_sequence() {
        let chain = chain_of_three();
        let one: HashSet<String> = ["b".to_string()].into_iter().collect();
        assert_eq!(makespan_stages(&chain, &one), 1);
        let all: HashSet<String> =
            ["a".to_string(), "b".to_string(), "c".to_string()].into_iter().collect();
        assert_eq!(
            makespan_stages(&chain, &all),
            3,
            "three links of one chain is three stages, not one set of three"
        );
    }

    #[test]
    fn a_set_whose_spool_cannot_be_priced_is_unranked_not_free() {
        // The whole point of `Option` here. Charging an unpriceable spool as
        // zero would make exactly the largest CTEs look free to materialize,
        // which is the bias the write term exists to correct -- and it would do
        // it silently, because a zero is still a number and the ranking would
        // still come out sorted.
        let chain = chain_of_three();
        let profiles = profiles_of(&[("a", 1.0, None), ("b", 2.0, Some(0.5)), ("c", 4.0, Some(0.5))]);

        let unpriceable: HashSet<String> = ["a".to_string()].into_iter().collect();
        assert!(
            barrier_chain(&chain, &profiles, &unpriceable).is_none(),
            "a member with no spool price leaves the whole set unranked"
        );
        // A set that avoids it is still rankable, so this is not just "nothing
        // can ever be priced".
        let fine: HashSet<String> = ["b".to_string()].into_iter().collect();
        assert!(barrier_chain(&chain, &profiles, &fine).is_some());
    }

    #[test]
    fn the_two_objectives_can_order_the_same_sets_differently() {
        // The reason the objective is a setting rather than a constant. These
        // are the numbers the NodeFusion coster produces: `cost` is work removed
        // (higher better), `makespan_s` is the path through the WITH chain
        // (lower better), and a set that removes the most work can easily be the
        // one that serializes the chain hardest.
        let costed = vec![
            CostedCombo {
                combo: vec!["a".into(), "b".into()],
                cost: 9.0,
                makespan_s: Some(7.0),
                stages: 2,
            },
            CostedCombo {
                combo: vec!["b".into()],
                cost: 4.0,
                makespan_s: Some(2.0),
                stages: 1,
            },
        ];

        let by_work = crate::opt::combo::order_by(costed.clone(), Objective::QueryTime);
        assert_eq!(by_work[0].combo, vec!["a".to_string(), "b".to_string()]);

        let by_path = crate::opt::combo::order_by(costed, Objective::Makespan);
        assert_eq!(
            by_path[0].combo,
            vec!["b".to_string()],
            "ranking by one and accepting on the other is how a search finds \
             improvements and promotes none of them"
        );
    }

    #[test]
    fn the_trial_budget_is_capped_in_the_objectives_own_measure() {
        // Capping wall clock while optimizing total work would cancel a
        // candidate for being slow when slow is not the complaint.
        let makespan = NodeFusionPass::from_config(
            &adaptive_config().with_nodefusion_objective(Objective::Makespan),
        );
        assert_eq!(makespan.budget_metric(), BudgetMetric::WallClock);
        let query_time = NodeFusionPass::from_config(
            &adaptive_config().with_nodefusion_objective(Objective::QueryTime),
        );
        assert_eq!(query_time.budget_metric(), BudgetMetric::NodeTime);
    }

    // ------------------------------------------------------------------
    // The spool constant
    // ------------------------------------------------------------------

    #[test]
    fn the_spool_default_is_per_backend_and_an_override_replaces_it() {
        let pass = NodeFusionPass::from_config(&adaptive_config());
        let empty = LearnedCostModel::new();

        // DuckDB buffers the CTE; Postgres materializes into a tuplestore that
        // can spill. Both are modelled guesses, and both are meant to be
        // reachable from a config rather than only by recompiling.
        assert!(
            pass.spool_factor_for(&empty, "WRITE", DialectType::DuckDB)
                < pass.spool_factor_for(&empty, "WRITE", DialectType::PostgreSQL),
            "a spool that spills should cost more than one that does not"
        );

        let tuned = NodeFusionPass::from_config(
            &adaptive_config().with_nodefusion_spool_cost_factor(Some(0.9)),
        );
        assert_eq!(tuned.spool_factor_for(&empty, "WRITE", DialectType::DuckDB), 0.9);
    }

    #[test]
    fn an_explicit_spool_rate_falls_back_rather_than_being_ignored() {
        // Expressed against the write constant, because that is the per-byte
        // rate the model knows how to apply to a plan region. With no write
        // constant to express it against there is nothing to divide by, and the
        // configured factor stands in -- loudly, because silently dropping a
        // number somebody set is the one outcome with no way to notice.
        let pass = NodeFusionPass::from_config(
            &adaptive_config()
                .with_nodefusion_spool_seconds_per_byte(Some(1e-9))
                .with_nodefusion_spool_cost_factor(Some(0.25)),
        );
        let empty = LearnedCostModel::new();
        assert_eq!(
            pass.spool_factor_for(&empty, "WRITE", DialectType::DuckDB),
            0.25,
            "no write constant means the explicit rate cannot be applied"
        );
    }

    #[test]
    fn a_spool_factor_of_zero_is_free_and_an_unpriceable_write_is_not() {
        // Two different answers that must not collapse into each other. Zero is
        // "the caller says this engine's spool is free"; `None` is "nobody
        // knows", and only the first is rankable.
        let model = LearnedCostModel::new();
        let roots: Vec<PlanNode> = Vec::new();
        assert_eq!(model.spool_cost("WRITE", &roots, 100.0, 0.0), Some(0.0));
        assert_eq!(model.spool_cost("WRITE", &roots, 100.0, 0.5), None);
    }

    // ------------------------------------------------------------------
    // State
    // ------------------------------------------------------------------

    #[test]
    fn a_legacy_nodefusion_state_row_decodes_and_is_not_mistaken_for_exhausted() {
        // A state row written by a build that predates the candidate list
        // decodes with an empty one, and without `candidates_built` a live
        // search would read that as "nothing left to try" and converge on its
        // baseline the moment it was deployed over.
        let legacy = r#"{"phase":"searching","baseline_ms":100,"best_ms":100}"#;
        let state: NodeFusionState = serde_json::from_str(legacy).expect("an old row must decode");
        assert_eq!(state.phase, "searching");
        assert_eq!(state.baseline_ms, 100);
        assert!(!state.candidates_built, "the search still has work to do");
        assert!(state.candidates.is_empty());
        assert!(state.best_set.is_none());
    }

    #[test]
    fn an_empty_best_set_is_a_configuration_and_not_an_absence() {
        // HMP can spell "nothing won yet" as an empty combo because its empty
        // combo *is* its baseline. Ours is not: the empty set means every CTE
        // plain, which is a real answer the search can arrive at, so the two
        // have to be different values.
        let mut state = NodeFusionState::default();
        assert!(state.best_set.is_none(), "the default rule is the incumbent");
        state.best_set = Some(Vec::new());
        let round_tripped: NodeFusionState =
            serde_json::from_str(&serde_json::to_string(&state).unwrap()).unwrap();
        assert_eq!(
            round_tripped.best_set,
            Some(Vec::new()),
            "'every CTE plain' must survive the round trip as itself"
        );
    }

    // ------------------------------------------------------------------
    // The search, driven end to end
    // ------------------------------------------------------------------

    /// A warehouse big enough for a materialization decision to have an effect
    /// worth measuring.
    ///
    /// `two_table_dag`'s twenty rows are fine for checking that a rewrite
    /// produces the right relations and useless for checking that a *search*
    /// finds anything: every configuration of a twenty-row DAG costs about the
    /// same, so the honest answer is always "the default rule is fine" and a
    /// test that asserted a promotion would be asserting noise.
    async fn setup_wide(conn: &DuckDBConnection) {
        conn.execute(
            "CREATE TABLE events AS SELECT \
                range AS event_id, \
                range % 5000 AS customer_id, \
                (range % 7)::VARCHAR AS channel, \
                range * 0.25 AS value \
             FROM range(200000)"
                .to_string(),
        )
        .await
        .unwrap();
        conn.execute(
            "CREATE TABLE customers AS SELECT \
                range AS customer_id, \
                CASE WHEN range % 3 = 0 THEN 'US' ELSE 'EU' END AS region \
             FROM range(5000)"
                .to_string(),
        )
        .await
        .unwrap();
    }

    /// A join-and-aggregate View that three Tables read.
    ///
    ///   events ─┐
    ///           ├─► enriched (View) ──┬──► by_region  (Table)
    ///   customers┘                    ├──► by_channel (Table)
    ///                                 └──► top_spend  (Table)
    ///
    /// Three readers of one expensive computation is the shape fusion exists
    /// for and the shape a materialization decision can actually move: plain,
    /// the fused query is free to unfold the join into each branch; materialized,
    /// it runs once. That is a difference the search should be able to see.
    fn wide_dag() -> Dag {
        let mut dag = make_dag(vec![
            node(
                "enriched",
                "SELECT e.event_id, e.customer_id, e.channel, e.value, c.region \
                 FROM events e JOIN customers c ON e.customer_id = c.customer_id \
                 WHERE e.value > 1",
                MaterializeMode::View,
                &[],
            ),
            node(
                "by_region",
                "SELECT region, count(*) AS n, sum(value) AS total FROM enriched GROUP BY region",
                MaterializeMode::Table,
                &["enriched"],
            ),
            node(
                "by_channel",
                "SELECT channel, count(*) AS n, sum(value) AS total FROM enriched GROUP BY channel",
                MaterializeMode::Table,
                &["enriched"],
            ),
            node(
                "top_spend",
                "SELECT customer_id, sum(value) AS total FROM enriched \
                 GROUP BY customer_id ORDER BY total DESC LIMIT 10",
                MaterializeMode::Table,
                &["enriched"],
            ),
        ]);
        dag.sources = vec![
            SourceNode {
                name: "events".to_string(),
                schema: Arc::new(duckdb::arrow::datatypes::Schema::empty()),
            },
            SourceNode {
                name: "customers".to_string(),
                schema: Arc::new(duckdb::arrow::datatypes::Schema::empty()),
            },
        ];
        dag
    }

    /// Run the adaptive search to convergence against a real DuckDB, the way
    /// `dee optimize` does: the batch driver supplies the executions the search
    /// would otherwise wait on the schedule for.
    ///
    /// This is the test that would catch the search being wired up wrong rather
    /// than merely compiling -- a baseline that never lands, a cursor that never
    /// advances, a promote that installs the trial instead of the winner. The
    /// unit tests above pin the pieces; this one pins that they add up.
    async fn search_to_convergence(objective: Objective) -> (Dag, OptimizeReport) {
        let conn = in_memory_conn().await;
        setup_wide(&conn).await;
        // Plans on, as `dee optimize` does it: the search ranks by pricing plan
        // regions with constants fitted to executed plans, so a run without
        // them leaves it with nothing to rank -- and it would look exactly like
        // "no candidate beats the default rule".
        let engine = Arc::new(
            SimpleEngine::new(Arc::clone(&conn))
                .unwrap()
                .with_profiling(crate::executor::ProfilingConfig {
                    collect_plans: true,
                    sample_interval: std::time::Duration::from_millis(250),
                }),
        );

        let config = adaptive_config()
            .with_nodefusion_objective(objective)
            .with_nodefusion_max_runs(3);
        config.validate().expect("the config under test must be valid");

        let mut optimizer =
            Optimizer::new_with_config(Arc::clone(&conn), Arc::clone(&engine), config);
        let mut dag = wide_dag();
        // A real store, not `NullStoreFactory`. A continuous pass keeps its
        // whole search in the store, so against a null one every step reads
        // "not registered", answers `Idle`, and the batch driver spends its
        // 512-iteration ceiling discovering that. The same is true of HMP; it
        // is the contract, not a quirk of this pass.
        let stores = MemoryStoreFactory::open().unwrap();
        let report = optimizer
            .run(&mut dag, "dag-adaptive", "pipeline", 1, &stores)
            .await
            .expect("the adaptive search should converge");
        (dag, report)
    }

    #[tokio::test]
    async fn the_search_converges_on_a_fused_dag_that_still_produces_the_same_rows() {
        let conn = in_memory_conn().await;
        setup_wide(&conn).await;
        let relations = ["by_region", "by_channel", "top_spend"];
        let expected = run_and_fingerprint(&conn, &wide_dag(), &relations).await;
        drop(conn);

        let (dag, report) = search_to_convergence(Objective::Makespan).await;

        // It actually fused, rather than converging on "leave it alone".
        assert!(
            dag.nodes.get("dee_fused".to_string()).is_some(),
            "the promoted DAG should be the fused one"
        );
        let pass = report
            .passes
            .iter()
            .find(|p| p.pass.contains("NodeFusion"))
            .expect("the pass should report");
        let crate::opt::PassDetail::NodeFusion(detail) = &pass.detail else {
            panic!("expected a NodeFusion detail, got {:?}", pass.detail);
        };
        assert!(detail.adaptive, "it ran as a search");
        assert_eq!(detail.objective.as_deref(), Some("makespan"));

        // It spent DAG runs, which is the property adaptive gives up and the
        // benchmark has to account for. A search reporting zero runs has not
        // measured anything.
        assert!(
            pass.dag_runs_used > 0,
            "the search measures a baseline and then candidates; it cannot be free"
        );

        // And whatever it chose, the DAG still delivers what it delivered
        // before. This is the non-negotiable one: the search may move the hint
        // around, never the rows.
        let conn = in_memory_conn().await;
        setup_wide(&conn).await;
        let actual = run_and_fingerprint(&conn, &dag, &relations).await;
        assert_eq!(
            expected, actual,
            "the searched materialization must not change what the DAG produces"
        );
    }

    #[tokio::test]
    async fn both_objectives_converge_and_report_the_measure_they_accepted_on() {
        // Not an assertion that the two pick *different* sets -- on a DAG this
        // small they may well agree, and asserting a disagreement would be
        // asserting a property of the fixture rather than of the search. What
        // has to hold is that each one runs to completion and records the
        // measure it was actually judging on, because ranking by one and
        // accepting on the other is the failure mode the setting exists to
        // prevent.
        for (objective, name) in [
            (Objective::Makespan, "makespan"),
            (Objective::QueryTime, "query_time"),
        ] {
            let (dag, report) = search_to_convergence(objective).await;
            assert!(dag.nodes.get("dee_fused".to_string()).is_some());
            let pass = report
                .passes
                .iter()
                .find(|p| p.pass.contains("NodeFusion"))
                .expect("the pass should report");
            let crate::opt::PassDetail::NodeFusion(detail) = &pass.detail else {
                panic!("expected a NodeFusion detail");
            };
            assert_eq!(detail.objective.as_deref(), Some(name));
            assert!(
                detail.baseline_runtime_ms.is_some(),
                "{name}: a search with no baseline has nothing to compare against"
            );
        }
    }

    #[tokio::test]
    async fn a_rule_driven_fusion_still_spends_no_dag_runs() {
        // The property adaptive gives up, asserted on the variant that keeps
        // it -- so a later change that quietly made every fusion a search would
        // fail here rather than in a benchmark months later.
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;
        let engine = Arc::new(SimpleEngine::new(Arc::clone(&conn)).unwrap());

        let mut optimizer = Optimizer::new_with_config(
            Arc::clone(&conn),
            Arc::clone(&engine),
            OptimizerConfig::default()
                .with_all_disabled()
                .with_nodefusion_pass()
                .with_nodefusion_naive_materialize_ctes(true),
        );
        let mut dag = layered_dag();
        let stores = MemoryStoreFactory::open().unwrap();
        let report = optimizer
            .run(&mut dag, "dag-rule", "pipeline", 1, &stores)
            .await
            .expect("the rewrite should succeed");

        assert_eq!(
            report.dag_runs_used, 0,
            "a rule-driven fusion decides everything from the DAG in front of it"
        );
        assert!(dag.nodes.get("dee_fused".to_string()).is_some());
    }


    #[tokio::test]
    async fn duckdb_plans_the_fused_query_the_same_with_and_without_the_hint() {
        // The finding that decides what the adaptive search can do on this
        // backend, pinned so it is a recorded fact rather than something
        // rediscovered the next time the search "mysteriously" finds nothing.
        //
        // DuckDB already materializes a CTE referenced more than once, so
        // `AS MATERIALIZED` asks for what it was going to do anyway and the
        // plan does not move. PostgreSQL is the opposite -- a CTE is inlined by
        // default since 12 -- and is the backend the search is actually for.
        //
        // If this test ever starts failing, that is good news: it means DuckDB
        // began distinguishing the two, and the search has a signal on it.
        let conn = in_memory_conn().await;
        setup_wide(&conn).await;
        let engine = SimpleEngine::new(Arc::clone(&conn)).unwrap();
        let mut dag = wide_dag();
        engine.resolve_schemas(&mut dag).await.unwrap();

        let dialect = dialect_for_db(&dag.db);
        let plan = NodeFusionPass::new(false, false, None).plan(&dag).unwrap();
        let decidable: HashSet<String> = plan.decidable().into_iter().collect();
        assert!(
            !decidable.is_empty(),
            "the fixture must have something the search could decide about"
        );

        let explain_of = async |set: &HashSet<String>| {
            let p = plan_fusion(&dag, MaterializeRule::Exact(set)).unwrap();
            let FusedSql { sql, .. } = build_fused_sql(&dag, &p, dialect).unwrap();
            // The SQL itself *does* differ -- that is the point of the hint.
            assert_eq!(sql.contains("AS MATERIALIZED"), !set.is_empty());
            conn.explain(&sql).await.unwrap().unwrap()
        };

        let plain = explain_of(&HashSet::new()).await;
        let hinted = explain_of(&decidable).await;

        let ops = |raw: &str| {
            let roots = conn.parse_plan(raw).unwrap();
            fn walk(n: &PlanNode, acc: &mut std::collections::BTreeMap<String, usize>) {
                *acc.entry(n.operator.clone()).or_insert(0) += 1;
                for c in &n.children {
                    walk(c, acc);
                }
            }
            let mut acc = std::collections::BTreeMap::new();
            for r in &roots {
                walk(r, &mut acc);
            }
            acc
        };
        assert_eq!(
            ops(&plain),
            ops(&hinted),
            "DuckDB plans the decidable CTEs identically hinted or not, so the fused plan \
             carries no signal a cost-based rule could rank on"
        );

        // The narrower claim is the true one. Materializing *every* CTE --
        // including the singly-referenced ones the search deliberately leaves
        // alone -- does change the plan, so a check written over all of them
        // would wrongly conclude the engine responds.
        let all: HashSet<String> = plan.ctes.iter().map(|c| c.node_id.clone()).collect();
        assert_ne!(
            ops(&plain),
            ops(&explain_of(&all).await),
            "the hint is not a global no-op on DuckDB; it is a no-op for the \
             multiply-referenced CTEs, which is exactly the decidable set"
        );
    }

    #[tokio::test]
    async fn a_search_with_no_signal_says_so_instead_of_converging_quietly() {
        // The consequence of the test above, and the behaviour that matters:
        // on DuckDB the search finds nothing, and it has to be possible to tell
        // that apart from "the default rule is optimal". They are the same
        // outcome and completely different facts.
        let conn = in_memory_conn().await;
        setup_wide(&conn).await;
        let model = LearnedCostModel::new();
        let pass = NodeFusionPass::from_config(&adaptive_config());
        let engine = SimpleEngine::new(Arc::clone(&conn)).unwrap();
        let mut dag = wide_dag();
        engine.resolve_schemas(&mut dag).await.unwrap();

        let setup = pass.build_candidates(&*conn, &dag, &model).await;
        assert!(setup.candidates.is_empty());
        let reason = setup
            .reason
            .expect("a search that tries nothing must say why it tried nothing");
        assert!(
            !reason.is_empty(),
            "the reason is what a reader has instead of guessing"
        );
    }

    #[test]
    fn the_makespan_axis_is_work_plus_waiting_and_not_waiting_alone() {
        // The failure this guards: ranked on the barrier chain alone, the
        // configuration with the fewest materializations wins every time --
        // barriers only ever add to that number -- so "makespan" would stop
        // being an objective and become a constant opinion that materializing
        // is bad. Adding the work term is what makes it a tradeoff again.
        //
        // Checked on the chain model directly rather than through a live
        // EXPLAIN, because the property is arithmetic and a fixture that
        // happened to make it true would not be evidence.
        let chain = chain_of_three();
        let profiles = profiles_of(&[
            ("a", 1.0, Some(0.5)),
            ("b", 2.0, Some(0.5)),
            ("c", 4.0, Some(0.5)),
        ]);
        let none = HashSet::new();
        let one: HashSet<String> = ["b".to_string()].into_iter().collect();

        // The chain alone says materializing is strictly worse, always.
        assert!(
            barrier_chain(&chain, &profiles, &none).unwrap()
                < barrier_chain(&chain, &profiles, &one).unwrap()
        );

        // With the work term, a configuration that removes enough work wins
        // despite adding a barrier. `work` here stands for what the engine's
        // plan would say; the point is only that the axis can go either way.
        let axis = |work: f64, set: &HashSet<String>| {
            work + barrier_chain(&chain, &profiles, set).unwrap()
        };
        assert!(
            axis(6.0, &one) < axis(10.0, &none),
            "a barrier that removes four units of work should beat no barrier \
             at all; on the chain alone it never could"
        );
    }
}
