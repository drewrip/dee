//! NodeFusion -- share the View work several Tables repeat, through one rollup.
//!
//! A dee DAG executes as one relation per node: every View is created as a
//! view and every Table as its own `CREATE TABLE ... AS`, so a Table built on
//! Views re-executes their SQL. When several Tables share upstream Views the
//! same work runs once per Table -- the duplication HMP and OMP spend measured
//! DAG runs deciding how to materialize away.
//!
//! This pass removes it structurally, following the rollup spec (`SRS.md`):
//!
//! 1. Count how many times each View's SQL runs across the stored builds --
//!    paths, not references -- and share every View that runs twice or more,
//!    together with the Views between it and a table.
//! 2. Paste the shared models into one new node, a `TempTable` whose query is a
//!    `WITH` chain, and stack the rows each reader needs into one `UNION ALL`
//!    discriminated by `kind`.
//! 3. Point each reader at the rollup. A Table whose query moved in becomes a
//!    projection of its kind; any other keeps its own SQL and reads the kind
//!    through a CTE of its own.
//!
//! ```sql
//! -- node dee_fused (TempTable)
//! WITH n_stg AS (<stg sql>),
//!      n_facts AS MATERIALIZED (<facts sql, its reference to stg now n_stg>),
//!      n_by_region_v AS (<by_region_v sql>),
//!      n_totals AS (<totals sql>)
//! SELECT 1 AS kind, "region", "d", CAST(NULL AS BIGINT) AS "n", ... FROM n_by_region_v
//! UNION ALL
//! SELECT 2 AS kind, CAST(NULL AS VARCHAR) AS "region", ..., "n", "s" FROM n_totals
//! ORDER BY kind
//!
//! -- node rpt_region: keeps its SQL          -- node totals: its query moved in
//! WITH dee_k1_by_region_v AS (               SELECT "n" AS "n", "s" AS "s"
//!   SELECT "region" AS "region", "d" AS "d"  FROM dee_fused WHERE kind = 2
//!   FROM dee_fused WHERE kind = 1)
//! SELECT * FROM dee_k1_by_region_v AS by_region_v
//! ```
//!
//! Which models are shared, the rollup's query and every reader's rewrite live
//! in [`rollup`]. This module plans under a materialization rule, installs the
//! result, and runs the adaptive search over that rule.
//!
//! # What is preserved
//!
//! Every relation the DAG created is still created, under the same name and as
//! the same kind of relation, with the same columns, types and rows. Views are
//! never rewritten. A Table whose query decides its stored order or reads the
//! clock -- `ORDER BY`, `LIMIT`, `current_timestamp`, `random()` -- keeps its
//! query and runs it itself, so its stored order and its timestamps stay its
//! own. Each branch fills the other kinds' columns with NULLs cast to the
//! engine's own type, read off the engine: PostgreSQL resolves `UNION` types
//! pairwise, and two leading untyped NULLs resolve to text.
//!
//! # What is materialized
//!
//! A CTE is emitted `AS MATERIALIZED` when two or more CTEs or output branches
//! read it inside the rollup. `nodefusion_materialize_ctes_override` names a
//! different set, and the adaptive search measures candidate sets against the
//! rule.
//!
//! # What is left alone
//!
//! A DAG where no View runs twice has nothing to share and is left exactly as
//! it was. A View that reads a stored node -- a Table, or a `TempTable` a
//! materialization search placed -- stays outside the rollup, since that node
//! is itself built from it; it is pasted into the stored builds that read it.
//! A reader whose query cannot be rewritten on the parsed AST keeps reading
//! what it read before, and the report names it.

use std::collections::{BTreeSet, HashMap, HashSet};
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
            bare_table_name, default_spool_factor, dialect_for_db, supports_materialized_hint,
        },
        dup::{SubtreeCost, SubtreeCostMethod},
        explain::{render_card_grid, render_ranked_table},
        learned::LearnedCostModel,
        report::{
            IterationStat, NodeFusionCte, NodeFusionDetail, NodeFusionKind, PassDetail,
            PassOutcome,
        },
        step::{
            BudgetMetric, OptimizationType, RegisterContext, StepContext, StepOutcome, StepPhase,
        },
        store::{OptStore, Registration},
    },
};

mod rollup;

/// The bare name of the fused node, before the schema prefix it inherits from
/// the nodes it stands in front of.
const FUSED_BASE: &str = "dee_fused";

/// The discriminator column every branch of the fused `UNION ALL` leads with.
const KIND_COLUMN: &str = "kind";

// ---------------------------------------------------------------------------
// The pass
// ---------------------------------------------------------------------------

pub struct NodeFusionPass {
    /// When set, the exact set of node IDs whose CTEs are materialized,
    /// replacing the rollup's own rule (two or more readers inside it). A name
    /// matches either as the full node ID or as its bare table name.
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
    /// `exec(V)` for every View, sorted by node ID.
    exec: Vec<(String, usize)>,
    /// The `WITH` chain, in emission order.
    ctes: Vec<ExplainCte>,
    kinds: Vec<NodeFusionKind>,
    /// The rollup's columns after `kind`, with the engine's type for each.
    columns: Vec<(String, String)>,
    untouched: Vec<(String, String)>,
}

struct ExplainCte {
    node_id: String,
    reason: String,
    cte_name: String,
    readers: usize,
    materialized: bool,
    decidable: bool,
}

impl NodeFusionPass {
    pub fn new(materialize_override: Option<Vec<String>>) -> Self {
        Self {
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
        let mut pass = Self::new(config.nodefusion_materialize_ctes_override.clone());
        if config.nodefusion_materialize_ctes || config.nodefusion_naive_materialize_ctes {
            warn!(
                "nodefusion: nodefusion_materialize_ctes and nodefusion_naive_materialize_ctes \
                 are no longer read. The rollup materializes exactly the CTEs with two or more \
                 readers inside it; use nodefusion_materialize_ctes_override to name a \
                 different set"
            );
        }
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
        if self.adaptive && self.materialize_override.is_some() {
            return Err(OptimizerError::Config(
                "nodefusion: the adaptive search and an explicit materialize-CTE override \
                 are incompatible -- the override pins the exact set the search exists to \
                 find"
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

/// One CTE in the rollup's `WITH` chain.
#[derive(Debug, Clone)]
struct CtePlan {
    node_id: String,
    cte_name: String,
    materialized: bool,
    /// A Table's own query, moved into the rollup (spec step 3.3).
    is_table: bool,
    /// The other CTEs this one reads. The `WITH` chain is a DAG of its own, and
    /// this is its edge set -- what the adaptive search walks to find the path a
    /// materialized CTE puts a barrier on. See [`barrier_chain`].
    reads: Vec<String>,
    /// How many CTEs and output branches read this one inside the rollup.
    ///
    /// More than one is what makes its `MATERIALIZED` flag *decidable*: a CTE
    /// read once is unfolded once whatever the hint says, so there is nothing
    /// to decide and nothing to measure.
    readers: usize,
    /// Whether this CTE's flag is the search's to choose.
    decidable: bool,
}

#[derive(Debug, Clone)]
struct FusionPlan {
    rollup: rollup::RollupPlan,
    /// The `WITH` chain, in emission order.
    ctes: Vec<CtePlan>,
    /// The CTEs emitted `AS MATERIALIZED`.
    materialized: BTreeSet<String>,
}

impl FusionPlan {
    /// The CTEs whose `MATERIALIZED` flag the adaptive search gets to choose,
    /// in the chain's own order: those read more than once inside the rollup.
    fn decidable(&self) -> Vec<String> {
        self.ctes
            .iter()
            .filter(|c| c.decidable)
            .map(|c| c.node_id.clone())
            .collect()
    }

    /// The set currently marked `AS MATERIALIZED`.
    fn materialized_set(&self) -> HashSet<String> {
        self.materialized.iter().cloned().collect()
    }
}

/// What decides which CTEs are emitted `AS MATERIALIZED`.
///
/// Split out of the pass so the adaptive search can plan the same DAG under a
/// set it is pricing without building a second pass to hold the setting.
#[derive(Debug, Clone, Copy)]
enum MaterializeRule<'a> {
    /// The rollup's rule -- two or more readers inside it -- unless an override
    /// names the set.
    Configured { over: Option<&'a [String]> },
    /// Exactly this set.
    Exact(&'a HashSet<String>),
}

/// Whether an override names `node_id`, as the full ID or its bare table name,
/// so `"wh"."main"."orders"` can be asked for as `orders`.
fn override_matches(list: &[String], node_id: &str) -> bool {
    list.iter()
        .any(|n| n == node_id || bare_table_name(n) == bare_table_name(node_id))
}

impl MaterializeRule<'_> {
    fn choose(&self, plan: &rollup::RollupPlan) -> BTreeSet<String> {
        let members = plan.selection.order.iter();
        match self {
            MaterializeRule::Exact(set) => members.filter(|id| set.contains(*id)).cloned().collect(),
            MaterializeRule::Configured { over: Some(list) } => members
                .filter(|id| override_matches(list, id))
                .cloned()
                .collect(),
            MaterializeRule::Configured { over: None } => plan.selection.default_materialized(),
        }
    }
}

impl NodeFusionPass {
    /// Decide what to share under the pass's own configuration.
    fn plan(&self, dag: &Dag) -> Result<FusionPlan, NotFusable> {
        plan_fusion(
            dag,
            MaterializeRule::Configured {
                over: self.materialize_override.as_deref(),
            },
        )
    }
}

/// Decide what the rollup shares and which of its CTEs are materialized, or
/// why this DAG has nothing to share.
///
/// A free function rather than a method because the adaptive search plans the
/// same DAG dozens of times under different materialization sets, and it has no
/// business constructing a pass to do it.
fn plan_fusion(dag: &Dag, rule: MaterializeRule<'_>) -> Result<FusionPlan, NotFusable> {
    let rollup = rollup::plan_rollup(dag)?;
    // Nothing to decide where the dialect ignores the hint.
    let hint = supports_materialized_hint(dialect_for_db(&dag.db));
    let materialized = if hint {
        rule.choose(&rollup)
    } else {
        BTreeSet::new()
    };
    let members: HashSet<&String> = rollup.selection.order.iter().collect();
    let ctes = rollup
        .selection
        .order
        .iter()
        .map(|id| {
            let mut reads: Vec<String> = dag
                .nodes
                .get(id.clone())
                .map(|n| {
                    n.depends_on
                        .iter()
                        .filter(|d| members.contains(d))
                        .cloned()
                        .collect()
                })
                .unwrap_or_default();
            reads.sort();
            let readers = rollup.selection.readers.get(id).copied().unwrap_or(0);
            CtePlan {
                node_id: id.clone(),
                cte_name: rollup.cte_names[id].clone(),
                materialized: materialized.contains(id),
                is_table: matches!(
                    rollup.selection.members.get(id),
                    Some(rollup::CteReason::TableQuery { .. })
                ),
                reads,
                readers,
                // A CTE read once is computed exactly once either way, so both
                // settings produce the same work and the search would be
                // spending runs on a coin flip.
                decidable: hint && readers > 1,
            }
        })
        .collect();
    Ok(FusionPlan {
        rollup,
        ctes,
        materialized,
    })
}

/// Why a model is in the rollup, in words.
fn describe_reason(reason: &rollup::CteReason) -> String {
    match reason {
        rollup::CteReason::Repeated { exec } => format!("its SQL runs {exec} times"),
        rollup::CteReason::OnPath => "on a path from a shared model to a table".to_string(),
        rollup::CteReason::Upstream => "read by a shared model".to_string(),
        rollup::CteReason::TableQuery { parent } => {
            format!("a table query over the shared {}", bare_table_name(parent))
        }
    }
}

fn describe_consumer(consumer: &rollup::KindConsumer) -> String {
    match consumer {
        rollup::KindConsumer::Own(id) => format!("{id} (its own query)"),
        rollup::KindConsumer::Direct(id) => format!("{id} (reads it)"),
        rollup::KindConsumer::Pasted(id) => format!("{id} (pasted into its readers)"),
    }
}

impl NodeFusionPass {
    /// Rewrite `dag` in place under this pass's configuration. Returns what
    /// happened, for the report.
    ///
    /// Needs the connection the DAG will run on: each kind's columns are read
    /// off the engine, so a branch can fill another kind's columns with a NULL
    /// of exactly the right type.
    pub async fn rewrite<C>(&mut self, conn: &C, dag: &mut Dag) -> Result<PassOutcome, OptimizerError>
    where
        C: Connector + Send + Sync,
    {
        // Cloned rather than borrowed: the rule holds a slice of
        // `materialize_override`, and `rewrite_under` needs `&mut self` for the
        // explain data it fills in.
        let over = self.materialize_override.clone();
        self.rewrite_under(conn, dag, MaterializeRule::Configured { over: over.as_deref() })
            .await
    }

    /// The CTEs the adaptive search would be allowed to decide about on `dag`:
    /// those read more than once inside the rollup.
    ///
    /// Public so tooling can enumerate the configurations the search can reach
    /// without reimplementing the reader count -- in particular the
    /// `nodefusion_validate` example, which checks every one of them against a
    /// real warehouse. Empty when the DAG has nothing to share.
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
    pub async fn rewrite_with<C>(
        &mut self,
        conn: &C,
        dag: &mut Dag,
        set: &HashSet<String>,
    ) -> Result<PassOutcome, OptimizerError>
    where
        C: Connector + Send + Sync,
    {
        self.rewrite_under(conn, dag, MaterializeRule::Exact(set)).await
    }

    async fn rewrite_under<C>(
        &mut self,
        conn: &C,
        dag: &mut Dag,
        rule: MaterializeRule<'_>,
    ) -> Result<PassOutcome, OptimizerError>
    where
        C: Connector + Send + Sync,
    {
        let plan = match plan_fusion(dag, rule) {
            Ok(plan) => plan,
            Err(why) => return Ok(self.not_fused(why)),
        };
        let columns = match rollup::resolve_kind_columns(conn, dag, &plan.rollup).await {
            Ok(columns) => columns,
            Err(why) => return Ok(self.not_fused(why)),
        };
        let emitted = match rollup::emit(dag, &plan.rollup, &columns, &plan.materialized) {
            Ok(emitted) => emitted,
            Err(why) => return Ok(self.not_fused(why)),
        };

        // Have the engine plan the rollup before installing it, so a query it
        // rejects is a failed optimization rather than a DAG that only breaks
        // at run time. The engine is the authority here: CTE bodies that name
        // no other member are carried through as authored, and may use syntax
        // the parser does not cover.
        if let Err(e) = conn.column_types(&emitted.fused_sql).await {
            return Err(OptimizerError::Exec(format!(
                "nodefusion built a rollup the engine rejects ({e}); refusing to install it"
            )));
        }
        debug!(
            "nodefusion: {} of {} CTE bod(ies) used as authored",
            emitted.verbatim_bodies,
            plan.ctes.len()
        );

        // On a copy, so a rewrite that leaves the graph inconsistent leaves the
        // DAG exactly as it was.
        let mut next = dag.clone();
        rollup::install(&mut next, &emitted)
            .map_err(|e| OptimizerError::Exec(format!("nodefusion: {e}")))?;
        *dag = next;

        let sel = &plan.rollup.selection;
        let mut exec: Vec<(String, usize)> = sel.exec.iter().map(|(k, v)| (k.clone(), *v)).collect();
        exec.sort();
        let kinds: Vec<NodeFusionKind> = sel
            .kinds
            .iter()
            .map(|k| NodeFusionKind {
                kind: k.kind,
                cte: k.cte.clone(),
                consumers: k.consumers.iter().map(describe_consumer).collect(),
                pushed_filter: emitted
                    .pushed
                    .iter()
                    .find(|(n, _)| *n == k.kind)
                    .map(|(_, p)| p.clone()),
            })
            .collect();
        let columns: Vec<(String, String)> = emitted
            .columns
            .iter()
            .map(|c| (c.name.clone(), c.ty.clone()))
            .collect();
        let views_inlined = plan.ctes.iter().filter(|c| !c.is_table).count();
        let outcome = format!(
            "shared {} model(s) through '{}', read back as {} kind(s) by {} rewritten node(s)",
            plan.ctes.len(),
            emitted.fused_id,
            kinds.len(),
            emitted.rewrites.len()
        );
        debug!("nodefusion: {outcome}");

        self.explain_data = Some(ExplainData {
            outcome: outcome.clone(),
            fused_id: Some(emitted.fused_id.clone()),
            exec: exec.clone(),
            ctes: plan
                .ctes
                .iter()
                .map(|c| ExplainCte {
                    node_id: c.node_id.clone(),
                    reason: describe_reason(&sel.members[&c.node_id]),
                    cte_name: c.cte_name.clone(),
                    readers: c.readers,
                    materialized: c.materialized,
                    decidable: c.decidable,
                })
                .collect(),
            kinds: kinds.clone(),
            columns: columns.clone(),
            untouched: emitted.untouched.clone(),
        });

        let mut record = PassOutcome::empty().with_detail(PassDetail::NodeFusion(
            NodeFusionDetail {
                fused_node: Some(emitted.fused_id.clone()),
                tables_fused: emitted.rewrites.len(),
                views_inlined,
                materialized_ctes: emitted.materialized.len(),
                fused_columns: emitted.columns.len(),
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
                exec_counts: exec,
                ctes: plan
                    .ctes
                    .iter()
                    .map(|c| NodeFusionCte {
                        node: c.node_id.clone(),
                        reason: describe_reason(&sel.members[&c.node_id]),
                        readers: c.readers,
                        materialized: c.materialized,
                    })
                    .collect(),
                kinds,
                columns,
                pushed_filters: emitted.pushed.clone(),
                untouched: emitted.untouched.clone(),
            },
        ));
        // One change: the DAG gained a rollup. The nodes rewritten to read it
        // are that change's consequence, not separate decisions.
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
            exec: Vec::new(),
            ctes: Vec::new(),
            kinds: Vec::new(),
            columns: Vec::new(),
            untouched: Vec::new(),
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
            exec_counts: Vec::new(),
            ctes: Vec::new(),
            kinds: Vec::new(),
            columns: Vec::new(),
            pushed_filters: Vec::new(),
            untouched: Vec::new(),
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
          <div class="subtle">Candidate sets of CTEs to mark <code>AS MATERIALIZED</code>, priced by EXPLAINing the rollup with each set applied and ordered by the objective. <b>Work removed</b> is relative to the default rule, higher is better; <b>Path</b> is the longest chain through the WITH clause with that set materialized, lower is better. The two disagree because materializing removes repeated computation and inserts a barrier, which is why the objective picks both the order candidates are trialled in and the test each has to pass. The spool term behind both is modelled rather than measured.</div>
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
                "Rollup node",
                data.fused_id.clone().unwrap_or_else(|| "-".into()),
            ),
            ("CTEs", data.ctes.len().to_string()),
            (
                "Materialized CTEs",
                data.ctes.iter().filter(|c| c.materialized).count().to_string(),
            ),
            ("Kinds", data.kinds.len().to_string()),
            ("Rollup columns", data.columns.len().to_string()),
        ]);

        let exec_rows: Vec<Vec<String>> = data
            .exec
            .iter()
            .map(|(id, n)| vec![id.clone(), n.to_string()])
            .collect();
        let exec_table = render_ranked_table(&["View", "Executions"], &exec_rows);

        let cte_rows: Vec<Vec<String>> = data
            .ctes
            .iter()
            .enumerate()
            .map(|(i, cte)| {
                vec![
                    (i + 1).to_string(),
                    cte.node_id.clone(),
                    cte.reason.clone(),
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
            &["#", "Model", "Why it is shared", "CTE", "Readers", "Hint", "Decided by"],
            &cte_rows,
        );

        let search_panel = self.search_panel();

        let kind_rows: Vec<Vec<String>> = data
            .kinds
            .iter()
            .map(|k| {
                vec![
                    k.kind.to_string(),
                    k.cte.clone(),
                    k.consumers.join("; "),
                    k.pushed_filter.clone().unwrap_or_else(|| "-".into()),
                ]
            })
            .collect();
        let kind_table = render_ranked_table(&["kind", "Model", "Read by", "Pushed filter"], &kind_rows);

        let column_rows: Vec<Vec<String>> = data
            .columns
            .iter()
            .map(|(name, ty)| vec![name.clone(), ty.clone()])
            .collect();
        let column_table = render_ranked_table(&["Column", "Type"], &column_rows);

        let untouched = if data.untouched.is_empty() {
            String::new()
        } else {
            format!(
                r#"<div class="subtle"><b>Left reading what they read before:</b> {}</div>"#,
                data.untouched
                    .iter()
                    .map(|(node, why)| format!("{node} ({why})"))
                    .collect::<Vec<_>>()
                    .join("; ")
            )
        };

        format!(
            r#"<div class="section-stack">
        {cards}
        {search_panel}
        <div class="panel">
          <h2>Executions</h2>
          <div class="subtle">How many times each View's SQL runs across the stored builds: the sum, over its readers, of 1 for a table and the reader's own count for a view. Paths, not references. A View that runs twice or more is what the rollup shares.</div>
          {exec_table}
        </div>
        <div class="panel">
          <h2>The WITH chain</h2>
          <div class="subtle">Every shared model, in topological order, and why it is in the rollup. A CTE is materialized when two or more CTEs or output branches read it inside the rollup.</div>
          {cte_table}
        </div>
        <div class="panel">
          <h2>Kinds</h2>
          <div class="subtle">The shared models something outside the rollup reads, one <code>UNION ALL</code> branch each. A table whose own query moved in becomes a projection of its kind; any other reader keeps its SQL and reads the kind through a CTE of its own. Views are never rewritten.</div>
          {untouched}
          {kind_table}
        </div>
        <div class="panel">
          <h2>Rollup columns</h2>
          <div class="subtle">After <code>kind</code>. A column shared by name and type appears once; a clash is renamed <code>model__column</code> and aliased back. Each branch fills the others with NULLs cast to the engine's own type.</div>
          {column_table}
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
    /// The rollup being priced and its kinds' column types, resolved once:
    /// neither depends on which CTEs are materialized.
    plan: &'a rollup::RollupPlan,
    columns: &'a rollup::KindColumns,
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
        let materialized: BTreeSet<String> = set.iter().cloned().collect();
        let sql = rollup::emit(self.dag, self.plan, self.columns, &materialized)
            .ok()?
            .fused_sql;
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
        columns: &rollup::KindColumns,
        coster: &dyn SubtreeCost,
        model: &LearnedCostModel,
        dialect: DialectType,
    ) -> Option<HashMap<String, CteProfile>>
    where
        C: Connector + Send + Sync,
    {
        let all: BTreeSet<String> = plan.ctes.iter().map(|c| c.node_id.clone()).collect();
        let sql = rollup::emit(dag, &plan.rollup, columns, &all).ok()?.fused_sql;
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
        for cte in &plan.ctes {
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
            return SearchSetup::nothing("this DAG has nothing for a rollup to share, so there is no WITH chain to search");
        };
        let decidable = plan.decidable();
        if decidable.is_empty() {
            return SearchSetup::nothing(
                "no CTE in the rollup has more than one reader, so every one of them is \
                 computed once whichever way the hint goes",
            );
        }

        let columns = match rollup::resolve_kind_columns(conn, dag, &plan.rollup).await {
            Ok(columns) => columns,
            Err(why) => {
                return SearchSetup::nothing(&format!(
                    "the rollup's column types could not be resolved: {why}"
                ));
            }
        };

        let coster = self.cost_model.coster(model);
        let Some(profiles) = self
            .profile_ctes(conn, dag, &plan, &columns, coster.as_ref(), model, dialect)
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
            plan: &plan.rollup,
            columns: &columns,
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
    async fn apply<C>(
        &mut self,
        conn: &C,
        dag: &mut Dag,
        set: Option<&[String]>,
    ) -> Result<PassOutcome, OptimizerError>
    where
        C: Connector + Send + Sync,
    {
        match set {
            None => self.rewrite(conn, dag).await,
            Some(members) => {
                let exact: HashSet<String> = members.iter().cloned().collect();
                self.rewrite_with(conn, dag, &exact).await
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
        let conn = Arc::clone(&ctx.conn);

        match state.phase.as_str() {
            // Unlike HMP, the baseline is *not* the DAG as it stands. HMP's
            // baseline is the authored DAG and its `Before` step does nothing;
            // ours has to be the fused DAG under the default rule, or every
            // candidate would be compared against an unfused control and would
            // "win" for reasons that have nothing to do with which CTEs are
            // materialized.
            "baseline" => {
                let record = self.apply(conn.as_ref(), ctx.dag, None).await?;
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
                    let record = self.apply(conn.as_ref(), ctx.dag, Some(&in_flight.set)).await?;
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
                    if self
                        .apply(conn.as_ref(), &mut trial, Some(&candidate.set))
                        .await
                        .is_err()
                    {
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
                    let record = self
                        .apply(conn.as_ref(), ctx.dag, Some(&candidate.set))
                        .await?;
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
        let conn = Arc::clone(&ctx.conn);
        let record = self.apply(conn.as_ref(), ctx.dag, best.as_deref()).await?;
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
            let conn = Arc::clone(&ctx.conn);
            let record = self.rewrite(conn.as_ref(), ctx.dag).await?;
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
    // tests exercise the same path a run does. A rollup that only looks
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
    /// `stg` runs once for each Table, which is exactly the repetition the
    /// rollup exists to remove.
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

    /// A shared chain with a CTE read twice inside the rollup:
    ///
    ///   orders ─► stg ─► facts ─┬─► by_region_v (View) ─► rpt_region (Table)
    ///                           └────────────────────────► totals     (Table)
    ///
    /// `facts` runs twice, so it and `stg` are shared; `by_region_v` is on the
    /// path to a table. `totals` reads `facts`, which `by_region_v` also reads
    /// inside the rollup, so its own query moves in -- and `facts` then has two
    /// readers there and is the one CTE the rule materializes.
    fn shared_chain_dag() -> Dag {
        make_dag(vec![
            node("stg", "SELECT order_id, region, amount FROM orders", MaterializeMode::View, &[]),
            node(
                "facts",
                "SELECT order_id, region, amount * 2 AS doubled FROM stg",
                MaterializeMode::View,
                &["stg"],
            ),
            node(
                "by_region_v",
                "SELECT region, sum(doubled) AS d FROM facts GROUP BY region",
                MaterializeMode::View,
                &["facts"],
            ),
            node(
                "rpt_region",
                "SELECT * FROM by_region_v",
                MaterializeMode::Table,
                &["by_region_v"],
            ),
            node(
                "totals",
                "SELECT count(*) AS n, sum(doubled) AS s FROM facts",
                MaterializeMode::Table,
                &["facts"],
            ),
        ])
    }

    /// DAG layout:
    ///
    ///   orders ─► shared (View) ─┬─► mid (View) ─► tbl_a (Table)
    ///                            └────────────────► tbl_b (Table)
    ///          └─► lonely (View) ──────────────────► tbl_b (Table)
    ///
    /// `shared` runs twice and `mid` is on its path to a table; `lonely` runs
    /// once and stays outside the rollup.
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

    /// DAG layout:
    ///
    ///   orders ─► stg (View) ─► base (Table) ─► enriched (View) ─► rollup (Table)
    ///
    /// Every View runs once: `enriched` reads the stored `base`, not `stg`'s SQL.
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
    /// The types matter as much as the rows: a NULL fill cast to a type read
    /// back off a node's Arrow schema used to turn DuckDB's HUGEINT into
    /// DECIMAL(38,0). The rows were identical; the delivered table was not.
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

    async fn fuse(conn: &Arc<DuckDBConnection>, dag: &mut Dag) -> PassOutcome {
        NodeFusionPass::new(None)
            .rewrite(conn.as_ref(), dag)
            .await
            .expect("the rewrite should not error")
    }

    fn detail(record: &PassOutcome) -> &NodeFusionDetail {
        let PassDetail::NodeFusion(detail) = &record.detail else {
            panic!("expected a NodeFusion detail");
        };
        detail
    }

    // ------------------------------------------------------------------
    // End to end: the rewritten DAG produces the same relations
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn test_fused_dag_produces_the_same_tables() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;

        let baseline = run_and_fingerprint(&conn, &two_table_dag(), &["by_region", "totals"]).await;

        // Deliberately not schema-resolved: the rollup's column types come from
        // the engine, not from the nodes' Arrow schemas.
        let mut dag = two_table_dag();
        let record = fuse(&conn, &mut dag).await;
        assert_eq!(record.changes_applied, 1, "{}", detail(&record).outcome);

        let fused = run_and_fingerprint(&conn, &dag, &["by_region", "totals"]).await;
        assert_eq!(
            baseline, fused,
            "the rollup must not change a Table's columns or its rows"
        );
    }

    #[tokio::test]
    async fn test_the_rewritten_dag_has_the_shape_the_rollup_promises() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;
        let mut dag = two_table_dag();
        fuse(&conn, &mut dag).await;

        let fused = dag.nodes.get("dee_fused".to_string()).expect("rollup node");
        assert_eq!(fused.materialize, MaterializeMode::TempTable);
        assert!(
            fused.depends_on.is_empty(),
            "the rollup reads nothing but the warehouse's sources"
        );
        assert!(fused.query_text.trim_end().ends_with("ORDER BY kind"));

        for table in ["by_region", "totals"] {
            let node = dag.nodes.get(table.to_string()).unwrap();
            assert_eq!(node.materialize, MaterializeMode::Table);
            assert_eq!(
                node.depends_on,
                HashSet::from(["dee_fused".to_string()]),
                "a rewritten Table reads the rollup and nothing else"
            );
            assert!(node.query_text.contains("WHERE kind = "), "{}", node.query_text);
        }

        // The View stays in the graph, stays a View, and is not rewritten.
        let stg = dag.nodes.get("stg".to_string()).expect("the View is kept");
        assert_eq!(stg.materialize, MaterializeMode::View);
        assert_eq!(
            stg.query_text,
            two_table_dag().nodes.get("stg".to_string()).unwrap().query_text
        );
    }

    #[tokio::test]
    async fn test_fusion_preserves_column_types() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;

        // `sum(order_id)` is a HUGEINT, which reaches an Arrow schema as
        // Decimal128(38, 0) and cannot be told apart from a real
        // DECIMAL(38,0) on the way back.
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
        fuse(&conn, &mut dag).await;
        let fused = run_and_fingerprint(&conn, &dag, &["wide", "narrow"]).await;

        assert_eq!(
            baseline, fused,
            "a rewritten Table must keep its column types, not only its rows"
        );
        assert!(
            baseline[0].0.iter().any(|c| c.contains("HUGEINT")),
            "the fixture must actually exercise HUGEINT -- got {:?}",
            baseline[0].0
        );
    }

    #[tokio::test]
    async fn test_a_table_query_moves_in_and_a_shared_cte_is_materialized() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;
        let relations = ["rpt_region", "totals"];
        let baseline = run_and_fingerprint(&conn, &shared_chain_dag(), &relations).await;

        let mut dag = shared_chain_dag();
        let record = fuse(&conn, &mut dag).await;
        assert_eq!(detail(&record).materialized_ctes, 1);

        let sql = dag.nodes.get("dee_fused".to_string()).unwrap().query_text.clone();
        assert!(sql.contains("n_facts AS MATERIALIZED ("), "{sql}");
        assert!(sql.contains("n_stg AS ("), "a CTE read once is plain: {sql}");

        let totals = dag.nodes.get("totals".to_string()).unwrap();
        assert!(
            totals.query_text.starts_with("SELECT ") && totals.query_text.contains("WHERE kind = "),
            "a Table query becomes a projection of its kind: {}",
            totals.query_text
        );

        let fused = run_and_fingerprint(&conn, &dag, &relations).await;
        assert_eq!(baseline, fused);
    }

    #[tokio::test]
    async fn test_a_view_outside_the_rollup_is_still_read_where_it_was() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;
        let baseline = run_and_fingerprint(&conn, &shared_view_dag(), &["tbl_a", "tbl_b"]).await;

        let mut dag = shared_view_dag();
        fuse(&conn, &mut dag).await;
        let tbl_b = dag.nodes.get("tbl_b".to_string()).unwrap();
        assert!(tbl_b.depends_on.contains("dee_fused"), "{:?}", tbl_b.depends_on);
        assert!(
            tbl_b.depends_on.contains("lonely"),
            "`lonely` runs once and is not shared, so `tbl_b` keeps reading it: {:?}",
            tbl_b.depends_on
        );

        let fused = run_and_fingerprint(&conn, &dag, &["tbl_a", "tbl_b"]).await;
        assert_eq!(baseline, fused);
    }

    #[tokio::test]
    async fn test_the_rewrite_is_a_function_of_the_dag_alone() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;

        // The same logical DAG, built by inserting its nodes in two different
        // orders. `Graph` is a `HashMap`, so that is enough to change how it
        // iterates -- and the rewrite must not notice. dee's DAGs are
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
            make_dag(nodes)
        };

        async fn texts(conn: &Arc<DuckDBConnection>, mut dag: Dag) -> Vec<(String, String)> {
            fuse(conn, &mut dag).await;
            let mut texts: Vec<(String, String)> = dag
                .nodes
                .nodes()
                .map(|n| (n.id.clone(), n.query_text.clone()))
                .collect();
            texts.sort();
            texts
        }

        assert_eq!(
            texts(&conn, build(false)).await,
            texts(&conn, build(true)).await,
            "the rollup and every rewritten Table must be byte-identical"
        );
    }

    // ------------------------------------------------------------------
    // Which CTEs are materialized
    // ------------------------------------------------------------------

    #[test]
    fn test_an_override_is_the_exact_set() {
        let dag = shared_chain_dag();
        let default = NodeFusionPass::new(None).plan(&dag).unwrap();
        assert_eq!(
            default.materialized,
            BTreeSet::from(["facts".to_string()]),
            "the rule: two or more readers inside the rollup"
        );

        let over = NodeFusionPass::new(Some(vec!["stg".to_string()]))
            .plan(&dag)
            .unwrap();
        assert_eq!(
            over.materialized,
            BTreeSet::from(["stg".to_string()]),
            "naming only `stg` takes `facts` back off"
        );
    }

    #[test]
    fn test_an_override_matches_a_bare_name_against_a_qualified_id() {
        let list = vec!["orders".to_string()];
        assert!(override_matches(&list, "\"wh\".\"main\".\"orders\""));
        assert!(!override_matches(&list, "\"wh\".\"main\".\"other\""));
    }

    // ------------------------------------------------------------------
    // The DAGs that are left alone
    // ------------------------------------------------------------------

    async fn assert_not_fused(conn: &Arc<DuckDBConnection>, dag: &mut Dag, expect: &str) {
        let mut before: Vec<(String, String)> = dag
            .nodes
            .nodes()
            .map(|n| (n.id.clone(), n.query_text.clone()))
            .collect();
        let n_before = dag.nodes.num_nodes();

        let record = fuse(conn, dag).await;
        let detail = detail(&record);
        assert!(
            detail.outcome.contains(expect),
            "expected an outcome mentioning '{expect}', got '{}'",
            detail.outcome
        );
        assert_eq!(detail.tables_fused, 0);
        assert_eq!(record.changes_applied, 0);
        assert_eq!(dag.nodes.num_nodes(), n_before, "no node was added");
        let mut after: Vec<(String, String)> = dag
            .nodes
            .nodes()
            .map(|n| (n.id.clone(), n.query_text.clone()))
            .collect();
        before.sort();
        after.sort();
        assert_eq!(before, after, "an unfusable DAG must be left exactly as it was");
    }

    const NOTHING_SHARED: &str = "nothing for a rollup to share";

    #[tokio::test]
    async fn test_a_view_read_once_is_not_worth_sharing() {
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
        assert_not_fused(&conn, &mut dag, NOTHING_SHARED).await;
    }

    #[tokio::test]
    async fn test_a_table_feeding_a_table_through_views_shares_nothing() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;
        assert_not_fused(&conn, &mut layered_dag(), NOTHING_SHARED).await;
    }

    #[tokio::test]
    async fn test_sharing_only_a_temp_table_is_left_alone() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;
        // A TempTable is stored once already, and a materialization search put
        // it there on purpose.
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
        assert_not_fused(&conn, &mut dag, NOTHING_SHARED).await;
    }

    #[tokio::test]
    async fn test_fusing_an_already_fused_dag_finds_nothing_left() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;
        let mut dag = two_table_dag();
        fuse(&conn, &mut dag).await;

        // Every Table now reads the rollup, so no View runs more than once.
        assert_not_fused(&conn, &mut dag, NOTHING_SHARED).await;
    }

    #[tokio::test]
    async fn test_a_node_the_parser_cannot_read() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;

        // `extract('year' from d)` is DuckDB-valid and polyglot-sql cannot
        // parse it. Here it sits in a shared leaf View that names no other
        // shared model, so nothing needs rewriting and it is carried through
        // exactly as authored.
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
        fuse(&conn, &mut dag).await;
        let fused = dag.nodes.get("dee_fused".to_string()).unwrap();
        assert!(
            fused.query_text.contains("extract('year'"),
            "the body should be used as authored, not regenerated -- got:\n{}",
            fused.query_text
        );
        let engine = SimpleEngine::new(Arc::clone(&conn)).unwrap();
        engine.run(&dag).await.expect("and the rewritten DAG still runs");
        engine.cleanup(&dag).await.unwrap();

        // The same syntax in a consumer cannot be rewritten, and is left
        // reading the real View rather than substituted textually. The other
        // consumer still shares.
        let build = || {
            make_dag(vec![
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
            ])
        };
        let baseline = run_and_fingerprint(&conn, &build(), &["by_region", "totals"]).await;
        let mut dag = build();
        let record = fuse(&conn, &mut dag).await;
        let d = detail(&record);
        assert_eq!(d.untouched.len(), 1, "{:?}", d.untouched);
        assert_eq!(d.untouched[0].0, "by_region");
        let by_region = dag.nodes.get("by_region".to_string()).unwrap();
        assert_eq!(by_region.depends_on, HashSet::from(["stg".to_string()]));
        assert_eq!(
            dag.nodes.get("totals".to_string()).unwrap().depends_on,
            HashSet::from(["dee_fused".to_string()])
        );
        assert_eq!(
            baseline,
            run_and_fingerprint(&conn, &dag, &["by_region", "totals"]).await
        );
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

        fuse(&conn, &mut dag).await;
        assert!(
            dag.nodes.get("\"wh\".\"dee_fused\"".to_string()).is_some(),
            "the rollup inherits the schema prefix of the nodes reading it"
        );

        let engine = SimpleEngine::new(Arc::clone(&conn)).unwrap();
        engine.run(&dag).await.expect("the rewritten DAG runs");
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
    /// landing pads, which are stored builds, so a View they materialize no
    /// longer runs more than once and there is less for the rollup to share.
    fn adaptive_config() -> OptimizerConfig {
        OptimizerConfig::default()
            .with_all_disabled()
            .with_nodefusion_pass()
            .with_nodefusion_adaptive_materialize_ctes(true)
    }

    #[test]
    fn adaptive_refuses_an_override_that_leaves_it_nothing_to_decide() {
        // An override pins the exact set the search exists to find, so the
        // runs would be spent re-deriving a fixed answer.
        let config = adaptive_config()
            .with_nodefusion_materialize_ctes_override(Some(vec!["stg".into()]));
        assert!(
            config.validate().is_err(),
            "a search with nothing to decide must be refused"
        );
        // And again at the pass, which is the backstop for a config row stored
        // before the rule existed.
        let pass = NodeFusionPass::from_config(&config);
        let err = pass.check_config().expect_err("the pass must refuse too");
        assert!(matches!(err, OptimizerError::Config(_)), "got {err:?}");
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

    #[test]
    fn only_a_cte_with_more_than_one_reader_is_the_searchs_to_decide() {
        let plan = NodeFusionPass::new(None).plan(&shared_chain_dag()).unwrap();
        assert_eq!(
            plan.decidable(),
            vec!["facts".to_string()],
            "only `facts` is read twice inside the rollup"
        );
        // Everything else is read exactly once -- by one CTE or its own output
        // branch -- so it is computed once whichever way the hint goes. A
        // search that trialled it would be spending DAG runs on a coin flip.
        for id in ["stg", "by_region_v", "totals"] {
            let cte = plan.ctes.iter().find(|c| c.node_id == id).unwrap();
            assert_eq!(cte.readers, 1, "{id}");
            assert!(!cte.decidable, "{id} is read once and has nothing to decide");
        }
    }

    #[tokio::test]
    async fn the_empty_set_is_a_configuration_and_delivers_the_same_rows() {
        // The search may move the hint around; it may not change what the DAG
        // produces. And the empty set -- every CTE plain -- is a configuration
        // it can install, not an absence.
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;
        let relations = ["rpt_region", "totals"];
        let baseline = run_and_fingerprint(&conn, &shared_chain_dag(), &relations).await;

        let mut dag = shared_chain_dag();
        let empty: HashSet<String> = HashSet::new();
        NodeFusionPass::new(None)
            .rewrite_with(conn.as_ref(), &mut dag, &empty)
            .await
            .unwrap();
        let sql = dag.nodes.get("dee_fused".to_string()).unwrap().query_text.clone();
        assert!(
            !sql.contains("MATERIALIZED"),
            "the empty set means every CTE plain; got {sql}"
        );

        let fused = run_and_fingerprint(&conn, &dag, &relations).await;
        assert_eq!(
            baseline, fused,
            "demoting the shared CTE changes the plan, never the rows"
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

    #[test]
    fn a_legacy_config_row_with_the_retired_switches_still_decodes() {
        // They are no longer read, but a stored config that carries them must
        // not start failing to decode -- `OptimizerConfig` refuses unknown
        // fields.
        let raw = serde_json::json!({
            "run_nodefusion_pass": true,
            "nodefusion_materialize_ctes": true,
            "nodefusion_naive_materialize_ctes": true,
        });
        let mut merged = serde_json::to_value(OptimizerConfig::default()).unwrap();
        for (k, v) in raw.as_object().unwrap() {
            merged[k] = v.clone();
        }
        let config: OptimizerConfig = serde_json::from_value(merged).expect("the old row decodes");
        let pass = NodeFusionPass::from_config(&config);
        assert!(pass.materialize_override.is_none());
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

    /// A join View that four stored builds read, one of them through a View.
    ///
    ///   events ─┐
    ///           ├─► enriched (View) ──┬──► by_region  (Table)
    ///   customers┘                    ├──► by_channel (Table)
    ///                                 ├──► top_spend  (Table)
    ///                                 └──► enriched_us (View) ─► us_count (Table)
    ///
    /// `enriched` runs four times, and `enriched_us` reads it inside the
    /// rollup, so `by_region` and `by_channel` move their queries in and
    /// `enriched` ends up with several readers there -- the one CTE whose
    /// materialization is a real question. `top_spend` orders and limits, so
    /// it keeps its own query.
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
                "enriched_us",
                "SELECT event_id, value FROM enriched WHERE region = 'US'",
                MaterializeMode::View,
                &["enriched"],
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
            node(
                "us_count",
                "SELECT count(*) AS n, sum(value) AS total FROM enriched_us",
                MaterializeMode::Table,
                &["enriched_us"],
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

    const WIDE_RELATIONS: [&str; 4] = ["by_region", "by_channel", "top_spend", "us_count"];

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
        let expected = run_and_fingerprint(&conn, &wide_dag(), &WIDE_RELATIONS).await;
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
        let actual = run_and_fingerprint(&conn, &dag, &WIDE_RELATIONS).await;
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
                .with_nodefusion_pass(),
        );
        let mut dag = two_table_dag();
        let stores = MemoryStoreFactory::open().unwrap();
        let report = optimizer
            .run(&mut dag, "dag-rule", "pipeline", 1, &stores)
            .await
            .expect("the rewrite should succeed");

        assert_eq!(
            report.dag_runs_used, 0,
            "a rule-driven rollup decides everything from the DAG in front of it"
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
        let dag = wide_dag();

        let plan = NodeFusionPass::new(None).plan(&dag).unwrap();
        let decidable: HashSet<String> = plan.decidable().into_iter().collect();
        assert!(
            !decidable.is_empty(),
            "the fixture must have something the search could decide about"
        );
        let columns = rollup::resolve_kind_columns(conn.as_ref(), &dag, &plan.rollup)
            .await
            .unwrap();

        let explain_of = async |set: &HashSet<String>| {
            let materialized: BTreeSet<String> = set.iter().cloned().collect();
            let sql = rollup::emit(&dag, &plan.rollup, &columns, &materialized)
                .unwrap()
                .fused_sql;
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
            "DuckDB plans the decidable CTEs identically hinted or not, so the rollup's plan \
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
        let dag = wide_dag();

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
