use async_trait::async_trait;
use log::{debug, info, warn};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    collections::{HashMap, HashSet},
    fs,
    marker::PhantomData,
    sync::{Arc, Mutex},
};

use crate::{
    connectors::Connector,
    dag::MaterializeMode,
    executor::{ExecStats, Executor},
    plan::OpKey,
    opt::{
        Dag, Optimization, OptimizerError, OptimizerConfig,
        combo::{canonical_order, combinations, search_combos},
        common::{dialect_for_db, make_temp},
        dup::{SubtreeCostMethod, duplicate_cost_set},
        leafset::{PlanArena, ViewRegionRequest, attribute_chain, has_top_level_group_by},
        learned::LearnedCostModel,
        explain::{render_bar_row, render_card_grid, render_ranked_table},
        pushdown::PushdownPass,
        resume::node_signature,
        report::{HmpDetail, IterationStat, PassDetail, PassOutcome},
        step::{
            OptimizationType, RegisterContext, StepContext, StepOutcome, StepPhase,
        },
        store::{OptStore, Registration},
    },
};

/// How HMP turns a run's plans into a per-View cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
// snake_case rather than lowercase: `NodeTime` has to encode as `node_time`,
// the same spelling the CLI and the benchmark config use for it.
#[serde(rename_all = "snake_case")]
pub enum HmpCostMethod {
    /// Leaf-set matching under DAG-order containment: a View is attributed the
    /// region of a consumer's plan whose scanned base relations are still
    /// contained in the View's own. See [`crate::opt::leafset`].
    #[default]
    Leafset,
    /// Match operators between plans by `(name, estimated cardinality)`. The
    /// original method: cheaper, but that key is the entire notion of operator
    /// identity across two plans, so an estimate that shifts by one row between
    /// the View's EXPLAIN and the consumer's EXPLAIN ANALYZE fails to match at
    /// all, and every View whose plan happens to contain the key is charged the
    /// operator's full cost.
    Signature,
    /// Rank Views by their own measured node time. Also the automatic fallback
    /// when a run carried no plans at all.
    NodeTime,
    /// Price a View's own plan with seconds-per-byte constants fitted to the
    /// EXPLAIN ANALYZE plans of every `CREATE TABLE` the engine has run. See
    /// [`crate::opt::learned`].
    ///
    /// Unlike the other three this carries knowledge between runs: the
    /// constants are a property of the engine and the machine, so what one DAG
    /// learns prices every other DAG's Views.
    LearnedCost,
    /// Ask the engine what each consumer would do with the View inlined into it
    /// as a materialized CTE, and charge the View every copy but one. See
    /// [`crate::opt::dup`].
    ///
    /// The only method that measures the duplication rather than inferring it
    /// from the graph, and the only one that talks to the database: it costs
    /// one EXPLAIN per consumer per candidate. `--hmp-dup-cost-model` chooses
    /// what the resulting plan regions are priced with;
    /// `--hmp-downstream-cost` has no effect, because a duplicate cost is
    /// already what this reports.
    DupAttribution,
}

impl HmpCostMethod {
    pub fn as_str(&self) -> &'static str {
        match self {
            HmpCostMethod::Leafset => "leafset",
            HmpCostMethod::Signature => "signature",
            HmpCostMethod::NodeTime => "node_time",
            HmpCostMethod::LearnedCost => "learned_cost",
            HmpCostMethod::DupAttribution => "dup_attribution",
        }
    }
}

impl std::str::FromStr for HmpCostMethod {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "leafset" => Ok(HmpCostMethod::Leafset),
            "signature" => Ok(HmpCostMethod::Signature),
            "node_time" | "nodetime" => Ok(HmpCostMethod::NodeTime),
            "learned_cost" | "learnedcost" => Ok(HmpCostMethod::LearnedCost),
            "dup_attribution" | "dupattribution" | "dup" => Ok(HmpCostMethod::DupAttribution),
            other => Err(format!(
                "unknown hmp cost method '{other}'; expected leafset, signature, \
                 node_time, learned_cost or dup_attribution"
            )),
        }
    }
}

/// Where HMP is in its search, as persisted between steps.
///
/// The old pass held all of this on the stack of a single `run()`, because
/// the whole search happened inside one call. Now that a step ends when the
/// DAG runs and the next one may not happen for hours -- or in another
/// process, after a restart -- everything the search needs to pick up where it
/// left off has to survive in the metadata database. This struct is exactly
/// that: what a `Before` step reads to decide what to try next, and what an
/// `After` step writes once it knows how the trial went.
/// Every field defaults, and the struct as a whole is `#[serde(default)]`, so a
/// row written by an older build decodes rather than hard-failing: `load_state`
/// turns a decode error into `OptimizerError::Store`, which would strand a
/// search that was merely persisted by a previous version. Removing a field is
/// already tolerated -- serde ignores unknown keys -- but adding one is not
/// unless the struct carries this.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
struct HmpState {
    /// `"baseline"` -- waiting for the first measurement, which is of the DAG
    /// as it stands. `"searching"` -- working through candidates.
    /// `"converged"` -- finished; the winner has been promoted.
    phase: String,
    baseline_ms: i64,
    best_ms: i64,
    best_combo: Vec<String>,
    /// The ranked candidate Views the search will explore.
    working_set: Vec<String>,
    /// Each candidate View's score from the baseline run, which seeds the
    /// enumeration's priority queue.
    baseline_scores: HashMap<String, f64>,
    /// DAG executions this search has consumed, baseline included.
    runs_used: usize,
    iterations: Vec<IterationStat>,
    /// Signatures of trial DAGs already measured, so two combos that reduce to
    /// the same DAG are not paid for twice.
    tried_sigs: Vec<String>,
    /// The combinations the search will trial, priced before any of them was
    /// run and ordered by the duplicate computation each removes.
    candidates: Vec<CandidateCombo>,
    /// Position in `candidates`.
    cursor: usize,
    /// Whether `candidates` has been enumerated yet.
    ///
    /// Not the same question as `candidates.is_empty()`, and the difference
    /// matters: a state row persisted by a build that predates the candidate
    /// list decodes with an empty one, and without this flag a live search
    /// would read that as "exhausted" and promote on its next step.
    candidates_built: bool,

    /// The candidate a `Before` step rewrote the DAG into and an `After` step
    /// is expected to report on.
    in_flight: Option<InFlight>,
}

/// One priced combination of Views: what materializing all of it together is
/// estimated to remove.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct CandidateCombo {
    /// Members, consumer-most first -- the order they must be materialized in.
    combo: Vec<String>,
    /// Duplicate computation the combination removes, from the telescoping
    /// chain in [`crate::opt::combo`].
    cost: f64,
    /// A member could not be priced and was charged zero, so `cost` is a floor.
    #[serde(default)]
    partial: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct InFlight {
    combo: Vec<String>,
    sig: String,
}

impl HmpState {
    fn new() -> Self {
        Self::default()
    }
}

impl Default for HmpState {
    fn default() -> Self {
        Self {
            phase: "baseline".to_string(),
            baseline_ms: 0,
            best_ms: i64::MAX,
            best_combo: Vec::new(),
            working_set: Vec::new(),
            baseline_scores: HashMap::new(),
            runs_used: 0,
            iterations: Vec::new(),
            tried_sigs: Vec::new(),
            candidates: Vec::new(),
            cursor: 0,
            candidates_built: false,
            in_flight: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HMPPass<C, E>
where
    C: Connector + Send + 'static + Sync,
    E: Executor<C> + Send + Sync,
{
    conn: Arc<C>,
    engine: Arc<E>,
    /// Rank VIEW candidates by the total cost of duplicate computation they
    /// introduce downstream, instead of an estimated cost to run the VIEW
    /// itself.
    downstream_cost: bool,
    /// Max number of DAG re-runs to spend searching for materialization
    /// candidates. Each attempted materialization (successful or not) costs
    /// one run, in addition to the initial baseline run.
    max_runs: usize,
    /// Fraction (0, 1.0] of total operator CPU time used to build the
    /// `working_set` of candidate operators to materialize.
    top_cpu_time: f64,
    /// When set, log a table of operator rankings after the baseline run.
    /// `Some("")` logs the table only; `Some(path)` also writes it to `path`.
    show_operators: Option<String>,
    /// When set, log a table of node (View) rankings after the baseline run.
    /// `Some("")` logs the table only; `Some(path)` also writes it to `path`.
    show_nodes: Option<String>,
    /// When set, rank VIEW candidates by total CPU time divided by the
    /// View's estimated cardinality (from its EXPLAIN plan), instead of raw
    /// total CPU time.
    normalize_with_cardinality: bool,
    /// How a View's cost is read off the run's plans.
    cost_method: HmpCostMethod,
    /// What `DupAttribution` prices a plan region with. Unused by the other
    /// methods, which have no notion of a region.
    dup_cost_model: SubtreeCostMethod,
    /// Cancel a trial once it has overrun the incumbent and finish the run
    /// under the incumbent instead, rather than measuring every candidate to
    /// completion.
    resume_trials: bool,
    /// Fraction by which a trial may overrun the incumbent before it is cut
    /// short. Only meaningful when `resume_trials` is set.
    budget_eps: f64,
    /// How much of a cancelled trial the resume may keep.
    reuse_policy: crate::opt::resume::ReusePolicy,
    /// Combinations the search may price before it starts spending DAG runs.
    ///
    /// Distinct from `max_runs`, which bounds executions: this one bounds how
    /// much of the combination space is *costed*, which is EXPLAIN-only.
    search_budget: usize,
    /// Run the PushdownPass before evaluating each candidate materialization
    /// combination, for more accurate cost measurements.
    use_pushdown: bool,
    /// Capture each iteration's CPU/memory/disk timeseries (already sampled
    /// by the profiled engine used for measurement) into its `IterationStat`.
    profile_iterations: bool,
    /// Which side of an execution to step on. `Both` by author's default: the
    /// search proposes before a run and learns after it, and either half alone
    /// is only part of a search.
    step_phase: StepPhase,
    /// Ranking tables from the baseline run, retained for `explain`. They
    /// describe the run the search started from, so they are computed once and
    /// kept rather than recomputed per step.
    operator_rows: Vec<OperatorRankingRow>,
    node_rows: Vec<NodeRankingRow>,
    /// Data collected during the last `step()`, used by `explain`.
    explain_data: Option<HMPExplainData>,
    /// The seconds-per-byte constants `LearnedCost` prices plans with.
    ///
    /// Behind a lock because `ranking_for` takes `&self` --- it is called from
    /// the offline benchmark as a pure function of a run --- while every run it
    /// sees teaches it something more. Loaded from and written back to the
    /// metadata store around each step, which is what makes the learning
    /// survive the process.
    learned: Arc<Mutex<LearnedCostModel>>,
    _phantom: PhantomData<E>,
}

/// Everything `Explain::explain` needs to describe what the last `run()`
/// did and why, retained from otherwise-local data computed during `run()`.
#[derive(Debug, Clone)]
struct HMPExplainData {
    baseline_ms: i64,
    final_ms: i64,
    runs_used: usize,
    max_runs: usize,
    top_cpu_time: f64,
    normalize_with_cardinality: bool,
    operator_rows: Vec<OperatorRankingRow>,
    node_rows: Vec<NodeRankingRow>,
    working_set: Vec<String>,
    best_combo: Vec<String>,
    iterations: Vec<IterationStat>,
    search_budget: usize,
    candidates_costed: usize,
}

/// One row of the `--hmp-show-operators` table.
#[derive(Serialize, Debug, Clone)]
struct OperatorRankingRow {
    rank: usize,
    operator: String,
    avg_runtime_s: f64,
    table_occurrences: usize,
    traced_views: Vec<String>,
}

/// One row of the `--hmp-show-nodes` ranking table: a View (out-degree > 1)
/// and the aggregate CPU time of every operator traced back to it.
///
/// Public so that a costing method can be evaluated offline against measured
/// ground truth -- see `ranking_for`.
#[derive(Serialize, Debug, Clone)]
pub struct NodeRankingRow {
    pub rank: usize,
    pub node: String,
    pub total_cpu_time_s: f64,
    /// Estimated cardinality of the View's own EXPLAIN plan, when available.
    pub cardinality: Option<f64>,
    /// The value nodes are ranked by: `total_cpu_time_s`, or (when
    /// `--hmp-normalize-with-cardinality` is set) `total_cpu_time_s` divided
    /// by `cardinality`.
    pub ranking_score: f64,
    /// Leaf-set matching only: the base relations this View reads, and the
    /// consumer plan operators its region was matched to.
    ///
    /// Empty under the other costing methods, which have no notion of either.
    /// They are the only way to tell a good attribution from a lucky one, which
    /// is why they reach the explain report rather than staying in a log line.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub leaves: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub matched: Vec<String>,
}

impl<C, E> HMPPass<C, E>
where
    C: Connector + Send + 'static + Sync,
    E: Executor<C> + Send + Sync,
{
    pub fn new(
        conn: Arc<C>,
        engine: Arc<E>,
        downstream_cost: bool,
        max_runs: usize,
        top_cpu_time: f64,
        show_operators: Option<String>,
        show_nodes: Option<String>,
        normalize_with_cardinality: bool,
        cost_method: HmpCostMethod,
        dup_cost_model: SubtreeCostMethod,
        use_pushdown: bool,
        search_budget: usize,
        profile_iterations: bool,
        resume_trials: bool,
        budget_eps: f64,
        reuse_policy: crate::opt::resume::ReusePolicy,
    ) -> Self {
        Self {
            conn,
            engine,
            downstream_cost,
            max_runs: max_runs.max(1),
            show_operators,
            show_nodes,
            normalize_with_cardinality,
            cost_method,
            dup_cost_model,
            use_pushdown,
            search_budget: search_budget.max(1),
            profile_iterations,
            resume_trials,
            reuse_policy,
            // Zero is the default and a meaningful setting -- stop the trial
            // the moment it can no longer win -- so only a nonsensical negative
            // falls back.
            budget_eps: if budget_eps >= 0.0 {
                budget_eps
            } else {
                crate::opt::common::DEFAULT_BUDGET_EPS
            },
            top_cpu_time: if top_cpu_time > 0.0 && top_cpu_time <= 1.0 {
                top_cpu_time
            } else {
                0.5
            },
            step_phase: StepPhase::Both,
            operator_rows: Vec::new(),
            node_rows: Vec::new(),
            explain_data: None,
            learned: Arc::new(Mutex::new(LearnedCostModel::new())),
            _phantom: PhantomData,
        }
    }

    /// Build a map from each operator found in the EXPLAIN ANALYZE plans of
    /// currently materialized (Table) nodes to its average runtime across
    /// occurrences, along with the occurrence count.
    fn operator_stats(&self, dag: &Dag, exec_stats: &ExecStats) -> HashMap<OpKey, (f64, usize)> {
        let mut timing_map: HashMap<OpKey, f64> = HashMap::new();
        let mut occurrence_map: HashMap<OpKey, usize> = HashMap::new();

        let mut materialized_node_count = 0;
        for node in dag.nodes.nodes() {
            if matches!(node.materialize, MaterializeMode::Table) {
                materialized_node_count += 1;
                if let Some(node_stat) = exec_stats.node_stats.get(&node.id)
                    && let Some(plan_str) = &node_stat.plan
                    && let Some(plans) = self.conn.parse_plan(plan_str)
                {
                    for plan in &plans {
                        plan.collect_operator_stats(&mut timing_map, &mut occurrence_map);
                    }
                }
            }
        }
        debug!("Analyzed {} materialized nodes", materialized_node_count);

        timing_map
            .into_iter()
            .map(|(sig, total)| {
                let occurrences = occurrence_map.get(&sig).cloned().unwrap_or(0);
                let avg = if occurrences > 0 {
                    total / occurrences as f64
                } else {
                    0.0
                };
                (sig, (avg, occurrences))
            })
            .collect()
    }

    /// Build the `--hmp-show-operators` table: operator key, its average
    /// runtime across occurrences, number of materialized Table plans the
    /// operator appears in, and every View whose EXPLAIN plan contains the
    /// operator. Rows are sorted by operator name for stable output.
    fn build_operator_table(
        conn: &C,
        dag: &Dag,
        exec_stats: &ExecStats,
        op_stats: &HashMap<OpKey, (f64, usize)>,
    ) -> Vec<OperatorRankingRow> {
        let mut entries: Vec<_> = op_stats.iter().collect();
        entries.sort_by(|a, b| a.0.name.cmp(&b.0.name).then(a.0.cardinality.cmp(&b.0.cardinality)));

        entries
            .into_iter()
            .enumerate()
            .map(|(i, (op_key, (avg_runtime, occurrences)))| OperatorRankingRow {
                rank: i + 1,
                operator: format!("{}(cardinality={})", op_key.name, op_key.cardinality),
                avg_runtime_s: *avg_runtime,
                table_occurrences: *occurrences,
                traced_views: Self::find_traced_views(conn, dag, op_key, exec_stats),
            })
            .collect()
    }

    /// Render the operator ranking table as aligned plain text.
    fn format_operator_table(rows: &[OperatorRankingRow]) -> String {
        let headers = [
            "Rank",
            "Operator",
            "Avg Runtime (s)",
            "Table Occurrences",
            "Traced View(s)",
        ];
        let rows_str: Vec<[String; 5]> = rows
            .iter()
            .map(|r| {
                [
                    r.rank.to_string(),
                    r.operator.clone(),
                    format!("{:.4}", r.avg_runtime_s),
                    r.table_occurrences.to_string(),
                    if r.traced_views.is_empty() {
                        "-".to_string()
                    } else {
                        r.traced_views.join(", ")
                    },
                ]
            })
            .collect();

        let mut widths: [usize; 5] = std::array::from_fn(|i| headers[i].len());
        for row in &rows_str {
            for (i, cell) in row.iter().enumerate() {
                widths[i] = widths[i].max(cell.len());
            }
        }

        let mut out = String::new();
        for (i, h) in headers.iter().enumerate() {
            out.push_str(&format!("{:<width$}  ", h, width = widths[i]));
        }
        out.push('\n');
        for (i, _) in headers.iter().enumerate() {
            out.push_str(&format!("{:-<width$}  ", "", width = widths[i]));
        }
        for row in &rows_str {
            out.push('\n');
            for (i, cell) in row.iter().enumerate() {
                out.push_str(&format!("{:<width$}  ", cell, width = widths[i]));
            }
        }
        out
    }

    /// Log the operator ranking table and, if `show_operators` carries a
    /// non-empty path, write it there too.
    fn log_operator_table(&self, dag: &Dag, exec_stats: &ExecStats, op_stats: &HashMap<OpKey, (f64, usize)>) {
        let Some(path) = &self.show_operators else {
            return;
        };

        let rows = Self::build_operator_table(self.conn.as_ref(), dag, exec_stats, op_stats);
        let table = Self::format_operator_table(&rows);
        info!("HMPPass operator rankings:\n{}", table);

        if !path.is_empty()
            && let Err(e) = fs::write(path, &table)
        {
            warn!("failed to write operator rankings to '{}': {}", path, e);
        }
    }

    /// Find every View node whose EXPLAIN plan contains the given operator,
    /// i.e. every View the operator can be traced back to (not just the one
    /// `find_materialization_candidate` would pick to materialize).
    fn find_traced_views(
        conn: &C,
        dag: &Dag,
        op_key: &OpKey,
        exec_stats: &ExecStats,
    ) -> Vec<String> {
        let mut views = Vec::new();
        for node in dag.nodes.nodes() {
            if !matches!(node.materialize, MaterializeMode::View) {
                continue;
            }
            let Some(node_stat) = exec_stats.node_stats.get(&node.id) else {
                continue;
            };
            let Some(plan_str) = &node_stat.plan else {
                continue;
            };
            if let Some(plans) = conn.parse_plan(plan_str)
                && plans.iter().any(|p| p.contains(op_key))
            {
                views.push(node.id.clone());
            }
        }
        views
    }

    /// Estimated cardinality of a View's own EXPLAIN plan, taken from the
    /// root operator of its (already-collected) query plan.
    fn view_cardinality(conn: &C, exec_stats: &ExecStats, view_id: &str) -> Option<f64> {
        let node_stat = exec_stats.node_stats.get(view_id)?;
        let plan_str = node_stat.plan.as_ref()?;
        let plans = conn.parse_plan(plan_str)?;
        // The root operator's estimate, matching the pre-refactor behaviour.
        plans.first()?.estimated_cardinality
    }

    /// Sum, for every View that is a branch point (out-degree > 1 and more
    /// than one downstream path to a TABLE/TEMP_TABLE node -- the only kind
    /// of View that materializing can actually deduplicate work for), the
    /// average runtime of every operator that traces back to it via
    /// `find_traced_views` (the same mapping the `--hmp-show-operators`
    /// table uses). This approximates the cost of running the View once,
    /// not the cost of the duplicate work it causes downstream.
    fn aggregate_cpu_time_avg(
        conn: &C,
        dag: &Dag,
        exec_stats: &ExecStats,
        op_stats: &HashMap<OpKey, (f64, usize)>,
    ) -> HashMap<String, f64> {
        let mut aggregate_cpu_time: HashMap<String, f64> = HashMap::new();
        for (op_key, (avg_runtime, _)) in op_stats {
            for view in Self::find_traced_views(conn, dag, op_key, exec_stats) {
                if dag.nodes.out_degree(&view) > 1 && dag.nodes.paths_to_sinks(&view) > 1 {
                    *aggregate_cpu_time.entry(view).or_insert(0.0) += avg_runtime;
                }
            }
        }
        aggregate_cpu_time
    }

    /// For `--hmp-downstream-cost`: rather than averaging an operator's cost
    /// across its occurrences, walk every occurrence of every operator in
    /// every materialized TABLE's EXPLAIN ANALYZE plan, and add its actual
    /// CPU cost to every branch-point View whose own EXPLAIN plan contains
    /// that operator. This totals the real cost of the duplicate
    /// computation a View introduces downstream, instead of estimating the
    /// cost of running the View itself.
    fn aggregate_downstream_cost(
        conn: &C,
        dag: &Dag,
        exec_stats: &ExecStats,
    ) -> HashMap<String, f64> {
        let mut aggregate_cpu_time: HashMap<String, f64> = HashMap::new();
        for node in dag.nodes.nodes() {
            if !matches!(node.materialize, MaterializeMode::Table) {
                continue;
            }
            let Some(node_stat) = exec_stats.node_stats.get(&node.id) else {
                continue;
            };
            let Some(plan_str) = &node_stat.plan else {
                continue;
            };
            let Some(plans) = conn.parse_plan(plan_str) else {
                continue;
            };

            let mut operators = Vec::new();
            for plan in &plans {
                plan.collect_operators(&mut operators);
            }
            for (op_key, cpu_cost) in operators {
                for view in Self::find_traced_views(conn, dag, &op_key, exec_stats) {
                    if dag.nodes.out_degree(&view) > 1 && dag.nodes.paths_to_sinks(&view) > 1 {
                        *aggregate_cpu_time.entry(view).or_insert(0.0) += cpu_cost;
                    }
                }
            }
        }
        aggregate_cpu_time
    }

    /// Build the `--hmp-show-nodes` ranking table from a per-View aggregate
    /// CPU time map (see `aggregate_cpu_time_avg` / `aggregate_downstream_cost`).
    /// Sorted by `ranking_score`, descending -- this is also the order
    /// `run()` searches down when picking which node to try materializing.
    /// `ranking_score` is `total_cpu_time_s`, or (when
    /// `normalize_with_cardinality` is set) `total_cpu_time_s` divided by the
    /// View's estimated cardinality, from its EXPLAIN plan.
    fn build_node_table(
        conn: &C,
        exec_stats: &ExecStats,
        aggregate_cpu_time: HashMap<String, f64>,
        normalize_with_cardinality: bool,
    ) -> Vec<NodeRankingRow> {
        let mut rows: Vec<(String, f64, Option<f64>, f64)> = aggregate_cpu_time
            .into_iter()
            .map(|(node, total_cpu_time_s)| {
                let cardinality = Self::view_cardinality(conn, exec_stats, &node);
                let ranking_score = match (normalize_with_cardinality, cardinality) {
                    (true, Some(c)) if c > 0.0 => total_cpu_time_s / c,
                    _ => total_cpu_time_s,
                };
                (node, total_cpu_time_s, cardinality, ranking_score)
            })
            .collect();
        rows.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));

        rows.into_iter()
            .enumerate()
            .map(|(i, (node, total_cpu_time_s, cardinality, ranking_score))| NodeRankingRow {
                rank: i + 1,
                node,
                total_cpu_time_s,
                cardinality,
                ranking_score,
                leaves: Vec::new(),
                matched: Vec::new(),
            })
            .collect()
    }

    /// Render the node ranking table as aligned plain text.
    fn format_node_table(rows: &[NodeRankingRow]) -> String {
        let headers = ["Rank", "Node", "Total CPU Time (s)", "Cardinality", "Ranking Score"];
        let rows_str: Vec<[String; 5]> = rows
            .iter()
            .map(|r| {
                [
                    r.rank.to_string(),
                    r.node.clone(),
                    format!("{:.4}", r.total_cpu_time_s),
                    r.cardinality
                        .map(|c| format!("{:.0}", c))
                        .unwrap_or_else(|| "-".to_string()),
                    format!("{:.4}", r.ranking_score),
                ]
            })
            .collect();

        let mut widths: [usize; 5] = std::array::from_fn(|i| headers[i].len());
        for row in &rows_str {
            for (i, cell) in row.iter().enumerate() {
                widths[i] = widths[i].max(cell.len());
            }
        }

        let mut out = String::new();
        for (i, h) in headers.iter().enumerate() {
            out.push_str(&format!("{:<width$}  ", h, width = widths[i]));
        }
        out.push('\n');
        for (i, _) in headers.iter().enumerate() {
            out.push_str(&format!("{:-<width$}  ", "", width = widths[i]));
        }
        for row in &rows_str {
            out.push('\n');
            for (i, cell) in row.iter().enumerate() {
                out.push_str(&format!("{:<width$}  ", cell, width = widths[i]));
            }
        }
        out
    }

    /// Log the node ranking table and, if `show_nodes` carries a non-empty
    /// path, write it there too.
    fn log_node_table(&self, node_ranking: &[NodeRankingRow]) {
        let Some(path) = &self.show_nodes else {
            return;
        };

        let table = Self::format_node_table(node_ranking);
        info!("HMPPass node rankings:\n{}", table);

        if !path.is_empty()
            && let Err(e) = fs::write(path, &table)
        {
            warn!("failed to write node rankings to '{}': {}", path, e);
        }
    }
}

/// The most of one consumer's measured time a single inlined View may be
/// charged. A consumer that does work of its own always keeps some.
const MAX_SHARE: f64 = 0.95;

/// The middle observation of `values`, or `None` if there are none.
fn median(values: &[u64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let mid = sorted.len() / 2;
    Some(if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] as f64 + sorted[mid] as f64) / 2.0
    } else {
        sorted[mid] as f64
    })
}

/// Canonical string signature of a DAG's structure. Used to detect when two
/// different materialization combinations produce an equivalent DAG (e.g.
/// after `make_temp`'s landing-pad insertion / view inlining), so we can
/// avoid re-running a trial we've effectively already tried.
fn dag_signature(dag: &Dag) -> String {
    let mut node_sigs: Vec<String> = dag.nodes.nodes().map(node_signature).collect();
    node_sigs.sort_unstable();
    node_sigs.join("|")
}
// ---------------------------------------------------------------------------
// The step interface
//
// HMP's search used to be a loop inside one call: measure a baseline, rank the
// views, then try candidate after candidate, running the DAG itself for each.
// Under the server that loop is turned inside out. The server runs the DAG --
// on a schedule, from a trigger, out of the queue -- and HMP gets a turn on
// either side of each execution: `Before` to rewrite the DAG into the next
// candidate, `After` to learn what that candidate cost. The search is the same
// search; what changed is that its iterations are the DAG's own runs, so a
// pipeline that runs nightly optimizes itself nightly instead of paying for a
// private burst of runs up front.
// ---------------------------------------------------------------------------

const STATE_TABLE: &str = "opt_hmp_state";
const TRIALS_TABLE: &str = "opt_hmp_trials";
/// The `LearnedCost` method's seconds-per-byte constants.
///
/// One row for the whole engine rather than one per DAG: a hash join's cost per
/// output byte is a property of the engine and the machine, not of the pipeline
/// that happened to contain it, so what one DAG measures should price the next
/// one's Views on its very first run.
const LEARNED_TABLE: &str = "opt_hmp_learned_cost";

impl<C, E> HMPPass<C, E>
where
    C: Connector + Send + 'static + Sync,
    E: Executor<C> + Send + Sync,
{
    /// Build from an [`OptimizerConfig`], the form the registry and the server
    /// carry settings in.
    pub fn from_config(conn: Arc<C>, engine: Arc<E>, config: &OptimizerConfig) -> Self {
        Self::new(
            conn,
            engine,
            config.hmp_downstream_cost,
            config.hmp_max_runs,
            config.hmp_top_cpu_time,
            config.hmp_show_operators.clone(),
            config.hmp_show_nodes.clone(),
            config.hmp_normalize_with_cardinality,
            config.hmp_cost_method,
            config.hmp_dup_cost_model,
            config.hmp_use_pushdown,
            config.hmp_search_budget,
            config.profile_iterations,
            config.trial_resume,
            config.trial_budget_eps,
            config.trial_reuse,
        )
    }

    async fn load_state(
        &self,
        store: &dyn OptStore,
        dag_id: &str,
    ) -> Result<Option<HmpState>, OptimizerError> {
        let rows = match store
            .query(
                &format!("SELECT state FROM {STATE_TABLE} WHERE dag_id = ?"),
                &[json!(dag_id)],
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
        state: &HmpState,
    ) -> Result<(), OptimizerError> {
        let encoded = serde_json::to_string(state)
            .map_err(|e| OptimizerError::Store(crate::opt::OptStoreError::Decode(e.to_string())))?;
        // Delete-then-insert rather than an upsert: the state is one row per
        // DAG and DuckDB's ON CONFLICT needs a constraint the pass would then
        // have to keep in step with this statement.
        store
            .execute(
                &format!("DELETE FROM {STATE_TABLE} WHERE dag_id = ?"),
                &[json!(dag_id)],
            )
            .await?;
        store
            .execute(
                &format!("INSERT INTO {STATE_TABLE} (dag_id, state, updated_at) VALUES (?, ?, now())"),
                &[json!(dag_id), json!(encoded)],
            )
            .await?;
        Ok(())
    }

    /// The seconds-per-byte constants recorded by every run so far.
    ///
    /// A missing table is an empty model, not an error: the pass is asked to
    /// rank before it has ever been registered in tests and benchmarks, and a
    /// model that has learned nothing is exactly what it should have then.
    async fn load_learned(&self, store: &dyn OptStore) -> LearnedCostModel {
        let rows = match store
            .query(&format!("SELECT model FROM {LEARNED_TABLE}"), &[])
            .await
        {
            Ok(rows) => rows,
            Err(e) => {
                if !crate::opt::store::is_missing_table(&e) {
                    warn!("HMPPass: could not read the learned cost model: {e}");
                }
                return LearnedCostModel::new();
            }
        };
        rows.first()
            .and_then(|r| r.get("model"))
            .and_then(|v| v.as_str())
            .and_then(|raw| match serde_json::from_str(raw) {
                Ok(model) => Some(model),
                Err(e) => {
                    warn!("HMPPass: discarding an undecodable learned cost model: {e}");
                    None
                }
            })
            .unwrap_or_default()
    }

    /// Write the constants back, replacing the single stored row.
    ///
    /// The whole model goes out each time rather than a delta, because it is
    /// the merge of what was read and what this run observed --- writing a delta
    /// would double-count everything already in the row.
    async fn save_learned(
        &self,
        store: &dyn OptStore,
        model: &LearnedCostModel,
    ) -> Result<(), OptimizerError> {
        let encoded = serde_json::to_string(model)
            .map_err(|e| OptimizerError::Store(crate::opt::OptStoreError::Decode(e.to_string())))?;
        store
            .execute(&format!("DELETE FROM {LEARNED_TABLE}"), &[])
            .await?;
        store
            .execute(
                &format!("INSERT INTO {LEARNED_TABLE} (model, updated_at) VALUES (?, now())"),
                &[json!(encoded)],
            )
            .await?;
        Ok(())
    }

    /// Whether this configuration prices anything with the seconds-per-byte
    /// constants, and so has a reason to read and write them.
    ///
    /// `LearnedCost` prices Views with them directly; `DupAttribution` prices
    /// plan regions with them when that is the region cost model it was given.
    /// Nothing else touches them, and every other DAG registered on this store
    /// pays for the query if they do.
    fn uses_learned_constants(&self) -> bool {
        match self.cost_method {
            HmpCostMethod::LearnedCost => true,
            HmpCostMethod::DupAttribution => {
                self.dup_cost_model == SubtreeCostMethod::LearnedCost
            }
            _ => false,
        }
    }

    /// Fold what earlier runs learned into this pass's in-memory model, so the
    /// ranking about to be computed is priced by everything known so far.
    async fn adopt_learned(&self, store: &dyn OptStore) {
        if !self.uses_learned_constants() {
            return;
        }
        let stored = self.load_learned(store).await;
        if let Ok(mut model) = self.learned.lock() {
            model.merge(&stored);
        }
    }

    /// Persist what the ranking just learned, so the next run starts from it.
    async fn publish_learned(&self, store: &dyn OptStore) {
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
        if let Err(e) = self.save_learned(store, &snapshot).await {
            warn!("HMPPass: could not persist the learned cost model: {e}");
        }
        // The stored row is now everything this pass knows, so keeping the
        // same observations in memory too would double them into the next
        // write. Start empty; the next `adopt_learned` reads them back.
        if let Ok(mut model) = self.learned.lock() {
            *model = LearnedCostModel::new();
        }
    }

    /// Rank candidate Views by leaf-set matching against every persisted
    /// consumer's EXPLAIN ANALYZE plan.
    ///
    /// Everything is denominated in the seconds a consumer's node actually
    /// took. Plan times are used only as *ratios within one plan*, to split a
    /// consumer's measured time among the Views inlined into it -- which is
    /// what keeps this indifferent to DuckDB reporting CPU time and Postgres
    /// wall time.
    ///
    /// Returns `None` when no plan in the run named a single relation. That is
    /// not "this DAG has no duplication", it is "this method cannot see" -- a
    /// plan format that does not carry relation names, or a run recorded before
    /// they were collected -- and the caller falls back rather than reporting a
    /// ranking of nothing.
    fn ranking_leafset(&self, dag: &Dag, stats: &ExecStats) -> Option<Vec<NodeRankingRow>> {
        let dialect = dialect_for_db(&dag.db);
        let source_names: Vec<String> = dag.sources.iter().map(|s| s.name.clone()).collect();
        let heights = dag.nodes.heights();

        // Leaf sets and the GROUP BY hint, computed once per View rather than
        // once per consumer.
        let views: Vec<&crate::dag::TransformNode> = dag
            .nodes
            .nodes()
            .filter(|n| n.materialize == MaterializeMode::View)
            .collect();
        let leaf_sets: HashMap<String, std::collections::BTreeSet<String>> = views
            .iter()
            .map(|n| (n.id.clone(), dag.nodes.leaf_sources(&n.id, &source_names)))
            .collect();
        let collapses: HashMap<String, bool> = views
            .iter()
            .map(|n| (n.id.clone(), has_top_level_group_by(&n.query_text, dialect)))
            .collect();
        // Which Views depend on each View. Only the already-placed ones
        // constrain a search, but the relation itself is a property of the DAG.
        let view_consumers: HashMap<String, HashSet<String>> = views
            .iter()
            .map(|n| {
                let consumers = dag
                    .nodes
                    .reachable(&n.id, |c| c.materialize == MaterializeMode::View)
                    .into_iter()
                    .collect();
                (n.id.clone(), consumers)
            })
            .collect();

        let mut saw_a_relation = false;
        // share[view][consumer] x the consumer's measured seconds.
        let mut attributed: HashMap<String, Vec<f64>> = HashMap::new();
        let mut widths: HashMap<String, Vec<u64>> = HashMap::new();
        let mut matched: HashMap<String, Vec<String>> = HashMap::new();

        for consumer in dag.nodes.nodes() {
            if !matches!(
                consumer.materialize,
                MaterializeMode::Table | MaterializeMode::TempTable
            ) {
                continue;
            }
            let Some(node_stat) = stats.node_stats.get(&consumer.id) else {
                continue;
            };
            let Some(plan_str) = &node_stat.plan else {
                continue;
            };
            let Some(plans) = self.conn.parse_plan(plan_str) else {
                continue;
            };
            let arena = PlanArena::build(&plans);
            saw_a_relation |= arena.nodes.iter().any(|n| !n.leaves.is_empty());
            let total = arena.total_time();
            if total <= 0.0 {
                continue;
            }
            let measured_s = node_stat.duration.num_milliseconds() as f64 / 1000.0;

            // The Views inlined into this consumer, consumer-most first: a View
            // can only be placed inside the region of a View that depends on it.
            let mut inlined: Vec<&crate::dag::TransformNode> = views
                .iter()
                .copied()
                .filter(|v| dag.nodes.frontier_materializes(&v.id).contains(&consumer.id))
                .collect();
            inlined.sort_by_key(|v| (heights.get(&v.id).copied().unwrap_or(0), v.id.clone()));

            let requests: Vec<ViewRegionRequest> = inlined
                .iter()
                .map(|v| ViewRegionRequest {
                    id: v.id.clone(),
                    leaves: leaf_sets.get(&v.id).cloned().unwrap_or_default(),
                    consumers: view_consumers.get(&v.id).cloned().unwrap_or_default(),
                    prefers_aggregate: collapses.get(&v.id).copied().unwrap_or(false),
                })
                .collect();

            for (view, attribution) in attribute_chain(&arena, &requests) {
                // Capped: a View is never charged the whole of a consumer that
                // also does work of its own, and an uncapped share turns one bad
                // match into a candidate that dominates the ranking.
                let share = (attribution.secs / total).min(MAX_SHARE);
                attributed
                    .entry(view.clone())
                    .or_default()
                    .push(share * measured_s);
                matched.entry(view.clone()).or_default().push(format!(
                    "{} in {} ({:.0}%)",
                    arena.nodes[attribution.node].operator,
                    consumer.id,
                    share * 100.0
                ));
                if let Some(rows) = attribution.cardinality {
                    widths.entry(view).or_default().push(rows);
                }
            }
        }

        if !saw_a_relation {
            return None;
        }

        let mut rows: Vec<NodeRankingRow> = attributed
            .into_iter()
            // Only a branch point can have its work deduplicated by being built
            // once, which is the same gate the other methods apply.
            .filter(|(view, _)| {
                dag.nodes.out_degree(view) > 1 && dag.nodes.paths_to_sinks(view) > 1
            })
            .map(|(view, mut per_consumer)| {
                per_consumer.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let sum: f64 = per_consumer.iter().sum();
                // The largest attribution, because each consumer runs a
                // possibly-differently-optimized copy and the largest is the one
                // least likely to have had work pushed out of it.
                let compute_secs = per_consumer.last().copied().unwrap_or(0.0);
                // What would simply vanish if the View were built once -- the
                // quantity the whole hypothesis is about.
                let duplicated_secs = sum - compute_secs;

                let total_cpu_time_s = if self.downstream_cost {
                    duplicated_secs
                } else {
                    compute_secs
                };
                // The median rather than the extreme: one View matches
                // differently in different consumers, because the optimizer
                // fuses adjacent Views differently depending on what else is in
                // the query, and a single bad match should not set the View's
                // estimated width.
                let cardinality = widths.get(&view).and_then(|w| median(w));
                let ranking_score = match (self.normalize_with_cardinality, cardinality) {
                    (true, Some(c)) if c > 0.0 => total_cpu_time_s / c,
                    _ => total_cpu_time_s,
                };
                let mut regions = matched.remove(&view).unwrap_or_default();
                regions.sort();
                NodeRankingRow {
                    rank: 0,
                    leaves: leaf_sets
                        .get(&view)
                        .map(|l| l.iter().cloned().collect())
                        .unwrap_or_default(),
                    matched: regions,
                    node: view,
                    total_cpu_time_s,
                    cardinality,
                    ranking_score,
                }
            })
            .filter(|r| r.ranking_score > 0.0)
            .collect();

        rows.sort_by(|a, b| {
            b.ranking_score
                .partial_cmp(&a.ranking_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.node.cmp(&b.node))
        });
        for (i, row) in rows.iter_mut().enumerate() {
            row.rank = i + 1;
        }
        Some(rows)
    }

    /// Fold everything this run's executed plans reveal into the learned cost
    /// model.
    ///
    /// Only TABLE/TEMP_TABLE nodes: they are the ones whose plan is an EXPLAIN
    /// ANALYZE with real timings behind it, and a constant fitted to an
    /// estimate is not fitted to anything.
    ///
    /// Separate from the ranking that uses it because two methods learn from a
    /// run --- `LearnedCost` prices Views with the constants directly, and
    /// `DupAttribution` prices plan regions with them --- and a method that
    /// forgot to learn first would silently price everything at the previous
    /// run's constants.
    fn learn_from(&self, dag: &Dag, stats: &ExecStats) {
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
                && let Some(plans) = self.conn.parse_plan(plan_str)
            {
                model.observe(&plans);
            }
        }
    }

    /// Rank candidate Views by pricing each View's own plan with
    /// seconds-per-byte constants fitted to the run's executed plans.
    ///
    /// Two halves. First it *learns*: every `CREATE TABLE` in this run came
    /// back with an EXPLAIN ANALYZE plan, and every operator in one of those
    /// carries a timing, a row count and a tuple width, which is a
    /// seconds-per-byte observation for that operator's type. Those fold into
    /// whatever the model already knew. Then it *prices*: a candidate View's
    /// own plan, which was never executed and has no timings at all, is costed
    /// operator by operator as output bytes times the constant for its type.
    ///
    /// That second half is the reason this method exists. Leaf-set matching can
    /// only price a View the DAG actually inlined somewhere measurable this
    /// run; a constant fitted to bytes prices any plan the engine will show,
    /// including one for a View whose consumers all went a different way.
    ///
    /// Returns `None` when nothing has been learned yet, or when no candidate
    /// could be priced --- the caller falls back rather than reporting a ranking
    /// of nothing.
    fn ranking_learned(&self, dag: &Dag, stats: &ExecStats) -> Option<Vec<NodeRankingRow>> {
        self.learn_from(dag, stats);
        let model = self.learned.lock().ok()?;
        if model.is_empty() {
            return None;
        }
        // The constants themselves, dearest first. A single outlier here
        // explains an otherwise inexplicable ranking, and reading it off a plan
        // by hand is not practical.
        if log::log_enabled!(log::Level::Debug) {
            let mut consts: Vec<(&String, f64, f64, f64, u64)> = model
                .operators()
                .filter_map(|(k, o)| {
                    Some((
                        k,
                        o.seconds_per_byte()?,
                        o.unweighted_seconds_per_byte()?,
                        o.bytes_per_tuple()?,
                        o.seconds_per_byte_n,
                    ))
                })
                .collect();
            consts.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            debug!(
                "HMPPass learned constants (weighted mean {:.4e} s/byte):",
                model.mean_seconds_per_byte().unwrap_or(0.0)
            );
            // Both means, because the gap between them is how much one small
            // observation was distorting the constant.
            for (key, spb, plain, bytes_per_tuple, n) in &consts {
                debug!(
                    "  {key:<44} {spb:>11.4e} s/byte  (unweighted {plain:>11.4e})                       {bytes_per_tuple:>8.1} bytes/tuple  n={n}"
                );
            }
        }

        let mut rows: Vec<NodeRankingRow> = Vec::new();
        for node in dag.nodes.nodes() {
            if node.materialize != MaterializeMode::View {
                continue;
            }
            // The same gate every other method applies: only a branch point has
            // work that materializing could deduplicate.
            if dag.nodes.out_degree(&node.id) <= 1 || dag.nodes.paths_to_sinks(&node.id) <= 1 {
                continue;
            }
            let Some(node_stat) = stats.node_stats.get(&node.id) else {
                continue;
            };
            let Some(plan_str) = &node_stat.plan else {
                continue;
            };
            let Some(plans) = self.conn.parse_plan(plan_str) else {
                continue;
            };
            let Some(once) = model.cost(&plans) else {
                continue;
            };

            // What one build costs, or what the duplication costs: the View is
            // inlined into each materialized consumer on its frontier and paid
            // for once per consumer, so all but one of those copies is what
            // materializing it would remove.
            let copies = dag.nodes.frontier_materializes(&node.id).len().max(1);
            let total_cpu_time_s = if self.downstream_cost {
                once * (copies - 1) as f64
            } else {
                once
            };

            let cardinality = plans.first().and_then(|p| p.rows());
            let ranking_score = match (self.normalize_with_cardinality, cardinality) {
                (true, Some(c)) if c > 0.0 => total_cpu_time_s / c,
                _ => total_cpu_time_s,
            };
            rows.push(NodeRankingRow {
                rank: 0,
                node: node.id.clone(),
                total_cpu_time_s,
                cardinality,
                ranking_score,
                leaves: Vec::new(),
                matched: Vec::new(),
            });
        }

        rows.retain(|r| r.ranking_score > 0.0);
        if rows.is_empty() {
            return None;
        }
        rows.sort_by(|a, b| {
            b.ranking_score
                .partial_cmp(&a.ranking_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.node.cmp(&b.node))
        });
        for (i, row) in rows.iter_mut().enumerate() {
            row.rank = i + 1;
        }
        // The cumulative share is what `top_cpu_time` cuts the working set at,
        // so a ranking whose leader takes 99.9% of it explains a search that
        // considered one candidate and stopped.
        if log::log_enabled!(log::Level::Debug) {
            let total: f64 = rows.iter().map(|r| r.ranking_score).sum();
            let mut cumulative = 0.0;
            debug!("HMPPass learned ranking ({} candidates):", rows.len());
            for row in &rows {
                cumulative += row.ranking_score;
                debug!(
                    "  {:<48} {:>11.4}s  {:>6.2}% cumulative",
                    row.node,
                    row.ranking_score,
                    if total > 0.0 { cumulative / total * 100.0 } else { 0.0 }
                );
            }
        }
        Some(rows)
    }

    /// Rank candidate Views by the duplicate computation each one causes,
    /// measured by asking the engine.
    ///
    /// The method itself lives in [`crate::opt::dup`]; this is the part that
    /// is HMP's: which Views are candidates at all, and what a ranking row
    /// made of one looks like.
    ///
    /// Unlike every other method here, the score is a duplicate cost by
    /// construction, so `--hmp-downstream-cost` neither applies nor is
    /// consulted. A View whose duplication comes out at zero or below --- the
    /// engine plans its copies for no more than one standalone build would
    /// cost --- is dropped from the ranking rather than ranked last, because
    /// there is nothing for materializing it to remove.
    ///
    /// Returns `None` when not one candidate could be measured, which is
    /// "this method could not see" and not "this DAG duplicates nothing": the
    /// caller falls back rather than searching an empty ranking.
    async fn ranking_dup(&self, dag: &Dag, stats: &ExecStats) -> Option<Vec<NodeRankingRow>> {
        // The regions about to be priced are priced by constants, so the
        // constants have to know about this run first.
        if self.dup_cost_model == SubtreeCostMethod::LearnedCost {
            self.learn_from(dag, stats);
        }

        // A snapshot rather than the live model: the loop below awaits an
        // EXPLAIN per candidate, and a `MutexGuard` cannot be held across an
        // await -- nor should a lock be held for the length of a database round
        // trip. Nothing learns during the loop, so a snapshot taken now prices
        // every candidate by the same constants, which is also what makes the
        // ranking a comparison rather than a drift.
        let snapshot = self.learned.lock().ok()?.clone();
        let coster = self.dup_cost_model.coster(&snapshot);

        let mut rows: Vec<NodeRankingRow> = Vec::new();
        let mut measured = 0usize;
        for node in dag.nodes.nodes() {
            if node.materialize != MaterializeMode::View {
                continue;
            }
            // The same gate every other method applies: only a branch point has
            // work that materializing could deduplicate.
            if dag.nodes.out_degree(&node.id) <= 1 || dag.nodes.paths_to_sinks(&node.id) <= 1 {
                continue;
            }
            let Some(attributed) = duplicate_cost_set(
                self.conn.as_ref(),
                dag,
                std::slice::from_ref(&node.id),
                coster.as_ref(),
            )
            .await
            else {
                continue;
            };
            measured += 1;

            // Only for the cardinality normalization, and optional: the set
            // coster EXPLAINs what it needs, so a run that collected no plans
            // no longer blocks the ranking -- it just cannot normalize it.
            let cardinality = stats
                .node_stats
                .get(&node.id)
                .and_then(|s| s.plan.as_ref())
                .and_then(|p| self.conn.parse_plan(p))
                .and_then(|plan| plan.first().and_then(|p| p.rows()));
            let ranking_score = match (self.normalize_with_cardinality, cardinality) {
                (true, Some(c)) if c > 0.0 => attributed.duplicate / c,
                _ => attributed.duplicate,
            };
            // What was matched, so a ranking can be checked against the plans
            // it came from rather than taken on faith -- the same role these
            // play under leaf-set matching.
            let matched = attributed
                .per_consumer
                .iter()
                .map(|(consumer, cost)| format!("{consumer} ({cost:.4})"))
                .collect();
            rows.push(NodeRankingRow {
                rank: 0,
                node: node.id.clone(),
                total_cpu_time_s: attributed.duplicate,
                cardinality,
                ranking_score,
                leaves: Vec::new(),
                matched,
            });
        }

        if measured == 0 {
            return None;
        }
        rows.retain(|r| r.ranking_score > 0.0);
        if rows.is_empty() {
            // Measured, and every candidate came out at nothing. That is an
            // answer -- this DAG has no duplication worth removing -- and
            // falling back to a method that would invent some is worse than
            // reporting it.
            debug!(
                "HMPPass: duplicate attribution measured {measured} candidate(s) and found no \
                 duplication in any of them"
            );
            return Some(rows);
        }
        rows.sort_by(|a, b| {
            b.ranking_score
                .partial_cmp(&a.ranking_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.node.cmp(&b.node))
        });
        for (i, row) in rows.iter_mut().enumerate() {
            row.rank = i + 1;
        }
        Some(rows)
    }

    /// The ranking a run's plans imply, as `(node, score)`.
    ///
    /// Public so a costing method can be scored offline against ground truth
    /// measured by materializing the Views it ranks.
    pub async fn ranking_for(&self, dag: &Dag, stats: &ExecStats) -> Vec<NodeRankingRow> {
        if self.cost_method == HmpCostMethod::NodeTime {
            return Self::ranking_from_node_times(dag, stats);
        }
        if self.cost_method == HmpCostMethod::DupAttribution {
            match self.ranking_dup(dag, stats).await {
                Some(rows) => return rows,
                None => warn!(
                    "HMPPass: duplicate attribution could not measure a single candidate -- \
                     the engine answered no EXPLAIN, or no consumer's plan kept the View's \
                     materialized CTE; falling back to leaf-set matching"
                ),
            }
        }
        if self.cost_method == HmpCostMethod::LearnedCost {
            match self.ranking_learned(dag, stats) {
                Some(rows) => return rows,
                None => warn!(
                    "HMPPass: the learned cost model has priced nothing in this run -- no \
                     executed plan carried operator timings and widths to fit it to; \
                     falling back to leaf-set matching"
                ),
            }
        }
        if matches!(
            self.cost_method,
            HmpCostMethod::Leafset | HmpCostMethod::LearnedCost | HmpCostMethod::DupAttribution
        ) {
            match self.ranking_leafset(dag, stats) {
                Some(rows) => return rows,
                None => warn!(
                    "HMPPass: no plan in this run named a relation, so leaf-set matching \
                     has nothing to match against; falling back to signature matching"
                ),
            }
        }
        let aggregate = if self.downstream_cost {
            Self::aggregate_downstream_cost(self.conn.as_ref(), dag, stats)
        } else {
            let op_stats = self.operator_stats(dag, stats);
            Self::aggregate_cpu_time_avg(self.conn.as_ref(), dag, stats, &op_stats)
        };
        Self::build_node_table(
            self.conn.as_ref(),
            stats,
            aggregate,
            self.normalize_with_cardinality,
        )
    }

    /// Rank candidate Views by their own measured node time.
    ///
    /// The fallback when a run carried no EXPLAIN ANALYZE plans -- plan
    /// collection is a property of the run group, and a continuous
    /// optimization has to cope with a run that was not asked to collect them.
    /// Node time is coarser than operator CPU attribution, but it ranks the
    /// same branch points in roughly the same order, which is enough to search
    /// from. Ranking nothing at all, by contrast, would silently turn HMP into
    /// a no-op.
    fn ranking_from_node_times(dag: &Dag, stats: &ExecStats) -> Vec<NodeRankingRow> {
        let mut rows: Vec<NodeRankingRow> = dag
            .nodes
            .nodes()
            .filter(|n| n.materialize == MaterializeMode::View)
            .filter(|n| dag.nodes.out_degree(&n.id) > 1 && dag.nodes.paths_to_sinks(&n.id) > 1)
            .filter_map(|n| {
                let seconds = stats.node_stats.get(&n.id)?.duration.num_milliseconds() as f64
                    / 1000.0;
                Some(NodeRankingRow {
                    rank: 0,
                    node: n.id.clone(),
                    total_cpu_time_s: seconds,
                    cardinality: None,
                    ranking_score: seconds,
                    leaves: Vec::new(),
                    matched: Vec::new(),
                })
            })
            .collect();
        rows.sort_by(|a, b| {
            b.ranking_score
                .partial_cmp(&a.ranking_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (i, row) in rows.iter_mut().enumerate() {
            row.rank = i + 1;
        }
        rows
    }

    /// Price the combinations this search will spend its run budget on.
    ///
    /// Returns them ordered by the duplicate computation each removes, most
    /// first. This is the whole reason the pass no longer needs a strategy: the
    /// order candidates are tried in is decided by costing them, not by
    /// guessing from singleton scores and finding out one DAG run at a time.
    ///
    /// Costing always goes through
    /// [`duplicate_cost_set`](crate::opt::dup::duplicate_cost_set), whatever
    /// `cost_method` is set to. `cost_method` decides the *ranking* -- and so
    /// which Views make the working set -- but a Leafset or node-time score is
    /// not a quantity of duplicated work, and mixing one into a set's dup cost
    /// would be adding incommensurable units. When `cost_method` is not
    /// `DupAttribution` this costs one extra singleton pass; that is the price of
    /// a coherent objective.
    async fn build_candidates(
        &self,
        dag: &Dag,
        working_set: &[String],
    ) -> Vec<CandidateCombo> {
        if working_set.is_empty() {
            return Vec::new();
        }

        // Same snapshot-then-build dance as `ranking_dup`: the model cannot be
        // held across an await.
        let model = match self.learned.lock() {
            Ok(m) => m.clone(),
            Err(_) => LearnedCostModel::new(),
        };
        let coster = self.dup_cost_model.coster(&model);

        let singletons: HashMap<String, f64> = self
            .node_rows
            .iter()
            .map(|r| (r.node.clone(), r.ranking_score))
            .collect();

        let ordered = canonical_order(dag, working_set);
        let costed = search_combos(
            self.conn.as_ref(),
            dag,
            &ordered,
            &singletons,
            coster.as_ref(),
            self.search_budget,
        )
        .await;

        if !costed.is_empty() {
            debug!(
                "HMPPass: {} candidate combination(s) priced, best {:.4}",
                costed.len(),
                costed.first().map(|c| c.cost).unwrap_or(0.0)
            );
            return costed
                .into_iter()
                .map(|c| CandidateCombo {
                    combo: c.combo,
                    cost: c.cost,
                    partial: false,
                })
                .collect();
        }

        // Nothing could be priced: the connector cannot EXPLAIN, or no plan was
        // priceable. An empty list here would read as "nothing worth doing" and
        // promote on the next step, which would quietly turn the pass off. Fall
        // back to the unpriced enumeration -- smallest combinations first, in
        // ranking order -- so the run budget is still spent on something.
        warn!(
            "HMPPass: no candidate combination could be costed; falling back to \
             trying them smallest-first in ranking order"
        );
        let cap = self.search_budget.max(self.max_runs);
        let mut out = Vec::new();
        for k in 1..=working_set.len() {
            for combo in combinations(working_set, k) {
                if out.len() >= cap {
                    return out;
                }
                out.push(CandidateCombo {
                    combo,
                    cost: 0.0,
                    partial: true,
                });
            }
        }
        out
    }

    /// The prefix of `ranking` whose cumulative score covers `top_cpu_time` of
    /// the total -- the candidates worth searching.
    fn working_set_from(&self, ranking: &[NodeRankingRow]) -> Vec<String> {
        let total: f64 = ranking.iter().map(|r| r.ranking_score).sum();
        if total <= 0.0 {
            return Vec::new();
        }
        let mut set = Vec::new();
        let mut cumulative = 0.0;
        for row in ranking {
            set.push(row.node.clone());
            cumulative += row.ranking_score;
            if cumulative / total >= self.top_cpu_time {
                break;
            }
        }
        set
    }
    /// Apply `combo` to `dag`, then optionally push predicates into it.
    async fn build_trial(&self, dag: &mut Dag, combo: &[String]) -> Result<(), OptimizerError> {
        for node_id in combo {
            make_temp(dag, node_id)?;
        }
        if self.use_pushdown {
            let mut pushdown = PushdownPass::new(self.conn.clone(), self.engine.clone());
            if let Err(e) = pushdown.rewrite(dag).await {
                debug!("HMPPass: pushdown failed for combo {combo:?}, continuing without it: {e}");
            }
        }
        Ok(())
    }
    /// Everything the report needs, from the state as it now stands.
    fn outcome_from(&self, state: &HmpState) -> PassOutcome {
        PassOutcome {
            dag_runs_used: state.runs_used as u32,
            changes_applied: state.best_combo.len() as u32,
            // Every iteration past the baseline is one candidate evaluated.
            candidates_considered: state.iterations.len().saturating_sub(1) as u32,
            working_set_size: state.working_set.len() as u32,
            iterations: state.iterations.clone(),
            detail: PassDetail::Hmp(HmpDetail {
                baseline_runtime_ms: state.baseline_ms,
                final_runtime_ms: if state.best_ms == i64::MAX {
                    state.baseline_ms
                } else {
                    state.best_ms
                },
                max_runs: self.max_runs,
                top_cpu_time: self.top_cpu_time,
                search_budget: self.search_budget,
                candidates_costed: state.candidates.len(),
                normalize_with_cardinality: self.normalize_with_cardinality,
                downstream_cost: self.downstream_cost,
                use_pushdown: self.use_pushdown,
                new_materializations: state.best_combo.clone(),
                working_set: state.working_set.clone(),
            }),
        }
    }

    fn remember_explain(&mut self, state: &HmpState) {
        self.explain_data = Some(HMPExplainData {
            baseline_ms: state.baseline_ms,
            final_ms: if state.best_ms == i64::MAX {
                state.baseline_ms
            } else {
                state.best_ms
            },
            runs_used: state.runs_used,
            max_runs: self.max_runs,
            top_cpu_time: self.top_cpu_time,
            normalize_with_cardinality: self.normalize_with_cardinality,
            operator_rows: self.operator_rows.clone(),
            node_rows: self.node_rows.clone(),
            working_set: state.working_set.clone(),
            best_combo: state.best_combo.clone(),
            iterations: state.iterations.clone(),
            search_budget: self.search_budget,
            candidates_costed: state.candidates.len(),
        });
    }

    /// The wall-clock cap this search's next trial runs under.
    ///
    /// `None` until something has been measured -- there is no incumbent to be
    /// worse than -- and `None` when resuming is off, because a budget without
    /// a resume behind it would cancel the user's pipeline and leave the tables
    /// unbuilt. Where it does apply, a candidate can never cost more than
    /// `1 + eps` times the best combination found so far.
    fn budget(&self, state: &HmpState) -> Option<i64> {
        if !self.resume_trials || state.best_ms == i64::MAX || state.best_ms <= 0 {
            return None;
        }
        Some(((state.best_ms as f64) * (1.0 + self.budget_eps)).round() as i64)
    }

    /// This search's incumbent as a DAG: the authored definition with the best
    /// combination measured so far materialized in it.
    ///
    /// An empty `best_combo` is not "no incumbent" -- it is the incumbent, the
    /// DAG as authored, which is what every candidate is being compared
    /// against. Returning `None` there would leave the *first* trials, the ones
    /// most likely to be bad guesses, running to completion.
    ///
    /// `None` only before a baseline has been measured, when there is genuinely
    /// nothing to be worse than.
    async fn incumbent_dag(&self, dag: &Dag, state: &HmpState) -> Option<Box<Dag>> {
        if state.best_ms == i64::MAX {
            return None;
        }
        let mut fallback = dag.clone();
        if !state.best_combo.is_empty() {
            self.build_trial(&mut fallback, &state.best_combo).await.ok()?;
        }
        Some(Box::new(fallback))
    }

    /// Decide and apply what this run should try.
    async fn step_before(
        &mut self,
        ctx: &mut StepContext<'_, C, E>,
    ) -> Result<StepOutcome, OptimizerError> {
        let Some(mut state) = self.load_state(ctx.store, ctx.dag_id).await? else {
            // Not registered, or registered and never stepped. Either way there
            // is nothing to propose yet.
            return Ok(StepOutcome::Idle);
        };

        match state.phase.as_str() {
            // The baseline is the DAG as it stands, so the right thing to do
            // before it is nothing at all.
            "baseline" => Ok(StepOutcome::Idle),
            "converged" => Ok(StepOutcome::Idle),
            "searching" => {
                if state.in_flight.is_some() {
                    // A trial was proposed and never reported on -- a run that
                    // failed or was cancelled. Re-propose it rather than
                    // scoring it from a run that never happened.
                    let combo = state.in_flight.as_ref().unwrap().combo.clone();
                    let fallback = self.incumbent_dag(ctx.dag, &state).await;
                    self.build_trial(ctx.dag, &combo).await?;
                    return Ok(StepOutcome::Trial {
                        reuse: self.reuse_policy,
                        label: describe(&combo),
                        budget_ms: self.budget(&state),
                        fallback,
                        record: Box::new(self.outcome_from(&state)),
                    });
                }

                if state.runs_used >= self.max_runs {
                    return self.promote(ctx, state).await;
                }

                // A state row written before the search kept a candidate list
                // decodes with an empty one. That is "not enumerated yet", not
                // "exhausted", and promoting here would silently end a search
                // that was merely persisted by an older build.
                if !state.candidates_built {
                    warn!(
                        "HMPPass: this search predates the candidate list and cannot be \
                         resumed mid-flight; promoting the best combination it had found"
                    );
                    return self.promote(ctx, state).await;
                }

                // Walk the candidates in the order they were priced, skipping
                // any that reduce to a DAG already measured: each costs nothing
                // but a rewrite, and paying a DAG run for a duplicate is the
                // expensive mistake.
                loop {
                    let Some(candidate) = state.candidates.get(state.cursor) else {
                        return self.promote(ctx, state).await;
                    };
                    let combo = candidate.combo.clone();
                    state.cursor += 1;

                    let mut trial = ctx.dag.clone();
                    // A combination that cannot be rewritten is not a failed
                    // search, it is a candidate that does not exist: `make_temp`
                    // declines to inline a View it cannot rebuild at the AST
                    // level rather than corrupt the query. Skipping it costs one
                    // rewrite; propagating it abandons the whole optimization
                    // over a candidate that was never going to be measured.
                    if let Err(e) = self.build_trial(&mut trial, &combo).await {
                        warn!("HMPPass: combo {combo:?} cannot be built, skipping it: {e}");
                        continue;
                    }
                    let sig = dag_signature(&trial);
                    let fallback = self.incumbent_dag(ctx.dag, &state).await;

                    if state.tried_sigs.contains(&sig) {
                        debug!("combo {combo:?} reduces to a DAG already tried, skipping");
                        continue;
                    }

                    state.tried_sigs.push(sig.clone());
                    state.in_flight = Some(InFlight {
                        combo: combo.clone(),
                        sig,
                    });
                    self.save_state(ctx.store, ctx.dag_id, &state).await?;

                    *ctx.dag = trial;
                    return Ok(StepOutcome::Trial {
                        reuse: self.reuse_policy,
                        label: describe(&combo),
                        budget_ms: self.budget(&state),
                        fallback,
                        record: Box::new(self.outcome_from(&state)),
                    });
                }
            }
            other => {
                warn!("HMPPass: unrecognized state '{other}'; leaving the DAG alone");
                Ok(StepOutcome::Idle)
            }
        }
    }

    /// Apply the best combination found and hand it over to be stored.
    ///
    /// `ctx.dag` is the committed definition here, not a trial: promotion
    /// happens on a `Before` step precisely so the winner is built from the
    /// DAG as authored rather than from whichever candidate ran last.
    async fn promote(
        &mut self,
        ctx: &mut StepContext<'_, C, E>,
        mut state: HmpState,
    ) -> Result<StepOutcome, OptimizerError> {
        state.phase = "converged".to_string();
        state.in_flight = None;

        for node_id in &state.best_combo {
            make_temp(ctx.dag, node_id)?;
        }
        self.save_state(ctx.store, ctx.dag_id, &state).await?;
        self.remember_explain(&state);

        debug!(
            "HMPPass converged: materialized {} view(s) using {}/{} runs",
            state.best_combo.len(),
            state.runs_used,
            self.max_runs
        );

        let record = Box::new(self.outcome_from(&state));
        if state.best_combo.is_empty() {
            // Nothing beat the baseline. Saying so is not the same as
            // promoting an unchanged DAG as a new version.
            Ok(StepOutcome::Done { record })
        } else {
            Ok(StepOutcome::Promote { record })
        }
    }

    /// Learn from the run that just finished.
    async fn step_after(
        &mut self,
        ctx: &mut StepContext<'_, C, E>,
    ) -> Result<StepOutcome, OptimizerError> {
        let Some(mut state) = self.load_state(ctx.store, ctx.dag_id).await? else {
            return Ok(StepOutcome::Idle);
        };
        if state.phase == "converged" {
            return Ok(StepOutcome::Idle);
        }
        let Some(run) = ctx.run.clone() else {
            return Ok(StepOutcome::Idle);
        };
        // A warmup is deliberately not a measurement: its whole purpose is to
        // absorb cold-cache cost that the numbers this search compares must
        // not contain.
        if !run.is_measured() {
            return Ok(StepOutcome::Idle);
        }
        // No stats on a measured run means the execution produced no usable
        // time -- it was cancelled at its budget, or it failed. Either way it is
        // a censored observation, and a censored observation is enough to
        // reject: what it says is "at least as slow as the cap", and the cap is
        // already worse than the best combination found so far.
        //
        // This must clear `in_flight`. Leaving it set would make the next
        // `Before` step re-propose the very candidate that was just cancelled,
        // and the search would spend its whole run budget on one bad combo.
        let Some(stats) = run.stats.as_ref() else {
            return self.reject_censored(ctx, state, &run.run_id).await;
        };

        let runtime_ms = stats.duration.num_milliseconds();

        // Everything earlier runs measured, before anything is priced by it.
        self.adopt_learned(ctx.store).await;

        if state.phase == "baseline" {
            state.baseline_ms = runtime_ms;
            state.best_ms = runtime_ms;
            state.runs_used = 1;
            state.iterations.push(
                IterationStat::new(1, runtime_ms)
                    .with_outcome("baseline")
                    .with_run_cost(ctx)
                    .with_samples(if self.profile_iterations {
                        stats.system_samples.clone()
                    } else {
                        Vec::new()
                    }),
            );

            let mut ranking = self.ranking_for(ctx.dag, stats).await;
            if ranking.is_empty() {
                debug!("HMPPass: no plan-derived ranking; falling back to node times");
                ranking = Self::ranking_from_node_times(ctx.dag, stats);
            }
            self.log_node_table(&ranking);
            let op_stats = self.operator_stats(ctx.dag, stats);
            self.log_operator_table(ctx.dag, stats, &op_stats);
            self.operator_rows =
                Self::build_operator_table(self.conn.as_ref(), ctx.dag, stats, &op_stats);

            state.baseline_scores = ranking
                .iter()
                .map(|r| (r.node.clone(), r.ranking_score))
                .collect();
            state.working_set = self.working_set_from(&ranking);
            self.node_rows = ranking;
            state.phase = "searching".to_string();

            debug!(
                "HMPPass baseline {runtime_ms}ms; working set of {} node(s): {:?}",
                state.working_set.len(),
                state.working_set
            );

            state.candidates = self.build_candidates(ctx.dag, &state.working_set).await;
            state.cursor = 0;
            state.candidates_built = true;
            self.record_trial(ctx.store, ctx.dag_id, &run.run_id, &state, runtime_ms, true)
                .await?;
            self.save_state(ctx.store, ctx.dag_id, &state).await?;
            self.publish_learned(ctx.store).await;
            self.remember_explain(&state);
            return Ok(StepOutcome::Idle);
        }

        let Some(in_flight) = state.in_flight.take() else {
            // A run that this search did not propose -- a scheduled run that
            // landed while nothing was in flight. It measured the committed
            // DAG, not a candidate, so there is nothing to attribute.
            return Ok(StepOutcome::Idle);
        };

        state.runs_used += 1;
        state.iterations.push(
            IterationStat::new(state.iterations.len() + 1, runtime_ms)
                .with_combo(in_flight.combo.clone())
                .with_outcome("ok")
                .with_run_cost(ctx)
                .with_samples(if self.profile_iterations {
                    stats.system_samples.clone()
                } else {
                    Vec::new()
                }),
        );
        // Fit the cost model to what this trial actually executed. The search
        // no longer re-ranks between rounds, so this is what keeps trial
        // evidence reaching the model -- and it reads the plans the run already
        // collected rather than issuing EXPLAINs of its own.
        self.learn_from(ctx.dag, stats);

        let improved = runtime_ms < state.best_ms;
        if improved {
            debug!(
                "combo {:?} improved runtime: {}ms -> {runtime_ms}ms",
                in_flight.combo, state.best_ms
            );
            state.best_ms = runtime_ms;
            state.best_combo = in_flight.combo.clone();
        } else {
            debug!(
                "combo {:?} did not improve runtime ({}ms -> {runtime_ms}ms)",
                in_flight.combo, state.best_ms
            );
        }

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
        self.publish_learned(ctx.store).await;
        self.remember_explain(&state);
        Ok(StepOutcome::Idle)
    }

    /// File a trial that produced no usable measurement and move the search on.
    async fn reject_censored(
        &mut self,
        ctx: &mut StepContext<'_, C, E>,
        mut state: HmpState,
        run_id: &str,
    ) -> Result<StepOutcome, OptimizerError> {
        let Some(in_flight) = state.in_flight.take() else {
            // Nothing was proposed, so nothing was censored. A run this search
            // did not cause failing is not its business.
            return Ok(StepOutcome::Idle);
        };
        if state.phase == "baseline" {
            // The baseline is the DAG as it stands and is not a candidate; a
            // search that recorded a censored baseline would compare every
            // later trial against a number no run produced.
            return Ok(StepOutcome::Idle);
        }

        state.runs_used += 1;
        // A lower bound, not a measurement: at least the budget. Recording it
        // as the candidate's runtime keeps the search from ever preferring it,
        // without claiming to know how bad it was.
        let censored_ms = self.budget(&state).unwrap_or(i64::MAX);
        debug!(
            "combo {:?} produced no usable measurement; rejecting as censored (>= {censored_ms}ms)",
            in_flight.combo
        );
        // `runtime_ms` is the censoring level, not a measurement. What the
        // iteration really cost -- the cut-short trial, the optimizer working
        // out what to keep, and the incumbent finishing the job -- comes off
        // the run itself, and only this path has all three.
        state.iterations.push(
            IterationStat::new(state.iterations.len() + 1, censored_ms)
                .with_combo(in_flight.combo.clone())
                .with_outcome("cancelled")
                .with_run_cost(ctx),
        );
        // A cancelled candidate is still a candidate that has been paid for.
        // `step_before` records the signature as it proposes, so this is
        // normally already present -- but a combo that reached here by any
        // other route must not be re-proposed either.
        if !state.tried_sigs.contains(&in_flight.sig) {
            state.tried_sigs.push(in_flight.sig.clone());
        }

        self.record_trial(ctx.store, ctx.dag_id, run_id, &state, censored_ms, false)
            .await?;
        self.save_state(ctx.store, ctx.dag_id, &state).await?;
        self.remember_explain(&state);
        Ok(StepOutcome::Idle)
    }

    async fn record_trial(
        &self,
        store: &dyn OptStore,
        dag_id: &str,
        run_id: &str,
        state: &HmpState,
        runtime_ms: i64,
        improved: bool,
    ) -> Result<(), OptimizerError> {
        let combo = state
            .iterations
            .last()
            .map(|i| i.combo.join(","))
            .unwrap_or_default();
        store
            .execute(
                &format!(
                    "INSERT INTO {TRIALS_TABLE} \
                     (dag_id, run_id, iteration, combo, runtime_ms, improved, recorded_at) \
                     VALUES (?, ?, ?, ?, ?, ?, now())"
                ),
                &[
                    json!(dag_id),
                    json!(run_id),
                    json!(state.iterations.len()),
                    json!(combo),
                    json!(runtime_ms),
                    json!(improved),
                ],
            )
            .await?;
        Ok(())
    }
}

/// A combination, for a log line or a report label.
fn describe(combo: &[String]) -> String {
    if combo.is_empty() {
        "baseline".to_string()
    } else {
        combo.join(", ")
    }
}

#[async_trait]
impl<C, E> Optimization<C, E> for HMPPass<C, E>
where
    C: Connector + Send + 'static + Sync,
    E: Executor<C> + Send + Sync,
{
    fn name(&self) -> &'static str {
        "hmp"
    }

    /// HMP decides by measurement, and the measurements it needs are runs of
    /// the DAG -- which the server is performing anyway.
    fn optimization_type(&self) -> OptimizationType {
        OptimizationType::Continuous
    }

    fn step_phase(&self) -> StepPhase {
        self.step_phase
    }

    fn set_step_phase(&mut self, phase: StepPhase) {
        self.step_phase = phase;
    }

    async fn register(
        &self,
        ctx: &RegisterContext<'_>,
    ) -> Result<Option<Registration>, OptimizerError> {
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
                         combo       VARCHAR,
                         runtime_ms  BIGINT,
                         improved    BOOLEAN,
                         recorded_at TIMESTAMPTZ NOT NULL
                     )"
                ),
                &[],
            )
            .await?;

        ctx.store
            .execute(
                &format!(
                    "CREATE TABLE IF NOT EXISTS {LEARNED_TABLE} (
                         model      VARCHAR NOT NULL,
                         updated_at TIMESTAMPTZ NOT NULL
                     )"
                ),
                &[],
            )
            .await?;

        // Registering is idempotent -- a server restart re-registers what a
        // DAG already had -- so an existing search is left exactly where it
        // was rather than restarted from its baseline.
        if self.load_state(ctx.store, ctx.dag_id).await?.is_none() {
            self.save_state(ctx.store, ctx.dag_id, &HmpState::new())
                .await?;
        }

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
        // Only this DAG's rows: the tables are shared by every DAG HMP is
        // registered on, so dropping them would take the others' searches with
        // it. They are dropped when the last one goes.
        for table in [STATE_TABLE, TRIALS_TABLE] {
            ctx.store
                .execute(
                    &format!("DELETE FROM {table} WHERE dag_id = ?"),
                    &[json!(ctx.dag_id)],
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
            // The learned constants have no `dag_id` to delete by -- they
            // describe the engine, not any one DAG -- so they go when the last
            // registration does, with everything else.
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
        match ctx.side {
            StepPhase::Before => self.step_before(ctx).await,
            StepPhase::After => self.step_after(ctx).await,
            StepPhase::Both => Ok(StepOutcome::Idle),
        }
    }

    fn explain(&self) -> Option<(String, String)> {
        Some(("HMPPass".to_string(), self.explain_html()))
    }
}

impl<C, E> HMPPass<C, E>
where
    C: Connector + Send + 'static + Sync,
    E: Executor<C> + Send + Sync,
{
    fn explain_html(&self) -> String {
        let Some(data) = &self.explain_data else {
            return r#"<div class="panel"><p class="subtle">HMPPass did not run.</p></div>"#
                .to_string();
        };

        let change_pct = if data.baseline_ms > 0 {
            (data.final_ms - data.baseline_ms) as f64 / data.baseline_ms as f64 * 100.0
        } else {
            0.0
        };

        let cards = render_card_grid(&[
            ("Baseline runtime", format!("{} ms", data.baseline_ms)),
            ("Final runtime", format!("{} ms", data.final_ms)),
            ("Change", format!("{change_pct:+.1}%")),
            ("Materializations chosen", data.best_combo.len().to_string()),
            (
                "Search budget used",
                format!("{}/{} runs", data.runs_used, data.max_runs),
            ),
            (
                "Combinations costed",
                format!("{}/{}", data.candidates_costed, data.search_budget),
            ),
            (
                "Ranking normalized by cardinality",
                if data.normalize_with_cardinality {
                    "yes"
                } else {
                    "no"
                }
                .to_string(),
            ),
        ]);

        let node_rows: Vec<Vec<String>> = data
            .node_rows
            .iter()
            .map(|r| {
                vec![
                    r.rank.to_string(),
                    r.node.clone(),
                    format!("{:.4}s", r.total_cpu_time_s),
                    r.cardinality
                        .map(|c| format!("{:.0}", c))
                        .unwrap_or_else(|| "-".to_string()),
                    format!("{:.4}", r.ranking_score),
                    if data.working_set.contains(&r.node) {
                        "yes".to_string()
                    } else {
                        "no".to_string()
                    },
                    if r.leaves.is_empty() {
                        "-".to_string()
                    } else {
                        r.leaves.join(", ")
                    },
                    if r.matched.is_empty() {
                        "-".to_string()
                    } else {
                        r.matched.join("; ")
                    },
                ]
            })
            .collect();
        let node_table = render_ranked_table(
            &[
                "Rank",
                "View",
                "Aggregate CPU time",
                "Cardinality",
                "Ranking score",
                "In working set",
                // Empty under signature matching, which has no notion of
                // either -- these are what make a leaf-set attribution
                // checkable rather than merely reported.
                "Reads",
                "Matched region",
            ],
            &node_rows,
        );

        let operator_rows: Vec<Vec<String>> = data
            .operator_rows
            .iter()
            .take(15)
            .map(|r| {
                vec![
                    r.rank.to_string(),
                    r.operator.clone(),
                    format!("{:.4}s", r.avg_runtime_s),
                    r.table_occurrences.to_string(),
                    r.traced_views.join(", "),
                ]
            })
            .collect();
        let operator_table = render_ranked_table(
            &[
                "Rank",
                "Operator",
                "Avg runtime",
                "Table occurrences",
                "Traced view(s)",
            ],
            &operator_rows,
        );

        let max_iter_ms = data
            .iterations
            .iter()
            .map(|i| i.runtime_ms)
            .max()
            .unwrap_or(1)
            .max(1);
        let iteration_bars: String = data
            .iterations
            .iter()
            .map(|it| {
                let label = if it.combo.is_empty() {
                    format!("Iteration {} (baseline)", it.iteration)
                } else {
                    format!("Iteration {}: materialize [{}]", it.iteration, it.combo.join(", "))
                };
                let is_winner = it.combo == data.best_combo && !it.combo.is_empty();
                let label = if is_winner {
                    format!("{label} — chosen")
                } else {
                    label
                };
                render_bar_row(
                    &label,
                    &format!("{} ms", it.runtime_ms),
                    it.runtime_ms as f64 / max_iter_ms as f64 * 100.0,
                )
            })
            .collect();

        let combinations_desc = format!(
            "Combinations of working-set nodes were priced before any of them was run: up to \
             {} of them, each costed by inlining its members as materialized CTEs and charging \
             every copy but one. A combination is costed as a chain -- each member against a DAG \
             in which the members before it are already tables -- so a node is not credited with \
             removing work that another member of the same combination already removed. {} were \
             costed, and the run budget was spent on them in descending order of the duplicate \
             computation they remove.",
            data.search_budget, data.candidates_costed
        );

        format!(
            r##"<div class="section-stack">
        {cards}
        <div class="panel">
          <h2>Why these nodes were considered</h2>
          <div class="subtle">Views with out-degree &gt; 1 and more than one downstream path to a TABLE/TEMP_TABLE node are candidates because materializing them can deduplicate work repeated by every downstream consumer. They're ranked by the aggregate CPU time of every operator (from the baseline's EXPLAIN plans) traced back to them. The working set walks this ranking, accumulating nodes until it covers {:.0}% of the total ranked CPU time.</div>
          {node_table}
        </div>
        <div class="panel">
          <h2>Operators traced back to candidate views</h2>
          <div class="subtle">Operators from materialized plans, with their average runtime across occurrences.</div>
          {operator_table}
        </div>
        <div class="panel">
          <h2>Combinations searched</h2>
          <div class="subtle">
            {}
            The combination with the lowest runtime was applied to the DAG.
          </div>
          <div class="plan-tree">{iteration_bars}</div>
        </div>
      </div>"##,
            data.top_cpu_time * 100.0,
            combinations_desc
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::ProfilingConfig;
    use crate::opt::{
        Optimizer, OptimizerConfig,
        store::{MemoryStore, MemoryStoreFactory},
    };
    use std::collections::HashSet;
    use crate::connectors::duckdb::{DuckDBConfig, DuckDBConnection};
    use crate::dag::TransformNode;
    use crate::executor::SimpleEngine;
    use crate::graph::Graph;
    use chrono::Utc;

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
            sources: vec![],
            max_parallelism: None,
        }
    }

    fn node_stats(plan: Option<String>) -> crate::executor::NodeStats {
        let now = Utc::now();
        crate::executor::NodeStats {
            start: now,
            finish: now,
            duration: chrono::TimeDelta::zero(),
            plan,
            rows_produced: None,
        }
    }

    async fn test_pass(
        search_budget: usize,
    ) -> HMPPass<DuckDBConnection, SimpleEngine<DuckDBConnection>> {
        let conn = in_memory_conn().await;
        let engine = Arc::new(SimpleEngine::new(Arc::clone(&conn)).unwrap());
        HMPPass::new(
            conn,
            engine,
            false,
            1,
            1.0,
            None,
            None,
            false,
            // The existing tests fabricate operator-signature plans, which is
            // what this method reads.
            HmpCostMethod::Signature,
            SubtreeCostMethod::default(),
            false,
            search_budget,
            false,
            true,
            crate::opt::common::DEFAULT_BUDGET_EPS,
            Default::default(),
        )
    }

    /// A DuckDB profiling plan for a consumer that joins `raw_a` and `raw_b`
    /// and then groups the result.
    fn grouped_join_plan() -> String {
        let scan = |table: &str, t: f64| {
            format!(
                r#"{{"operator_name":"SEQ_SCAN","operator_timing":{t},
                     "operator_cardinality":1000,
                     "extra_info":{{"Table":"{table}","Estimated Cardinality":1000}},
                     "children":[]}}"#
            )
        };
        format!(
            r#"{{"operator_name":"HASH_GROUP_BY","operator_timing":1.0,
                 "operator_cardinality":10,
                 "extra_info":{{"Estimated Cardinality":10}},
                 "children":[
                   {{"operator_name":"HASH_JOIN","operator_timing":6.0,
                     "operator_cardinality":1000,
                     "extra_info":{{"Estimated Cardinality":1000}},
                     "children":[{}, {}]}}]}}"#,
            scan("raw_a", 1.0),
            scan("raw_b", 2.0)
        )
    }

    fn leafset_dag() -> Dag {
        //   raw_a, raw_b (declared sources)
        //        |
        //     joined (View, branch point: two Table consumers)
        //      /        \
        //   out_a(T)   out_b(T)
        let mut dag = make_dag(vec![
            node(
                "joined",
                "SELECT k, v FROM raw_a JOIN raw_b USING (k)",
                MaterializeMode::View,
                &[],
            ),
            node(
                "out_a",
                "SELECT k, count(*) FROM joined GROUP BY k",
                MaterializeMode::Table,
                &["joined"],
            ),
            node(
                "out_b",
                "SELECT k, count(*) FROM joined GROUP BY k",
                MaterializeMode::Table,
                &["joined"],
            ),
        ]);
        dag.sources = ["raw_a", "raw_b"]
            .iter()
            .map(|name| crate::dag::SourceNode {
                name: name.to_string(),
                schema: std::sync::Arc::new(duckdb::arrow::datatypes::Schema::empty()),
            })
            .collect();
        dag
    }

    fn leafset_stats(consumer_ms: i64) -> ExecStats {
        let now = Utc::now();
        let mut consumer = node_stats(Some(grouped_join_plan()));
        consumer.duration = chrono::TimeDelta::milliseconds(consumer_ms);
        let mut node_stats_map = HashMap::new();
        node_stats_map.insert("out_a".to_string(), consumer.clone());
        node_stats_map.insert("out_b".to_string(), consumer);
        node_stats_map.insert("joined".to_string(), node_stats(None));
        ExecStats {
            start: now,
            finish: now,
            duration: chrono::TimeDelta::milliseconds(consumer_ms * 2),
            node_stats: node_stats_map,
            system_samples: Vec::new(),
        }
    }

    #[tokio::test]
    async fn leafset_charges_a_view_the_region_of_the_plan_it_owns() {
        let mut pass = test_pass(2).await;
        pass.cost_method = HmpCostMethod::Leafset;
        let dag = leafset_dag();

        let rows = pass
            .ranking_leafset(&dag, &leafset_stats(1000))
            .expect("the plans name relations");
        assert_eq!(rows.len(), 1, "only the branch-point View is a candidate");
        assert_eq!(rows[0].node, "joined");

        // The view reads both relations but does not group, so its region is
        // the join and its two scans -- 9s of the plan's 10s -- and the
        // consumer measured 1s. `compute_secs` is the largest of the two
        // identical consumers.
        assert!(
            (rows[0].total_cpu_time_s - 0.9).abs() < 1e-6,
            "expected a 0.9 share of one consumer, got {}",
            rows[0].total_cpu_time_s
        );
        // The join emitted 1000 rows; the GROUP BY above it emitted 10. Landing
        // on the aggregate would report the view 100x too narrow, which is what
        // makes a too-large intermediate look safe to persist.
        assert_eq!(rows[0].cardinality, Some(1000.0));
    }

    #[tokio::test]
    async fn leafset_downstream_cost_reports_only_what_deduplication_removes() {
        let mut pass = test_pass(2).await;
        pass.cost_method = HmpCostMethod::Leafset;
        pass.downstream_cost = true;
        let rows = pass
            .ranking_leafset(&leafset_dag(), &leafset_stats(1000))
            .expect("the plans name relations");
        // Two consumers each pay 0.9s; building the view once removes one of
        // them, not both.
        assert!(
            (rows[0].total_cpu_time_s - 0.9).abs() < 1e-6,
            "got {}",
            rows[0].total_cpu_time_s
        );
    }

    // ---- LearnedCost -----------------------------------------------------

    /// A profiled consumer plan: every operator carries a timing, a row count
    /// and a `result_set_size`, which is what a seconds-per-byte constant is
    /// fitted to.
    ///
    ///   SEQ_SCAN raw_a  1.0s / 1000 rows / 24000 bytes -> 1/24000 s/byte
    ///   SEQ_SCAN raw_b  2.0s / 1000 rows / 24000 bytes -> 2/24000 s/byte
    ///   HASH_JOIN       6.0s / 1000 rows / 32000 bytes -> 6/32000 s/byte
    ///   HASH_GROUP_BY   1.0s /   10 rows /   400 bytes -> 1/400   s/byte
    fn profiled_consumer_plan() -> String {
        let scan = |table: &str, time: f64| {
            format!(
                r#"{{"operator_name":"SEQ_SCAN","operator_timing":{time},
                     "operator_cardinality":1000,"result_set_size":24000,
                     "extra_info":{{"Table":"{table}","Estimated Cardinality":1000}},
                     "children":[]}}"#
            )
        };
        format!(
            r#"{{"operator_name":"HASH_GROUP_BY","operator_timing":1.0,
                 "operator_cardinality":10,"result_set_size":400,
                 "extra_info":{{"Aggregates":["count_star()"],"Estimated Cardinality":10}},
                 "children":[
                   {{"operator_name":"HASH_JOIN","operator_timing":6.0,
                     "operator_cardinality":1000,"result_set_size":32000,
                     "extra_info":{{"Estimated Cardinality":1000}},
                     "children":[{}, {}]}}]}}"#,
            scan("raw_a", 1.0),
            scan("raw_b", 2.0)
        )
    }

    /// The View's own plan: a plain `EXPLAIN (FORMAT JSON)`, so estimates and
    /// nothing else -- no timings, no sizes. Pricing this is the whole point.
    fn unexecuted_view_plan() -> String {
        let scan = |table: &str| {
            format!(
                r#"{{"name":"SEQ_SCAN",
                     "extra_info":{{"Table":"{table}","Estimated Cardinality":1000}},
                     "children":[]}}"#
            )
        };
        format!(
            r#"[{{"name":"HASH_JOIN","extra_info":{{"Estimated Cardinality":1000}},
                  "children":[{}, {}]}}]"#,
            scan("raw_a"),
            scan("raw_b")
        )
    }

    /// `leafset_dag`, but with the View's own plan attached and the consumers'
    /// plans profiled.
    fn learned_stats(consumers: &[&str]) -> ExecStats {
        let now = Utc::now();
        let mut node_stats_map = HashMap::new();
        for id in consumers {
            node_stats_map.insert(id.to_string(), node_stats(Some(profiled_consumer_plan())));
        }
        node_stats_map.insert("joined".to_string(), node_stats(Some(unexecuted_view_plan())));
        ExecStats {
            start: now,
            finish: now,
            duration: chrono::TimeDelta::milliseconds(1000),
            node_stats: node_stats_map,
            system_samples: Vec::new(),
        }
    }

    fn learned_dag(consumers: &[&str]) -> Dag {
        let mut nodes = vec![node(
            "joined",
            "SELECT k, v FROM raw_a JOIN raw_b USING (k)",
            MaterializeMode::View,
            &[],
        )];
        for id in consumers {
            nodes.push(node(
                id,
                "SELECT k, count(*) FROM joined GROUP BY k",
                MaterializeMode::Table,
                &["joined"],
            ));
        }
        let mut dag = make_dag(nodes);
        dag.sources = ["raw_a", "raw_b"]
            .iter()
            .map(|name| crate::dag::SourceNode {
                name: name.to_string(),
                schema: std::sync::Arc::new(duckdb::arrow::datatypes::Schema::empty()),
            })
            .collect();
        dag
    }

    #[tokio::test]
    async fn learned_cost_prices_a_views_own_plan_from_what_the_tables_measured() {
        let mut pass = test_pass(2).await;
        pass.cost_method = HmpCostMethod::LearnedCost;

        let rows = pass
            .ranking_learned(&learned_dag(&["out_a", "out_b"]), &learned_stats(&["out_a", "out_b"]))
            .expect("the executed plans carry timings and widths");

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].node, "joined");
        // HASH_JOIN:  1000 rows x 32 bytes x 6/32000 s/byte  = 6.0
        // SEQ_SCAN:   1000 rows x 24 bytes x 1.5/24000 s/byte = 1.5, twice.
        //
        // The View's plan carries no widths at all -- DuckDB reports none
        // without profiling -- so both came from what the same operator types
        // measured on the consumers.
        assert!(
            (rows[0].total_cpu_time_s - 9.0).abs() < 1e-9,
            "cost was {}",
            rows[0].total_cpu_time_s
        );
        assert_eq!(rows[0].cardinality, Some(1000.0));
    }

    #[tokio::test]
    async fn learned_cost_downstream_charges_every_copy_but_the_one_that_would_remain() {
        let mut pass = test_pass(2).await;
        pass.cost_method = HmpCostMethod::LearnedCost;
        pass.downstream_cost = true;

        let consumers = ["out_a", "out_b", "out_c"];
        let rows = pass
            .ranking_learned(&learned_dag(&consumers), &learned_stats(&consumers))
            .expect("the executed plans carry timings and widths");

        // Inlined into three consumers and paid for three times; materializing
        // it leaves one, so two builds' worth is what would vanish.
        assert!(
            (rows[0].total_cpu_time_s - 18.0).abs() < 1e-9,
            "cost was {}",
            rows[0].total_cpu_time_s
        );
    }

    #[tokio::test]
    async fn learned_cost_keeps_a_separate_constant_per_aggregate_function() {
        // Same operator, same bytes, different function and a very different
        // time. One constant fitted to both would price each at the mean.
        let plan = |func: &str, seconds: f64| {
            format!(
                r#"{{"operator_name":"HASH_GROUP_BY","operator_timing":{seconds},
                     "operator_cardinality":100,"result_set_size":1000,
                     "extra_info":{{"Aggregates":["{func}"],"Estimated Cardinality":100}},
                     "children":[]}}"#
            )
        };
        let now = Utc::now();
        let dag = learned_dag(&["out_a", "out_b"]);
        let mut node_stats_map = HashMap::new();
        node_stats_map.insert("out_a".to_string(), node_stats(Some(plan("count_star()", 1.0))));
        node_stats_map.insert("out_b".to_string(), node_stats(Some(plan("string_agg(#1)", 9.0))));
        node_stats_map.insert("joined".to_string(), node_stats(None));
        let stats = ExecStats {
            start: now,
            finish: now,
            duration: chrono::TimeDelta::milliseconds(1000),
            node_stats: node_stats_map,
            system_samples: Vec::new(),
        };

        let mut pass = test_pass(2).await;
        pass.cost_method = HmpCostMethod::LearnedCost;
        // No View plan to rank, but the learning half still runs.
        let _ = pass.ranking_learned(&dag, &stats);

        let model = pass.learned.lock().unwrap();
        let cheap = model.seconds_per_byte("HASH_GROUP_BY[count_star]").unwrap();
        let dear = model.seconds_per_byte("HASH_GROUP_BY[string_agg]").unwrap();
        assert!((cheap - 1.0 / 1000.0).abs() < 1e-12);
        assert!((dear - 9.0 / 1000.0).abs() < 1e-12);
    }

    #[tokio::test]
    async fn learned_constants_outlive_the_pass_that_measured_them() {
        let store = MemoryStore::open("hmp").unwrap();
        let consumers = ["out_a", "out_b"];

        // One pass learns from a run and writes what it learned away.
        let mut first = test_pass(2).await;
        first.cost_method = HmpCostMethod::LearnedCost;
        store
            .execute(
                &format!(
                    "CREATE TABLE IF NOT EXISTS {LEARNED_TABLE} \
                     (model VARCHAR NOT NULL, updated_at TIMESTAMPTZ NOT NULL)"
                ),
                &[],
            )
            .await
            .unwrap();
        first
            .ranking_learned(&learned_dag(&consumers), &learned_stats(&consumers))
            .expect("learned something");
        first.publish_learned(&store).await;

        // A second pass, which has measured nothing itself, prices the same
        // View identically -- and does it from a run in which no consumer
        // carried a plan at all.
        let mut second = test_pass(2).await;
        second.cost_method = HmpCostMethod::LearnedCost;
        second.adopt_learned(&store).await;

        let now = Utc::now();
        let bare = ExecStats {
            start: now,
            finish: now,
            duration: chrono::TimeDelta::milliseconds(1000),
            node_stats: [
                ("out_a".to_string(), node_stats(None)),
                ("out_b".to_string(), node_stats(None)),
                ("joined".to_string(), node_stats(Some(unexecuted_view_plan()))),
            ]
            .into_iter()
            .collect(),
            system_samples: Vec::new(),
        };
        let rows = second
            .ranking_learned(&learned_dag(&consumers), &bare)
            .expect("the stored constants are enough on their own");
        assert!((rows[0].total_cpu_time_s - 9.0).abs() < 1e-9);
    }

    // ---------------------------------------------------------------------
    // DupAttribution
    // ---------------------------------------------------------------------

    /// A DAG with a branch-point View both of whose consumers compute all of
    /// it, ranked against a live engine. `Cardinality` rather than the default
    /// `LearnedCost` for the region cost model, so the ranking is a function of
    /// the plans alone and does not depend on what the fixture happened to
    /// measure.
    #[tokio::test]
    async fn dup_attribution_ranks_a_branch_point_by_what_its_copies_cost() {
        let conn = in_memory_conn().await;
        conn.execute(
            "CREATE TABLE raw AS SELECT i AS id, i % 7 AS g, i * 1.5 AS amt \
             FROM range(1000) t(i)"
                .to_string(),
        )
        .await
        .unwrap();
        let engine = Arc::new(SimpleEngine::new(Arc::clone(&conn)).unwrap());
        let mut pass: HMPPass<DuckDBConnection, SimpleEngine<DuckDBConnection>> = HMPPass::new(
            Arc::clone(&conn),
            engine,
            false,
            1,
            1.0,
            None,
            None,
            false,
            HmpCostMethod::DupAttribution,
            SubtreeCostMethod::Cardinality,
            false,
            2,
            false,
            true,
            crate::opt::common::DEFAULT_BUDGET_EPS,
            Default::default(),
        );
        pass.cost_method = HmpCostMethod::DupAttribution;

        let shared = "SELECT g, sum(amt) AS total FROM raw GROUP BY g";
        // `lonely` is a View too, and an expensive one, but only one node
        // reads it -- so it must not appear in a ranking of duplication.
        let lonely = "SELECT id, amt * 2 AS doubled FROM raw";
        let dag = make_dag(vec![
            node("shared", shared, MaterializeMode::View, &[]),
            node("lonely", lonely, MaterializeMode::View, &[]),
            node(
                "out_a",
                "SELECT count(*) AS n FROM shared",
                MaterializeMode::Table,
                &["shared"],
            ),
            node(
                "out_b",
                "SELECT max(total) AS biggest FROM shared",
                MaterializeMode::Table,
                &["shared"],
            ),
            node(
                "out_c",
                "SELECT sum(doubled) AS s FROM lonely",
                MaterializeMode::Table,
                &["lonely"],
            ),
        ]);

        // What the run would have collected: each View's own EXPLAIN plan.
        let mut stats_map = HashMap::new();
        for (id, sql) in [("shared", shared), ("lonely", lonely)] {
            let plan = conn.explain(sql).await.unwrap().unwrap();
            stats_map.insert(id.to_string(), node_stats(Some(plan)));
        }
        let now = Utc::now();
        let stats = ExecStats {
            start: now,
            finish: now,
            duration: chrono::TimeDelta::zero(),
            node_stats: stats_map,
            system_samples: Vec::new(),
        };

        let rows = pass
            .ranking_dup(&dag, &stats)
            .await
            .expect("the engine planned both consumers of `shared`");

        assert_eq!(
            rows.iter().map(|r| r.node.as_str()).collect::<Vec<_>>(),
            vec!["shared"],
            "a View with one consumer duplicates nothing and must not be ranked"
        );
        assert!(rows[0].total_cpu_time_s > 0.0, "{:?}", rows[0]);
        // The per-consumer breakdown reaches the report, because a ranking
        // nobody can check against the plans behind it is a ranking taken on
        // faith.
        assert_eq!(rows[0].matched.len(), 2, "{:?}", rows[0].matched);
        assert!(
            rows[0].matched.iter().any(|m| m.starts_with("out_a"))
                && rows[0].matched.iter().any(|m| m.starts_with("out_b")),
            "{:?}",
            rows[0].matched
        );
    }

    /// Attribution no longer depends on the run having collected plans.
    ///
    /// It used to: the View's own plan was the `own` term and had to come from
    /// `stats.node_stats`, so a run without plans declined outright. The set
    /// coster EXPLAINs the build-once probe itself, so the only thing a missing
    /// plan now costs is the cardinality normalization.
    #[tokio::test]
    async fn dup_attribution_still_measures_when_the_run_carried_no_plans() {
        let mut pass = test_pass(2).await;
        pass.cost_method = HmpCostMethod::DupAttribution;
        pass.dup_cost_model = SubtreeCostMethod::Cardinality;

        let dag = make_dag(vec![
            node("v", "SELECT 1 AS x", MaterializeMode::View, &[]),
            node("a", "SELECT x FROM v", MaterializeMode::Table, &["v"]),
            node("b", "SELECT x FROM v", MaterializeMode::Table, &["v"]),
        ]);
        let now = Utc::now();
        let stats = ExecStats {
            start: now,
            finish: now,
            duration: chrono::TimeDelta::zero(),
            node_stats: HashMap::new(),
            system_samples: Vec::new(),
        };

        let rows = pass
            .ranking_dup(&dag, &stats)
            .await
            .expect("the set coster EXPLAINs what it needs, so this still measures");
        assert!(
            rows.iter().all(|r| r.cardinality.is_none()),
            "with no stored plan there is nothing to normalize by: {rows:?}"
        );
        // And the dispatcher still works from here.
        let _ = pass.ranking_for(&dag, &stats).await;
    }

    /// The learned constants are a cost model `DupAttribution` uses too, so a
    /// pass configured that way must read and write them like `LearnedCost`
    /// does -- and must not when it is pricing regions some other way.
    #[tokio::test]
    async fn dup_attribution_shares_the_learned_constants_only_when_it_prices_with_them() {
        let mut pass = test_pass(2).await;
        pass.cost_method = HmpCostMethod::DupAttribution;

        pass.dup_cost_model = SubtreeCostMethod::LearnedCost;
        assert!(pass.uses_learned_constants());

        pass.dup_cost_model = SubtreeCostMethod::Cardinality;
        assert!(!pass.uses_learned_constants());
    }

    #[tokio::test]
    async fn learned_cost_declines_rather_than_ranking_nothing_when_it_has_learned_nothing() {
        let mut pass = test_pass(2).await;
        pass.cost_method = HmpCostMethod::LearnedCost;
        // Consumers with no plans at all: nothing to fit a constant to.
        let dag = learned_dag(&["out_a", "out_b"]);
        let now = Utc::now();
        let stats = ExecStats {
            start: now,
            finish: now,
            duration: chrono::TimeDelta::milliseconds(1000),
            node_stats: [
                ("out_a".to_string(), node_stats(None)),
                ("out_b".to_string(), node_stats(None)),
                ("joined".to_string(), node_stats(Some(unexecuted_view_plan()))),
            ]
            .into_iter()
            .collect(),
            system_samples: Vec::new(),
        };

        assert!(pass.ranking_learned(&dag, &stats).is_none());
        // And `ranking_for` falls through to leaf-set matching rather than
        // returning a ranking of nothing.
        let _ = pass.ranking_for(&dag, &stats).await;
    }

    #[tokio::test]
    async fn leafset_declines_rather_than_ranking_nothing_when_plans_name_no_relation() {
        // A plan format that does not carry relation names must fall back, not
        // report that the DAG has no duplication.
        let mut pass = test_pass(2).await;
        pass.cost_method = HmpCostMethod::Leafset;
        let dag = leafset_dag();
        let mut stats = leafset_stats(1000);
        let bare = Some(
            r#"{"operator_name":"HASH_JOIN","operator_timing":1.0,
                "extra_info":{"Estimated Cardinality":10},"children":[]}"#
                .to_string(),
        );
        for id in ["out_a", "out_b"] {
            stats.node_stats.get_mut(id).unwrap().plan = bare.clone();
        }
        assert!(pass.ranking_leafset(&dag, &stats).is_none());
        // And `ranking_for` takes the fallback rather than returning nothing.
        let _ = pass.ranking_for(&dag, &stats).await;
    }
    // End-to-end smoke test for the search, driven through the step interface:
    // the batch driver supplies the executions, and HMP proposes a candidate
    // before each and scores it after. The search must converge, must stop at
    // its run budget, and must trial candidates in the order it priced them --
    // the whole point of pricing them before spending runs.
    #[tokio::test]
    async fn search_respects_the_run_budget_and_trials_in_priced_order() {
        let conn = in_memory_conn().await;
        conn.execute(
            "CREATE TABLE orders AS SELECT range AS order_id, range % 4 AS region \
             FROM range(500)"
                .to_string(),
        )
        .await
        .unwrap();

        let mut dag = make_dag(vec![
            node(
                "heavy_a",
                "SELECT order_id, region FROM orders WHERE region = 0",
                MaterializeMode::View,
                &[],
            ),
            node(
                "heavy_b",
                "SELECT order_id, region FROM orders WHERE region = 1",
                MaterializeMode::View,
                &[],
            ),
            node("sink_a1", "SELECT * FROM heavy_a", MaterializeMode::Table, &["heavy_a"]),
            node("sink_a2", "SELECT * FROM heavy_a", MaterializeMode::Table, &["heavy_a"]),
            node("sink_b1", "SELECT * FROM heavy_b", MaterializeMode::Table, &["heavy_b"]),
            node("sink_b2", "SELECT * FROM heavy_b", MaterializeMode::Table, &["heavy_b"]),
        ]);

        let engine = Arc::new(
            SimpleEngine::new(Arc::clone(&conn))
                .unwrap()
                .with_profiling(ProfilingConfig {
                    collect_plans: true,
                    ..Default::default()
                }),
        );

        let config = OptimizerConfig::default()
            .with_all_disabled()
            .with_hmp_pass()
            .with_hmp_max_runs(4)
            .with_hmp_top_cpu_time(1.0)
            .with_hmp_use_pushdown(false)
            .with_hmp_cost_method(HmpCostMethod::DupAttribution)
            .with_hmp_dup_cost_model(SubtreeCostMethod::Cardinality)
            .with_hmp_search_budget(8);

        let stores = MemoryStoreFactory::open().unwrap();
        let mut optimizer = Optimizer::new_with_config(conn, engine, config);
        let report = optimizer
            .run(&mut dag, "dag-1", "search", 1, &stores)
            .await
            .unwrap();

        let hmp = report.pass("HMPPass").expect("HMP should have reported");
        assert!(
            hmp.dag_runs_used <= 4,
            "the search must not exceed its run budget, spent {}",
            hmp.dag_runs_used
        );
        assert!(hmp.dag_runs_used >= 1, "the baseline alone is one run");

        let PassDetail::Hmp(detail) = &hmp.detail else {
            panic!("HMP must report HMP detail");
        };
        assert!(
            detail.candidates_costed > 0,
            "the search should have priced candidates before spending runs"
        );
        assert!(
            detail.candidates_costed <= 8,
            "costing must stay inside the search budget, priced {}",
            detail.candidates_costed
        );

        // Every combo that was actually trialled must appear in the order the
        // pricing put them in, and never before a combo priced above it.
        let trialled: Vec<&Vec<String>> = hmp
            .iterations
            .iter()
            .map(|it| &it.combo)
            .filter(|c| !c.is_empty())
            .collect();
        assert!(
            !trialled.is_empty(),
            "at least one candidate should have been trialled and named"
        );
    }

    #[tokio::test]
    async fn a_cancelled_trial_is_rejected_rather_than_re_proposed() {
        // The failure this guards against: a candidate cancelled at its budget
        // leaves `in_flight` set, the next `Before` step re-installs the very
        // same combo, and the search spends its whole run budget on one bad
        // candidate -- forever, if the candidate is reliably slow.
        let store = MemoryStore::open("hmp").unwrap();
        let mut pass = test_pass(2).await;
        let conn = Arc::clone(&pass.conn);
        let engine = Arc::clone(&pass.engine);

        pass.register(&RegisterContext {
            store: &store,
            dag_id: "dag-1",
            dag_name: "pipeline",
        })
        .await
        .unwrap();

        let mut state = pass.load_state(&store, "dag-1").await.unwrap().unwrap();
        state.phase = "searching".to_string();
        state.best_ms = 1000;
        state.best_combo = vec!["kept".to_string()];
        state.working_set = vec!["a".to_string(), "b".to_string()];
        state.candidates_built = true;
        state.in_flight = Some(InFlight {
            combo: vec!["a".to_string()],
            sig: "sig-a".to_string(),
        });
        pass.save_state(&store, "dag-1", &state).await.unwrap();

        let mut dag = make_dag(vec![node("x", "SELECT 1", MaterializeMode::Table, &[])]);
        let mut ctx = StepContext {
            store: &store,
            conn,
            engine,
            dag: &mut dag,
            dag_id: "dag-1",
            dag_name: "pipeline",
            dag_version: 1,
            side: StepPhase::After,
            run: Some(crate::opt::RunContext {
                run_id: "r1".into(),
                run_group_id: "g1".into(),
                run_phase: crate::opt::run_phase::MEASURE.into(),
                rep_index: 0,
                // A measured run with no stats: cancelled at its budget.
                stats: None,
                resumed: None,
            }),
        };
        let outcome = pass.step(&mut ctx).await.unwrap();
        assert!(matches!(outcome, StepOutcome::Idle));

        let after = pass.load_state(&store, "dag-1").await.unwrap().unwrap();
        assert!(
            after.in_flight.is_none(),
            "the cancelled candidate is still in flight and will be re-proposed"
        );
        assert_eq!(after.runs_used, 1, "the cancelled run still cost a run");
        // Recorded as a lower bound -- at least the budget -- and never as a
        // new best.
        assert_eq!(after.best_ms, 1000);
        assert_eq!(after.best_combo, vec!["kept".to_string()]);
        assert!(
            after.tried_sigs.iter().any(|sig| sig == "sig-a"),
            "the cancelled combo must stay recorded as tried, or the next \
             Before step re-proposes it"
        );
        assert_eq!(
            after.iterations.last().unwrap().outcome.as_deref(),
            Some("cancelled")
        );
    }

    /// A cancelled iteration must carry what it actually cost, in parts.
    ///
    /// `runtime_ms` there is the censoring level -- "at least this slow" --
    /// and reporting it as the iteration's cost understates a cancelled run by
    /// exactly the resume nobody counted.
    #[tokio::test]
    async fn a_cancelled_iteration_records_the_trial_the_overhead_and_the_resume() {
        let store = MemoryStore::open("hmp").unwrap();
        let mut pass = test_pass(2).await;
        let dag = make_dag(vec![
            node("v", "SELECT 1 AS x", MaterializeMode::View, &[]),
            node("a", "SELECT x FROM v", MaterializeMode::Table, &["v"]),
            node("b", "SELECT x FROM v", MaterializeMode::Table, &["v"]),
        ]);
        pass.register(&RegisterContext {
            store: &store,
            dag_id: "dag-1",
            dag_name: "pipeline",
        })
        .await
        .unwrap();

        // A search already past its baseline, with a candidate in flight.
        let mut state = pass.load_state(&store, "dag-1").await.unwrap().unwrap();
        state.phase = "searching".to_string();
        state.baseline_ms = 1000;
        state.best_ms = 1000;
        state.working_set = vec!["v".to_string()];
        state.candidates_built = true;
        state.runs_used = 1;
        state.in_flight = Some(InFlight {
            combo: vec!["v".to_string()],
            sig: "sig-1".to_string(),
        });
        pass.save_state(&store, "dag-1", &state).await.unwrap();

        let mut working = dag.clone();
        let mut ctx = StepContext {
            store: &store,
            conn: Arc::clone(&pass.conn),
            engine: Arc::clone(&pass.engine),
            dag: &mut working,
            dag_id: "dag-1",
            dag_name: "pipeline",
            dag_version: 1,
            side: StepPhase::After,
            run: Some(crate::opt::RunContext {
                run_id: "r1".into(),
                run_group_id: "g1".into(),
                run_phase: crate::opt::run_phase::MEASURE.into(),
                rep_index: 0,
                // Cancelled: no stats, so no measurement...
                stats: None,
                // ...but the run still happened, in three parts.
                resumed: Some(crate::opt::ResumeTiming {
                    trial_ms: 1000,
                    overhead_ms: 120,
                    resume_ms: 800,
                    trial_node_time_ms: 2400,
                    resume_node_time_ms: 1500,
                }),
            }),
        };
        pass.step(&mut ctx).await.unwrap();

        let after = pass.load_state(&store, "dag-1").await.unwrap().unwrap();
        let it = after
            .iterations
            .last()
            .expect("the cancelled trial was filed");
        assert_eq!(it.outcome.as_deref(), Some("cancelled"));
        assert_eq!(it.trial_ms, Some(1000));
        assert_eq!(it.resume_overhead_ms, Some(120));
        assert_eq!(it.resume_ms, Some(800));
        // What it cost, against what the search is allowed to believe about
        // the candidate's speed. The two are different numbers and the
        // distinction is the whole point of keeping both.
        assert_eq!(it.total_ms(), 1920);
        assert_eq!(it.runtime_ms, 1000);
        // Both halves of the run did database work, and both are charged.
        assert_eq!(it.node_time_ms, Some(3900));
    }

    /// An iteration that finished reports the database time of its own run and
    /// no resume breakdown -- there was no resume to break down.
    #[tokio::test]
    async fn a_completed_iteration_records_node_time_and_no_resume() {
        let store = MemoryStore::open("hmp").unwrap();
        let mut pass = test_pass(2).await;
        let dag = make_dag(vec![
            node("v", "SELECT 1 AS x", MaterializeMode::View, &[]),
            node("a", "SELECT x FROM v", MaterializeMode::Table, &["v"]),
            node("b", "SELECT x FROM v", MaterializeMode::Table, &["v"]),
        ]);
        pass.register(&RegisterContext {
            store: &store,
            dag_id: "dag-1",
            dag_name: "pipeline",
        })
        .await
        .unwrap();

        let now = Utc::now();
        let mut node_stats_map = HashMap::new();
        for (id, ms) in [("a", 700i64), ("b", 500i64)] {
            node_stats_map.insert(
                id.to_string(),
                crate::executor::NodeStats {
                    start: now,
                    finish: now,
                    duration: chrono::TimeDelta::milliseconds(ms),
                    plan: None,
                    rows_produced: None,
                },
            );
        }
        let stats = ExecStats {
            start: now,
            finish: now,
            duration: chrono::TimeDelta::milliseconds(900),
            node_stats: node_stats_map,
            system_samples: Vec::new(),
        };

        let mut working = dag.clone();
        let mut ctx = StepContext {
            store: &store,
            conn: Arc::clone(&pass.conn),
            engine: Arc::clone(&pass.engine),
            dag: &mut working,
            dag_id: "dag-1",
            dag_name: "pipeline",
            dag_version: 1,
            side: StepPhase::After,
            run: Some(crate::opt::RunContext {
                run_id: "r1".into(),
                run_group_id: "g1".into(),
                run_phase: crate::opt::run_phase::MEASURE.into(),
                rep_index: 0,
                stats: Some(stats),
                resumed: None,
            }),
        };
        pass.step(&mut ctx).await.unwrap();

        let after = pass.load_state(&store, "dag-1").await.unwrap().unwrap();
        let it = after.iterations.last().expect("the baseline was filed");
        // 700 + 500 of node time inside a 900ms wall clock: the two nodes ran
        // concurrently, which is exactly why both numbers are kept.
        assert_eq!(it.node_time_ms, Some(1200));
        assert_eq!(it.runtime_ms, 900);
        assert_eq!(it.total_ms(), 900);
        assert_eq!(it.trial_ms, None);
        assert_eq!(it.resume_ms, None);
    }

    #[tokio::test]
    async fn a_cancelled_baseline_is_not_recorded_as_one() {
        // The baseline is the DAG as it stands, not a candidate. Recording a
        // censored baseline would give every later trial a number no run
        // produced to beat.
        let store = MemoryStore::open("hmp").unwrap();
        let mut pass = test_pass(2).await;
        let conn = Arc::clone(&pass.conn);
        let engine = Arc::clone(&pass.engine);
        pass.register(&RegisterContext {
            store: &store,
            dag_id: "dag-1",
            dag_name: "pipeline",
        })
        .await
        .unwrap();

        let mut dag = make_dag(vec![node("x", "SELECT 1", MaterializeMode::Table, &[])]);
        let mut ctx = StepContext {
            store: &store,
            conn,
            engine,
            dag: &mut dag,
            dag_id: "dag-1",
            dag_name: "pipeline",
            dag_version: 1,
            side: StepPhase::After,
            run: Some(crate::opt::RunContext {
                run_id: "r1".into(),
                run_group_id: "g1".into(),
                run_phase: crate::opt::run_phase::MEASURE.into(),
                rep_index: 0,
                stats: None,
                resumed: None,
            }),
        };
        pass.step(&mut ctx).await.unwrap();
        let after = pass.load_state(&store, "dag-1").await.unwrap().unwrap();
        assert_eq!(after.phase, "baseline");
        assert_eq!(after.baseline_ms, 0);
        assert_eq!(after.runs_used, 0);
    }

    #[tokio::test]
    async fn a_budget_only_exists_once_there_is_something_to_be_worse_than() {
        let pass = test_pass(2).await;
        let mut state = HmpState::new();
        assert_eq!(pass.budget(&state), None, "nothing has been measured yet");
        state.best_ms = 1000;
        // Exactly the incumbent: the search only asks whether a candidate is
        // faster, so a trial that reaches the incumbent's time has already
        // answered no and is stopped there.
        assert_eq!(pass.budget(&state), Some(1000));
    }

    #[tokio::test]
    async fn a_zero_budget_eps_is_honoured_rather_than_read_as_unset() {
        // `0.0` is the default and a deliberate setting. Coercing it to a
        // fallback -- which an `if eps > 0.0` check would -- would silently
        // restore the slack this is meant to remove.
        let mut pass = test_pass(2).await;
        pass.budget_eps = 0.0;
        let mut state = HmpState::new();
        state.best_ms = 800;
        assert_eq!(pass.budget(&state), Some(800));
    }

    // A continuous optimization's whole premise is that its search survives
    // between runs, so state written by one step has to be what the next one
    // reads. Register, step, and check the search actually moved.
    /// Costing can come back with nothing at all -- a connector that cannot
    /// EXPLAIN, or a run whose plans were never collected. The search must fall
    /// back to trying combinations unpriced rather than hand back an empty
    /// list, because an empty candidate list promotes on the next step and so
    /// turns the pass off without saying so.
    #[tokio::test]
    async fn an_uncostable_search_falls_back_to_plain_combinations() {
        let pass = test_pass(32).await;
        let dag = make_dag(vec![
            node("a", "SELECT 1", MaterializeMode::View, &[]),
            node("b", "SELECT 2", MaterializeMode::View, &[]),
            node("t1", "SELECT * FROM a", MaterializeMode::Table, &["a"]),
            node("t2", "SELECT * FROM b", MaterializeMode::Table, &["b"]),
        ]);
        // The learned model has been fitted to nothing, so the region coster
        // declines to price anything it is handed.

        let working_set = vec!["a".to_string(), "b".to_string()];
        let candidates = pass.build_candidates(&dag, &working_set).await;

        assert!(
            !candidates.is_empty(),
            "an uncostable search must still offer candidates, or HMP silently \
             stops optimizing"
        );
        assert!(
            candidates.iter().all(|c| c.partial),
            "unpriced candidates must be flagged as such rather than claim a cost"
        );
        assert!(
            candidates.iter().take(2).all(|c| c.combo.len() == 1),
            "the fallback tries combinations smallest-first: {candidates:?}"
        );
        assert!(
            candidates.iter().any(|c| c.combo.len() == 2),
            "and still reaches the pairs: {candidates:?}"
        );
    }

    /// A state row written before the candidate list existed must still
    /// decode, and must not read as an exhausted search.
    ///
    /// `load_state` turns a decode failure into a hard error, so without the
    /// struct-level `#[serde(default)]` every search in flight at upgrade time
    /// would break; and without `candidates_built` a search that decoded fine
    /// would still promote immediately, because an empty candidate list is
    /// indistinguishable from a finished one.
    #[test]
    fn a_legacy_state_row_decodes_and_is_not_mistaken_for_exhausted() {
        // Shaped like a pre-change persisted state, mid-search: the cursors
        // that no longer exist, and none of the fields that replaced them.
        let legacy = r#"{
            "phase": "searching",
            "baseline_ms": 1200,
            "best_ms": 900,
            "best_combo": ["a"],
            "working_set": ["a", "b"],
            "baseline_scores": {"a": 2.0, "b": 1.0},
            "working_order": ["a", "b"],
            "runs_used": 2,
            "iterations": [],
            "tried_sigs": ["sig-a"],
            "k": 2,
            "combo_index": 1,
            "round_score_sums": {},
            "round_score_counts": {},
            "beams": [{"combo": ["a"], "runtime_ms": 900}],
            "node_cursor": 1,
            "beam_cursor": 0,
            "proposals": [],
            "tried_combos": {"sig-a": 900},
            "in_flight": null
        }"#;

        let state: HmpState =
            serde_json::from_str(legacy).expect("a state row from an older build must decode");

        assert_eq!(state.phase, "searching");
        assert_eq!(state.best_ms, 900);
        assert_eq!(state.tried_sigs, vec!["sig-a".to_string()]);
        assert!(
            !state.candidates_built,
            "the row predates the candidate list, so the search has not enumerated yet"
        );
        assert!(state.candidates.is_empty());
    }

    #[tokio::test]
    async fn search_state_persists_between_steps() {
        let store = MemoryStore::open("hmp").unwrap();
        let pass = test_pass(2).await;

        let ctx = RegisterContext {
            store: &store,
            dag_id: "dag-1",
            dag_name: "pipeline",
        };
        let registration = pass.register(&ctx).await.unwrap();
        assert_eq!(
            registration.map(|r| r.tables),
            Some(vec![
                "opt_hmp_state".to_string(),
                "opt_hmp_trials".to_string(),
                "opt_hmp_learned_cost".to_string(),
            ]),
            "HMP keeps state, so it must say which tables hold it"
        );

        let state = pass.load_state(&store, "dag-1").await.unwrap().unwrap();
        assert_eq!(state.phase, "baseline", "a fresh search has measured nothing");

        // Registering again must not restart a search in progress -- a server
        // restart re-registers everything a DAG had.
        let mut advanced = state.clone();
        advanced.phase = "searching".to_string();
        advanced.best_ms = 1234;
        pass.save_state(&store, "dag-1", &advanced).await.unwrap();
        pass.register(&ctx).await.unwrap();

        let reloaded = pass.load_state(&store, "dag-1").await.unwrap().unwrap();
        assert_eq!(reloaded.phase, "searching");
        assert_eq!(reloaded.best_ms, 1234);

        // Deregistering takes the state with it.
        pass.deregister(&ctx).await.unwrap();
        assert!(
            pass.load_state(&store, "dag-1").await.unwrap().is_none(),
            "deregistering must not leave the search behind"
        );
    }

    // Two DAGs registered on HMP share its tables, so deregistering one must
    // not take the other's search with it.
    #[tokio::test]
    async fn deregistering_one_dag_leaves_another_dags_search_alone() {
        let store = MemoryStore::open("hmp").unwrap();
        let pass = test_pass(2).await;

        for dag_id in ["dag-1", "dag-2"] {
            pass.register(&RegisterContext {
                store: &store,
                dag_id,
                dag_name: dag_id,
            })
            .await
            .unwrap();
        }

        pass.deregister(&RegisterContext {
            store: &store,
            dag_id: "dag-1",
            dag_name: "dag-1",
        })
        .await
        .unwrap();

        assert!(pass.load_state(&store, "dag-1").await.unwrap().is_none());
        assert!(
            pass.load_state(&store, "dag-2").await.unwrap().is_some(),
            "the other DAG's search must survive"
        );
    }

    // The namespace check is what keeps an optimization out of the tables
    // holding every run, plan and connection credential dee has recorded.
    #[tokio::test]
    async fn an_optimization_cannot_reach_outside_its_own_tables() {
        let store = MemoryStore::open("hmp").unwrap();
        let error = store
            .execute("DROP TABLE IF EXISTS connections", &[])
            .await
            .expect_err("a write outside the namespace must be refused");
        assert!(
            error.to_string().contains("connections"),
            "the error should name the table it refused: {error}"
        );
        // Its own tables are still reachable.
        store
            .execute("CREATE TABLE IF NOT EXISTS opt_hmp_state (dag_id VARCHAR)", &[])
            .await
            .expect("its own namespace must be writable");
    }
}
