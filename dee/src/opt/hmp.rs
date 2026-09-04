use async_trait::async_trait;
use log::{debug, info, warn};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    collections::{HashMap, HashSet},
    fs,
    marker::PhantomData,
    sync::Arc,
};

use crate::{
    connectors::Connector,
    dag::MaterializeMode,
    executor::{ExecStats, Executor},
    plan::OpKey,
    opt::{
        Dag, Optimization, OptimizerError, OptimizerConfig,
        common::{dialect_for_db, make_temp},
        leafset::{PlanArena, ViewRegionRequest, attribute_chain, has_top_level_group_by},
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

/// Strategy HMP uses to search through the node ranking when deciding
/// which VIEWs to materialize.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HMPStrategy {
    /// Walk the node ranking, trying all k-sized combinations smallest-first
    /// (singles, pairs, triples, ...). This is the default / original behavior.
    #[default]
    Breadth,
    /// Walk the node ranking sequentially, committing each materialization
    /// that improves performance before trying the next node down the ranking.
    Greedy,
}

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
}

impl HmpCostMethod {
    pub fn as_str(&self) -> &'static str {
        match self {
            HmpCostMethod::Leafset => "leafset",
            HmpCostMethod::Signature => "signature",
            HmpCostMethod::NodeTime => "node_time",
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
            other => Err(format!(
                "unknown hmp cost method '{other}'; expected leafset, signature or node_time"
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    /// Each candidate's score from the baseline run, the fallback for a node
    /// no later trial observed.
    baseline_scores: HashMap<String, f64>,
    /// `working_set` in the order the search is currently trying it, re-derived
    /// between rounds from the freshest evidence.
    working_order: Vec<String>,
    /// DAG executions this search has consumed, baseline included.
    runs_used: usize,
    iterations: Vec<IterationStat>,
    /// Signatures of trial DAGs already measured, so two combos that reduce to
    /// the same DAG are not paid for twice.
    tried_sigs: Vec<String>,

    // --- Breadth cursor -------------------------------------------------
    /// Combination size currently being enumerated.
    k: usize,
    /// Position within the size-`k` combinations of `working_order`.
    combo_index: usize,
    /// Ranking scores observed during this round, accumulated as trials are
    /// measured. Replaces holding every trial's DAG and ExecStats: the
    /// refinement only ever used them to derive these numbers, and deriving
    /// each once as it arrives is both cheaper and the only version that
    /// survives being written to a database.
    round_score_sums: HashMap<String, f64>,
    round_score_counts: HashMap<String, usize>,

    // --- Greedy cursor --------------------------------------------------
    beams: Vec<BeamState>,
    /// Position in `working_order` -- the node currently being considered.
    node_cursor: usize,
    /// Which of `beams` is being expanded at `node_cursor`.
    beam_cursor: usize,
    /// Beams proposed so far at `node_cursor`, pruned to `beam_width` when the
    /// node is finished.
    proposals: Vec<BeamState>,
    /// Measured runtime by trial-DAG signature, so a combo the beam search
    /// re-proposes is scored from the existing measurement rather than re-run.
    tried_combos: HashMap<String, i64>,

    /// The candidate a `Before` step rewrote the DAG into and an `After` step
    /// is expected to report on.
    in_flight: Option<InFlight>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct InFlight {
    combo: Vec<String>,
    sig: String,
}

impl HmpState {
    fn new() -> Self {
        Self {
            phase: "baseline".to_string(),
            baseline_ms: 0,
            best_ms: i64::MAX,
            best_combo: Vec::new(),
            working_set: Vec::new(),
            baseline_scores: HashMap::new(),
            working_order: Vec::new(),
            runs_used: 0,
            iterations: Vec::new(),
            tried_sigs: Vec::new(),
            k: 1,
            combo_index: 0,
            round_score_sums: HashMap::new(),
            round_score_counts: HashMap::new(),
            beams: Vec::new(),
            node_cursor: 0,
            beam_cursor: 0,
            proposals: Vec::new(),
            tried_combos: HashMap::new(),
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
    /// Strategy for searching through the node ranking.
    strategy: HMPStrategy,
    /// How a View's cost is read off the run's plans.
    cost_method: HmpCostMethod,
    /// Cancel a trial once it has overrun the incumbent and finish the run
    /// under the incumbent instead, rather than measuring every candidate to
    /// completion.
    resume_trials: bool,
    /// Fraction by which a trial may overrun the incumbent before it is cut
    /// short. Only meaningful when `resume_trials` is set.
    budget_eps: f64,
    /// Run the PushdownPass before evaluating each candidate materialization
    /// combination, for more accurate cost measurements.
    use_pushdown: bool,
    /// Number of hypotheses the `Greedy` strategy's beam search keeps alive
    /// at each step. Unused by the `Breadth` strategy.
    beam_width: usize,
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
    strategy: HMPStrategy,
    beam_width: usize,
}

/// A single hypothesis in the `Greedy` strategy's beam search: a
/// materialization combo and the runtime it measured at.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BeamState {
    combo: Vec<String>,
    runtime_ms: i64,
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
#[derive(Serialize, Debug, Clone)]
struct NodeRankingRow {
    rank: usize,
    node: String,
    total_cpu_time_s: f64,
    /// Estimated cardinality of the View's own EXPLAIN plan, when available.
    cardinality: Option<f64>,
    /// The value nodes are ranked by: `total_cpu_time_s`, or (when
    /// `--hmp-normalize-with-cardinality` is set) `total_cpu_time_s` divided
    /// by `cardinality`.
    ranking_score: f64,
    /// Leaf-set matching only: the base relations this View reads, and the
    /// consumer plan operators its region was matched to.
    ///
    /// Empty under the other costing methods, which have no notion of either.
    /// They are the only way to tell a good attribution from a lucky one, which
    /// is why they reach the explain report rather than staying in a log line.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    leaves: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    matched: Vec<String>,
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
        strategy: HMPStrategy,
        cost_method: HmpCostMethod,
        use_pushdown: bool,
        beam_width: usize,
        profile_iterations: bool,
        resume_trials: bool,
        budget_eps: f64,
    ) -> Self {
        Self {
            conn,
            engine,
            downstream_cost,
            max_runs: max_runs.max(1),
            show_operators,
            show_nodes,
            normalize_with_cardinality,
            strategy,
            cost_method,
            use_pushdown,
            beam_width: beam_width.max(1),
            profile_iterations,
            resume_trials,
            budget_eps: if budget_eps > 0.0 {
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

/// All k-sized combinations of `items`, preserving relative order.
fn combinations(items: &[String], k: usize) -> Vec<Vec<String>> {
    fn helper(items: &[String], start: usize, k: usize, combo: &mut Vec<String>, out: &mut Vec<Vec<String>>) {
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
            config.hmp_strategy,
            config.hmp_cost_method,
            config.hmp_use_pushdown,
            config.hmp_beam_width,
            config.profile_iterations,
            config.trial_resume,
            config.trial_budget_eps,
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

    /// The ranking a run's plans imply, as `(node, score)`.
    fn ranking_for(&self, dag: &Dag, stats: &ExecStats) -> Vec<NodeRankingRow> {
        if self.cost_method == HmpCostMethod::NodeTime {
            return Self::ranking_from_node_times(dag, stats);
        }
        if self.cost_method == HmpCostMethod::Leafset {
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

    /// Reorder `nodes` by the scores accumulated during the round that just
    /// finished, falling back to the baseline score for a node no trial saw.
    ///
    /// The same refinement the breadth search always did between combination
    /// sizes; it reads accumulated sums rather than replaying stored
    /// observations, because the sums are what it computed from them anyway.
    fn reorder_by(
        nodes: &[String],
        sums: &HashMap<String, f64>,
        counts: &HashMap<String, usize>,
        baseline_scores: &HashMap<String, f64>,
    ) -> Vec<String> {
        let mut ordered = nodes.to_vec();
        ordered.sort_by(|a, b| {
            let score_of = |n: &String| -> f64 {
                match counts.get(n) {
                    Some(&count) if count > 0 => {
                        sums.get(n).copied().unwrap_or(0.0) / count as f64
                    }
                    _ => baseline_scores.get(n).copied().unwrap_or(0.0),
                }
            };
            score_of(b)
                .partial_cmp(&score_of(a))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        ordered
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

    /// The next combination the breadth search should try, advancing the
    /// cursor past any that reduce to a DAG already measured.
    ///
    /// `None` means the enumeration is exhausted.
    fn next_breadth_combo(&self, state: &mut HmpState) -> Option<Vec<String>> {
        loop {
            if state.k > state.working_order.len() {
                return None;
            }
            let combos = combinations(&state.working_order, state.k);
            if state.combo_index >= combos.len() {
                // Round finished: refine the order from what this round
                // measured, then start on the next size.
                state.k += 1;
                state.combo_index = 0;
                state.working_order = Self::reorder_by(
                    &state.working_order,
                    &state.round_score_sums,
                    &state.round_score_counts,
                    &state.baseline_scores,
                );
                state.round_score_sums.clear();
                state.round_score_counts.clear();
                continue;
            }
            let combo = combos[state.combo_index].clone();
            state.combo_index += 1;
            return Some(combo);
        }
    }

    /// The next combination the greedy beam search should try.
    fn next_greedy_combo(&self, state: &mut HmpState) -> Option<Vec<String>> {
        if state.beams.is_empty() {
            state.beams.push(BeamState {
                combo: Vec::new(),
                runtime_ms: state.best_ms,
            });
            state.proposals = state.beams.clone();
        }
        loop {
            if state.node_cursor >= state.working_order.len() {
                return None;
            }
            let node_id = state.working_order[state.node_cursor].clone();

            if state.beam_cursor >= state.beams.len() {
                // Every beam has been expanded at this node: prune to the
                // width and move on. Beams carried forward unchanged are among
                // the proposals, which is what lets the search drop a node it
                // committed to earlier.
                let mut proposals = std::mem::take(&mut state.proposals);
                proposals.sort_by_key(|p| p.runtime_ms);
                proposals.dedup_by(|a, b| a.combo == b.combo);
                proposals.truncate(self.beam_width.max(1));
                state.beams = proposals;
                state.node_cursor += 1;
                state.beam_cursor = 0;
                state.proposals = state.beams.clone();
                continue;
            }

            let beam = state.beams[state.beam_cursor].clone();
            state.beam_cursor += 1;
            if beam.combo.contains(&node_id) {
                continue;
            }
            let mut combo = beam.combo.clone();
            combo.push(node_id);
            return Some(combo);
        }
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
                strategy: format!("{:?}", self.strategy),
                beam_width: self.beam_width,
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
            strategy: self.strategy,
            beam_width: self.beam_width,
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
                        label: describe(&combo),
                        budget_ms: self.budget(&state),
                        fallback,
                        record: Box::new(self.outcome_from(&state)),
                    });
                }

                if state.runs_used >= self.max_runs {
                    return self.promote(ctx, state).await;
                }

                // Skip candidates that reduce to a DAG already measured; each
                // costs nothing but a rewrite, and paying a DAG run for a
                // duplicate is the expensive mistake.
                loop {
                    let combo = match self.strategy {
                        HMPStrategy::Breadth => self.next_breadth_combo(&mut state),
                        HMPStrategy::Greedy => self.next_greedy_combo(&mut state),
                    };
                    let Some(combo) = combo else {
                        return self.promote(ctx, state).await;
                    };

                    let mut trial = ctx.dag.clone();
                    self.build_trial(&mut trial, &combo).await?;
                    let sig = dag_signature(&trial);
                    let fallback = self.incumbent_dag(ctx.dag, &state).await;

                    if let Some(&measured) = state.tried_combos.get(&sig) {
                        // Already measured under another combo: score this
                        // beam from that measurement instead of re-running.
                        state.proposals.push(BeamState {
                            combo: combo.clone(),
                            runtime_ms: measured,
                        });
                        continue;
                    }
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

        if state.phase == "baseline" {
            state.baseline_ms = runtime_ms;
            state.best_ms = runtime_ms;
            state.runs_used = 1;
            state.iterations.push(IterationStat {
                iteration: 1,
                runtime_ms,
                combo: Vec::new(),
                outcome: Some("baseline".to_string()),
                system_samples: if self.profile_iterations {
                    stats.system_samples.clone()
                } else {
                    Vec::new()
                },
            });

            let mut ranking = self.ranking_for(ctx.dag, stats);
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
            state.working_order = state.working_set.clone();
            self.node_rows = ranking;
            state.phase = "searching".to_string();

            debug!(
                "HMPPass baseline {runtime_ms}ms; working set of {} node(s): {:?}",
                state.working_set.len(),
                state.working_set
            );
            self.record_trial(ctx.store, ctx.dag_id, &run.run_id, &state, runtime_ms, true)
                .await?;
            self.save_state(ctx.store, ctx.dag_id, &state).await?;
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
        state.iterations.push(IterationStat {
            iteration: state.iterations.len() + 1,
            runtime_ms,
            combo: in_flight.combo.clone(),
            outcome: Some("ok".to_string()),
            system_samples: if self.profile_iterations {
                stats.system_samples.clone()
            } else {
                Vec::new()
            },
        });
        state.tried_combos.insert(in_flight.sig, runtime_ms);
        state.proposals.push(BeamState {
            combo: in_flight.combo.clone(),
            runtime_ms,
        });

        // Accumulate the ranking this trial implies, for the reordering the
        // breadth search does between combination sizes.
        for row in self.ranking_for(ctx.dag, stats) {
            *state.round_score_sums.entry(row.node.clone()).or_insert(0.0) += row.ranking_score;
            *state.round_score_counts.entry(row.node).or_insert(0) += 1;
        }

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
        // as the candidate's runtime keeps the beam and the de-duplication from
        // ever preferring it, without claiming to know how bad it was.
        let censored_ms = self.budget(&state).unwrap_or(i64::MAX);
        debug!(
            "combo {:?} produced no usable measurement; rejecting as censored (>= {censored_ms}ms)",
            in_flight.combo
        );
        state.iterations.push(IterationStat {
            iteration: state.iterations.len() + 1,
            runtime_ms: censored_ms,
            combo: in_flight.combo.clone(),
            outcome: Some("cancelled".to_string()),
            system_samples: Vec::new(),
        });
        state.tried_combos.insert(in_flight.sig, censored_ms);
        state.proposals.push(BeamState {
            combo: in_flight.combo,
            runtime_ms: censored_ms,
        });

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

        // Registering is idempotent -- a server restart re-registers what a
        // DAG already had -- so an existing search is left exactly where it
        // was rather than restarted from its baseline.
        if self.load_state(ctx.store, ctx.dag_id).await?.is_none() {
            self.save_state(ctx.store, ctx.dag_id, &HmpState::new())
                .await?;
        }

        Ok(Some(Registration::new([STATE_TABLE, TRIALS_TABLE])))
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
            for table in [STATE_TABLE, TRIALS_TABLE] {
                ctx.store
                    .execute(&format!("DROP TABLE IF EXISTS {table}"), &[])
                    .await?;
            }
        }
        Ok(Some(Registration::new([STATE_TABLE, TRIALS_TABLE])))
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
                "Search strategy",
                match data.strategy {
                    HMPStrategy::Breadth => "breadth",
                    HMPStrategy::Greedy => "greedy",
                }
                .to_string(),
            ),
            (
                "Beam width",
                match data.strategy {
                    HMPStrategy::Greedy => data.beam_width.to_string(),
                    HMPStrategy::Breadth => "n/a".to_string(),
                },
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

        let combinations_desc = match data.strategy {
            HMPStrategy::Breadth => {
                "Combinations of working-set nodes were tried smallest-first (singles, then pairs, \
                 ...) until the run budget was exhausted. Between sizes, the search order was \
                 refined using the EXPLAIN ANALYZE plans collected from the previous size's trials."
                    .to_string()
            }
            HMPStrategy::Greedy => format!(
                "A beam search (width {}) walked the node ranking, keeping the {} \
                 best-performing materialization combos alive at each step -- including the option \
                 of leaving a node out -- until the run budget was exhausted.",
                data.beam_width, data.beam_width
            ),
        };

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
        beam_width: usize,
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
            HMPStrategy::Breadth,
            // The existing tests fabricate operator-signature plans, which is
            // what this method reads.
            HmpCostMethod::Signature,
            false,
            beam_width,
            false,
            true,
            crate::opt::common::DEFAULT_BUDGET_EPS,
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
        let _ = pass.ranking_for(&dag, &stats);
    }

    // "shelf" is a materialized Table whose EXPLAIN plan roots two operators:
    // OP1 (cheap, traced only to b_view) and OP2 (expensive, traced only to
    // c_view). Baseline ranks b_view above c_view, but the fabricated trial
    // observation shows the opposite (c_view now costs more than b_view) --
    // simulating what a real trial's fresh EXPLAIN ANALYZE plans can reveal
    // once other nodes have already been materialized. `refine_node_order`
    // should re-rank to match the observation, not the stale baseline.
    #[tokio::test]
    async fn refine_node_order_promotes_node_with_higher_observed_cost() {
        let pass = test_pass(2).await;

        let shelf_plan = r#"{"operator_name":"ROOT","operator_timing":0.0,"children":[
            {"operator_name":"OP1","operator_timing":2.0,"extra_info":{"Estimated Cardinality":"1"},"children":[]},
            {"operator_name":"OP2","operator_timing":8.0,"extra_info":{"Estimated Cardinality":"1"},"children":[]}
        ]}"#;
        let b_view_plan = r#"[{"operator_name":"OP1","operator_timing":2.0,"extra_info":{"Estimated Cardinality":"1"},"children":[]}]"#;
        let c_view_plan = r#"[{"operator_name":"OP2","operator_timing":8.0,"extra_info":{"Estimated Cardinality":"1"},"children":[]}]"#;

        let dag = make_dag(vec![
            node("shelf", "SELECT 1 AS x", MaterializeMode::Table, &[]),
            node("b_view", "SELECT 1 AS x", MaterializeMode::View, &[]),
            node("c_view", "SELECT 1 AS x", MaterializeMode::View, &[]),
            node("b_sink1", "SELECT * FROM b_view", MaterializeMode::Table, &["b_view"]),
            node("b_sink2", "SELECT * FROM b_view", MaterializeMode::Table, &["b_view"]),
            node("c_sink1", "SELECT * FROM c_view", MaterializeMode::Table, &["c_view"]),
            node("c_sink2", "SELECT * FROM c_view", MaterializeMode::Table, &["c_view"]),
        ]);

        let mut node_stats_map = HashMap::new();
        node_stats_map.insert("shelf".to_string(), node_stats(Some(shelf_plan.to_string())));
        node_stats_map.insert("b_view".to_string(), node_stats(Some(b_view_plan.to_string())));
        node_stats_map.insert("c_view".to_string(), node_stats(Some(c_view_plan.to_string())));

        let exec_stats = ExecStats {
            start: Utc::now(),
            finish: Utc::now(),
            duration: chrono::TimeDelta::zero(),
            node_stats: node_stats_map,
            system_samples: vec![],
        };

        let baseline_scores: HashMap<String, f64> = [
            ("b_view".to_string(), 10.0),
            ("c_view".to_string(), 5.0),
        ]
        .into_iter()
        .collect();
        let nodes = vec!["b_view".to_string(), "c_view".to_string()];

        // The search now accumulates each trial's ranking as it is measured
        // rather than replaying stored observations, so the equivalent of one
        // observation is that trial's ranking folded into the round's sums.
        let mut sums: HashMap<String, f64> = HashMap::new();
        let mut counts: HashMap<String, usize> = HashMap::new();
        for row in pass.ranking_for(&dag, &exec_stats) {
            *sums.entry(row.node.clone()).or_insert(0.0) += row.ranking_score;
            *counts.entry(row.node).or_insert(0) += 1;
        }

        let refined = HMPPass::<DuckDBConnection, SimpleEngine<DuckDBConnection>>::reorder_by(
            &nodes,
            &sums,
            &counts,
            &baseline_scores,
        );

        assert_eq!(refined, vec!["c_view".to_string(), "b_view".to_string()]);
    }

    // Nodes absent from every observation (e.g. materialized in all trials
    // seen so far) fall back to the baseline score instead of being treated
    // as zero-cost.
    #[tokio::test]
    async fn refine_node_order_falls_back_to_baseline_for_unobserved_nodes() {

        let baseline_scores: HashMap<String, f64> = [
            ("b_view".to_string(), 10.0),
            ("c_view".to_string(), 5.0),
        ]
        .into_iter()
        .collect();
        let nodes = vec!["c_view".to_string(), "b_view".to_string()];

        // No observations at all -- order should be untouched apart from
        // falling back fully to baseline_scores (b_view still ranks first).
        let refined = HMPPass::<DuckDBConnection, SimpleEngine<DuckDBConnection>>::reorder_by(
            &nodes,
            &HashMap::new(),
            &HashMap::new(),
            &baseline_scores,
        );

        assert_eq!(refined, vec!["b_view".to_string(), "c_view".to_string()]);
    }

    // End-to-end smoke test for the Greedy strategy's beam search, now driven
    // through the step interface: the batch driver supplies the executions,
    // and HMP proposes a candidate before each and scores it after. This is
    // the test that the inversion preserved the search -- the beam still
    // converges, still stops at its budget, and the budget is still counted in
    // DAG runs.
    #[tokio::test]
    async fn greedy_beam_search_runs_and_respects_budget() {
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
            .with_hmp_strategy(HMPStrategy::Greedy)
            .with_hmp_max_runs(4)
            .with_hmp_top_cpu_time(1.0)
            .with_hmp_use_pushdown(false)
            .with_hmp_beam_width(2);

        let stores = MemoryStoreFactory::open().unwrap();
        let mut optimizer = Optimizer::new_with_config(conn, engine, config);
        let report = optimizer
            .run(&mut dag, "dag-1", "greedy", 1, &stores)
            .await
            .unwrap();

        let hmp = report.pass("HMPPass").expect("HMP should have reported");
        assert!(
            hmp.dag_runs_used <= 4,
            "the search must not exceed its run budget, spent {}",
            hmp.dag_runs_used
        );
        assert!(hmp.dag_runs_used >= 1, "the baseline alone is one run");
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
        state.working_order = state.working_set.clone();
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
        assert_eq!(
            after.tried_combos.get("sig-a").copied(),
            Some(1250),
            "the censored observation must be filed at the budget, not discarded"
        );
        assert_eq!(
            after.iterations.last().unwrap().outcome.as_deref(),
            Some("cancelled")
        );
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
        assert_eq!(pass.budget(&state), Some(1250));
    }

    // A continuous optimization's whole premise is that its search survives
    // between runs, so state written by one step has to be what the next one
    // reads. Register, step, and check the search actually moved.
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
            Some(vec!["opt_hmp_state".to_string(), "opt_hmp_trials".to_string()]),
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
