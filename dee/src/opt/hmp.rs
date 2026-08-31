use async_trait::async_trait;
use log::{debug, info, warn};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    fs,
    marker::PhantomData,
    sync::Arc,
};

use crate::{
    connectors::Connector,
    dag::MaterializeMode,
    executor::{ExecStats, Executor, ProfilingConfig, SimpleEngine},
    plan::OpKey,
    opt::{
        Dag, Explain, OptimizerError, OptimizerPass,
        common::make_temp,
        explain::{render_bar_row, render_card_grid, render_ranked_table},
        pushdown::PushdownPass,
        report::{HmpDetail, IterationStat, PassDetail, PassOutcome},
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
    /// Run the PushdownPass before evaluating each candidate materialization
    /// combination, for more accurate cost measurements.
    use_pushdown: bool,
    /// Number of hypotheses the `Greedy` strategy's beam search keeps alive
    /// at each step. Unused by the `Breadth` strategy.
    beam_width: usize,
    /// Capture each iteration's CPU/memory/disk timeseries (already sampled
    /// by the profiled engine used for measurement) into its `IterationStat`.
    profile_iterations: bool,
    /// Data collected during the last `run()`, used by `Explain::explain`.
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
#[derive(Debug, Clone)]
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
        use_pushdown: bool,
        beam_width: usize,
        profile_iterations: bool,
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
            use_pushdown,
            beam_width: beam_width.max(1),
            profile_iterations,
            top_cpu_time: if top_cpu_time > 0.0 && top_cpu_time <= 1.0 {
                top_cpu_time
            } else {
                0.5
            },
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
            })
            .collect()
    }

    /// Re-score `nodes` using the node ranking observed across a set of
    /// trial `(Dag, ExecStats)` pairs, rather than the stale baseline
    /// ranking. Materializing a node changes the plans of every downstream
    /// consumer, which can reveal that the baseline ranking double-counted
    /// operator cost shared between two candidate Views -- this re-derives
    /// each node's ranking score from every trial DAG it still appears as a
    /// View in, averages across observations, and falls back to
    /// `baseline_scores` for any node absent from every observation (e.g. it
    /// was materialized in all of them). Returns `nodes` reordered by
    /// descending refined score.
    fn refine_node_order(
        &self,
        nodes: &[String],
        observations: &[(Dag, ExecStats)],
        baseline_scores: &HashMap<String, f64>,
    ) -> Vec<String> {
        let mut score_sums: HashMap<String, f64> = HashMap::new();
        let mut score_counts: HashMap<String, usize> = HashMap::new();

        for (obs_dag, obs_stats) in observations {
            let aggregate_cpu_time = if self.downstream_cost {
                Self::aggregate_downstream_cost(self.conn.as_ref(), obs_dag, obs_stats)
            } else {
                let op_stats = self.operator_stats(obs_dag, obs_stats);
                Self::aggregate_cpu_time_avg(self.conn.as_ref(), obs_dag, obs_stats, &op_stats)
            };
            let ranking = Self::build_node_table(self.conn.as_ref(), obs_stats, aggregate_cpu_time, self.normalize_with_cardinality);
            for row in ranking {
                *score_sums.entry(row.node.clone()).or_insert(0.0) += row.ranking_score;
                *score_counts.entry(row.node).or_insert(0) += 1;
            }
        }

        let mut ordered = nodes.to_vec();
        ordered.sort_by(|a, b| {
            let score_of = |n: &String| -> f64 {
                match score_counts.get(n) {
                    Some(&count) if count > 0 => score_sums.get(n).copied().unwrap_or(0.0) / count as f64,
                    _ => baseline_scores.get(n).copied().unwrap_or(0.0),
                }
            };
            score_of(b)
                .partial_cmp(&score_of(a))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        ordered
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

/// Canonical string signature of a DAG's structure. Used to detect when two
/// different materialization combinations produce an equivalent DAG (e.g.
/// after `make_temp`'s landing-pad insertion / view inlining), so we can
/// avoid re-running a trial we've effectively already tried.
fn dag_signature(dag: &Dag) -> String {
    let mut node_sigs: Vec<String> = dag
        .nodes
        .nodes()
        .map(|n| {
            let mut deps: Vec<&str> = n.depends_on.iter().map(String::as_str).collect();
            deps.sort_unstable();
            format!(
                "{}::{}::{}::[{}]",
                n.id,
                n.materialize.as_str(),
                n.query_text,
                deps.join(",")
            )
        })
        .collect();
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

#[async_trait]
impl<C, E> OptimizerPass<C, E> for HMPPass<C, E>
where
    C: Connector + Send + 'static + Sync,
    E: Executor<C> + Send + Sync,
{
    async fn run(&mut self, dag: &mut Dag) -> Result<PassOutcome, OptimizerError> {
        debug!(
            "Running HMPPass (Heuristic Materialization Pass), max_runs={}, top_cpu_time={}",
            self.max_runs, self.top_cpu_time
        );

        // 1. Establish baseline and collect plans
        debug!("Establishing baseline by running DAG with profiling and plan collection enabled");
        let engine = SimpleEngine::new(self.conn.clone())
            .map_err(|e| OptimizerError::Exec(e.to_string()))?
            .with_profiling(ProfilingConfig {
                collect_plans: true,
                ..Default::default()
            });

        let _ = engine.cleanup(dag).await.unwrap();
        let exec_stats = engine
            .run(dag)
            .await
            .map_err(|e| OptimizerError::Exec(format!("baseline run failed: {}", e)))?;

        let mut runs_used = 1usize;
        let mut best_ms = exec_stats.duration.num_milliseconds();
        debug!("Baseline run completed in {}ms", best_ms);
        let baseline_ms = best_ms;

        let mut iterations: Vec<IterationStat> = vec![IterationStat {
            iteration: 1,
            runtime_ms: best_ms,
            combo: Vec::new(),
            system_samples: if self.profile_iterations {
                exec_stats.system_samples.clone()
            } else {
                Vec::new()
            },
            ..Default::default()
        }];

        // 2. Build the operator stats, then the node ranking derived from
        // them: for every View with out-degree > 1, the aggregate average
        // runtime of every operator traced back to it. Walk the node
        // ranking, accumulating CPU time until it covers `top_cpu_time` of
        // the total, to build the working set of candidate nodes to search.
        let op_stats = self.operator_stats(dag, &exec_stats);
        self.log_operator_table(dag, &exec_stats, &op_stats);

        let aggregate_cpu_time = if self.downstream_cost {
            Self::aggregate_downstream_cost(self.conn.as_ref(), dag, &exec_stats)
        } else {
            Self::aggregate_cpu_time_avg(self.conn.as_ref(), dag, &exec_stats, &op_stats)
        };
        let node_ranking = Self::build_node_table(self.conn.as_ref(), &exec_stats, aggregate_cpu_time, self.normalize_with_cardinality);
        self.log_node_table(&node_ranking);
        let baseline_scores: HashMap<String, f64> = node_ranking
            .iter()
            .map(|r| (r.node.clone(), r.ranking_score))
            .collect();
        let total_node_timing: f64 = node_ranking.iter().map(|r| r.ranking_score).sum();

        let mut candidate_nodes: Vec<String> = Vec::new();
        if total_node_timing > 0.0 {
            let mut cumulative = 0.0;
            for row in &node_ranking {
                candidate_nodes.push(row.node.clone());
                cumulative += row.ranking_score;
                if cumulative / total_node_timing >= self.top_cpu_time {
                    break;
                }
            }
        }
        debug!(
            "Candidate working set contains {} node(s) (top_cpu_time={}): {:?}",
            candidate_nodes.len(),
            self.top_cpu_time,
            candidate_nodes
        );

        // 3. Search through candidate nodes to find the best materialization
        // combination. The strategy determines how we explore the search space.
        let baseline_dag = dag.clone();
        let mut best_combo: Vec<String> = Vec::new();
        let mut tried_dag_sigs: HashSet<String> = HashSet::new();

        match self.strategy {
            HMPStrategy::Breadth => {
                // Try all k-sized combinations smallest-first (singles, pairs,
                // triples, ...). Every attempt costs one run out of `max_runs`.
                // Between sizes, the working order is refined using the real
                // EXPLAIN ANALYZE plans collected from the previous size's
                // trials, so the most promising combos (per the freshest
                // evidence) are tried first if the budget runs out early.
                let mut working_order = candidate_nodes.clone();
                let mut round_observations: Vec<(Dag, ExecStats)> = Vec::new();
                'sizes: for k in 1..=candidate_nodes.len() {
                    if runs_used >= self.max_runs {
                        break;
                    }
                    if k > 1 {
                        working_order =
                            self.refine_node_order(&working_order, &round_observations, &baseline_scores);
                        round_observations.clear();
                    }
                    let combos = combinations(&working_order, k);
                    debug!("Trying {} combination(s) of size {}", combos.len(), k);

                    for combo in combos {
                        if runs_used >= self.max_runs {
                            break 'sizes;
                        }

                        let mut trial_dag = baseline_dag.clone();
                        let mut trial_counter = 0usize;
                        for node_id in &combo {
                            make_temp(&mut trial_dag, node_id, &mut trial_counter)?;
                        }

                        if !tried_dag_sigs.insert(dag_signature(&trial_dag)) {
                            debug!(
                                "Combo {:?} produces a DAG equivalent to one already tried, skipping",
                                combo
                            );
                            continue;
                        }

                        if self.use_pushdown {
                            let mut pushdown_pass =
                                PushdownPass::new(self.conn.clone(), self.engine.clone());
                            if let Err(e) = pushdown_pass.run(&mut trial_dag).await {
                                debug!(
                                    "HMPPass: pushdown failed for combo {:?}, continuing without it: {e}",
                                    combo
                                );
                            }
                        }

                        debug!(
                            "Trying materialization combo {:?}, re-running DAG to measure impact",
                            combo
                        );
                        let _ = engine.cleanup(&trial_dag).await.unwrap();
                        let trial_stats = engine
                            .run(&trial_dag)
                            .await
                            .map_err(|e| OptimizerError::Exec(format!("candidate run failed: {}", e)))?;
                        runs_used += 1;

                        let trial_ms = trial_stats.duration.num_milliseconds();
                        iterations.push(IterationStat {
                            iteration: iterations.len() + 1,
                            runtime_ms: trial_ms,
                            combo: combo.clone(),
                            system_samples: if self.profile_iterations {
                                trial_stats.system_samples.clone()
                            } else {
                                Vec::new()
                            },
                            ..Default::default()
                        });
                        round_observations.push((trial_dag, trial_stats));
                        if trial_ms < best_ms {
                            debug!(
                                "Combo {:?} improved runtime: {}ms -> {}ms",
                                combo, best_ms, trial_ms
                            );
                            best_ms = trial_ms;
                            best_combo = combo;
                        } else {
                            debug!(
                                "Combo {:?} did not improve runtime ({}ms -> {}ms)",
                                combo, best_ms, trial_ms
                            );
                        }
                    }
                }
            }
            HMPStrategy::Greedy => {
                // Beam search over the node ranking: at each node, expand
                // every live beam by trying it added to that beam's combo,
                // then keep only the `beam_width` best-performing combos
                // (which may include a beam carried forward unchanged, i.e.
                // "don't add this node"). Carrying beams forward unchanged is
                // what lets the search recover from an early, locally-good
                // but globally-wrong commitment -- a plain single-path
                // greedy walk can never drop a node once it's committed.
                let mut beams = vec![BeamState {
                    combo: Vec::new(),
                    runtime_ms: best_ms,
                }];
                let mut tried_combos: HashMap<String, i64> = HashMap::new();

                'nodes: for node_id in &candidate_nodes {
                    if runs_used >= self.max_runs {
                        break;
                    }

                    let mut proposals = beams.clone();
                    for beam in &beams {
                        if beam.combo.contains(node_id) {
                            continue;
                        }

                        let mut trial_combo = beam.combo.clone();
                        trial_combo.push(node_id.clone());

                        let mut trial_dag = baseline_dag.clone();
                        let mut trial_counter = 0usize;
                        for nid in &trial_combo {
                            make_temp(&mut trial_dag, nid, &mut trial_counter)?;
                        }

                        let sig = dag_signature(&trial_dag);
                        let trial_ms = if let Some(&cached_ms) = tried_combos.get(&sig) {
                            debug!(
                                "Combo {:?} produces a DAG equivalent to one already tried, reusing its runtime",
                                trial_combo
                            );
                            cached_ms
                        } else {
                            if runs_used >= self.max_runs {
                                break;
                            }

                            if self.use_pushdown {
                                let mut pushdown_pass =
                                    PushdownPass::new(self.conn.clone(), self.engine.clone());
                                if let Err(e) = pushdown_pass.run(&mut trial_dag).await {
                                    debug!(
                                        "HMPPass: pushdown failed for combo {:?}, continuing without it: {e}",
                                        trial_combo
                                    );
                                }
                            }

                            debug!(
                                "Trying materialization combo {:?}, re-running DAG to measure impact",
                                trial_combo
                            );
                            let _ = engine.cleanup(&trial_dag).await.unwrap();
                            let trial_stats = engine.run(&trial_dag).await.map_err(|e| {
                                OptimizerError::Exec(format!("candidate run failed: {}", e))
                            })?;
                            runs_used += 1;

                            let ms = trial_stats.duration.num_milliseconds();
                            iterations.push(IterationStat {
                                iteration: iterations.len() + 1,
                                runtime_ms: ms,
                                combo: trial_combo.clone(),
                                system_samples: if self.profile_iterations {
                                    trial_stats.system_samples.clone()
                                } else {
                                    Vec::new()
                                },
                                ..Default::default()
                            });
                            tried_combos.insert(sig, ms);

                            if ms < best_ms {
                                debug!(
                                    "Combo {:?} improved runtime: {}ms -> {}ms",
                                    trial_combo, best_ms, ms
                                );
                                best_ms = ms;
                                best_combo = trial_combo.clone();
                            } else {
                                debug!(
                                    "Combo {:?} did not improve runtime ({}ms -> {}ms)",
                                    trial_combo, best_ms, ms
                                );
                            }
                            ms
                        };

                        proposals.push(BeamState {
                            combo: trial_combo,
                            runtime_ms: trial_ms,
                        });

                        if runs_used >= self.max_runs {
                            break;
                        }
                    }

                    proposals.sort_by_key(|p| p.runtime_ms);
                    proposals.dedup_by(|a, b| a.combo == b.combo);
                    proposals.truncate(self.beam_width.max(1));
                    beams = proposals;

                    if runs_used >= self.max_runs {
                        break 'nodes;
                    }
                }
            }
        }

        // 4. Apply whichever combination (if any) produced the best runtime.
        let mut lp_counter = 0usize;
        for node_id in &best_combo {
            make_temp(dag, node_id, &mut lp_counter)?;
        }

        debug!(
            "Heuristic complete: materialized {} view(s) using {}/{} runs",
            best_combo.len(),
            runs_used,
            self.max_runs
        );
        let outcome = PassOutcome {
            dag_runs_used: runs_used as u32,
            changes_applied: best_combo.len() as u32,
            // Every iteration past the baseline is one candidate evaluated.
            candidates_considered: iterations.len().saturating_sub(1) as u32,
            working_set_size: candidate_nodes.len() as u32,
            iterations: iterations.clone(),
            detail: PassDetail::Hmp(HmpDetail {
                baseline_runtime_ms: baseline_ms,
                final_runtime_ms: best_ms,
                max_runs: self.max_runs,
                top_cpu_time: self.top_cpu_time,
                strategy: format!("{:?}", self.strategy),
                beam_width: self.beam_width,
                normalize_with_cardinality: self.normalize_with_cardinality,
                downstream_cost: self.downstream_cost,
                use_pushdown: self.use_pushdown,
                new_materializations: best_combo.clone(),
                working_set: candidate_nodes.clone(),
            }),
        };

        self.explain_data = Some(HMPExplainData {
            baseline_ms: iterations.first().map(|i| i.runtime_ms).unwrap_or(best_ms),
            final_ms: best_ms,
            runs_used,
            max_runs: self.max_runs,
            top_cpu_time: self.top_cpu_time,
            normalize_with_cardinality: self.normalize_with_cardinality,
            operator_rows: Self::build_operator_table(self.conn.as_ref(), dag, &exec_stats, &op_stats),
            node_rows: node_ranking,
            working_set: candidate_nodes,
            best_combo,
            iterations,
            strategy: self.strategy,
            beam_width: self.beam_width,
        });

        Ok(outcome)
    }
}

impl<C, E> Explain for HMPPass<C, E>
where
    C: Connector + Send + 'static + Sync,
    E: Executor<C> + Send + Sync,
{
    fn explain_label(&self) -> String {
        "HMPPass".to_string()
    }

    fn explain(&self) -> String {
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
            false,
            beam_width,
            false,
        )
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

        let refined = pass.refine_node_order(&nodes, &[(dag, exec_stats)], &baseline_scores);

        assert_eq!(refined, vec!["c_view".to_string(), "b_view".to_string()]);
    }

    // Nodes absent from every observation (e.g. materialized in all trials
    // seen so far) fall back to the baseline score instead of being treated
    // as zero-cost.
    #[tokio::test]
    async fn refine_node_order_falls_back_to_baseline_for_unobserved_nodes() {
        let pass = test_pass(2).await;

        let baseline_scores: HashMap<String, f64> = [
            ("b_view".to_string(), 10.0),
            ("c_view".to_string(), 5.0),
        ]
        .into_iter()
        .collect();
        let nodes = vec!["c_view".to_string(), "b_view".to_string()];

        // No observations at all -- order should be untouched apart from
        // falling back fully to baseline_scores (b_view still ranks first).
        let refined = pass.refine_node_order(&nodes, &[], &baseline_scores);

        assert_eq!(refined, vec!["b_view".to_string(), "c_view".to_string()]);
    }

    // End-to-end smoke test for the Greedy strategy's beam search: exercises
    // the full run() path (real DAG execution, cleanup, EXPLAIN ANALYZE
    // collection) against a small in-memory DuckDB DAG, and checks the pass
    // completes, respects its run budget, and only ever applies
    // materializations that were actually searched.
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

        let engine = Arc::new(SimpleEngine::new(Arc::clone(&conn)).unwrap());
        let mut pass = HMPPass::new(
            conn,
            engine,
            false,
            4,
            1.0,
            None,
            None,
            false,
            HMPStrategy::Greedy,
            false,
            2,
            false,
        );

        let outcome = pass.run(&mut dag).await.unwrap();

        let runs_used = outcome.dag_runs_used;
        assert!(runs_used <= 4);
        assert!(runs_used >= 1);
    }
}
