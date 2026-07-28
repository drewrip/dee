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
    opt::{
        Dag, Explain, OptimizerError, OptimizerPass,
        common::make_temp,
        explain::{render_bar_row, render_card_grid, render_ranked_table},
        pushdown::PushdownPass,
    },
};

/// Strategy HMP uses to search through the node ranking when deciding
/// which VIEWs to materialize.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
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
    no_plan_dups: bool,
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
}

#[derive(Deserialize, Debug)]
struct DuckDBPlan {
    operator_name: Option<String>,
    #[serde(alias = "name")]
    name: Option<String>,
    #[serde(default)]
    operator_timing: Option<f64>,
    #[serde(default)]
    extra_info: HashMap<String, serde_json::Value>,
    #[serde(default)]
    children: Vec<DuckDBPlan>,
}

#[derive(Hash, Eq, PartialEq, Clone, Debug)]
struct OpKey {
    name: String,
    cardinality: String,
}

/// Runtime of a single iteration attempted by the pass, in the order it was
/// run. Iteration 1 is always the baseline (no materializations applied).
#[derive(Serialize, Debug, Clone)]
struct IterationStat {
    iteration: usize,
    runtime_ms: i64,
    /// Materialization combo tried at this iteration; empty for the
    /// baseline (iteration 1).
    #[serde(default)]
    combo: Vec<String>,
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

impl DuckDBPlan {
    fn get_sig(&self) -> Option<OpKey> {
        let name = self.operator_name.clone().or_else(|| self.name.clone())?;
        let cardinality = self
            .extra_info
            .get("Estimated Cardinality")
            .and_then(|v| {
                if let Some(s) = v.as_str() {
                    Some(s.to_string())
                } else if let Some(f) = v.as_f64() {
                    Some(f.to_string())
                } else if let Some(i) = v.as_i64() {
                    Some(i.to_string())
                } else {
                    None
                }
            })
            .unwrap_or_else(|| "0".to_string());
        Some(OpKey { name, cardinality })
    }

    fn collect_operator_stats(
        &self,
        timing_map: &mut HashMap<OpKey, f64>,
        occurrence_map: &mut HashMap<OpKey, usize>,
        no_plan_dups: bool,
        plan_sigs: &mut HashSet<OpKey>,
    ) {
        if let Some(sig) = self.get_sig() {
            if let Some(t) = self.operator_timing {
                *timing_map.entry(sig.clone()).or_insert(0.0) += t;
            }

            if no_plan_dups {
                plan_sigs.insert(sig);
            } else {
                *occurrence_map.entry(sig).or_insert(0) += 1;
            }
        }
        for child in &self.children {
            child.collect_operator_stats(timing_map, occurrence_map, no_plan_dups, plan_sigs);
        }
    }

    fn contains_operator(&self, target: &OpKey) -> bool {
        if let Some(sig) = self.get_sig() {
            if sig == *target {
                return true;
            }
        }
        for child in &self.children {
            if child.contains_operator(target) {
                return true;
            }
        }
        false
    }
}

impl<C, E> HMPPass<C, E>
where
    C: Connector + Send + 'static + Sync,
    E: Executor<C> + Send + Sync,
{
    pub fn new(
        conn: Arc<C>,
        engine: Arc<E>,
        no_plan_dups: bool,
        max_runs: usize,
        top_cpu_time: f64,
        show_operators: Option<String>,
        show_nodes: Option<String>,
        normalize_with_cardinality: bool,
        strategy: HMPStrategy,
        use_pushdown: bool,
    ) -> Self {
        Self {
            conn,
            engine,
            no_plan_dups,
            max_runs: max_runs.max(1),
            show_operators,
            show_nodes,
            normalize_with_cardinality,
            strategy,
            use_pushdown,
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
                if let Some(node_stat) = exec_stats.node_stats.get(&node.id) {
                    if let Some(plan_str) = &node_stat.plan {
                        if let Ok(plan) = serde_json::from_str::<DuckDBPlan>(plan_str) {
                            if self.no_plan_dups {
                                let mut plan_sigs = HashSet::new();
                                plan.collect_operator_stats(
                                    &mut timing_map,
                                    &mut occurrence_map,
                                    true,
                                    &mut plan_sigs,
                                );
                                for sig in plan_sigs {
                                    *occurrence_map.entry(sig).or_insert(0) += 1;
                                }
                            } else {
                                plan.collect_operator_stats(
                                    &mut timing_map,
                                    &mut occurrence_map,
                                    false,
                                    &mut HashSet::new(),
                                );
                            }
                        }
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
                traced_views: Self::find_traced_views(dag, op_key, exec_stats),
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

        let rows = Self::build_operator_table(dag, exec_stats, op_stats);
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
    fn find_traced_views(dag: &Dag, op_key: &OpKey, exec_stats: &ExecStats) -> Vec<String> {
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
            if let Ok(plans) = serde_json::from_str::<Vec<DuckDBPlan>>(plan_str)
                && plans.iter().any(|p| p.contains_operator(op_key))
            {
                views.push(node.id.clone());
            }
        }
        views
    }

    /// Estimated cardinality of a View's own EXPLAIN plan, taken from the
    /// root operator of its (already-collected) query plan.
    fn view_cardinality(exec_stats: &ExecStats, view_id: &str) -> Option<f64> {
        let node_stat = exec_stats.node_stats.get(view_id)?;
        let plan_str = node_stat.plan.as_ref()?;
        let plans: Vec<DuckDBPlan> = serde_json::from_str(plan_str).ok()?;
        let root = plans.first()?;
        let sig = root.get_sig()?;
        sig.cardinality.parse::<f64>().ok()
    }

    /// Build the `--hmp-show-nodes` ranking table: for every View with
    /// out-degree > 1 and more than one downstream path to a TABLE/TEMP_TABLE
    /// node (a branch point, the only kind of View that materializing can
    /// actually deduplicate work for), sum the average runtime of every
    /// operator that traces back to it via `find_traced_views` (the same
    /// mapping the `--hmp-show-operators` table uses). Sorted by
    /// `ranking_score`, descending -- this is also the order `run()` searches
    /// down when picking which node to try materializing. `ranking_score` is
    /// `total_cpu_time_s`, or (when `normalize_with_cardinality` is set)
    /// `total_cpu_time_s` divided by the View's estimated cardinality, from
    /// its EXPLAIN plan.
    fn build_node_table(
        dag: &Dag,
        exec_stats: &ExecStats,
        op_stats: &HashMap<OpKey, (f64, usize)>,
        normalize_with_cardinality: bool,
    ) -> Vec<NodeRankingRow> {
        let mut aggregate_cpu_time: HashMap<String, f64> = HashMap::new();
        for (op_key, (avg_runtime, _)) in op_stats {
            for view in Self::find_traced_views(dag, op_key, exec_stats) {
                if dag.nodes.out_degree(&view) > 1 && dag.nodes.paths_to_sinks(&view) > 1 {
                    *aggregate_cpu_time.entry(view).or_insert(0.0) += avg_runtime;
                }
            }
        }

        let mut rows: Vec<(String, f64, Option<f64>, f64)> = aggregate_cpu_time
            .into_iter()
            .map(|(node, total_cpu_time_s)| {
                let cardinality = Self::view_cardinality(exec_stats, &node);
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
    async fn run(&mut self, dag: &mut Dag) -> Result<HashMap<String, String>, OptimizerError> {
        debug!(
            "Running HMPPass (Heuristic Materialization Pass), max_runs={}, top_cpu_time={}",
            self.max_runs, self.top_cpu_time
        );
        let mut stats = HashMap::new();

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
        stats.insert("baseline_runtime_ms".into(), best_ms.to_string());

        let mut iterations: Vec<IterationStat> = vec![IterationStat {
            iteration: 1,
            runtime_ms: best_ms,
            combo: Vec::new(),
        }];

        // 2. Build the operator stats, then the node ranking derived from
        // them: for every View with out-degree > 1, the aggregate average
        // runtime of every operator traced back to it. Walk the node
        // ranking, accumulating CPU time until it covers `top_cpu_time` of
        // the total, to build the working set of candidate nodes to search.
        let op_stats = self.operator_stats(dag, &exec_stats);
        self.log_operator_table(dag, &exec_stats, &op_stats);

        let node_ranking = Self::build_node_table(
            dag,
            &exec_stats,
            &op_stats,
            self.normalize_with_cardinality,
        );
        self.log_node_table(&node_ranking);
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
                'sizes: for k in 1..=candidate_nodes.len() {
                    if runs_used >= self.max_runs {
                        break;
                    }
                    let combos = combinations(&candidate_nodes, k);
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
                        });
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
                // Walk the node ranking sequentially. For each node, try
                // materializing it along with all previously-committed nodes.
                // If performance improves, commit that node and continue down
                // the ranking.
                let mut committed: Vec<String> = Vec::new();
                for node_id in &candidate_nodes {
                    if runs_used >= self.max_runs {
                        break;
                    }

                    let mut trial_combo = committed.clone();
                    trial_combo.push(node_id.clone());

                    let mut trial_dag = baseline_dag.clone();
                    let mut trial_counter = 0usize;
                    for nid in &trial_combo {
                        make_temp(&mut trial_dag, nid, &mut trial_counter)?;
                    }

                    if !tried_dag_sigs.insert(dag_signature(&trial_dag)) {
                        debug!(
                            "Combo {:?} produces a DAG equivalent to one already tried, skipping",
                            trial_combo
                        );
                        continue;
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
                    let trial_stats = engine
                        .run(&trial_dag)
                        .await
                        .map_err(|e| OptimizerError::Exec(format!("candidate run failed: {}", e)))?;
                    runs_used += 1;

                    let trial_ms = trial_stats.duration.num_milliseconds();
                    iterations.push(IterationStat {
                        iteration: iterations.len() + 1,
                        runtime_ms: trial_ms,
                        combo: trial_combo.clone(),
                    });
                    if trial_ms < best_ms {
                        debug!(
                            "Combo {:?} improved runtime: {}ms -> {}ms",
                            trial_combo, best_ms, trial_ms
                        );
                        best_ms = trial_ms;
                        best_combo = trial_combo.clone();
                        committed = trial_combo;
                    } else {
                        debug!(
                            "Combo {:?} did not improve runtime ({}ms -> {}ms)",
                            trial_combo, best_ms, trial_ms
                        );
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
        stats.insert(
            "new_materializations".into(),
            if best_combo.is_empty() {
                "none".into()
            } else {
                best_combo.join(",")
            },
        );
        stats.insert(
            "materialization_count".into(),
            best_combo.len().to_string(),
        );
        stats.insert("working_set_size".into(), candidate_nodes.len().to_string());
        stats.insert("runs_used".into(), runs_used.to_string());
        stats.insert("final_runtime_ms".into(), best_ms.to_string());
        stats.insert(
            "iterations".into(),
            serde_json::to_string(&iterations)
                .map_err(|e| OptimizerError::Exec(format!("failed to serialize iterations: {e}")))?,
        );

        self.explain_data = Some(HMPExplainData {
            baseline_ms: iterations.first().map(|i| i.runtime_ms).unwrap_or(best_ms),
            final_ms: best_ms,
            runs_used,
            max_runs: self.max_runs,
            top_cpu_time: self.top_cpu_time,
            normalize_with_cardinality: self.normalize_with_cardinality,
            operator_rows: Self::build_operator_table(dag, &exec_stats, &op_stats),
            node_rows: node_ranking,
            working_set: candidate_nodes,
            best_combo,
            iterations,
            strategy: self.strategy,
        });

        Ok(stats)
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
          <div class="subtle">Combinations of working-set nodes were tried smallest-first (singles, then pairs, ...) until the run budget was exhausted. The combination with the lowest runtime was applied to the DAG.</div>
          <div class="plan-tree">{iteration_bars}</div>
        </div>
      </div>"##,
            data.top_cpu_time * 100.0
        )
    }
}
