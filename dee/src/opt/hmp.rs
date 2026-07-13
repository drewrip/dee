use async_trait::async_trait;
use log::debug;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    marker::PhantomData,
    sync::Arc,
};

use crate::{
    connectors::Connector,
    dag::MaterializeMode,
    executor::{ExecStats, Executor, ProfilingConfig, SimpleEngine},
    opt::{Dag, OptimizerError, OptimizerPass, common::make_temp},
};

#[derive(Debug, Clone)]
pub struct HMPPass<C, E>
where
    C: Connector + Send + 'static + Sync,
    E: Executor<C> + Send + Sync,
{
    conn: Arc<C>,
    no_plan_dups: bool,
    /// Max number of DAG re-runs to spend searching for materialization
    /// candidates. Each attempted materialization (successful or not) costs
    /// one run, in addition to the initial baseline run.
    max_runs: usize,
    /// Fraction (0, 1.0] of total operator CPU time used to build the
    /// `working_set` of candidate operators to materialize.
    top_cpu_time: f64,
    _phantom: PhantomData<E>,
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
    pub fn new(conn: Arc<C>, no_plan_dups: bool, max_runs: usize, top_cpu_time: f64) -> Self {
        Self {
            conn,
            no_plan_dups,
            max_runs: max_runs.max(1),
            top_cpu_time: if top_cpu_time > 0.0 && top_cpu_time <= 1.0 {
                top_cpu_time
            } else {
                0.5
            },
            _phantom: PhantomData,
        }
    }

    /// Rank operators found in the EXPLAIN ANALYZE plans of currently
    /// materialized (Table) nodes by their potential duplication time,
    /// descending. Only operators seen more than once are returned, since
    /// those are the only ones a new materialization could deduplicate.
    fn rank_operators(&self, dag: &Dag, exec_stats: &ExecStats) -> Vec<(OpKey, f64, f64)> {
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

        let mut ranked_ops: Vec<_> = timing_map
            .into_iter()
            .filter(|(sig, _)| occurrence_map.get(sig).cloned().unwrap_or(0) > 1)
            .map(|(sig, t)| {
                let n = occurrence_map.get(&sig).cloned().unwrap_or(1) as f64;
                let potential_duplication_time = if n > 0.0 { t - t / n } else { 0.0 };
                (sig, t, potential_duplication_time)
            })
            .collect();

        ranked_ops.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        if !ranked_ops.is_empty() {
            debug!("Top 5 bottlenecks by potential duplication time across all plans:");
            for (i, (op, timing, pdt)) in ranked_ops.iter().take(5).enumerate() {
                debug!(
                    "  {}. {:?} - PDT: {:.4}s (Total Timing: {:.4}s, Found {} times)",
                    i + 1,
                    op,
                    pdt,
                    timing,
                    occurrence_map.get(op).unwrap_or(&0)
                );
            }
        }

        ranked_ops
    }

    /// Find the node that should be materialized for a given operator: the
    /// nearest branch point (out-degree > 1) downstream of the View node
    /// whose EXPLAIN plan contains the operator.
    fn find_materialization_candidate(
        dag: &Dag,
        op_key: &OpKey,
        exec_stats: &ExecStats,
    ) -> Option<String> {
        let sorted_node_ids = dag.nodes.topological_sort();

        let mut candidate_node_id = None;
        for node_id in &sorted_node_ids {
            let node = dag.nodes.get(node_id.clone())?;
            if matches!(node.materialize, MaterializeMode::View) {
                if let Some(node_stat) = exec_stats.node_stats.get(node_id) {
                    if let Some(plan_str) = &node_stat.plan {
                        // Plain EXPLAIN JSON is usually an array
                        if let Ok(plans) = serde_json::from_str::<Vec<DuckDBPlan>>(plan_str) {
                            if plans.iter().any(|p| p.contains_operator(op_key)) {
                                debug!(
                                    "Operator found in view node '{}' (closest to source)",
                                    node_id
                                );
                                candidate_node_id = Some(node_id.clone());
                                break;
                            }
                        }
                    }
                }
            }
        }

        // Follow the graph until a branch is found
        let mut current_id = candidate_node_id?;
        while dag.nodes.out_degree(&current_id) == 1 {
            let next_node = dag
                .nodes
                .nodes()
                .find(|n| n.depends_on.contains(&current_id))
                .map(|n| n.id.clone());

            if let Some(next_id) = next_node {
                debug!(
                    "Node '{}' has out-degree 1, moving downstream to '{}'",
                    current_id, next_id
                );
                current_id = next_id;
            } else {
                break;
            }
        }

        let out_degree = dag.nodes.out_degree(&current_id);
        if out_degree > 1 {
            debug!(
                "Found optimal branch point at node '{}' (out-degree={})",
                current_id, out_degree
            );
            Some(current_id)
        } else {
            debug!(
                "Stopped at node '{}' with out-degree {}, skipping candidate",
                current_id, out_degree
            );
            None
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
        }];

        // 2. Build the working_set: walk the operator ranking, accumulating
        // CPU time until it covers `top_cpu_time` of the total.
        let ranked_ops = self.rank_operators(dag, &exec_stats);
        let total_timing: f64 = ranked_ops.iter().map(|(_, t, _)| t).sum();

        let mut working_set_ops: Vec<OpKey> = Vec::new();
        if total_timing > 0.0 {
            let mut cumulative = 0.0;
            for (op_key, timing, _) in &ranked_ops {
                working_set_ops.push(op_key.clone());
                cumulative += timing;
                if cumulative / total_timing >= self.top_cpu_time {
                    break;
                }
            }
        }
        debug!(
            "Working set contains {} operator(s) (top_cpu_time={})",
            working_set_ops.len(),
            self.top_cpu_time
        );

        // Map each working_set operator back to the node that should be
        // materialized for it, deduplicating candidates found via more than
        // one operator.
        let mut candidate_nodes: Vec<String> = Vec::new();
        for op_key in &working_set_ops {
            if let Some(candidate) = Self::find_materialization_candidate(dag, op_key, &exec_stats)
                && !candidate_nodes.contains(&candidate)
            {
                candidate_nodes.push(candidate);
            }
        }
        debug!(
            "Working set maps to {} distinct candidate node(s): {:?}",
            candidate_nodes.len(),
            candidate_nodes
        );

        // 3. Search combinations of candidate nodes, starting with singles,
        // then pairs, then triples, etc. Every attempt (successful or not)
        // costs one run out of `max_runs`, so larger combination sizes are
        // only reached once every smaller size has been fully explored
        // within budget.
        let baseline_dag = dag.clone();
        let mut best_combo: Vec<String> = Vec::new();
        let mut tried_dag_sigs: HashSet<String> = HashSet::new();

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
        stats.insert("working_set_size".into(), working_set_ops.len().to_string());
        stats.insert("runs_used".into(), runs_used.to_string());
        stats.insert("final_runtime_ms".into(), best_ms.to_string());
        stats.insert(
            "iterations".into(),
            serde_json::to_string(&iterations)
                .map_err(|e| OptimizerError::Exec(format!("failed to serialize iterations: {e}")))?,
        );

        Ok(stats)
    }
}
