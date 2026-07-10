use async_trait::async_trait;
use log::debug;
use serde::Deserialize;
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
    pub fn new(conn: Arc<C>, no_plan_dups: bool, max_runs: usize) -> Self {
        Self {
            conn,
            no_plan_dups,
            max_runs: max_runs.max(1),
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

#[async_trait]
impl<C, E> OptimizerPass<C, E> for HMPPass<C, E>
where
    C: Connector + Send + 'static + Sync,
    E: Executor<C> + Send + Sync,
{
    async fn run(&mut self, dag: &mut Dag) -> Result<HashMap<String, String>, OptimizerError> {
        debug!(
            "Running HMPPass (Heuristic Materialization Pass), max_runs={}",
            self.max_runs
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
        let mut exec_stats = engine
            .run(dag)
            .await
            .map_err(|e| OptimizerError::Exec(format!("baseline run failed: {}", e)))?;

        let mut runs_used = 1usize;
        let mut best_ms = exec_stats.duration.num_milliseconds();
        debug!("Baseline run completed in {}ms", best_ms);
        stats.insert("baseline_runtime_ms".into(), best_ms.to_string());

        // 2. Iteratively materialize views: rank operators, try the top
        // candidates in order, lock in the first one that improves runtime,
        // then re-rank against the new DAG. Stop once the run budget is
        // exhausted or a full ranking pass finds nothing that helps.
        let mut materialized_views: Vec<String> = Vec::new();
        let mut rejected_views: HashSet<String> = HashSet::new();
        let mut lp_counter = 0usize;

        while runs_used < self.max_runs {
            debug!(
                "Searching for common expensive operators (run {}/{})",
                runs_used + 1,
                self.max_runs
            );
            let ranked_ops = self.rank_operators(dag, &exec_stats);

            let mut locked_in = false;
            for (op_key, timing, pdt) in ranked_ops {
                if runs_used >= self.max_runs {
                    break;
                }

                debug!(
                    "Evaluating operator {:?} (PDT={:.4}s, timing={:.4}s)",
                    op_key, pdt, timing
                );

                let candidate = match Self::find_materialization_candidate(dag, &op_key, &exec_stats)
                {
                    Some(c) => c,
                    None => continue,
                };

                if materialized_views.contains(&candidate) || rejected_views.contains(&candidate) {
                    debug!(
                        "Node '{}' already evaluated this run, skipping",
                        candidate
                    );
                    continue;
                }

                let snapshot = dag.clone();
                make_temp(dag, &candidate, &mut lp_counter)?;

                debug!(
                    "Trying materialization of '{}', re-running DAG to measure impact",
                    candidate
                );
                let _ = engine.cleanup(dag).await.unwrap();
                let candidate_stats = engine
                    .run(dag)
                    .await
                    .map_err(|e| OptimizerError::Exec(format!("candidate run failed: {}", e)))?;
                runs_used += 1;

                let candidate_ms = candidate_stats.duration.num_milliseconds();
                if candidate_ms < best_ms {
                    debug!(
                        "Materializing '{}' improved runtime: {}ms -> {}ms",
                        candidate, best_ms, candidate_ms
                    );
                    best_ms = candidate_ms;
                    exec_stats = candidate_stats;
                    materialized_views.push(candidate);
                    locked_in = true;
                    break;
                } else {
                    debug!(
                        "Materializing '{}' did not improve runtime ({}ms -> {}ms), reverting",
                        candidate, best_ms, candidate_ms
                    );
                    *dag = snapshot;
                    rejected_views.insert(candidate);
                }
            }

            if !locked_in {
                debug!("No candidate improved runtime this round, stopping search");
                break;
            }
        }

        debug!(
            "Heuristic complete: materialized {} view(s) using {}/{} runs",
            materialized_views.len(),
            runs_used,
            self.max_runs
        );
        stats.insert(
            "new_materializations".into(),
            if materialized_views.is_empty() {
                "none".into()
            } else {
                materialized_views.join(",")
            },
        );
        stats.insert(
            "materialization_count".into(),
            materialized_views.len().to_string(),
        );
        stats.insert("runs_used".into(), runs_used.to_string());
        stats.insert("final_runtime_ms".into(), best_ms.to_string());

        Ok(stats)
    }
}
