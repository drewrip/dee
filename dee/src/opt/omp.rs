use async_trait::async_trait;
use itertools::{Itertools, repeat_n};
use log::debug;
use std::{
    collections::{HashMap, HashSet},
    fs,
    marker::PhantomData,
    sync::Arc,
};

use crate::{
    connectors::Connector,
    dag::MaterializeMode,
    executor::Executor,
    opt::{Dag, OMPStrategy, OptimizationMetric, OptimizerError, OptimizerPass},
};

#[derive(Debug, Clone)]
pub struct OMPPass<C, E>
where
    C: Connector + Send + 'static + Sync,
    E: Executor<C> + Send + Sync,
{
    conn: Arc<C>,
    engine: Arc<E>,
    metric: OptimizationMetric,
    strategy: OMPStrategy,
    _phantom: PhantomData<C>,
}

impl<C, E> OMPPass<C, E>
where
    C: Connector + Send + 'static + Sync,
    E: Executor<C> + Send + Sync,
{
    pub fn new(conn: Arc<C>, engine: Arc<E>, metric: OptimizationMetric) -> Self {
        Self {
            conn,
            engine,
            metric,
            strategy: OMPStrategy::default(),
            _phantom: PhantomData,
        }
    }

    pub fn set_strategy(&mut self, strategy: OMPStrategy) {
        self.strategy = strategy;
    }

    async fn evaluate_dag(&self, dag: &Dag) -> Result<f32, OptimizerError> {
        match self.metric {
            OptimizationMetric::Runtime => {
                let run_result = self
                    .engine
                    .run(dag)
                    .await
                    .map_err(|e| OptimizerError::Exec(e.to_string()))?;
                Ok(run_result.duration.num_milliseconds() as f32)
            }
            OptimizationMetric::Cost => {
                let cost = self
                    .engine
                    .cost(dag)
                    .await
                    .map_err(|e| OptimizerError::Exec(e.to_string()))?;
                Ok(cost.unwrap_or(0.0))
            }
        }
    }

    fn collect_profile_stats(
        &self,
        node: &serde_json::Value,
        total_cpu_time: f32,
        stats: &mut HashMap<String, f32>,
    ) {
        if let (Some(name), Some(timing)) = (
            node.get("operator_name").and_then(|v| v.as_str()),
            node.get("operator_timing").and_then(|v| v.as_f64()),
        ) {
            let extra_info = node
                .get("extra_info")
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            let key = format!("{}:{}", name, extra_info.to_string());
            let rel_cpu = (timing as f32) / total_cpu_time;
            *stats.entry(key).or_insert(0.0) += rel_cpu;
        }

        if let Some(children) = node.get("children").and_then(|v| v.as_array()) {
            for child in children {
                self.collect_profile_stats(child, total_cpu_time, stats);
            }
        }
    }

    fn collect_explain_keys(&self, node: &serde_json::Value, keys: &mut HashSet<String>) {
        if let Some(name) = node.get("name").and_then(|v| v.as_str()) {
            let extra_info = node
                .get("extra_info")
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            let key = format!("{}:{}", name, extra_info.to_string());
            keys.insert(key);
        }

        if let Some(children) = node.get("children").and_then(|v| v.as_array()) {
            for child in children {
                self.collect_explain_keys(child, keys);
            }
        }
    }

    async fn run_strategy_1(
        &mut self,
        dag: &mut Dag,
    ) -> Result<HashMap<String, String>, OptimizerError> {
        debug!("Running OMPPass Strategy 1 (Exhaustive)");
        let mut stats = HashMap::new();
        let top_n = 3;
        let _ = self.engine.cleanup(dag).await.unwrap();

        let baseline_val = self.evaluate_dag(dag).await?;
        let mut best_val = baseline_val;

        let mut candidates: Vec<(String, usize)> = dag
            .nodes
            .nodes()
            .filter(|n| matches!(n.materialize, MaterializeMode::View))
            .cloned()
            .map(|n| (n.id.clone(), dag.nodes.out_degree(&n.id)))
            .collect();

        candidates.sort_by_key(|k| k.1);
        let top_candidates: Vec<_> = candidates.iter().rev().take(top_n).collect();

        let plans: Vec<Vec<MaterializeMode>> = repeat_n(
            [MaterializeMode::View, MaterializeMode::Table].into_iter(),
            top_candidates.len(),
        )
        .multi_cartesian_product()
        .collect();

        let mut work_dag = dag.clone();
        let mut best_plan: Vec<MaterializeMode> = top_candidates
            .iter()
            .map(|c| dag.nodes.get(c.0.clone()).unwrap().materialize)
            .collect();
        let baseline_plan = best_plan.clone();

        for (i, plan) in plans.iter().enumerate() {
            debug!("OMPPass: iter {}/{}", i + 1, plans.len());
            let _ = self.engine.cleanup(dag).await.unwrap();

            for node in plan.iter().enumerate() {
                let node_id = top_candidates.get(node.0).unwrap().0.clone();
                work_dag
                    .nodes
                    .get_mut(node_id.clone())
                    .ok_or(OptimizerError::Exec("missing node".to_string()))?
                    .materialize = node.1.clone();
            }

            let current_val = self.evaluate_dag(&work_dag).await?;

            stats.insert(format!("attempt_{}", i + 1), current_val.to_string());
            if current_val < best_val {
                best_val = current_val;
                best_plan = plan.clone();
            } else {
                for node in baseline_plan.iter().enumerate() {
                    let node_id = top_candidates.get(node.0).unwrap().0.clone();
                    work_dag
                        .nodes
                        .get_mut(node_id.clone())
                        .ok_or(OptimizerError::Exec("missing node".to_string()))?
                        .materialize = node.1.clone();
                }
            }
        }

        stats.insert("metric".into(), format!("{:?}", self.metric));
        stats.insert("baseline_value".into(), baseline_val.to_string());
        stats.insert("best_value".into(), best_val.to_string());
        let change = (best_val - baseline_val) / baseline_val;
        stats.insert("opt_change".into(), change.to_string());
        debug!(
            "OMPPass change: {:.2} -> {:.2} ({:.2}%)",
            baseline_val,
            best_val,
            change * 100.0,
        );

        let mut new_mats = vec![];
        for node in best_plan.clone().into_iter().enumerate() {
            let node_id = top_candidates.get(node.0).unwrap().0.clone();
            if matches!(node.1, MaterializeMode::Table) {
                new_mats.push(node_id.clone());
            }
            dag.nodes
                .get_mut(node_id)
                .ok_or(OptimizerError::Exec("missing node".to_string()))?
                .materialize = node.1.clone();
        }

        stats.insert("best_plan".into(), format!("{:?}", new_mats));
        Ok(stats)
    }

    async fn run_strategy_2(
        &mut self,
        dag: &mut Dag,
    ) -> Result<HashMap<String, String>, OptimizerError> {
        debug!("Running OMPPass Strategy 2 (Heuristic)");
        let mut stats = HashMap::new();

        // 1. Setup temp dir for plans
        let tmp_dir = tempfile::tempdir().map_err(|e| OptimizerError::Exec(e.to_string()))?;
        let plan_dir = tmp_dir.path().to_str().unwrap_or_default().to_string();

        // 2. Run baseline with plan collection
        let engine = E::new(self.conn.clone())
            .map_err(|e| OptimizerError::Exec(e.to_string()))?
            .with_plan_dir(plan_dir.clone());

        let _ = engine.cleanup(dag).await;
        let _ = engine
            .run(dag)
            .await
            .map_err(|e| OptimizerError::Exec(e.to_string()))?;

        // 3. Analyze materialized tables
        let mut master_map: HashMap<String, f32> = HashMap::new();

        for node in dag.nodes.nodes() {
            if matches!(node.materialize, MaterializeMode::Table) {
                let plan_path = tmp_dir.path().join(format!("table_{}.json", node.id));
                if plan_path.exists() {
                    let plan_content = fs::read_to_string(plan_path)
                        .map_err(|e| OptimizerError::Exec(e.to_string()))?;
                    let root: serde_json::Value = serde_json::from_str(&plan_content)
                        .map_err(|e| OptimizerError::Exec(e.to_string()))?;

                    let cpu_time =
                        root.get("cpu_time").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
                    let mut node_map = HashMap::new();
                    self.collect_profile_stats(&root, cpu_time, &mut node_map);

                    for (k, v) in node_map {
                        *master_map.entry(k).or_insert(0.0) += v;
                    }
                }
            }
        }

        // 4. Create sorted masterlist
        let mut masterlist: Vec<(String, f32)> = master_map.into_iter().collect();
        masterlist.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        debug!("masterlist = {:?}", masterlist);
        // 5. Search through VIEWs with outdegree > 1
        let mut view_candidates: Vec<_> = dag
            .nodes
            .nodes()
            .filter(|n| matches!(n.materialize, MaterializeMode::View))
            .map(|n| (n.id.clone(), dag.nodes.out_degree(&n.id)))
            .filter(|c| c.1 > 1)
            .collect();

        // Sort by outdegree desc
        view_candidates.sort_by(|a, b| b.1.cmp(&a.1));

        for (view_id, _) in view_candidates {
            let plan_path = tmp_dir.path().join(format!("view_{}.json", view_id));
            if plan_path.exists() {
                let plan_content = fs::read_to_string(plan_path)
                    .map_err(|e| OptimizerError::Exec(e.to_string()))?;
                let explain_nodes: serde_json::Value = serde_json::from_str(&plan_content)
                    .map_err(|e| OptimizerError::Exec(e.to_string()))?;

                let mut view_operators = HashSet::new();
                if let Some(nodes_arr) = explain_nodes.as_array() {
                    for node in nodes_arr {
                        self.collect_explain_keys(node, &mut view_operators);
                    }
                } else {
                    self.collect_explain_keys(&explain_nodes, &mut view_operators);
                }

                // 6. Match against masterlist
                for (op_key, rel_cpu) in &masterlist {
                    if *rel_cpu < 0.7 {
                        continue;
                    }
                    if view_operators.contains(op_key) {
                        debug!(
                            "Strategy 2: Switching view '{}' to table based on operator '{}' (val: {})",
                            view_id, op_key, rel_cpu
                        );
                        dag.nodes
                            .get_mut(view_id.clone())
                            .ok_or(OptimizerError::Exec("missing node".to_string()))?
                            .materialize = MaterializeMode::Table;
                        stats.insert("switched_view".into(), view_id);
                        stats.insert("match_operator".into(), op_key.clone());
                        stats.insert("match_value".into(), rel_cpu.to_string());
                        return Ok(stats);
                    }
                }
            }
        }

        Ok(stats)
    }
}

#[async_trait]
impl<C, E> OptimizerPass<C, E> for OMPPass<C, E>
where
    C: Connector + Send + 'static + Sync,
    E: Executor<C> + Send + Sync,
{
    async fn run(&mut self, dag: &mut Dag) -> Result<HashMap<String, String>, OptimizerError> {
        debug!("Running OMPPass (strategy: {:?})", self.strategy);
        match self.strategy {
            OMPStrategy::Exhaustive => self.run_strategy_1(dag).await,
            OMPStrategy::Heuristic => self.run_strategy_2(dag).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectors::duckdb::{DuckDBConnection, DuckDBProfile};
    use crate::dag::TransformNode;
    use crate::executor::SimpleEngine;
    use std::collections::HashSet;

    #[tokio::test]
    async fn test_omp_pass_cost_metric() {
        let profile = DuckDBProfile::new_from_path(":memory:".to_string());
        let conn = DuckDBConnection::new(profile).await.unwrap();
        let engine = Arc::new(SimpleEngine::new(conn.clone()).unwrap());

        let mut nodes = HashMap::new();
        // N1 is a table
        nodes.insert(
            "n1".to_string(),
            TransformNode {
                id: "n1".to_string(),
                query_text: "SELECT 1 AS id".to_string(),
                materialize: MaterializeMode::Table,
                depends_on: HashSet::new(),
            },
        );
        // N2 is a view with high out-degree (simulated by having N3 and N4 depend on it)
        nodes.insert(
            "n2".to_string(),
            TransformNode {
                id: "n2".to_string(),
                query_text: "SELECT * FROM n1".to_string(),
                materialize: MaterializeMode::View,
                depends_on: HashSet::from_iter(vec!["n1".to_string()]),
            },
        );
        nodes.insert(
            "n3".to_string(),
            TransformNode {
                id: "n3".to_string(),
                query_text: "SELECT * FROM n2".to_string(),
                materialize: MaterializeMode::View,
                depends_on: HashSet::from_iter(vec!["n2".to_string()]),
            },
        );
        nodes.insert(
            "n4".to_string(),
            TransformNode {
                id: "n4".to_string(),
                query_text: "SELECT * FROM n2".to_string(),
                materialize: MaterializeMode::View,
                depends_on: HashSet::from_iter(vec!["n2".to_string()]),
            },
        );

        let mut dag = Dag {
            db: "duckdb".to_string(),
            nodes: crate::graph::Graph::new(nodes),
            sources: vec![],
        };

        let mut pass = OMPPass::new(conn, engine, OptimizationMetric::Cost);
        let stats = pass.run(&mut dag).await.unwrap();

        assert_eq!(stats.get("metric").unwrap(), "Cost");
        assert!(stats.contains_key("baseline_value"));
        assert!(stats.contains_key("best_value"));
    }
}
