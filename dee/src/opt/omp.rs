use async_trait::async_trait;
use itertools::{Itertools, repeat_n};
use log::debug;
use std::{collections::HashMap, marker::PhantomData, sync::Arc};

use crate::{
    connectors::Connector,
    dag::MaterializeMode,
    executor::Executor,
    opt::{Dag, OptimizerError, OptimizerPass},
};

#[derive(Debug, Clone, Copy, Default)]
pub enum OMPCostMetric {
    #[default]
    Actual,
    Estimate,
}

#[derive(Debug, Clone, Copy, Default)]
pub enum OMPCentrality {
    #[default]
    OutDegree,
    Paths,
}

#[derive(Debug, Clone)]
pub struct OMPPass<C, E>
where
    C: Connector + Send + 'static + Sync,
    E: Executor<C> + Send + Sync,
{
    engine: Arc<E>,
    top_n: Option<usize>,
    cost_metric: OMPCostMetric,
    centrality: OMPCentrality,
    _phantom: PhantomData<C>,
}

impl<C, E> OMPPass<C, E>
where
    C: Connector + Send + 'static + Sync,
    E: Executor<C> + Send + Sync,
{
    pub fn new(
        _conn: Arc<C>,
        engine: Arc<E>,
        top_n: Option<usize>,
        cost_metric: OMPCostMetric,
        centrality: OMPCentrality,
    ) -> Self {
        Self {
            engine,
            top_n,
            cost_metric,
            centrality,
            _phantom: PhantomData,
        }
    }
}

#[async_trait]
impl<C, E> OptimizerPass<C, E> for OMPPass<C, E>
where
    C: Connector + Send + 'static + Sync,
    E: Executor<C> + Send + Sync,
{
    async fn run(&mut self, dag: &mut Dag) -> Result<HashMap<String, String>, OptimizerError> {
        debug!(
            "Running OMPPass with metric: {:?}, centrality: {:?}",
            self.cost_metric, self.centrality
        );
        let mut stats = HashMap::new();
        let _ = self.engine.cleanup(dag).await.unwrap();

        let baseline_cost = match self.cost_metric {
            OMPCostMetric::Actual => self
                .engine
                .run(dag)
                .await
                .map(|r| r.duration.num_milliseconds() as f32)
                .map_err(|_| OptimizerError::Exec("couldn't get baseline runtime".to_string()))?,
            OMPCostMetric::Estimate => self
                .engine
                .cost(dag)
                .await
                .map_err(|_| OptimizerError::Exec("couldn't get baseline cost".to_string()))?
                .ok_or(OptimizerError::Exec(
                    "no cost estimate available".to_string(),
                ))?,
        };

        let mut best_cost = baseline_cost;
        let mut candidates: Vec<(String, usize)> = dag
            .nodes
            .nodes()
            .filter(|n| matches!(n.materialize, MaterializeMode::View))
            .cloned()
            .map(|n| {
                let rank = match self.centrality {
                    OMPCentrality::OutDegree => dag.nodes.out_degree(&n.id),
                    OMPCentrality::Paths => dag.nodes.paths_to_sinks(&n.id),
                };
                (n.id.clone(), rank)
            })
            .filter(|(_, d)| *d > 1)
            .collect();

        candidates.sort_by_key(|k| k.1);
        let top_candidates: Vec<_> = if let Some(n) = self.top_n {
            candidates.iter().rev().take(n).collect()
        } else {
            candidates.iter().rev().collect()
        };

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
            debug!("OMPPass: iter {}", i + 1);
            let _ = self.engine.cleanup(dag).await.unwrap();

            for node in plan.iter().enumerate() {
                let node_id = top_candidates.get(node.0).unwrap().0.clone();
                work_dag
                    .nodes
                    .get_mut(node_id.clone())
                    .ok_or(OptimizerError::Exec("missing node".to_string()))?
                    .materialize = node.1.clone();
            }

            let current_cost = match self.cost_metric {
                OMPCostMetric::Actual => self
                    .engine
                    .run(&work_dag)
                    .await
                    .map(|r| r.duration.num_milliseconds() as f32)
                    .map_err(|e| OptimizerError::Exec(format!("test dag run failed - {}", e)))?,
                OMPCostMetric::Estimate => self
                    .engine
                    .cost(&work_dag)
                    .await
                    .map_err(|e| OptimizerError::Exec(format!("test dag cost failed - {}", e)))?
                    .ok_or(OptimizerError::Exec(
                        "no cost estimate available".to_string(),
                    ))?,
            };

            stats.insert(format!("attempt_{}", i + 1), current_cost.to_string());
            if current_cost < best_cost {
                best_cost = current_cost;
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

        stats.insert("baseline_value".into(), baseline_cost.to_string());
        stats.insert("best_value".into(), best_cost.to_string());
        let change = (best_cost - baseline_cost) / (baseline_cost);
        stats.insert("opt_change".into(), change.to_string());
        debug!(
            "OMPPass change: {:.2} -> {:.2} ({:.2}%)",
            baseline_cost,
            best_cost,
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
}
