use async_trait::async_trait;
use itertools::{Itertools, repeat_n};
use log::debug;
use std::{collections::HashMap, marker::PhantomData, sync::Arc};

use crate::{
    connectors::Connector,
    dag::MaterializeMode,
    executor::{Executor, ExecutorError},
    opt::{Dag, OptimizerError, OptimizerPass},
};

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
    centrality: OMPCentrality,
    early_termination: bool,
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
        centrality: OMPCentrality,
        early_termination: bool,
    ) -> Self {
        Self {
            engine,
            top_n,
            centrality,
            early_termination,
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
        debug!("Running OMPPass with centrality: {:?}", self.centrality);
        let mut stats = HashMap::new();
        let _ = self.engine.cleanup(dag).await.unwrap();

        let baseline_cost = self
            .engine
            .run(dag)
            .await
            .map(|r| r.duration.num_milliseconds() as f32)
            .map_err(|e| OptimizerError::Exec(format!("couldn't get baseline runtime: {e}")))?;

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
            [MaterializeMode::View, MaterializeMode::TempTable].into_iter(),
            top_candidates.len(),
        )
        .multi_cartesian_product()
        .collect();

        let mut work_dag = dag.clone();
        let mut best_plan: Vec<MaterializeMode> = top_candidates
            .iter()
            .map(|c| dag.nodes.get(c.0.clone()).unwrap().materialize)
            .collect();
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

            let current_cost = if self.early_termination {
                let cancel_tx = self.engine.cancel_sender();
                // Reset any prior cancellation before starting the run.
                cancel_tx.send(false).ok();
                let budget_ms = best_cost as u64;
                let cancel_tx_timer = Arc::clone(&cancel_tx);
                let timer = tokio::spawn(async move {
                    tokio::time::sleep(std::time::Duration::from_millis(budget_ms)).await;
                    cancel_tx_timer.send(true).ok();
                });
                let result = self.engine.run(&work_dag).await;
                timer.abort();
                match result {
                    Ok(r) => r.duration.num_milliseconds() as f32,
                    Err(ExecutorError::Cancelled) => {
                        debug!("OMPPass: plan {} cancelled after {}ms", i + 1, budget_ms);
                        stats.insert(
                            format!("attempt_{}", i + 1),
                            format!("cancelled({})", budget_ms),
                        );
                        continue;
                    }
                    Err(e) => {
                        return Err(OptimizerError::Exec(format!(
                            "test dag run failed - {}",
                            e
                        )));
                    }
                }
            } else {
                self.engine
                    .run(&work_dag)
                    .await
                    .map(|r| r.duration.num_milliseconds() as f32)
                    .map_err(|e| OptimizerError::Exec(format!("test dag run failed - {}", e)))?
            };

            stats.insert(format!("attempt_{}", i + 1), current_cost.to_string());
            if self.early_termination {
                // Completed within the timeout window — it's the new best by definition
                best_cost = current_cost;
                best_plan = plan.clone();
            } else if current_cost < best_cost {
                best_cost = current_cost;
                best_plan = plan.clone();
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
            if matches!(node.1, MaterializeMode::TempTable) {
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
