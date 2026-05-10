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

#[derive(Debug, Clone)]
pub struct OMPPass<C, E>
where
    C: Connector + Send + 'static + Sync,
    E: Executor<C> + Send + Sync,
{
    engine: Arc<E>,
    _phantom: PhantomData<C>,
}

impl<C, E> OMPPass<C, E>
where
    C: Connector + Send + 'static + Sync,
    E: Executor<C> + Send + Sync,
{
    pub fn new(_conn: Arc<C>, engine: Arc<E>) -> Self {
        Self {
            engine,
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
        debug!("Running OMPPass");
        let mut stats = HashMap::new();
        let top_n = 3;
        let _ = self.engine.cleanup(dag).await.unwrap();
        let run_result = self
            .engine
            .run(dag)
            .await
            .map_err(|_| OptimizerError::Exec("couldn't get baseline".to_string()))?;
        let mut best_runtime = run_result.duration.num_milliseconds();
        let baseline_runtime = best_runtime;
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
            let run_result = self
                .engine
                .run(&work_dag)
                .await
                .map_err(|e| OptimizerError::Exec(format!("test dag run failed - {}", e)))?;

            stats.insert(
                format!("attempt_{}", i + 1),
                run_result.duration.num_milliseconds().to_string(),
            );
            if run_result.duration.num_milliseconds() < best_runtime {
                best_runtime = run_result.duration.num_milliseconds();
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

        stats.insert("baseline_runtime".into(), baseline_runtime.to_string());
        stats.insert("best_runtime".into(), best_runtime.to_string());
        let change = (best_runtime as f32 - baseline_runtime as f32) / (baseline_runtime as f32);
        stats.insert("opt_change".into(), change.to_string());
        debug!(
            "OMPPass change: {:.2}ms -> {:.2}ms ({:.2}%)",
            baseline_runtime,
            best_runtime,
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
