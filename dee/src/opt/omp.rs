use async_trait::async_trait;
use chrono::{TimeDelta, Utc};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use log::debug;
use std::{collections::HashSet, sync::Arc};

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
    conn: Arc<C>,
    engine: Arc<E>,
}

impl<C, E> OMPPass<C, E>
where
    C: Connector + Send + 'static + Sync,
    E: Executor<C> + Send + Sync,
{
    pub fn new(conn: Arc<C>, engine: Arc<E>) -> Self {
        Self { conn, engine }
    }
}

#[async_trait]
impl<C, E> OptimizerPass<C, E> for OMPPass<C, E>
where
    C: Connector + Send + 'static + Sync,
    E: Executor<C> + Send + Sync,
{
    async fn run(&mut self, dag: &mut Dag) -> Result<usize, OptimizerError> {
        debug!("Running OMPPass");
        let _ = self.engine.cleanup(dag).await.unwrap();
        let stats = self
            .engine
            .run(dag)
            .await
            .map_err(|_| OptimizerError::Exec("couldn't get baseline".to_string()))?;
        let mut best_runtime = stats.duration.num_milliseconds();
        let baseline_runtime = best_runtime;
        let mut candidates: Vec<String> = dag
            .nodes
            .nodes()
            .filter(|n| matches!(n.materialize, MaterializeMode::View))
            .map(|n| n.id.clone())
            .collect();

        let mut new_mats = HashSet::new();

        let mut work_dag = dag.clone();
        let iter_budget = 10;
        let mut i = 0;
        while i < iter_budget && candidates.len() > 0 {
            let _ = self.engine.cleanup(dag).await.unwrap();
            let this_idx = i % candidates.len();
            let test = candidates[this_idx].clone();
            work_dag
                .nodes
                .get_mut(test.clone())
                .ok_or(OptimizerError::Exec("missing node".to_string()))?
                .materialize = MaterializeMode::Table;
            let stats = self
                .engine
                .run(&work_dag)
                .await
                .map_err(|e| OptimizerError::Exec(format!("test dag run failed - {}", e)))?;

            if stats.duration.num_milliseconds() < best_runtime {
                let id = work_dag
                    .nodes
                    .get_mut(test)
                    .ok_or(OptimizerError::Exec("missing node".to_string()))?
                    .id
                    .clone();
                new_mats.insert(id);
                candidates.remove(this_idx);
                best_runtime = stats.duration.num_milliseconds();
            } else {
                work_dag
                    .nodes
                    .get_mut(test)
                    .ok_or(OptimizerError::Exec("missing node".to_string()))?
                    .materialize = MaterializeMode::View;
            }
            i += 1;
        }

        debug!(
            "OMPPass change: {:.2}ms -> {:.2}ms ({:.2}%)",
            baseline_runtime,
            best_runtime,
            ((best_runtime as f32 - baseline_runtime as f32) / (baseline_runtime as f32)) * 100.0
        );

        // now resolve
        for new_mat in new_mats {
            let node = dag
                .nodes
                .get_mut(new_mat)
                .ok_or(OptimizerError::Exec("couldn't apply changes".to_string()))?;
            node.materialize = MaterializeMode::Table;
        }

        Ok(0)
    }
}
