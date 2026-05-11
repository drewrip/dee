use async_trait::async_trait;
use chrono::{DateTime, TimeDelta, Utc};
use futures::{StreamExt, stream::FuturesUnordered};
use log::debug;

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use thiserror::Error;

use crate::{
    connectors::Connector,
    dag::{Dag, MaterializeMode},
};

#[derive(Error, Debug)]
pub enum ExecutorError {
    #[error("couldn't execute DAG - {0}")]
    Exec(String),
}

#[async_trait]
pub trait Executor<C>
where
    C: Connector + Send,
{
    type ExecutionEngine;

    fn new(conn: Arc<C>) -> Result<Self::ExecutionEngine, ExecutorError>;
    async fn run(&self, dag: &Dag) -> Result<ExecStats, ExecutorError>;
    async fn cleanup(&self, dag: &Dag) -> Result<usize, ExecutorError>;
    async fn cost(&self, dag: &Dag) -> Result<Option<f32>, ExecutorError>;
}

#[derive(Debug)]
pub struct SimpleEngine<C>
where
    C: Connector,
{
    conn: Arc<C>,
}

#[async_trait]
impl<C> Executor<C> for SimpleEngine<C>
where
    C: Connector + Send + Sync + 'static,
{
    type ExecutionEngine = Self;

    fn new(conn: Arc<C>) -> Result<SimpleEngine<C>, ExecutorError> {
        Ok(SimpleEngine { conn })
    }

    async fn run(&self, dag: &Dag) -> Result<ExecStats, ExecutorError> {
        let mut work_graph = dag.nodes.clone();
        let mut work_queue = FuturesUnordered::new();
        let initial_size = work_graph.num_nodes();
        let mut finished = 0;
        let mut in_progress = HashSet::new();

        let node_stats = HashMap::new();
        let start = Utc::now();
        while work_graph.num_nodes() > 0 {
            let next_nodes: Vec<_> = work_graph
                .sources()
                .filter(|n| !in_progress.contains(n))
                .collect();

            // pop off all nodes with no dependencies and run them
            for node_id in next_nodes.into_iter() {
                let tn = dag.nodes.get(node_id.clone()).unwrap().clone();
                let conn = Arc::clone(&self.conn);
                debug!("running node tidx={}", node_id);
                debug!("work_queue.len()={}", work_queue.len());
                in_progress.insert(node_id.clone());
                work_queue.push(tokio::spawn(async move {
                    let res = conn
                        .new_relation(tn.materialize, tn.id.clone(), tn.query_text)
                        .await
                        .map_err(|e| ExecutorError::Exec(e.to_string()))?;
                    debug!("new_relation ({}, {:?})", tn.id, tn.materialize);
                    Ok((res, node_id.clone()))
                }));
            }

            // wait for work_queue to empty
            while let Some(item) = work_queue.next().await {
                let (_, node_id) =
                    item.map_err(|j| ExecutorError::Exec(format!("join error - {}", j)))??;
                debug!("recv result for nidx={:?}", node_id);
                in_progress.remove(&node_id);
                work_graph.remove(node_id.clone());
                finished += 1;
                debug!("finished {}/{} nodes", finished, initial_size);
            }
        }
        debug!("work_queue cleared");
        let finish = Utc::now();

        let exec_stats = ExecStats {
            start,
            finish,
            duration: finish - start,
            node_stats,
        };
        Ok(exec_stats)
    }

    async fn cleanup(&self, dag: &Dag) -> Result<usize, ExecutorError> {
        let mut num_deleted = 0;
        for node in dag.nodes.nodes() {
            num_deleted += self
                .conn
                .drop_relation(MaterializeMode::View, node.id.clone())
                .await
                .unwrap_or(0);
            num_deleted += self
                .conn
                .drop_relation(MaterializeMode::Table, node.id.clone())
                .await
                .unwrap_or(0);
        }
        debug!("cleanup, {} relations dropped", num_deleted);
        Ok(num_deleted)
    }

    async fn cost(&self, dag: &Dag) -> Result<Option<f32>, ExecutorError> {
        let sorted_nodes = dag.nodes.topological_sort();
        self.conn
            .execute("SET disabled_optimizers = 'cte_filter_pusher';".to_string())
            .await
            .map_err(|e| ExecutorError::Exec(e.to_string()))?;

        let mut total_cost = 0.0;
        let mut cost_exists = false;

        for (i, node_id) in sorted_nodes.into_iter().enumerate() {
            let node = dag
                .nodes
                .get(node_id.clone())
                .ok_or(ExecutorError::Exec("node missing".to_string()))?;

            match node.materialize {
                MaterializeMode::View => {
                    self.conn
                        .new_relation(
                            MaterializeMode::View,
                            node.id.clone(),
                            node.query_text.clone(),
                        )
                        .await
                        .map_err(|e| ExecutorError::Exec(e.to_string()))?;
                }
                MaterializeMode::Table => {
                    let wrapped_query = format!(
                        "WITH __dee_dummy_scan_{} AS MATERIALIZED ({}) SELECT * FROM __dee_dummy_scan_{}",
                        i, node.query_text, i
                    );

                    let cost_res = self
                        .conn
                        .cost(wrapped_query.clone())
                        .await
                        .map_err(|e| ExecutorError::Exec(e.to_string()))?;

                    if let Some(c) = cost_res {
                        total_cost += c;
                        cost_exists = true;
                    }

                    self.conn
                        .new_relation(MaterializeMode::View, node.id.clone(), wrapped_query)
                        .await
                        .map_err(|e| ExecutorError::Exec(e.to_string()))?;
                }
            }
        }

        self.conn
            .execute("SET disabled_optimizers = '';".to_string())
            .await
            .map_err(|e| ExecutorError::Exec(e.to_string()))?;

        if cost_exists {
            Ok(Some(total_cost))
        } else {
            Ok(None)
        }
    }
}

#[derive(Clone, Debug)]
pub struct ExecStats {
    pub start: DateTime<Utc>,
    pub finish: DateTime<Utc>,
    pub duration: TimeDelta,
    pub node_stats: HashMap<String, NodeStats>,
}

#[derive(Clone, Debug)]
pub struct NodeStats {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectors::duckdb::{DuckDBConnection, DuckDBProfile};
    use crate::dag::TransformNode;
    use std::collections::HashSet;

    #[tokio::test]
    async fn test_simple_engine_cost() {
        let profile = DuckDBProfile::new_from_path(":memory:".to_string());
        let conn = DuckDBConnection::new(profile).await.unwrap();
        let engine = SimpleEngine::new(conn.clone()).unwrap();

        let mut nodes = HashMap::new();
        nodes.insert(
            "n1".to_string(),
            TransformNode {
                id: "n1".to_string(),
                query_text: "SELECT 1 AS id".to_string(),
                materialize: MaterializeMode::Table,
                depends_on: HashSet::new(),
            },
        );
        nodes.insert(
            "n2".to_string(),
            TransformNode {
                id: "n2".to_string(),
                query_text: "SELECT * FROM n1".to_string(),
                materialize: MaterializeMode::View,
                depends_on: HashSet::from_iter(vec!["n1".to_string()]),
            },
        );

        let dag = Dag {
            db: "duckdb".to_string(),
            nodes: crate::graph::Graph::new(nodes),
            sources: vec![],
        };

        let cost = engine.cost(&dag).await.unwrap();
        assert!(cost.is_some());
        assert!(cost.unwrap() > 0.0);
    }
}
