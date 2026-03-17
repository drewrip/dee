use crate::{connectors::Connector, dag::Dag};

use async_trait::async_trait;
use futures::{StreamExt, stream::FuturesUnordered};
use petgraph::Direction::{self};
use std::sync::Arc;

use thiserror::Error;

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
    async fn run(&mut self, dag: Dag) -> Result<usize, ExecutorError>;
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

    async fn run(&mut self, dag: Dag) -> Result<usize, ExecutorError> {
        let mut work_graph = dag.graph.clone();
        let mut work_queue = FuturesUnordered::new();

        let mut total_size = 0;
        while work_graph.node_count() > 0 {
            let next_nodes: Vec<_> = work_graph.externals(Direction::Incoming).collect();

            // pop off all nodes with no dependencies and run them
            let transform_idx = next_nodes
                .clone()
                .iter()
                .map(|n| *work_graph.node_weight(*n).unwrap())
                .collect::<Vec<u32>>();

            for tidx in &transform_idx {
                let tn = dag.nodes.get(*tidx as usize).unwrap().clone();
                let conn = Arc::clone(&self.conn);
                work_queue.push(tokio::spawn(async move {
                    conn.execute(tn.query_text)
                        .await
                        .map_err(|e| ExecutorError::Exec(format!("exec error - {}", e)))
                }));
            }

            // Remove queued nodes from the Graph
            for node in next_nodes {
                work_graph.remove_node(node);
            }

            // wait for at least one of the nodes to finish;
            if let Some(res) = work_queue.next().await {
                total_size +=
                    res.map_err(|j| ExecutorError::Exec(format!("join error - {}", j)))??;
            }
        }
        // wait for work_queue to empty
        while let Some(item) = work_queue.next().await {
            total_size +=
                item.map_err(|j| ExecutorError::Exec(format!("join error - {}", j)))??;
        }
        Ok(total_size)
    }
}
