use crate::{connectors::Connector, dag::Dag};

use async_trait::async_trait;
use futures::{StreamExt, stream::FuturesUnordered};
use log::debug;
use petgraph::Direction::{self};
use std::{collections::HashSet, sync::Arc};

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
    async fn run(&self, dag: Dag) -> Result<usize, ExecutorError>;
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

    async fn run(&self, dag: Dag) -> Result<usize, ExecutorError> {
        let mut work_graph = dag.graph.clone();
        let mut work_queue = FuturesUnordered::new();

        let mut total_size = 0;
        let mut in_progress = HashSet::new();
        while (work_graph.node_count() + work_queue.len()) > 0 {
            let next_nodes: Vec<_> = work_graph
                .externals(Direction::Incoming)
                .filter(|n| !in_progress.contains(n))
                .collect();

            // pop off all nodes with no dependencies and run them
            let transform_idx = next_nodes
                .clone()
                .iter()
                .map(|n| *work_graph.node_weight(*n).unwrap())
                .collect::<Vec<u32>>();

            for (tidx, nidx) in transform_idx.iter().zip(next_nodes) {
                let tn = dag.nodes.get(*tidx as usize).unwrap().clone();
                let conn = Arc::clone(&self.conn);
                debug!("running node tidx={}", tidx);
                debug!("work_queue.len()={}", work_queue.len());
                in_progress.insert(nidx);
                work_queue.push(tokio::spawn(async move {
                    let res = conn
                        .new_relation(tn.materialize, tn.id, tn.query_text)
                        .await
                        .map_err(|e| ExecutorError::Exec(e.to_string()))?;
                    Ok((res, nidx))
                }));
            }

            // wait for at least one of the nodes to finish;
            if let Some(res) = work_queue.next().await {
                let (rs_size, nidx) =
                    res.map_err(|j| ExecutorError::Exec(format!("join error - {}", j)))??;
                total_size += rs_size;
                work_graph.remove_node(nidx);
                in_progress.remove(&nidx);
            }
        }
        debug!("finished adding new work to the queue");
        debug!("work_queue.len()={}", work_queue.len());
        // wait for work_queue to empty
        while let Some(item) = work_queue.next().await {
            let (rs_size, nidx) =
                item.map_err(|j| ExecutorError::Exec(format!("join error - {}", j)))??;
            total_size += rs_size;
            work_graph.remove_node(nidx);
            in_progress.remove(&nidx);
        }
        debug!("work_queue cleared");
        Ok(total_size)
    }
}
