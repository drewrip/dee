use async_trait::async_trait;
use chrono::{DateTime, TimeDelta, Utc};
use futures::{StreamExt, stream::FuturesUnordered};
use log::{debug, warn};

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::Duration,
};

use thiserror::Error;
use tokio::sync::{Mutex, watch};

use crate::{
    connectors::Connector,
    dag::{Dag, MaterializeMode},
    profile::SystemUsageSample,
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
    plans_dir: Option<String>,
    profiling: Option<ProfilingConfig>,
}

impl<C> SimpleEngine<C>
where
    C: Connector,
{
    pub fn with_plans_dir(mut self, plans_dir: String) -> Self {
        self.plans_dir = Some(plans_dir);
        self
    }

    pub fn with_profiling(mut self, profiling: ProfilingConfig) -> Self {
        self.profiling = Some(profiling);
        self
    }
}

#[derive(Clone, Debug)]
pub struct ProfilingConfig {
    pub sample_interval: Duration,
    pub collect_plans: bool,
}

impl Default for ProfilingConfig {
    fn default() -> Self {
        Self {
            sample_interval: Duration::from_millis(250),
            collect_plans: false,
        }
    }
}

async fn sample_connector_usage<C>(
    conn: &Arc<C>,
    start: DateTime<Utc>,
) -> Result<SystemUsageSample, crate::connectors::ConnectorError>
where
    C: Connector + Send + Sync + 'static,
{
    let timestamp = Utc::now();
    let cpu_percent = conn.sample_system_cpu_usage().await?;
    let memory_bytes = conn.sample_system_memory_usage().await?;
    Ok(SystemUsageSample {
        timestamp,
        elapsed_ms: (timestamp - start).num_milliseconds(),
        cpu_percent,
        memory_bytes,
    })
}

async fn spawn_sampler<C>(
    conn: Arc<C>,
    profiling: ProfilingConfig,
    start: DateTime<Utc>,
) -> (
    watch::Sender<bool>,
    tokio::task::JoinHandle<Vec<SystemUsageSample>>,
)
where
    C: Connector + Send + Sync + 'static,
{
    let (stop_tx, mut stop_rx) = watch::channel(false);
    let samples = Arc::new(Mutex::new(Vec::new()));
    let sampler_samples = Arc::clone(&samples);
    let handle = tokio::spawn(async move {
        if let Ok(sample) = sample_connector_usage(&conn, start).await {
            sampler_samples.lock().await.push(sample);
        }

        let mut interval = tokio::time::interval(profiling.sample_interval);
        interval.tick().await;

        loop {
            tokio::select! {
                changed = stop_rx.changed() => {
                    if changed.is_err() || *stop_rx.borrow() {
                        break;
                    }
                }
                _ = interval.tick() => {
                    match sample_connector_usage(&conn, start).await {
                        Ok(sample) => sampler_samples.lock().await.push(sample),
                        Err(err) => warn!("failed to collect profiling sample: {}", err),
                    }
                }
            }
        }

        if let Ok(sample) = sample_connector_usage(&conn, start).await {
            sampler_samples.lock().await.push(sample);
        }

        samples.lock().await.clone()
    });

    (stop_tx, handle)
}

#[async_trait]
impl<C> Executor<C> for SimpleEngine<C>
where
    C: Connector + Send + Sync + 'static,
{
    type ExecutionEngine = Self;

    fn new(conn: Arc<C>) -> Result<SimpleEngine<C>, ExecutorError> {
        Ok(SimpleEngine {
            conn,
            plans_dir: None,
            profiling: None,
        })
    }

    async fn run(&self, dag: &Dag) -> Result<ExecStats, ExecutorError> {
        let mut work_graph = dag.nodes.clone();
        let mut work_queue = FuturesUnordered::new();
        let initial_size = work_graph.num_nodes();
        let mut finished = 0;
        let mut in_progress = HashSet::new();

        let node_stats = HashMap::new();
        let start = Utc::now();
        let (sampler_stop, sampler_handle) = if let Some(profiling) = self.profiling.clone() {
            let (stop, handle) = spawn_sampler(Arc::clone(&self.conn), profiling, start).await;
            (Some(stop), Some(handle))
        } else {
            (None, None)
        };

        let collect_plans = self.profiling.as_ref().map(|p| p.collect_plans).unwrap_or(false);

        let mut node_stats = node_stats;
        while work_graph.num_nodes() > 0 {
            let next_nodes: Vec<_> = work_graph
                .sources()
                .filter(|n| !in_progress.contains(n))
                .collect();

            // pop off all nodes with no dependencies and run them
            for node_id in next_nodes.into_iter() {
                let tn = dag.nodes.get(node_id.clone()).unwrap().clone();
                let conn = Arc::clone(&self.conn);
                let plans_dir = self.plans_dir.clone();
                let collect_plans = collect_plans;
                debug!("running node tidx={}", node_id);
                debug!("work_queue.len()={}", work_queue.len());
                in_progress.insert(node_id.clone());
                work_queue.push(tokio::spawn(async move {
                    let node_start = Utc::now();
                    let (res, plan) = if plans_dir.is_some() || collect_plans {
                        let (res, plan) = conn
                            .new_relation_and_explain(tn.materialize, tn.id.clone(), tn.query_text)
                            .await
                            .map_err(|e| ExecutorError::Exec(e.to_string()))?;

                        if let Some(plan_str) = plan.clone() {
                            if let Some(dir) = plans_dir {
                                let rel_type = match tn.materialize {
                                    MaterializeMode::Table => "table",
                                    MaterializeMode::View => "view",
                                };
                                let filename = format!("{}_{}.json", tn.id, rel_type);
                                let path = std::path::Path::new(&dir).join(filename);
                                if let Some(parent) = path.parent() {
                                    std::fs::create_dir_all(parent)
                                        .map_err(|e| ExecutorError::Exec(e.to_string()))?;
                                }
                                std::fs::write(path, plan_str)
                                    .map_err(|e| ExecutorError::Exec(e.to_string()))?;
                            }
                        }
                        (res, plan)
                    } else {
                        let res = conn.new_relation(tn.materialize, tn.id.clone(), tn.query_text)
                            .await
                            .map_err(|e| ExecutorError::Exec(e.to_string()))?;
                        (res, None)
                    };
                    let node_finish = Utc::now();

                    debug!("new_relation ({}, {:?})", tn.id, tn.materialize);
                    Ok((
                        res,
                        node_id.clone(),
                        NodeStats {
                            start: node_start,
                            finish: node_finish,
                            duration: node_finish - node_start,
                            plan,
                        },
                    ))
                }));
            }
            // wait for work_queue to empty
            while let Some(item) = work_queue.next().await {
                let (_, node_id, stats) =
                    item.map_err(|j| ExecutorError::Exec(format!("join error - {}", j)))??;
                debug!("recv result for nidx={:?}", node_id);
                in_progress.remove(&node_id);
                work_graph.remove(node_id.clone());
                node_stats.insert(node_id.clone(), stats);
                finished += 1;
                debug!("finished {}/{} nodes", finished, initial_size);
            }
        }
        debug!("work_queue cleared");
        let finish = Utc::now();

        let system_samples = if let (Some(stop), Some(handle)) = (sampler_stop, sampler_handle) {
            let _ = stop.send(true);
            handle
                .await
                .map_err(|j| ExecutorError::Exec(format!("sampler join error - {}", j)))?
        } else {
            Vec::new()
        };

        let exec_stats = ExecStats {
            start,
            finish,
            duration: finish - start,
            node_stats,
            system_samples,
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
    pub system_samples: Vec<SystemUsageSample>,
}

#[derive(Clone, Debug)]
pub struct NodeStats {
    pub start: DateTime<Utc>,
    pub finish: DateTime<Utc>,
    pub duration: TimeDelta,
    pub plan: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectors::duckdb::{DuckDBConfig, DuckDBConnection};
    use crate::dag::TransformNode;
    use std::collections::HashSet;

    #[tokio::test]
    async fn test_simple_engine_cost() {
        let config = DuckDBConfig::new_from_path(":memory:".to_string());
        let conn = DuckDBConnection::new(config).await.unwrap();
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
