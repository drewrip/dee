use async_trait::async_trait;
use chrono::{DateTime, TimeDelta, Utc};
use tokio::task::JoinSet;
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
    #[error("execution cancelled")]
    Cancelled,
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
    fn cancel_sender(&self) -> Arc<watch::Sender<bool>>;

    /// Resolve the Arrow output schema for every node in `dag` and store it in
    /// `TransformNode::schema`.
    ///
    /// Strategy:
    /// 1. Walk nodes in topological order and create each as a **VIEW**
    ///    (regardless of its configured `materialize` mode).  Because the DB
    ///    engine resolves the SQL itself, this works for any SQL dialect —
    ///    including dialect-specific functions that DataFusion cannot plan.
    /// 2. Walk nodes in topological order again and call `get_schema` on each
    ///    to retrieve the output Arrow schema from the live DB.
    /// 3. Store the resolved schema on the corresponding `TransformNode`.
    /// 4. Clean up all views created in step 1.
    ///
    /// Fails if any view cannot be created, any schema cannot be fetched, or
    /// cleanup fails.
    async fn resolve_schemas(&self, dag: &mut Dag) -> Result<(), ExecutorError>;
}

#[derive(Debug)]
pub struct SimpleEngine<C>
where
    C: Connector,
{
    conn: Arc<C>,
    plans_dir: Option<String>,
    profiling: Option<ProfilingConfig>,
    cancel_tx: Arc<watch::Sender<bool>>,
    cancel_rx: watch::Receiver<bool>,
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
        let (cancel_tx, cancel_rx) = watch::channel(false);
        Ok(SimpleEngine {
            conn,
            plans_dir: None,
            profiling: None,
            cancel_tx: Arc::new(cancel_tx),
            cancel_rx,
        })
    }

    fn cancel_sender(&self) -> Arc<watch::Sender<bool>> {
        Arc::clone(&self.cancel_tx)
    }

    async fn run(&self, dag: &Dag) -> Result<ExecStats, ExecutorError> {
        let mut work_graph = dag.nodes.clone();
        let mut work_queue: JoinSet<Result<(usize, String, NodeStats), ExecutorError>> =
            JoinSet::new();
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

        let collect_plans = self
            .profiling
            .as_ref()
            .map(|p| p.collect_plans)
            .unwrap_or(false);

        let mut node_stats = node_stats;
        while work_graph.num_nodes() > 0 {
            if *self.cancel_rx.borrow() {
                debug!("SimpleEngine: cancellation requested, stopping execution");
                // Abort all queued tasks and wait for them to finish before
                // cleaning up.  Simply dropping a JoinHandle detaches the task
                // rather than cancelling it, so in-flight CREATE TABLE/VIEW
                // statements would race against the cleanup that follows.
                // abort_all() + drain guarantees no tasks are still modifying
                // the database when cleanup runs.
                work_queue.abort_all();
                while work_queue.join_next().await.is_some() {}
                if let Some(ref stop) = sampler_stop {
                    let _ = stop.send(true);
                }
                self.cleanup(dag).await?;
                return Err(ExecutorError::Cancelled);
            }

            let next_nodes: Vec<_> = work_graph
                .sources()
                .filter(|n| !in_progress.contains(n))
                .collect();

            debug!("next_nodes = {}", next_nodes.len());

            // queue all currently-runnable nodes
            for node_id in next_nodes.into_iter() {
                let tn = dag.nodes.get(node_id.clone()).unwrap().clone();
                let conn = Arc::clone(&self.conn);
                let plans_dir = self.plans_dir.clone();
                let collect_plans = collect_plans;
                debug!("running node tidx={}", node_id);
                debug!("work_queue.len()={}", work_queue.len());
                in_progress.insert(node_id.clone());
                work_queue.spawn(async move {
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
                                    MaterializeMode::TempTable => "temp_table",
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
                        let res = conn
                            .new_relation(tn.materialize, tn.id.clone(), tn.query_text)
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
                });
            }
            // wait for one node to finish, then loop back to queue any newly-runnable nodes
            if let Some(item) = work_queue.join_next().await {
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
            num_deleted += self
                .conn
                .drop_relation(MaterializeMode::TempTable, node.id.clone())
                .await
                .unwrap_or(0);
        }
        debug!("cleanup, {} relations dropped", num_deleted);
        Ok(num_deleted)
    }

    async fn resolve_schemas(&self, dag: &mut Dag) -> Result<(), ExecutorError> {
        const PREFIX: &str = "dee_tmp_";

        let topo = dag.nodes.topological_sort();

        // Build a mapping from original node ID → prefixed node ID so we can
        // rename every reference inside query_text and depends_on.  The prefix
        // keeps these short-lived views from colliding with any relation the
        // real DAG execution has already materialised.
        let rename_map: std::collections::HashMap<String, String> = topo
            .iter()
            .map(|id| {
                // Strip any schema prefix before inserting the dee_tmp_ tag so
                // the final name stays valid SQL.
                // e.g. "warehouse"."main"."foo" → "warehouse"."main"."dee_tmp_foo"
                let prefixed = if let Some(pos) = id.rfind("\".\"") {
                    let schema_part = &id[..pos + 2]; // up to and including the last `".`
                    let bare = id[pos + 3..].trim_matches('"');
                    format!("{schema_part}\"{PREFIX}{bare}\"")
                } else {
                    format!("{PREFIX}{id}")
                };
                (id.clone(), prefixed)
            })
            .collect();

        // Clone the DAG and apply renames so every node ID, depends_on entry,
        // and query_text reference uses the prefixed name.
        let mut tmp_dag = dag.clone();
        // Replace node IDs and their query text / deps in the cloned graph.
        // We rebuild the node map entirely to avoid borrow conflicts.
        let original_nodes: Vec<_> = topo
            .iter()
            .filter_map(|id| tmp_dag.nodes.get(id.clone()))
            .collect();

        let mut renamed_nodes: Vec<crate::dag::TransformNode> = original_nodes
            .into_iter()
            .map(|n| {
                let new_id = rename_map[&n.id].clone();

                // Rewrite query_text: replace every original node name with its
                // prefixed counterpart.  Longer names are replaced first to
                // avoid partial matches (e.g. "foo" matching inside "foo_bar").
                let mut new_query = n.query_text.clone();
                let mut sorted_originals: Vec<&String> = rename_map.keys().collect();
                sorted_originals.sort_by_key(|k| std::cmp::Reverse(k.len()));
                for orig in sorted_originals {
                    new_query = new_query.replace(orig.as_str(), &rename_map[orig]);
                }

                let new_deps = n
                    .depends_on
                    .iter()
                    .map(|dep| rename_map.get(dep).cloned().unwrap_or_else(|| dep.clone()))
                    .collect();

                crate::dag::TransformNode {
                    id: new_id,
                    query_text: new_query,
                    materialize: n.materialize.clone(),
                    depends_on: new_deps,
                    schema: None,
                }
            })
            .collect();

        // Replace tmp_dag's graph with the renamed nodes.
        let mut new_graph = crate::graph::Graph::new(std::collections::HashMap::new());
        for node in renamed_nodes.drain(..) {
            new_graph
                .add_node(node)
                .map_err(|e| ExecutorError::Exec(format!("resolve_schemas: rename failed: {e}")))?;
        }
        tmp_dag.nodes = new_graph;

        // Step 1 — create every renamed node as a VIEW in the DB.
        // Walk in the original topological order (sources first) using rename_map
        // to look up each prefixed ID — this avoids relying on tmp_dag's sort order,
        // which can differ due to HashMap non-determinism.
        debug!("resolve_schemas: creating {} node(s) as views", topo.len());
        for orig_id in &topo {
            let tmp_id = &rename_map[orig_id];
            let node = tmp_dag
                .nodes
                .get(tmp_id.clone())
                .ok_or_else(|| ExecutorError::Exec(format!("node '{tmp_id}' not found")))?;

            self.conn
                .new_relation(MaterializeMode::View, node.id.clone(), node.query_text.clone())
                .await
                .map_err(|e| {
                    ExecutorError::Exec(format!(
                        "resolve_schemas: failed to create view for '{tmp_id}': {e}"
                    ))
                })?;
        }

        // Step 2 — fetch schemas using the prefixed names; map results back to
        // the original node IDs so they can be stored on `dag`.
        // Iterate over `topo` (the original order) and look up the prefixed name
        // via rename_map — never zip with tmp_topo, whose order may differ because
        // the rebuilt HashMap has non-deterministic iteration order.
        debug!("resolve_schemas: fetching schemas for {} node(s)", topo.len());
        let mut resolved: Vec<(String, datafusion::arrow::datatypes::SchemaRef)> = Vec::new();
        for orig_id in &topo {
            let tmp_id = &rename_map[orig_id];
            match self.conn.get_schema(tmp_id.clone()).await {
                Some(Ok(schema)) => {
                    debug!("resolve_schemas: resolved schema for '{orig_id}' (via '{tmp_id}')");
                    resolved.push((orig_id.clone(), schema));
                }
                Some(Err(e)) => {
                    let _ = self.cleanup(&tmp_dag).await;
                    return Err(ExecutorError::Exec(format!(
                        "resolve_schemas: get_schema failed for '{orig_id}': {e}"
                    )));
                }
                None => {
                    let _ = self.cleanup(&tmp_dag).await;
                    return Err(ExecutorError::Exec(format!(
                        "resolve_schemas: no schema available for '{orig_id}'"
                    )));
                }
            }
        }

        // Step 3 — store the schemas on the original DAG nodes.
        for (node_id, schema) in resolved {
            if let Some(node) = dag.nodes.get_mut(node_id.clone()) {
                node.schema = Some(schema);
            }
        }

        // Step 4 — clean up the prefixed temporary views; never touches the
        // real DAG's relations.
        self.cleanup(&tmp_dag).await.map_err(|e| {
            ExecutorError::Exec(format!("resolve_schemas: cleanup failed: {e}"))
        })?;

        debug!("resolve_schemas: complete");
        Ok(())
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
