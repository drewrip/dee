use async_trait::async_trait;
use chrono::{DateTime, TimeDelta, Utc};
use log::{debug, warn};
use serde::{Deserialize, Serialize};
use tokio::task::JoinSet;

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::Duration,
};

use thiserror::Error;
use tokio::sync::{Mutex, watch};

use crate::{
    connectors::Connector,
    dag::{Dag, MaterializeMode, TransformNode},
    graph::Graph,
    profile::SystemUsageSample,
};

/// Split a possibly-qualified, possibly double-quoted SQL identifier (e.g.
/// `foo`, `"schema".foo`, `"cat"."schema"."foo bar"`) into its dot-separated
/// parts, stripping quotes and unescaping doubled quotes (`""` -> `"`) within
/// quoted parts. A `.` inside a quoted part is not treated as a separator.
fn split_qualified_identifier(id: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = id.chars().peekable();

    while let Some(c) = chars.next() {
        match c {
            '"' => {
                if in_quotes && chars.peek() == Some(&'"') {
                    current.push('"');
                    chars.next();
                } else {
                    in_quotes = !in_quotes;
                }
            }
            '.' if !in_quotes => parts.push(std::mem::take(&mut current)),
            _ => current.push(c),
        }
    }
    parts.push(current);
    parts
}

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

    /// Clear a latched cancellation so this engine can run again.
    ///
    /// `run` returns `ExecutorError::Cancelled` as soon as it observes the
    /// flag, but nothing lowers it again, so a cancelled engine would refuse
    /// every subsequent `run`. Callers that keep an engine alive across
    /// executions must call this after handling a cancellation.
    pub fn reset_cancel(&self) {
        let _ = self.cancel_tx.send(false);
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
    let disk = conn.sample_system_disk_usage().await?;
    Ok(SystemUsageSample {
        timestamp,
        elapsed_ms: (timestamp - start).num_milliseconds(),
        cpu_percent,
        memory_bytes,
        disk_bytes: disk.disk_bytes,
        read_bytes: disk.read_bytes,
        written_bytes: disk.written_bytes,
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

            // Every node whose dependencies are met, capped by the DAG's
            // parallelism setting. The cap is what makes the loop below a
            // scheduler rather than a fan-out: with `None` every runnable node
            // starts the moment it becomes runnable, which is what dee has
            // always done and remains the default.
            //
            // Sorted before the cap is applied, because `sources()` walks a
            // HashMap and its order is not stable across runs. Unbounded that
            // is harmless -- everything runnable is started either way -- but
            // under a cap it decides *which* nodes are deferred, and a
            // ParallelismTuning trial that scheduled a different subset each
            // time would be measuring the ordering rather than the setting.
            let available = match dag.max_parallelism {
                Some(limit) => limit.max(1).saturating_sub(in_progress.len()),
                None => usize::MAX,
            };
            let mut runnable: Vec<_> = work_graph
                .sources()
                .filter(|n| !in_progress.contains(n))
                .collect();
            runnable.sort();
            let next_nodes: Vec<_> = runnable.into_iter().take(available).collect();

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
                            // Rows the backend reported writing. Meaningful
                            // for TABLE/TEMP_TABLE (CTAS row count); a VIEW
                            // writes nothing, so this is 0 there.
                            rows_produced: Some(res as u64),
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

        // Parse a possibly-qualified, possibly-quoted identifier (e.g.
        // `"cat"."schema"."table"`) into its dot-separated parts, then
        // re-quote every part with double quotes so the result is always a
        // valid DuckDB qualified identifier.
        let prefixed_name = |id: &str| -> String {
            match split_qualified_identifier(id).as_slice() {
                [catalog, schema, table] => {
                    format!("\"{catalog}\".\"{schema}\".\"{PREFIX}{table}\"")
                }
                [schema, table] => {
                    format!("\"{schema}\".\"{PREFIX}{table}\"")
                }
                [table] => {
                    format!("{PREFIX}{table}")
                }
                _ => format!("{PREFIX}{id}"),
            }
        };

        let topo = dag.nodes.topological_sort();

        // Build a mapping from original node ID → prefixed node ID so we can
        // rename every reference inside query_text and depends_on.  The prefix
        // keeps these short-lived views from colliding with any relation the
        // real DAG execution has already materialised.
        let rename_map: HashMap<String, String> = topo
            .iter()
            .map(|id| (id.clone(), prefixed_name(id)))
            .collect();

        // Clone the DAG and apply renames so every node ID, depends_on entry,
        // and query_text reference uses the prefixed name.
        let mut tmp_dag = dag.clone();
        let original_nodes: Vec<_> = topo
            .iter()
            .filter_map(|id| tmp_dag.nodes.get(id.clone()))
            .collect();

        let mut renamed_nodes: Vec<TransformNode> = original_nodes
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

                TransformNode {
                    id: new_id,
                    query_text: new_query,
                    materialize: n.materialize.clone(),
                    depends_on: new_deps,
                    schema: None,
                }
            })
            .collect();

        let mut new_graph = Graph::new(HashMap::new());
        for node in renamed_nodes.drain(..) {
            new_graph
                .add_node(node)
                .map_err(|e| ExecutorError::Exec(format!("resolve_schemas: rename failed: {e}")))?;
        }
        tmp_dag.nodes = new_graph;

        // Build prefixed names for DAG source tables.  These are created as
        // temporary views so we can get up-to-date schemas directly from the DB
        // rather than relying on whatever was serialised in the DAG file.
        let source_tmp_names: Vec<String> = dag.sources.iter()
            .map(|src| prefixed_name(&src.name))
            .collect();

        // Helper: drop all source temp views (best-effort, used in error paths).
        let drop_source_tmps = |conn: &Arc<C>, names: &[String]| {
            let conn = Arc::clone(conn);
            let names = names.to_vec();
            async move {
                for name in names {
                    conn.drop_relation(MaterializeMode::View, name).await.ok();
                }
            }
        };

        // Step 1 — create all node temp views in topological order.
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
                .map_err(|e| ExecutorError::Exec(format!(
                    "resolve_schemas: failed to create view for '{tmp_id}': {e}"
                )))?;
        }

        // Step 1b — create one temp VIEW per source using SELECT * FROM <name> LIMIT 0
        // so the DB engine resolves the live schema for us.
        debug!("resolve_schemas: creating {} source view(s)", dag.sources.len());
        for (src, tmp_name) in dag.sources.iter().zip(source_tmp_names.iter()) {
            let query = format!("SELECT * FROM {} LIMIT 0", src.name);
            if let Err(e) = self.conn
                .new_relation(MaterializeMode::View, tmp_name.clone(), query)
                .await
            {
                let _ = self.cleanup(&tmp_dag).await;
                return Err(ExecutorError::Exec(format!(
                    "resolve_schemas: failed to create view for source '{}': {e}", src.name
                )));
            }
        }

        // Step 2 — fetch schemas for all nodes.
        debug!("resolve_schemas: fetching schemas for {} node(s)", topo.len());
        let mut resolved_nodes: Vec<(String, duckdb::arrow::datatypes::SchemaRef)> = Vec::new();
        for orig_id in &topo {
            let tmp_id = &rename_map[orig_id];
            match self.conn.get_schema(tmp_id.clone()).await {
                Some(Ok(schema)) => {
                    debug!("resolve_schemas: resolved schema for '{orig_id}' (via '{tmp_id}')");
                    resolved_nodes.push((orig_id.clone(), schema));
                }
                Some(Err(e)) => {
                    drop_source_tmps(&self.conn, &source_tmp_names).await;
                    let _ = self.cleanup(&tmp_dag).await;
                    return Err(ExecutorError::Exec(format!(
                        "resolve_schemas: get_schema failed for '{orig_id}': {e}"
                    )));
                }
                None => {
                    drop_source_tmps(&self.conn, &source_tmp_names).await;
                    let _ = self.cleanup(&tmp_dag).await;
                    return Err(ExecutorError::Exec(format!(
                        "resolve_schemas: no schema available for '{orig_id}'"
                    )));
                }
            }
        }

        // Step 2b — fetch schemas for all sources.
        debug!("resolve_schemas: fetching schemas for {} source(s)", dag.sources.len());
        let mut resolved_sources: Vec<duckdb::arrow::datatypes::SchemaRef> = Vec::new();
        for (src, tmp_name) in dag.sources.iter().zip(source_tmp_names.iter()) {
            match self.conn.get_schema(tmp_name.clone()).await {
                Some(Ok(schema)) => {
                    debug!("resolve_schemas: resolved schema for source '{}' (via '{tmp_name}')", src.name);
                    resolved_sources.push(schema);
                }
                Some(Err(e)) => {
                    drop_source_tmps(&self.conn, &source_tmp_names).await;
                    let _ = self.cleanup(&tmp_dag).await;
                    return Err(ExecutorError::Exec(format!(
                        "resolve_schemas: get_schema failed for source '{}': {e}", src.name
                    )));
                }
                None => {
                    drop_source_tmps(&self.conn, &source_tmp_names).await;
                    let _ = self.cleanup(&tmp_dag).await;
                    return Err(ExecutorError::Exec(format!(
                        "resolve_schemas: no schema available for source '{}'", src.name
                    )));
                }
            }
        }

        // Step 3 — store resolved schemas on nodes and sources.
        for (node_id, schema) in resolved_nodes {
            if let Some(node) = dag.nodes.get_mut(node_id.clone()) {
                node.schema = Some(schema);
            }
        }
        for (src, schema) in dag.sources.iter_mut().zip(resolved_sources) {
            src.schema = schema;
        }

        // Step 4 — clean up all temporary views.
        drop_source_tmps(&self.conn, &source_tmp_names).await;
        self.cleanup(&tmp_dag)
            .await
            .map_err(|e| ExecutorError::Exec(format!("resolve_schemas: cleanup failed: {e}")))?;

        debug!("resolve_schemas: complete");
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExecStats {
    pub start: DateTime<Utc>,
    pub finish: DateTime<Utc>,
    pub duration: TimeDelta,
    pub node_stats: HashMap<String, NodeStats>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub system_samples: Vec<SystemUsageSample>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeStats {
    pub start: DateTime<Utc>,
    pub finish: DateTime<Utc>,
    pub duration: TimeDelta,
    /// Raw backend EXPLAIN / EXPLAIN ANALYZE JSON, when plan collection is on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan: Option<String>,
    /// Rows the backend reported writing for this node, when it reports one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rows_produced: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::split_qualified_identifier;
    use crate::connectors::duckdb::{DuckDBConfig, DuckDBConnection};
    use crate::dag::TransformNode;

    /// `n` independent nodes, each a query slow enough that two running
    /// concurrently overlap observably.
    fn wide_dag(n: usize, max_parallelism: Option<usize>) -> Dag {
        let mut map = HashMap::new();
        for i in 0..n {
            let id = format!("n{i}");
            map.insert(
                id.clone(),
                TransformNode {
                    id,
                    // A range big enough to take real time, aggregated so the
                    // result stays tiny.
                    query_text: "SELECT sum(i) AS s FROM range(4000000) t(i)".to_string(),
                    materialize: MaterializeMode::Table,
                    depends_on: HashSet::new(),
                    schema: None,
                },
            );
        }
        Dag {
            db: "duckdb".to_string(),
            nodes: Graph::new(map),
            sources: Vec::new(),
            max_parallelism,
        }
    }

    /// The most node intervals that were ever open at the same instant.
    fn peak_overlap(stats: &ExecStats) -> usize {
        let mut edges: Vec<(i64, i64)> = Vec::new();
        for node in stats.node_stats.values() {
            edges.push((node.start.timestamp_micros(), 1));
            edges.push((node.finish.timestamp_micros(), -1));
        }
        // A node that finishes exactly as another starts is not an overlap, so
        // ends sort before starts at equal timestamps.
        edges.sort_by_key(|(at, delta)| (*at, *delta));
        let mut open = 0i64;
        let mut peak = 0i64;
        for (_, delta) in edges {
            open += delta;
            peak = peak.max(open);
        }
        peak as usize
    }

    async fn run_wide(n: usize, cap: Option<usize>) -> ExecStats {
        // More pool connections than nodes: with a single-connection pool the
        // queries would serialize on the pool rather than on the scheduler,
        // and the cap would look like it worked when nothing had been capped.
        let conn = DuckDBConnection::new(
            DuckDBConfig::new_from_path(":memory:".to_string())
                .with_num_connections(n as u32 + 1),
        )
        .await
        .expect("in-memory duckdb");
        let engine = SimpleEngine::new(conn).expect("engine");
        let dag = wide_dag(n, cap);
        let stats = engine.run(&dag).await.expect("run");
        assert_eq!(stats.node_stats.len(), n, "every node ran");
        stats
    }

    // The connector's duckdb calls are synchronous inside an async fn, so on
    // tokio's default current-thread test runtime every spawned node would run
    // to completion before the next one started -- and the cap assertions
    // would hold whether or not the cap did anything. These need real threads.
    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_a_cap_of_one_serializes_the_dag() {
        // The strongest statement the knob makes: with a cap of one, no two
        // nodes are ever in flight together. If this does not hold,
        // ParallelismTuning is measuring a setting that does nothing.
        let stats = run_wide(4, Some(1)).await;
        assert_eq!(peak_overlap(&stats), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_a_cap_bounds_how_many_nodes_are_in_flight() {
        let stats = run_wide(6, Some(2)).await;
        assert!(
            peak_overlap(&stats) <= 2,
            "cap of 2 allowed {} nodes at once",
            peak_overlap(&stats)
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_an_uncapped_dag_still_fans_out() {
        // The default has to stay what it was. A cap accidentally applied to
        // `None` would slow every DAG dee has ever run.
        let stats = run_wide(4, None).await;
        assert!(
            peak_overlap(&stats) > 1,
            "an uncapped DAG ran its nodes one at a time"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_a_cap_wider_than_the_dag_does_not_hold_anything_back() {
        let stats = run_wide(3, Some(16)).await;
        assert!(peak_overlap(&stats) > 1);
    }

    #[test]
    fn splits_bare_identifier() {
        assert_eq!(split_qualified_identifier("foo"), vec!["foo"]);
    }

    #[test]
    fn splits_partial_identifier() {
        assert_eq!(
            split_qualified_identifier("schema.foo"),
            vec!["schema", "foo"]
        );
    }

    #[test]
    fn splits_full_identifier() {
        assert_eq!(
            split_qualified_identifier("cat.schema.foo"),
            vec!["cat", "schema", "foo"]
        );
    }

    #[test]
    fn strips_quotes_from_parts() {
        assert_eq!(
            split_qualified_identifier("\"schema\".\"foo\""),
            vec!["schema", "foo"]
        );
    }

    #[test]
    fn dot_inside_quotes_is_not_a_separator() {
        assert_eq!(
            split_qualified_identifier("\"schema\".\"foo.bar\""),
            vec!["schema", "foo.bar"]
        );
    }

    #[test]
    fn unescapes_doubled_quotes_inside_quoted_part() {
        assert_eq!(
            split_qualified_identifier("\"foo \"\"bar\"\"\""),
            vec!["foo \"bar\""]
        );
    }

    #[test]
    fn mixed_quoted_and_bare_parts() {
        assert_eq!(
            split_qualified_identifier("cat.\"My Schema\".foo"),
            vec!["cat", "My Schema", "foo"]
        );
    }
}
