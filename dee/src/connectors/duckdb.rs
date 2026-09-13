use crate::{
    connectors::{Connector, ConnectorError, DiskUsageSample, PushdownInfo},
    dag::MaterializeMode,
};
use async_trait::async_trait;
use duckdb::arrow::datatypes::SchemaRef;
use duckdb::{Config, DuckdbConnectionManager, InterruptHandle, params};
use log::{info, trace};
use r2d2::Pool;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    path::PathBuf,
    process::Command,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};
use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};
use tempfile;

/// Shape of a single node in DuckDB's `EXPLAIN (FORMAT JSON)` output, just
/// enough of it to find scan operators and read back what they pushed down.
#[derive(Deserialize, Debug, Default)]
struct ExplainNode {
    #[serde(default)]
    children: Vec<ExplainNode>,
    #[serde(default)]
    extra_info: ExplainExtraInfo,
}

#[derive(Deserialize, Debug, Default)]
struct ExplainExtraInfo {
    #[serde(rename = "Table")]
    table: Option<String>,
    #[serde(rename = "Projections", default, deserialize_with = "string_or_vec")]
    projections: Vec<String>,
    #[serde(rename = "Filters", default, deserialize_with = "string_or_vec")]
    filters: Vec<String>,
}

/// DuckDB's `EXPLAIN (FORMAT JSON)` serializes a single-element
/// `Projections`/`Filters` list as a bare string rather than a one-element
/// array, so this accepts either shape.
fn string_or_vec<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StringOrVec {
        Single(String),
        Multiple(Vec<String>),
    }
    Ok(match StringOrVec::deserialize(deserializer)? {
        StringOrVec::Single(s) => vec![s],
        StringOrVec::Multiple(v) => v,
    })
}

/// Walk `node` and its children, recording the pushed-down projections and
/// filters of every scan operator (identified by the presence of a `Table`
/// key in `extra_info`) into `out`, keyed by the relation's bare name (any
/// catalog/schema qualification DuckDB reports is stripped).
///
/// Each scan gets its own entry in the relation's `Vec`: a query that scans
/// one relation twice (a self-join, or two UNION branches) reports two
/// independent sets of filters, and folding them together would turn two
/// alternatives into one conjunction.
fn collect_scan_pushdowns(node: &ExplainNode, out: &mut HashMap<String, Vec<PushdownInfo>>) {
    if let Some(table) = &node.extra_info.table {
        let relation = table.rsplit('.').next().unwrap_or(table).to_string();
        let mut info = PushdownInfo::default();
        for p in &node.extra_info.projections {
            if !info.projections.contains(p) {
                info.projections.push(p.clone());
            }
        }
        for f in &node.extra_info.filters {
            // DuckDB reports internal runtime filters (e.g. "optional: Dynamic
            // Filter (overall_rank)") that aren't real predicates to push down.
            if f.starts_with("optional:") {
                continue;
            }
            if !info.filters.contains(f) {
                info.filters.push(f.clone());
            }
        }
        out.entry(relation).or_default().push(info);
    }
    for child in &node.children {
        collect_scan_pushdowns(child, out);
    }
}

/// How many pooled connections the DuckDB connector may hand out at once.
///
/// Not a tuning knob: it is a ceiling high enough never to bind, because the
/// number of connections in use has to be the DAG's degree of parallelism and
/// nothing else. Pooled DuckDB connections are `try_clone()`s of one
/// `Connection`, so they share a database, a buffer pool and a `threads`
/// setting -- a pool is not extra engine capacity, it is purely a second cap
/// on how many node queries can be in flight. Setting that cap independently
/// of [`Dag::max_parallelism`](crate::dag::Dag::max_parallelism) makes the
/// effective concurrency `min(cap, pool)`, which silently bounds an uncapped
/// DAG at the pool size and makes any ladder rung at or above it identical to
/// the baseline. That is a measurement bug, not a slow default:
/// `ParallelismTuning` compares rungs against a baseline it believes is
/// unbounded.
const POOL_CEILING: u32 = 1024;

#[derive(Serialize, Deserialize, Clone)]
pub struct DuckDBConfig {
    pub database: PathBuf,
    pub threads: Option<i64>,
    pub max_memory: Option<String>,
}

impl DuckDBConfig {
    pub fn new_from_path(path: String) -> Self {
        Self {
            database: PathBuf::from(path),
            threads: None,
            max_memory: None,
        }
    }

    pub fn with_threads(mut self, num_threads: i64) -> Self {
        self.threads = Some(num_threads);
        self
    }

    pub fn with_max_memory(mut self, mem_str: String) -> Self {
        self.max_memory = Some(mem_str);
        self
    }
}

/// The settings that go into [`Connector::cost_backend_key`].
///
/// Everything here changes how bytes reach the disk: the block size the
/// storage layer lays rows out in, when a checkpoint flushes the WAL, how much
/// the write buffer will hold, whether the engine may pick the order-preserving
/// sink (which writes row groups as they arrive rather than buffering them),
/// and how hard it compresses on the way.
///
/// `threads` is deliberately absent. It looks like it belongs, but DuckDB sums
/// a write operator's `operator_timing` across threads, so the measured cost of
/// a write does not move with it --- 1, 4 and 8 threads were measured within 2%
/// of each other on the same write. Keying on it would split one engine's
/// samples across every thread count a benchmark sweep happens to use.
///
/// Note also that `checkpoint_threshold` and `write_buffer_row_group_memory_limit`
/// were measured as inert for the write rate on DuckDB 1.5.5 (1 MB / 16 MB /
/// 4 GB and 32 MB / 1 GB respectively all gave the same cost). They are kept
/// because they describe the storage path and a future version may make them
/// matter, and because splitting two models that turn out to agree costs a
/// re-fit while pooling two that do not costs a wrong price.
const COST_RELEVANT_SETTINGS: [&str; 7] = [
    "default_block_size",
    "checkpoint_threshold",
    "wal_autocheckpoint",
    "write_buffer_row_group_memory_limit",
    "memory_limit",
    "preserve_insertion_order",
    "force_compression",
];

/// How long to wait for interrupted statements to actually stop before giving
/// up and saying so. Generous: the alternative to waiting is letting a caller
/// drop a relation another statement is still writing.
const INTERRUPT_QUIESCE_TIMEOUT: Duration = Duration::from_secs(30);

pub struct DuckDBConnection {
    pub pool: Pool<DuckdbConnectionManager>,
    /// Interrupt handles for the queries currently executing on this
    /// connector, keyed by an arbitrary ticket.
    ///
    /// Cancelling a DAG means cancelling the SQL, not just dropping the future
    /// that is waiting on it. DuckDB's driver call is blocking and contains no
    /// await point, so aborting the task cannot stop it -- the statement runs
    /// to completion regardless, and anything that then drops or rebuilds the
    /// relation races a write that is still in progress. An interrupt handle is
    /// the only thing that actually stops the query, so every statement that
    /// builds a relation registers one for as long as it runs.
    inflight: Arc<Mutex<HashMap<u64, Arc<InterruptHandle>>>>,
    next_ticket: Arc<AtomicU64>,
}

/// Registers an interrupt handle for as long as it is alive, so a query is
/// always deregistered on the way out -- including when it fails, and including
/// when it is the interrupt itself that made it fail.
struct Inflight {
    registry: Arc<Mutex<HashMap<u64, Arc<InterruptHandle>>>>,
    ticket: u64,
}

impl Inflight {
    fn register(
        registry: &Arc<Mutex<HashMap<u64, Arc<InterruptHandle>>>>,
        next: &Arc<AtomicU64>,
        handle: Arc<InterruptHandle>,
    ) -> Self {
        let ticket = next.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut g) = registry.lock() {
            g.insert(ticket, handle);
        }
        Self {
            registry: Arc::clone(registry),
            ticket,
        }
    }
}

impl Drop for Inflight {
    fn drop(&mut self) {
        if let Ok(mut g) = self.registry.lock() {
            g.remove(&self.ticket);
        }
    }
}

fn materialize_mode_in_duckdb(mode: MaterializeMode) -> String {
    match mode {
        MaterializeMode::Table | MaterializeMode::TempTable => "TABLE".to_string(),
        MaterializeMode::View => "VIEW".to_string(),
    }
}

fn parse_duckdb_size_bytes(value: &str) -> Option<u64> {
    let trimmed = value.trim();
    let mut parts = trimmed.split_whitespace();
    let quantity = parts.next()?.parse::<f64>().ok()?;
    let unit = parts.next().unwrap_or("B").to_ascii_uppercase();
    let multiplier = match unit.as_str() {
        "B" => 1.0,
        "KB" | "KIB" => 1024.0,
        "MB" | "MIB" => 1024.0 * 1024.0,
        "GB" | "GIB" => 1024.0 * 1024.0 * 1024.0,
        "TB" | "TIB" => 1024.0 * 1024.0 * 1024.0 * 1024.0,
        _ => return None,
    };
    Some((quantity * multiplier).round() as u64)
}

fn sample_process_cpu_usage(pid: u32) -> Result<Option<f64>, ConnectorError> {
    let output = Command::new("ps")
        .args(["-o", "%cpu=", "-p", &pid.to_string()])
        .output()
        .map_err(|e| ConnectorError::Execute(format!("Failed to run ps for cpu usage: {}", e)))?;

    if !output.status.success() {
        return Err(ConnectorError::Execute(format!(
            "ps exited with status {} while sampling cpu usage",
            output.status
        )));
    }

    let stdout = String::from_utf8(output.stdout).map_err(|e| {
        ConnectorError::Execute(format!("Failed to decode ps cpu usage output: {}", e))
    })?;

    Ok(stdout.trim().parse::<f64>().ok())
}

fn sample_process_disk_io(pid: u32) -> (Option<u64>, Option<u64>) {
    let mut sys = System::new();
    let pid = Pid::from_u32(pid);
    sys.refresh_processes_specifics(
        ProcessesToUpdate::Some(&[pid]),
        ProcessRefreshKind::everything(),
    );
    sys.process(pid)
        .map(|process| {
            let usage = process.disk_usage();
            (Some(usage.total_read_bytes), Some(usage.total_written_bytes))
        })
        .unwrap_or((None, None))
}

/// Rows written by a `CREATE TABLE AS` / `CREATE TEMP TABLE AS`, read from a
/// DuckDB `enable_profiling='json'` plan.
///
/// The `*_CREATE_TABLE_AS` operator reports `operator_cardinality = 1` (the
/// single count row the statement returns), so the real figure is the
/// cardinality of its input operator.
fn rows_written_from_plan(json_str: &str) -> Option<usize> {
    fn op_name(node: &serde_json::Value) -> Option<&str> {
        node.get("operator_name")
            .or_else(|| node.get("name"))
            .and_then(|v| v.as_str())
    }

    fn find(node: &serde_json::Value) -> Option<usize> {
        if let Some(name) = op_name(node)
            && name.ends_with("CREATE_TABLE_AS")
        {
            let child = node.get("children")?.as_array()?.first()?;
            return child
                .get("operator_cardinality")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize);
        }
        for child in node.get("children")?.as_array()? {
            if let Some(found) = find(child) {
                return Some(found);
            }
        }
        None
    }

    find(&serde_json::from_str::<serde_json::Value>(json_str).ok()?)
}

impl DuckDBConnection {
    /// Run `f` on a blocking pool thread with the connection's interrupt handle
    /// registered for the duration.
    ///
    /// Two things this buys, and both are required for a cancellable DAG:
    /// the runtime keeps its workers, and [`interrupt_inflight`] has something
    /// to interrupt. The handle is deregistered by `Inflight`'s `Drop`, so a
    /// query that fails -- including one the interrupt killed -- still cleans
    /// up after itself.
    async fn blocking<F, R>(&self, f: F) -> Result<R, ConnectorError>
    where
        F: FnOnce(&duckdb::Connection) -> Result<R, ConnectorError> + Send + 'static,
        R: Send + 'static,
    {
        let pool = self.pool.clone();
        let registry = Arc::clone(&self.inflight);
        let next = Arc::clone(&self.next_ticket);
        tokio::task::spawn_blocking(move || {
            let conn = pool.get().map_err(|_| {
                ConnectorError::Execute("didn't get connection from pool".to_string())
            })?;
            let _guard = Inflight::register(&registry, &next, conn.interrupt_handle());
            f(&conn)
        })
        .await
        .map_err(|e| ConnectorError::Execute(format!("blocking task failed: {e}")))?
    }
}

/// DuckDB's two write sinks, and the walk that says which one a plan gets.
///
/// DuckDB picks `BATCH_CREATE_TABLE_AS` when the pipeline feeding the write
/// still carries batch order, and writes row groups straight out as they
/// arrive; otherwise it picks `CREATE_TABLE_AS` and buffers first. The two
/// differ by about 2x per byte on real DAGs, and only the second one has the
/// dead zone where small writes never reach the disk at all --- so which sink a
/// candidate would get has to be predicted before its build can be priced.
pub const DUCKDB_BATCH_WRITE: &str = "BATCH_CREATE_TABLE_AS";
pub const DUCKDB_BUFFERED_WRITE: &str = "CREATE_TABLE_AS";

/// The relation name the write-path probe plans against.
///
/// Never created: `EXPLAIN` of a `CREATE OR REPLACE TABLE` plans the statement
/// and stops. The name only has to be one no DAG would use, so that if a future
/// DuckDB ever did materialize during `EXPLAIN` it would not land on anything.
const WRITE_PATH_PROBE: &str = "dee_write_path_probe";

#[async_trait]
impl Connector for DuckDBConnection {
    type Config = DuckDBConfig;
    type Connection = DuckDBConnection;

    async fn new(config: Self::Config) -> Result<Arc<Self::Connection>, ConnectorError> {
        let mut conf = Config::default();
        if let Some(max_mem) = config.max_memory {
            conf = conf
                .max_memory(&max_mem)
                .map_err(|_| ConnectorError::Create("set max memory problem".to_string()))?;
        }
        if let Some(threads) = config.threads {
            conf = conf
                .threads(threads)
                .map_err(|_| ConnectorError::Create("set threads problem".to_string()))?;
        }

        conf = conf
            .access_mode(duckdb::AccessMode::ReadWrite)
            .map_err(|_| ConnectorError::Create("set access_mode".to_string()))?;

        let manager = DuckdbConnectionManager::file_with_flags(config.database, conf)
            .map_err(|e| ConnectorError::Create(format!("connection manager - {}", e)))?;
        let pool = Pool::builder()
            .connection_timeout(Duration::from_hours(2))
            // `min_idle(0)` so connections are cloned on demand rather than
            // pre-built to `max_size`: what is live is what the executor
            // actually has in flight, which is the DAG's parallelism.
            .min_idle(Some(0))
            .max_size(POOL_CEILING)
            .build(manager)
            .map_err(|_| ConnectorError::Create("r2d2 pool".to_string()))?;

        {
            let conn = pool.get().map_err(|_| {
                ConnectorError::Create("couldn't get connection for ICU setup".to_string())
            })?;
            conn.execute_batch("INSTALL icu; LOAD icu;").map_err(|e| {
                ConnectorError::Create(format!("failed to install/load ICU: {}", e))
            })?;
        }

        Ok(Arc::new(Self {
            pool,
            inflight: Arc::new(Mutex::new(HashMap::new())),
            next_ticket: Arc::new(AtomicU64::new(0)),
        }))
    }

    async fn execute(&self, query_text: String) -> Result<usize, ConnectorError> {
        // On a blocking pool thread rather than inline: the driver call blocks
        // with no await inside it, so running it on a runtime worker both
        // starves the runtime and makes the task impossible to cancel.
        self.blocking(move |conn| {
            conn.execute(&query_text.clone(), params![]).map_err(|e| {
                ConnectorError::Execute(format!("{} - query_text:\n{}", e, query_text))
            })
        })
        .await
    }

    async fn interrupt_inflight(&self) -> usize {
        let snapshot = || -> Vec<Arc<InterruptHandle>> {
            match self.inflight.lock() {
                Ok(g) => g.values().cloned().collect(),
                Err(_) => Vec::new(),
            }
        };
        let signalled = snapshot().len();
        if signalled == 0 {
            return 0;
        }

        // Signal, then wait for the statements to unwind. The task that issued
        // one may already have been aborted while the query itself is still
        // running on a blocking thread, so the registry -- not the task set --
        // is what says whether the engine is quiet. Re-signal each round: a
        // statement can start between the snapshot and the interrupt.
        let deadline = std::time::Instant::now() + INTERRUPT_QUIESCE_TIMEOUT;
        loop {
            let handles = snapshot();
            if handles.is_empty() {
                return signalled;
            }
            for h in &handles {
                h.interrupt();
            }
            if std::time::Instant::now() >= deadline {
                log::warn!(
                    "{} query(ies) still running {:?} after being interrupted; \
                     giving up waiting for them",
                    handles.len(),
                    INTERRUPT_QUIESCE_TIMEOUT
                );
                return signalled;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    async fn new_relation(
        &self,
        relation_type: MaterializeMode,
        name: String,
        query_text: String,
    ) -> Result<usize, ConnectorError> {
        let rel_type = materialize_mode_in_duckdb(relation_type);
        trace!("creating new_relation ({}, {})", rel_type, name);
        let tmpl_query = format!(
            "CREATE OR REPLACE {} {} AS ({})",
            rel_type, name, query_text
        );
        self.execute(tmpl_query).await
    }

    async fn new_relation_and_explain(
        &self,
        relation_type: MaterializeMode,
        name: String,
        query_text: String,
    ) -> Result<(usize, Option<String>), ConnectorError> {
        self.blocking(move |conn| match relation_type {
            MaterializeMode::View => {
                let explain_query = format!("EXPLAIN (FORMAT JSON) {}", query_text);
                let mut stmt = conn.prepare(&explain_query).map_err(|e| {
                    ConnectorError::Execute(format!("Failed to prepare explain: {}", e))
                })?;

                let json_str: String = stmt
                    .query_row([], |row| {
                        let col_count = row.as_ref().column_count();
                        if col_count >= 2 {
                            row.get(1)
                        } else {
                            row.get(0)
                        }
                    })
                    .map_err(|e| {
                        ConnectorError::Execute(format!("Failed to execute explain: {}", e))
                    })?;

                let rel_type = materialize_mode_in_duckdb(relation_type);
                let tmpl_query = format!("CREATE {} {} AS ({})", rel_type, name, query_text);
                let res = conn.execute(&tmpl_query, params![]).map_err(|e| {
                    ConnectorError::Execute(format!(
                        "{} - query_text:\n{}",
                        e.to_string(),
                        tmpl_query
                    ))
                })?;
                Ok((res, Some(json_str)))
            }
            MaterializeMode::Table | MaterializeMode::TempTable => {
                let temp_file = tempfile::Builder::new()
                    .suffix(".json")
                    .tempfile()
                    .map_err(|e| {
                        ConnectorError::Execute(format!("Failed to create temp file: {}", e))
                    })?;
                let temp_path = temp_file
                    .path()
                    .to_str()
                    .ok_or(ConnectorError::Execute("Invalid temp path".to_string()))?;

                conn.execute("SET enable_profiling = 'json';", [])
                    .map_err(|e| {
                        ConnectorError::Execute(format!("Failed to enable profiling: {}", e))
                    })?;
                conn.execute(&format!("SET profiling_output = '{}';", temp_path), [])
                    .map_err(|e| {
                        ConnectorError::Execute(format!("Failed to set profiling output: {}", e))
                    })?;

                let rel_type = materialize_mode_in_duckdb(relation_type);
                let tmpl_query = format!("CREATE {} {} AS ({})", rel_type, name, query_text);
                let res = conn.execute(&tmpl_query, params![]).map_err(|e| {
                    ConnectorError::Execute(format!(
                        "{} - query_text:\n{}",
                        e.to_string(),
                        tmpl_query
                    ))
                })?;

                conn.execute("RESET enable_profiling;", []).map_err(|e| {
                    ConnectorError::Execute(format!("Failed to disable profiling: {}", e))
                })?;
                conn.execute("RESET profiling_output;", []).map_err(|e| {
                    ConnectorError::Execute(format!("Failed to reset profiling output: {}", e))
                })?;

                let json_str = std::fs::read_to_string(temp_path).map_err(|e| {
                    ConnectorError::Execute(format!("Failed to read profiling output: {}", e))
                })?;

                // DuckDB's `execute` reports 0 changed rows for a CTAS, and
                // the CREATE_TABLE_AS operator's own cardinality is 1 (the
                // count row it returns). The number of rows actually written
                // is the cardinality of that operator's input, so read it off
                // the profiling plan we already have rather than paying for a
                // separate COUNT(*).
                let rows_written = rows_written_from_plan(&json_str).unwrap_or(res);

                Ok((rows_written, Some(json_str)))
            }
        })
        .await
    }

    async fn drop_relation(
        &self,
        relation_type: MaterializeMode,
        name: String,
    ) -> Result<usize, ConnectorError> {
        let rel_type = materialize_mode_in_duckdb(relation_type);
        trace!("attempt drop_relation ({}, {})", rel_type, name);
        let tmpl_query = format!("DROP {} IF EXISTS {}", rel_type, name);
        self.execute(tmpl_query).await
    }

    async fn get_schema(&self, name: String) -> Option<Result<SchemaRef, ConnectorError>> {
        info!("attempt to fetch arrow schema for {}", name);
        let conn = match self.pool.get() {
            Ok(c) => c,
            Err(e) => {
                return Some(Err(ConnectorError::Execute(format!(
                    "couldn't get connection from pool: {e}"
                ))));
            }
        };
        // Execute with LIMIT 0 via query_arrow so DuckDB populates the arrow
        // schema pointer before we call get_schema().  A plain prepare() +
        // schema() panics because the arrow array pointer is only set after
        // execution.  LIMIT 0 returns zero rows so there is no data transfer.
        let tmpl_query = format!("SELECT * FROM {} LIMIT 0", name);
        let mut stmt = match conn.prepare(&tmpl_query) {
            Ok(s) => s,
            Err(e) => {
                return Some(Err(ConnectorError::Execute(format!(
                    "couldn't prepare schema query for {name}: {e}"
                ))));
            }
        };
        match stmt.query_arrow([]) {
            Ok(arrow) => Some(Ok(arrow.get_schema())),
            Err(e) => Some(Err(ConnectorError::Execute(format!(
                "couldn't execute schema query for {name}: {e}"
            )))),
        }
    }

    async fn pushdown(
        &self,
        query_text: &str,
    ) -> Result<Option<HashMap<String, Vec<PushdownInfo>>>, ConnectorError> {
        let conn = self
            .pool
            .get()
            .map_err(|_| ConnectorError::Execute("didn't get connection from pool".to_string()))?;

        let explain_query = format!("EXPLAIN (FORMAT JSON) {}", query_text);
        let mut stmt = conn.prepare(&explain_query).map_err(|e| {
            ConnectorError::Execute(format!("Failed to prepare explain: {}", e))
        })?;

        let json_str: String = stmt
            .query_row([], |row| {
                let col_count = row.as_ref().column_count();
                if col_count >= 2 {
                    row.get(1)
                } else {
                    row.get(0)
                }
            })
            .map_err(|e| ConnectorError::Execute(format!("Failed to execute explain: {}", e)))?;

        let plans: Vec<ExplainNode> = serde_json::from_str(&json_str).map_err(|e| {
            ConnectorError::Execute(format!(
                "Failed to parse explain JSON: {e} - json:\n{json_str}"
            ))
        })?;

        let mut result: HashMap<String, Vec<PushdownInfo>> = HashMap::new();
        for plan in &plans {
            collect_scan_pushdowns(plan, &mut result);
        }

        Ok(Some(result))
    }

    async fn explain(&self, query_text: &str) -> Result<Option<String>, ConnectorError> {
        let explain_query = format!("EXPLAIN (FORMAT JSON) {}", query_text);
        self.blocking(move |conn| {
            let mut stmt = conn.prepare(&explain_query).map_err(|e| {
                ConnectorError::Execute(format!("Failed to prepare explain: {e}"))
            })?;
            // Two columns (`explain_key`, `explain_value`) in every version
            // that prints a header; one in those that do not.
            let json_str: String = stmt
                .query_row([], |row| {
                    if row.as_ref().column_count() >= 2 {
                        row.get(1)
                    } else {
                        row.get(0)
                    }
                })
                .map_err(|e| {
                    ConnectorError::Execute(format!(
                        "Failed to execute explain: {e} - query_text:\n{explain_query}"
                    ))
                })?;
            Ok(Some(json_str))
        })
        .await
    }

    fn parse_plan(&self, json: &str) -> Option<Vec<crate::plan::PlanNode>> {
        crate::plan::parse_duckdb_plan(json)
    }

    fn time_basis(&self) -> crate::plan::TimeBasis {
        crate::plan::TimeBasis::CpuTime
    }

    async fn sample_system_memory_usage(&self) -> Result<Option<u64>, ConnectorError> {
        let conn = self
            .pool
            .get()
            .map_err(|_| ConnectorError::Execute("didn't get connection from pool".to_string()))?;

        let mut stmt = conn
            .prepare("SELECT memory_usage FROM pragma_database_size()")
            .map_err(|e| {
                ConnectorError::Execute(format!("Failed to prepare memory usage sample: {}", e))
            })?;

        let memory_usage: String = stmt
            .query_row([], |row| row.get(0))
            .map_err(|e| ConnectorError::Execute(format!("Failed to query memory usage: {}", e)))?;

        Ok(parse_duckdb_size_bytes(&memory_usage))
    }

    /// DuckDB's `threads` setting: one pool shared by every query on the
    /// database, so it bounds the whole engine and not just one statement.
    async fn write_path_for(&self, query_text: &str) -> Result<Option<String>, ConnectorError> {
        // `CREATE OR REPLACE`, and a name of our own: a plain `CREATE TABLE`
        // whose name is already taken plans as a bare `CREATE_TABLE` and
        // answers nothing. `EXPLAIN` neither creates nor replaces --- verified
        // against a populated table of that name, which survived intact.
        let probe = format!("EXPLAIN (FORMAT JSON) CREATE OR REPLACE TABLE {WRITE_PATH_PROBE} AS ({query_text})");
        let json = self
            .blocking(move |conn| {
                let mut stmt = conn.prepare(&probe).map_err(|e| {
                    ConnectorError::Execute(format!("Failed to prepare write-path probe: {e}"))
                })?;
                let json: String = stmt
                    .query_row([], |row| {
                        if row.as_ref().column_count() >= 2 {
                            row.get(1)
                        } else {
                            row.get(0)
                        }
                    })
                    .map_err(|e| {
                        ConnectorError::Execute(format!("write-path probe failed: {e}"))
                    })?;
                Ok(Some(json))
            })
            .await;
        // A query the engine will not plan as a write is not an error worth
        // failing a costing over --- the caller falls back to inferring it.
        let Ok(Some(json)) = json else {
            return Ok(None);
        };
        let root = crate::plan::parse_duckdb_plan(&json)
            .and_then(|p| p.into_iter().next())
            .filter(|n| crate::plan::is_write_operator(&n.operator))
            .map(|n| n.operator.to_ascii_uppercase());
        Ok(root)
    }

    async fn cost_backend_key(&self) -> Result<Option<String>, ConnectorError> {
        let conn = self
            .pool
            .get()
            .map_err(|_| ConnectorError::Execute("didn't get connection from pool".to_string()))?;
        let version: String = conn
            .query_row("SELECT version()", [], |row| row.get(0))
            .map_err(|e| ConnectorError::Execute(format!("reading version - {e}")))?;
        // The file, not the alias: two DAGs can attach the same database under
        // different names, and the same name can be attached to two files.
        // `:memory:` for an in-memory database, which is what DuckDB reports.
        let path: String = conn
            .query_row(
                "SELECT path FROM duckdb_databases() WHERE database_name = current_database()",
                [],
                |row| row.get::<_, Option<String>>(0).map(|p| p.unwrap_or_default()),
            )
            .map_err(|e| ConnectorError::Execute(format!("reading database path - {e}")))?;
        let path = if path.is_empty() {
            ":memory:".to_string()
        } else {
            // Canonicalized so a relative path and an absolute one to the same
            // file share a model rather than fitting two.
            std::fs::canonicalize(&path)
                .map(|p| p.display().to_string())
                .unwrap_or(path)
        };

        let mut settings = Vec::new();
        for name in COST_RELEVANT_SETTINGS {
            let value: String = conn
                .query_row(
                    "SELECT value FROM duckdb_settings() WHERE name = ?",
                    [name],
                    |row| row.get::<_, Option<String>>(0).map(|v| v.unwrap_or_default()),
                )
                // A setting this build does not have is recorded as absent
                // rather than skipped: skipping it would make the key depend on
                // which reads happened to succeed.
                .unwrap_or_else(|_| "-".to_string());
            settings.push(format!("{name}={value}"));
        }

        Ok(Some(format!(
            "duckdb {version} db={path} {}",
            settings.join(" ")
        )))
    }

    async fn parallelism_budget(&self) -> Result<Option<usize>, ConnectorError> {
        let conn = self
            .pool
            .get()
            .map_err(|_| ConnectorError::Execute("didn't get connection from pool".to_string()))?;
        let threads: i64 = conn
            .query_row("SELECT current_setting('threads')", [], |row| row.get(0))
            .map_err(|e| ConnectorError::Execute(format!("reading threads - {e}")))?;
        Ok((threads > 0).then_some(threads as usize))
    }

    async fn sample_system_cpu_usage(&self) -> Result<Option<f64>, ConnectorError> {
        sample_process_cpu_usage(std::process::id())
    }

    async fn sample_system_disk_usage(&self) -> Result<DiskUsageSample, ConnectorError> {
        let conn = self
            .pool
            .get()
            .map_err(|_| ConnectorError::Execute("didn't get connection from pool".to_string()))?;

        let mut stmt = conn
            .prepare("SELECT database_size FROM pragma_database_size()")
            .map_err(|e| {
                ConnectorError::Execute(format!("Failed to prepare disk usage sample: {}", e))
            })?;

        let database_size: String = stmt
            .query_row([], |row| row.get(0))
            .map_err(|e| ConnectorError::Execute(format!("Failed to query disk usage: {}", e)))?;

        let (read_bytes, written_bytes) = sample_process_disk_io(std::process::id());

        Ok(DiskUsageSample {
            disk_bytes: parse_duckdb_size_bytes(&database_size),
            read_bytes,
            written_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_duckdb_size_bytes() {
        assert_eq!(parse_duckdb_size_bytes("44.0 KiB"), Some(45056));
        assert_eq!(parse_duckdb_size_bytes("1.0 B"), Some(1));
    }

    #[test]
    fn test_sample_process_cpu_usage() {
        let cpu = sample_process_cpu_usage(std::process::id()).unwrap();
        assert!(cpu.is_some());
        assert!(cpu.unwrap() >= 0.0);
    }

    #[test]
    fn test_sample_process_disk_io() {
        // Values may legitimately be `None` on platforms sysinfo doesn't
        // support per-process disk counters on; just assert this doesn't panic.
        let _ = sample_process_disk_io(std::process::id());
    }

    async fn in_memory_conn() -> Arc<DuckDBConnection> {
        let config = DuckDBConfig::new_from_path(":memory:".to_string());
        DuckDBConnection::new(config).await.unwrap()
    }

    #[tokio::test]
    async fn test_pushdown_reports_filter_and_projection() {
        let conn = in_memory_conn().await;
        conn.execute(
            "CREATE TABLE t AS SELECT range AS a, range*2 AS b, range*3 AS c FROM range(100)"
                .to_string(),
        )
        .await
        .unwrap();

        let result = conn
            .pushdown("SELECT a, b FROM t WHERE a > 10 AND c < 50")
            .await
            .unwrap()
            .expect("duckdb connector should support pushdown");

        let scans = result.get("t").expect("scan of t should be reported");
        assert_eq!(scans.len(), 1, "one scan of t");
        let t = &scans[0];
        assert_eq!(t.projections, vec!["a", "b"]);
        assert_eq!(t.filters.len(), 2);
        assert!(t.filters.iter().any(|f| f.contains('a')));
        assert!(t.filters.iter().any(|f| f.contains('c')));
    }

    #[tokio::test]
    async fn test_pushdown_single_projection_and_filter_not_treated_as_chars() {
        let conn = in_memory_conn().await;
        conn.execute("CREATE TABLE t AS SELECT range AS a FROM range(10)".to_string())
            .await
            .unwrap();

        let result = conn
            .pushdown("SELECT a FROM t WHERE a > 5")
            .await
            .unwrap()
            .unwrap();

        let t = &result.get("t").unwrap()[0];
        assert_eq!(t.projections, vec!["a"]);
        assert_eq!(t.filters, vec!["a>5"]);
    }

    #[tokio::test]
    async fn test_pushdown_sees_through_views_to_base_table() {
        let conn = in_memory_conn().await;
        conn.execute(
            "CREATE TABLE t AS SELECT range AS a, range*2 AS b, range*3 AS c FROM range(100)"
                .to_string(),
        )
        .await
        .unwrap();
        conn.execute("CREATE VIEW v AS SELECT a, b, c FROM t WHERE a > 5".to_string())
            .await
            .unwrap();

        let result = conn.pushdown("SELECT a FROM v").await.unwrap().unwrap();

        // Only the base table shows up; the view is inlined by DuckDB's planner.
        assert!(result.contains_key("t"));
        assert!(!result.contains_key("v"));
        assert_eq!(result["t"][0].projections, vec!["a"]);
    }

    // A self-join scans one relation twice with a different predicate on each
    // side. Reporting them as one merged entry would let the caller AND two
    // alternatives together and ask for rows that satisfy both.
    #[tokio::test]
    async fn test_pushdown_reports_each_scan_of_a_relation_separately() {
        let conn = in_memory_conn().await;
        conn.execute(
            "CREATE TABLE t AS SELECT range AS a, range % 7 AS b FROM range(100)".to_string(),
        )
        .await
        .unwrap();

        let result = conn
            .pushdown("SELECT x.a FROM t x JOIN t y ON x.b = y.b WHERE x.a < 10 AND y.a > 90")
            .await
            .unwrap()
            .unwrap();

        let scans = result.get("t").expect("scans of t should be reported");
        assert_eq!(scans.len(), 2, "each scan reported separately: {scans:?}");
        assert!(
            scans.iter().any(|s| s.filters.iter().any(|f| f.contains('<')))
                && scans.iter().any(|s| s.filters.iter().any(|f| f.contains('>'))),
            "each scan keeps its own predicate: {scans:?}"
        );
    }

    #[tokio::test]
    async fn test_pushdown_no_filter_reports_empty_filters() {
        let conn = in_memory_conn().await;
        conn.execute("CREATE TABLE t AS SELECT range AS a FROM range(10)".to_string())
            .await
            .unwrap();

        let result = conn.pushdown("SELECT a FROM t").await.unwrap().unwrap();

        let t = &result.get("t").unwrap()[0];
        assert_eq!(t.projections, vec!["a"]);
        assert!(t.filters.is_empty());
    }
}

#[cfg(test)]
mod write_path_tests {
    use super::{DUCKDB_BATCH_WRITE, DUCKDB_BUFFERED_WRITE, WRITE_PATH_PROBE};
    use crate::connectors::Connector;
    use crate::connectors::duckdb::{DuckDBConfig, DuckDBConnection};

    #[tokio::test]
    async fn asking_the_engine_settles_the_window_the_walk_cannot() {
        let c = DuckDBConnection::new(DuckDBConfig::new_from_path(":memory:".to_string()))
            .await
            .unwrap();
        c.execute(
            "CREATE TABLE src AS SELECT i::BIGINT a, (i%97)::BIGINT b, (i%1000)::BIGINT g \
             FROM range(200000) t(i)"
                .to_string(),
        )
        .await
        .unwrap();

        let unpartitioned = "SELECT a, row_number() OVER (ORDER BY b) r FROM src";
        let partitioned = "SELECT a, row_number() OVER (PARTITION BY g) r FROM src";
        assert_eq!(
            c.write_path_for(unpartitioned).await.unwrap().as_deref(),
            Some(DUCKDB_BATCH_WRITE)
        );
        assert_eq!(
            c.write_path_for(partitioned).await.unwrap().as_deref(),
            Some(DUCKDB_BUFFERED_WRITE)
        );

        // And the two ordinary shapes, so the probe is answering about the
        // pipeline and not just echoing something constant.
        assert_eq!(
            c.write_path_for("SELECT * FROM src").await.unwrap().as_deref(),
            Some(DUCKDB_BATCH_WRITE)
        );
        assert_eq!(
            c.write_path_for("SELECT g, count(*) n FROM src GROUP BY g")
                .await
                .unwrap()
                .as_deref(),
            Some(DUCKDB_BUFFERED_WRITE)
        );
    }

    #[tokio::test]
    async fn the_write_path_probe_creates_and_replaces_nothing() {
        let c = DuckDBConnection::new(DuckDBConfig::new_from_path(":memory:".to_string()))
            .await
            .unwrap();
        c.execute("CREATE TABLE src AS SELECT 1 AS a".to_string())
            .await
            .unwrap();
        // A populated table sitting exactly where the probe plans to write.
        c.execute(format!("CREATE TABLE {WRITE_PATH_PROBE} AS SELECT 42 AS keep_me"))
            .await
            .unwrap();

        assert!(c.write_path_for("SELECT * FROM src").await.unwrap().is_some());

        let schema = c
            .get_schema(WRITE_PATH_PROBE.to_string())
            .await
            .expect("the probe table is still there")
            .expect("and still readable");
        assert_eq!(
            schema.fields().len(),
            1,
            "the probe replaced a real table: {schema:?}"
        );
        assert_eq!(schema.field(0).name(), "keep_me", "column was overwritten");
    }

    #[tokio::test]
    async fn an_unplannable_query_answers_none_rather_than_erroring() {
        let c = DuckDBConnection::new(DuckDBConfig::new_from_path(":memory:".to_string()))
            .await
            .unwrap();
        assert_eq!(
            c.write_path_for("SELECT * FROM no_such_relation").await.unwrap(),
            None
        );
    }
}
