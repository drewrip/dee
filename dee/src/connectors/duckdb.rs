use crate::{
    connectors::{Connector, ConnectorError, DiskUsageSample, PushdownInfo},
    dag::MaterializeMode,
};
use async_trait::async_trait;
use duckdb::arrow::datatypes::SchemaRef;
use duckdb::{Config, DuckdbConnectionManager, params};
use log::{info, trace};
use r2d2::Pool;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, path::PathBuf, process::Command, sync::Arc, time::Duration};
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
fn collect_scan_pushdowns(node: &ExplainNode, out: &mut HashMap<String, PushdownInfo>) {
    if let Some(table) = &node.extra_info.table {
        let relation = table.rsplit('.').next().unwrap_or(table).to_string();
        let entry = out.entry(relation).or_default();
        for p in &node.extra_info.projections {
            if !entry.projections.contains(p) {
                entry.projections.push(p.clone());
            }
        }
        for f in &node.extra_info.filters {
            // DuckDB reports internal runtime filters (e.g. "optional: Dynamic
            // Filter (overall_rank)") that aren't real predicates to push down.
            if f.starts_with("optional:") {
                continue;
            }
            if !entry.filters.contains(f) {
                entry.filters.push(f.clone());
            }
        }
    }
    for child in &node.children {
        collect_scan_pushdowns(child, out);
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct DuckDBConfig {
    pub database: PathBuf,
    pub num_connections: u32,
    pub threads: Option<i64>,
    pub max_memory: Option<String>,
}

impl DuckDBConfig {
    pub fn new_from_path(path: String) -> Self {
        Self {
            database: PathBuf::from(path),
            num_connections: 1,
            threads: None,
            max_memory: None,
        }
    }

    pub fn with_num_connections(mut self, num_connections: u32) -> Self {
        self.num_connections = num_connections;
        self
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

pub struct DuckDBConnection {
    pub pool: Pool<DuckdbConnectionManager>,
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
            .max_size(config.num_connections)
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

        Ok(Arc::new(Self { pool }))
    }

    async fn execute(&self, query_text: String) -> Result<usize, ConnectorError> {
        let conn = self
            .pool
            .get()
            .map_err(|_| ConnectorError::Execute("didn't get connection from pool".to_string()))?;
        conn.execute(&query_text.clone(), params![]).map_err(|e| {
            ConnectorError::Execute(format!("{} - query_text:\n{}", e.to_string(), query_text))
        })
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
        let conn = self
            .pool
            .get()
            .map_err(|_| ConnectorError::Execute("didn't get connection from pool".to_string()))?;

        match relation_type {
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
        }
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
    ) -> Result<Option<HashMap<String, PushdownInfo>>, ConnectorError> {
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

        let mut result: HashMap<String, PushdownInfo> = HashMap::new();
        for plan in &plans {
            collect_scan_pushdowns(plan, &mut result);
        }

        Ok(Some(result))
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

        let t = result.get("t").expect("scan of t should be reported");
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

        let t = result.get("t").unwrap();
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
        assert_eq!(result["t"].projections, vec!["a"]);
    }

    #[tokio::test]
    async fn test_pushdown_no_filter_reports_empty_filters() {
        let conn = in_memory_conn().await;
        conn.execute("CREATE TABLE t AS SELECT range AS a FROM range(10)".to_string())
            .await
            .unwrap();

        let result = conn.pushdown("SELECT a FROM t").await.unwrap().unwrap();

        let t = result.get("t").unwrap();
        assert_eq!(t.projections, vec!["a"]);
        assert!(t.filters.is_empty());
    }
}
