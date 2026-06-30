use crate::{
    connectors::{
        Connector, ConnectorError, ExplainSupport, ProfilingSupport,
        RelationOps, SchemaSupport, SystemMetrics,
    },
    dag::MaterializeMode,
};
use async_trait::async_trait;
use datafusion::arrow::datatypes::SchemaRef;
use duckdb::{Config, DuckdbConnectionManager, params};
use log::{info, trace};
use r2d2::Pool;
use serde::{Deserialize, Serialize};
use std::{path::PathBuf, process::Command, sync::Arc, time::Duration};
use tempfile;

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
        .map_err(|e| ConnectorError::Execute { err: e.to_string(), query: "ps".to_string() })?;

    if !output.status.success() {
        return Err(ConnectorError::Execute { err: format!("ps exited with status {}", output.status), query: "ps".to_string() });
    }

    let stdout = String::from_utf8(output.stdout).map_err(|e| {
        ConnectorError::Execute { err: e.to_string(), query: "ps".to_string() }
    })?;

    Ok(stdout.trim().parse::<f64>().ok())
}

// ---------------------------------------------------------------------------
// Capability trait implementations
// ---------------------------------------------------------------------------

#[async_trait]
impl SchemaSupport for DuckDBConnection {
    async fn get_schema(&self, name: &str) -> Result<SchemaRef, ConnectorError> {
        info!("attempt to fetch arrow schema for {}", name);
        let conn = self.pool.get().map_err(|e| {
            ConnectorError::Execute { err: e.to_string(), query: "".to_string() }
        })?;

        // Execute with LIMIT 0 via query_arrow so DuckDB populates the arrow
        // schema pointer before we call get_schema().  A plain prepare() +
        // schema() panics because the arrow array pointer is only set after
        // execution.  LIMIT 0 returns zero rows so there is no data transfer.
        let tmpl_query = format!("SELECT * FROM {} LIMIT 0", name);
        let mut stmt = conn.prepare(&tmpl_query).map_err(|e| {
            ConnectorError::Execute { err: e.to_string(), query: tmpl_query.clone() }
        })?;

        stmt.query_arrow([])
            .map_err(|e| ConnectorError::Execute { err: e.to_string(), query: tmpl_query })
            .map(|arrow| arrow.get_schema())
    }
}

#[async_trait]
impl ExplainSupport for DuckDBConnection {
    async fn explain(&self, query: &str) -> Result<String, ConnectorError> {
        let conn = self.pool.get().map_err(|_| {
            ConnectorError::Execute { err: "didn't get connection from pool".to_string(), query: query.to_string() }
        })?;

        let explain_query = format!("EXPLAIN (FORMAT JSON) {}", query);
        let mut stmt = conn.prepare(&explain_query).map_err(|e| {
            ConnectorError::Execute { err: e.to_string(), query: explain_query.clone() }
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
                ConnectorError::Execute { err: e.to_string(), query: query.to_string() }
            })?;

        Ok(json_str)
    }
}

#[async_trait]
impl ProfilingSupport for DuckDBConnection {
    type PlanData = String;

    async fn execute_with_profile(
        &self,
        query: &str,
    ) -> Result<(usize, Option<Self::PlanData>), ConnectorError> {
        let conn = self.pool.get().map_err(|_| {
            ConnectorError::Execute { err: "didn't get connection from pool".to_string(), query: query.to_string() }
        })?;

        let temp_file = tempfile::Builder::new()
            .suffix(".json")
            .tempfile()
            .map_err(|e| {
                ConnectorError::Execute { err: e.to_string(), query: "".to_string() }
            })?;
        let temp_path = temp_file
            .path()
            .to_str()
            .ok_or(ConnectorError::Execute { err: "Invalid temp path".to_string(), query: "".to_string() })?;

        conn.execute("SET enable_profiling = 'json';", [])
            .map_err(|e| {
                ConnectorError::Execute { err: e.to_string(), query: "SET enable_profiling = 'json';".to_string() }
            })?;
        conn.execute(&format!("SET profiling_output = '{}';", temp_path), [])
            .map_err(|e| {
                ConnectorError::Execute { err: e.to_string(), query: format!("SET profiling_output = '{}';", temp_path) }
            })?;

        let res = conn.execute(query, params![]).map_err(|e| {
            ConnectorError::Execute { err: e.to_string(), query: query.to_string() }
        })?;

        conn.execute("RESET enable_profiling;", []).map_err(|e| {
            ConnectorError::Execute { err: e.to_string(), query: "RESET enable_profiling;".to_string() }
        })?;
        conn.execute("RESET profiling_output;", []).map_err(|e| {
            ConnectorError::Execute { err: e.to_string(), query: "RESET profiling_output;".to_string() }
        })?;

        let json_str = std::fs::read_to_string(temp_path).map_err(|e| {
            ConnectorError::Execute { err: e.to_string(), query: "".to_string() }
        })?;

        Ok((res, Some(json_str)))
    }
}

#[async_trait]
impl RelationOps for DuckDBConnection {
    async fn create_relation(
        &self,
        relation_type: MaterializeMode,
        name: String,
        query: String,
    ) -> Result<usize, ConnectorError> {
        let rel_type = materialize_mode_in_duckdb(relation_type);
        trace!("creating new_relation ({}, {})", rel_type, name);
        let tmpl_query = format!(
            "CREATE OR REPLACE {} {} AS ({})",
            rel_type, name, query
        );
        self.execute(tmpl_query).await
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
}

#[async_trait]
impl SystemMetrics for DuckDBConnection {
    async fn sample_cpu(&self) -> Result<Option<f64>, ConnectorError> {
        sample_process_cpu_usage(std::process::id())
    }

    async fn sample_memory(&self) -> Result<Option<u64>, ConnectorError> {
        let conn = self.pool.get().map_err(|_| {
            ConnectorError::Execute { err: "didn't get connection from pool".to_string(), query: "".to_string() }
        })?;

        let mut stmt = conn
            .prepare("SELECT memory_usage FROM pragma_database_size()")
            .map_err(|e| {
                ConnectorError::Execute { err: e.to_string(), query: "SELECT memory_usage FROM pragma_database_size()".to_string() }
            })?;

        let memory_usage: String = stmt
            .query_row([], |row| row.get(0))
            .map_err(|e| {
                ConnectorError::Execute { err: e.to_string(), query: "SELECT memory_usage FROM pragma_database_size()".to_string() }
            })?;

        Ok(parse_duckdb_size_bytes(&memory_usage))
    }
}

// ---------------------------------------------------------------------------
// Connector impl (keeps backward-compatible method names)
// ---------------------------------------------------------------------------

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
            .map_err(|_| ConnectorError::Execute { err: "didn't get connection from pool".to_string(), query: query_text.clone() })?;
        conn.execute(&query_text.clone(), params![]).map_err(|e| {
            ConnectorError::Execute { err: e.to_string(), query: query_text }
        })
    }

    async fn dialect(&self) -> &'static str {
        "duckdb"
    }

    // Delegate deprecated Connector methods to capability traits
    async fn new_relation(
        &self,
        relation_type: MaterializeMode,
        name: String,
        query_text: String,
    ) -> Result<usize, ConnectorError> {
        RelationOps::create_relation(self, relation_type, name, query_text).await
    }

    async fn drop_relation(
        &self,
        relation_type: MaterializeMode,
        name: String,
    ) -> Result<usize, ConnectorError> {
        RelationOps::drop_relation(self, relation_type, name).await
    }

    async fn get_schema(
        &self,
        name: String,
    ) -> Option<Result<SchemaRef, ConnectorError>> {
        Some(SchemaSupport::get_schema(self, &name).await)
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
}
