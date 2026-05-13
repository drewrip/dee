use crate::{
    connectors::{Connector, ConnectorError},
    dag::MaterializeMode,
};
use async_trait::async_trait;
use datafusion::arrow::datatypes::SchemaRef;
use duckdb::{Config, DuckdbConnectionManager, params};
use log::debug;
use r2d2::Pool;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, path::PathBuf, sync::Arc, time::Duration};
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

#[derive(Deserialize, Debug)]
struct DuckDBExplainNode {
    name: String,
    children: Vec<DuckDBExplainNode>,
    #[serde(default)]
    extra_info: HashMap<String, serde_json::Value>,
}

fn get_duckdb_weight(name: &str) -> f32 {
    match name {
        "HASH_JOIN" => 15.0,
        "NESTED_LOOP_JOIN" => 20.0,
        "CROSS_PRODUCT" => 100.0,
        "SEQ_SCAN" => 1.0,
        "INDEX_SCAN" => 0.5,
        "FILTER" => 0.1,
        "PROJECTION" => 0.1,
        "ORDER_BY" => 2.0,
        "TOP_N" => 0.1,
        "AGGREGATE" => 2.0,
        "HASH_GROUP_BY" => 15.0,
        "DISTINCT" => 2.0,
        "UNION" => 0.5,
        _ => 1.0,
    }
}

fn compute_duckdb_node_cost(node: &DuckDBExplainNode, in_dummy_cte: bool) -> f32 {
    let cte_name = node
        .extra_info
        .get("CTE Name")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let is_dummy_cte =
        (node.name == "CTE" || node.name == "CTE_SCAN") && cte_name.starts_with("__dee_dummy_scan");

    let weight = get_duckdb_weight(&node.name);
    let cardinality = node
        .extra_info
        .get("Estimated Cardinality")
        .and_then(|v| {
            if let Some(s) = v.as_str() {
                s.parse::<f32>().ok()
            } else {
                v.as_f64().map(|f| f as f32)
            }
        })
        .unwrap_or(1.0);

    let current_cost = cardinality * weight;

    if is_dummy_cte && in_dummy_cte {
        return current_cost;
    }

    let next_in_dummy = in_dummy_cte || is_dummy_cte;
    let children_cost: f32 = node
        .children
        .iter()
        .map(|child| compute_duckdb_node_cost(child, next_in_dummy))
        .sum();
    current_cost + children_cost
}

fn materialize_mode_in_duckdb(mode: MaterializeMode) -> String {
    match mode {
        MaterializeMode::Table => "TABLE".to_string(),
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
        //debug!("creating new_relation ({}, {})", rel_type, name);
        let tmpl_query = format!("CREATE {} {} AS ({})", rel_type, name, query_text);
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
            MaterializeMode::Table => {
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

                Ok((res, Some(json_str)))
            }
        }
    }

    async fn drop_relation(
        &self,
        relation_type: MaterializeMode,
        name: String,
    ) -> Result<usize, ConnectorError> {
        let rel_type = materialize_mode_in_duckdb(relation_type);
        debug!("attempt drop_relation ({}, {})", rel_type, name);
        let tmpl_query = format!("DROP {} IF EXISTS {}", rel_type, name);
        self.execute(tmpl_query).await
    }

    async fn get_schema(&self, name: String) -> Option<Result<SchemaRef, ConnectorError>> {
        debug!("attempt to fetch arrow schema for {}", name);
        let conn = self
            .pool
            .get()
            .map_err(|_| ConnectorError::Execute("didn't get connection from pool".to_string()))
            .unwrap();
        let tmpl_query = format!("SELECT * FROM {}", name);
        let stmt = conn.prepare(&tmpl_query).unwrap();
        let schema = stmt.schema().clone();
        Some(Ok(schema))
    }

    async fn cost(&self, query: String) -> Result<Option<f32>, ConnectorError> {
        let explain_query = format!("EXPLAIN (FORMAT json) {}", query);
        let conn = self
            .pool
            .get()
            .map_err(|_| ConnectorError::Execute("didn't get connection from pool".to_string()))?;

        let mut stmt = conn
            .prepare(&explain_query)
            .map_err(|e| ConnectorError::Execute(format!("Failed to prepare explain: {}", e)))?;

        let json_str: String = stmt
            .query_row([], |row| {
                // DuckDB JSON explain might return two columns: (key, value)
                // or just one column (value).
                let col_count = row.as_ref().column_count();
                if col_count >= 2 {
                    row.get(1)
                } else {
                    row.get(0)
                }
            })
            .map_err(|e| ConnectorError::Execute(format!("Failed to execute explain: {}", e)))?;

        let nodes: Vec<DuckDBExplainNode> = serde_json::from_str(&json_str)
            .map_err(|e| ConnectorError::Execute(format!("Failed to parse explain JSON: {}", e)))?;

        Ok(Some(
            nodes
                .iter()
                .map(|n| compute_duckdb_node_cost(n, false))
                .sum(),
        ))
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_duckdb_cost() {
        let config = DuckDBConfig::new_from_path(":memory:".to_string());
        let conn = DuckDBConnection::new(config).await.unwrap();

        // Create a dummy table to have some plan
        conn.execute("CREATE TABLE t1 AS SELECT 1 AS id".to_string())
            .await
            .unwrap();

        let cost = conn.cost("SELECT * FROM t1".to_string()).await.unwrap();
        assert!(cost.unwrap() > 0.0);
    }

    #[test]
    fn test_parse_duckdb_size_bytes() {
        assert_eq!(parse_duckdb_size_bytes("44.0 KiB"), Some(45056));
        assert_eq!(parse_duckdb_size_bytes("1.0 B"), Some(1));
    }
}
