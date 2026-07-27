use async_trait::async_trait;
use ::duckdb::arrow::datatypes::SchemaRef;
use std::{collections::HashMap, sync::Arc};
use thiserror::Error;

use crate::dag::MaterializeMode;

/// All pre-implemented connectors
pub mod duckdb;
pub mod postgres;

#[derive(Error, Debug)]
pub enum ConnectorError {
    #[error("couldn't create a connection to the DB - {0}")]
    Create(String),
    #[error("couldn't execute query against connector - {0}")]
    Execute(String),
}

/// What can be pushed down into a single relation, as reported by a
/// connector's native query planner (e.g. DuckDB's `EXPLAIN (FORMAT JSON)`).
///
/// `projections` are the column names the plan actually reads from the
/// relation; `filters` are raw SQL predicate strings (in the connector's own
/// dialect) that the plan applies directly against a scan of the relation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PushdownInfo {
    pub projections: Vec<String>,
    pub filters: Vec<String>,
}

#[async_trait]
pub trait Connector {
    type Config;
    type Connection;

    async fn new(config: Self::Config) -> Result<Arc<Self::Connection>, ConnectorError>;

    async fn execute(&self, query_text: String) -> Result<usize, ConnectorError>;

    async fn new_relation(
        &self,
        relation_type: MaterializeMode,
        name: String,
        query_text: String,
    ) -> Result<usize, ConnectorError>;

    async fn new_relation_and_explain(
        &self,
        relation_type: MaterializeMode,
        name: String,
        query_text: String,
    ) -> Result<(usize, Option<String>), ConnectorError> {
        let res = self.new_relation(relation_type, name, query_text).await?;
        Ok((res, None))
    }

    async fn drop_relation(
        &self,
        relation_type: MaterializeMode,
        name: String,
    ) -> Result<usize, ConnectorError>;

    async fn get_schema(&self, name: String) -> Option<Result<SchemaRef, ConnectorError>>;

    /// Ask the connector's own query planner what can be pushed down into
    /// each relation `query_text` scans, keyed by relation name.
    ///
    /// Returns `Ok(None)` when the connector has no native way to answer
    /// this (e.g. Postgres today). Returns `Ok(Some(map))` — possibly with
    /// an empty map if the query scans no relations directly (e.g. a
    /// constant-only `SELECT`) — when the connector could analyze the query.
    async fn pushdown(
        &self,
        _query_text: &str,
    ) -> Result<Option<HashMap<String, PushdownInfo>>, ConnectorError> {
        Ok(None)
    }

    async fn sample_system_cpu_usage(&self) -> Result<Option<f64>, ConnectorError> {
        Ok(None)
    }

    async fn sample_system_memory_usage(&self) -> Result<Option<u64>, ConnectorError> {
        Ok(None)
    }
}
