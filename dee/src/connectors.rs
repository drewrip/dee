use async_trait::async_trait;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::physical_plan::ExecutionPlan;
use std::sync::Arc;
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

    async fn sample_system_cpu_usage(&self) -> Result<Option<f64>, ConnectorError> {
        Ok(None)
    }

    async fn sample_system_memory_usage(&self) -> Result<Option<u64>, ConnectorError> {
        Ok(None)
    }

    /// Parse `EXPLAIN (FORMAT JSON)` output into a DataFusion
    /// [`ExecutionPlan`].
    ///
    /// Returns `Some(plan)` when the connector can convert its explain
    async fn explain_to_logical_plan(
        &self,
        _json_plan: &str,
        _schema: SchemaRef,
    ) -> Option<Arc<dyn ExecutionPlan>> {
        None
    }
}
