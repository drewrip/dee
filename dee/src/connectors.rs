use async_trait::async_trait;
use datafusion::arrow::datatypes::SchemaRef;
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
    type Profile;
    type Connection;

    async fn new(profile: Self::Profile) -> Result<Arc<Self::Connection>, ConnectorError>;

    async fn execute(&self, query_text: String) -> Result<usize, ConnectorError>;

    async fn new_relation(
        &self,
        relation_type: MaterializeMode,
        name: String,
        query_text: String,
    ) -> Result<usize, ConnectorError>;

    async fn drop_relation(
        &self,
        relation_type: MaterializeMode,
        name: String,
    ) -> Result<usize, ConnectorError>;

    async fn get_schema(&self, name: String) -> Option<Result<SchemaRef, ConnectorError>>;

    async fn cost(&self, _query: String) -> Result<Option<f32>, ConnectorError> {
        Ok(None)
    }
}
