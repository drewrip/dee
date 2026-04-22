use crate::{
    connectors::{Connector, ConnectorError},
    dag::MaterializeMode,
};
use async_trait::async_trait;
use datafusion::arrow::datatypes::SchemaRef;
use log::debug;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Serialize, Deserialize, Clone)]
pub struct PostgresProfile {}

impl PostgresProfile {
    pub fn new() -> Self {
        Self {}
    }
}

pub struct PostgresConnection {}

#[async_trait]
impl Connector for PostgresConnection {
    type Profile = PostgresProfile;
    type Connection = PostgresConnection;

    fn new(profile: Self::Profile) -> Result<Arc<Self::Connection>, ConnectorError> {
        let conn = PostgresConnection {};
        Ok(Arc::new(conn))
    }

    async fn execute(&self, query_text: String) -> Result<usize, ConnectorError> {
        Ok(0)
    }

    async fn new_relation(
        &self,
        relation_type: MaterializeMode,
        name: String,
        query_text: String,
    ) -> Result<usize, ConnectorError> {
        Ok(0)
    }

    async fn drop_relation(
        &self,
        relation_type: MaterializeMode,
        name: String,
    ) -> Result<usize, ConnectorError> {
        Ok(0)
    }

    async fn get_schema(&self, name: String) -> Option<Result<SchemaRef, ConnectorError>> {
        None
    }
}
