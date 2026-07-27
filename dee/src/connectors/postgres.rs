use crate::{
    connectors::{Connector, ConnectorError},
    dag::MaterializeMode,
};
use async_trait::async_trait;
use duckdb::arrow::datatypes::SchemaRef;
use serde::{Deserialize, Serialize};
use sqlx::{
    ConnectOptions, Executor, PgPool, Row,
    postgres::{PgConnectOptions, PgPoolOptions},
};
use std::{sync::Arc, time::Duration};

#[derive(Serialize, Deserialize, Clone)]
pub struct PostgresConfig {
    host: String,
    port: Option<i32>,
    user: String,
    password: String,
    database: String,
    num_connections: Option<u32>,
}

impl PostgresConfig {}

pub struct PostgresConnection {
    pool: PgPool,
}

fn materialize_mode_in_pg(mode: MaterializeMode) -> String {
    match mode {
        MaterializeMode::Table | MaterializeMode::TempTable => "TABLE".to_string(),
        MaterializeMode::View => "VIEW".to_string(),
    }
}

#[async_trait]
impl Connector for PostgresConnection {
    type Config = PostgresConfig;
    type Connection = PostgresConnection;

    async fn new(config: Self::Config) -> Result<Arc<Self::Connection>, ConnectorError> {
        let conn_options = PgConnectOptions::new_without_pgpass()
            .host(&config.host)
            .port(config.port.unwrap_or(5432) as u16)
            .username(&config.user)
            .password(&config.password)
            .database(&config.database)
            .log_slow_statements(log::LevelFilter::Off, Duration::from_hours(2));

        let pool = PgPoolOptions::new()
            .max_connections(config.num_connections.unwrap_or(4))
            .connect_with(conn_options)
            .await
            .map_err(|_| ConnectorError::Create("couldn't create PgPool".into()))?;
        let pg_conn = PostgresConnection { pool };
        Ok(Arc::new(pg_conn))
    }

    async fn execute(&self, query_text: String) -> Result<usize, ConnectorError> {
        let mut conn = self.pool.acquire().await.map_err(|e| {
            ConnectorError::Execute(format!("couldn't retrieve connection from pool - {}", e))
        })?;
        let temp_q: &str = &query_text;
        let rows = conn
            .execute(temp_q)
            .await
            .map_err(|e| ConnectorError::Execute(format!("couldn't execute SQL - {}", e)))?;
        Ok(rows.rows_affected() as usize)
    }

    async fn new_relation(
        &self,
        relation_type: MaterializeMode,
        name: String,
        query_text: String,
    ) -> Result<usize, ConnectorError> {
        let ddl_text = match relation_type {
            MaterializeMode::View => format!("CREATE OR REPLACE VIEW {} AS ({})", name, query_text),
            MaterializeMode::Table | MaterializeMode::TempTable => {
                format!("CREATE TABLE {} AS ({})", name, query_text)
            }
        };
        self.execute(ddl_text).await
    }

    async fn drop_relation(
        &self,
        relation_type: MaterializeMode,
        name: String,
    ) -> Result<usize, ConnectorError> {
        let text_rel_type = materialize_mode_in_pg(relation_type);
        let ddl_text = format!("DROP {} IF EXISTS {} CASCADE", text_rel_type, name);
        self.execute(ddl_text).await
    }

    async fn get_schema(&self, _name: String) -> Option<Result<SchemaRef, ConnectorError>> {
        None
    }

    async fn sample_system_memory_usage(&self) -> Result<Option<u64>, ConnectorError> {
        let row = sqlx::query(
            "SELECT COALESCE(SUM(total_bytes), 0)::BIGINT AS memory_bytes FROM pg_backend_memory_contexts",
        )
        .fetch_one(&self.pool)
        .await
        .map_err(|e| ConnectorError::Execute(format!("Failed to sample memory usage: {}", e)))?;

        let memory_bytes: i64 = row.try_get("memory_bytes").map_err(|e| {
            ConnectorError::Execute(format!("Failed to decode memory usage sample: {}", e))
        })?;

        Ok(Some(memory_bytes.max(0) as u64))
    }
}
