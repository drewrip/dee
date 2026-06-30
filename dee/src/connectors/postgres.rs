use crate::{
    connectors::{
        Connector, ConnectorError, ExplainSupport, ProfilingSupport,
        RelationOps, SchemaSupport, SystemMetrics,
    },
    dag::MaterializeMode,
};
use async_trait::async_trait;
use datafusion::arrow::datatypes::SchemaRef;
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

// ---------------------------------------------------------------------------
// Capability trait implementations
// ---------------------------------------------------------------------------

#[async_trait]
impl SchemaSupport for PostgresConnection {
    async fn get_schema(&self, name: &str) -> Result<SchemaRef, ConnectorError> {
        // Postgres does not support direct Arrow schema retrieval.
        // Return an error indicating the capability is absent.
        Err(ConnectorError::Schema(format!(
            "Postgres does not support schema retrieval for '{}'", name
        )))
    }
}

#[async_trait]
impl ExplainSupport for PostgresConnection {
    async fn explain(&self, query: &str) -> Result<String, ConnectorError> {
        let rows = sqlx::query(
            "EXPLAIN (FORMAT JSON) $1",
        )
        .bind(query)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| ConnectorError::Execute { err: e.to_string(), query: query.to_string() })?;

        if rows.is_empty() {
            return Err(ConnectorError::ExplainNotSupported);
        }

        // Extract the JSON array string from the first column
        let explain_output: String = rows[0].get(0);
        Ok(explain_output)
    }
}

#[async_trait]
impl ProfilingSupport for PostgresConnection {
    type PlanData = String;

    async fn execute_with_profile(
        &self,
        _query: &str,
    ) -> Result<(usize, Option<Self::PlanData>), ConnectorError> {
        // Postgres does not support DuckDB-style profiling.
        Err(ConnectorError::ExplainNotSupported)
    }
}

#[async_trait]
impl RelationOps for PostgresConnection {
    async fn create_relation(
        &self,
        relation_type: MaterializeMode,
        name: String,
        query: String,
    ) -> Result<usize, ConnectorError> {
        let ddl_text = match relation_type {
            MaterializeMode::View => format!("CREATE OR REPLACE VIEW {} AS ({})", name, query),
            MaterializeMode::Table | MaterializeMode::TempTable => {
                format!("CREATE TABLE {} AS ({})", name, query)
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
}

#[async_trait]
impl SystemMetrics for PostgresConnection {
    async fn sample_cpu(&self) -> Result<Option<f64>, ConnectorError> {
        // No CPU sampling for Postgres (default)
        Ok(None)
    }

    async fn sample_memory(&self) -> Result<Option<u64>, ConnectorError> {
        let row = sqlx::query(
            "SELECT COALESCE(SUM(total_bytes), 0)::BIGINT AS memory_bytes FROM pg_backend_memory_contexts",
        )
        .fetch_one(&self.pool)
        .await
        .map_err(|e| ConnectorError::Execute { err: e.to_string(), query: "SELECT COALESCE(SUM(total_bytes), 0)::BIGINT AS memory_bytes FROM pg_backend_memory_contexts".to_string() })?;

        let memory_bytes: i64 = row.try_get("memory_bytes").map_err(|e| {
            ConnectorError::Execute { err: e.to_string(), query: "memory_bytes decode".to_string() }
        })?;

        Ok(Some(memory_bytes.max(0) as u64))
    }
}

// ---------------------------------------------------------------------------
// Connector impl (keeps backward-compatible method names)
// ---------------------------------------------------------------------------

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
            ConnectorError::Execute { err: format!("couldn't retrieve connection from pool - {}", e), query: query_text.clone() }
        })?;
        let temp_q: &str = &query_text;
        let rows = conn
            .execute(temp_q)
            .await
            .map_err(|e| ConnectorError::Execute { err: format!("couldn't execute SQL - {}", e), query: query_text })?;
        Ok(rows.rows_affected() as usize)
    }

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

    async fn dialect(&self) -> &'static str {
        "postgresql"
    }

}
