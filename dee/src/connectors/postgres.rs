use crate::{
    connectors::{Connector, ConnectorError},
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
pub struct PostgresProfile {
    host: String,
    port: Option<i32>,
    user: String,
    password: String,
    database: String,
    num_connections: Option<u32>,
}

impl PostgresProfile {}

pub struct PostgresConnection {
    pool: PgPool,
}

#[derive(Deserialize, Debug)]
struct PostgresExplainNode {
    #[serde(rename = "Node Type")]
    node_type: String,
    #[serde(rename = "Plan Rows")]
    plan_rows: f32,
    #[serde(rename = "Plans")]
    #[serde(default)]
    plans: Vec<PostgresExplainNode>,
}

#[derive(Deserialize, Debug)]
struct PostgresExplainWrapper {
    #[serde(rename = "Plan")]
    plan: PostgresExplainNode,
}

fn get_postgres_weight(node_type: &str) -> f32 {
    match node_type {
        "Hash Join" => 2.0,
        "Nested Loop" => 5.0,
        "Merge Join" => 2.0,
        "Seq Scan" => 1.0,
        "Index Scan" => 0.5,
        "Index Only Scan" => 0.4,
        "Bitmap Heap Scan" => 0.8,
        "Bitmap Index Scan" => 0.4,
        "Sort" => 2.0,
        "Aggregate" => 2.0,
        "Hash" => 1.0,
        "Limit" => 0.1,
        _ => 1.0,
    }
}

fn compute_postgres_node_cost(node: &PostgresExplainNode) -> f32 {
    let weight = get_postgres_weight(&node.node_type);
    let current_cost = node.plan_rows * weight;
    let children_cost: f32 = node.plans.iter().map(compute_postgres_node_cost).sum();
    current_cost + children_cost
}

fn materialize_mode_in_pg(mode: MaterializeMode) -> String {
    match mode {
        MaterializeMode::Table => "TABLE".to_string(),
        MaterializeMode::View => "VIEW".to_string(),
    }
}

#[async_trait]
impl Connector for PostgresConnection {
    type Profile = PostgresProfile;
    type Connection = PostgresConnection;

    async fn new(profile: Self::Profile) -> Result<Arc<Self::Connection>, ConnectorError> {
        let conn_options = PgConnectOptions::new_without_pgpass()
            .host(&profile.host)
            .port(profile.port.unwrap_or(5432) as u16)
            .username(&profile.user)
            .password(&profile.password)
            .database(&profile.database)
            .log_slow_statements(log::LevelFilter::Off, Duration::from_hours(2));

        let pool = PgPoolOptions::new()
            .max_connections(profile.num_connections.unwrap_or(4))
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
            MaterializeMode::Table => {
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

    async fn cost(&self, query: String) -> Result<Option<f32>, ConnectorError> {
        let explain_query = format!("EXPLAIN (FORMAT JSON) {}", query);
        let row = sqlx::query(&explain_query)
            .fetch_one(&self.pool)
            .await
            .map_err(|e| ConnectorError::Execute(format!("Failed to execute explain: {}", e)))?;

        let json_value: serde_json::Value = row
            .try_get(0)
            .map_err(|e| ConnectorError::Execute(format!("Failed to get explain JSON: {}", e)))?;

        let wrappers: Vec<PostgresExplainWrapper> = serde_json::from_value(json_value).map_err(|e| {
            ConnectorError::Execute(format!("Failed to parse explain JSON: {}", e))
        })?;

        let total_cost = wrappers
            .iter()
            .map(|w| compute_postgres_node_cost(&w.plan))
            .sum();

        Ok(Some(total_cost))
    }
}
