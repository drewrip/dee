use serde::{Deserialize, Serialize};

use crate::connectors::{duckdb::DuckDBConfig, postgres::PostgresConfig};

#[derive(Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum Connection {
    #[serde(rename = "duckdb")]
    DuckDB(DuckDBConfig),
    #[serde(rename = "postgres")]
    Postgres(PostgresConfig),
}
