use serde::{Deserialize, Serialize};

use crate::connectors::{duckdb::DuckDBProfile, postgres::PostgresProfile};

#[derive(Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Profile {
    #[serde(rename = "duckdb")]
    DuckDB(DuckDBProfile),
    #[serde(rename = "postgres")]
    Postgres(PostgresProfile),
}
