use serde::{Deserialize, Serialize};

use crate::connectors::duckdb::DuckDBProfile;

#[derive(Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ProfileType {
    #[serde(rename = "duckdb")]
    DuckDB(DuckDBProfile),
}

#[derive(Serialize, Deserialize)]
pub struct Profile {
    pub name: String,
    pub profile: ProfileType,
}
