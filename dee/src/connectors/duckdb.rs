use std::path::PathBuf;

use duckdb::{Connection, params};

use crate::connectors::{Connector, ConnectorError};

pub struct DuckDBProfile {
    pub db: DuckDBType,
}

impl DuckDBProfile {
    pub fn new_with_path(path: PathBuf) -> Self {
        Self {
            db: DuckDBType::File(path),
        }
    }

    pub fn new_in_memory() -> Self {
        Self {
            db: DuckDBType::Ephemeral,
        }
    }
}

pub enum DuckDBType {
    File(PathBuf),
    Ephemeral,
}

pub struct DuckDBConnection {
    pub conn: Connection,
}

impl Connector for DuckDBConnection {
    type Profile = DuckDBProfile;
    type Connection = DuckDBConnection;

    fn new(profile: Self::Profile) -> Result<Self::Connection, ConnectorError> {
        let conn = match profile.db {
            DuckDBType::Ephemeral => Connection::open_in_memory(),
            DuckDBType::File(path) => Connection::open(path),
        }
        .map_err(|_| ConnectorError::Create("duckdb issue".to_string()))?;

        Ok(Self::Connection { conn })
    }

    fn execute(&mut self, query_text: String) -> Result<usize, ConnectorError> {
        self.conn
            .execute(&query_text, params![])
            .map_err(|e| ConnectorError::Execute(format!("{}", e.to_string())))
    }
}
