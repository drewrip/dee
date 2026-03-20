use std::{path::PathBuf, sync::Arc, time::Duration};

use crate::{
    connectors::{Connector, ConnectorError},
    dag::MaterializeMode,
};
use async_trait::async_trait;
use duckdb::{Config, DuckdbConnectionManager, params};
use log::debug;
use r2d2::Pool;

pub struct DuckDBProfile {
    pub db: DuckDBType,
    pub num_connections: u32,
    pub threads: Option<i64>,
    pub max_memory: Option<String>,
}

impl DuckDBProfile {
    pub fn new_with_path(path: PathBuf) -> Self {
        Self {
            db: DuckDBType::File(path),
            ..Default::default()
        }
    }

    pub fn new_in_memory() -> Self {
        Self {
            db: DuckDBType::Ephemeral,
            ..Default::default()
        }
    }

    pub fn with_num_connections(mut self, num_connections: u32) -> Self {
        self.num_connections = num_connections;
        self
    }

    pub fn with_threads(mut self, num_threads: i64) -> Self {
        self.threads = Some(num_threads);
        self
    }

    pub fn with_max_memory(mut self, mem_str: String) -> Self {
        self.max_memory = Some(mem_str);
        self
    }
}

impl Default for DuckDBProfile {
    fn default() -> Self {
        Self {
            db: DuckDBType::Ephemeral,
            num_connections: 4,
            threads: None,
            max_memory: None,
        }
    }
}

pub enum DuckDBType {
    File(PathBuf),
    Ephemeral,
}

pub struct DuckDBConnection {
    pub pool: Pool<DuckdbConnectionManager>,
}

fn materialize_mode_in_duckdb(mode: MaterializeMode) -> String {
    match mode {
        MaterializeMode::Table => "TABLE".to_string(),
        MaterializeMode::View => "VIEW".to_string(),
        MaterializeMode::Incremental => {
            debug!("not sure how to handle Incremental mode in DuckDB... defaulting to VIEW");
            "VIEW".to_string()
        }
    }
}

#[async_trait]
impl Connector for DuckDBConnection {
    type Profile = DuckDBProfile;
    type Connection = DuckDBConnection;

    fn new(profile: Self::Profile) -> Result<Arc<Self::Connection>, ConnectorError> {
        let mut conf = Config::default();
        if let Some(max_mem) = profile.max_memory {
            conf = conf
                .max_memory(&max_mem)
                .map_err(|_| ConnectorError::Create("set max memory problem".to_string()))?;
        }
        if let Some(threads) = profile.threads {
            conf = conf
                .threads(threads)
                .map_err(|_| ConnectorError::Create("set threads problem".to_string()))?;
        }

        conf = conf
            .access_mode(duckdb::AccessMode::ReadWrite)
            .map_err(|_| ConnectorError::Create("set access_mode".to_string()))?;

        let manager = match profile.db {
            DuckDBType::File(path) => DuckdbConnectionManager::file_with_flags(path, conf),
            DuckDBType::Ephemeral => DuckdbConnectionManager::memory_with_flags(conf),
        }
        .map_err(|e| ConnectorError::Create(format!("connection manager - {}", e)))?;
        let pool = Pool::builder()
            .connection_timeout(Duration::from_hours(2))
            .max_size(profile.num_connections)
            .build(manager)
            .map_err(|_| ConnectorError::Create("r2d2 pool".to_string()))?;
        Ok(Arc::new(Self { pool }))
    }

    async fn execute(&self, query_text: String) -> Result<usize, ConnectorError> {
        let conn = self
            .pool
            .get()
            .map_err(|_| ConnectorError::Execute("didn't get connection from pool".to_string()))?;
        conn.execute(&query_text.clone(), params![]).map_err(|e| {
            ConnectorError::Execute(format!("{} - query_text:\n{}", e.to_string(), query_text))
        })
    }

    async fn new_relation(
        &self,
        relation_type: MaterializeMode,
        name: String,
        query_text: String,
    ) -> Result<usize, ConnectorError> {
        let rel_type = materialize_mode_in_duckdb(relation_type);
        //debug!("creating new_relation ({}, {})", rel_type, name);
        let tmpl_query = format!("CREATE {} '{}' AS ({})", rel_type, name, query_text);
        self.execute(tmpl_query).await
    }

    async fn drop_relation(
        &self,
        relation_type: MaterializeMode,
        name: String,
    ) -> Result<usize, ConnectorError> {
        let rel_type = materialize_mode_in_duckdb(relation_type);
        debug!("attempt drop_relation ({}, {})", rel_type, name);
        let tmpl_query = format!("DROP {} IF EXISTS {}", rel_type, name);
        self.execute(tmpl_query).await
    }
}
