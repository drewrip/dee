use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Where the server listens, what it writes to, and the knobs that govern how
/// eagerly it works.
#[derive(Clone, Debug)]
pub struct ServerConfig {
    pub bind: SocketAddr,
    pub metadata_db: PathBuf,
    pub tick_interval: Duration,
    pub max_concurrent_runs: usize,
    pub connector_idle_ttl: Duration,
    pub store_pool_size: u32,
}

pub const DEFAULT_PORT: u16 = 8471;

impl Default for ServerConfig {
    fn default() -> Self {
        ServerConfig {
            bind: SocketAddr::from(([127, 0, 0, 1], DEFAULT_PORT)),
            metadata_db: default_data_dir().join("dee.duckdb"),
            tick_interval: Duration::from_secs(1),
            max_concurrent_runs: 4,
            connector_idle_ttl: Duration::from_secs(900),
            store_pool_size: 8,
        }
    }
}

/// `$DEE_HOME`, else `~/.dee`, else the working directory. The last fallback
/// only matters on a machine with no home directory set.
pub fn default_data_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("DEE_HOME") {
        return PathBuf::from(dir);
    }
    match std::env::var("HOME") {
        Ok(home) if !home.is_empty() => Path::new(&home).join(".dee"),
        _ => PathBuf::from("."),
    }
}
