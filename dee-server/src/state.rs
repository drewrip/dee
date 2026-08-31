use std::sync::Arc;

use chrono::{DateTime, Utc};

use crate::config::ServerConfig;
use crate::exec::connectors::ConnectorCache;
use crate::exec::manager::RunManager;
use crate::store::Store;

/// Everything a request handler needs. Cheap to clone: `Store` is a pool
/// handle and the rest is shared behind an `Arc`.
#[derive(Clone)]
pub struct AppState {
    pub store: Store,
    pub config: Arc<ServerConfig>,
    pub connectors: Arc<ConnectorCache>,
    pub runs: Arc<RunManager>,
    pub instance_id: String,
    pub started_at: DateTime<Utc>,
}

impl AppState {
    pub fn new(store: Store, config: ServerConfig, instance_id: String) -> Self {
        let connectors = Arc::new(ConnectorCache::new(config.connector_idle_ttl));
        let runs = Arc::new(RunManager::new(config.max_concurrent_runs));
        AppState {
            store,
            config: Arc::new(config),
            connectors,
            runs,
            instance_id,
            started_at: Utc::now(),
        }
    }
}

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
