use std::sync::Arc;

use chrono::{DateTime, Utc};
use tokio::sync::Notify;

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
    /// Kicked whenever the queue might have become dispatchable -- a group
    /// finishing, an entry being enqueued. The dispatcher also ticks, so a
    /// missed signal costs latency, never a stuck queue.
    pub queue_wake: Arc<Notify>,
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
            queue_wake: Arc::new(Notify::new()),
            instance_id,
            started_at: Utc::now(),
        }
    }

    /// Tell the queue dispatcher to look again.
    pub fn wake_queue(&self) {
        self.queue_wake.notify_one();
    }
}

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
