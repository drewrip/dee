//! Cached connection pools, keyed by connection name and config hash.
//!
//! This is the main thing the daemon buys over the old CLI. Building a
//! connector is expensive: `DuckDBConnection::new` runs `INSTALL icu; LOAD
//! icu;` and builds an r2d2 pool, `PostgresConnection::new` builds a `PgPool`.
//! Every `dee-cli run` used to pay that on startup; here it is paid once per
//! (connection, config) and amortized across every run and every optimizer
//! candidate.
//!
//! The cost of caching is that a handle holds its database open. A DuckDB
//! warehouse file cannot be opened by a second process while we hold it, so
//! entries are dropped when their connection is edited (the config hash keys
//! the cache) or when they go unused for `idle_ttl`.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use dee::connections::Connection;
use dee::connectors::{Connector, duckdb::DuckDBConnection, postgres::PostgresConnection};
use tokio::sync::{Mutex, OnceCell};

use crate::error::ServerError;
use crate::store::repo::connections::ConnectionRow;

/// A live connector.
///
/// `Connector` has associated types and an `async fn new` returning
/// `Arc<Self::Connection>`, so it is not object-safe and `Box<dyn Connector>`
/// is impossible. Enum dispatch mirrors `dee::connections::Connection`, which
/// is the same two-arm shape the old CLI used at its single call site.
#[derive(Clone)]
pub enum ConnectorHandle {
    DuckDb(Arc<DuckDBConnection>),
    Postgres(Arc<PostgresConnection>),
}

impl ConnectorHandle {
    /// Whether operator timings from this backend are CPU time or wall time.
    /// Recorded per run: DuckDB reports one and Postgres the other, so the two
    /// must never be silently averaged together.
    pub fn time_basis(&self) -> &'static str {
        match self {
            ConnectorHandle::DuckDb(c) => c.time_basis().as_str(),
            ConnectorHandle::Postgres(c) => c.time_basis().as_str(),
        }
    }
}

struct Entry {
    /// A `OnceCell` rather than the handle itself so two concurrent misses for
    /// the same key wait on one construction instead of building two pools
    /// against the same database.
    cell: Arc<OnceCell<ConnectorHandle>>,
    last_used: Instant,
}

pub struct ConnectorCache {
    entries: Mutex<HashMap<CacheKey, Entry>>,
    idle_ttl: Duration,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct CacheKey {
    name: String,
    config_hash: String,
}

impl ConnectorCache {
    pub fn new(idle_ttl: Duration) -> Self {
        ConnectorCache {
            entries: Mutex::new(HashMap::new()),
            idle_ttl,
        }
    }

    /// Get the connector for `row`, building it if this is the first use of
    /// this configuration.
    pub async fn acquire(&self, row: &ConnectionRow) -> Result<ConnectorHandle, ServerError> {
        let key = CacheKey {
            name: row.name.clone(),
            config_hash: row.config_hash.clone(),
        };

        let cell = {
            let mut entries = self.entries.lock().await;
            // A new config hash supersedes every older pool for this name, and
            // dropping it here is what releases the previous warehouse file.
            entries.retain(|k, _| k.name != key.name || k.config_hash == key.config_hash);

            let entry = entries.entry(key).or_insert_with(|| Entry {
                cell: Arc::new(OnceCell::new()),
                last_used: Instant::now(),
            });
            entry.last_used = Instant::now();
            entry.cell.clone()
        };
        // The lock is released before construction: building a pool can take
        // seconds, and holding the map would serialize every other target.

        let connection = row.connection()?;
        cell.get_or_try_init(|| build(connection)).await.cloned()
    }

    /// Forget every pool for `name`, releasing its database.
    pub async fn invalidate(&self, name: &str) {
        self.entries.lock().await.retain(|k, _| k.name != name);
    }

    /// Drop pools unused for longer than the idle TTL.
    ///
    /// Without this a long-lived server pins every warehouse it has ever
    /// touched -- which for a benchmark sweep is one file per cell -- and
    /// eventually runs out of file descriptors.
    pub async fn sweep_idle(&self) -> usize {
        let ttl = self.idle_ttl;
        let mut entries = self.entries.lock().await;
        let before = entries.len();
        entries.retain(|_, e| e.last_used.elapsed() < ttl);
        before - entries.len()
    }

    #[cfg(test)]
    pub async fn len(&self) -> usize {
        self.entries.lock().await.len()
    }
}

async fn build(connection: Connection) -> Result<ConnectorHandle, ServerError> {
    match connection {
        Connection::DuckDB(config) => {
            let conn = DuckDBConnection::new(config)
                .await
                .map_err(|e| ServerError::BadRequest(format!("connecting to duckdb: {e}")))?;
            Ok(ConnectorHandle::DuckDb(conn))
        }
        Connection::Postgres(config) => {
            let conn = PostgresConnection::new(config)
                .await
                .map_err(|e| ServerError::BadRequest(format!("connecting to postgres: {e}")))?;
            Ok(ConnectorHandle::Postgres(conn))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::content_hash;
    use chrono::Utc;
    use serde_json::json;

    fn row(name: &str, path: &std::path::Path) -> ConnectionRow {
        let config = json!({
            "type": "duckdb", "database": path
        });
        ConnectionRow {
            name: name.to_string(),
            kind: "duckdb".into(),
            config_hash: content_hash(&config),
            config: config.to_string(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    #[tokio::test]
    async fn test_a_repeated_acquire_reuses_one_pool() {
        // Reuse is the point of the daemon: building a DuckDB connector runs
        // INSTALL/LOAD icu and builds a pool, which the old CLI paid per run.
        let dir = tempfile::tempdir().unwrap();
        let cache = ConnectorCache::new(Duration::from_secs(60));
        let row = row("wh", &dir.path().join("w.duckdb"));

        let first = cache.acquire(&row).await.unwrap();
        let second = cache.acquire(&row).await.unwrap();

        assert_eq!(cache.len().await, 1);
        match (first, second) {
            (ConnectorHandle::DuckDb(a), ConnectorHandle::DuckDb(b)) => {
                assert!(Arc::ptr_eq(&a, &b), "a second acquire rebuilt the pool");
            }
            _ => panic!("expected duckdb handles"),
        }
    }

    #[tokio::test]
    async fn test_concurrent_first_acquires_build_only_one_pool() {
        // Two runs starting at once must not open two pools onto one DuckDB
        // file, which is what a plain "check then build" would do.
        let dir = tempfile::tempdir().unwrap();
        let cache = Arc::new(ConnectorCache::new(Duration::from_secs(60)));
        let row = Arc::new(row("wh", &dir.path().join("w.duckdb")));

        let mut handles = Vec::new();
        for _ in 0..8 {
            let cache = cache.clone();
            let row = row.clone();
            handles.push(tokio::spawn(async move { cache.acquire(&row).await }));
        }
        let mut first: Option<Arc<DuckDBConnection>> = None;
        for h in handles {
            match h.await.unwrap().unwrap() {
                ConnectorHandle::DuckDb(c) => match &first {
                    None => first = Some(c),
                    Some(f) => assert!(Arc::ptr_eq(f, &c), "a concurrent acquire built a second pool"),
                },
                _ => panic!("expected duckdb handles"),
            }
        }
        assert_eq!(cache.len().await, 1);
    }

    #[tokio::test]
    async fn test_a_changed_config_supersedes_the_old_pool() {
        // The old entry must be dropped, not merely shadowed: while it lives
        // it holds the previous warehouse file open, and DuckDB allows only
        // one process per file.
        let dir = tempfile::tempdir().unwrap();
        let cache = ConnectorCache::new(Duration::from_secs(60));

        let before = row("wh", &dir.path().join("a.duckdb"));
        let after = row("wh", &dir.path().join("b.duckdb"));
        assert_ne!(before.config_hash, after.config_hash);

        cache.acquire(&before).await.unwrap();
        cache.acquire(&after).await.unwrap();

        assert_eq!(cache.len().await, 1, "the superseded pool is still cached");
    }

    #[tokio::test]
    async fn test_invalidate_releases_a_targets_pools() {
        let dir = tempfile::tempdir().unwrap();
        let cache = ConnectorCache::new(Duration::from_secs(60));
        cache.acquire(&row("wh", &dir.path().join("a.duckdb"))).await.unwrap();
        cache.acquire(&row("other", &dir.path().join("b.duckdb"))).await.unwrap();

        cache.invalidate("wh").await;

        assert_eq!(cache.len().await, 1, "invalidate must only drop the named target");
    }

    #[tokio::test]
    async fn test_idle_pools_are_swept() {
        // A sweep's whole job is to stop a long-lived server pinning every
        // warehouse a benchmark run ever touched.
        let dir = tempfile::tempdir().unwrap();
        let cache = ConnectorCache::new(Duration::from_millis(0));
        cache.acquire(&row("wh", &dir.path().join("a.duckdb"))).await.unwrap();

        assert_eq!(cache.sweep_idle().await, 1);
        assert_eq!(cache.len().await, 0);
    }

    #[tokio::test]
    async fn test_a_pool_in_use_is_not_swept() {
        let dir = tempfile::tempdir().unwrap();
        let cache = ConnectorCache::new(Duration::from_secs(3600));
        cache.acquire(&row("wh", &dir.path().join("a.duckdb"))).await.unwrap();

        assert_eq!(cache.sweep_idle().await, 0);
        assert_eq!(cache.len().await, 1);
    }

    #[tokio::test]
    async fn test_time_basis_matches_the_backend() {
        // Recorded per run because DuckDB reports CPU time and Postgres wall
        // time; the strings must be the ones the benchmark schema documents.
        let dir = tempfile::tempdir().unwrap();
        let cache = ConnectorCache::new(Duration::from_secs(60));
        let handle = cache.acquire(&row("wh", &dir.path().join("a.duckdb"))).await.unwrap();
        assert_eq!(handle.time_basis(), "cpu_time");
    }
}
