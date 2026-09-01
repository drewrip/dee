//! The metadata store as an optimization sees it.
//!
//! `dee::opt::OptStore` is the port an optimization keeps its state through;
//! this is the server's implementation of it. Each handle is scoped to one
//! optimization's table namespace, so a pass can create, read and drop the
//! tables it owns and nothing else -- the metadata database also holds every
//! run, plan and connection credential dee has recorded, and a materialization
//! search has no business anywhere near them.
//!
//! The SQL-to-JSON translation itself lives in `dee::opt::store`, shared with
//! the in-process store the library's own tests use, so the two cannot drift.

use std::sync::Arc;

use async_trait::async_trait;
use dee::opt::store::{OptStore, OptStoreError, execute_on, query_on, table_prefix};
use dee::opt::OptStoreFactory;
use serde_json::Value;

use crate::store::Store;

/// One optimization's view of the metadata database.
pub struct ScopedStore {
    store: Store,
    optimization: String,
    prefix: String,
}

impl ScopedStore {
    pub fn new(store: Store, optimization: &str) -> Self {
        Self {
            store,
            optimization: optimization.to_string(),
            prefix: table_prefix(optimization),
        }
    }
}

#[async_trait]
impl OptStore for ScopedStore {
    fn table_prefix(&self) -> &str {
        &self.prefix
    }

    async fn execute(&self, sql: &str, params: &[Value]) -> Result<usize, OptStoreError> {
        let (optimization, prefix) = (self.optimization.clone(), self.prefix.clone());
        let (sql, params) = (sql.to_string(), params.to_vec());
        // Through `Store::write`, so an optimization's writes serialize
        // against every other writer exactly as a run's do. DuckDB aborts a
        // transaction on a write-write conflict, and a step that lost one
        // would look to the pass like a measurement that never happened.
        self.store
            .write(move |conn| {
                execute_on(conn, &optimization, &prefix, &sql, &params)
                    .map_err(|e| crate::store::StoreError::Pool(e.to_string()))
            })
            .await
            .map_err(|e| OptStoreError::Backend(e.to_string()))
    }

    async fn query(&self, sql: &str, params: &[Value]) -> Result<Vec<Value>, OptStoreError> {
        let (optimization, prefix) = (self.optimization.clone(), self.prefix.clone());
        let (sql, params) = (sql.to_string(), params.to_vec());
        self.store
            .read(move |conn| {
                query_on(conn, &optimization, &prefix, &sql, &params)
                    .map_err(|e| crate::store::StoreError::Pool(e.to_string()))
            })
            .await
            .map_err(|e| OptStoreError::Backend(e.to_string()))
    }
}

/// Hands out a [`ScopedStore`] per optimization.
#[derive(Clone)]
pub struct StoreFactory {
    store: Store,
}

impl StoreFactory {
    pub fn new(store: Store) -> Self {
        Self { store }
    }
}

impl OptStoreFactory for StoreFactory {
    fn store_for(&self, optimization: &str) -> Arc<dyn OptStore> {
        Arc::new(ScopedStore::new(self.store.clone(), optimization))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn scoped(optimization: &str) -> ScopedStore {
        ScopedStore::new(Store::open_temporary().unwrap(), optimization)
    }

    #[tokio::test]
    async fn test_an_optimization_can_build_and_read_its_own_tables() {
        let store = scoped("hmp").await;
        store
            .execute(
                "CREATE TABLE IF NOT EXISTS opt_hmp_state (dag_id VARCHAR, state VARCHAR)",
                &[],
            )
            .await
            .unwrap();
        store
            .execute(
                "INSERT INTO opt_hmp_state (dag_id, state) VALUES (?, ?)",
                &[serde_json::json!("d1"), serde_json::json!("{\"phase\":\"baseline\"}")],
            )
            .await
            .unwrap();

        let rows = store
            .query(
                "SELECT dag_id, state FROM opt_hmp_state WHERE dag_id = ?",
                &[serde_json::json!("d1")],
            )
            .await
            .unwrap();

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["dag_id"], serde_json::json!("d1"));
        assert_eq!(rows[0]["state"], serde_json::json!("{\"phase\":\"baseline\"}"));
    }

    #[tokio::test]
    async fn test_an_optimization_cannot_read_the_servers_own_tables() {
        // The guarantee that makes it safe to let optimization code write SQL
        // against the database holding connection credentials.
        let store = scoped("hmp").await;
        let error = store
            .query("SELECT config FROM connections", &[])
            .await
            .expect_err("reading outside the namespace must be refused");
        assert!(error.to_string().contains("connections"));
    }

    #[tokio::test]
    async fn test_one_optimization_cannot_touch_anothers_tables() {
        // Namespaces are per optimization, not merely "not the server's", so a
        // bug in one pass cannot corrupt another's search.
        let store = scoped("hmp").await;
        let error = store
            .execute("DROP TABLE IF EXISTS opt_omp_state", &[])
            .await
            .expect_err("another optimization's tables must be refused");
        assert!(error.to_string().contains("opt_omp_state"));
    }
}
