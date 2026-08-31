//! The metadata store: an r2d2 pool over a DuckDB file, reached from async
//! code through `spawn_blocking`.
//!
//! Two facts about the `duckdb` crate shape this module. Every API on
//! `Connection` is blocking, so any use from tokio needs `spawn_blocking`
//! regardless of the surrounding design. And `DuckdbConnectionManager` holds a
//! single `Arc<Mutex<Connection>>` behind the pool, so a pool is N handles onto
//! one in-process database rather than N independent connections -- concurrent
//! readers are fine under MVCC.
//!
//! Writes additionally take a `tokio::sync::Mutex` permit. DuckDB's optimistic
//! concurrency aborts a transaction on a write-write conflict, and while dee's
//! writes mostly touch disjoint rows, serializing them costs nothing at a few
//! hundred rows per run and removes the conflict-retry question entirely.

pub mod repo;
pub mod schema;

use std::path::Path;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use duckdb::{Connection, DuckdbConnectionManager};
use thiserror::Error;

use schema::MIGRATIONS;

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("metadata database error: {0}")]
    Duck(#[from] duckdb::Error),

    /// Opening failed in a way that looks like another process holds the file.
    /// DuckDB allows only one read-write process per file, so this is the
    /// error a second `dee serve` on the same metadata database hits.
    #[error(
        "metadata database '{path}' is already open by another dee server \
         (DuckDB allows one process at a time): {source}"
    )]
    AlreadyOpen {
        path: String,
        source: duckdb::Error,
    },

    #[error("connection pool error: {0}")]
    Pool(String),

    #[error("store task failed: {0}")]
    Join(String),

    #[error("{0} not found")]
    NotFound(String),

    #[error("stored {what} is not valid JSON: {source}")]
    Decode {
        what: &'static str,
        source: serde_json::Error,
    },
}

#[derive(Clone)]
pub struct Store {
    pool: r2d2::Pool<DuckdbConnectionManager>,
    /// Held across every write so two writers never race into a DuckDB
    /// transaction conflict.
    write: Arc<tokio::sync::Mutex<()>>,
}

impl Store {
    /// Open (creating if absent) the metadata database at `path` and bring its
    /// schema up to date.
    ///
    /// The file is created mode 0600 on Unix: connection configs, including
    /// Postgres passwords, are stored in it.
    pub fn open(path: &Path, pool_size: u32) -> Result<Self, StoreError> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| StoreError::Pool(format!("creating {}: {e}", parent.display())))?;
            }
        }

        let manager = DuckdbConnectionManager::file(path).map_err(|e| {
            if is_lock_error(&e) {
                StoreError::AlreadyOpen {
                    path: path.display().to_string(),
                    source: e,
                }
            } else {
                StoreError::Duck(e)
            }
        })?;
        let pool = r2d2::Pool::builder()
            .max_size(pool_size.max(1))
            .build(manager)
            .map_err(|e| StoreError::Pool(e.to_string()))?;

        restrict_permissions(path);

        let store = Store {
            pool,
            write: Arc::new(tokio::sync::Mutex::new(())),
        };
        {
            let conn = store.checkout()?;
            migrate(&conn)?;
        }
        Ok(store)
    }

    /// An in-memory store, for tests that do not care about a file.
    #[cfg(test)]
    pub fn open_temporary() -> Result<Self, StoreError> {
        let manager = DuckdbConnectionManager::memory()?;
        let pool = r2d2::Pool::builder()
            .max_size(4)
            .build(manager)
            .map_err(|e| StoreError::Pool(e.to_string()))?;
        let store = Store {
            pool,
            write: Arc::new(tokio::sync::Mutex::new(())),
        };
        {
            let conn = store.checkout()?;
            migrate(&conn)?;
        }
        Ok(store)
    }

    fn checkout(&self) -> Result<r2d2::PooledConnection<DuckdbConnectionManager>, StoreError> {
        self.pool.get().map_err(|e| StoreError::Pool(e.to_string()))
    }

    /// Run `f` against the database. Concurrent with other reads.
    pub async fn read<T, F>(&self, f: F) -> Result<T, StoreError>
    where
        F: FnOnce(&Connection) -> Result<T, StoreError> + Send + 'static,
        T: Send + 'static,
    {
        let pool = self.pool.clone();
        tokio::task::spawn_blocking(move || {
            let conn = pool.get().map_err(|e| StoreError::Pool(e.to_string()))?;
            f(&conn)
        })
        .await
        .map_err(|e| StoreError::Join(e.to_string()))?
    }

    /// Run `f` against the database, serialized against every other write.
    /// Reads still proceed concurrently.
    ///
    /// `f` may open a transaction; multi-table writes should, so a failure
    /// partway cannot leave a run half-recorded.
    pub async fn write<T, F>(&self, f: F) -> Result<T, StoreError>
    where
        F: FnOnce(&Connection) -> Result<T, StoreError> + Send + 'static,
        T: Send + 'static,
    {
        let _permit = self.write.lock().await;
        self.read(f).await
    }
}

/// DuckDB reports a busy file through its IO error text rather than a distinct
/// variant, so this matches on the message. Being wrong only costs a less
/// specific error message.
fn is_lock_error(e: &duckdb::Error) -> bool {
    let text = e.to_string().to_ascii_lowercase();
    text.contains("lock") || text.contains("being used by another process")
}

#[cfg(unix)]
fn restrict_permissions(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    if let Err(e) = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)) {
        log::warn!(
            "could not restrict permissions on {}: {e}; it holds connection credentials",
            path.display()
        );
    }
}

#[cfg(not(unix))]
fn restrict_permissions(_path: &Path) {}

/// Apply every migration this binary knows about that the database has not
/// seen, in order. Each runs in its own transaction, so a failure leaves the
/// recorded version behind rather than a half-applied schema.
fn migrate(conn: &Connection) -> Result<(), StoreError> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS schema_migrations (
             version    INTEGER PRIMARY KEY,
             applied_at TIMESTAMPTZ NOT NULL
         );",
    )?;

    let applied = current_version(conn)?;

    for migration in MIGRATIONS {
        if migration.version <= applied {
            continue;
        }
        log::info!("applying metadata migration {}", migration.version);
        conn.execute_batch("BEGIN TRANSACTION;")?;
        let result = (|| -> Result<(), StoreError> {
            conn.execute_batch(migration.sql)?;
            conn.execute(
                "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                duckdb::params![migration.version, Utc::now()],
            )?;
            Ok(())
        })();
        match result {
            Ok(()) => conn.execute_batch("COMMIT;")?,
            Err(e) => {
                let _ = conn.execute_batch("ROLLBACK;");
                return Err(e);
            }
        }
    }
    Ok(())
}

pub fn current_version(conn: &Connection) -> Result<i32, StoreError> {
    let version: Option<i32> = conn.query_row(
        "SELECT max(version) FROM schema_migrations",
        [],
        |row| row.get(0),
    )?;
    Ok(version.unwrap_or(0))
}

/// SQL for binding a Rust string list into a `VARCHAR[]` column.
///
/// duckdb-rs has no `ToSql`/`FromSql` for lists, so JSON is the boundary
/// format: bind the list as a JSON string and let DuckDB build the real list.
/// The column stays a native `VARCHAR[]`, which is what makes a
/// DuckDB-to-parquet export of the metadata produce a genuine list column.
pub const LIST_PARAM: &str = "from_json(?, '[\"VARCHAR\"]')";

/// The value to bind wherever [`LIST_PARAM`] appears in a statement.
pub fn list_param(items: &[String]) -> String {
    serde_json::to_string(items).expect("a list of strings always serializes")
}

/// Read a `VARCHAR[]` column back. Select it as `to_json(col)`.
pub fn parse_list(json: &str) -> Result<Vec<String>, StoreError> {
    serde_json::from_str(json).map_err(|source| StoreError::Decode {
        what: "list column",
        source,
    })
}

/// A time-ordered id. UUIDv7 embeds its timestamp, so `ORDER BY id` is
/// chronological and rows need no separate sequence column.
pub fn new_id() -> String {
    uuid::Uuid::now_v7().to_string()
}

/// Record this server's presence. `stopped_at` stays null until a clean
/// shutdown, which is how the orphan sweep tells a crash from an exit.
pub async fn register_instance(
    store: &Store,
    instance_id: String,
    bind: String,
    version: String,
) -> Result<(), StoreError> {
    let pid = std::process::id() as i32;
    let now = Utc::now();
    store
        .write(move |conn| {
            conn.execute(
                "INSERT INTO server_instances
                     (instance_id, pid, version, bind, started_at, stopped_at)
                 VALUES (?, ?, ?, ?, ?, NULL)",
                duckdb::params![instance_id, pid, version, bind, now],
            )?;
            Ok(())
        })
        .await
}

pub async fn mark_instance_stopped(store: &Store, instance_id: String) -> Result<(), StoreError> {
    let now = Utc::now();
    store
        .write(move |conn| {
            conn.execute(
                "UPDATE server_instances SET stopped_at = ? WHERE instance_id = ?",
                duckdb::params![now, instance_id],
            )?;
            Ok(())
        })
        .await
}

/// How many jobs the previous life of this database left mid-flight.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SweptOrphans {
    pub runs: usize,
    pub run_groups: usize,
    pub optimizations: usize,
}

/// Move everything left `queued` or `running` by a departed server into a
/// terminal `orphaned` state.
///
/// Without this, a killed server leaves rows that claim to be running forever:
/// history reads as permanently in-flight, and the overlap check would refuse
/// to ever schedule that DAG again.
pub async fn sweep_orphans(store: &Store) -> Result<SweptOrphans, StoreError> {
    let now = Utc::now();
    store
        .write(move |conn| {
            const REASON: &str = "server restarted while this was in flight";
            let mut swept = SweptOrphans::default();
            for (table, counter) in [
                ("runs", 0usize),
                ("run_groups", 1),
                ("optimizations", 2),
            ] {
                let n = conn.execute(
                    &format!(
                        "UPDATE {table} SET status = 'orphaned', finished_at = ?, error = ?
                         WHERE status IN ('queued', 'running')"
                    ),
                    duckdb::params![now, REASON],
                )?;
                match counter {
                    0 => swept.runs = n,
                    1 => swept.run_groups = n,
                    _ => swept.optimizations = n,
                }
            }
            Ok(swept)
        })
        .await
}

/// When this database was last migrated. Surfaced by `/v1/info`.
pub async fn last_migrated_at(store: &Store) -> Result<Option<DateTime<Utc>>, StoreError> {
    store
        .read(|conn| {
            let ts: Option<DateTime<Utc>> = conn.query_row(
                "SELECT max(applied_at) FROM schema_migrations",
                [],
                |row| row.get(0),
            )?;
            Ok(ts)
        })
        .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn temp_store() -> Store {
        Store::open_temporary().expect("open temporary store")
    }

    #[tokio::test]
    async fn test_migrations_bring_a_fresh_database_to_the_latest_version() {
        let store = temp_store();
        let version = store.read(|c| current_version(c)).await.unwrap();
        assert_eq!(version, schema::latest_version());
        assert!(version > 0, "there should be at least one migration");
    }

    #[tokio::test]
    async fn test_migrating_an_already_current_database_is_a_no_op() {
        // `open` runs migrate; running it again must not fail on the
        // already-created tables or double-count a version.
        let store = temp_store();
        store
            .write(|c| {
                migrate(c)?;
                migrate(c)?;
                Ok(())
            })
            .await
            .unwrap();

        let rows: i64 = store
            .read(|c| Ok(c.query_row("SELECT count(*) FROM schema_migrations", [], |r| r.get(0))?))
            .await
            .unwrap();
        assert_eq!(rows as usize, MIGRATIONS.len());
    }

    #[tokio::test]
    async fn test_every_declared_table_exists_after_migration() {
        // Guards against a table being added to the plan but not the SQL, and
        // against a typo that silently creates a differently-named table.
        let store = temp_store();
        let expected = [
            "schema_migrations",
            "server_instances",
            "connections",
            "dags",
            "dag_versions",
            "dag_version_nodes",
            "dag_version_sources",
            "schedules",
            "schedule_skips",
            "run_groups",
            "runs",
            "node_executions",
            "plans",
            "run_samples",
            "optimizations",
            "optimization_passes",
            "optimization_iterations",
            "events",
        ];
        for table in expected {
            let n: i64 = store
                .read(move |c| {
                    Ok(c.query_row(
                        "SELECT count(*) FROM duckdb_tables() WHERE table_name = ?",
                        duckdb::params![table],
                        |r| r.get(0),
                    )?)
                })
                .await
                .unwrap();
            assert_eq!(n, 1, "table '{table}' is missing after migration");
        }
    }

    #[tokio::test]
    async fn test_list_and_timestamp_columns_round_trip() {
        // `depends_on VARCHAR[]` and `TIMESTAMPTZ` are load-bearing across the
        // schema, so prove both survive a write/read cycle rather than
        // discovering it when a run tries to record itself.
        let store = temp_store();
        let now = Utc::now();

        store
            .write(move |c| {
                c.execute(
                    "INSERT INTO dag_versions
                        (dag_id, version, content_hash, definition, sql_dialect,
                         node_count, source_count, origin, created_at)
                     VALUES ('d', 1, 'h', '{}', 'duckdb', 1, 0, 'submitted', ?)",
                    duckdb::params![now],
                )?;
                c.execute(
                    &format!(
                        "INSERT INTO dag_version_nodes
                            (dag_id, version, node_id, materialize, query_text,
                             depends_on, out_degree, paths_to_sinks)
                         VALUES ('d', 1, 'n', 'view', 'select 1', {LIST_PARAM}, 2, 3)"
                    ),
                    duckdb::params![list_param(&[
                        "\"w\".\"m\".\"a\"".to_string(),
                        "b".to_string(),
                    ])],
                )?;
                Ok(())
            })
            .await
            .unwrap();

        let (deps, created): (String, DateTime<Utc>) = store
            .read(|c| {
                Ok(c.query_row(
                    "SELECT to_json(n.depends_on), v.created_at
                     FROM dag_version_nodes n JOIN dag_versions v USING (dag_id, version)",
                    [],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )?)
            })
            .await
            .unwrap();

        // Node ids are quoted, fully-qualified relation names, so the encoding
        // has to survive embedded double quotes.
        assert_eq!(
            parse_list(&deps).unwrap(),
            vec!["\"w\".\"m\".\"a\"".to_string(), "b".to_string()]
        );
        assert_eq!(created.timestamp_millis(), now.timestamp_millis());
    }

    #[tokio::test]
    async fn test_concurrent_writes_are_serialized() {
        // The write permit is what lets callers read-modify-write without
        // guarding each site themselves. If it stops working, this counter
        // loses increments.
        let store = temp_store();
        store
            .write(|c| {
                c.execute_batch("CREATE TABLE counter (n BIGINT); INSERT INTO counter VALUES (0);")?;
                Ok(())
            })
            .await
            .unwrap();

        let mut handles = Vec::new();
        for _ in 0..64 {
            let store = store.clone();
            handles.push(tokio::spawn(async move {
                store
                    .write(|c| {
                        let n: i64 = c.query_row("SELECT n FROM counter", [], |r| r.get(0))?;
                        c.execute("UPDATE counter SET n = ?", duckdb::params![n + 1])?;
                        Ok(())
                    })
                    .await
            }));
        }
        for h in handles {
            h.await.unwrap().unwrap();
        }

        let n: i64 = store
            .read(|c| Ok(c.query_row("SELECT n FROM counter", [], |r| r.get(0))?))
            .await
            .unwrap();
        assert_eq!(n, 64, "a concurrent write was lost");
    }

    #[tokio::test]
    async fn test_reads_are_not_blocked_by_the_write_permit() {
        // Reads deliberately skip the permit. Hold it and prove a read still
        // completes, so a long write cannot stall the API.
        let store = temp_store();
        let permit = store.write.clone().lock_owned().await;

        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();
        let s = store.clone();
        let read = tokio::spawn(async move {
            s.read(|conn| Ok(conn.query_row("SELECT 1", [], |r| r.get::<_, i32>(0))?))
                .await
                .unwrap();
            c.fetch_add(1, Ordering::SeqCst);
        });

        tokio::time::timeout(std::time::Duration::from_secs(5), read)
            .await
            .expect("read blocked behind the write permit")
            .unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 1);
        drop(permit);
    }

    #[tokio::test]
    async fn test_sweep_orphans_terminates_in_flight_work() {
        let store = temp_store();
        let now = Utc::now();
        store
            .write(move |c| {
                for (id, status) in [("r1", "running"), ("r2", "queued"), ("r3", "succeeded")] {
                    c.execute(
                        "INSERT INTO runs
                            (run_id, run_group_id, dag_id, dag_version, target, status,
                             queued_at, instance_id)
                         VALUES (?, 'g', 'd', 1, 't', ?, ?, 'i')",
                        duckdb::params![id, status, now],
                    )?;
                }
                c.execute(
                    "INSERT INTO run_groups
                        (run_group_id, dag_id, dag_version, target, trigger, status,
                         created_at, instance_id)
                     VALUES ('g', 'd', 1, 't', 'manual', 'running', ?, 'i')",
                    duckdb::params![now],
                )?;
                Ok(())
            })
            .await
            .unwrap();

        let swept = sweep_orphans(&store).await.unwrap();
        assert_eq!(swept.runs, 2);
        assert_eq!(swept.run_groups, 1);
        assert_eq!(swept.optimizations, 0);

        let (orphaned, succeeded): (i64, i64) = store
            .read(|c| {
                Ok(c.query_row(
                    "SELECT count(*) FILTER (WHERE status = 'orphaned'),
                            count(*) FILTER (WHERE status = 'succeeded')
                     FROM runs",
                    [],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )?)
            })
            .await
            .unwrap();
        assert_eq!(orphaned, 2);
        assert_eq!(succeeded, 1, "a finished run must not be swept");

        // A second sweep has nothing left to do.
        assert_eq!(sweep_orphans(&store).await.unwrap(), SweptOrphans::default());
    }

    #[tokio::test]
    async fn test_instance_lifecycle_records_a_clean_exit() {
        let store = temp_store();
        register_instance(&store, "i1".into(), "127.0.0.1:1".into(), "0.1.0".into())
            .await
            .unwrap();

        let stopped: Option<DateTime<Utc>> = store
            .read(|c| {
                Ok(c.query_row("SELECT stopped_at FROM server_instances", [], |r| r.get(0))?)
            })
            .await
            .unwrap();
        assert!(stopped.is_none(), "a live instance has no stopped_at");

        mark_instance_stopped(&store, "i1".into()).await.unwrap();
        let stopped: Option<DateTime<Utc>> = store
            .read(|c| {
                Ok(c.query_row("SELECT stopped_at FROM server_instances", [], |r| r.get(0))?)
            })
            .await
            .unwrap();
        assert!(stopped.is_some(), "a clean exit must be recorded");
    }

    #[tokio::test]
    async fn test_open_creates_the_file_and_survives_reopening() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("dee.duckdb");

        let store = Store::open(&path, 4).unwrap();
        store
            .write(|c| {
                c.execute(
                    "INSERT INTO dags
                        (dag_id, name, current_version, created_at, updated_at)
                     VALUES ('d', 'churn', 1, now(), now())",
                    [],
                )?;
                Ok(())
            })
            .await
            .unwrap();
        drop(store);

        let reopened = Store::open(&path, 4).unwrap();
        let name: String = reopened
            .read(|c| Ok(c.query_row("SELECT name FROM dags", [], |r| r.get(0))?))
            .await
            .unwrap();
        assert_eq!(name, "churn");
    }

    #[cfg(unix)]
    #[test]
    fn test_metadata_file_is_not_world_readable() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dee.duckdb");
        let _store = Store::open(&path, 2).unwrap();

        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o077, 0, "the file holds connection credentials");
    }
}
