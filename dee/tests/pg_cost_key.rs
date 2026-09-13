//! The cost-model backend key on Postgres.
//!
//! The DuckDB half of this is covered in-tree
//! (`opt::hmp::tests::the_duckdb_backend_key_names_the_file_and_its_storage_settings`),
//! which runs against an in-memory database. Postgres needs a live server, so
//! these are `#[ignore]`d and run explicitly:
//!
//! ```bash
//! cargo test -p dee --test pg_cost_key -- --ignored --nocapture
//! ```
//!
//! What they establish is that a Postgres server identifies itself well enough
//! that its seconds-per-byte constants are never pooled with another engine's
//! --- and that the settings deciding what a write costs are part of that
//! identity, since `synchronous_commit = off` and `on` do not share a constant.

use dee::connectors::Connector;
use dee::connectors::postgres::{PostgresConfig, PostgresConnection};
use serde_json::json;

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

async fn conn() -> std::sync::Arc<PostgresConnection> {
    let cfg: PostgresConfig = serde_json::from_value(json!({
        "host": env_or("DEE_PG_HOST", "0.0.0.0"),
        "port": env_or("DEE_PG_PORT", "5432").parse::<i32>().unwrap(),
        "user": env_or("DEE_PG_USER", "runner"),
        "password": env_or("DEE_PG_PASSWORD", "password"),
        "database": env_or("DEE_PG_DB", "benchmark"),
        "num_connections": 2,
    }))
    .unwrap();
    PostgresConnection::new(cfg).await.expect("postgres")
}

#[tokio::test]
#[ignore = "needs a live Postgres"]
async fn the_postgres_key_names_the_server_and_its_durability_settings() {
    let key = conn()
        .await
        .cost_backend_key()
        .await
        .expect("the key query ran")
        .expect("a key");
    println!("{key}");

    assert!(key.starts_with("postgres "), "{key}");
    // The server, not just the engine: two clusters on one host, or one cluster
    // holding two databases on different tablespaces, are different engines as
    // far as a write rate is concerned.
    assert!(key.contains("server="), "{key}");
    assert!(key.contains(&format!("/{}", env_or("DEE_PG_DB", "benchmark"))), "{key}");
    // The settings that decide how much work a durable write is. The gap
    // between `synchronous_commit` off and on is larger than any effect the
    // write model tries to capture.
    for setting in [
        "wal_level",
        "synchronous_commit",
        "fsync",
        "full_page_writes",
        "wal_compression",
        "max_wal_size",
        "checkpoint_timeout",
        "checkpoint_completion_target",
    ] {
        assert!(key.contains(&format!("{setting}=")), "{setting} missing from {key}");
    }
    // Readable through `pg_settings`, which every role can read: a key that
    // depended on the connected role's privileges would split one server's
    // samples by whoever happened to run the DAG.
    assert!(!key.contains("restricted"), "a setting was unreadable: {key}");
}

/// The key does not drift between reads. It is the primary key of the stored
/// constants, so a key that changed shape run to run would refit from nothing
/// every time.
#[tokio::test]
#[ignore = "needs a live Postgres"]
async fn the_postgres_key_is_stable_across_connections() {
    let a = conn().await.cost_backend_key().await.unwrap().unwrap();
    let b = conn().await.cost_backend_key().await.unwrap().unwrap();
    assert_eq!(a, b);
}

/// A DuckDB key and a Postgres key are never equal, which is the property the
/// whole change exists for: before it, one unkeyed row served both.
#[tokio::test]
#[ignore = "needs a live Postgres"]
async fn a_postgres_key_is_never_a_duckdb_key() {
    use dee::connectors::duckdb::{DuckDBConfig, DuckDBConnection};
    let pg = conn().await.cost_backend_key().await.unwrap().unwrap();
    let duck = DuckDBConnection::new(DuckDBConfig::new_from_path(":memory:".to_string()))
        .await
        .unwrap()
        .cost_backend_key()
        .await
        .unwrap()
        .unwrap();
    assert_ne!(pg, duck);
}
