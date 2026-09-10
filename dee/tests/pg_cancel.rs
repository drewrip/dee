//! Mid-query cancellation on Postgres.
//!
//! The DuckDB half of this property is covered in-tree
//! (`executor::tests::test_a_budget_actually_stops_a_long_running_query`),
//! which can run against an in-memory database. Postgres needs a live server,
//! so these are `#[ignore]`d and run explicitly:
//!
//! ```bash
//! cargo test -p dee --test pg_cancel -- --ignored --nocapture
//! ```
//!
//! What they establish is the property cancel-and-resume rests on: a budget
//! cuts a *single long-running statement* short. If the budget only stopped
//! waiting for the statement, the overrun would be however long the longest
//! node takes -- which is exactly the tail the feature exists to bound.

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use dee::connectors::postgres::{PostgresConfig, PostgresConnection};
use dee::connectors::Connector;
use dee::dag::{Dag, MaterializeMode, TransformNode};
use dee::executor::{Executor, RunOptions, SimpleEngine, StopReason};
use dee::graph::Graph;
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
        "num_connections": 8,
    }))
    .unwrap();
    PostgresConnection::new(cfg).await.expect("postgres")
}

/// One node that would take a minute. Nothing to schedule around it, so the
/// only way to stop the run is to stop the statement.
fn one_slow_node(id: &str) -> Dag {
    let mut map = HashMap::new();
    map.insert(
        id.to_string(),
        TransformNode {
            id: id.to_string(),
            query_text: "SELECT 1 AS n FROM (SELECT pg_sleep(60)) AS s".to_string(),
            materialize: MaterializeMode::Table,
            depends_on: HashSet::new(),
            schema: None,
        },
    );
    Dag {
        db: "postgres".to_string(),
        nodes: Graph::new(map),
        sources: Vec::new(),
        max_parallelism: None,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn a_budget_stops_a_long_running_postgres_query() {
    let engine = SimpleEngine::new(conn().await).expect("engine");
    let dag = one_slow_node("dee_cancel_probe");
    let started = Instant::now();
    let outcome = engine
        .run_with(
            &dag,
            RunOptions {
                budget: Some(Duration::from_millis(300)),
                cleanup_on_cancel: true,
                ..RunOptions::default()
            },
        )
        .await
        .expect("run");
    let elapsed = started.elapsed();

    assert_eq!(outcome.stopped, Some(StopReason::Budget));
    assert!(
        outcome.completed.is_empty(),
        "a 60s sleep cannot have finished in {elapsed:?}"
    );
    assert!(
        elapsed < Duration::from_secs(10),
        "the budget did not bind: the run took {elapsed:?} against a 300ms budget, \
         which means the statement kept going after it was cancelled"
    );
    eprintln!("postgres budget stop took {elapsed:?} against a 60s statement");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn nothing_is_still_running_once_a_stopped_postgres_run_returns() {
    // "Signalled" is not "stopped". The resume drops and rebuilds relations
    // immediately after this returns, so a backend still writing one would
    // race that.
    let connector = conn().await;
    let engine = SimpleEngine::new(std::sync::Arc::clone(&connector)).expect("engine");
    let dag = one_slow_node("dee_cancel_probe2");
    let outcome = engine
        .run_with(
            &dag,
            RunOptions {
                budget: Some(Duration::from_millis(300)),
                cleanup_on_cancel: true,
                ..RunOptions::default()
            },
        )
        .await
        .expect("run");
    assert_eq!(outcome.stopped, Some(StopReason::Budget));

    assert_eq!(
        connector.interrupt_inflight().await,
        0,
        "a backend was still running a statement after the run returned"
    );
    connector
        .drop_relation(MaterializeMode::Table, "dee_cancel_probe2".into())
        .await
        .ok();
    connector
        .new_relation(
            MaterializeMode::Table,
            "dee_cancel_probe2".into(),
            "SELECT 1 AS n".into(),
        )
        .await
        .expect("the relation could not be rebuilt after the cancelled run");
    connector
        .drop_relation(MaterializeMode::Table, "dee_cancel_probe2".into())
        .await
        .ok();
}
