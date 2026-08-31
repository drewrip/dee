//! Executing a run group.
//!
//! This is `dee-cli/src/run.rs`'s repetition loop moved into the server: for
//! each run in the group, optionally clean up, then execute the DAG and record
//! what happened. The differences are that the connection pool is already warm,
//! and that every observation is persisted instead of accumulated in memory
//! and printed.

use std::sync::Arc;

use chrono::Utc;
use dee::dag::Dag;
use dee::executor::{Executor, ExecutorError, ProfilingConfig, SimpleEngine};
use tokio::sync::watch;

use crate::error::ServerError;
use crate::exec::connectors::ConnectorHandle;
use crate::state::AppState;
use crate::store::repo::{connections, dags, runs};

/// Run every repetition in `group_id`, recording each.
///
/// Errors are reported through the store rather than returned: by the time a
/// group is dispatched nobody is awaiting it, so a failure that only propagated
/// upward would vanish.
pub async fn drive_group(state: AppState, group_id: String) {
    let outcome = drive_group_inner(state.clone(), group_id.clone()).await;

    let error = match outcome {
        Ok(()) => None,
        Err(e) => {
            let message = e.to_string();
            log::error!("run group {group_id} failed: {message}");
            let _ = runs::log_event(
                &state.store,
                None,
                Some(group_id.clone()),
                None,
                "error",
                message.clone(),
            )
            .await;
            Some(message)
        }
    };

    if let Err(e) = runs::finalize_group(&state.store, group_id.clone(), error).await {
        log::error!("could not finalize run group {group_id}: {e}");
    }
    state.runs.finish(&group_id).await;
    // The DAG is free now, so whatever was waiting behind it can start without
    // waiting for the dispatcher's next tick.
    state.wake_queue();
}

async fn drive_group_inner(state: AppState, group_id: String) -> Result<(), ServerError> {
    let group = runs::get_group(&state.store, group_id.clone())
        .await?
        .ok_or_else(|| ServerError::NotFound("run group", group_id.clone()))?;

    let definition = dags::definition(&state.store, group.dag_id.clone(), group.dag_version)
        .await?
        .ok_or_else(|| {
            ServerError::NotFound(
                "dag version",
                format!("{} v{}", group.dag_name, group.dag_version),
            )
        })?;
    let dag = Dag::try_from(definition)
        .map_err(|e| ServerError::Internal(format!("stored dag no longer parses: {e}")))?;

    // Node materializations come from the definition rather than from
    // ExecStats, which does not carry them.
    let materializations: Vec<(String, String)> = dag
        .nodes
        .nodes()
        .map(|n| (n.id.clone(), n.materialize.as_str().to_string()))
        .collect();

    let connection = connections::get(&state.store, group.target.clone())
        .await?
        .ok_or_else(|| ServerError::NotFound("connection", group.target.clone()))?;
    let handle = state.connectors.acquire(&connection).await?;

    runs::mark_group_running(&state.store, group_id.clone()).await?;
    runs::log_event(
        &state.store,
        None,
        Some(group_id.clone()),
        Some(group.dag_id.clone()),
        "info",
        format!(
            "starting {} on '{}' (v{}, {} warmup(s), {} repetition(s))",
            group.dag_name, group.target, group.dag_version, group.warmups, group.repetitions
        ),
    )
    .await?;

    let series = runs::runs_in_group(&state.store, group_id.clone()).await?;

    match handle {
        ConnectorHandle::DuckDb(conn) => {
            run_series(&state, &group, &dag, &materializations, series, conn, "duckdb_json").await
        }
        ConnectorHandle::Postgres(conn) => {
            run_series(&state, &group, &dag, &materializations, series, conn, "postgres_json").await
        }
    }
}

async fn run_series<C>(
    state: &AppState,
    group: &runs::RunGroupRow,
    dag: &Dag,
    materializations: &[(String, String)],
    series: Vec<runs::RunRow>,
    conn: Arc<C>,
    plan_format: &str,
) -> Result<(), ServerError>
where
    C: dee::connectors::Connector + Send + Sync + 'static,
{
    let plan_time_basis = conn.time_basis().as_str().to_string();

    // One engine for the whole group. Building it is cheap; what matters is
    // that every repetition shares the already-warm pool, which is what makes
    // per-repetition timings measure the DAG and not connection setup.
    let mut engine = SimpleEngine::new(Arc::clone(&conn))
        .map_err(|e| ServerError::Internal(format!("building the engine: {e}")))?;
    if group.collect_plans || group.sample_interval_ms.is_some() {
        engine = engine.with_profiling(ProfilingConfig {
            collect_plans: group.collect_plans,
            sample_interval: std::time::Duration::from_millis(
                group.sample_interval_ms.unwrap_or(250).max(1) as u64,
            ),
        });
    }

    let cancel = engine.cancel_sender();
    state
        .runs
        .register_cancel(&group.run_group_id, cancel.clone())
        .await;

    for run in series {
        // A cancel that lands between repetitions must stop the series, not
        // just the repetition that was in flight.
        if *cancel.borrow() {
            break;
        }

        runs::mark_run_running(&state.store, run.run_id.clone()).await?;

        let cleanup_started = Utc::now();
        if group.cleanup_before {
            if let Err(e) = engine.cleanup(dag).await {
                let message = format!("cleanup before {} rep {}: {e}", run.phase, run.rep_index);
                runs::mark_run_terminal(
                    &state.store,
                    run.run_id.clone(),
                    runs::status::FAILED,
                    Some(message.clone()),
                )
                .await?;
                return Err(ServerError::Internal(message));
            }
        }
        let cleanup_ms = (Utc::now() - cleanup_started).num_milliseconds();

        match engine.run(dag).await {
            Ok(stats) => {
                let duration_ms = stats.duration.num_milliseconds();
                runs::record_success(
                    &state.store,
                    run.run_id.clone(),
                    stats,
                    materializations.to_vec(),
                    plan_format.to_string(),
                    plan_time_basis.clone(),
                    cleanup_ms,
                )
                .await?;
                runs::log_event(
                    &state.store,
                    Some(run.run_id.clone()),
                    Some(group.run_group_id.clone()),
                    Some(group.dag_id.clone()),
                    "info",
                    format!(
                        "{} {}/{} finished in {duration_ms}ms",
                        run.phase,
                        run.rep_index + 1,
                        if run.phase == "warmup" {
                            group.warmups
                        } else {
                            group.repetitions
                        }
                    ),
                )
                .await?;
            }
            Err(ExecutorError::Cancelled) => {
                runs::mark_run_terminal(
                    &state.store,
                    run.run_id.clone(),
                    runs::status::CANCELLED,
                    Some("cancelled".into()),
                )
                .await?;
                // The engine latches its cancel flag and never lowers it, so
                // clear it before the engine could be used again.
                engine.reset_cancel();
                break;
            }
            Err(e) => {
                let message = e.to_string();
                runs::mark_run_terminal(
                    &state.store,
                    run.run_id.clone(),
                    runs::status::FAILED,
                    Some(message.clone()),
                )
                .await?;
                // Later repetitions of a failed series would measure nothing
                // useful; finalize_group marks them skipped.
                return Err(ServerError::Internal(message));
            }
        }
    }

    Ok(())
}

/// A cancel handle, so the API can stop a group it did not start.
pub type CancelHandle = Arc<watch::Sender<bool>>;
