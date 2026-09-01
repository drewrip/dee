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

use dee::opt::{RunContext, StepPhase};

use crate::error::ServerError;
use crate::exec::connectors::ConnectorHandle;
use crate::exec::stepper;
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
    let dag = Dag::try_from(definition.clone())
        .map_err(|e| ServerError::Internal(format!("stored dag no longer parses: {e}")))?;

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
            run_series(
                &state,
                &group,
                &dag,
                &definition.sources,
                series,
                conn,
                "duckdb_json",
            )
            .await
        }
        ConnectorHandle::Postgres(conn) => {
            run_series(
                &state,
                &group,
                &dag,
                &definition.sources,
                series,
                conn,
                "postgres_json",
            )
            .await
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_series<C>(
    state: &AppState,
    group: &runs::RunGroupRow,
    dag: &Dag,
    sources: &[dee::file::DagFileSource],
    series: Vec<runs::RunRow>,
    conn: Arc<C>,
    plan_format: &str,
) -> Result<(), ServerError>
where
    C: dee::connectors::Connector + Send + Sync + 'static,
{
    let plan_time_basis = conn.time_basis().as_str().to_string();

    // Which continuous optimizations this DAG is under. Built before the
    // engine, because whether a search that ranks by operator cost is attached
    // decides whether this group has to collect plans.
    let bare_engine = Arc::new(
        SimpleEngine::new(Arc::clone(&conn))
            .map_err(|e| ServerError::Internal(format!("building the engine: {e}")))?,
    );
    let mut steppers =
        stepper::build(state, &group.dag_id, Arc::clone(&conn), Arc::clone(&bare_engine)).await?;
    let collect_plans = group.collect_plans || stepper::wants_plans(&steppers);

    // One engine for the whole group. Building it is cheap; what matters is
    // that every repetition shares the already-warm pool, which is what makes
    // per-repetition timings measure the DAG and not connection setup.
    let mut engine = SimpleEngine::new(Arc::clone(&conn))
        .map_err(|e| ServerError::Internal(format!("building the engine: {e}")))?;
    if collect_plans || group.sample_interval_ms.is_some() {
        engine = engine.with_profiling(ProfilingConfig {
            collect_plans,
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

    // The definition this group's repetitions execute, and the version they are
    // recorded against.
    //
    // A group normally pins both: nothing that happens between repetitions
    // changes what they run, which is what makes their timings comparable to
    // each other. A continuous optimization converging mid-group is the one
    // exception, and deliberately so -- it has just produced a better version
    // of this very DAG, and running the remaining repetitions against the
    // superseded one would measure a DAG dee has already moved past. The
    // repetitions on either side of the promotion are then measuring different
    // things, which is exactly what `runs.dag_version` records per run.
    let mut base = dag.clone();
    let mut base_version = group.dag_version;

    for run in series {
        // A cancel that lands between repetitions must stop the series, not
        // just the repetition that was in flight.
        if *cancel.borrow() {
            break;
        }

        runs::mark_run_running(&state.store, run.run_id.clone()).await?;

        // What this execution actually runs. A `Before` step may rewrite it
        // into a candidate, which is why every repetition starts from a fresh
        // copy of the stored definition rather than carrying the last one's
        // rewrite forward.
        let mut working = base.clone();
        let before = stepper::step_all(
            &mut steppers,
            Arc::clone(&conn),
            Arc::clone(&bare_engine),
            &mut working,
            &group.dag_id,
            &group.dag_name,
            base_version,
            StepPhase::Before,
            Some(RunContext {
                run_id: run.run_id.clone(),
                run_group_id: group.run_group_id.clone(),
                run_phase: run.phase.clone(),
                rep_index: run.rep_index,
                stats: None,
            }),
        )
        .await;
        for (name, label) in &before.trials {
            runs::log_event(
                &state.store,
                Some(run.run_id.clone()),
                Some(group.run_group_id.clone()),
                Some(group.dag_id.clone()),
                "info",
                format!("{name} is trying {label} on this run"),
            )
            .await?;
        }
        // A promotion from a `Before` step is the search finishing: the DAG it
        // produced is stored, and this run executes it rather than a candidate.
        if !before.promoted.is_empty() || !before.finished.is_empty() {
            let promoted = stepper::apply(
                state,
                &group.dag_id,
                &group.dag_name,
                base_version,
                sources,
                before,
            )
            .await;
            // This run executes the promotion, so it is a run of the new
            // version. The group was dispatched against the old one, and
            // leaving the run pointing there would make `dee runs list`
            // disagree with what the run actually did.
            if let Some(version) = promoted {
                // Every repetition from here on runs the promotion too.
                base = working.clone();
                base_version = version;
                runs::log_event(
                    &state.store,
                    Some(run.run_id.clone()),
                    Some(group.run_group_id.clone()),
                    Some(group.dag_id.clone()),
                    "info",
                    format!(
                        "an optimization converged before this run; \
                         it and the repetitions after it execute v{version}"
                    ),
                )
                .await?;
            }
        }

        // The run rows were created when the group was, against the version
        // current then. Once an optimization has promoted, that is no longer
        // the version these repetitions execute, and a run that says otherwise
        // would put the pre- and post-convergence timings under one label.
        if base_version != group.dag_version {
            runs::set_run_version(&state.store, run.run_id.clone(), base_version).await?;
        }

        let dag = &working;

        // Node materializations come from the DAG that is about to run rather
        // than from ExecStats, which does not carry them -- and from `working`
        // rather than the stored definition, because a continuous optimization
        // may have rewritten this run into a candidate, and a recorded run
        // must describe what it executed.
        let materializations: Vec<(String, String)> = dag
            .nodes
            .nodes()
            .map(|n| (n.id.clone(), n.materialize.as_str().to_string()))
            .collect();

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

                // What the candidate cost. Stepped before the run is recorded
                // so an optimization that converges here has its result stored
                // alongside the run that decided it.
                let mut observed = working.clone();
                let after = stepper::step_all(
                    &mut steppers,
                    Arc::clone(&conn),
                    Arc::clone(&bare_engine),
                    &mut observed,
                    &group.dag_id,
                    &group.dag_name,
                    group.dag_version,
                    StepPhase::After,
                    Some(RunContext {
                        run_id: run.run_id.clone(),
                        run_group_id: group.run_group_id.clone(),
                        run_phase: run.phase.clone(),
                        rep_index: run.rep_index,
                        stats: Some(stats.clone()),
                    }),
                )
                .await;
                if !after.is_empty() {
                    if let Some(version) = stepper::apply(
                        state,
                        &group.dag_id,
                        &group.dag_name,
                        base_version,
                        sources,
                        after,
                    )
                    .await
                    {
                        // Promoted after the run, so it describes the next
                        // repetition rather than this one; this run keeps the
                        // version it executed.
                        base = observed.clone();
                        base_version = version;
                    }
                }

                runs::record_success(
                    &state.store,
                    run.run_id.clone(),
                    stats,
                    materializations,
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
