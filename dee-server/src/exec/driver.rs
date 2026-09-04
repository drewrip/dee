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
use dee::executor::{Executor, ExecutorError, ProfilingConfig, RunOptions, SimpleEngine, StopReason};
use dee::opt::resume;
use tokio::sync::watch;

use dee::opt::{RunContext, StepPhase};

use crate::error::ServerError;
use crate::exec::connectors::ConnectorHandle;
use crate::exec::stepper;
use crate::state::AppState;
use crate::store::repo::{connections, dags, runs};

/// How much longer than the cancelled candidate's own budget the resume that
/// finishes the run may take. It is a delivery, not an experiment, so it must
/// not be cut short again -- but it stays bounded so a pathological engine
/// state cannot hang the group.
const RESUME_BUDGET_MULTIPLE: u32 = 3;

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
        let mut before = stepper::step_all(
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
        for trial in &before.trials {
            runs::log_event(
                &state.store,
                Some(run.run_id.clone()),
                Some(group.run_group_id.clone()),
                Some(group.dag_id.clone()),
                "info",
                format!("{} is trying {} on this run", trial.name, trial.label),
            )
            .await?;
        }
        // At most one candidate can be budgeted per run: two searches cancelling
        // the same execution would each attribute the stop to their own
        // candidate, and only one of them would be right. The first trial that
        // offers both a budget and an incumbent wins; anything else runs to
        // completion, which is what happened before this existed.
        //
        // Taken out of the report rather than borrowed from it: the report is
        // consumed by `apply` just below, and this has to outlive the run.
        let budgeted = before
            .trials
            .iter()
            .position(|t| t.budget_ms.is_some_and(|ms| ms > 0) && t.fallback.is_some())
            .map(|i| before.trials.remove(i));
        let budget = budgeted.as_ref().map(|t| {
            std::time::Duration::from_millis(t.budget_ms.expect("filtered above") as u64)
        });
        if let (Some(trial), Some(budget)) = (budgeted.as_ref(), budget) {
            log::debug!(
                "{}'s candidate {} runs under a {}ms budget, with a fallback to finish the run",
                trial.name,
                trial.label,
                budget.as_millis()
            );
        } else if let Some(trial) = before.trials.first() {
            log::debug!(
                "{}'s candidate {} runs to completion (budget {:?}, fallback {})",
                trial.name,
                trial.label,
                trial.budget_ms,
                trial.fallback.is_some()
            );
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

        let executed = engine
            .run_with(
                dag,
                RunOptions {
                    budget,
                    // Whatever a cancelled candidate built is what the resume
                    // reuses, so a budgeted run must not tidy up after itself.
                    cleanup_on_cancel: budget.is_none(),
                    ..RunOptions::default()
                },
            )
            .await;

        match executed {
            // A candidate that overran its budget. The verdict is drawn from
            // the censored run itself, so the `After` step happens *first* and
            // is handed no stats; only then is the delivery paid, under the
            // incumbent, on the warehouse the candidate half-filled.
            Ok(outcome) if outcome.stopped == Some(StopReason::Budget) => {
                let trial = budgeted
                    .as_ref()
                    .expect("a budget is only set with a trial behind it");
                let incumbent = trial.fallback.as_deref().expect("filtered above");
                let trial_ms = outcome.stats.duration.num_milliseconds();
                runs::log_event(
                    &state.store,
                    Some(run.run_id.clone()),
                    Some(group.run_group_id.clone()),
                    Some(group.dag_id.clone()),
                    "info",
                    format!(
                        "{}'s candidate {} exceeded its {trial_ms}ms budget after {}/{} node(s); \
                         finishing this run under the incumbent",
                        trial.name,
                        trial.label,
                        outcome.completed.len(),
                        working.nodes.num_nodes()
                    ),
                )
                .await?;

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
                        // No stats is the censored observation: "at least as
                        // slow as the cap". Every continuous pass reads it as a
                        // rejection and moves its search on.
                        stats: None,
                    }),
                )
                .await;
                if !after.is_empty()
                    && let Some(version) = stepper::apply(
                        state,
                        &group.dag_id,
                        &group.dag_name,
                        base_version,
                        sources,
                        after,
                    )
                    .await
                {
                    base = observed.clone();
                    base_version = version;
                }

                let plan = resume::plan(&working, incumbent, &outcome.completed);
                let reused = plan.reusable.len();
                let kept = plan.reusable.clone();
                resume::drop_relations(conn.as_ref(), &plan.to_drop).await;
                let resumed = engine
                    .run_with(
                        incumbent,
                        RunOptions {
                            skip: plan.reusable,
                            // A delivery must not be cut short again, but a
                            // pathological engine state must not hang the group.
                            budget: budget.map(|b| b * RESUME_BUDGET_MULTIPLE),
                            cleanup_on_cancel: false,
                        },
                    )
                    .await;

                let delivered = match resumed {
                    Ok(r) if r.stopped.is_none() => r,
                    Ok(_) | Err(_) => {
                        let message = format!(
                            "{}'s candidate was cancelled and the run could not be finished                              under the incumbent",
                            trial.name
                        );
                        runs::mark_run_terminal(
                            &state.store,
                            run.run_id.clone(),
                            runs::status::FAILED,
                            Some(message.clone()),
                        )
                        .await?;
                        engine.reset_cancel();
                        return Err(ServerError::Internal(message));
                    }
                };

                // One record for the whole delivery: the nodes the incumbent
                // built, plus the candidate's nodes that were *kept*. A node the
                // candidate built and the resume then dropped and rebuilt has
                // the resume's row already, and a landing pad the incumbent does
                // not have is not part of what was delivered at all -- recording
                // either would describe a warehouse that does not exist.
                //
                // The elapsed time is *not* a measurement of either DAG: the
                // resume started from a warm, half-built warehouse. That is what
                // `delivery` records, so nothing downstream compares it to a
                // clean run.
                let mut stats = delivered.stats;
                let resume_ms = stats.duration.num_milliseconds();
                for (id, node) in outcome.stats.node_stats {
                    if kept.contains(&id) {
                        stats.node_stats.entry(id).or_insert(node);
                    }
                }
                stats.start = outcome.stats.start;
                stats.duration = stats.finish - stats.start;

                runs::record_success(
                    &state.store,
                    run.run_id.clone(),
                    runs::Delivery::resumed(trial_ms, resume_ms),
                    stats,
                    // What is actually in the warehouse is the incumbent's
                    // relations, not the cancelled candidate's.
                    incumbent
                        .nodes
                        .nodes()
                        .map(|n| (n.id.clone(), n.materialize.as_str().to_string()))
                        .collect(),
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
                        "finished under the incumbent in {resume_ms}ms, reusing {reused} \
                         relation(s) the cancelled candidate had already built"
                    ),
                )
                .await?;
            }
            Ok(outcome) if outcome.stopped.is_none() => {
                let stats = outcome.stats;
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
                    runs::Delivery::direct(),
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
            // The user asked for this run to stop. Finishing it behind their
            // back would be wrong, so this stays exactly what it was.
            Ok(_) | Err(ExecutorError::Cancelled) => {
                // A budgeted run was told not to tidy up after itself, because
                // a resume was going to reuse what it built. A user's cancel
                // ends that: nothing will read those relations, and leaving a
                // half-built warehouse behind is not what cancelling did before
                // budgets existed.
                if budget.is_some()
                    && let Err(e) = engine.cleanup(dag).await
                {
                    log::warn!("cleanup after cancelling {}: {e}", run.run_id);
                }
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
