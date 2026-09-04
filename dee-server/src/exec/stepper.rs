//! Stepping a DAG's registered optimizations around one of its runs.
//!
//! This is where the new interface earns its keep. A continuous optimization
//! improves a DAG by measuring it running, and the server is already running
//! it -- on a schedule, from a trigger, out of the queue. So rather than an
//! optimization buying its own runs in one expensive burst, it gets a turn on
//! either side of each execution the DAG was going to perform anyway: `Before`
//! to install the candidate this run should try, `After` to learn what it
//! cost.
//!
//! Two consequences follow, and both are deliberate:
//!
//! * A trial is never persisted. The `Before` step rewrites the in-memory DAG
//!   for that one execution; only a converged optimization mints a version.
//!   Otherwise a search would bury a DAG's real history under its rejected
//!   experiments.
//! * A cancellation budget is honoured here, but only halfway: the run is the
//!   DAG's real work, so abandoning a losing candidate is not enough on its
//!   own -- the run is then finished under that search's incumbent, rebuilding
//!   only what the cancelled candidate never got to. A trial that offers a
//!   budget without an incumbent to fall back on is measured to completion,
//!   because a pipeline that did not run is not an outcome a search may choose.

use std::sync::Arc;

use dee::connectors::Connector;
use dee::dag::Dag;
use dee::executor::Executor;
use dee::opt::store::OptStore;
use dee::opt::{
    Optimization, OptimizationType, RunContext, StepContext, StepOutcome, StepPhase, registry,
};

use crate::error::ServerError;
use crate::state::AppState;
use crate::store::optstore::ScopedStore;
use crate::store::repo::{dags, registrations};

/// One registered optimization, built and ready to step.
pub struct Stepper<C, E>
where
    C: Connector + Send + Sync + 'static,
    E: Executor<C> + Send + Sync + 'static,
{
    pub name: String,
    pub step_phase: StepPhase,
    optimization: Box<dyn Optimization<C, E>>,
    store: Arc<dyn OptStore>,
}

/// A candidate one optimization installed on this run.
pub struct InstalledTrial {
    /// The optimization that proposed it.
    pub name: String,
    /// Human-readable identity of the candidate.
    pub label: String,
    /// Cancel the run once it has taken this long.
    pub budget_ms: Option<i64>,
    /// The DAG to finish the run under if the candidate is cancelled: that
    /// search's incumbent. Without one there is nothing better to fall back to,
    /// and the candidate is measured to completion however slow it turns out --
    /// a pipeline that did not run is not an outcome a search gets to choose.
    pub fallback: Option<Box<Dag>>,
}

/// What stepping a DAG's optimizations produced for one execution.
#[derive(Default)]
pub struct StepReport {
    /// Every candidate installed on this run.
    pub trials: Vec<InstalledTrial>,
    /// Optimizations that converged, with the DAG each promoted.
    pub promoted: Vec<(String, Dag)>,
    /// Optimizations that finished without a rewrite worth storing.
    pub finished: Vec<String>,
}

impl StepReport {
    pub fn is_empty(&self) -> bool {
        self.trials.is_empty() && self.promoted.is_empty() && self.finished.is_empty()
    }
}

/// Build the continuous optimizations registered on `dag_id` that step on
/// `side`.
///
/// Rebuilt per run rather than held across them: an optimization's decisions
/// come from the metadata database, not from anything it holds in memory, so
/// there is nothing to keep alive -- and a rebuilt one behaves identically
/// after a restart, which is the property that makes a search survive one.
pub async fn build<C, E>(
    state: &AppState,
    dag_id: &str,
    conn: Arc<C>,
    engine: Arc<E>,
) -> Result<Vec<Stepper<C, E>>, ServerError>
where
    C: Connector + Send + Sync + 'static,
    E: Executor<C> + Send + Sync + 'static,
{
    let registered = registrations::active_for_dag(&state.store, dag_id.to_string()).await?;
    let mut steppers = Vec::new();

    for row in registered {
        // A `Once` optimization is invoked explicitly, not around runs. It
        // being registered says what a DAG is under; it does not make every
        // run re-apply a rewrite that is already in the stored definition.
        if row.optimization_type() != OptimizationType::Continuous {
            continue;
        }
        let config = row.config.clone().unwrap_or_default();
        let Some(mut optimization) =
            registry::build::<C, E>(&row.name, conn.clone(), engine.clone(), &config)
        else {
            log::warn!(
                "'{}' is registered on this DAG but no such optimization exists; skipping",
                row.name
            );
            continue;
        };
        optimization.set_step_phase(row.step_phase());

        steppers.push(Stepper {
            name: row.name.clone(),
            step_phase: row.step_phase(),
            optimization,
            store: Arc::new(ScopedStore::new(state.store.clone(), &row.name)),
        });
    }

    Ok(steppers)
}

/// Whether any of these optimizations needs EXPLAIN ANALYZE plans.
///
/// HMP ranks candidate views by the operator CPU time their plans attribute to
/// them. A run that collected none leaves it ranking by node time instead --
/// workable, but coarser -- so the driver turns plan collection on when a
/// search that wants them is attached, rather than letting the quality of the
/// optimization depend on how the run happened to be triggered.
pub fn wants_plans<C, E>(steppers: &[Stepper<C, E>]) -> bool
where
    C: Connector + Send + Sync + 'static,
    E: Executor<C> + Send + Sync + 'static,
{
    steppers
        .iter()
        .any(|s| matches!(s.name.as_str(), "hmp" | "omp"))
}

/// Step every optimization whose phase includes `side`.
///
/// `dag` is the working copy for this one execution: a `Before` step may
/// rewrite it into a candidate, and that rewrite lives and dies with the run.
#[allow(clippy::too_many_arguments)]
pub async fn step_all<C, E>(
    steppers: &mut [Stepper<C, E>],
    conn: Arc<C>,
    engine: Arc<E>,
    dag: &mut Dag,
    dag_id: &str,
    dag_name: &str,
    dag_version: i32,
    side: StepPhase,
    run: Option<RunContext>,
) -> StepReport
where
    C: Connector + Send + Sync + 'static,
    E: Executor<C> + Send + Sync + 'static,
{
    let mut report = StepReport::default();

    for stepper in steppers.iter_mut() {
        if !stepper.step_phase.includes(side) {
            continue;
        }

        let mut ctx = StepContext {
            store: stepper.store.as_ref(),
            conn: conn.clone(),
            engine: engine.clone(),
            dag,
            dag_id,
            dag_name,
            dag_version,
            side,
            run: run.clone(),
        };

        // A step that fails must not fail the run. The DAG is the user's work;
        // an optimization is an opinion about it, and an opinion that errors
        // is one to log and set aside.
        match stepper.optimization.step(&mut ctx).await {
            Ok(StepOutcome::Idle) => {}
            Ok(StepOutcome::Trial {
                label,
                budget_ms,
                fallback,
                ..
            }) => {
                log::info!("{}: trying {label} on this run", stepper.name);
                report.trials.push(InstalledTrial {
                    name: stepper.name.clone(),
                    label,
                    budget_ms,
                    fallback,
                });
            }
            Ok(StepOutcome::Promote { .. }) | Ok(StepOutcome::Rewrote { .. }) => {
                log::info!("{}: converged; promoting its result", stepper.name);
                report.promoted.push((stepper.name.clone(), ctx.dag.clone()));
            }
            Ok(StepOutcome::Done { .. }) => {
                log::info!(
                    "{}: converged with nothing that beat the DAG as it stands",
                    stepper.name
                );
                report.finished.push(stepper.name.clone());
            }
            Err(e) => {
                log::warn!("{} failed to step: {e}", stepper.name);
            }
        }
    }

    report
}

/// A promotion becomes a new version of the DAG, attributed to the version it
/// came from -- the same lineage `dee optimize --save` records. Either way the
/// registration is marked finished, so a converged search stops being stepped
/// while its state and trial history stay readable.
/// Store what converged optimizations decided, returning the version any of
/// them promoted.
pub async fn apply(
    state: &AppState,
    dag_id: &str,
    dag_name: &str,
    source_version: i32,
    sources: &[dee::file::DagFileSource],
    report: StepReport,
) -> Option<i32> {
    let mut promoted_version = None;
    for (name, dag) in report.promoted {
        let mut definition = dee::file::DagFile::from(dag);
        // Resolved schemas are a cache of warehouse state, not part of the
        // definition. Letting them reach the stored DAG would make a version's
        // content hash depend on the warehouse, so an optimization that
        // changed nothing would still mint a version.
        definition.sources = sources.to_vec();

        let submitted = dags::submit(
            &state.store,
            dags::SubmitRequest {
                derived_from_version: Some(source_version),
                ..dags::SubmitRequest::new(
                    dag_name.to_string(),
                    definition,
                    dags::Origin::Optimized,
                )
            },
        )
        .await;

        match submitted {
            Ok(result) => {
                log::info!(
                    "{name} promoted {dag_name} v{source_version} -> v{}",
                    result.version
                );
                promoted_version = Some(result.version);
                if let Err(e) = registrations::mark_finished(
                    &state.store,
                    dag_id.to_string(),
                    name.clone(),
                    Some(result.version),
                )
                .await
                {
                    log::error!("could not mark {name} finished: {e}");
                }
            }
            Err(e) => log::error!("{name} converged but its result could not be stored: {e}"),
        }
    }

    for name in report.finished {
        if let Err(e) =
            registrations::mark_finished(&state.store, dag_id.to_string(), name.clone(), None).await
        {
            log::error!("could not mark {name} finished: {e}");
        }
    }

    promoted_version
}


// ---------------------------------------------------------------------------
// Registration
//
// An optimization builds its own state tables, so registering one means asking
// it to rather than declaring its schema somewhere else -- the pass is the only
// thing that knows what it needs to remember. Constructing one needs a
// connector, which is why registering resolves the DAG's target: it is also
// the point at which "this DAG has a reachable warehouse" is worth finding out,
// rather than on the first scheduled run at 3am.
// ---------------------------------------------------------------------------

use dee::executor::SimpleEngine;
use dee::opt::{OptimizerConfig, RegisterContext};

use crate::exec::connectors::ConnectorHandle;
use crate::store::repo::connections;

/// Which side of a registration to perform.
#[derive(Clone, Copy, PartialEq)]
pub enum Registering {
    Setup,
    Teardown,
}

async fn perform<C>(
    conn: Arc<C>,
    store: &dyn OptStore,
    name: &str,
    dag_id: &str,
    dag_name: &str,
    config: &OptimizerConfig,
    direction: Registering,
) -> Result<Vec<String>, ServerError>
where
    C: Connector + Send + Sync + 'static,
{
    let engine = Arc::new(
        SimpleEngine::new(Arc::clone(&conn))
            .map_err(|e| ServerError::Internal(format!("building the engine: {e}")))?,
    );
    let optimization = registry::build::<C, SimpleEngine<C>>(name, conn, engine, config)
        .ok_or_else(|| ServerError::BadRequest(format!("no optimization named '{name}'")))?;

    let ctx = RegisterContext {
        store,
        dag_id,
        dag_name,
    };
    let registration = match direction {
        Registering::Setup => optimization.register(&ctx).await,
        Registering::Teardown => optimization.deregister(&ctx).await,
    }
    .map_err(|e| ServerError::Internal(e.to_string()))?;

    // `None` means the optimization keeps no state -- Pushdown decides
    // everything from the DAG in front of it -- which is a real answer, not a
    // failure to set anything up.
    Ok(registration.map(|r| r.tables).unwrap_or_default())
}

/// Register or deregister `name` on a DAG, returning the tables it owns.
pub async fn registration(
    state: &AppState,
    store: &dyn OptStore,
    name: &str,
    dag_id: &str,
    dag_name: &str,
    target: &str,
    config: &OptimizerConfig,
    direction: Registering,
) -> Result<Vec<String>, ServerError> {
    let connection = connections::get(&state.store, target.to_string())
        .await?
        .ok_or_else(|| ServerError::NotFound("connection", target.to_string()))?;

    match state.connectors.acquire(&connection).await? {
        ConnectorHandle::DuckDb(conn) => {
            perform(conn, store, name, dag_id, dag_name, config, direction).await
        }
        ConnectorHandle::Postgres(conn) => {
            perform(conn, store, name, dag_id, dag_name, config, direction).await
        }
    }
}
