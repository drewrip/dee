//! Running the optimizer against a registered DAG.
//!
//! Structurally the same as `driver.rs`: claim the DAG, do the work against a
//! cached connector, record what happened. An optimization holds the same
//! per-DAG claim as a run because it *is* a series of runs -- HMP and OMP
//! execute candidate DAGs against the live warehouse, so an optimization and a
//! scheduled run would otherwise fight over the same relation names.

use std::sync::Arc;

use dee::dag::Dag;
use dee::executor::{Executor, SimpleEngine};
use dee::file::DagFile;
use dee::opt::explain::render_explain_html;
use dee::opt::{Optimizer, OptimizerConfig};

use crate::error::ServerError;
use crate::exec::connectors::ConnectorHandle;
use crate::state::AppState;
use crate::store::repo::{connections, dags, optimizations, runs};

pub struct OptimizeJob {
    pub optimization_id: String,
    pub dag_id: String,
    pub dag_name: String,
    pub source_version: i32,
    pub target: String,
    pub config: OptimizerConfig,
    /// Store the rewritten DAG as a new version of the same DAG.
    pub save_as_version: bool,
    pub explain: bool,
}

/// Nobody awaits this, so failures are recorded rather than returned.
pub async fn drive(state: AppState, job: OptimizeJob) {
    let optimization_id = job.optimization_id.clone();
    let dag_id = job.dag_id.clone();

    match drive_inner(state.clone(), job).await {
        Ok(()) => {}
        Err(e) => {
            let message = e.to_string();
            log::error!("optimization {optimization_id} failed: {message}");
            if let Err(e) = optimizations::record_failure(
                &state.store,
                optimization_id.clone(),
                runs::status::FAILED,
                message,
            )
            .await
            {
                log::error!("could not record the failure of {optimization_id}: {e}");
            }
        }
    }

    let _ = dag_id;
    state.runs.finish(&optimization_id).await;
    // An optimization holds the same per-DAG claim as a run, so finishing one
    // can unblock a queue -- and the entries behind it pick up the rewrite.
    state.wake_queue();
}

async fn drive_inner(state: AppState, job: OptimizeJob) -> Result<(), ServerError> {
    let definition = dags::definition(&state.store, job.dag_id.clone(), job.source_version)
        .await?
        .ok_or_else(|| {
            ServerError::NotFound(
                "dag version",
                format!("{} v{}", job.dag_name, job.source_version),
            )
        })?;
    let dag = Dag::try_from(definition.clone())
        .map_err(|e| ServerError::Internal(format!("stored dag no longer parses: {e}")))?;

    let connection = connections::get(&state.store, job.target.clone())
        .await?
        .ok_or_else(|| ServerError::NotFound("connection", job.target.clone()))?;

    let (report, explain_sections, mut optimized) =
        match state.connectors.acquire(&connection).await? {
            ConnectorHandle::DuckDb(conn) => optimize_with(conn, dag, job.config.clone()).await?,
            ConnectorHandle::Postgres(conn) => optimize_with(conn, dag, job.config.clone()).await?,
        };

    // The optimizer calls `resolve_schemas`, which fills in `sources[].columns`
    // by querying the warehouse. Those are a cache of warehouse state, not part
    // of the definition, and letting them reach the stored DAG would make a
    // version's content hash depend on the warehouse -- the same file would
    // hash differently before and after resolution, and an optimization that
    // changed nothing would still mint a version. Keep what the author wrote.
    optimized.sources = definition.sources.clone();

    // Save before recording the report, so `result_version` can point at it.
    let result_version = if job.save_as_version {
        let submitted = dags::submit(
            &state.store,
            dags::SubmitRequest {
                derived_from_version: Some(job.source_version),
                optimization_id: Some(job.optimization_id.clone()),
                ..dags::SubmitRequest::new(
                    job.dag_name.clone(),
                    optimized,
                    dags::Origin::Optimized,
                )
            },
        )
        .await?;
        Some(submitted.version)
    } else {
        None
    };

    let explain_html = if job.explain && !explain_sections.is_empty() {
        Some(render_explain_html(&explain_sections))
    } else {
        None
    };

    optimizations::record_success(
        &state.store,
        job.optimization_id,
        report,
        result_version,
        explain_html,
    )
    .await?;
    Ok(())
}

async fn optimize_with<C>(
    conn: Arc<C>,
    mut dag: Dag,
    config: OptimizerConfig,
) -> Result<
    (
        dee::opt::report::OptimizeReport,
        Vec<(String, String)>,
        DagFile,
    ),
    ServerError,
>
where
    C: dee::connectors::Connector + Send + Sync + 'static,
{
    let engine = Arc::new(
        SimpleEngine::new(Arc::clone(&conn))
            .map_err(|e| ServerError::Internal(format!("building the engine: {e}")))?,
    );

    // The optimizer measures candidate DAGs by running them, so it has to
    // start from a warehouse without this DAG's relations already in it --
    // exactly what `dee-cli opt` did before its first pass.
    engine
        .cleanup(&dag)
        .await
        .map_err(|e| ServerError::Internal(format!("preparing the warehouse: {e}")))?;

    let mut optimizer =
        Optimizer::new_with_config(conn, engine, config).stats_on_passes(true);
    let report = optimizer
        .run(&mut dag)
        .await
        .map_err(|e| ServerError::Internal(format!("optimizing: {e}")))?;

    let sections = optimizer.explain_sections().to_vec();
    Ok((report, sections, DagFile::from(dag)))
}
