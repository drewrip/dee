use axum::Json;
use axum::extract::{Path, Query, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use chrono::Utc;
use dee::file::DagFile;
use dee::opt::OptimizerConfig;
use dee::opt::report::OptimizeReport;
use serde::{Deserialize, Serialize};

use crate::api::dags::lookup;
use crate::api::runs::WaitQuery;
use crate::error::ServerError;
use crate::exec::optimize::{self, OptimizeJob};
use crate::state::AppState;
use crate::store::repo::{dags, optimizations, runs};

#[derive(Deserialize, Default)]
pub struct OptimizeBody {
    #[serde(default)]
    pub version: Option<i32>,
    #[serde(default)]
    pub target: Option<String>,
    /// Partial configs are accepted; absent fields take the same defaults the
    /// CLI would have used.
    #[serde(default)]
    pub config: OptimizerConfig,
    /// Store the rewrite as a new version of this DAG.
    #[serde(default)]
    pub save_as_version: bool,
    #[serde(default)]
    pub explain: bool,
}

#[derive(Serialize)]
pub struct OptimizeAccepted {
    pub optimization_id: String,
    pub dag: String,
    pub source_version: i32,
    pub target: String,
    pub status: String,
}

pub async fn start(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Query(query): Query<WaitQuery>,
    body: Option<Json<OptimizeBody>>,
) -> Result<(StatusCode, Json<OptimizeAccepted>), ServerError> {
    let body = body.map(|Json(b)| b).unwrap_or_default();
    let dag = lookup(&state, &name).await?;
    let source_version = body.version.unwrap_or(dag.current_version);

    if dags::definition(&state.store, dag.dag_id.clone(), source_version)
        .await?
        .is_none()
    {
        return Err(ServerError::NotFound(
            "dag version",
            format!("{name} v{source_version}"),
        ));
    }

    let target = body
        .target
        .or_else(|| dag.default_target.clone())
        .ok_or_else(|| {
            ServerError::BadRequest(format!("'{name}' has no target; pass one with the request"))
        })?;

    let mut config = body.config;
    reject_server_side_paths(&config)?;
    // The explain sections are only collected when the optimizer is told to,
    // and asking for the HTML afterwards is too late.
    config.explain = body.explain;

    let optimization_id = optimizations::create(
        &state.store,
        dag.dag_id.clone(),
        source_version,
        target.clone(),
        &config,
        state.instance_id.clone(),
    )
    .await?;

    if let Some(blocking) = state.runs.claim(&dag.dag_id, &optimization_id).await {
        optimizations::record_failure(
            &state.store,
            optimization_id,
            runs::status::SKIPPED,
            format!("another job for '{name}' was already running"),
        )
        .await?;
        return Err(ServerError::Conflict(format!(
            "'{name}' already has an active job ({blocking}); \
             an optimization runs the DAG, so it cannot share the warehouse"
        )));
    }

    let job = OptimizeJob {
        optimization_id: optimization_id.clone(),
        dag_id: dag.dag_id.clone(),
        dag_name: name.clone(),
        source_version,
        target: target.clone(),
        config,
        save_as_version: body.save_as_version,
        explain: body.explain,
    };
    tokio::spawn(optimize::drive(state.clone(), job));

    let status = if query.wait {
        wait_for(&state, &optimization_id, query.timeout_s.unwrap_or(7200)).await?
    } else {
        runs::status::RUNNING.to_string()
    };

    Ok((
        StatusCode::ACCEPTED,
        Json(OptimizeAccepted {
            optimization_id,
            dag: name,
            source_version,
            target,
            status,
        }),
    ))
}

async fn wait_for(
    state: &AppState,
    optimization_id: &str,
    timeout_s: u64,
) -> Result<String, ServerError> {
    let deadline = Utc::now() + chrono::Duration::seconds(timeout_s as i64);
    let mut interval = std::time::Duration::from_millis(50);
    loop {
        let row = optimizations::get(&state.store, optimization_id.to_string())
            .await?
            .ok_or_else(|| ServerError::NotFound("optimization", optimization_id.to_string()))?;
        if runs::status::is_terminal(&row.status) {
            return Ok(row.status);
        }
        if Utc::now() > deadline {
            return Err(ServerError::BadRequest(format!(
                "optimization {optimization_id} did not finish within {timeout_s}s; \
                 it is still running"
            )));
        }
        tokio::time::sleep(interval).await;
        interval = (interval * 2).min(std::time::Duration::from_millis(1000));
    }
}

#[derive(Deserialize, Default)]
pub struct ListQuery {
    #[serde(default)]
    pub dag: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
}

pub async fn list(
    State(state): State<AppState>,
    Query(query): Query<ListQuery>,
) -> Result<Json<Vec<optimizations::OptimizationRow>>, ServerError> {
    Ok(Json(
        optimizations::list(
            &state.store,
            query.dag,
            query.limit.unwrap_or(50).clamp(1, 10_000),
        )
        .await?,
    ))
}

#[derive(Serialize)]
pub struct OptimizationDetail {
    #[serde(flatten)]
    pub optimization: optimizations::OptimizationRow,
    pub passes: Vec<optimizations::PassRow>,
    pub iterations: Vec<optimizations::IterationRow>,
}

pub async fn get(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<OptimizationDetail>, ServerError> {
    let optimization = fetch(&state, &id).await?;
    Ok(Json(OptimizationDetail {
        optimization,
        passes: optimizations::passes(&state.store, id.clone()).await?,
        iterations: optimizations::iterations(&state.store, id).await?,
    }))
}

/// The `OptimizeReport` exactly as the library produced it -- the same bytes
/// the old `dee-cli opt --report-json` wrote.
pub async fn report(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<OptimizeReport>, ServerError> {
    fetch(&state, &id).await?;
    optimizations::report(&state.store, id.clone())
        .await?
        .map(Json)
        .ok_or_else(|| {
            ServerError::NotFound("optimize report", format!("{id} (it may not have finished)"))
        })
}

pub async fn explain(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Response, ServerError> {
    fetch(&state, &id).await?;
    let html = optimizations::explain_html(&state.store, id.clone())
        .await?
        .ok_or_else(|| {
            ServerError::NotFound(
                "explain report",
                format!("{id}; request one by optimizing with explain enabled"),
            )
        })?;
    Ok((
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        html,
    )
        .into_response())
}

/// The DAG this optimization produced, for a caller that wants the rewrite
/// without having saved it as a version.
pub async fn result_dag(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<DagFile>, ServerError> {
    let optimization = fetch(&state, &id).await?;
    let version = optimization.result_version.ok_or_else(|| {
        ServerError::NotFound(
            "optimized dag",
            format!("{id}; it was not saved as a version"),
        )
    })?;
    dags::definition(&state.store, optimization.dag_id, version)
        .await?
        .map(Json)
        .ok_or_else(|| ServerError::NotFound("dag version", format!("{id} result")))
}

async fn fetch(
    state: &AppState,
    id: &str,
) -> Result<optimizations::OptimizationRow, ServerError> {
    optimizations::get(&state.store, id.to_string())
        .await?
        .ok_or_else(|| ServerError::NotFound("optimization", id.to_string()))
}

/// `hmp_show_operators` and `hmp_show_nodes` name files the optimizer writes.
///
/// In a daemon those become arbitrary filesystem writes chosen by whoever can
/// reach the API. The empty string keeps the diagnostic (it logs the table),
/// so the useful half of the option survives.
fn reject_server_side_paths(config: &OptimizerConfig) -> Result<(), ServerError> {
    for (name, value) in [
        ("hmp_show_operators", &config.hmp_show_operators),
        ("hmp_show_nodes", &config.hmp_show_nodes),
    ] {
        if let Some(path) = value {
            if !path.is_empty() {
                return Err(ServerError::BadRequest(format!(
                    "{name} cannot name a file on the server; pass \"\" to log the table instead"
                )));
            }
        }
    }
    Ok(())
}
