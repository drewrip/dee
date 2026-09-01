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
    /// Overrides for this optimization only, layered over the DAG's stored
    /// configuration. Only the keys present here are overridden, which is what
    /// lets `dee optimize pipeline --omp-top 5` change one parameter without
    /// silently reverting the rest to defaults. Absent entirely means "use the
    /// DAG's configuration as submitted".
    #[serde(default)]
    pub config: Option<serde_json::Value>,
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
    /// The configuration this optimization actually ran under, after the
    /// DAG's settings and the request's overrides were resolved. Echoed back
    /// because with two sources for it, "which settings did this use" must not
    /// be something a caller has to reconstruct.
    pub config: OptimizerConfig,
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

    let mut config = resolve_config(dag.optimizer_config.clone(), body.config)?;
    crate::api::reject_server_side_paths(&config)?;
    // The explain sections are only collected when the optimizer is told to,
    // and asking for the HTML afterwards is too late.
    config.explain = body.explain;

    // As in `trigger`: the in-memory claim below cannot see entries still
    // waiting in the queue, and an optimization runs the DAG, so it must not
    // start on top of a queue that is mid-drain.
    if let Some(blocking) = runs::active_job(&state.store, dag.dag_id.clone()).await? {
        return Err(ServerError::Conflict(format!(
            "'{name}' already has an active job ({blocking}); an optimization runs the DAG, \
             so it cannot share the warehouse with a run or a queued one"
        )));
    }

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
        config: config.clone(),
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
            config,
        }),
    ))
}

/// Layer a request's overrides onto the DAG's stored configuration.
///
/// A shallow merge is the whole of it, because `OptimizerConfig` is flat: every
/// field is a scalar the caller either named or did not. Deserializing the
/// merged object rather than patching a struct is what keeps `deny_unknown_fields`
/// doing its job -- a misspelled `omp_topp` is rejected here instead of being
/// dropped and quietly running under the wrong settings.
pub(crate) fn resolve_config(
    stored: Option<OptimizerConfig>,
    overrides: Option<serde_json::Value>,
) -> Result<OptimizerConfig, ServerError> {
    let base = stored.unwrap_or_default();
    let Some(overrides) = overrides else {
        return Ok(base);
    };
    let overrides = overrides.as_object().cloned().ok_or_else(|| {
        ServerError::BadRequest("config must be a json object of optimizer settings".into())
    })?;

    let mut merged = serde_json::to_value(&base)
        .map_err(|e| ServerError::Internal(format!("serializing the optimizer config: {e}")))?;
    let object = merged
        .as_object_mut()
        .expect("OptimizerConfig serializes as an object");
    for (key, value) in overrides {
        object.insert(key, value);
    }

    serde_json::from_value(merged)
        .map_err(|e| ServerError::BadRequest(format!("invalid optimizer config: {e}")))
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn stored() -> OptimizerConfig {
        OptimizerConfig {
            run_omp_pass: true,
            run_hmp_pass: false,
            run_pushdown_pass: true,
            omp_top: Some(5),
            hmp_max_runs: 4,
            ..OptimizerConfig::default()
        }
    }

    #[test]
    fn test_no_overrides_runs_the_dags_own_settings() {
        // The point of storing a config with the DAG: `dee optimize pipeline`
        // sends nothing and still benchmarks the pass it was submitted for.
        let resolved = resolve_config(Some(stored()), None).unwrap();
        assert!(resolved.run_omp_pass);
        assert!(!resolved.run_hmp_pass);
        assert_eq!(resolved.omp_top, Some(5));
        assert_eq!(resolved.hmp_max_runs, 4);
    }

    #[test]
    fn test_an_override_changes_one_field_and_leaves_the_rest() {
        let resolved =
            resolve_config(Some(stored()), Some(json!({"omp_top": 9}))).unwrap();
        assert_eq!(resolved.omp_top, Some(9));
        // Everything the request did not name still comes from the DAG. A
        // merge that reverted these to defaults would silently re-enable HMP
        // and turn an OMP benchmark into something else.
        assert!(!resolved.run_hmp_pass);
        assert_eq!(resolved.hmp_max_runs, 4);
    }

    #[test]
    fn test_a_dag_with_no_settings_falls_back_to_dees_defaults() {
        let resolved = resolve_config(None, Some(json!({"omp_top": 2}))).unwrap();
        assert_eq!(resolved.omp_top, Some(2));
        assert_eq!(
            resolved.hmp_max_runs,
            OptimizerConfig::default().hmp_max_runs
        );
    }

    #[test]
    fn test_a_null_override_is_a_value_not_an_absence() {
        // `--omp-top` has no "unset" spelling of its own, so a request that
        // sends null must clear the stored value rather than inherit it.
        let resolved =
            resolve_config(Some(stored()), Some(json!({"omp_top": null}))).unwrap();
        assert_eq!(resolved.omp_top, None);
    }

    #[test]
    fn test_a_misspelled_setting_is_rejected_rather_than_ignored() {
        // Silently dropping it would run the benchmark under settings nobody
        // chose, and the result would look valid.
        let error = resolve_config(Some(stored()), Some(json!({"omp_topp": 9})))
            .expect_err("unknown fields must not be accepted");
        assert!(
            error.to_string().contains("omp_topp"),
            "the error should name the offending field: {error}"
        );
    }

    #[test]
    fn test_a_config_that_is_not_an_object_is_a_bad_request() {
        let error = resolve_config(None, Some(json!("omp"))).expect_err("not an object");
        assert!(matches!(error, ServerError::BadRequest(_)));
    }
}
