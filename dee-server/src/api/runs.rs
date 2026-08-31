use axum::Json;
use axum::extract::{Path, Query, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use chrono::Utc;
use dee::dag::Dag;
use dee::executor::{ExecStats, NodeStats};
use dee::profile::{
    DagRunProfile, ProfileReport, build_dag_run_profile, render_profile_html, SystemUsageSample,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::api::dags::lookup;
use crate::error::ServerError;
use crate::exec::driver;
use crate::state::AppState;
use crate::store::repo::{dags, runs};

#[derive(Deserialize, Default)]
pub struct TriggerBody {
    #[serde(default)]
    pub version: Option<i32>,
    #[serde(default)]
    pub target: Option<String>,
    #[serde(default)]
    pub warmups: Option<i32>,
    #[serde(default)]
    pub repetitions: Option<i32>,
    /// Drop every relation the DAG defines before each repetition. True by
    /// default, which is what makes repeated measurements comparable.
    #[serde(default)]
    pub cleanup_before: Option<bool>,
    #[serde(default)]
    pub collect_plans: Option<bool>,
    #[serde(default)]
    pub sample_interval_ms: Option<i32>,
}

#[derive(Deserialize, Default)]
pub struct WaitQuery {
    #[serde(default)]
    pub wait: bool,
    #[serde(default)]
    pub timeout_s: Option<u64>,
}

#[derive(Serialize)]
pub struct TriggerResult {
    pub run_group_id: String,
    pub run_ids: Vec<String>,
    pub dag: String,
    pub version: i32,
    pub target: String,
    pub status: String,
}

pub async fn trigger(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Query(query): Query<WaitQuery>,
    body: Option<Json<TriggerBody>>,
) -> Result<(StatusCode, Json<TriggerResult>), ServerError> {
    let body = body.map(|Json(b)| b).unwrap_or_default();
    let dag = lookup(&state, &name).await?;
    let version = body.version.unwrap_or(dag.current_version);

    if dags::definition(&state.store, dag.dag_id.clone(), version)
        .await?
        .is_none()
    {
        return Err(ServerError::NotFound(
            "dag version",
            format!("{name} v{version}"),
        ));
    }

    let target = body
        .target
        .or_else(|| dag.default_target.clone())
        .ok_or_else(|| {
            ServerError::BadRequest(format!(
                "'{name}' has no target; set one with `dee dag submit --target` or pass one here"
            ))
        })?;

    // Claim before creating rows so a rejected trigger leaves no trace beyond
    // the conflict it reports.
    let request = runs::RunRequest {
        dag_id: dag.dag_id.clone(),
        dag_version: version,
        target: target.clone(),
        trigger: "manual".into(),
        scheduled_for: None,
        warmups: body.warmups.unwrap_or(0),
        repetitions: body.repetitions.unwrap_or(1),
        cleanup_before: body.cleanup_before.unwrap_or(true),
        collect_plans: body.collect_plans.unwrap_or(false),
        sample_interval_ms: body.sample_interval_ms,
    };

    let created =
        runs::create_group(&state.store, request, state.instance_id.clone()).await?;

    if let Some(blocking) = state
        .runs
        .claim(&dag.dag_id, &created.run_group_id)
        .await
    {
        // Losing the claim means another job for this DAG is already running.
        // Record the collision the same way the scheduler does, so history
        // shows a refused trigger rather than a gap.
        runs::finalize_group(
            &state.store,
            created.run_group_id.clone(),
            Some(format!("another job for '{name}' is already running")),
        )
        .await?;
        return Err(ServerError::Conflict(format!(
            "'{name}' already has an active job ({blocking}); \
             wait for it or cancel it first"
        )));
    }

    let group_id = created.run_group_id.clone();
    tokio::spawn(driver::drive_group(state.clone(), group_id.clone()));

    let status = if query.wait {
        wait_for_group(&state, &group_id, query.timeout_s.unwrap_or(3600)).await?
    } else {
        runs::status::QUEUED.to_string()
    };

    Ok((
        StatusCode::ACCEPTED,
        Json(TriggerResult {
            run_group_id: created.run_group_id,
            run_ids: created.run_ids,
            dag: name,
            version,
            target,
            status,
        }),
    ))
}

/// Poll until the group reaches a terminal state.
///
/// Long-polling rather than streaming: a client that just wants the result
/// should not have to hold a connection open interpreting events.
async fn wait_for_group(
    state: &AppState,
    group_id: &str,
    timeout_s: u64,
) -> Result<String, ServerError> {
    let deadline = Utc::now() + chrono::Duration::seconds(timeout_s as i64);
    let mut interval = std::time::Duration::from_millis(25);

    loop {
        let group = runs::get_group(&state.store, group_id.to_string())
            .await?
            .ok_or_else(|| ServerError::NotFound("run group", group_id.to_string()))?;
        if runs::status::is_terminal(&group.status) {
            return Ok(group.status);
        }
        if Utc::now() > deadline {
            return Err(ServerError::BadRequest(format!(
                "run group {group_id} did not finish within {timeout_s}s; it is still running"
            )));
        }
        tokio::time::sleep(interval).await;
        // Back off so a long DAG is not polled hundreds of times a second.
        interval = (interval * 2).min(std::time::Duration::from_millis(500));
    }
}

#[derive(Deserialize, Default)]
pub struct ListQuery {
    #[serde(default)]
    pub dag: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub group: Option<String>,
    #[serde(default)]
    pub phase: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
}

pub async fn list(
    State(state): State<AppState>,
    Query(query): Query<ListQuery>,
) -> Result<Json<Vec<runs::RunRow>>, ServerError> {
    let filter = runs::RunFilter {
        dag_name: query.dag,
        status: query.status,
        run_group_id: query.group,
        phase: query.phase,
        limit: query.limit.unwrap_or(50).clamp(1, 10_000),
    };
    Ok(Json(runs::list_runs(&state.store, filter).await?))
}

pub async fn get(
    State(state): State<AppState>,
    Path(run_id): Path<String>,
) -> Result<Json<runs::RunRow>, ServerError> {
    Ok(Json(fetch_run(&state, &run_id).await?))
}

pub async fn nodes(
    State(state): State<AppState>,
    Path(run_id): Path<String>,
) -> Result<Json<Vec<runs::NodeExecutionRow>>, ServerError> {
    fetch_run(&state, &run_id).await?;
    Ok(Json(runs::node_executions(&state.store, run_id).await?))
}

pub async fn plans(
    State(state): State<AppState>,
    Path(run_id): Path<String>,
) -> Result<Json<Vec<runs::PlanRow>>, ServerError> {
    fetch_run(&state, &run_id).await?;
    Ok(Json(runs::plans(&state.store, run_id).await?))
}

pub async fn samples(
    State(state): State<AppState>,
    Path(run_id): Path<String>,
) -> Result<Json<Vec<runs::SampleRow>>, ServerError> {
    fetch_run(&state, &run_id).await?;
    Ok(Json(runs::samples(&state.store, run_id).await?))
}

pub async fn events(
    State(state): State<AppState>,
    Path(run_id): Path<String>,
) -> Result<Json<Vec<runs::EventRow>>, ServerError> {
    Ok(Json(runs::events_for_run(&state.store, run_id).await?))
}

pub async fn cancel(
    State(state): State<AppState>,
    Path(run_id): Path<String>,
) -> Result<(StatusCode, Json<serde_json::Value>), ServerError> {
    let run = fetch_run(&state, &run_id).await?;
    cancel_group(&state, &run.run_group_id).await
}

pub async fn cancel_group_route(
    State(state): State<AppState>,
    Path(group_id): Path<String>,
) -> Result<(StatusCode, Json<serde_json::Value>), ServerError> {
    cancel_group(&state, &group_id).await
}

async fn cancel_group(
    state: &AppState,
    group_id: &str,
) -> Result<(StatusCode, Json<serde_json::Value>), ServerError> {
    let group = runs::get_group(&state.store, group_id.to_string())
        .await?
        .ok_or_else(|| ServerError::NotFound("run group", group_id.to_string()))?;

    if runs::status::is_terminal(&group.status) {
        return Err(ServerError::Conflict(format!(
            "run group {group_id} already finished ({})",
            group.status
        )));
    }

    let signalled = state.runs.cancel(group_id).await;
    // 202, not 200: the engine only checks its cancel flag between node
    // dispatches, so a long-running node keeps going until it returns.
    Ok((
        StatusCode::ACCEPTED,
        Json(serde_json::json!({
            "run_group_id": group_id,
            "requested": signalled,
            "detail": if signalled {
                "cancellation requested; the current node finishes first"
            } else {
                "this group is not running on this server"
            }
        })),
    ))
}

pub async fn group(
    State(state): State<AppState>,
    Path(group_id): Path<String>,
) -> Result<Json<GroupDetail>, ServerError> {
    let group = runs::get_group(&state.store, group_id.clone())
        .await?
        .ok_or_else(|| ServerError::NotFound("run group", group_id.clone()))?;
    let series = runs::runs_in_group(&state.store, group_id).await?;
    Ok(Json(GroupDetail {
        group,
        runs: series,
    }))
}

#[derive(Serialize)]
pub struct GroupDetail {
    #[serde(flatten)]
    pub group: runs::RunGroupRow,
    pub runs: Vec<runs::RunRow>,
}

/// Rebuild the `ProfileReport` for a whole group.
///
/// This is deliberately the exact type the old `dee-cli run --report-json`
/// wrote, built by the library's own `build_dag_run_profile`, so any consumer
/// of that file works unchanged against the server.
pub async fn group_report(
    State(state): State<AppState>,
    Path(group_id): Path<String>,
) -> Result<Json<ProfileReport>, ServerError> {
    Ok(Json(build_group_report(&state, &group_id).await?))
}

pub async fn run_report(
    State(state): State<AppState>,
    Path(run_id): Path<String>,
) -> Result<Json<ProfileReport>, ServerError> {
    let run = fetch_run(&state, &run_id).await?;
    let mut report = build_group_report(&state, &run.run_group_id).await?;
    report.runs.retain(|r| r.rep_index as i32 == run.rep_index
        && r.phase == run.phase);
    Ok(Json(report))
}

pub async fn group_report_html(
    State(state): State<AppState>,
    Path(group_id): Path<String>,
) -> Result<Response, ServerError> {
    let report = build_group_report(&state, &group_id).await?;
    let html = render_profile_html(&report)
        .map_err(|e| ServerError::Internal(format!("rendering the profile: {e}")))?;
    Ok((
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        html,
    )
        .into_response())
}

async fn build_group_report(
    state: &AppState,
    group_id: &str,
) -> Result<ProfileReport, ServerError> {
    let group = runs::get_group(&state.store, group_id.to_string())
        .await?
        .ok_or_else(|| ServerError::NotFound("run group", group_id.to_string()))?;
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

    let series = runs::runs_in_group(&state.store, group_id.to_string()).await?;
    let mut profiles: Vec<DagRunProfile> = Vec::new();

    for run in series {
        // Only completed runs have anything to report.
        if run.status != runs::status::SUCCEEDED {
            continue;
        }
        let nodes = runs::node_executions(&state.store, run.run_id.clone()).await?;
        let plans = runs::plans(&state.store, run.run_id.clone()).await?;
        let samples = runs::samples(&state.store, run.run_id.clone()).await?;

        let plan_by_node: HashMap<String, String> = plans
            .into_iter()
            .map(|p| (p.node_id, p.plan_json))
            .collect();

        let node_stats: HashMap<String, NodeStats> = nodes
            .into_iter()
            .map(|n| {
                let stats = NodeStats {
                    start: n.started_at,
                    finish: n.finished_at,
                    duration: chrono::TimeDelta::milliseconds(n.duration_ms),
                    plan: plan_by_node.get(&n.node_id).cloned(),
                    rows_produced: n.rows_produced.map(|r| r as u64),
                };
                (n.node_id, stats)
            })
            .collect();

        let start = run.started_at.unwrap_or(run.queued_at);
        let finish = run.finished_at.unwrap_or(start);
        let stats = ExecStats {
            start,
            finish,
            duration: chrono::TimeDelta::milliseconds(run.duration_ms.unwrap_or(0)),
            node_stats,
            system_samples: samples
                .into_iter()
                .map(|s| SystemUsageSample {
                    timestamp: s.timestamp,
                    elapsed_ms: s.elapsed_ms,
                    cpu_percent: s.cpu_percent,
                    memory_bytes: s.memory_bytes.map(|v| v as u64),
                    disk_bytes: s.disk_bytes.map(|v| v as u64),
                    read_bytes: s.read_bytes.map(|v| v as u64),
                    written_bytes: s.written_bytes.map(|v| v as u64),
                })
                .collect(),
        };

        // The old CLI put the DAG's file path here; there is no file now, so
        // the DAG's name and version identify it instead.
        let label = format!("{}@v{}", group.dag_name, group.dag_version);
        let mut profile = build_dag_run_profile(&label, &dag, &stats);
        profile.phase = run.phase.clone();
        profile.rep_index = run.rep_index as usize;
        profiles.push(profile);
    }

    Ok(ProfileReport {
        generated_at: Utc::now(),
        runs: profiles,
    })
}

async fn fetch_run(state: &AppState, run_id: &str) -> Result<runs::RunRow, ServerError> {
    runs::get_run(&state.store, run_id.to_string())
        .await?
        .ok_or_else(|| ServerError::NotFound("run", run_id.to_string()))
}
