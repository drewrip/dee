use axum::Json;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};

use crate::api::dags::lookup;
use crate::error::ServerError;
use crate::sched::cron::CronSpec;
use crate::state::AppState;
use crate::store::repo::{connections, schedules};

#[derive(Deserialize)]
pub struct SetSchedule {
    pub cron: String,
    #[serde(default = "default_timezone")]
    pub timezone: String,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub target: Option<String>,
}

fn default_timezone() -> String {
    "UTC".to_string()
}
fn default_true() -> bool {
    true
}

#[derive(Serialize)]
pub struct ScheduleView {
    #[serde(flatten)]
    pub schedule: schedules::ScheduleRow,
    /// What the expression means, in words. Cron is famously easy to get
    /// subtly wrong, and this is the cheapest possible check on intent.
    pub description: String,
}

pub async fn set(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(body): Json<SetSchedule>,
) -> Result<Json<ScheduleView>, ServerError> {
    let dag = lookup(&state, &name).await?;

    // Validated here rather than at fire time, so a typo is an error the
    // author sees now instead of a schedule that silently never runs.
    let spec = CronSpec::parse(&body.cron, &body.timezone)
        .map_err(|e| ServerError::BadRequest(e.to_string()))?;

    if let Some(target) = &body.target {
        if connections::get(&state.store, target.clone()).await?.is_none() {
            return Err(ServerError::BadRequest(format!(
                "no connection named '{target}'"
            )));
        }
    }
    if body.target.is_none() && dag.default_target.is_none() {
        return Err(ServerError::BadRequest(format!(
            "'{name}' has no target, so a schedule would have nothing to run against; \
             set one on the dag or pass one here"
        )));
    }

    let next_fire_at = if body.enabled {
        spec.next_after(chrono::Utc::now())
    } else {
        None
    };

    schedules::upsert(
        &state.store,
        dag.dag_id.clone(),
        body.cron,
        body.timezone,
        body.enabled,
        body.target,
        next_fire_at,
    )
    .await?;

    view(&state, &dag.dag_id, &name).await.map(Json)
}

pub async fn get(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<ScheduleView>, ServerError> {
    let dag = lookup(&state, &name).await?;
    view(&state, &dag.dag_id, &name).await.map(Json)
}

pub async fn list(
    State(state): State<AppState>,
) -> Result<Json<Vec<schedules::ScheduleRow>>, ServerError> {
    Ok(Json(schedules::list(&state.store).await?))
}

pub async fn delete(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<StatusCode, ServerError> {
    let dag = lookup(&state, &name).await?;
    if !schedules::delete(&state.store, dag.dag_id).await? {
        return Err(ServerError::NotFound("schedule", name));
    }
    Ok(StatusCode::NO_CONTENT)
}

pub async fn pause(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<ScheduleView>, ServerError> {
    set_enabled(state, name, false).await
}

pub async fn resume(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<ScheduleView>, ServerError> {
    set_enabled(state, name, true).await
}

async fn set_enabled(
    state: AppState,
    name: String,
    enabled: bool,
) -> Result<Json<ScheduleView>, ServerError> {
    let dag = lookup(&state, &name).await?;
    let schedule = schedules::get(&state.store, dag.dag_id.clone())
        .await?
        .ok_or_else(|| ServerError::NotFound("schedule", name.clone()))?;

    // Resuming computes the next firing from now, so a schedule paused for a
    // week does not come back owing a week of runs.
    let next_fire_at = if enabled {
        CronSpec::parse(&schedule.cron, &schedule.timezone)
            .map_err(|e| ServerError::Internal(e.to_string()))?
            .next_after(chrono::Utc::now())
    } else {
        None
    };

    schedules::set_enabled(&state.store, dag.dag_id.clone(), enabled, next_fire_at).await?;
    view(&state, &dag.dag_id, &name).await.map(Json)
}

#[derive(Deserialize, Default)]
pub struct SkipQuery {
    #[serde(default)]
    pub limit: Option<usize>,
}

pub async fn skips(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Query(query): Query<SkipQuery>,
) -> Result<Json<Vec<schedules::SkipRow>>, ServerError> {
    lookup(&state, &name).await?;
    Ok(Json(
        schedules::skips(
            &state.store,
            Some(name),
            query.limit.unwrap_or(50).clamp(1, 10_000),
        )
        .await?,
    ))
}

async fn view(
    state: &AppState,
    dag_id: &str,
    name: &str,
) -> Result<ScheduleView, ServerError> {
    let schedule = schedules::get(&state.store, dag_id.to_string())
        .await?
        .ok_or_else(|| ServerError::NotFound("schedule", name.to_string()))?;
    let description = CronSpec::parse(&schedule.cron, &schedule.timezone)
        .map(|_| describe(&schedule.cron, &schedule.timezone))
        .unwrap_or_else(|e| e.to_string());
    Ok(ScheduleView {
        schedule,
        description,
    })
}

fn describe(expr: &str, timezone: &str) -> String {
    use croner::Cron;
    use std::str::FromStr;
    match Cron::from_str(expr) {
        Ok(cron) => format!("{} ({timezone})", cron.describe()),
        Err(_) => format!("{expr} ({timezone})"),
    }
}
