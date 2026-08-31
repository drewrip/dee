//! `POST /v1/dags/{name}/queue` and the endpoints for looking at what it made.
//!
//! Enqueueing N entries is the benchmark primitive: N run groups for one DAG,
//! executed strictly one after another because a DAG can only have one job in
//! flight. It is not the same thing as one group with N repetitions -- a group
//! pins one version and runs under one driver against one engine, so nothing
//! that happens between its repetitions can change what it runs. Separate
//! entries can, which is what makes a queue the right shape for watching a DAG
//! adapt across its own history.

use axum::Json;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};

use crate::api::dags::lookup;
use crate::api::runs::{TriggerBody, WaitQuery, wait_for_group};
use crate::error::ServerError;
use crate::state::AppState;
use crate::store::repo::{dags, runs};

/// An upper bound on one request, not on the queue. It exists so a typo in a
/// benchmark script cannot write a million rows before anyone notices.
const MAX_COUNT: i32 = 10_000;

#[derive(Deserialize, Default)]
pub struct EnqueueBody {
    /// How many entries to add. Each is a separate run group and runs on its
    /// own turn; use `repetitions` for repeats inside one group.
    #[serde(default)]
    pub count: Option<i32>,
    #[serde(flatten)]
    pub run: TriggerBody,
}

#[derive(Serialize)]
pub struct EnqueuedEntry {
    pub run_group_id: String,
    /// 1-based place in this DAG's queue at the moment of enqueueing.
    pub position: i32,
    pub run_ids: Vec<String>,
}

#[derive(Serialize)]
pub struct EnqueueResult {
    pub dag: String,
    pub version: i32,
    /// False when the version was left to float, so each entry resolves to
    /// whatever is current when its turn comes.
    pub pinned_version: bool,
    pub target: String,
    pub count: i32,
    pub entries: Vec<EnqueuedEntry>,
    pub status: String,
}

pub async fn enqueue(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Query(query): Query<WaitQuery>,
    body: Option<Json<EnqueueBody>>,
) -> Result<(StatusCode, Json<EnqueueResult>), ServerError> {
    let body = body.map(|Json(b)| b).unwrap_or_default();
    let count = body.count.unwrap_or(1);
    if count < 1 || count > MAX_COUNT {
        return Err(ServerError::BadRequest(format!(
            "count must be between 1 and {MAX_COUNT}, not {count}"
        )));
    }

    let dag = lookup(&state, &name).await?;
    let pinned_version = body.run.version.is_some();
    let version = body.run.version.unwrap_or(dag.current_version);

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
        .run
        .target
        .clone()
        .or_else(|| dag.default_target.clone())
        .ok_or_else(|| {
            ServerError::BadRequest(format!(
                "'{name}' has no target; set one with `dee dag submit --target` or pass one here"
            ))
        })?;

    // Where these entries land behind whatever is already queued for this DAG.
    let ahead = runs::list_queue(&state.store, Some(name.clone()), true, MAX_COUNT as usize)
        .await?
        .len() as i32;

    let mut entries = Vec::with_capacity(count as usize);
    for i in 0..count {
        let request = runs::RunRequest {
            dag_id: dag.dag_id.clone(),
            dag_version: version,
            target: target.clone(),
            trigger: "queue".into(),
            scheduled_for: None,
            warmups: body.run.warmups.unwrap_or(0),
            repetitions: body.run.repetitions.unwrap_or(1),
            cleanup_before: body.run.cleanup_before.unwrap_or(true),
            collect_plans: body.run.collect_plans.unwrap_or(false),
            sample_interval_ms: body.run.sample_interval_ms,
            queued: true,
            pin_version: pinned_version,
        };
        let created =
            runs::create_group(&state.store, request, state.instance_id.clone()).await?;
        entries.push(EnqueuedEntry {
            run_group_id: created.run_group_id,
            position: ahead + i + 1,
            run_ids: created.run_ids,
        });
    }

    // Nothing here claims the DAG. The dispatcher does that when the entry's
    // turn comes, which is the whole point: enqueueing never conflicts.
    state.wake_queue();

    let status = if query.wait {
        let timeout_s = query.timeout_s.unwrap_or(3600);
        let mut last = runs::status::SUCCEEDED.to_string();
        for entry in &entries {
            // Sequential by construction, so waiting on each in turn is the
            // same as waiting on the whole batch -- and the timeout applies
            // per entry, which is what a caller sizing it from one run expects.
            let outcome = wait_for_group(&state, &entry.run_group_id, timeout_s).await?;
            if outcome != runs::status::SUCCEEDED {
                last = outcome;
            }
        }
        last
    } else {
        runs::status::QUEUED.to_string()
    };

    Ok((
        StatusCode::ACCEPTED,
        Json(EnqueueResult {
            dag: name,
            version,
            pinned_version,
            target,
            count,
            entries,
            status,
        }),
    ))
}

#[derive(Deserialize, Default)]
pub struct ListQuery {
    #[serde(default)]
    pub dag: Option<String>,
    /// Include entries that have already finished. Off by default: the queue
    /// is a picture of what is left to do.
    #[serde(default)]
    pub all: bool,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Serialize)]
pub struct QueueEntry {
    #[serde(flatten)]
    pub group: runs::RunGroupRow,
    /// 1-based place among this DAG's entries that have not started. `None`
    /// once the entry leaves the queue.
    pub position: Option<i32>,
}

pub async fn list(
    State(state): State<AppState>,
    Query(query): Query<ListQuery>,
) -> Result<Json<Vec<QueueEntry>>, ServerError> {
    let limit = query.limit.unwrap_or(100).clamp(1, 10_000);
    let groups = runs::list_queue(&state.store, query.dag, !query.all, limit).await?;

    // Position is per DAG, because that is the queue a caller is actually in:
    // entries for different DAGs do not wait on each other.
    let mut next: std::collections::HashMap<String, i32> = std::collections::HashMap::new();
    let entries = groups
        .into_iter()
        .map(|group| {
            let position = (group.queue_state.as_deref() == Some(runs::queue_state::PENDING)
                && group.status == runs::status::QUEUED)
                .then(|| {
                    let slot = next.entry(group.dag_id.clone()).or_insert(0);
                    *slot += 1;
                    *slot
                });
            QueueEntry { group, position }
        })
        .collect();

    Ok(Json(entries))
}

/// Drop one entry that has not started yet.
///
/// A running entry is a run like any other, so it is cancelled through
/// `/v1/run-groups/{id}/cancel` rather than here -- the distinction being that
/// one has a warehouse to unwind and the other does not.
pub async fn drop_entry(
    State(state): State<AppState>,
    Path(group_id): Path<String>,
) -> Result<Json<serde_json::Value>, ServerError> {
    if runs::drop_pending(&state.store, group_id.clone()).await? {
        return Ok(Json(serde_json::json!({
            "run_group_id": group_id,
            "dropped": 1,
        })));
    }

    let group = runs::get_group(&state.store, group_id.clone())
        .await?
        .ok_or_else(|| ServerError::NotFound("queue entry", group_id.clone()))?;
    Err(ServerError::Conflict(format!(
        "{group_id} is not waiting in the queue (it is {}); cancel it with \
         `dee cancel {group_id}`",
        group.status
    )))
}

#[derive(Deserialize, Default)]
pub struct ClearQuery {
    #[serde(default)]
    pub dag: Option<String>,
}

pub async fn clear(
    State(state): State<AppState>,
    Query(query): Query<ClearQuery>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let dag_id = match &query.dag {
        Some(name) => Some(lookup(&state, name).await?.dag_id),
        None => None,
    };
    let dropped = runs::clear_pending(&state.store, dag_id).await?;
    Ok(Json(serde_json::json!({
        "dropped": dropped,
        "dag": query.dag,
    })))
}
