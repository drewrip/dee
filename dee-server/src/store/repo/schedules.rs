//! Schedules and the record of windows that produced no run.

use chrono::{DateTime, Utc};
use serde::Serialize;

use crate::store::{Store, StoreError, new_id};

#[derive(Debug, Clone, Serialize)]
pub struct ScheduleRow {
    pub dag_id: String,
    pub dag_name: String,
    pub cron: String,
    pub timezone: String,
    pub enabled: bool,
    pub catchup: bool,
    pub overlap_policy: String,
    pub target: Option<String>,
    pub next_fire_at: Option<DateTime<Utc>>,
    pub last_fire_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SkipRow {
    pub skip_id: String,
    pub dag_id: String,
    pub dag_name: String,
    pub scheduled_for: DateTime<Utc>,
    pub detected_at: DateTime<Utc>,
    pub reason: String,
    pub blocking_run_id: Option<String>,
    pub windows_skipped: i32,
    pub detail: Option<String>,
}

/// Reasons a window produced nothing. Recorded rather than logged, because
/// "the schedule did nothing" is otherwise indistinguishable from "the server
/// was down".
pub mod reason {
    /// Another job for this DAG was still running.
    pub const OVERLAP: &str = "overlap";
    /// Windows that elapsed while the server was not running.
    pub const MISSED_WINDOW: &str = "missed_window";
    /// The schedule names no connection and the DAG has no default.
    pub const NO_TARGET: &str = "no_target";
    /// The fire itself failed.
    pub const ERROR: &str = "error";
}

const SELECT: &str = "SELECT s.dag_id, d.name, s.cron, s.timezone, s.enabled, s.catchup,
        s.overlap_policy, s.target, s.next_fire_at, s.last_fire_at
    FROM schedules s JOIN dags d USING (dag_id)";

fn row_from(row: &duckdb::Row<'_>) -> duckdb::Result<ScheduleRow> {
    Ok(ScheduleRow {
        dag_id: row.get(0)?,
        dag_name: row.get(1)?,
        cron: row.get(2)?,
        timezone: row.get(3)?,
        enabled: row.get(4)?,
        catchup: row.get(5)?,
        overlap_policy: row.get(6)?,
        target: row.get(7)?,
        next_fire_at: row.get(8)?,
        last_fire_at: row.get(9)?,
    })
}

#[allow(clippy::too_many_arguments)]
pub async fn upsert(
    store: &Store,
    dag_id: String,
    cron: String,
    timezone: String,
    enabled: bool,
    target: Option<String>,
    next_fire_at: Option<DateTime<Utc>>,
) -> Result<(), StoreError> {
    let now = Utc::now();
    store
        .write(move |conn| {
            let exists: i64 = conn.query_row(
                "SELECT count(*) FROM schedules WHERE dag_id = ?",
                duckdb::params![dag_id],
                |r| r.get(0),
            )?;
            if exists > 0 {
                conn.execute(
                    "UPDATE schedules
                     SET cron = ?, timezone = ?, enabled = ?, target = ?,
                         next_fire_at = ?, updated_at = ?
                     WHERE dag_id = ?",
                    duckdb::params![cron, timezone, enabled, target, next_fire_at, now, dag_id],
                )?;
            } else {
                conn.execute(
                    "INSERT INTO schedules
                        (dag_id, cron, timezone, enabled, target, next_fire_at,
                         created_at, updated_at)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    duckdb::params![
                        dag_id, cron, timezone, enabled, target, next_fire_at, now, now
                    ],
                )?;
            }
            Ok(())
        })
        .await
}

pub async fn get(store: &Store, dag_id: String) -> Result<Option<ScheduleRow>, StoreError> {
    store
        .read(move |conn| {
            let mut stmt = conn.prepare(&format!("{SELECT} WHERE s.dag_id = ?"))?;
            let mut rows = stmt.query_map(duckdb::params![dag_id], row_from)?;
            rows.next().transpose().map_err(StoreError::from)
        })
        .await
}

pub async fn list(store: &Store) -> Result<Vec<ScheduleRow>, StoreError> {
    store
        .read(|conn| {
            let mut stmt = conn.prepare(&format!(
                "{SELECT} ORDER BY s.next_fire_at NULLS LAST, d.name"
            ))?;
            Ok(stmt.query_map([], row_from)?.collect::<duckdb::Result<Vec<_>>>()?)
        })
        .await
}

/// Every enabled schedule, whether due or not. The scheduler filters in Rust
/// so it can parse each cron once and report parse failures per schedule.
pub async fn enabled(store: &Store) -> Result<Vec<ScheduleRow>, StoreError> {
    store
        .read(|conn| {
            let mut stmt = conn.prepare(&format!("{SELECT} WHERE s.enabled ORDER BY d.name"))?;
            Ok(stmt.query_map([], row_from)?.collect::<duckdb::Result<Vec<_>>>()?)
        })
        .await
}

pub async fn set_enabled(
    store: &Store,
    dag_id: String,
    enabled: bool,
    next_fire_at: Option<DateTime<Utc>>,
) -> Result<bool, StoreError> {
    let now = Utc::now();
    store
        .write(move |conn| {
            let n = conn.execute(
                "UPDATE schedules SET enabled = ?, next_fire_at = ?, updated_at = ?
                 WHERE dag_id = ?",
                duckdb::params![enabled, next_fire_at, now, dag_id],
            )?;
            Ok(n > 0)
        })
        .await
}

/// Move a schedule forward after it fired (or was skipped).
pub async fn advance(
    store: &Store,
    dag_id: String,
    next_fire_at: Option<DateTime<Utc>>,
    last_fire_at: Option<DateTime<Utc>>,
) -> Result<(), StoreError> {
    let now = Utc::now();
    store
        .write(move |conn| {
            conn.execute(
                "UPDATE schedules
                 SET next_fire_at = ?, last_fire_at = coalesce(?, last_fire_at), updated_at = ?
                 WHERE dag_id = ?",
                duckdb::params![next_fire_at, last_fire_at, now, dag_id],
            )?;
            Ok(())
        })
        .await
}

pub async fn delete(store: &Store, dag_id: String) -> Result<bool, StoreError> {
    store
        .write(move |conn| {
            let n = conn.execute(
                "DELETE FROM schedules WHERE dag_id = ?",
                duckdb::params![dag_id],
            )?;
            Ok(n > 0)
        })
        .await
}

pub async fn record_skip(
    store: &Store,
    dag_id: String,
    scheduled_for: DateTime<Utc>,
    reason: &'static str,
    blocking_run_id: Option<String>,
    windows_skipped: i32,
    detail: Option<String>,
) -> Result<String, StoreError> {
    let skip_id = new_id();
    let id = skip_id.clone();
    let now = Utc::now();
    store
        .write(move |conn| {
            conn.execute(
                "INSERT INTO schedule_skips
                    (skip_id, dag_id, scheduled_for, detected_at, reason, blocking_run_id,
                     windows_skipped, detail)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                duckdb::params![
                    id,
                    dag_id,
                    scheduled_for,
                    now,
                    reason,
                    blocking_run_id,
                    windows_skipped,
                    detail
                ],
            )?;
            Ok(())
        })
        .await?;
    Ok(skip_id)
}

pub async fn skips(
    store: &Store,
    dag_name: Option<String>,
    limit: usize,
) -> Result<Vec<SkipRow>, StoreError> {
    store
        .read(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT s.skip_id, s.dag_id, d.name, s.scheduled_for, s.detected_at, s.reason,
                        s.blocking_run_id, s.windows_skipped, s.detail
                 FROM schedule_skips s JOIN dags d USING (dag_id)
                 WHERE (?1 IS NULL OR d.name = ?1)
                 ORDER BY s.detected_at DESC LIMIT ?2",
            )?;
            Ok(stmt
                .query_map(duckdb::params![dag_name, limit as i64], |r| {
                    Ok(SkipRow {
                        skip_id: r.get(0)?,
                        dag_id: r.get(1)?,
                        dag_name: r.get(2)?,
                        scheduled_for: r.get(3)?,
                        detected_at: r.get(4)?,
                        reason: r.get(5)?,
                        blocking_run_id: r.get(6)?,
                        windows_skipped: r.get(7)?,
                        detail: r.get(8)?,
                    })
                })?
                .collect::<duckdb::Result<Vec<_>>>()?)
        })
        .await
}
