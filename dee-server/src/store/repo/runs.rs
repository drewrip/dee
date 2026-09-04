//! Run groups, runs, and everything a run observed.
//!
//! A *group* is one trigger. A scheduled fire makes a group of one; a
//! benchmark trigger with `warmups=2, repetitions=9` makes eleven runs sharing
//! a group. That is how the old `--repeat`/`--warmups` semantics survive the
//! move to a server: the whole series executes in one driver task against one
//! already-warm connection pool, so per-repetition timings still measure the
//! DAG rather than process startup and pool construction.

use chrono::{DateTime, Utc};
use dee::executor::ExecStats;
use serde::Serialize;

use crate::store::{Store, StoreError, new_id};

/// Terminal and in-flight states shared by runs, groups and optimizations.
pub mod status {
    pub const QUEUED: &str = "queued";
    pub const RUNNING: &str = "running";
    pub const SUCCEEDED: &str = "succeeded";
    pub const FAILED: &str = "failed";
    pub const CANCELLED: &str = "cancelled";
    pub const SKIPPED: &str = "skipped";
    pub const ORPHANED: &str = "orphaned";

    /// States that mean work is still owned by a server.
    pub const ACTIVE: [&str; 2] = [QUEUED, RUNNING];

    pub fn is_terminal(status: &str) -> bool {
        !ACTIVE.contains(&status)
    }
}

/// Where a group sits in the run queue.
///
/// A group with no queue state was dispatched the moment it was created --
/// `dee trigger` and the scheduler both do that. `Pending` and `Dispatched`
/// only ever appear on groups that arrived through `POST /v1/dags/{name}/queue`.
pub mod queue_state {
    pub const PENDING: &str = "pending";
    pub const DISPATCHED: &str = "dispatched";
}

/// What a caller asks for when triggering.
#[derive(Debug, Clone)]
pub struct RunRequest {
    pub dag_id: String,
    pub dag_version: i32,
    pub target: String,
    pub trigger: String,
    pub scheduled_for: Option<DateTime<Utc>>,
    pub warmups: i32,
    pub repetitions: i32,
    pub cleanup_before: bool,
    pub collect_plans: bool,
    pub sample_interval_ms: Option<i32>,
    /// Park the group in the queue instead of driving it now. The dispatcher
    /// starts it when the DAG is free and everything ahead of it has finished.
    pub queued: bool,
    /// False when `dag_version` was resolved from the DAG's current version
    /// rather than named by the caller, which is what lets a queued entry
    /// re-resolve to a newer version before it runs.
    pub pin_version: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct RunGroupRow {
    pub run_group_id: String,
    pub dag_id: String,
    pub dag_name: String,
    pub dag_version: i32,
    pub target: String,
    pub trigger: String,
    pub scheduled_for: Option<DateTime<Utc>>,
    pub warmups: i32,
    pub repetitions: i32,
    pub cleanup_before: bool,
    pub collect_plans: bool,
    pub sample_interval_ms: Option<i32>,
    pub status: String,
    pub created_at: DateTime<Utc>,
    pub finished_at: Option<DateTime<Utc>>,
    pub error: Option<String>,
    /// `None` for a group that never went through the queue.
    pub queue_state: Option<String>,
    pub pin_version: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct RunRow {
    pub run_id: String,
    pub run_group_id: String,
    pub dag_id: String,
    pub dag_name: String,
    pub dag_version: i32,
    pub target: String,
    pub phase: String,
    pub rep_index: i32,
    pub status: String,
    pub queued_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub finished_at: Option<DateTime<Utc>>,
    pub duration_ms: Option<i64>,
    pub node_time_ms: Option<i64>,
    pub node_count: Option<i32>,
    pub rows_produced: Option<i64>,
    pub cleanup_ms: Option<i64>,
    pub peak_engine_mem_bytes: Option<i64>,
    pub db_size_bytes: Option<i64>,
    pub plan_time_basis: Option<String>,
    /// How this run's tables reached the warehouse: `direct`, or `resumed`
    /// after a cancelled candidate. **Only a `direct` run's `duration_ms` is
    /// comparable to another run's** -- see [`Delivery`].
    pub delivery: Option<String>,
    /// What the cancelled candidate had spent when it was stopped, and what the
    /// resume that finished the run took. Both `None` on a direct delivery.
    pub trial_elapsed_ms: Option<i64>,
    pub resume_elapsed_ms: Option<i64>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct NodeExecutionRow {
    pub run_id: String,
    pub node_id: String,
    pub materialize: String,
    pub started_at: DateTime<Utc>,
    pub finished_at: DateTime<Utc>,
    pub duration_ms: i64,
    pub rows_produced: Option<i64>,
    pub has_plan: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct PlanRow {
    pub run_id: String,
    pub node_id: String,
    pub plan_format: String,
    pub plan_json: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SampleRow {
    pub run_id: String,
    pub seq: i32,
    pub timestamp: DateTime<Utc>,
    pub elapsed_ms: i64,
    pub cpu_percent: Option<f64>,
    pub memory_bytes: Option<i64>,
    pub disk_bytes: Option<i64>,
    pub read_bytes: Option<i64>,
    pub written_bytes: Option<i64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EventRow {
    pub event_id: String,
    pub ts: DateTime<Utc>,
    pub level: String,
    pub message: String,
}

pub struct CreatedGroup {
    pub run_group_id: String,
    pub run_ids: Vec<String>,
}

/// Create a group and its runs, all `queued`.
///
/// Warmups come first and are tagged `phase = 'warmup'`; they are recorded so
/// they are visible, but must be excluded from any aggregate.
///
/// `request.queued` decides who starts the group: the caller, right now, or
/// the queue dispatcher when the DAG frees up.
pub async fn create_group(
    store: &Store,
    request: RunRequest,
    instance_id: String,
) -> Result<CreatedGroup, StoreError> {
    let run_group_id = new_id();
    let now = Utc::now();
    let repetitions = request.repetitions.max(1);
    let warmups = request.warmups.max(0);

    let mut runs = Vec::new();
    for i in 0..warmups {
        runs.push((new_id(), "warmup".to_string(), i));
    }
    for i in 0..repetitions {
        runs.push((new_id(), "measure".to_string(), i));
    }
    let run_ids: Vec<String> = runs.iter().map(|(id, _, _)| id.clone()).collect();

    let group_id = run_group_id.clone();
    store
        .write(move |conn| {
            conn.execute_batch("BEGIN TRANSACTION;")?;
            let result = (|| -> Result<(), StoreError> {
                conn.execute(
                    "INSERT INTO run_groups
                        (run_group_id, dag_id, dag_version, target, trigger, scheduled_for,
                         warmups, repetitions, cleanup_before, collect_plans,
                         sample_interval_ms, status, created_at, instance_id,
                         queue_state, pin_version)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    duckdb::params![
                        group_id,
                        request.dag_id,
                        request.dag_version,
                        request.target,
                        request.trigger,
                        request.scheduled_for,
                        warmups,
                        repetitions,
                        request.cleanup_before,
                        request.collect_plans,
                        request.sample_interval_ms,
                        status::QUEUED,
                        now,
                        instance_id,
                        request.queued.then(|| queue_state::PENDING),
                        request.pin_version
                    ],
                )?;
                for (run_id, phase, rep_index) in &runs {
                    conn.execute(
                        "INSERT INTO runs
                            (run_id, run_group_id, dag_id, dag_version, target, phase,
                             rep_index, status, queued_at, instance_id)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        duckdb::params![
                            run_id,
                            group_id,
                            request.dag_id,
                            request.dag_version,
                            request.target,
                            phase,
                            rep_index,
                            status::QUEUED,
                            now,
                            instance_id
                        ],
                    )?;
                }
                Ok(())
            })();
            match result {
                Ok(()) => {
                    conn.execute_batch("COMMIT;")?;
                    Ok(())
                }
                Err(e) => {
                    let _ = conn.execute_batch("ROLLBACK;");
                    Err(e)
                }
            }
        })
        .await?;

    Ok(CreatedGroup {
        run_group_id,
        run_ids,
    })
}

/// Whether this DAG already has work in flight, and what is blocking.
///
/// Optimizations count: they run the DAG against the same warehouse, so an
/// optimize and a scheduled run would fight over the same relation names.
/// Entries still waiting in the queue count too, which is what stops a manual
/// trigger or a schedule from cutting in front of a queue that is mid-drain.
/// The dispatcher itself asks [`dispatch_blocker`] instead.
pub async fn active_job(store: &Store, dag_id: String) -> Result<Option<String>, StoreError> {
    store
        .read(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT run_group_id FROM run_groups
                 WHERE dag_id = ?1 AND status IN ('queued', 'running')
                 UNION ALL
                 SELECT optimization_id FROM optimizations
                 WHERE dag_id = ?1 AND status IN ('queued', 'running')
                 LIMIT 1",
            )?;
            let mut rows = stmt.query_map(duckdb::params![dag_id], |r| r.get::<_, String>(0))?;
            rows.next().transpose().map_err(StoreError::from)
        })
        .await
}

/// What actually blocks the queue from starting `dag_id` right now.
///
/// Deliberately narrower than [`active_job`]: entries still sitting in the
/// queue are excluded, because otherwise the first pending entry would report
/// itself -- and every entry behind it -- as a reason not to start. Only work
/// that has been handed to a driver, or an optimization, counts.
pub async fn dispatch_blocker(
    store: &Store,
    dag_id: String,
) -> Result<Option<String>, StoreError> {
    store
        .read(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT run_group_id FROM run_groups
                 WHERE dag_id = ?1 AND status IN ('queued', 'running')
                   AND (queue_state IS NULL OR queue_state = 'dispatched')
                 UNION ALL
                 SELECT optimization_id FROM optimizations
                 WHERE dag_id = ?1 AND status IN ('queued', 'running')
                 LIMIT 1",
            )?;
            let mut rows = stmt.query_map(duckdb::params![dag_id], |r| r.get::<_, String>(0))?;
            rows.next().transpose().map_err(StoreError::from)
        })
        .await
}

/// The front of the queue, oldest first.
///
/// Entries for different DAGs are interleaved here; the dispatcher decides
/// which of them can actually start. Ordering by id after `created_at` is what
/// makes N entries submitted in one burst dequeue in submission order.
pub async fn next_pending(store: &Store, limit: usize) -> Result<Vec<RunGroupRow>, StoreError> {
    store
        .read(move |conn| {
            let mut stmt = conn.prepare(&format!(
                "{GROUP_SELECT}
                 WHERE g.queue_state = 'pending' AND g.status = 'queued'
                 ORDER BY g.created_at, g.run_group_id
                 LIMIT ?"
            ))?;
            Ok(stmt
                .query_map(duckdb::params![limit as i64], group_from)?
                .collect::<duckdb::Result<Vec<_>>>()?)
        })
        .await
}

/// Take an entry out of the queue. Returns false if something already did.
///
/// This is the queue's compare-and-set: the transition out of `pending`
/// happens in the same statement that checks for it, so a dispatch racing a
/// `DELETE /v1/queue/{id}` cannot both win.
pub async fn mark_dispatched(store: &Store, group_id: String) -> Result<bool, StoreError> {
    store
        .write(move |conn| {
            let n = conn.execute(
                "UPDATE run_groups SET queue_state = ?
                 WHERE run_group_id = ? AND queue_state = 'pending' AND status = 'queued'",
                duckdb::params![queue_state::DISPATCHED, group_id],
            )?;
            Ok(n == 1)
        })
        .await
}

/// Put a dispatched entry back at its original place in the queue.
///
/// Only used when the in-memory claim is lost between the compare-and-set and
/// the spawn. Position is preserved because ordering is by `created_at`, which
/// this does not touch.
pub async fn requeue(store: &Store, group_id: String) -> Result<(), StoreError> {
    store
        .write(move |conn| {
            conn.execute(
                "UPDATE run_groups SET queue_state = ?
                 WHERE run_group_id = ? AND queue_state = 'dispatched' AND status = 'queued'",
                duckdb::params![queue_state::PENDING, group_id],
            )?;
            Ok(())
        })
        .await
}

/// Repoint a queued group at a different version of its DAG.
///
/// The group's runs carry `dag_version` too -- the benchmark reads it from
/// there -- so both move together or the two disagree about what ran.
pub async fn set_group_version(
    store: &Store,
    group_id: String,
    version: i32,
) -> Result<(), StoreError> {
    store
        .write(move |conn| {
            conn.execute_batch("BEGIN TRANSACTION;")?;
            let result = (|| -> Result<(), StoreError> {
                conn.execute(
                    "UPDATE run_groups SET dag_version = ? WHERE run_group_id = ?",
                    duckdb::params![version, group_id],
                )?;
                conn.execute(
                    "UPDATE runs SET dag_version = ? WHERE run_group_id = ?",
                    duckdb::params![version, group_id],
                )?;
                Ok(())
            })();
            match result {
                Ok(()) => {
                    conn.execute_batch("COMMIT;")?;
                    Ok(())
                }
                Err(e) => {
                    let _ = conn.execute_batch("ROLLBACK;");
                    Err(e)
                }
            }
        })
        .await
}

/// The queue, front first. Includes entries the dispatcher has already started.
pub async fn list_queue(
    store: &Store,
    dag_name: Option<String>,
    active_only: bool,
    limit: usize,
) -> Result<Vec<RunGroupRow>, StoreError> {
    store
        .read(move |conn| {
            let mut stmt = conn.prepare(&format!(
                "{GROUP_SELECT}
                 WHERE g.queue_state IS NOT NULL
                   AND (?1 IS NULL OR d.name = ?1)
                   AND (NOT ?2 OR g.status IN ('queued', 'running'))
                 ORDER BY g.created_at, g.run_group_id
                 LIMIT ?3"
            ))?;
            Ok(stmt
                .query_map(
                    duckdb::params![dag_name, active_only, limit as i64],
                    group_from,
                )?
                .collect::<duckdb::Result<Vec<_>>>()?)
        })
        .await
}

/// Drop one entry that has not started. Returns false if it already has.
pub async fn drop_pending(store: &Store, group_id: String) -> Result<bool, StoreError> {
    Ok(cancel_pending(store, CancelPending::One(group_id)).await? > 0)
}

/// Drop every entry that has not started, optionally for one DAG.
pub async fn clear_pending(
    store: &Store,
    dag_id: Option<String>,
) -> Result<usize, StoreError> {
    cancel_pending(store, CancelPending::All(dag_id)).await
}

enum CancelPending {
    One(String),
    All(Option<String>),
}

/// Cancelled, not deleted: a benchmark that queued fifty runs and abandoned
/// thirty should still be able to say so afterwards.
async fn cancel_pending(store: &Store, what: CancelPending) -> Result<usize, StoreError> {
    let now = Utc::now();
    let (group_id, dag_id) = match what {
        CancelPending::One(id) => (Some(id), None),
        CancelPending::All(dag_id) => (None, dag_id),
    };
    store
        .write(move |conn| {
            const REASON: &str = "removed from the run queue before it started";
            // The predicate is written once and applied to both tables through
            // the group id, so a partially cancelled entry is not possible.
            conn.execute_batch("BEGIN TRANSACTION;")?;
            let result = (|| -> Result<usize, StoreError> {
                conn.execute(
                    "UPDATE runs SET status = ?1, finished_at = ?2, error = ?3
                     WHERE status IN ('queued', 'running')
                       AND run_group_id IN (
                           SELECT run_group_id FROM run_groups
                           WHERE queue_state = 'pending' AND status = 'queued'
                             AND (?4 IS NULL OR run_group_id = ?4)
                             AND (?5 IS NULL OR dag_id = ?5))",
                    duckdb::params![status::CANCELLED, now, REASON, group_id, dag_id],
                )?;
                let n = conn.execute(
                    "UPDATE run_groups SET status = ?1, finished_at = ?2, error = ?3
                     WHERE queue_state = 'pending' AND status = 'queued'
                       AND (?4 IS NULL OR run_group_id = ?4)
                       AND (?5 IS NULL OR dag_id = ?5)",
                    duckdb::params![status::CANCELLED, now, REASON, group_id, dag_id],
                )?;
                Ok(n)
            })();
            match result {
                Ok(n) => {
                    conn.execute_batch("COMMIT;")?;
                    Ok(n)
                }
                Err(e) => {
                    let _ = conn.execute_batch("ROLLBACK;");
                    Err(e)
                }
            }
        })
        .await
}

const RUN_SELECT: &str = "SELECT r.run_id, r.run_group_id, r.dag_id, d.name, r.dag_version,
        r.target, r.phase, r.rep_index, r.status, r.queued_at, r.started_at, r.finished_at,
        r.duration_ms, r.node_time_ms, r.node_count, r.rows_produced, r.cleanup_ms,
        r.peak_engine_mem_bytes, r.db_size_bytes, r.plan_time_basis,
        r.delivery, r.trial_elapsed_ms, r.resume_elapsed_ms, r.error
    FROM runs r JOIN dags d USING (dag_id)";

fn run_from(row: &duckdb::Row<'_>) -> duckdb::Result<RunRow> {
    Ok(RunRow {
        run_id: row.get(0)?,
        run_group_id: row.get(1)?,
        dag_id: row.get(2)?,
        dag_name: row.get(3)?,
        dag_version: row.get(4)?,
        target: row.get(5)?,
        phase: row.get(6)?,
        rep_index: row.get(7)?,
        status: row.get(8)?,
        queued_at: row.get(9)?,
        started_at: row.get(10)?,
        finished_at: row.get(11)?,
        duration_ms: row.get(12)?,
        node_time_ms: row.get(13)?,
        node_count: row.get(14)?,
        rows_produced: row.get(15)?,
        cleanup_ms: row.get(16)?,
        peak_engine_mem_bytes: row.get(17)?,
        db_size_bytes: row.get(18)?,
        plan_time_basis: row.get(19)?,
        delivery: row.get(20)?,
        trial_elapsed_ms: row.get(21)?,
        resume_elapsed_ms: row.get(22)?,
        error: row.get(23)?,
    })
}

pub async fn get_run(store: &Store, run_id: String) -> Result<Option<RunRow>, StoreError> {
    store
        .read(move |conn| {
            let mut stmt = conn.prepare(&format!("{RUN_SELECT} WHERE r.run_id = ?"))?;
            let mut rows = stmt.query_map(duckdb::params![run_id], run_from)?;
            rows.next().transpose().map_err(StoreError::from)
        })
        .await
}

/// Runs in a group, warmups first, then measured repetitions in order. This is
/// also the order the driver executes them in.
pub async fn runs_in_group(store: &Store, group_id: String) -> Result<Vec<RunRow>, StoreError> {
    store
        .read(move |conn| {
            let mut stmt = conn.prepare(&format!(
                "{RUN_SELECT} WHERE r.run_group_id = ?
                 ORDER BY CASE r.phase WHEN 'warmup' THEN 0 ELSE 1 END, r.rep_index"
            ))?;
            Ok(stmt
                .query_map(duckdb::params![group_id], run_from)?
                .collect::<duckdb::Result<Vec<_>>>()?)
        })
        .await
}

#[derive(Debug, Clone, Default)]
pub struct RunFilter {
    pub dag_name: Option<String>,
    pub status: Option<String>,
    pub run_group_id: Option<String>,
    pub phase: Option<String>,
    pub limit: usize,
}

pub async fn list_runs(store: &Store, filter: RunFilter) -> Result<Vec<RunRow>, StoreError> {
    store
        .read(move |conn| {
            // Built with literal-free predicates that no-op when a filter is
            // absent, so there is one prepared shape rather than a matrix of
            // hand-concatenated SQL.
            let mut stmt = conn.prepare(&format!(
                "{RUN_SELECT}
                 WHERE (?1 IS NULL OR d.name = ?1)
                   AND (?2 IS NULL OR r.status = ?2)
                   AND (?3 IS NULL OR r.run_group_id = ?3)
                   AND (?4 IS NULL OR r.phase = ?4)
                 -- Newest series first, but execution order within a series:
                 -- warmups then repetitions, which is how they actually ran.
                 ORDER BY r.queued_at DESC,
                          CASE r.phase WHEN 'warmup' THEN 0 ELSE 1 END,
                          r.rep_index
                 LIMIT ?5"
            ))?;
            Ok(stmt
                .query_map(
                    duckdb::params![
                        filter.dag_name,
                        filter.status,
                        filter.run_group_id,
                        filter.phase,
                        filter.limit as i64
                    ],
                    run_from,
                )?
                .collect::<duckdb::Result<Vec<_>>>()?)
        })
        .await
}

const GROUP_SELECT: &str = "SELECT g.run_group_id, g.dag_id, d.name, g.dag_version, g.target,
        g.trigger, g.scheduled_for, g.warmups, g.repetitions, g.cleanup_before,
        g.collect_plans, g.sample_interval_ms, g.status, g.created_at, g.finished_at, g.error,
        g.queue_state, g.pin_version
    FROM run_groups g JOIN dags d USING (dag_id)";

fn group_from(row: &duckdb::Row<'_>) -> duckdb::Result<RunGroupRow> {
    Ok(RunGroupRow {
        run_group_id: row.get(0)?,
        dag_id: row.get(1)?,
        dag_name: row.get(2)?,
        dag_version: row.get(3)?,
        target: row.get(4)?,
        trigger: row.get(5)?,
        scheduled_for: row.get(6)?,
        warmups: row.get(7)?,
        repetitions: row.get(8)?,
        cleanup_before: row.get(9)?,
        collect_plans: row.get(10)?,
        sample_interval_ms: row.get(11)?,
        status: row.get(12)?,
        created_at: row.get(13)?,
        finished_at: row.get(14)?,
        error: row.get(15)?,
        queue_state: row.get(16)?,
        pin_version: row.get(17)?,
    })
}

pub async fn get_group(store: &Store, group_id: String) -> Result<Option<RunGroupRow>, StoreError> {
    store
        .read(move |conn| {
            let mut stmt = conn.prepare(&format!("{GROUP_SELECT} WHERE g.run_group_id = ?"))?;
            let mut rows = stmt.query_map(duckdb::params![group_id], group_from)?;
            rows.next().transpose().map_err(StoreError::from)
        })
        .await
}

pub async fn mark_group_running(store: &Store, group_id: String) -> Result<(), StoreError> {
    store
        .write(move |conn| {
            conn.execute(
                "UPDATE run_groups SET status = ? WHERE run_group_id = ?",
                duckdb::params![status::RUNNING, group_id],
            )?;
            Ok(())
        })
        .await
}

/// Close out a group, deriving its status from the runs it produced.
pub async fn finalize_group(
    store: &Store,
    group_id: String,
    error: Option<String>,
) -> Result<String, StoreError> {
    let now = Utc::now();
    store
        .write(move |conn| {
            // A group is only a success if every one of its runs succeeded: a
            // benchmark series with a failed repetition is not a clean result.
            let (failed, cancelled, unfinished): (i64, i64, i64) = conn.query_row(
                "SELECT count(*) FILTER (WHERE status = 'failed'),
                        count(*) FILTER (WHERE status = 'cancelled'),
                        count(*) FILTER (WHERE status IN ('queued', 'running'))
                 FROM runs WHERE run_group_id = ?",
                duckdb::params![group_id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )?;

            // Runs left behind by an aborted series never ran at all.
            if unfinished > 0 {
                conn.execute(
                    "UPDATE runs SET status = ?, finished_at = ?
                     WHERE run_group_id = ? AND status IN ('queued', 'running')",
                    duckdb::params![status::SKIPPED, now, group_id],
                )?;
            }

            let group_status = if failed > 0 {
                status::FAILED
            } else if cancelled > 0 {
                status::CANCELLED
            } else {
                status::SUCCEEDED
            };

            conn.execute(
                "UPDATE run_groups SET status = ?, finished_at = ?, error = ?
                 WHERE run_group_id = ?",
                duckdb::params![group_status, now, error, group_id],
            )?;
            Ok(group_status.to_string())
        })
        .await
}

pub async fn mark_run_running(store: &Store, run_id: String) -> Result<(), StoreError> {
    let now = Utc::now();
    store
        .write(move |conn| {
            conn.execute(
                "UPDATE runs SET status = ?, started_at = ? WHERE run_id = ?",
                duckdb::params![status::RUNNING, now, run_id],
            )?;
            Ok(())
        })
        .await
}

/// Point a run at the version it actually executed.
///
/// A run group is dispatched against whatever version was current at the time,
/// but a continuous optimization that converges on a `Before` step promotes its
/// result and that run executes the promotion. Leaving the run pointing at the
/// version it was dispatched for would mean `dee runs list` disagreed with what
/// the run did.
pub async fn set_run_version(
    store: &Store,
    run_id: String,
    dag_version: i32,
) -> Result<(), StoreError> {
    store
        .write(move |conn| {
            conn.execute(
                "UPDATE runs SET dag_version = ? WHERE run_id = ?",
                duckdb::params![dag_version, run_id],
            )?;
            Ok(())
        })
        .await
}

pub async fn mark_run_terminal(
    store: &Store,
    run_id: String,
    run_status: &'static str,
    error: Option<String>,
) -> Result<(), StoreError> {
    let now = Utc::now();
    store
        .write(move |conn| {
            conn.execute(
                "UPDATE runs SET status = ?, finished_at = ?, error = ? WHERE run_id = ?",
                duckdb::params![run_status, now, error, run_id],
            )?;
            Ok(())
        })
        .await
}

/// How a run's tables reached the warehouse.
///
/// A run whose candidate was cancelled mid-way and finished under the search's
/// incumbent still delivered every relation the consumer asked for, so it
/// succeeded. Its wall time, though, measures neither DAG: the second half
/// started from a warm, half-built warehouse. Recording that here is what lets
/// every comparison filter on it, rather than each caller having to reconstruct
/// from the event log whether a run is comparable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Delivery {
    kind: &'static str,
    trial_ms: Option<i64>,
    resume_ms: Option<i64>,
}

impl Delivery {
    /// One execution of one DAG. The only kind whose elapsed time means
    /// anything.
    pub fn direct() -> Self {
        Self {
            kind: delivery::DIRECT,
            trial_ms: None,
            resume_ms: None,
        }
    }

    /// A cancelled candidate followed by a resume under the incumbent.
    pub fn resumed(trial_ms: i64, resume_ms: i64) -> Self {
        Self {
            kind: delivery::RESUMED,
            trial_ms: Some(trial_ms),
            resume_ms: Some(resume_ms),
        }
    }

    pub fn as_str(&self) -> &'static str {
        self.kind
    }

    /// Whether this run's elapsed time may be compared with another's.
    pub fn is_measurement(&self) -> bool {
        self.kind == delivery::DIRECT
    }
}

/// The values `runs.delivery` takes.
pub mod delivery {
    /// The run executed one DAG start to finish.
    pub const DIRECT: &str = "direct";
    /// A candidate was cancelled at its budget and the run finished under the
    /// incumbent. Never a measurement -- see [`super::Delivery`].
    pub const RESUMED: &str = "resumed";

    /// The SQL predicate every runtime comparison must carry.
    pub const COMPARABLE: &str = "delivery = 'direct'";
}

/// Everything one successful execution observed, written as one transaction.
///
/// A partially recorded run would be worse than a missing one: it would look
/// complete while under-reporting node time and rows.
pub async fn record_success(
    store: &Store,
    run_id: String,
    delivery: Delivery,
    stats: ExecStats,
    materializations: Vec<(String, String)>,
    plan_format: String,
    plan_time_basis: String,
    cleanup_ms: i64,
) -> Result<(), StoreError> {
    let node_time_ms: i64 = stats
        .node_stats
        .values()
        .map(|n| n.duration.num_milliseconds())
        .sum();
    let node_count = stats.node_stats.len() as i32;
    let rows_produced: Option<i64> = {
        let total: i64 = stats
            .node_stats
            .values()
            .filter_map(|n| n.rows_produced)
            .map(|r| r as i64)
            .sum();
        // Distinguish "reported zero rows" from "the backend reports nothing".
        if stats.node_stats.values().any(|n| n.rows_produced.is_some()) {
            Some(total)
        } else {
            None
        }
    };
    let peak_memory = stats
        .system_samples
        .iter()
        .filter_map(|s| s.memory_bytes)
        .max()
        .map(|v| v as i64);
    let peak_disk = stats
        .system_samples
        .iter()
        .filter_map(|s| s.disk_bytes)
        .max()
        .map(|v| v as i64);
    let duration_ms = stats.duration.num_milliseconds();
    let now = Utc::now();

    store
        .write(move |conn| {
            conn.execute_batch("BEGIN TRANSACTION;")?;
            let result = (|| -> Result<(), StoreError> {
                conn.execute(
                    "UPDATE runs
                     SET status = ?, started_at = ?, finished_at = ?, duration_ms = ?,
                         node_time_ms = ?, node_count = ?, rows_produced = ?, cleanup_ms = ?,
                         peak_engine_mem_bytes = ?, db_size_bytes = ?, plan_time_basis = ?,
                         delivery = ?, trial_elapsed_ms = ?, resume_elapsed_ms = ?
                     WHERE run_id = ?",
                    duckdb::params![
                        status::SUCCEEDED,
                        stats.start,
                        stats.finish,
                        duration_ms,
                        node_time_ms,
                        node_count,
                        rows_produced,
                        cleanup_ms,
                        peak_memory,
                        peak_disk,
                        plan_time_basis,
                        delivery.as_str(),
                        delivery.trial_ms,
                        delivery.resume_ms,
                        run_id
                    ],
                )?;

                let modes: std::collections::HashMap<&str, &str> = materializations
                    .iter()
                    .map(|(id, mode)| (id.as_str(), mode.as_str()))
                    .collect();

                for (node_id, node) in &stats.node_stats {
                    let has_plan = node.plan.is_some();
                    conn.execute(
                        "INSERT INTO node_executions
                            (run_id, node_id, materialize, started_at, finished_at,
                             duration_ms, rows_produced, has_plan)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        duckdb::params![
                            run_id,
                            node_id,
                            modes.get(node_id.as_str()).copied().unwrap_or("view"),
                            node.start,
                            node.finish,
                            node.duration.num_milliseconds(),
                            node.rows_produced.map(|r| r as i64),
                            has_plan
                        ],
                    )?;
                    if let Some(plan) = &node.plan {
                        conn.execute(
                            "INSERT INTO plans (run_id, node_id, plan_format, plan_json)
                             VALUES (?, ?, ?, ?)",
                            duckdb::params![run_id, node_id, plan_format, plan],
                        )?;
                    }
                }

                for (seq, sample) in stats.system_samples.iter().enumerate() {
                    conn.execute(
                        "INSERT INTO run_samples
                            (run_id, seq, timestamp, elapsed_ms, cpu_percent, memory_bytes,
                             disk_bytes, read_bytes, written_bytes)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        duckdb::params![
                            run_id,
                            seq as i32,
                            sample.timestamp,
                            sample.elapsed_ms,
                            sample.cpu_percent,
                            sample.memory_bytes.map(|v| v as i64),
                            sample.disk_bytes.map(|v| v as i64),
                            sample.read_bytes.map(|v| v as i64),
                            sample.written_bytes.map(|v| v as i64)
                        ],
                    )?;
                }
                let _ = now;
                Ok(())
            })();
            match result {
                Ok(()) => {
                    conn.execute_batch("COMMIT;")?;
                    Ok(())
                }
                Err(e) => {
                    let _ = conn.execute_batch("ROLLBACK;");
                    Err(e)
                }
            }
        })
        .await
}

pub async fn node_executions(
    store: &Store,
    run_id: String,
) -> Result<Vec<NodeExecutionRow>, StoreError> {
    store
        .read(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT run_id, node_id, materialize, started_at, finished_at, duration_ms,
                        rows_produced, has_plan
                 FROM node_executions WHERE run_id = ? ORDER BY started_at, node_id",
            )?;
            Ok(stmt
                .query_map(duckdb::params![run_id], |r| {
                    Ok(NodeExecutionRow {
                        run_id: r.get(0)?,
                        node_id: r.get(1)?,
                        materialize: r.get(2)?,
                        started_at: r.get(3)?,
                        finished_at: r.get(4)?,
                        duration_ms: r.get(5)?,
                        rows_produced: r.get(6)?,
                        has_plan: r.get(7)?,
                    })
                })?
                .collect::<duckdb::Result<Vec<_>>>()?)
        })
        .await
}

pub async fn plans(store: &Store, run_id: String) -> Result<Vec<PlanRow>, StoreError> {
    store
        .read(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT run_id, node_id, plan_format, plan_json
                 FROM plans WHERE run_id = ? ORDER BY node_id",
            )?;
            Ok(stmt
                .query_map(duckdb::params![run_id], |r| {
                    Ok(PlanRow {
                        run_id: r.get(0)?,
                        node_id: r.get(1)?,
                        plan_format: r.get(2)?,
                        plan_json: r.get(3)?,
                    })
                })?
                .collect::<duckdb::Result<Vec<_>>>()?)
        })
        .await
}

pub async fn samples(store: &Store, run_id: String) -> Result<Vec<SampleRow>, StoreError> {
    store
        .read(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT run_id, seq, timestamp, elapsed_ms, cpu_percent, memory_bytes,
                        disk_bytes, read_bytes, written_bytes
                 FROM run_samples WHERE run_id = ? ORDER BY seq",
            )?;
            Ok(stmt
                .query_map(duckdb::params![run_id], |r| {
                    Ok(SampleRow {
                        run_id: r.get(0)?,
                        seq: r.get(1)?,
                        timestamp: r.get(2)?,
                        elapsed_ms: r.get(3)?,
                        cpu_percent: r.get(4)?,
                        memory_bytes: r.get(5)?,
                        disk_bytes: r.get(6)?,
                        read_bytes: r.get(7)?,
                        written_bytes: r.get(8)?,
                    })
                })?
                .collect::<duckdb::Result<Vec<_>>>()?)
        })
        .await
}

/// Record one lifecycle event. Deliberately coarse: this backs `dee runs
/// logs`, not a log firehose.
pub async fn log_event(
    store: &Store,
    run_id: Option<String>,
    run_group_id: Option<String>,
    dag_id: Option<String>,
    level: &'static str,
    message: String,
) -> Result<(), StoreError> {
    let event_id = new_id();
    let now = Utc::now();
    store
        .write(move |conn| {
            conn.execute(
                "INSERT INTO events
                    (event_id, run_id, run_group_id, optimization_id, dag_id, ts, level, message)
                 VALUES (?, ?, ?, NULL, ?, ?, ?, ?)",
                duckdb::params![event_id, run_id, run_group_id, dag_id, now, level, message],
            )?;
            Ok(())
        })
        .await
}

pub async fn events_for_run(store: &Store, run_id: String) -> Result<Vec<EventRow>, StoreError> {
    store
        .read(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT event_id, ts, level, message FROM events
                 WHERE run_id = ?1 OR run_group_id = ?1 ORDER BY event_id",
            )?;
            Ok(stmt
                .query_map(duckdb::params![run_id], |r| {
                    Ok(EventRow {
                        event_id: r.get(0)?,
                        ts: r.get(1)?,
                        level: r.get(2)?,
                        message: r.get(3)?,
                    })
                })?
                .collect::<duckdb::Result<Vec<_>>>()?)
        })
        .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Store;
    use dee::executor::NodeStats;
    use dee::profile::SystemUsageSample;
    use std::collections::HashMap;

    async fn seeded() -> (Store, String) {
        let store = Store::open_temporary().unwrap();
        store
            .write(|c| {
                c.execute(
                    "INSERT INTO dags (dag_id, name, current_version, default_target,
                                       created_at, updated_at)
                     VALUES ('d1', 'sales', 1, 'wh', now(), now())",
                    [],
                )?;
                Ok(())
            })
            .await
            .unwrap();
        (store, "d1".to_string())
    }

    fn request(dag_id: &str, warmups: i32, repetitions: i32) -> RunRequest {
        RunRequest {
            dag_id: dag_id.to_string(),
            dag_version: 1,
            target: "wh".into(),
            trigger: "manual".into(),
            scheduled_for: None,
            warmups,
            repetitions,
            cleanup_before: true,
            collect_plans: false,
            sample_interval_ms: None,
            queued: false,
            pin_version: true,
        }
    }

    fn queued_request(dag_id: &str, pin_version: bool) -> RunRequest {
        RunRequest {
            queued: true,
            pin_version,
            ..request(dag_id, 0, 1)
        }
    }

    #[tokio::test]
    async fn test_the_queue_dequeues_in_submission_order() {
        let (store, dag_id) = seeded().await;
        let mut submitted = Vec::new();
        for _ in 0..5 {
            submitted.push(
                create_group(&store, queued_request(&dag_id, true), "i".into())
                    .await
                    .unwrap()
                    .run_group_id,
            );
        }

        // Ids are UUIDv7, so entries created inside one millisecond still come
        // back in the order they were written.
        let order: Vec<String> = next_pending(&store, 10)
            .await
            .unwrap()
            .into_iter()
            .map(|g| g.run_group_id)
            .collect();
        assert_eq!(order, submitted);
    }

    #[tokio::test]
    async fn test_a_waiting_entry_blocks_a_trigger_but_not_the_dispatcher() {
        let (store, dag_id) = seeded().await;
        let entry = create_group(&store, queued_request(&dag_id, true), "i".into())
            .await
            .unwrap()
            .run_group_id;

        // A manual trigger must not cut the line...
        assert_eq!(
            active_job(&store, dag_id.clone()).await.unwrap().as_deref(),
            Some(entry.as_str())
        );
        // ...but the queue must not treat its own front entry as a blocker,
        // or nothing would ever start.
        assert!(dispatch_blocker(&store, dag_id.clone()).await.unwrap().is_none());

        assert!(mark_dispatched(&store, entry.clone()).await.unwrap());
        assert_eq!(
            dispatch_blocker(&store, dag_id).await.unwrap().as_deref(),
            Some(entry.as_str()),
            "once started it blocks like any other job"
        );
    }

    #[tokio::test]
    async fn test_taking_an_entry_out_of_the_queue_happens_exactly_once() {
        let (store, dag_id) = seeded().await;
        let entry = create_group(&store, queued_request(&dag_id, true), "i".into())
            .await
            .unwrap()
            .run_group_id;

        assert!(mark_dispatched(&store, entry.clone()).await.unwrap());
        assert!(
            !mark_dispatched(&store, entry.clone()).await.unwrap(),
            "a second dispatch of the same entry must lose"
        );

        // A lost claim puts it back where it was, still ahead of nothing.
        requeue(&store, entry.clone()).await.unwrap();
        assert!(mark_dispatched(&store, entry).await.unwrap());
    }

    #[tokio::test]
    async fn test_dropping_a_waiting_entry_cancels_it_and_its_runs() {
        let (store, dag_id) = seeded().await;
        let entry = create_group(
            &store,
            RunRequest {
                queued: true,
                ..request(&dag_id, 1, 3)
            },
            "i".into(),
        )
        .await
        .unwrap()
        .run_group_id;

        assert!(drop_pending(&store, entry.clone()).await.unwrap());
        let group = get_group(&store, entry.clone()).await.unwrap().unwrap();
        assert_eq!(group.status, status::CANCELLED);
        // Cancelled, not deleted: a benchmark that abandoned thirty of fifty
        // queued runs should still be able to say so afterwards.
        let series = runs_in_group(&store, entry.clone()).await.unwrap();
        assert_eq!(series.len(), 4);
        assert!(series.iter().all(|r| r.status == status::CANCELLED));

        assert!(
            !drop_pending(&store, entry).await.unwrap(),
            "dropping it twice is not a second cancellation"
        );
    }

    #[tokio::test]
    async fn test_a_started_entry_cannot_be_dropped_from_the_queue() {
        let (store, dag_id) = seeded().await;
        let entry = create_group(&store, queued_request(&dag_id, true), "i".into())
            .await
            .unwrap()
            .run_group_id;
        mark_dispatched(&store, entry.clone()).await.unwrap();

        // It has a warehouse to unwind now, so it is cancelled as a run, not
        // dropped as a queue entry.
        assert!(!drop_pending(&store, entry.clone()).await.unwrap());
        let group = get_group(&store, entry).await.unwrap().unwrap();
        assert_eq!(group.status, status::QUEUED);
    }

    #[tokio::test]
    async fn test_clearing_one_dags_queue_leaves_the_others_alone() {
        let (store, dag_id) = seeded().await;
        let other = "d2";
        store
            .write(move |conn| {
                conn.execute(
                    "INSERT INTO dags (dag_id, name, current_version, created_at, updated_at)
                     VALUES ('d2', 'churn', 1, now(), now())",
                    [],
                )?;
                Ok(())
            })
            .await
            .unwrap();

        for _ in 0..3 {
            create_group(&store, queued_request(&dag_id, true), "i".into())
                .await
                .unwrap();
        }
        create_group(&store, queued_request(other, true), "i".into())
            .await
            .unwrap();

        assert_eq!(clear_pending(&store, Some(dag_id)).await.unwrap(), 3);
        let left = next_pending(&store, 10).await.unwrap();
        assert_eq!(left.len(), 1);
        assert_eq!(left[0].dag_id, other);

        assert_eq!(clear_pending(&store, None).await.unwrap(), 1);
        assert!(next_pending(&store, 10).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_listing_the_queue_hides_finished_entries_unless_asked() {
        let (store, dag_id) = seeded().await;
        let done = create_group(&store, queued_request(&dag_id, true), "i".into())
            .await
            .unwrap()
            .run_group_id;
        let waiting = create_group(&store, queued_request(&dag_id, true), "i".into())
            .await
            .unwrap()
            .run_group_id;
        finalize_group(&store, done.clone(), None).await.unwrap();

        // An ordinary trigger is not a queue entry and never appears here.
        create_group(&store, request(&dag_id, 0, 1), "i".into())
            .await
            .unwrap();

        let active = list_queue(&store, None, true, 10).await.unwrap();
        assert_eq!(
            active.iter().map(|g| g.run_group_id.clone()).collect::<Vec<_>>(),
            vec![waiting.clone()]
        );

        let all = list_queue(&store, None, false, 10).await.unwrap();
        assert_eq!(
            all.iter().map(|g| g.run_group_id.clone()).collect::<Vec<_>>(),
            vec![done, waiting]
        );
    }

    #[tokio::test]
    async fn test_repointing_an_entry_moves_the_group_and_its_runs_together() {
        let (store, dag_id) = seeded().await;
        let entry = create_group(
            &store,
            RunRequest {
                queued: true,
                pin_version: false,
                ..request(&dag_id, 1, 2)
            },
            "i".into(),
        )
        .await
        .unwrap()
        .run_group_id;

        set_group_version(&store, entry.clone(), 7).await.unwrap();

        let group = get_group(&store, entry.clone()).await.unwrap().unwrap();
        assert_eq!(group.dag_version, 7);
        assert!(!group.pin_version);
        // The two must not be able to disagree about what ran.
        let series = runs_in_group(&store, entry).await.unwrap();
        assert!(series.iter().all(|r| r.dag_version == 7));
    }

    fn exec_stats(node_ms: i64, rows: Option<u64>) -> ExecStats {
        let start = Utc::now();
        let finish = start + chrono::TimeDelta::milliseconds(node_ms);
        let mut node_stats = HashMap::new();
        node_stats.insert(
            "a".to_string(),
            NodeStats {
                start,
                finish,
                duration: chrono::TimeDelta::milliseconds(node_ms),
                plan: Some("{\"plan\":true}".into()),
                rows_produced: rows,
            },
        );
        ExecStats {
            start,
            finish,
            duration: chrono::TimeDelta::milliseconds(node_ms),
            node_stats,
            system_samples: vec![SystemUsageSample {
                timestamp: start,
                elapsed_ms: 0,
                cpu_percent: Some(12.5),
                memory_bytes: Some(4096),
                disk_bytes: Some(8192),
                read_bytes: None,
                written_bytes: None,
            }],
        }
    }

    #[tokio::test]
    async fn test_a_group_creates_warmups_then_repetitions_in_execution_order() {
        // This is the old --warmups/--repeat contract: W untimed runs, then N
        // timed ones, all in one series.
        let (store, dag_id) = seeded().await;
        let created = create_group(&store, request(&dag_id, 2, 3), "i".into())
            .await
            .unwrap();
        assert_eq!(created.run_ids.len(), 5);

        let series = runs_in_group(&store, created.run_group_id).await.unwrap();
        let shape: Vec<(String, i32)> = series
            .iter()
            .map(|r| (r.phase.clone(), r.rep_index))
            .collect();
        assert_eq!(
            shape,
            vec![
                ("warmup".into(), 0),
                ("warmup".into(), 1),
                ("measure".into(), 0),
                ("measure".into(), 1),
                ("measure".into(), 2),
            ]
        );
        assert!(series.iter().all(|r| r.status == status::QUEUED));
    }

    #[tokio::test]
    async fn test_a_group_always_has_at_least_one_repetition() {
        let (store, dag_id) = seeded().await;
        let created = create_group(&store, request(&dag_id, 0, 0), "i".into())
            .await
            .unwrap();
        assert_eq!(created.run_ids.len(), 1);
    }

    #[tokio::test]
    async fn test_active_job_sees_both_runs_and_optimizations() {
        // An optimization runs the DAG against the same warehouse, so it has
        // to block a scheduled run just as another run would.
        let (store, dag_id) = seeded().await;
        assert!(active_job(&store, dag_id.clone()).await.unwrap().is_none());

        let created = create_group(&store, request(&dag_id, 0, 1), "i".into())
            .await
            .unwrap();
        assert_eq!(
            active_job(&store, dag_id.clone()).await.unwrap(),
            Some(created.run_group_id.clone())
        );

        finalize_group(&store, created.run_group_id, None).await.unwrap();
        assert!(active_job(&store, dag_id.clone()).await.unwrap().is_none());

        store
            .write(|c| {
                c.execute(
                    "INSERT INTO optimizations (optimization_id, dag_id, source_version, target,
                                                status, config, instance_id)
                     VALUES ('o1', 'd1', 1, 'wh', 'running', '{}', 'i')",
                    [],
                )?;
                Ok(())
            })
            .await
            .unwrap();
        assert_eq!(
            active_job(&store, dag_id).await.unwrap(),
            Some("o1".to_string())
        );
    }

    #[tokio::test]
    async fn test_recording_a_success_captures_timings_plans_and_samples() {
        let (store, dag_id) = seeded().await;
        let created = create_group(&store, request(&dag_id, 0, 1), "i".into())
            .await
            .unwrap();
        let run_id = created.run_ids[0].clone();

        record_success(
            &store,
            run_id.clone(),
            Delivery::direct(),
            exec_stats(42, Some(1000)),
            vec![("a".into(), "table".into())],
            "duckdb_json".into(),
            "cpu_time".into(),
            7,
        )
        .await
        .unwrap();

        let run = get_run(&store, run_id.clone()).await.unwrap().unwrap();
        assert_eq!(run.status, status::SUCCEEDED);
        assert_eq!(run.duration_ms, Some(42));
        assert_eq!(run.node_time_ms, Some(42));
        assert_eq!(run.node_count, Some(1));
        assert_eq!(run.rows_produced, Some(1000));
        assert_eq!(run.cleanup_ms, Some(7));
        assert_eq!(run.plan_time_basis.as_deref(), Some("cpu_time"));
        // Peaks come from the sampler, not from the node stats.
        assert_eq!(run.peak_engine_mem_bytes, Some(4096));
        assert_eq!(run.db_size_bytes, Some(8192));

        let nodes = node_executions(&store, run_id.clone()).await.unwrap();
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].materialize, "table");
        assert!(nodes[0].has_plan);

        assert_eq!(plans(&store, run_id.clone()).await.unwrap().len(), 1);
        assert_eq!(samples(&store, run_id).await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_rows_produced_is_null_when_the_backend_reports_none() {
        // Postgres does not always report row counts; a null must not be
        // recorded as a genuine zero.
        let (store, dag_id) = seeded().await;
        let created = create_group(&store, request(&dag_id, 0, 1), "i".into())
            .await
            .unwrap();
        let run_id = created.run_ids[0].clone();

        record_success(
            &store,
            run_id.clone(),
            Delivery::direct(),
            exec_stats(10, None),
            vec![("a".into(), "view".into())],
            "postgres_json".into(),
            "wall_time".into(),
            0,
        )
        .await
        .unwrap();

        let run = get_run(&store, run_id).await.unwrap().unwrap();
        assert_eq!(run.rows_produced, None);
    }

    #[tokio::test]
    async fn test_a_resumed_run_is_recorded_as_not_comparable() {
        // The run delivered its tables, so it succeeded -- but its wall time
        // covers a cancelled candidate plus a warm-start resume and measures
        // neither DAG. Anything that compares runtimes has to be able to see
        // that from the row alone.
        let (store, dag_id) = seeded().await;
        let created = create_group(&store, request(&dag_id, 0, 1), "i".into())
            .await
            .unwrap();
        record_success(
            &store,
            created.run_ids[0].clone(),
            Delivery::resumed(1250, 400),
            exec_stats(5, Some(1)),
            vec![],
            "duckdb_json".into(),
            "cpu_time".into(),
            0,
        )
        .await
        .unwrap();

        let run = get_run(&store, created.run_ids[0].clone())
            .await
            .unwrap()
            .expect("the run was recorded");
        assert_eq!(run.status, status::SUCCEEDED);
        assert_eq!(run.delivery.as_deref(), Some(delivery::RESUMED));
        assert_eq!(run.trial_elapsed_ms, Some(1250));
        assert_eq!(run.resume_elapsed_ms, Some(400));
        assert!(!Delivery::resumed(1250, 400).is_measurement());

        // And the predicate every comparison carries actually excludes it.
        let comparable: i64 = store
            .read(move |c| {
                Ok(c.query_row(
                    &format!("SELECT count(*) FROM runs WHERE {}", delivery::COMPARABLE),
                    [],
                    |r| r.get(0),
                )?)
            })
            .await
            .unwrap();
        assert_eq!(comparable, 0);
    }

    #[tokio::test]
    async fn test_an_ordinary_run_stays_comparable() {
        let (store, dag_id) = seeded().await;
        let created = create_group(&store, request(&dag_id, 0, 1), "i".into())
            .await
            .unwrap();
        record_success(
            &store,
            created.run_ids[0].clone(),
            Delivery::direct(),
            exec_stats(5, Some(1)),
            vec![],
            "duckdb_json".into(),
            "cpu_time".into(),
            0,
        )
        .await
        .unwrap();
        let run = get_run(&store, created.run_ids[0].clone())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(run.delivery.as_deref(), Some(delivery::DIRECT));
        assert_eq!(run.trial_elapsed_ms, None);
        assert!(Delivery::direct().is_measurement());
    }

    #[tokio::test]
    async fn test_a_group_succeeds_only_if_every_run_did() {
        let (store, dag_id) = seeded().await;
        let created = create_group(&store, request(&dag_id, 0, 2), "i".into())
            .await
            .unwrap();

        record_success(
            &store,
            created.run_ids[0].clone(),
            Delivery::direct(),
            exec_stats(5, Some(1)),
            vec![],
            "duckdb_json".into(),
            "cpu_time".into(),
            0,
        )
        .await
        .unwrap();
        mark_run_terminal(
            &store,
            created.run_ids[1].clone(),
            status::FAILED,
            Some("boom".into()),
        )
        .await
        .unwrap();

        let outcome = finalize_group(&store, created.run_group_id, None).await.unwrap();
        assert_eq!(outcome, status::FAILED);
    }

    #[tokio::test]
    async fn test_finalizing_marks_never_started_repetitions_skipped() {
        // An aborted series leaves queued runs behind. Left alone they would
        // read as permanently pending.
        let (store, dag_id) = seeded().await;
        let created = create_group(&store, request(&dag_id, 0, 3), "i".into())
            .await
            .unwrap();
        mark_run_terminal(
            &store,
            created.run_ids[0].clone(),
            status::CANCELLED,
            None,
        )
        .await
        .unwrap();

        let outcome = finalize_group(&store, created.run_group_id.clone(), None)
            .await
            .unwrap();
        assert_eq!(outcome, status::CANCELLED);

        let series = runs_in_group(&store, created.run_group_id).await.unwrap();
        let skipped = series.iter().filter(|r| r.status == status::SKIPPED).count();
        assert_eq!(skipped, 2);
        assert!(series.iter().all(|r| status::is_terminal(&r.status)));
    }

    #[tokio::test]
    async fn test_listing_filters_and_orders_by_execution_order() {
        let (store, dag_id) = seeded().await;
        let created = create_group(&store, request(&dag_id, 1, 2), "i".into())
            .await
            .unwrap();

        let all = list_runs(
            &store,
            RunFilter {
                limit: 50,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(all.len(), 3);
        // Warmups run first, so they list first within a series.
        assert_eq!(all[0].phase, "warmup");

        let measured = list_runs(
            &store,
            RunFilter {
                phase: Some("measure".into()),
                limit: 50,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(measured.len(), 2);

        let by_dag = list_runs(
            &store,
            RunFilter {
                dag_name: Some("nope".into()),
                limit: 50,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert!(by_dag.is_empty());

        let by_group = list_runs(
            &store,
            RunFilter {
                run_group_id: Some(created.run_group_id),
                limit: 50,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(by_group.len(), 3);
    }

    #[tokio::test]
    async fn test_events_are_reachable_from_a_run_and_from_its_group() {
        let (store, dag_id) = seeded().await;
        let created = create_group(&store, request(&dag_id, 0, 1), "i".into())
            .await
            .unwrap();
        let run_id = created.run_ids[0].clone();

        log_event(
            &store,
            None,
            Some(created.run_group_id.clone()),
            Some(dag_id),
            "info",
            "starting".into(),
        )
        .await
        .unwrap();
        log_event(
            &store,
            Some(run_id.clone()),
            Some(created.run_group_id.clone()),
            None,
            "info",
            "finished".into(),
        )
        .await
        .unwrap();

        // Group-level events matter to a run: "starting" belongs to the series
        // the run is part of.
        let events = events_for_run(&store, run_id).await.unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].message, "finished");

        let group_events = events_for_run(&store, created.run_group_id).await.unwrap();
        assert_eq!(group_events.len(), 2);
        // UUIDv7 ids sort chronologically, so no sequence column is needed.
        assert_eq!(group_events[0].message, "starting");
    }
}
