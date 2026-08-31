//! Optimizer runs and everything they observed.
//!
//! The full typed `OptimizeReport` is kept verbatim in `report`, so no field
//! the library adds later is silently dropped and any existing consumer of
//! `--report-json` works unchanged. The flattened tables beside it are what
//! anything actually filters or aggregates on -- in particular
//! `optimization_iterations`, which is the search trace a history-seeded
//! optimizer would read to avoid re-measuring combinations it already knows.

use chrono::{DateTime, Utc};
use dee::opt::OptimizerConfig;
use dee::opt::report::OptimizeReport;
use serde::Serialize;

use crate::store::{LIST_PARAM, Store, StoreError, list_param, new_id, parse_list};

#[derive(Debug, Clone, Serialize)]
pub struct OptimizationRow {
    pub optimization_id: String,
    pub dag_id: String,
    pub dag_name: String,
    pub source_version: i32,
    pub result_version: Option<i32>,
    pub target: String,
    pub status: String,
    pub started_at: Option<DateTime<Utc>>,
    pub finished_at: Option<DateTime<Utc>>,
    pub wall_ms: Option<i64>,
    pub baseline_runtime_ms: Option<i64>,
    pub final_runtime_ms: Option<i64>,
    pub dag_runs_used: Option<i32>,
    pub total_changes_applied: Option<i32>,
    pub nodes_before: Option<i32>,
    pub nodes_after: Option<i32>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PassRow {
    pub pass_order: i32,
    pub pass_name: String,
    pub wall_ms: i64,
    pub dag_runs_used: i32,
    pub changes_applied: i32,
    pub candidates_considered: i32,
    pub working_set_size: i32,
    pub detail: serde_json::Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct IterationRow {
    pub pass_order: i32,
    pub pass_name: String,
    pub iteration: i32,
    pub runtime_ms: i64,
    pub combo: Vec<String>,
    pub outcome: Option<String>,
}

/// Create the row before any work starts, so an optimization is visible (and
/// blocks the DAG) from the moment it is accepted.
pub async fn create(
    store: &Store,
    dag_id: String,
    source_version: i32,
    target: String,
    config: &OptimizerConfig,
    instance_id: String,
) -> Result<String, StoreError> {
    let optimization_id = new_id();
    let config_json = serde_json::to_string(config).map_err(|source| StoreError::Decode {
        what: "optimizer config",
        source,
    })?;
    let id = optimization_id.clone();
    let now = Utc::now();

    store
        .write(move |conn| {
            conn.execute(
                "INSERT INTO optimizations
                    (optimization_id, dag_id, source_version, target, status, started_at,
                     config, instance_id)
                 VALUES (?, ?, ?, ?, 'running', ?, ?, ?)",
                duckdb::params![id, dag_id, source_version, target, now, config_json, instance_id],
            )?;
            Ok(())
        })
        .await?;
    Ok(optimization_id)
}

/// Record a completed optimization: header, per-pass rows, and the search trace.
pub async fn record_success(
    store: &Store,
    optimization_id: String,
    report: OptimizeReport,
    result_version: Option<i32>,
    explain_html: Option<String>,
) -> Result<(), StoreError> {
    let report_json = serde_json::to_string(&report).map_err(|source| StoreError::Decode {
        what: "optimize report",
        source,
    })?;
    let total_changes = report.total_changes_applied() as i32;

    store
        .write(move |conn| {
            conn.execute_batch("BEGIN TRANSACTION;")?;
            let result = (|| -> Result<(), StoreError> {
                conn.execute(
                    "UPDATE optimizations
                     SET status = 'succeeded', started_at = ?, finished_at = ?, wall_ms = ?,
                         baseline_runtime_ms = ?, final_runtime_ms = ?, dag_runs_used = ?,
                         total_changes_applied = ?, nodes_before = ?, nodes_after = ?,
                         result_version = ?, report = ?, explain_html = ?
                     WHERE optimization_id = ?",
                    duckdb::params![
                        report.started_at,
                        report.finished_at,
                        report.wall_ms as i64,
                        report.baseline_runtime_ms.map(|v| v as i64),
                        report.final_runtime_ms.map(|v| v as i64),
                        report.dag_runs_used as i32,
                        total_changes,
                        report.nodes_before as i32,
                        report.nodes_after as i32,
                        result_version,
                        report_json,
                        explain_html,
                        optimization_id
                    ],
                )?;

                for pass in &report.passes {
                    let detail = serde_json::to_string(&pass.detail).map_err(|source| {
                        StoreError::Decode {
                            what: "pass detail",
                            source,
                        }
                    })?;
                    conn.execute(
                        "INSERT INTO optimization_passes
                            (optimization_id, pass_order, pass_name, started_at, finished_at,
                             wall_ms, dag_runs_used, changes_applied, candidates_considered,
                             working_set_size, detail)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        duckdb::params![
                            optimization_id,
                            pass.order as i32,
                            pass.pass,
                            pass.started_at,
                            pass.finished_at,
                            pass.wall_ms as i64,
                            pass.dag_runs_used as i32,
                            pass.changes_applied as i32,
                            pass.candidates_considered as i32,
                            pass.working_set_size as i32,
                            detail
                        ],
                    )?;

                    for iteration in &pass.iterations {
                        let samples = if iteration.system_samples.is_empty() {
                            None
                        } else {
                            Some(serde_json::to_string(&iteration.system_samples).map_err(
                                |source| StoreError::Decode {
                                    what: "iteration samples",
                                    source,
                                },
                            )?)
                        };
                        let peak_rss = iteration
                            .system_samples
                            .iter()
                            .filter_map(|s| s.memory_bytes)
                            .max()
                            .map(|v| v as i64);
                        conn.execute(
                            &format!(
                                "INSERT INTO optimization_iterations
                                    (optimization_id, pass_order, pass_name, iteration,
                                     runtime_ms, combo, outcome, cpu_seconds, peak_rss_bytes,
                                     samples)
                                 VALUES (?, ?, ?, ?, ?, {LIST_PARAM}, ?, NULL, ?, ?)"
                            ),
                            duckdb::params![
                                optimization_id,
                                pass.order as i32,
                                pass.pass,
                                iteration.iteration as i32,
                                iteration.runtime_ms as i64,
                                list_param(&iteration.combo),
                                iteration.outcome,
                                peak_rss,
                                samples
                            ],
                        )?;
                    }
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
        .await
}

pub async fn record_failure(
    store: &Store,
    optimization_id: String,
    status: &'static str,
    error: String,
) -> Result<(), StoreError> {
    let now = Utc::now();
    store
        .write(move |conn| {
            conn.execute(
                "UPDATE optimizations SET status = ?, finished_at = ?, error = ?
                 WHERE optimization_id = ?",
                duckdb::params![status, now, error, optimization_id],
            )?;
            Ok(())
        })
        .await
}

const SELECT: &str = "SELECT o.optimization_id, o.dag_id, d.name, o.source_version,
        o.result_version, o.target, o.status, o.started_at, o.finished_at, o.wall_ms,
        o.baseline_runtime_ms, o.final_runtime_ms, o.dag_runs_used, o.total_changes_applied,
        o.nodes_before, o.nodes_after, o.error
    FROM optimizations o JOIN dags d USING (dag_id)";

fn row_from(row: &duckdb::Row<'_>) -> duckdb::Result<OptimizationRow> {
    Ok(OptimizationRow {
        optimization_id: row.get(0)?,
        dag_id: row.get(1)?,
        dag_name: row.get(2)?,
        source_version: row.get(3)?,
        result_version: row.get(4)?,
        target: row.get(5)?,
        status: row.get(6)?,
        started_at: row.get(7)?,
        finished_at: row.get(8)?,
        wall_ms: row.get(9)?,
        baseline_runtime_ms: row.get(10)?,
        final_runtime_ms: row.get(11)?,
        dag_runs_used: row.get(12)?,
        total_changes_applied: row.get(13)?,
        nodes_before: row.get(14)?,
        nodes_after: row.get(15)?,
        error: row.get(16)?,
    })
}

pub async fn get(store: &Store, id: String) -> Result<Option<OptimizationRow>, StoreError> {
    store
        .read(move |conn| {
            let mut stmt = conn.prepare(&format!("{SELECT} WHERE o.optimization_id = ?"))?;
            let mut rows = stmt.query_map(duckdb::params![id], row_from)?;
            rows.next().transpose().map_err(StoreError::from)
        })
        .await
}

pub async fn list(
    store: &Store,
    dag_name: Option<String>,
    limit: usize,
) -> Result<Vec<OptimizationRow>, StoreError> {
    store
        .read(move |conn| {
            let mut stmt = conn.prepare(&format!(
                "{SELECT} WHERE (?1 IS NULL OR d.name = ?1)
                 ORDER BY o.started_at DESC LIMIT ?2"
            ))?;
            Ok(stmt
                .query_map(duckdb::params![dag_name, limit as i64], row_from)?
                .collect::<duckdb::Result<Vec<_>>>()?)
        })
        .await
}

/// The stored `OptimizeReport`, exactly as the library produced it.
pub async fn report(store: &Store, id: String) -> Result<Option<OptimizeReport>, StoreError> {
    let body: Option<Option<String>> = store
        .read(move |conn| {
            let mut stmt =
                conn.prepare("SELECT report FROM optimizations WHERE optimization_id = ?")?;
            let mut rows = stmt.query_map(duckdb::params![id], |r| r.get::<_, Option<String>>(0))?;
            rows.next().transpose().map_err(StoreError::from)
        })
        .await?;

    match body.flatten() {
        Some(text) => serde_json::from_str(&text)
            .map(Some)
            .map_err(|source| StoreError::Decode {
                what: "optimize report",
                source,
            }),
        None => Ok(None),
    }
}

pub async fn explain_html(store: &Store, id: String) -> Result<Option<String>, StoreError> {
    let body: Option<Option<String>> = store
        .read(move |conn| {
            let mut stmt =
                conn.prepare("SELECT explain_html FROM optimizations WHERE optimization_id = ?")?;
            let mut rows = stmt.query_map(duckdb::params![id], |r| r.get::<_, Option<String>>(0))?;
            rows.next().transpose().map_err(StoreError::from)
        })
        .await?;
    Ok(body.flatten())
}

pub async fn passes(store: &Store, id: String) -> Result<Vec<PassRow>, StoreError> {
    store
        .read(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT pass_order, pass_name, wall_ms, dag_runs_used, changes_applied,
                        candidates_considered, working_set_size, detail
                 FROM optimization_passes WHERE optimization_id = ? ORDER BY pass_order",
            )?;
            let rows = stmt
                .query_map(duckdb::params![id], |r| {
                    Ok((
                        r.get::<_, i32>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, i64>(2)?,
                        r.get::<_, i32>(3)?,
                        r.get::<_, i32>(4)?,
                        r.get::<_, i32>(5)?,
                        r.get::<_, i32>(6)?,
                        r.get::<_, String>(7)?,
                    ))
                })?
                .collect::<duckdb::Result<Vec<_>>>()?;

            rows.into_iter()
                .map(|(order, name, wall, runs, changes, candidates, working, detail)| {
                    Ok(PassRow {
                        pass_order: order,
                        pass_name: name,
                        wall_ms: wall,
                        dag_runs_used: runs,
                        changes_applied: changes,
                        candidates_considered: candidates,
                        working_set_size: working,
                        detail: serde_json::from_str(&detail).map_err(|source| {
                            StoreError::Decode {
                                what: "pass detail",
                                source,
                            }
                        })?,
                    })
                })
                .collect()
        })
        .await
}

pub async fn iterations(store: &Store, id: String) -> Result<Vec<IterationRow>, StoreError> {
    store
        .read(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT pass_order, pass_name, iteration, runtime_ms, to_json(combo), outcome
                 FROM optimization_iterations WHERE optimization_id = ?
                 ORDER BY pass_order, iteration",
            )?;
            let rows = stmt
                .query_map(duckdb::params![id], |r| {
                    Ok((
                        r.get::<_, i32>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, i32>(2)?,
                        r.get::<_, i64>(3)?,
                        r.get::<_, String>(4)?,
                        r.get::<_, Option<String>>(5)?,
                    ))
                })?
                .collect::<duckdb::Result<Vec<_>>>()?;

            rows.into_iter()
                .map(|(order, name, iteration, runtime, combo, outcome)| {
                    Ok(IterationRow {
                        pass_order: order,
                        pass_name: name,
                        iteration,
                        runtime_ms: runtime,
                        combo: parse_list(&combo)?,
                        outcome,
                    })
                })
                .collect()
        })
        .await
}
