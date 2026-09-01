//! Which optimizations a DAG is under.
//!
//! Read on every run to decide what to step, so it is deliberately small: the
//! identity of the optimization, when it steps, and whether it has finished.
//! How an optimization searches, and everything it learned doing so, lives in
//! tables the optimization creates for itself.

use chrono::{DateTime, Utc};
use dee::opt::{OptimizationType, OptimizerConfig, StepPhase};
use serde::Serialize;

use crate::store::{LIST_PARAM, Store, StoreError, list_param, parse_list};

#[derive(Debug, Clone, Serialize)]
pub struct RegistrationRow {
    pub dag_id: String,
    pub dag_name: String,
    pub name: String,
    pub optimization_type: String,
    pub step_phase: String,
    pub config: Option<OptimizerConfig>,
    pub tables: Vec<String>,
    pub finished_at: Option<DateTime<Utc>>,
    pub result_version: Option<i32>,
    pub registered_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl RegistrationRow {
    pub fn step_phase(&self) -> StepPhase {
        self.step_phase.parse().unwrap_or(StepPhase::Both)
    }

    pub fn optimization_type(&self) -> OptimizationType {
        self.optimization_type
            .parse()
            .unwrap_or(OptimizationType::Continuous)
    }

    /// Whether this registration should still be stepped.
    ///
    /// A converged optimization stays registered so its state and trial
    /// history remain readable; it just stops being asked.
    pub fn is_active(&self) -> bool {
        self.finished_at.is_none()
    }
}

const SELECT: &str = "SELECT o.dag_id, d.name, o.name, o.optimization_type, o.step_phase,
        o.config, to_json(o.tables), o.finished_at, o.result_version,
        o.registered_at, o.updated_at
    FROM dag_optimizations o JOIN dags d USING (dag_id)";

fn row_from(row: &duckdb::Row<'_>) -> duckdb::Result<(RegistrationRow, Option<String>, String)> {
    Ok((
        RegistrationRow {
            dag_id: row.get(0)?,
            dag_name: row.get(1)?,
            name: row.get(2)?,
            optimization_type: row.get(3)?,
            step_phase: row.get(4)?,
            config: None,
            tables: Vec::new(),
            finished_at: row.get(7)?,
            result_version: row.get(8)?,
            registered_at: row.get(9)?,
            updated_at: row.get(10)?,
        },
        row.get::<_, Option<String>>(5)?,
        row.get::<_, String>(6)?,
    ))
}

fn decode(
    row: (RegistrationRow, Option<String>, String),
) -> Result<RegistrationRow, StoreError> {
    let (mut registration, config, tables) = row;
    registration.tables = parse_list(&tables)?;
    registration.config = match config {
        Some(raw) => Some(serde_json::from_str(&raw).map_err(|source| StoreError::Decode {
            what: "optimizer config",
            source,
        })?),
        None => None,
    };
    Ok(registration)
}

pub struct Register {
    pub dag_id: String,
    pub name: String,
    pub optimization_type: OptimizationType,
    pub step_phase: StepPhase,
    pub config: OptimizerConfig,
    pub tables: Vec<String>,
}

/// Record a registration, replacing any the DAG already had for this
/// optimization.
///
/// Replacing rather than refusing is what makes re-registering the way to
/// change an optimization's settings, and what lets a server restart
/// re-establish what a DAG already had without a special path.
pub async fn upsert(store: &Store, request: Register) -> Result<(), StoreError> {
    let config = serde_json::to_string(&request.config).map_err(|source| StoreError::Decode {
        what: "optimizer config",
        source,
    })?;
    let now = Utc::now();
    store
        .write(move |conn| {
            conn.execute(
                "DELETE FROM dag_optimizations WHERE dag_id = ? AND name = ?",
                duckdb::params![request.dag_id, request.name],
            )?;
            conn.execute(
                &format!(
                    "INSERT INTO dag_optimizations
                        (dag_id, name, optimization_type, step_phase, config, tables,
                         finished_at, result_version, registered_at, updated_at)
                     VALUES (?, ?, ?, ?, ?, {LIST_PARAM}, NULL, NULL, ?, ?)"
                ),
                duckdb::params![
                    request.dag_id,
                    request.name,
                    request.optimization_type.as_str(),
                    request.step_phase.as_str(),
                    config,
                    list_param(&request.tables),
                    now,
                    now,
                ],
            )?;
            Ok(())
        })
        .await
}

/// Every optimization registered on `dag_id`.
pub async fn for_dag(store: &Store, dag_id: String) -> Result<Vec<RegistrationRow>, StoreError> {
    store
        .read(move |conn| {
            let mut stmt = conn.prepare(&format!("{SELECT} WHERE o.dag_id = ? ORDER BY o.name"))?;
            let rows = stmt
                .query_map(duckdb::params![dag_id], |row| row_from(row))?
                .collect::<duckdb::Result<Vec<_>>>()?;
            rows.into_iter().map(decode).collect()
        })
        .await
}

/// Only the ones that should still be stepped.
pub async fn active_for_dag(
    store: &Store,
    dag_id: String,
) -> Result<Vec<RegistrationRow>, StoreError> {
    Ok(for_dag(store, dag_id)
        .await?
        .into_iter()
        .filter(RegistrationRow::is_active)
        .collect())
}

pub async fn get(
    store: &Store,
    dag_id: String,
    name: String,
) -> Result<Option<RegistrationRow>, StoreError> {
    store
        .read(move |conn| {
            let mut stmt = conn.prepare(&format!("{SELECT} WHERE o.dag_id = ? AND o.name = ?"))?;
            let mut rows = stmt.query_map(duckdb::params![dag_id, name], |row| row_from(row))?;
            match rows.next() {
                Some(row) => decode(row?).map(Some),
                None => Ok(None),
            }
        })
        .await
}

pub async fn list(store: &Store, limit: usize) -> Result<Vec<RegistrationRow>, StoreError> {
    store
        .read(move |conn| {
            let mut stmt =
                conn.prepare(&format!("{SELECT} ORDER BY o.registered_at DESC LIMIT ?"))?;
            let rows = stmt
                .query_map(duckdb::params![limit as i64], |row| row_from(row))?
                .collect::<duckdb::Result<Vec<_>>>()?;
            rows.into_iter().map(decode).collect()
        })
        .await
}

/// Change when an optimization steps.
pub async fn set_step_phase(
    store: &Store,
    dag_id: String,
    name: String,
    phase: StepPhase,
) -> Result<bool, StoreError> {
    store
        .write(move |conn| {
            let changed = conn.execute(
                "UPDATE dag_optimizations SET step_phase = ?, updated_at = ?
                 WHERE dag_id = ? AND name = ?",
                duckdb::params![phase.as_str(), Utc::now(), dag_id, name],
            )?;
            Ok(changed == 1)
        })
        .await
}

/// Mark an optimization as converged, optionally naming the version it
/// promoted. It stays registered; it just stops being stepped.
pub async fn mark_finished(
    store: &Store,
    dag_id: String,
    name: String,
    result_version: Option<i32>,
) -> Result<(), StoreError> {
    store
        .write(move |conn| {
            conn.execute(
                "UPDATE dag_optimizations
                 SET finished_at = ?, result_version = ?, updated_at = ?
                 WHERE dag_id = ? AND name = ?",
                duckdb::params![Utc::now(), result_version, Utc::now(), dag_id, name],
            )?;
            Ok(())
        })
        .await
}

pub async fn remove(store: &Store, dag_id: String, name: String) -> Result<bool, StoreError> {
    store
        .write(move |conn| {
            let removed = conn.execute(
                "DELETE FROM dag_optimizations WHERE dag_id = ? AND name = ?",
                duckdb::params![dag_id, name],
            )?;
            Ok(removed == 1)
        })
        .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::repo::dags;
    use dee::file::DagFile;

    async fn a_dag(store: &Store, name: &str) -> String {
        let definition: DagFile = serde_json::from_str(
            r#"{"db":"wh","nodes":[{"id":"a","query_text":"SELECT 1",
                "materialize":"table","depends_on":[]}],"sources":[]}"#,
        )
        .unwrap();
        dags::submit(
            store,
            dags::SubmitRequest::new(name.to_string(), definition, dags::Origin::Submitted),
        )
        .await
        .unwrap()
        .dag_id
    }

    #[tokio::test]
    async fn test_a_registration_round_trips() {
        let store = Store::open_temporary().unwrap();
        let dag_id = a_dag(&store, "pipeline").await;

        upsert(
            &store,
            Register {
                dag_id: dag_id.clone(),
                name: "hmp".into(),
                optimization_type: OptimizationType::Continuous,
                step_phase: StepPhase::Both,
                config: OptimizerConfig::default(),
                tables: vec!["opt_hmp_state".into(), "opt_hmp_trials".into()],
            },
        )
        .await
        .unwrap();

        let row = get(&store, dag_id, "hmp".into()).await.unwrap().unwrap();
        assert_eq!(row.optimization_type(), OptimizationType::Continuous);
        assert_eq!(row.step_phase(), StepPhase::Both);
        assert_eq!(row.tables, ["opt_hmp_state", "opt_hmp_trials"]);
        assert!(row.is_active(), "a fresh registration has not converged");
    }

    #[tokio::test]
    async fn test_registering_twice_replaces_rather_than_duplicates() {
        // Re-registering is how a restart re-establishes what a DAG had, and
        // how settings are changed. Two rows would mean two searches over one
        // set of tables.
        let store = Store::open_temporary().unwrap();
        let dag_id = a_dag(&store, "pipeline").await;

        for phase in [StepPhase::Both, StepPhase::After] {
            upsert(
                &store,
                Register {
                    dag_id: dag_id.clone(),
                    name: "hmp".into(),
                    optimization_type: OptimizationType::Continuous,
                    step_phase: phase,
                    config: OptimizerConfig::default(),
                    tables: vec![],
                },
            )
            .await
            .unwrap();
        }

        let rows = for_dag(&store, dag_id).await.unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].step_phase(), StepPhase::After);
    }

    #[tokio::test]
    async fn test_a_converged_optimization_stops_being_stepped_but_stays_visible() {
        // The distinction the driver reads: finished optimizations are not
        // stepped, but removing them from the listing would hide what they
        // decided and why.
        let store = Store::open_temporary().unwrap();
        let dag_id = a_dag(&store, "pipeline").await;

        upsert(
            &store,
            Register {
                dag_id: dag_id.clone(),
                name: "hmp".into(),
                optimization_type: OptimizationType::Continuous,
                step_phase: StepPhase::Both,
                config: OptimizerConfig::default(),
                tables: vec![],
            },
        )
        .await
        .unwrap();

        mark_finished(&store, dag_id.clone(), "hmp".into(), Some(3))
            .await
            .unwrap();

        assert!(active_for_dag(&store, dag_id.clone()).await.unwrap().is_empty());
        let listed = for_dag(&store, dag_id).await.unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].result_version, Some(3));
        assert!(!listed[0].is_active());
    }

    #[tokio::test]
    async fn test_the_step_phase_can_be_read_back_after_being_set() {
        // The configurable setting, through the layer that actually persists it.
        let store = Store::open_temporary().unwrap();
        let dag_id = a_dag(&store, "pipeline").await;
        upsert(
            &store,
            Register {
                dag_id: dag_id.clone(),
                name: "hmp".into(),
                optimization_type: OptimizationType::Continuous,
                step_phase: StepPhase::Both,
                config: OptimizerConfig::default(),
                tables: vec![],
            },
        )
        .await
        .unwrap();

        assert!(
            set_step_phase(&store, dag_id.clone(), "hmp".into(), StepPhase::Before)
                .await
                .unwrap()
        );
        let row = get(&store, dag_id.clone(), "hmp".into()).await.unwrap().unwrap();
        assert_eq!(row.step_phase(), StepPhase::Before);

        assert!(
            !set_step_phase(&store, dag_id, "omp".into(), StepPhase::Before)
                .await
                .unwrap(),
            "setting the phase of something not registered must report that"
        );
    }
}
