//! Attaching optimizations to a DAG.
//!
//! `POST /v1/dags/{name}/optimize` still exists and still means "optimize this
//! DAG now, to convergence". These routes are the other shape the server made
//! possible: an optimization *registered* on a DAG, stepping around its runs
//! and improving it over its lifetime.
//!
//! The difference that matters operationally is what each costs. An optimize
//! request performs its own DAG runs, so it takes the same exclusive per-DAG
//! claim a run does and cannot share the warehouse with the schedule.
//! Registration performs none -- a continuous optimization spends the runs the
//! DAG was going to perform anyway -- so it is a cheap metadata write that
//! never blocks anything.

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use dee::opt::{OptimizerConfig, StepPhase, registry};
use serde::{Deserialize, Serialize};

use crate::api::dags::lookup;
use crate::error::ServerError;
use crate::state::AppState;
use crate::store::optstore::ScopedStore;
use crate::store::repo::registrations;

#[derive(Deserialize, Default)]
pub struct RegisterBody {
    /// Which optimization: `parallelism`, `hmp`, `omp`, `pushdown`.
    pub name: String,
    /// `before`, `after` or `both`. Defaults to the optimization author's
    /// choice, which is what the caller wants unless they have a reason.
    #[serde(default)]
    pub step_phase: Option<String>,
    /// Settings to register under, layered over the DAG's stored
    /// configuration. Pinned at registration: a search whose parameters
    /// changed halfway through would be comparing measurements taken under
    /// different rules.
    #[serde(default)]
    pub config: Option<serde_json::Value>,
}

#[derive(Serialize)]
pub struct RegistrationView {
    pub dag: String,
    pub name: String,
    pub optimization_type: String,
    pub step_phase: String,
    /// Tables this optimization created to keep its state in. Empty for one
    /// that keeps none.
    pub tables: Vec<String>,
    /// False once the optimization has converged. It stays registered so its
    /// state and history remain readable; it is simply no longer stepped.
    pub active: bool,
    pub result_version: Option<i32>,
    pub config: Option<OptimizerConfig>,
}

impl From<registrations::RegistrationRow> for RegistrationView {
    fn from(row: registrations::RegistrationRow) -> Self {
        Self {
            dag: row.dag_name.clone(),
            name: row.name.clone(),
            optimization_type: row.optimization_type.clone(),
            step_phase: row.step_phase.clone(),
            tables: row.tables.clone(),
            active: row.is_active(),
            result_version: row.result_version,
            config: row.config.clone(),
        }
    }
}

pub async fn register(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(body): Json<RegisterBody>,
) -> Result<(StatusCode, Json<RegistrationView>), ServerError> {
    let dag = lookup(&state, &name).await?;

    let info = registry::info(&body.name).ok_or_else(|| {
        ServerError::BadRequest(format!(
            "no optimization named '{}'; dee has {}",
            body.name,
            registry::names().join(", ")
        ))
    })?;

    let step_phase = match &body.step_phase {
        Some(raw) => raw.parse::<StepPhase>().map_err(ServerError::BadRequest)?,
        None => info.default_step_phase,
    };

    let mut config = crate::api::optimize::resolve_config(dag.optimizer_config.clone(), body.config)?;
    crate::api::reject_server_side_paths(&config)?;
    // Pass selection is decided by which optimizations are registered, so the
    // flags that mean "also run this one" would be a second, contradictory
    // answer to the same question. Driven off the registry rather than a list
    // of flags, so an optimization added there cannot be left switched on here
    // by a stored config that happened to mention it.
    for pass in registry::names() {
        config.set_pass(pass, pass == body.name);
    }

    // Let the optimization build whatever it keeps state in. Doing this before
    // recording the registration means a registration never names tables that
    // do not exist.
    let target = dag.default_target.clone().ok_or_else(|| {
        ServerError::BadRequest(format!(
            "'{name}' has no target; set one before registering an optimization on it"
        ))
    })?;
    let store = ScopedStore::new(state.store.clone(), &body.name);
    let tables = crate::exec::stepper::registration(
        &state,
        &store,
        &body.name,
        &dag.dag_id,
        &name,
        &target,
        &config,
        crate::exec::stepper::Registering::Setup,
    )
    .await?;

    registrations::upsert(
        &state.store,
        registrations::Register {
            dag_id: dag.dag_id.clone(),
            name: body.name.clone(),
            optimization_type: info.optimization_type,
            step_phase,
            config: config.clone(),
            tables: tables.clone(),
        },
    )
    .await?;

    Ok((
        StatusCode::CREATED,
        Json(RegistrationView {
            dag: name,
            name: body.name,
            optimization_type: info.optimization_type.as_str().to_string(),
            step_phase: step_phase.as_str().to_string(),
            tables,
            active: true,
            result_version: None,
            config: Some(config),
        }),
    ))
}

pub async fn list(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<Vec<RegistrationView>>, ServerError> {
    let dag = lookup(&state, &name).await?;
    Ok(Json(
        registrations::for_dag(&state.store, dag.dag_id)
            .await?
            .into_iter()
            .map(RegistrationView::from)
            .collect(),
    ))
}

#[derive(Deserialize)]
pub struct PhaseBody {
    /// `before`, `after` or `both`.
    pub step_phase: String,
}

/// Change when a registered optimization steps.
pub async fn set_phase(
    State(state): State<AppState>,
    Path((name, optimization)): Path<(String, String)>,
    Json(body): Json<PhaseBody>,
) -> Result<Json<RegistrationView>, ServerError> {
    let dag = lookup(&state, &name).await?;
    let phase = body
        .step_phase
        .parse::<StepPhase>()
        .map_err(ServerError::BadRequest)?;

    if !registrations::set_step_phase(
        &state.store,
        dag.dag_id.clone(),
        optimization.clone(),
        phase,
    )
    .await?
    {
        return Err(ServerError::NotFound(
            "registered optimization",
            format!("{optimization} on {name}"),
        ));
    }

    let row = registrations::get(&state.store, dag.dag_id, optimization)
        .await?
        .ok_or_else(|| ServerError::NotFound("registered optimization", name.clone()))?;
    Ok(Json(RegistrationView::from(row)))
}

#[derive(Serialize)]
pub struct Deregistered {
    pub dag: String,
    pub name: String,
    /// Tables that were torn down. Empty for an optimization that kept no
    /// state -- which is not the same as one whose teardown did nothing.
    pub tables: Vec<String>,
}

pub async fn deregister(
    State(state): State<AppState>,
    Path((name, optimization)): Path<(String, String)>,
) -> Result<Json<Deregistered>, ServerError> {
    let dag = lookup(&state, &name).await?;
    let row = registrations::get(&state.store, dag.dag_id.clone(), optimization.clone())
        .await?
        .ok_or_else(|| {
            ServerError::NotFound(
                "registered optimization",
                format!("{optimization} on {name}"),
            )
        })?;

    let config = row.config.clone().unwrap_or_default();
    let target = dag.default_target.clone().ok_or_else(|| {
        ServerError::BadRequest(format!("'{name}' has no target to tear its state down against"))
    })?;
    let store = ScopedStore::new(state.store.clone(), &optimization);
    let tables = crate::exec::stepper::registration(
        &state,
        &store,
        &optimization,
        &dag.dag_id,
        &name,
        &target,
        &config,
        crate::exec::stepper::Registering::Teardown,
    )
    .await?;

    registrations::remove(&state.store, dag.dag_id, optimization.clone()).await?;

    Ok(Json(Deregistered {
        dag: name,
        name: optimization,
        tables,
    }))
}

/// Every optimization dee can register, with the facts that decide how it
/// behaves. Clients read this rather than hard-coding the list.
#[derive(Serialize)]
pub struct AvailableOptimization {
    pub name: &'static str,
    pub optimization_type: &'static str,
    pub default_step_phase: &'static str,
    pub doc: &'static str,
}

pub async fn available() -> Json<Vec<AvailableOptimization>> {
    Json(
        registry::OPTIMIZATIONS
            .iter()
            .map(|o| AvailableOptimization {
                name: o.name,
                optimization_type: o.optimization_type.as_str(),
                default_step_phase: o.default_step_phase.as_str(),
                doc: o.doc,
            })
            .collect(),
    )
}
