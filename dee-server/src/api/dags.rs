use axum::Json;
use axum::extract::{Path, Query, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use dee::dag::Dag;
use dee::file::DagFile;
use serde::{Deserialize, Serialize};

use crate::error::ServerError;
use crate::state::AppState;
use crate::store::repo::{connections, dags};

#[derive(Deserialize)]
pub struct SubmitDag {
    pub name: String,
    pub definition: DagFile,
    #[serde(default)]
    pub target: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
}

#[derive(Serialize)]
pub struct SubmitResult {
    pub name: String,
    pub dag_id: String,
    pub version: i32,
    pub content_hash: String,
    /// False when the definition matched a version this DAG already had.
    pub created: bool,
    /// Non-fatal problems with the definition. `Dag::try_from` only logs
    /// these, so without surfacing them here they would reach nobody.
    pub warnings: Vec<String>,
}

pub async fn submit(
    State(state): State<AppState>,
    Json(body): Json<SubmitDag>,
) -> Result<(StatusCode, Json<SubmitResult>), ServerError> {
    validate_name(&body.name)?;

    if let Some(target) = &body.target {
        if connections::get(&state.store, target.clone())
            .await?
            .is_none()
        {
            return Err(ServerError::BadRequest(format!(
                "no connection named '{target}'; register it with `dee connection add`"
            )));
        }
    }

    let warnings = inspect(&body.definition)?;

    let submitted = dags::submit(
        &state.store,
        body.name.clone(),
        body.definition,
        body.target,
        body.description,
        dags::Origin::Submitted,
        None,
        None,
    )
    .await?;

    let status = if submitted.created {
        StatusCode::CREATED
    } else {
        StatusCode::OK
    };
    Ok((
        status,
        Json(SubmitResult {
            name: body.name,
            dag_id: submitted.dag_id,
            version: submitted.version,
            content_hash: submitted.content_hash,
            created: submitted.created,
            warnings,
        }),
    ))
}

pub async fn list(State(state): State<AppState>) -> Result<Json<Vec<dags::DagRow>>, ServerError> {
    Ok(Json(dags::list(&state.store).await?))
}

pub async fn get(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<dags::DagRow>, ServerError> {
    Ok(Json(lookup(&state, &name).await?))
}

pub async fn versions(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<Vec<dags::DagVersionRow>>, ServerError> {
    let dag = lookup(&state, &name).await?;
    Ok(Json(dags::versions(&state.store, dag.dag_id).await?))
}

#[derive(Serialize)]
pub struct VersionDetail {
    #[serde(flatten)]
    pub dag: dags::DagRow,
    pub version: i32,
    pub definition: DagFile,
    pub nodes: Vec<dags::DagNodeRow>,
}

pub async fn version(
    State(state): State<AppState>,
    Path((name, version)): Path<(String, i32)>,
) -> Result<Json<VersionDetail>, ServerError> {
    let dag = lookup(&state, &name).await?;
    let definition = dags::definition(&state.store, dag.dag_id.clone(), version)
        .await?
        .ok_or_else(|| ServerError::NotFound("dag version", format!("{name} v{version}")))?;
    let nodes = dags::nodes(&state.store, dag.dag_id.clone(), version).await?;
    Ok(Json(VersionDetail {
        dag,
        version,
        definition,
        nodes,
    }))
}

pub async fn delete(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<StatusCode, ServerError> {
    let dag = lookup(&state, &name).await?;
    dags::delete(&state.store, dag.dag_id).await?;
    Ok(StatusCode::NO_CONTENT)
}

#[derive(Deserialize)]
pub struct GraphQuery {
    #[serde(default)]
    pub format: Option<String>,
    #[serde(default)]
    pub version: Option<i32>,
}

/// Render a version's graph. `svg` and `dot` reuse the library's renderers, so
/// what the API serves and what `dee draw` produces locally are the same code.
pub async fn graph(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Query(query): Query<GraphQuery>,
) -> Result<Response, ServerError> {
    let dag_row = lookup(&state, &name).await?;
    let version = query.version.unwrap_or(dag_row.current_version);
    let definition = dags::definition(&state.store, dag_row.dag_id.clone(), version)
        .await?
        .ok_or_else(|| ServerError::NotFound("dag version", format!("{name} v{version}")))?;

    let source_names: Vec<String> = definition.sources.iter().map(|s| s.name.clone()).collect();
    let dag = Dag::try_from(definition)
        .map_err(|e| ServerError::Internal(format!("stored dag no longer parses: {e}")))?;

    Ok(match query.format.as_deref().unwrap_or("svg") {
        "dot" => (
            [(header::CONTENT_TYPE, "text/vnd.graphviz; charset=utf-8")],
            dag.nodes.draw(),
        )
            .into_response(),
        "svg" => (
            [(header::CONTENT_TYPE, "image/svg+xml")],
            dag.nodes.draw_svg(&source_names),
        )
            .into_response(),
        other => {
            return Err(ServerError::BadRequest(format!(
                "unknown graph format '{other}'; use 'svg' or 'dot'"
            )));
        }
    })
}

pub(crate) async fn lookup(state: &AppState, name: &str) -> Result<dags::DagRow, ServerError> {
    dags::get(&state.store, name.to_string())
        .await?
        .ok_or_else(|| ServerError::NotFound("dag", name.to_string()))
}

/// Reject a definition that cannot run, and report the things that are legal
/// but probably not what the author meant.
fn inspect(definition: &DagFile) -> Result<Vec<String>, ServerError> {
    // `Dag::try_from` runs `Graph::check`, so a dangling dependency fails here
    // rather than at the first scheduled run, hours later.
    let dag = Dag::try_from(definition.clone())
        .map_err(|e| ServerError::BadRequest(format!("invalid dag: {e}")))?;

    let mut warnings = Vec::new();
    for sink in dag.nodes.sinks() {
        if let Some(node) = dag.nodes.get(sink.clone()) {
            if node.materialize == dee::dag::MaterializeMode::TempTable {
                warnings.push(format!(
                    "'{sink}' is a sink materialized as a temp table, so its output does not \
                     outlive the run that produces it"
                ));
            }
        }
    }
    Ok(warnings)
}

fn validate_name(name: &str) -> Result<(), ServerError> {
    if name.is_empty() {
        return Err(ServerError::BadRequest("dag name is empty".into()));
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.')
    {
        return Err(ServerError::BadRequest(format!(
            "dag name '{name}' may only contain letters, digits, '_', '-' and '.'"
        )));
    }
    Ok(())
}
