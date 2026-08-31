use axum::Json;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use chrono::Utc;
use dee::connections::Connection;
use serde::{Deserialize, Serialize};

use crate::error::ServerError;
use crate::state::AppState;
use crate::store::repo::connections::{self, ConnectionView};

#[derive(Deserialize)]
pub struct CreateConnection {
    pub name: String,
    pub config: Connection,
}

#[derive(Deserialize, Default)]
pub struct UpsertQuery {
    #[serde(default)]
    pub upsert: bool,
}

#[derive(Serialize)]
pub struct CreatedConnection {
    name: String,
    kind: String,
    replaced: bool,
}

pub async fn create(
    State(state): State<AppState>,
    Query(query): Query<UpsertQuery>,
    Json(body): Json<CreateConnection>,
) -> Result<(StatusCode, Json<CreatedConnection>), ServerError> {
    validate_name(&body.name)?;
    reject_metadata_database(&state, &body.config)?;

    let exists = connections::get(&state.store, body.name.clone())
        .await?
        .is_some();
    if exists && !query.upsert {
        return Err(ServerError::Conflict(format!(
            "connection '{}' already exists; set upsert to replace it",
            body.name
        )));
    }

    let kind = connections::kind_of(&body.config)?;
    let replaced = connections::upsert(&state.store, body.name.clone(), body.config).await?;

    // A replaced connection's cached pool still points at the old database, so
    // it must go before anything can use the new configuration.
    if replaced {
        state.connectors.invalidate(&body.name).await;
    }

    let status = if replaced {
        StatusCode::OK
    } else {
        StatusCode::CREATED
    };
    Ok((
        status,
        Json(CreatedConnection {
            name: body.name,
            kind,
            replaced,
        }),
    ))
}

pub async fn list(State(state): State<AppState>) -> Result<Json<Vec<ConnectionView>>, ServerError> {
    let rows = connections::list(&state.store).await?;
    let views = rows
        .iter()
        .map(|r| r.redacted())
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Json(views))
}

pub async fn get(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<ConnectionView>, ServerError> {
    let row = connections::get(&state.store, name.clone())
        .await?
        .ok_or_else(|| ServerError::NotFound("connection", name))?;
    Ok(Json(row.redacted()?))
}

pub async fn delete(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<StatusCode, ServerError> {
    let referenced = connections::referenced_by(&state.store, name.clone()).await?;
    if !referenced.is_empty() {
        return Err(ServerError::Conflict(format!(
            "connection '{name}' is still the target of: {}",
            referenced.join(", ")
        )));
    }
    if !connections::delete(&state.store, name.clone()).await? {
        return Err(ServerError::NotFound("connection", name));
    }
    state.connectors.invalidate(&name).await;
    Ok(StatusCode::NO_CONTENT)
}

#[derive(Serialize)]
pub struct TestResult {
    ok: bool,
    latency_ms: i64,
    time_basis: Option<String>,
    error: Option<String>,
}

/// Build (or reuse) the connector and run a trivial query against it.
///
/// A pool built here stays cached, so the first real run does not pay for it
/// again -- `dee connection test` doubles as a warm-up.
pub async fn test(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<TestResult>, ServerError> {
    use dee::connectors::Connector;

    let row = connections::get(&state.store, name.clone())
        .await?
        .ok_or_else(|| ServerError::NotFound("connection", name))?;

    let started = Utc::now();
    let handle = match state.connectors.acquire(&row).await {
        Ok(handle) => handle,
        // A connection that cannot be reached is a fact about the world, not a
        // malformed request, so it is reported in the body with ok:false
        // rather than as an HTTP error.
        Err(e) => {
            return Ok(Json(TestResult {
                ok: false,
                latency_ms: (Utc::now() - started).num_milliseconds(),
                time_basis: None,
                error: Some(e.to_string()),
            }));
        }
    };

    let probe = match &handle {
        crate::exec::connectors::ConnectorHandle::DuckDb(c) => {
            c.execute("SELECT 1".to_string()).await.err().map(|e| e.to_string())
        }
        crate::exec::connectors::ConnectorHandle::Postgres(c) => {
            c.execute("SELECT 1".to_string()).await.err().map(|e| e.to_string())
        }
    };

    Ok(Json(TestResult {
        ok: probe.is_none(),
        latency_ms: (Utc::now() - started).num_milliseconds(),
        time_basis: Some(handle.time_basis().to_string()),
        error: probe,
    }))
}

fn validate_name(name: &str) -> Result<(), ServerError> {
    if name.is_empty() {
        return Err(ServerError::BadRequest("connection name is empty".into()));
    }
    // Names appear in URL paths, so keep them to characters that need no
    // escaping and cannot be confused with a path segment.
    if !name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.')
    {
        return Err(ServerError::BadRequest(format!(
            "connection name '{name}' may only contain letters, digits, '_', '-' and '.'"
        )));
    }
    Ok(())
}

/// Refuse a DuckDB connection pointed at the server's own metadata database.
///
/// The data connector and the metadata store use the same crate, and DuckDB
/// permits one read-write process per file. Accepting this would either fail
/// confusingly at run time or corrupt the server's own state.
fn reject_metadata_database(state: &AppState, config: &Connection) -> Result<(), ServerError> {
    let Connection::DuckDB(duck) = config else {
        return Ok(());
    };
    let target = std::fs::canonicalize(&duck.database).unwrap_or_else(|_| duck.database.clone());
    let metadata = std::fs::canonicalize(&state.config.metadata_db)
        .unwrap_or_else(|_| state.config.metadata_db.clone());
    if target == metadata {
        return Err(ServerError::BadRequest(format!(
            "'{}' is this server's metadata database and cannot also be a warehouse",
            duck.database.display()
        )));
    }
    Ok(())
}
