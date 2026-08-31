//! Named connection targets.
//!
//! The stored `config` is the whole serde-tagged `dee::connections::Connection`
//! JSON. Deserializing it is the only decode path, so adding a connector
//! variant to the library needs no change here and no migration.

use chrono::{DateTime, Utc};
use dee::connections::Connection;
use serde::Serialize;
use serde_json::Value;

use crate::hash::content_hash;
use crate::store::{Store, StoreError};

#[derive(Debug, Clone)]
pub struct ConnectionRow {
    pub name: String,
    pub kind: String,
    /// The tagged `Connection` JSON, with credentials intact. Never put this
    /// in a response; use [`ConnectionRow::redacted`].
    pub config: String,
    pub config_hash: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// What a client is allowed to see. Credentials are removed, because a
/// connection is readable by anyone who can reach the API while the password
/// only ever needs to travel inward.
#[derive(Debug, Clone, Serialize)]
pub struct ConnectionView {
    pub name: String,
    pub kind: String,
    pub config: Value,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Keys whose values are replaced before a config leaves the server.
const SECRET_KEYS: &[&str] = &["password", "secret", "token"];

pub const REDACTED: &str = "\u{2022}\u{2022}\u{2022}redacted\u{2022}\u{2022}\u{2022}";

/// Replace secret-looking values anywhere in `config`.
///
/// This works on the JSON rather than the typed struct deliberately:
/// `PostgresConfig`'s fields are private with no accessors, so there is no way
/// to rebuild a redacted copy through the type. Walking the JSON also means a
/// credential added to a future connector variant is redacted without anyone
/// remembering to update this list's call sites.
pub fn redact(config: &Value) -> Value {
    match config {
        Value::Object(map) => {
            let mut out = serde_json::Map::new();
            for (key, value) in map {
                let is_secret = SECRET_KEYS
                    .iter()
                    .any(|s| key.to_ascii_lowercase().contains(s));
                if is_secret && !value.is_null() {
                    out.insert(key.clone(), Value::String(REDACTED.to_string()));
                } else {
                    out.insert(key.clone(), redact(value));
                }
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(items.iter().map(redact).collect()),
        other => other.clone(),
    }
}

impl ConnectionRow {
    pub fn redacted(&self) -> Result<ConnectionView, StoreError> {
        let config: Value = serde_json::from_str(&self.config).map_err(|source| {
            StoreError::Decode {
                what: "connection config",
                source,
            }
        })?;
        Ok(ConnectionView {
            name: self.name.clone(),
            kind: self.kind.clone(),
            config: redact(&config),
            created_at: self.created_at,
            updated_at: self.updated_at,
        })
    }

    /// The typed connection, for actually building a pool.
    pub fn connection(&self) -> Result<Connection, StoreError> {
        serde_json::from_str(&self.config).map_err(|source| StoreError::Decode {
            what: "connection config",
            source,
        })
    }
}

/// The `type` tag of a `Connection`, without a match that would need updating
/// for every new variant.
pub fn kind_of(connection: &Connection) -> Result<String, StoreError> {
    let value = serde_json::to_value(connection).map_err(|source| StoreError::Decode {
        what: "connection config",
        source,
    })?;
    Ok(value
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string())
}

const SELECT: &str = "SELECT name, kind, config, config_hash, created_at, updated_at
                      FROM connections";

fn row_from(row: &duckdb::Row<'_>) -> duckdb::Result<ConnectionRow> {
    Ok(ConnectionRow {
        name: row.get(0)?,
        kind: row.get(1)?,
        config: row.get(2)?,
        config_hash: row.get(3)?,
        created_at: row.get(4)?,
        updated_at: row.get(5)?,
    })
}

/// Insert or replace a connection.
///
/// Returns whether a row already existed, so the API can answer 201 versus 200
/// and refuse a duplicate when the caller did not ask to overwrite.
pub async fn upsert(
    store: &Store,
    name: String,
    connection: Connection,
) -> Result<bool, StoreError> {
    let config = serde_json::to_value(&connection).map_err(|source| StoreError::Decode {
        what: "connection config",
        source,
    })?;
    let kind = kind_of(&connection)?;
    let hash = content_hash(&config);
    let config = config.to_string();
    let now = Utc::now();

    store
        .write(move |conn| {
            let existed: i64 = conn.query_row(
                "SELECT count(*) FROM connections WHERE name = ?",
                duckdb::params![name],
                |r| r.get(0),
            )?;
            if existed > 0 {
                conn.execute(
                    "UPDATE connections
                     SET kind = ?, config = ?, config_hash = ?, updated_at = ?
                     WHERE name = ?",
                    duckdb::params![kind, config, hash, now, name],
                )?;
            } else {
                conn.execute(
                    "INSERT INTO connections
                        (name, kind, config, config_hash, created_at, updated_at)
                     VALUES (?, ?, ?, ?, ?, ?)",
                    duckdb::params![name, kind, config, hash, now, now],
                )?;
            }
            Ok(existed > 0)
        })
        .await
}

pub async fn get(store: &Store, name: String) -> Result<Option<ConnectionRow>, StoreError> {
    store
        .read(move |conn| {
            let mut stmt = conn.prepare(&format!("{SELECT} WHERE name = ?"))?;
            let mut rows = stmt.query_map(duckdb::params![name], row_from)?;
            match rows.next() {
                Some(row) => Ok(Some(row?)),
                None => Ok(None),
            }
        })
        .await
}

pub async fn list(store: &Store) -> Result<Vec<ConnectionRow>, StoreError> {
    store
        .read(|conn| {
            let mut stmt = conn.prepare(&format!("{SELECT} ORDER BY name"))?;
            let rows = stmt.query_map([], row_from)?;
            Ok(rows.collect::<duckdb::Result<Vec<_>>>()?)
        })
        .await
}

/// Names of DAGs or schedules that would be left pointing at nothing if this
/// connection went away.
pub async fn referenced_by(store: &Store, name: String) -> Result<Vec<String>, StoreError> {
    store
        .read(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT DISTINCT d.name
                 FROM dags d LEFT JOIN schedules s ON s.dag_id = d.dag_id
                 WHERE d.default_target = ?1 OR s.target = ?1
                 ORDER BY d.name",
            )?;
            let rows = stmt.query_map(duckdb::params![name], |r| r.get::<_, String>(0))?;
            Ok(rows.collect::<duckdb::Result<Vec<_>>>()?)
        })
        .await
}

pub async fn delete(store: &Store, name: String) -> Result<bool, StoreError> {
    store
        .write(move |conn| {
            let n = conn.execute(
                "DELETE FROM connections WHERE name = ?",
                duckdb::params![name],
            )?;
            Ok(n > 0)
        })
        .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn duckdb_connection(path: &str) -> Connection {
        serde_json::from_value(json!({
            "type": "duckdb", "database": path, "num_connections": 2
        }))
        .unwrap()
    }

    fn postgres_connection(password: &str) -> Connection {
        serde_json::from_value(json!({
            "type": "postgres", "host": "h", "port": 5432, "user": "u",
            "password": password, "database": "d", "num_connections": 4
        }))
        .unwrap()
    }

    #[test]
    fn test_redact_removes_credentials_and_keeps_everything_else() {
        let config = json!({
            "type": "postgres", "host": "db.internal", "user": "runner",
            "password": "hunter2", "port": 5432
        });
        let redacted = redact(&config);

        assert_eq!(redacted["password"], json!(REDACTED));
        assert_eq!(redacted["host"], json!("db.internal"));
        assert_eq!(redacted["user"], json!("runner"));
        assert_eq!(redacted["port"], json!(5432));
    }

    #[test]
    fn test_redact_reaches_nested_and_repeated_values() {
        let config = json!({
            "outer": {"password": "a", "inner": [{"api_token": "b"}]},
            "note": "no secret here",
        });
        let redacted = redact(&config);

        assert_eq!(redacted["outer"]["password"], json!(REDACTED));
        assert_eq!(redacted["outer"]["inner"][0]["api_token"], json!(REDACTED));
        assert_eq!(redacted["note"], json!("no secret here"));
    }

    #[test]
    fn test_redact_leaves_an_absent_credential_absent() {
        // Writing a placeholder over a null would make an unset password look
        // like a set one.
        let redacted = redact(&json!({"password": null}));
        assert_eq!(redacted["password"], json!(null));
    }

    #[test]
    fn test_redacted_view_never_carries_the_real_password() {
        let row = ConnectionRow {
            name: "pg".into(),
            kind: "postgres".into(),
            config: serde_json::to_string(&postgres_connection("hunter2")).unwrap(),
            config_hash: "h".into(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        let serialized = serde_json::to_string(&row.redacted().unwrap()).unwrap();
        assert!(!serialized.contains("hunter2"), "{serialized}");
    }

    #[test]
    fn test_kind_comes_from_the_serde_tag() {
        assert_eq!(kind_of(&duckdb_connection("w.duckdb")).unwrap(), "duckdb");
        assert_eq!(kind_of(&postgres_connection("p")).unwrap(), "postgres");
    }

    #[tokio::test]
    async fn test_upsert_reports_whether_it_replaced_and_updates_the_hash() {
        let store = crate::store::Store::open_temporary().unwrap();

        assert!(!upsert(&store, "wh".into(), duckdb_connection("a.duckdb")).await.unwrap());
        let first = get(&store, "wh".into()).await.unwrap().unwrap();

        assert!(upsert(&store, "wh".into(), duckdb_connection("b.duckdb")).await.unwrap());
        let second = get(&store, "wh".into()).await.unwrap().unwrap();

        // The hash keying the connector cache must move, or the server would
        // keep using a pool pointed at the previous database.
        assert_ne!(first.config_hash, second.config_hash);
        assert_eq!(list(&store).await.unwrap().len(), 1, "upsert must not duplicate");
    }

    #[tokio::test]
    async fn test_stored_config_round_trips_back_into_a_typed_connection() {
        let store = crate::store::Store::open_temporary().unwrap();
        upsert(&store, "pg".into(), postgres_connection("hunter2")).await.unwrap();

        let row = get(&store, "pg".into()).await.unwrap().unwrap();
        // The password has to survive storage intact: redaction is for
        // responses, not for what the server connects with.
        assert!(matches!(row.connection().unwrap(), Connection::Postgres(_)));
        assert!(row.config.contains("hunter2"));
    }

    #[tokio::test]
    async fn test_referenced_by_finds_dags_and_schedules_pointing_at_a_target() {
        let store = crate::store::Store::open_temporary().unwrap();
        upsert(&store, "wh".into(), duckdb_connection("w.duckdb")).await.unwrap();

        store
            .write(|c| {
                c.execute(
                    "INSERT INTO dags (dag_id, name, current_version, default_target,
                                       created_at, updated_at)
                     VALUES ('d1', 'churn', 1, 'wh', now(), now()),
                            ('d2', 'other', 1, NULL, now(), now())",
                    [],
                )?;
                c.execute(
                    "INSERT INTO schedules (dag_id, cron, target, created_at, updated_at)
                     VALUES ('d2', '0 * * * *', 'wh', now(), now())",
                    [],
                )?;
                Ok(())
            })
            .await
            .unwrap();

        // Both routes to a target count: the DAG's default and a schedule's
        // override. Deleting the connection would strand either.
        let referenced = referenced_by(&store, "wh".into()).await.unwrap();
        assert_eq!(referenced, vec!["churn".to_string(), "other".to_string()]);
        assert!(referenced_by(&store, "unused".into()).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_delete_reports_whether_anything_was_removed() {
        let store = crate::store::Store::open_temporary().unwrap();
        upsert(&store, "wh".into(), duckdb_connection("w.duckdb")).await.unwrap();

        assert!(delete(&store, "wh".into()).await.unwrap());
        assert!(!delete(&store, "wh".into()).await.unwrap());
        assert!(get(&store, "wh".into()).await.unwrap().is_none());
    }
}
