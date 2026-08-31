//! The DAG registry: named DAGs, their immutable versioned definitions, and
//! the exploded node table derived from each version.

use chrono::{DateTime, Utc};
use dee::dag::Dag;
use dee::file::DagFile;
use serde::Serialize;

use crate::hash::dag_hash;
use crate::store::{LIST_PARAM, Store, StoreError, list_param, new_id, parse_list};

/// How a version came to exist. `Optimized` versions carry the version they
/// were derived from, which is what makes "did the optimizer's rewrite
/// actually run faster" answerable from history alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    Submitted,
    Optimized,
    Converted,
}

impl Origin {
    pub fn as_str(&self) -> &'static str {
        match self {
            Origin::Submitted => "submitted",
            Origin::Optimized => "optimized",
            Origin::Converted => "converted",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct DagRow {
    pub dag_id: String,
    pub name: String,
    pub description: Option<String>,
    pub current_version: i32,
    pub default_target: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DagVersionRow {
    pub version: i32,
    pub content_hash: String,
    pub sql_dialect: Option<String>,
    pub node_count: i32,
    pub source_count: i32,
    pub origin: String,
    pub derived_from_version: Option<i32>,
    pub optimization_id: Option<String>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DagNodeRow {
    pub node_id: String,
    pub materialize: String,
    pub query_text: String,
    pub depends_on: Vec<String>,
    /// How many nodes read this one.
    pub out_degree: i32,
    /// Materialization points (Table/TempTable nodes) reachable from here,
    /// via `Graph::paths_to_sinks` -- the value OMP filters and ranks
    /// candidates by, so it explains the optimizer's choices rather than
    /// approximating them.
    pub paths_to_sinks: i32,
}

/// The outcome of a submission.
pub struct Submitted {
    pub dag_id: String,
    pub version: i32,
    pub content_hash: String,
    /// False when the definition matched an existing version, which is then
    /// returned unchanged.
    pub created: bool,
}

const DAG_SELECT: &str = "SELECT dag_id, name, description, current_version, default_target,
                                 created_at, updated_at FROM dags";

fn dag_from(row: &duckdb::Row<'_>) -> duckdb::Result<DagRow> {
    Ok(DagRow {
        dag_id: row.get(0)?,
        name: row.get(1)?,
        description: row.get(2)?,
        current_version: row.get(3)?,
        default_target: row.get(4)?,
        created_at: row.get(5)?,
        updated_at: row.get(6)?,
    })
}

pub async fn get(store: &Store, name: String) -> Result<Option<DagRow>, StoreError> {
    store
        .read(move |conn| {
            let mut stmt = conn.prepare(&format!("{DAG_SELECT} WHERE name = ?"))?;
            let mut rows = stmt.query_map(duckdb::params![name], dag_from)?;
            rows.next().transpose().map_err(StoreError::from)
        })
        .await
}

pub async fn list(store: &Store) -> Result<Vec<DagRow>, StoreError> {
    store
        .read(|conn| {
            let mut stmt = conn.prepare(&format!("{DAG_SELECT} ORDER BY name"))?;
            Ok(stmt.query_map([], dag_from)?.collect::<duckdb::Result<Vec<_>>>()?)
        })
        .await
}

/// Register a definition under `name`, creating the DAG if it is new.
///
/// Idempotent by content: if `definition` hashes to a version this DAG already
/// has, that version is returned and nothing is written. The benchmark harness
/// and any CI resubmit constantly, and without this each would grow the
/// history by one identical version per invocation.
pub async fn submit(
    store: &Store,
    name: String,
    definition: DagFile,
    target: Option<String>,
    description: Option<String>,
    origin: Origin,
    derived_from_version: Option<i32>,
    optimization_id: Option<String>,
) -> Result<Submitted, StoreError> {
    let content_hash = dag_hash(&definition).map_err(|source| StoreError::Decode {
        what: "dag definition",
        source,
    })?;
    let sql_dialect = definition
        .metadata
        .as_ref()
        .and_then(|m| m.sql_dialect.clone());
    let source_count = definition.sources.len() as i32;
    let node_count = definition.nodes.len() as i32;

    // Build the graph once, outside the write, so the transaction holds the
    // store only for as long as the inserts take.
    let nodes = explode(&definition)?;
    let sources = definition
        .sources
        .iter()
        .map(|s| {
            let columns = serde_json::to_string(&s.columns).map_err(|source| StoreError::Decode {
                what: "dag source columns",
                source,
            })?;
            Ok((s.name.clone(), columns))
        })
        .collect::<Result<Vec<_>, StoreError>>()?;
    let body = serde_json::to_string(&definition).map_err(|source| StoreError::Decode {
        what: "dag definition",
        source,
    })?;

    let now = Utc::now();
    let origin = origin.as_str();

    store
        .write(move |conn| {
            conn.execute_batch("BEGIN TRANSACTION;")?;
            let result = (|| -> Result<Submitted, StoreError> {
                let existing: Option<(String, i32)> = {
                    let mut stmt =
                        conn.prepare("SELECT dag_id, current_version FROM dags WHERE name = ?")?;
                    let mut rows = stmt.query_map(duckdb::params![name], |r| {
                        Ok((r.get::<_, String>(0)?, r.get::<_, i32>(1)?))
                    })?;
                    rows.next().transpose()?
                };

                let (dag_id, next_version) = match &existing {
                    Some((dag_id, current)) => {
                        // Same content as a version we already hold: hand that
                        // one back rather than recording a duplicate.
                        let matched: Option<i32> = {
                            let mut stmt = conn.prepare(
                                "SELECT version FROM dag_versions
                                 WHERE dag_id = ? AND content_hash = ?",
                            )?;
                            let mut rows = stmt.query_map(
                                duckdb::params![dag_id, content_hash],
                                |r| r.get::<_, i32>(0),
                            )?;
                            rows.next().transpose()?
                        };
                        if let Some(version) = matched {
                            if let Some(target) = &target {
                                conn.execute(
                                    "UPDATE dags SET default_target = ?, updated_at = ?
                                     WHERE dag_id = ?",
                                    duckdb::params![target, now, dag_id],
                                )?;
                            }
                            conn.execute_batch("COMMIT;")?;
                            return Ok(Submitted {
                                dag_id: dag_id.clone(),
                                version,
                                content_hash,
                                created: false,
                            });
                        }
                        (dag_id.clone(), current + 1)
                    }
                    None => (new_id(), 1),
                };

                if existing.is_none() {
                    conn.execute(
                        "INSERT INTO dags (dag_id, name, description, current_version,
                                           default_target, created_at, updated_at)
                         VALUES (?, ?, ?, ?, ?, ?, ?)",
                        duckdb::params![
                            dag_id, name, description, next_version, target, now, now
                        ],
                    )?;
                } else {
                    conn.execute(
                        "UPDATE dags
                         SET current_version = ?,
                             default_target = coalesce(?, default_target),
                             description = coalesce(?, description),
                             updated_at = ?
                         WHERE dag_id = ?",
                        duckdb::params![next_version, target, description, now, dag_id],
                    )?;
                }

                conn.execute(
                    "INSERT INTO dag_versions
                        (dag_id, version, content_hash, definition, sql_dialect, node_count,
                         source_count, origin, derived_from_version, optimization_id, created_at)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    duckdb::params![
                        dag_id,
                        next_version,
                        content_hash,
                        body,
                        sql_dialect,
                        node_count,
                        source_count,
                        origin,
                        derived_from_version,
                        optimization_id,
                        now
                    ],
                )?;

                for node in &nodes {
                    conn.execute(
                        &format!(
                            "INSERT INTO dag_version_nodes
                                (dag_id, version, node_id, materialize, query_text,
                                 depends_on, out_degree, paths_to_sinks)
                             VALUES (?, ?, ?, ?, ?, {LIST_PARAM}, ?, ?)"
                        ),
                        duckdb::params![
                            dag_id,
                            next_version,
                            node.node_id,
                            node.materialize,
                            node.query_text,
                            list_param(&node.depends_on),
                            node.out_degree,
                            node.paths_to_sinks
                        ],
                    )?;
                }
                for (source_name, columns) in &sources {
                    conn.execute(
                        "INSERT INTO dag_version_sources (dag_id, version, name, columns)
                         VALUES (?, ?, ?, ?)",
                        duckdb::params![dag_id, next_version, source_name, columns],
                    )?;
                }

                Ok(Submitted {
                    dag_id,
                    version: next_version,
                    content_hash,
                    created: true,
                })
            })();

            match result {
                Ok(submitted) => {
                    if submitted.created {
                        conn.execute_batch("COMMIT;")?;
                    }
                    Ok(submitted)
                }
                Err(e) => {
                    let _ = conn.execute_batch("ROLLBACK;");
                    Err(e)
                }
            }
        })
        .await
}

/// Derive the per-node structural facts a client would otherwise have to
/// recompute from the definition JSON.
fn explode(definition: &DagFile) -> Result<Vec<DagNodeRow>, StoreError> {
    let dag = Dag::try_from(definition.clone())
        .map_err(|e| StoreError::NotFound(format!("invalid dag: {e}")))?;

    let mut rows: Vec<DagNodeRow> = dag
        .nodes
        .nodes()
        .map(|node| {
            let mut depends_on: Vec<String> = node.depends_on.iter().cloned().collect();
            depends_on.sort();
            DagNodeRow {
                node_id: node.id.clone(),
                materialize: node.materialize.as_str().to_string(),
                query_text: node.query_text.clone(),
                depends_on,
                out_degree: dag.nodes.out_degree(&node.id) as i32,
                paths_to_sinks: dag.nodes.paths_to_sinks(&node.id) as i32,
            }
        })
        .collect();
    rows.sort_by(|a, b| a.node_id.cmp(&b.node_id));
    Ok(rows)
}

pub async fn versions(store: &Store, dag_id: String) -> Result<Vec<DagVersionRow>, StoreError> {
    store
        .read(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT version, content_hash, sql_dialect, node_count, source_count, origin,
                        derived_from_version, optimization_id, created_at
                 FROM dag_versions WHERE dag_id = ? ORDER BY version DESC",
            )?;
            let rows = stmt.query_map(duckdb::params![dag_id], |r| {
                Ok(DagVersionRow {
                    version: r.get(0)?,
                    content_hash: r.get(1)?,
                    sql_dialect: r.get(2)?,
                    node_count: r.get(3)?,
                    source_count: r.get(4)?,
                    origin: r.get(5)?,
                    derived_from_version: r.get(6)?,
                    optimization_id: r.get(7)?,
                    created_at: r.get(8)?,
                })
            })?;
            Ok(rows.collect::<duckdb::Result<Vec<_>>>()?)
        })
        .await
}

/// The stored definition of one version, parsed back into a `DagFile`.
pub async fn definition(
    store: &Store,
    dag_id: String,
    version: i32,
) -> Result<Option<DagFile>, StoreError> {
    let body: Option<String> = store
        .read(move |conn| {
            let mut stmt = conn
                .prepare("SELECT definition FROM dag_versions WHERE dag_id = ? AND version = ?")?;
            let mut rows =
                stmt.query_map(duckdb::params![dag_id, version], |r| r.get::<_, String>(0))?;
            rows.next().transpose().map_err(StoreError::from)
        })
        .await?;

    match body {
        Some(text) => serde_json::from_str(&text)
            .map(Some)
            .map_err(|source| StoreError::Decode {
                what: "dag definition",
                source,
            }),
        None => Ok(None),
    }
}

pub async fn nodes(
    store: &Store,
    dag_id: String,
    version: i32,
) -> Result<Vec<DagNodeRow>, StoreError> {
    store
        .read(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT node_id, materialize, query_text, to_json(depends_on),
                        out_degree, paths_to_sinks
                 FROM dag_version_nodes WHERE dag_id = ? AND version = ? ORDER BY node_id",
            )?;
            let rows = stmt.query_map(duckdb::params![dag_id, version], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, String>(3)?,
                    r.get::<_, i32>(4)?,
                    r.get::<_, i32>(5)?,
                ))
            })?;
            rows.collect::<duckdb::Result<Vec<_>>>()?
                .into_iter()
                .map(|(node_id, materialize, query_text, deps, out_degree, paths_to_sinks)| {
                    Ok(DagNodeRow {
                        node_id,
                        materialize,
                        query_text,
                        depends_on: parse_list(&deps)?,
                        out_degree,
                        paths_to_sinks,
                    })
                })
                .collect()
        })
        .await
}

/// Remove a DAG and everything recorded about it.
///
/// History is deleted with the DAG rather than orphaned: a run row whose DAG no
/// longer exists cannot be interpreted, since the definition it ran is gone.
pub async fn delete(store: &Store, dag_id: String) -> Result<(), StoreError> {
    store
        .write(move |conn| {
            conn.execute_batch("BEGIN TRANSACTION;")?;
            let result = (|| -> Result<(), StoreError> {
                conn.execute(
                    "DELETE FROM node_executions WHERE run_id IN
                        (SELECT run_id FROM runs WHERE dag_id = ?)",
                    duckdb::params![dag_id],
                )?;
                conn.execute(
                    "DELETE FROM plans WHERE run_id IN
                        (SELECT run_id FROM runs WHERE dag_id = ?)",
                    duckdb::params![dag_id],
                )?;
                conn.execute(
                    "DELETE FROM run_samples WHERE run_id IN
                        (SELECT run_id FROM runs WHERE dag_id = ?)",
                    duckdb::params![dag_id],
                )?;
                conn.execute(
                    "DELETE FROM optimization_passes WHERE optimization_id IN
                        (SELECT optimization_id FROM optimizations WHERE dag_id = ?)",
                    duckdb::params![dag_id],
                )?;
                conn.execute(
                    "DELETE FROM optimization_iterations WHERE optimization_id IN
                        (SELECT optimization_id FROM optimizations WHERE dag_id = ?)",
                    duckdb::params![dag_id],
                )?;
                for table in [
                    "events",
                    "optimizations",
                    "runs",
                    "run_groups",
                    "schedule_skips",
                    "schedules",
                    "dag_version_nodes",
                    "dag_version_sources",
                    "dag_versions",
                    "dags",
                ] {
                    conn.execute(
                        &format!("DELETE FROM {table} WHERE dag_id = ?"),
                        duckdb::params![dag_id],
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
        .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Store;

    fn parse(json: &str) -> DagFile {
        serde_json::from_str(json).unwrap()
    }

    /// a -> b -> {c, d}, where b is a table.
    fn diamond() -> DagFile {
        parse(
            r#"{"nodes":[
                {"id":"a","query_text":"select 1","depends_on":[],"materialize":"view"},
                {"id":"b","query_text":"select 2","depends_on":["a"],"materialize":"table"},
                {"id":"c","query_text":"select 3","depends_on":["b"],"materialize":"table"},
                {"id":"d","query_text":"select 4","depends_on":["b"],"materialize":"table"}
            ],"sources":[]}"#,
        )
    }

    async fn store_with(dag: DagFile) -> (Store, Submitted) {
        let store = Store::open_temporary().unwrap();
        let submitted = submit(
            &store,
            "d".into(),
            dag,
            None,
            None,
            Origin::Submitted,
            None,
            None,
        )
        .await
        .unwrap();
        (store, submitted)
    }

    #[tokio::test]
    async fn test_a_first_submission_creates_version_one() {
        let (store, submitted) = store_with(diamond()).await;
        assert_eq!(submitted.version, 1);
        assert!(submitted.created);
        assert_eq!(get(&store, "d".into()).await.unwrap().unwrap().current_version, 1);
    }

    #[tokio::test]
    async fn test_resubmitting_identical_content_returns_the_same_version() {
        // The benchmark harness and CI resubmit constantly; each must not add
        // a version.
        let (store, first) = store_with(diamond()).await;
        let again = submit(
            &store, "d".into(), diamond(), None, None, Origin::Submitted, None, None,
        )
        .await
        .unwrap();

        assert!(!again.created);
        assert_eq!(again.version, first.version);
        assert_eq!(versions(&store, first.dag_id).await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_changed_content_creates_a_new_version() {
        let (store, first) = store_with(diamond()).await;
        let mut changed = diamond();
        changed.nodes[0].query_text = "select 99".into();

        let second = submit(
            &store, "d".into(), changed, None, None, Origin::Submitted, None, None,
        )
        .await
        .unwrap();

        assert!(second.created);
        assert_eq!(second.version, 2);
        assert_ne!(second.content_hash, first.content_hash);
        assert_eq!(get(&store, "d".into()).await.unwrap().unwrap().current_version, 2);
    }

    #[tokio::test]
    async fn test_an_optimized_version_records_what_it_came_from() {
        // This link is what lets a later question -- did the rewrite actually
        // run faster -- be answered from history without re-running anything.
        let (store, first) = store_with(diamond()).await;
        let mut optimized = diamond();
        optimized.nodes[0].materialize = Some("temp_table".into());

        let second = submit(
            &store,
            "d".into(),
            optimized,
            None,
            None,
            Origin::Optimized,
            Some(first.version),
            Some("opt-1".into()),
        )
        .await
        .unwrap();

        let rows = versions(&store, first.dag_id).await.unwrap();
        let newest = rows.iter().find(|v| v.version == second.version).unwrap();
        assert_eq!(newest.origin, "optimized");
        assert_eq!(newest.derived_from_version, Some(1));
        assert_eq!(newest.optimization_id.as_deref(), Some("opt-1"));
    }

    #[tokio::test]
    async fn test_node_facts_are_the_optimizers_own_numbers() {
        let (store, submitted) = store_with(diamond()).await;
        let nodes = nodes(&store, submitted.dag_id, 1).await.unwrap();
        let by_id = |id: &str| nodes.iter().find(|n| n.node_id == id).unwrap().clone();

        assert_eq!(by_id("b").out_degree, 2);
        assert_eq!(by_id("a").out_degree, 1);

        // `Graph::paths_to_sinks` counts reachable materialization points, not
        // paths to childless nodes: from `a` that is b, c and d. A
        // path-counting definition would say 2. OMP filters on this number, so
        // storing the optimizer's version is what makes the column explain its
        // choices.
        assert_eq!(by_id("a").paths_to_sinks, 3);
        assert_eq!(by_id("b").paths_to_sinks, 3);
        assert_eq!(by_id("c").paths_to_sinks, 1);
    }

    #[tokio::test]
    async fn test_dependencies_survive_the_json_list_encoding() {
        let dag = parse(
            r#"{"nodes":[
                {"id":"\"w\".\"m\".\"a\"","query_text":"q","depends_on":[],"materialize":"view"},
                {"id":"z","query_text":"q","depends_on":["\"w\".\"m\".\"a\""],"materialize":"table"}
            ],"sources":[]}"#,
        );
        let (store, submitted) = store_with(dag).await;
        let nodes = nodes(&store, submitted.dag_id, 1).await.unwrap();
        let z = nodes.iter().find(|n| n.node_id == "z").unwrap();
        assert_eq!(z.depends_on, vec![r#""w"."m"."a""#.to_string()]);
    }

    #[tokio::test]
    async fn test_the_stored_definition_round_trips() {
        let (store, submitted) = store_with(diamond()).await;
        let back = super::definition(&store, submitted.dag_id, 1).await.unwrap().unwrap();
        assert_eq!(back.nodes.len(), 4);
        assert_eq!(crate::hash::dag_hash(&back).unwrap(), submitted.content_hash);
    }

    #[tokio::test]
    async fn test_delete_removes_the_dag_and_its_history() {
        let (store, submitted) = store_with(diamond()).await;
        let dag_id = submitted.dag_id.clone();
        store
            .write({
                let dag_id = dag_id.clone();
                move |c| {
                    c.execute(
                        "INSERT INTO runs (run_id, run_group_id, dag_id, dag_version, target,
                                           status, queued_at, instance_id)
                         VALUES ('r', 'g', ?, 1, 't', 'succeeded', now(), 'i')",
                        duckdb::params![dag_id],
                    )?;
                    c.execute(
                        "INSERT INTO node_executions (run_id, node_id, materialize, started_at,
                                                      finished_at, duration_ms)
                         VALUES ('r', 'a', 'view', now(), now(), 1)",
                        [],
                    )?;
                    Ok(())
                }
            })
            .await
            .unwrap();

        delete(&store, dag_id).await.unwrap();

        assert!(get(&store, "d".into()).await.unwrap().is_none());
        // Run history is deleted with the DAG rather than orphaned: a run whose
        // definition is gone cannot be interpreted.
        let leftovers: i64 = store
            .read(|c| {
                Ok(c.query_row(
                    "SELECT (SELECT count(*) FROM runs) + (SELECT count(*) FROM node_executions)
                            + (SELECT count(*) FROM dag_versions)
                            + (SELECT count(*) FROM dag_version_nodes)",
                    [],
                    |r| r.get(0),
                )?)
            })
            .await
            .unwrap();
        assert_eq!(leftovers, 0);
    }
}
