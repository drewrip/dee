use std::collections::HashSet;

use polyglot_sql::dialects::DialectType;

use crate::{
    dag::{Dag, MaterializeMode, TransformNode},
    opt::OptimizerError,
};

// ---------------------------------------------------------------------------
// dialect_for_db — map a DAG sql_dialect string to a polyglot-sql DialectType
// ---------------------------------------------------------------------------

/// Return the polyglot-sql [`DialectType`] for `db`.
///
/// Matches common dialect names case-insensitively. Defaults to
/// [`DialectType::DuckDB`] when the dialect is unknown or empty, because
/// DuckDB is the primary target engine and its dialect is the safest default.
/// Fraction by which a trial may overrun the best configuration known before it
/// is worth abandoning.
///
/// A candidate already slower than the best known setting needs no exact
/// runtime to be rejected, so there is nothing to learn from letting it
/// finish -- only a censored observation ("at least this bad"), which is all
/// the acceptance tests consume. Shared by every search that budgets a trial,
/// so that "how much worse a run may get" is one number rather than one per
/// pass.
pub const DEFAULT_BUDGET_EPS: f64 = 0.25;

pub fn dialect_for_db(db: &str) -> DialectType {
    match db.to_lowercase().as_str() {
        "duckdb" => DialectType::DuckDB,
        "postgresql" | "postgres" => DialectType::PostgreSQL,
        "mysql" => DialectType::MySQL,
        "sqlite" => DialectType::SQLite,
        "bigquery" => DialectType::BigQuery,
        "default" => DialectType::Generic,
        _ => DialectType::DuckDB,
    }
}

// ---------------------------------------------------------------------------
// make_temp
// ---------------------------------------------------------------------------

/// Safely rewrite `dag` so that `view_name` can be backed by a TempTable
/// without creating any `TempTable → View` edges.
///
/// Algorithm:
/// 1. Create a landing-pad node `lp_<view_name>` (TempTable, `SELECT * FROM
///    view_name`).  Add edge `view_name → lp`.  Deriving the name from the
///    node being materialized keeps landing pads collision-free when several
///    nodes are materialized in the same trial.
/// 2. Find the materialization frontier `M` = `frontier_materializes(view_name)`:
///    the nearest Table / TempTable nodes downstream from `view_name`.
/// 3. For each `m` in `M`, iteratively inline every intermediate View that lies
///    on a path between `view_name` and `m` by substituting the view's SQL as a
///    subquery in `m`'s query text (graph-minor / edge-contraction style).
/// 4. Replace every reference to `view_name` in `m`'s query text with `lp`.
/// 5. Rebase `m` onto `lp` by replacing the `view_name` entry in `m.depends_on`
///    with `lp`.
///
/// After the call:
/// - `view_name` is still a View; only `lp` is a TempTable.
/// - Every direct successor of `lp` is a Table or TempTable.
/// - No `TempTable → View` edge exists in the graph.
///
/// Returns the name of the created landing-pad node.
pub fn make_temp(dag: &mut Dag, view_name: &str) -> Result<String, OptimizerError> {
    // 2. Compute the materialization frontier BEFORE inserting the landing pad,
    //    so lp itself is not included in the frontier set.
    let frontier: HashSet<String> = dag.nodes.frontier_materializes(view_name);

    // 1. Create the landing-pad TempTable, named after the node it backs.
    let lp_name = landing_pad_name(view_name);

    let mut lp_deps = HashSet::new();
    lp_deps.insert(view_name.to_string());
    dag.nodes.add_node_unchecked(TransformNode {
        id: lp_name.clone(),
        query_text: format!("SELECT * FROM {view_name}"),
        materialize: MaterializeMode::TempTable,
        depends_on: lp_deps,
        schema: None,
    });

    // 3–5. For each frontier node m, inline intermediate views then rebase onto lp.
    for m_id in &frontier {
        // Iteratively inline any direct View dependency of m that has
        // view_name as a transitive dependency (i.e., sits between view_name
        // and m on the data-flow path).
        loop {
            let view_dep: Option<String> = dag.nodes.get(m_id.clone()).and_then(|m_node| {
                m_node
                    .depends_on
                    .iter()
                    .find(|dep| {
                        if *dep == view_name {
                            return false; // handled in step 4–5
                        }
                        let is_view = dag
                            .nodes
                            .get((*dep).clone())
                            .map(|d| matches!(d.materialize, MaterializeMode::View))
                            .unwrap_or(false);
                        is_view && is_transitive_dep(dag, dep, view_name)
                    })
                    .cloned()
            });

            match view_dep {
                None => break,
                Some(v_id) => {
                    let view_sql = dag
                        .nodes
                        .get(v_id.clone())
                        .ok_or_else(|| {
                            OptimizerError::Exec(format!(
                                "make_temp: intermediate view '{v_id}' not found"
                            ))
                        })?
                        .query_text
                        .clone();

                    let view_deps: Vec<String> = dag
                        .nodes
                        .get(v_id.clone())
                        .map(|v| v.depends_on.iter().cloned().collect())
                        .unwrap_or_default();

                    let m_node = dag.nodes.get_mut(m_id.clone()).ok_or_else(|| {
                        OptimizerError::Exec(format!("make_temp: node '{m_id}' not found"))
                    })?;

                    // Substitute the view name with an inline subquery.
                    m_node.query_text = m_node
                        .query_text
                        .replace(v_id.as_str(), &format!("({view_sql})"));
                    m_node.depends_on.remove(&v_id);
                    for dep in view_deps {
                        m_node.depends_on.insert(dep);
                    }
                }
            }
        }

        // 4 & 5. Replace view_name with lp and rebase the dependency.
        let m_node = dag
            .nodes
            .get_mut(m_id.clone())
            .ok_or_else(|| OptimizerError::Exec(format!("make_temp: node '{m_id}' not found")))?;

        m_node.query_text = m_node.query_text.replace(view_name, &lp_name);
        if m_node.depends_on.remove(view_name) {
            m_node.depends_on.insert(lp_name.clone());
        }
    }

    Ok(lp_name)
}

/// The landing-pad node ID for `node_id`: the same schema prefix, with the
/// base name prefixed by `lp_`.
///
/// Examples:
///   `"warehouse"."main"."foo"` → `"warehouse"."main"."lp_foo"`
///   `foo`                      → `lp_foo`
///
/// Deriving the name from the node keeps landing pads unique, so materializing
/// several nodes in one pass cannot collide.
pub fn landing_pad_name(node_id: &str) -> String {
    // Use the same schema prefix as node_id so the executor places the landing
    // pad in the same catalog/schema.
    let prefix = schema_prefix(node_id);
    let base = &node_id[prefix.len()..];
    if base.starts_with('"') {
        format!("{prefix}\"lp_{}\"", base.trim_matches('"'))
    } else {
        format!("{prefix}lp_{base}")
    }
}

/// Extract the schema prefix from a qualified node ID.
///
/// Examples:
///   `"warehouse"."main"."foo"` → `"warehouse"."main".`
///   `"foo"`                    → `` (empty — no prefix)
///
/// The landing pad inherits this prefix so it lands in the same catalog/schema.
fn schema_prefix(node_id: &str) -> String {
    // Qualified identifiers join segments with `"."`.  Find the last occurrence
    // of that separator and return everything up to and including it.
    if let Some(pos) = node_id.rfind("\".\"") {
        // pos is the index of `"` before the last `.`
        // include the closing `"` and the `.`: advance by 2 to end after `".`
        format!("{}\".", &node_id[..pos])
    } else {
        String::new()
    }
}

/// Returns `true` if `dep` appears in the transitive dependency set of `node_id`.
pub(crate) fn is_transitive_dep(dag: &Dag, node_id: &str, dep: &str) -> bool {
    let node = match dag.nodes.get(node_id.to_string()) {
        Some(n) => n,
        None => return false,
    };
    if node.depends_on.contains(dep) {
        return true;
    }
    node.depends_on
        .iter()
        .any(|parent| is_transitive_dep(dag, parent, dep))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dag::{Dag, MaterializeMode, TransformNode},
        graph::Graph,
    };
    use std::collections::HashMap;

    fn make_dag(nodes: Vec<TransformNode>) -> Dag {
        let mut map = HashMap::new();
        for n in nodes {
            map.insert(n.id.clone(), n);
        }
        Dag {
            db: "test".into(),
            nodes: Graph::new(map),
            sources: vec![],
            max_parallelism: None,
        }
    }

    fn node(id: &str, mode: MaterializeMode, deps: &[&str], query: &str) -> TransformNode {
        TransformNode {
            id: id.to_string(),
            query_text: query.to_string(),
            materialize: mode,
            depends_on: deps.iter().map(|s| s.to_string()).collect(),
            schema: None,
        }
    }

    // Layout: n (View) --> m (Table)
    //
    // After make_temp(n):
    //   n (View) --> lp_n (TempTable) --> m (Table)
    //   m.query_text references lp_n, not n
    //   m.depends_on = {lp_n}
    #[test]
    fn test_make_temp_direct_table_dep() {
        let mut dag = make_dag(vec![
            node("n", MaterializeMode::View, &[], "SELECT 1 AS x"),
            node("m", MaterializeMode::Table, &["n"], "SELECT x FROM n"),
        ]);

        make_temp(&mut dag, "n").unwrap();

        // lp_n exists and is TempTable
        let lp = dag.nodes.get("lp_n".to_string()).expect("lp_n must exist");
        assert!(matches!(lp.materialize, MaterializeMode::TempTable));
        assert_eq!(lp.query_text, "SELECT * FROM n");
        assert!(lp.depends_on.contains("n"));

        // m now references lp_n, not n
        let m = dag.nodes.get("m".to_string()).unwrap();
        assert!(m.query_text.contains("lp_n"), "m must reference lp_n");
        assert!(
            !m.query_text.contains(" n"),
            "m must not reference n directly"
        );
        assert!(m.depends_on.contains("lp_n"));
        assert!(!m.depends_on.contains("n"));

        // n is still a View
        let n = dag.nodes.get("n".to_string()).unwrap();
        assert!(matches!(n.materialize, MaterializeMode::View));
    }

    // Layout: n (View) --> v1 (View) --> m (Table)
    //
    // After make_temp(n):
    //   n (View) --> lp_n (TempTable) --> m (Table, v1 inlined)
    //   No TempTable → View edge.
    #[test]
    fn test_make_temp_intermediate_view_inlined() {
        let mut dag = make_dag(vec![
            node("n", MaterializeMode::View, &[], "SELECT 1 AS x"),
            node(
                "v1",
                MaterializeMode::View,
                &["n"],
                "SELECT x FROM n WHERE x > 0",
            ),
            node("m", MaterializeMode::Table, &["v1"], "SELECT x FROM v1"),
        ]);

        make_temp(&mut dag, "n").unwrap();

        // m must depend on lp_n only, not v1 or n
        let m = dag.nodes.get("m".to_string()).unwrap();
        assert!(m.depends_on.contains("lp_n"), "m must depend on lp_n");
        assert!(!m.depends_on.contains("v1"), "m must not depend on v1");
        assert!(!m.depends_on.contains("n"), "m must not depend on n");

        // m's query must reference lp_n (v1 was inlined then n replaced by lp_n)
        assert!(
            m.query_text.contains("lp_n"),
            "m query must reference lp_n; got: {}",
            m.query_text
        );

        // No TempTable → View edge: lp_n's only successor is m (Table)
        let lp = dag.nodes.get("lp_n".to_string()).unwrap();
        assert!(matches!(lp.materialize, MaterializeMode::TempTable));
        assert!(lp.depends_on.contains("n"));
    }

    // Layout: n (View) --> v1 (View) --> m1 (Table)
    //                  \-> m2 (Table)
    //
    // After make_temp(n), both m1 and m2 must be rebased onto lp_n.
    #[test]
    fn test_make_temp_multiple_frontier_nodes() {
        let mut dag = make_dag(vec![
            node("n", MaterializeMode::View, &[], "SELECT 1 AS x"),
            node("v1", MaterializeMode::View, &["n"], "SELECT x FROM n"),
            node("m1", MaterializeMode::Table, &["v1"], "SELECT x FROM v1"),
            node("m2", MaterializeMode::Table, &["n"], "SELECT x FROM n"),
        ]);

        make_temp(&mut dag, "n").unwrap();

        let m1 = dag.nodes.get("m1".to_string()).unwrap();
        assert!(m1.depends_on.contains("lp_n"));
        assert!(!m1.depends_on.contains("n"));
        assert!(!m1.depends_on.contains("v1"));

        let m2 = dag.nodes.get("m2".to_string()).unwrap();
        assert!(m2.depends_on.contains("lp_n"));
        assert!(!m2.depends_on.contains("n"));
    }

    #[test]
    fn test_dialect_for_db() {
        assert_eq!(dialect_for_db("duckdb"), DialectType::DuckDB);
        assert_eq!(dialect_for_db("DuckDB"), DialectType::DuckDB);
        assert_eq!(dialect_for_db("postgres"), DialectType::PostgreSQL);
        assert_eq!(dialect_for_db("postgresql"), DialectType::PostgreSQL);
        assert_eq!(dialect_for_db("mysql"), DialectType::MySQL);
        assert_eq!(dialect_for_db("sqlite"), DialectType::SQLite);
        assert_eq!(dialect_for_db("bigquery"), DialectType::BigQuery);
        assert_eq!(dialect_for_db("default"), DialectType::Generic);
        assert_eq!(dialect_for_db("unknown"), DialectType::DuckDB);
    }
}
