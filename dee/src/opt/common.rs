use std::collections::HashSet;

use crate::{
    dag::{Dag, MaterializeMode, TransformNode},
    opt::OptimizerError,
};

/// Safely rewrite `dag` so that `view_name` can be backed by a TempTable
/// without creating any `TempTable → View` edges.
///
/// Algorithm:
/// 1. Create a landing-pad node `lp_<counter>` (TempTable, `SELECT * FROM
///    view_name`).  Add edge `view_name → lp`.
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
pub fn make_temp(
    dag: &mut Dag,
    view_name: &str,
    counter: &mut usize,
) -> Result<String, OptimizerError> {
    // 2. Compute the materialization frontier BEFORE inserting the landing pad,
    //    so lp itself is not included in the frontier set.
    let frontier: HashSet<String> = dag.nodes.frontier_materializes(view_name);

    // 1. Create the landing-pad TempTable.
    // Use the same schema prefix as view_name so the executor places the
    // landing pad in the same catalog/schema (e.g. "warehouse"."main"."lp_0").
    // schema_prefix("warehouse"."main"."foo") → "warehouse"."main".
    let prefix = schema_prefix(view_name);
    let lp_name = if prefix.is_empty() {
        format!("lp_{counter}")
    } else {
        format!("{prefix}\"lp_{counter}\"")
    };
    *counter += 1;

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
            let view_dep: Option<String> = dag
                .nodes
                .get(m_id.clone())
                .and_then(|m_node| {
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
                    m_node.query_text =
                        m_node.query_text.replace(v_id.as_str(), &format!("({view_sql})"));
                    m_node.depends_on.remove(&v_id);
                    for dep in view_deps {
                        m_node.depends_on.insert(dep);
                    }
                }
            }
        }

        // 4 & 5. Replace view_name with lp and rebase the dependency.
        let m_node = dag.nodes.get_mut(m_id.clone()).ok_or_else(|| {
            OptimizerError::Exec(format!("make_temp: node '{m_id}' not found"))
        })?;

        m_node.query_text = m_node.query_text.replace(view_name, &lp_name);
        if m_node.depends_on.remove(view_name) {
            m_node.depends_on.insert(lp_name.clone());
        }
    }

    Ok(lp_name)
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
fn is_transitive_dep(dag: &Dag, node_id: &str, dep: &str) -> bool {
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
    //   n (View) --> lp_0 (TempTable) --> m (Table)
    //   m.query_text references lp_0, not n
    //   m.depends_on = {lp_0}
    #[test]
    fn test_make_temp_direct_table_dep() {
        let mut dag = make_dag(vec![
            node("n", MaterializeMode::View, &[], "SELECT 1 AS x"),
            node("m", MaterializeMode::Table, &["n"], "SELECT x FROM n"),
        ]);

        let mut counter = 0;
        make_temp(&mut dag, "n", &mut counter).unwrap();

        // lp_0 exists and is TempTable
        let lp = dag.nodes.get("lp_0".to_string()).expect("lp_0 must exist");
        assert!(matches!(lp.materialize, MaterializeMode::TempTable));
        assert_eq!(lp.query_text, "SELECT * FROM n");
        assert!(lp.depends_on.contains("n"));

        // m now references lp_0, not n
        let m = dag.nodes.get("m".to_string()).unwrap();
        assert!(m.query_text.contains("lp_0"), "m must reference lp_0");
        assert!(!m.query_text.contains(" n"), "m must not reference n directly");
        assert!(m.depends_on.contains("lp_0"));
        assert!(!m.depends_on.contains("n"));

        // n is still a View
        let n = dag.nodes.get("n".to_string()).unwrap();
        assert!(matches!(n.materialize, MaterializeMode::View));
    }

    // Layout: n (View) --> v1 (View) --> m (Table)
    //
    // After make_temp(n):
    //   n (View) --> lp_0 (TempTable) --> m (Table, v1 inlined)
    //   No TempTable → View edge.
    #[test]
    fn test_make_temp_intermediate_view_inlined() {
        let mut dag = make_dag(vec![
            node("n", MaterializeMode::View, &[], "SELECT 1 AS x"),
            node("v1", MaterializeMode::View, &["n"], "SELECT x FROM n WHERE x > 0"),
            node("m", MaterializeMode::Table, &["v1"], "SELECT x FROM v1"),
        ]);

        let mut counter = 0;
        make_temp(&mut dag, "n", &mut counter).unwrap();

        // m must depend on lp_0 only, not v1 or n
        let m = dag.nodes.get("m".to_string()).unwrap();
        assert!(m.depends_on.contains("lp_0"), "m must depend on lp_0");
        assert!(!m.depends_on.contains("v1"), "m must not depend on v1");
        assert!(!m.depends_on.contains("n"), "m must not depend on n");

        // m's query must reference lp_0 (v1 was inlined then n replaced by lp_0)
        assert!(
            m.query_text.contains("lp_0"),
            "m query must reference lp_0; got: {}",
            m.query_text
        );

        // No TempTable → View edge: lp_0's only successor is m (Table)
        let lp = dag.nodes.get("lp_0".to_string()).unwrap();
        assert!(matches!(lp.materialize, MaterializeMode::TempTable));
        assert!(lp.depends_on.contains("n"));
    }

    // Layout: n (View) --> v1 (View) --> m1 (Table)
    //                  \-> m2 (Table)
    //
    // After make_temp(n), both m1 and m2 must be rebased onto lp_0.
    #[test]
    fn test_make_temp_multiple_frontier_nodes() {
        let mut dag = make_dag(vec![
            node("n", MaterializeMode::View, &[], "SELECT 1 AS x"),
            node("v1", MaterializeMode::View, &["n"], "SELECT x FROM n"),
            node("m1", MaterializeMode::Table, &["v1"], "SELECT x FROM v1"),
            node("m2", MaterializeMode::Table, &["n"], "SELECT x FROM n"),
        ]);

        let mut counter = 0;
        make_temp(&mut dag, "n", &mut counter).unwrap();

        let m1 = dag.nodes.get("m1".to_string()).unwrap();
        assert!(m1.depends_on.contains("lp_0"));
        assert!(!m1.depends_on.contains("n"));
        assert!(!m1.depends_on.contains("v1"));

        let m2 = dag.nodes.get("m2".to_string()).unwrap();
        assert!(m2.depends_on.contains("lp_0"));
        assert!(!m2.depends_on.contains("n"));
    }
}
