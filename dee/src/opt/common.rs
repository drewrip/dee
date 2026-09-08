use std::collections::{HashMap, HashSet};

use polyglot_sql::{dialects::DialectType, expressions::Expression};

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
/// is abandoned. **Zero: a trial is stopped the moment it stops being able to
/// win.**
///
/// The acceptance tests ask one question -- is this candidate faster than the
/// incumbent? Once a trial's elapsed time reaches the incumbent's, the answer
/// is already no, and every further second buys a more precise measurement of
/// a configuration that has been rejected either way. So the budget is exactly
/// the incumbent, and the slack that used to sit on top of it is gone.
///
/// The cost of the slack was not the slack itself: a cancelled run is charged
/// the budget *plus* the resume that finishes it, so widening the budget widens
/// the worst delivered run twice over. Shared by every search that budgets a
/// trial, so "how long may a losing candidate run" is one number rather than
/// one per pass.
pub const DEFAULT_BUDGET_EPS: f64 = 0.0;

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
// AST-level reference rewriting
//
// Shared by `make_temp` and the pushdown pass. Both need to substitute one
// relation reference for another inside a query, and both need it to happen on
// the parsed AST rather than the raw text: a plain `str::replace` will happily
// rewrite `WHERE env = 'staging'` into a comparison against a table name, or
// corrupt a longer identifier that merely contains the node's name, and hand
// the engine a query that means something else entirely.
// ---------------------------------------------------------------------------

/// Extract the bare (unquoted, unqualified) table name from a node ID that may
/// be a one-, two-, or three-part quoted identifier such as
/// `"warehouse"."main"."stg_accounts"`.
pub(crate) fn bare_table_name(node_id: &str) -> String {
    node_id
        .split('.')
        .last()
        .unwrap_or(node_id)
        .trim_matches('"')
        .to_string()
}

/// `true` if `table` is a reference to the DAG node `node_id`.
///
/// Node IDs can be one-, two-, or three-part quoted identifiers such as
/// `"warehouse"."main"."stg_orders"`. Matching compares from the right and
/// only on the parts both sides actually spell out, so an unqualified
/// `stg_orders` in a query matches the node, while a *different* schema's
/// `"warehouse"."raw"."stg_orders"` does not.
pub(crate) fn table_ref_matches(table: &polyglot_sql::expressions::TableRef, node_id: &str) -> bool {
    let parts: Vec<&str> = node_id.split('.').map(|p| p.trim_matches('"')).collect();
    let Some(name) = parts.last() else {
        return false;
    };
    if !table.name.name.eq_ignore_ascii_case(name) {
        return false;
    }
    let qualifiers: Vec<&str> = parts[..parts.len() - 1].to_vec();
    let refs: Vec<&str> = [table.catalog.as_ref(), table.schema.as_ref()]
        .into_iter()
        .flatten()
        .map(|i| i.name.as_str())
        .collect();
    // Compare the qualifiers both sides spell out, right-aligned.
    for (r, q) in refs.iter().rev().zip(qualifiers.iter().rev()) {
        if !r.eq_ignore_ascii_case(q) {
            return false;
        }
    }
    true
}

/// Rewrite every reference to a DAG node in `sql` to the name it is
/// materialized under for analysis, at the AST level.
///
/// `mapping` is keyed by node ID. Matching happens on parsed table
/// references (see [`table_ref_matches`]), so a node ID that also occurs as a
/// substring of a string literal, a column name, or a longer identifier is
/// left alone — unlike the plain `str::replace` this replaced, which would
/// happily rewrite `WHERE env = 'staging'` into a comparison against a
/// scratch table name and hand the connector a query that means something
/// else entirely.
///
/// Returns `None` if `sql` doesn't parse or can't be regenerated; callers
/// fall back to textual substitution, which is what this did before.
pub(crate) fn rewrite_node_refs(
    sql: &str,
    mapping: &HashMap<String, String>,
    dialect: DialectType,
) -> Option<String> {
    let parsed = polyglot_sql::parse_one(sql, dialect).ok()?;
    let rewritten = polyglot_sql::traversal::transform(parsed, &|node| {
        let Expression::Table(table) = &node else {
            return Ok(Some(node));
        };
        let Some((_, new_name)) = mapping
            .iter()
            .find(|(node_id, _)| table_ref_matches(table, node_id))
        else {
            return Ok(Some(node));
        };
        let mut table = table.clone();
        table.name = polyglot_sql::expressions::Identifier::new(new_name.clone());
        table.schema = None;
        table.catalog = None;
        Ok(Some(Expression::Table(table)))
    })
    .ok()?;
    polyglot_sql::generate(&rewritten, dialect).ok()
}

/// Inline `view_sql` (the query text of `view_id`) into `table_sql` by
/// replacing every AST-level table reference to `view_id` with a
/// parenthesized, aliased subquery wrapping `view_sql` — the AST-based
/// counterpart of a plain `str::replace`.
///
/// Operating on the parsed AST (rather than raw substring substitution)
/// avoids matching `view_id`'s name where it merely appears as a substring
/// of an unrelated, longer identifier, and lets the original table's alias
/// (or, if it had none, its own name — so any qualified column references
/// elsewhere in the query keep resolving) carry over onto the new subquery
/// precisely, rather than by accident of leftover trailing text.
///
/// Every occurrence of `view_id` in `table_sql` is replaced (matching
/// `str::replace`'s multi-occurrence behavior for self-joins etc.), each
/// getting its own independent copy of `view_sql`'s AST.
///
/// Returns `None` if `table_sql` or `view_sql` doesn't parse, if
/// regenerating the rewritten AST fails, or if `view_id` was not found
/// anywhere in `table_sql` — callers fall back to plain string substitution
/// in all of those cases.
pub(crate) fn inline_view_ast(
    table_sql: &str,
    view_id: &str,
    view_sql: &str,
    dialect: DialectType,
) -> Option<String> {
    let bare_view = bare_table_name(view_id);
    let table_expr = polyglot_sql::parse_one(table_sql, dialect).ok()?;
    let view_expr = polyglot_sql::parse_one(view_sql, dialect).ok()?;

    let replaced_any = std::cell::Cell::new(false);
    let rewritten = polyglot_sql::traversal::transform(table_expr, &|node| {
        let Expression::Table(t) = &node else {
            return Ok(Some(node));
        };
        if !table_ref_matches(t, view_id) {
            return Ok(Some(node));
        }
        replaced_any.set(true);
        let alias = t
            .alias
            .clone()
            .unwrap_or_else(|| polyglot_sql::expressions::Identifier::new(bare_view.clone()));
        Ok(Some(Expression::Subquery(Box::new(
            polyglot_sql::expressions::Subquery {
                this: view_expr.clone(),
                alias: Some(alias),
                column_aliases: t.column_aliases.clone(),
                alias_explicit_as: t.alias_explicit_as,
                alias_keyword: None,
                order_by: None,
                limit: None,
                offset: None,
                distribute_by: None,
                sort_by: None,
                cluster_by: None,
                lateral: false,
                modifiers_inside: true,
                trailing_comments: vec![],
                inferred_type: None,
            },
        ))))
    })
    .ok()?;

    if !replaced_any.get() {
        return None;
    }

    polyglot_sql::generate(&rewritten, dialect).ok()
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
    let dialect = dialect_for_db(&dag.db);
    let lp_name = landing_pad_name(view_name);

    // Already promoted. Re-running would be actively destructive rather than
    // merely redundant: the pad is now a TempTable child of the view, so it is
    // itself the whole materialized frontier, and the rebase below would
    // repoint the pad at *itself* -- a relation that reads itself, and a cycle
    // in a graph whose topological sort gives up silently on one.
    //
    // Reachable in ordinary use: HMP and OMP both step the same working DAG,
    // and they routinely rank the same hot view first.
    if dag
        .nodes
        .get(lp_name.clone())
        .is_some_and(|lp| lp.depends_on.contains(view_name))
    {
        return Ok(lp_name);
    }

    // 2. Compute the materialization frontier BEFORE inserting the landing pad,
    //    so lp itself is not included in the frontier set.
    let frontier: HashSet<String> = dag.nodes.frontier_materializes(view_name);

    // 1. Create the landing-pad TempTable, named after the node it backs.

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

                    // Substitute the view name with an inline subquery, on the
                    // AST. A plain `str::replace` here corrupts a name that
                    // occurs inside a string literal, a column alias, or a
                    // longer identifier, and drops the alias the subquery needs
                    // for qualified column references to keep resolving.
                    match inline_view_ast(&m_node.query_text, &v_id, &view_sql, dialect) {
                        Some(rewritten) => m_node.query_text = rewritten,
                        None => {
                            return Err(OptimizerError::Exec(format!(
                                "make_temp: could not inline view '{v_id}' into '{m_id}' at the \
                                 AST level; refusing to fall back to text substitution, which \
                                 would silently change what the query means"
                            )));
                        }
                    }
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

        // Rename the reference on the AST, for the same reason as above.
        let mapping = HashMap::from([(view_name.to_string(), lp_name.clone())]);
        match rewrite_node_refs(&m_node.query_text, &mapping, dialect) {
            Some(rewritten) => m_node.query_text = rewritten,
            None => {
                return Err(OptimizerError::Exec(format!(
                    "make_temp: could not repoint '{m_id}' from '{view_name}' to '{lp_name}' at \
                     the AST level; refusing to fall back to text substitution"
                )));
            }
        }
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

    // ---------------------------------------------------------------------
    // The rebase must preserve what each consumer means.
    //
    // Every one of these corrupted the consumer under the old `str::replace`,
    // and a corrupted consumer is not merely a mis-measured candidate: the
    // resume reuses relations across DAGs on the strength of dee's rewrites
    // being correctness-preserving, so a rewrite that changes meaning is a
    // delivered-data bug.
    // ---------------------------------------------------------------------

    /// `orders` is the promoted view; the consumer also reads a relation whose
    /// name merely *contains* it.
    #[test]
    fn test_rebase_does_not_touch_a_longer_identifier() {
        let mut dag = make_dag(vec![
            node("orders", MaterializeMode::View, &[], "SELECT * FROM raw"),
            node(
                "m",
                MaterializeMode::Table,
                &["orders"],
                "SELECT o.id FROM orders o JOIN orders_summary s ON s.id = o.id",
            ),
        ]);
        make_temp(&mut dag, "orders").unwrap();
        let sql = dag.nodes.get("m".to_string()).unwrap().query_text.clone();
        assert!(
            sql.contains("orders_summary"),
            "the unrelated relation was renamed: {sql}"
        );
        assert!(
            !sql.contains("lp_orders_summary"),
            "a longer identifier was corrupted: {sql}"
        );
        assert!(sql.contains("lp_orders"), "the reference was not repointed: {sql}");
    }

    /// The name appears inside a string literal, where it is data, not a
    /// relation.
    #[test]
    fn test_rebase_does_not_touch_a_string_literal() {
        let mut dag = make_dag(vec![
            node("orders", MaterializeMode::View, &[], "SELECT * FROM raw"),
            node(
                "m",
                MaterializeMode::Table,
                &["orders"],
                "SELECT count(*) AS n FROM orders WHERE src = 'orders'",
            ),
        ]);
        make_temp(&mut dag, "orders").unwrap();
        let sql = dag.nodes.get("m".to_string()).unwrap().query_text.clone();
        assert!(
            sql.contains("'orders'"),
            "a string literal was rewritten, silently changing which rows match: {sql}"
        );
    }

    /// The name appears as an output column alias, which is part of this
    /// relation's contract with everything downstream.
    #[test]
    fn test_rebase_does_not_rename_an_output_column() {
        let mut dag = make_dag(vec![
            node("orders", MaterializeMode::View, &[], "SELECT * FROM raw"),
            node(
                "m",
                MaterializeMode::Table,
                &["orders"],
                "SELECT id AS orders FROM orders",
            ),
        ]);
        make_temp(&mut dag, "orders").unwrap();
        let sql = dag.nodes.get("m".to_string()).unwrap().query_text.clone();
        assert!(
            !sql.contains("AS lp_orders") && !sql.contains("as lp_orders"),
            "the output column was renamed, changing the relation's schema: {sql}"
        );
    }

    /// An inlined view must keep the name the consumer's qualified references
    /// resolve against, or every `v1.x` in the consumer dangles.
    #[test]
    fn test_inlining_an_intermediate_view_keeps_its_alias() {
        let mut dag = make_dag(vec![
            node("base", MaterializeMode::View, &[], "SELECT id, amt FROM raw"),
            node(
                "v1",
                MaterializeMode::View,
                &["base"],
                "SELECT id, amt FROM base",
            ),
            node(
                "m",
                MaterializeMode::Table,
                &["v1"],
                "SELECT v1.id, v1.amt FROM v1",
            ),
        ]);
        make_temp(&mut dag, "base").unwrap();
        let sql = dag.nodes.get("m".to_string()).unwrap().query_text.clone();
        assert!(
            sql.contains("v1"),
            "the inlined subquery lost the alias that `v1.id` resolves against: {sql}"
        );
        assert!(sql.contains("lp_base"), "the pad was not wired in: {sql}");
    }

    /// Applying the same promotion twice must not build a relation that reads
    /// itself. Reachable whenever two passes rank the same hot view first.
    #[test]
    fn test_make_temp_twice_leaves_no_self_loop() {
        let mut dag = make_dag(vec![
            node("v", MaterializeMode::View, &[], "SELECT * FROM raw"),
            node("m", MaterializeMode::Table, &["v"], "SELECT * FROM v"),
        ]);
        make_temp(&mut dag, "v").unwrap();
        make_temp(&mut dag, "v").unwrap();

        let lp = dag.nodes.get("lp_v".to_string()).unwrap();
        assert!(
            !lp.depends_on.contains("lp_v"),
            "the pad depends on itself: {:?}",
            lp.depends_on
        );
        assert!(
            !lp.query_text.contains("lp_v"),
            "the pad reads itself: {}",
            lp.query_text
        );
        assert!(
            lp.depends_on.contains("v"),
            "the pad stopped backing its view: {:?}",
            lp.depends_on
        );
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
