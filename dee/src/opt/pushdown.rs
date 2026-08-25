use async_trait::async_trait;
use duckdb::arrow::datatypes::SchemaRef;
use log::{debug, trace, warn};
use polyglot_sql::{
    dialects::DialectType,
    expressions::{Expression, Null, Select, With},
    traversal::ExpressionWalk,
};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use crate::{
    connectors::Connector,
    dag::MaterializeMode,
    executor::Executor,
    opt::common::dialect_for_db,
    opt::{
        Dag, Explain, OptimizerError, OptimizerPass,
        explain::{render_card_grid, render_ranked_table},
        report::{PassDetail, PassOutcome, PushdownDetail, PushdownOutcome},
    },
};

/// Extract the bare (unquoted, unqualified) table name from a node ID that may
/// be a one-, two-, or three-part quoted identifier such as
/// `"warehouse"."main"."stg_accounts"`.
fn bare_table_name(node_id: &str) -> String {
    node_id
        .split('.')
        .last()
        .unwrap_or(node_id)
        .trim_matches('"')
        .to_string()
}

// ---------------------------------------------------------------------------
// Dead-column elimination
//
// When a TempTable's rewritten schema is pruned to only the columns a
// downstream frontier query actually needs, that frontier's own SQL text can
// still contain a *nested* subquery selecting extra columns from the
// TempTable that were never propagated further up (e.g. `SELECT x.a FROM
// (SELECT a, b, c FROM staging) x` — `b`/`c` are dead beyond that subquery).
// If we don't also prune those references, the frontier's own SQL breaks at
// bind time once the TempTable no longer physically has those columns.
//
// This targets exactly that case: for every *nested* SELECT (never the
// outermost statement — its own projection list is the frontier's real
// output, not a "dead" intermediate) whose FROM clause is a single, unjoined
// reference to `source`, prune its projection list down to `keep` (the
// authoritative column set computed from the connector's pushdown analysis).
// ---------------------------------------------------------------------------

/// Return the name of the *physical source column* a SELECT-list item reads,
/// if and only if that item is a plain pass-through column reference —
/// `Expression::Column`, or `Expression::Alias` wrapping one (e.g. `a AS
/// foo`, which still reads physical column `a`; the rename is preserved,
/// only the keep/drop decision is based on `a`).
///
/// Returns `None` for anything else — computed expressions, aggregates,
/// function calls, `CASE`, `*`, literals, ... — which are always left
/// untouched: `keep`/`required_cols` only ever contains `source`'s own
/// physical column names, so testing a *computed* column's output alias
/// (e.g. `mean_temp` in `avg(avg_temp) AS mean_temp`) against that set would
/// almost always miss and wrongly drop it.
fn column_output_name(expr: &Expression) -> Option<String> {
    match expr {
        Expression::Column(col) => Some(col.name.name.clone()),
        Expression::Alias(alias) => match &alias.this {
            Expression::Column(col) => Some(col.name.name.clone()),
            _ => None,
        },
        _ => None,
    }
}

/// `true` if any of `select`'s own SELECT-list items is a `*`/`table.*` star.
fn select_has_star(select: &Select) -> bool {
    select
        .expressions
        .iter()
        .any(|e| matches!(e, Expression::Star(_)))
}

/// `true` if `select` has exactly one, unjoined FROM source and that source
/// is a direct reference to `bare_source` — the only shape this pass will
/// prune, so it never has to reason about join ambiguity or DISTINCT/`*`.
fn is_single_unjoined_source(select: &Select, bare_source: &str) -> bool {
    if !select.joins.is_empty() || select.distinct || select.distinct_on.is_some() {
        return false;
    }
    if select_has_star(select) {
        return false;
    }
    match &select.from {
        Some(from) if from.expressions.len() == 1 => matches!(
            &from.expressions[0],
            Expression::Table(t) if t.name.name.eq_ignore_ascii_case(bare_source)
        ),
        _ => false,
    }
}

/// `true` if any of `select`'s FROM/JOIN targets is a direct reference to
/// `bare_source` (regardless of how many other sources/joins are present).
fn select_scans_source(select: &Select, bare_source: &str) -> bool {
    let is_match = |e: &Expression| matches!(e, Expression::Table(t) if t.name.name.eq_ignore_ascii_case(bare_source));
    select
        .from
        .as_ref()
        .map(|f| f.expressions.iter().any(is_match))
        .unwrap_or(false)
        || select.joins.iter().any(|j| is_match(&j.this))
}

/// Every column `select` references *outside* of a bare pass-through
/// select-list position — i.e. inside WHERE/GROUP BY/HAVING/ORDER BY/JOIN-ON,
/// or inside a select-list item that computes something (an aggregate,
/// function call, arithmetic, `CASE`, ...) rather than just reading a column
/// straight through. [`prune_dead_source_columns`] only ever rewrites bare
/// pass-through select-list items (see `column_output_name`) — everywhere
/// else is always left exactly as written, so every column referenced there
/// must always survive pruning of `source`'s own schema.
fn columns_needed_regardless(select: &Select) -> HashSet<String> {
    let mut trimmed = select.clone();
    trimmed
        .expressions
        .retain(|e| column_output_name(e).is_none());
    polyglot_sql::ast_transforms::get_column_names(&Expression::Select(Box::new(trimmed)))
        .into_iter()
        .collect()
}

/// Return every column name referenced by a scan of `source` in `sql` that
/// [`prune_dead_source_columns`] will *not* rewrite. These columns must
/// survive projection pruning of `source` even if the connector's own
/// analysis reports them as unused overall, since we won't touch the text
/// that still references them. Two cases:
///
/// - A select eligible for pruning (single, unjoined, no `DISTINCT`/`*`):
///   only its bare pass-through select-list items are actually subject to
///   the prune/keep decision (see [`columns_needed_regardless`]) —
///   everything else in it always survives regardless.
/// - A select *not* eligible for pruning (joined, `DISTINCT`, or the
///   outermost statement, which is never pruned regardless of shape):
///   nothing in it is ever rewritten, so *every* column it references
///   always survives.
///
/// Returns `None` if pruning `source`'s projection must be abandoned
/// entirely for this frontier: a `SELECT *`/`table.*` anywhere touching a
/// scan of `source` (e.g. `SELECT pp.*, ... FROM source AS pp`) needs *every*
/// column, and unlike a plain column reference, a star gives us no column
/// names to collect at all — there's nothing a keep-list could represent.
///
/// Where it *can* return a concrete set, this is intentionally over-broad
/// when it can't cleanly attribute a column to `source` specifically (e.g.
/// inside a join) — it just returns every column name in scope, and the
/// caller only keeps the ones that actually exist in `source`'s schema, so a
/// false positive here costs a little pruning, never correctness.
fn collect_unprunable_source_columns(
    sql: &str,
    source_id: &str,
    dialect: DialectType,
) -> Option<HashSet<String>> {
    let bare_source = bare_table_name(source_id);
    let mut out = HashSet::new();
    let Ok(root) = polyglot_sql::parse_one(sql, dialect) else {
        // Can't analyze what we can't parse — fail closed like the `SELECT
        // *`/`table.*` case, not open like "this frontier needs nothing".
        return None;
    };

    // Every "top-level" select — the root itself, or a branch reached from
    // it purely through UNION (never through a nested subquery/derived-table
    // FROM position) — has its own projection list left untouched by
    // `prune_dead_source_columns` no matter its shape (see
    // `prune_root_query`), so any such select that directly scans `source`
    // needs every column it references to survive, not just its bare
    // pass-through columns.
    let mut top_level_selects = HashSet::new();
    collect_top_level_select_ptrs(&root, &mut top_level_selects);

    for node in root.dfs() {
        if let Expression::Select(select) = node {
            if !select_scans_source(select, &bare_source) {
                continue;
            }
            if select_has_star(select) {
                return None;
            }
            if top_level_selects.contains(&(node as *const Expression)) {
                out.extend(polyglot_sql::ast_transforms::get_column_names(&root));
            } else if is_single_unjoined_source(select, &bare_source) {
                out.extend(columns_needed_regardless(select));
            } else {
                out.extend(polyglot_sql::ast_transforms::get_column_names(node));
            }
        }
    }

    Some(out)
}

/// Collect the identities of every "top-level" select reachable from `expr`
/// by descending only through `UNION`/`INTERSECT`/`EXCEPT` branches — i.e.
/// every select whose own output row shape is part of the frontier query's
/// final result, never a droppable intermediate. `prune_root_query` never
/// rewrites these selects' own projection lists (only their FROM/JOIN/CTE
/// positions), matching the guarantee the plain root-`Select` case already
/// relied on before `UNION` support existed here.
fn collect_top_level_select_ptrs<'a>(
    expr: &'a Expression,
    out: &mut HashSet<*const Expression>,
) {
    match expr {
        Expression::Select(_) => {
            out.insert(expr as *const Expression);
        }
        Expression::Union(u) => {
            collect_top_level_select_ptrs(&u.left, out);
            collect_top_level_select_ptrs(&u.right, out);
        }
        Expression::Intersect(i) => {
            collect_top_level_select_ptrs(&i.left, out);
            collect_top_level_select_ptrs(&i.right, out);
        }
        Expression::Except(e) => {
            collect_top_level_select_ptrs(&e.left, out);
            collect_top_level_select_ptrs(&e.right, out);
        }
        _ => {}
    }
}

/// Prune every CTE's definition in `with` (if any) as a derived table — same
/// treatment as a FROM/JOIN position, since a CTE's body is exactly that,
/// just referenced by name instead of inline.
fn prune_with_ctes(with: &mut Option<With>, bare_source: &str, keep: &HashSet<String>) {
    if let Some(with) = with.as_mut() {
        for cte in with.ctes.iter_mut() {
            let taken = std::mem::replace(&mut cte.this, Expression::Null(Null));
            cte.this = prune_derived_table(taken, bare_source, keep);
        }
    }
}

/// Descend into every FROM/JOIN/CTE position of `select` (but never mutate
/// `select`'s own projection list) so nested derived tables get pruned.
fn descend_into_from_positions(select: &mut Select, bare_source: &str, keep: &HashSet<String>) {
    if let Some(mut from) = select.from.take() {
        from.expressions = from
            .expressions
            .into_iter()
            .map(|e| prune_derived_table(e, bare_source, keep))
            .collect();
        select.from = Some(from);
    }
    for join in select.joins.iter_mut() {
        let taken = std::mem::replace(&mut join.this, Expression::Null(Null));
        join.this = prune_derived_table(taken, bare_source, keep);
    }
    prune_with_ctes(&mut select.with, bare_source, keep);
}

/// Applied to an expression sitting in a FROM/JOIN/CTE position: recurse into
/// its own nested derived tables first (bottom-up), then, if this expression
/// is itself a single unjoined scan of `bare_source`, prune its projection
/// list down to `keep`.
///
/// `UNION`/`INTERSECT`/`EXCEPT` branches are recursed into structurally
/// (their own output rows are never pruned — same as this function never
/// touching a `Select`'s own projection list beyond the single-unjoined-scan
/// case — only their FROM/JOIN/CTE positions), matching
/// [`collect_top_level_select_ptrs`]'s notion of which selects are eligible.
fn prune_derived_table(expr: Expression, bare_source: &str, keep: &HashSet<String>) -> Expression {
    // A parenthesized derived table (`FROM (SELECT ...) AS alias`) parses as
    // `Expression::Subquery`, wrapping the actual `Select` in `.this`. Unwrap
    // it, recurse/prune the inner Select, then rewrap so the alias survives.
    if let Expression::Subquery(mut sub) = expr {
        sub.this = prune_derived_table(sub.this, bare_source, keep);
        return Expression::Subquery(sub);
    }

    match expr {
        // `Union`/`Intersect`/`Except` implement `Drop` (to iteratively
        // flatten deeply left-recursive chains), which forbids partially
        // moving out of `.left`/`.right` — take each field via
        // `mem::replace` first instead.
        Expression::Union(mut u) => {
            let left = std::mem::replace(&mut u.left, Expression::Null(Null));
            let right = std::mem::replace(&mut u.right, Expression::Null(Null));
            u.left = prune_derived_table(left, bare_source, keep);
            u.right = prune_derived_table(right, bare_source, keep);
            prune_with_ctes(&mut u.with, bare_source, keep);
            return Expression::Union(u);
        }
        Expression::Intersect(mut i) => {
            let left = std::mem::replace(&mut i.left, Expression::Null(Null));
            let right = std::mem::replace(&mut i.right, Expression::Null(Null));
            i.left = prune_derived_table(left, bare_source, keep);
            i.right = prune_derived_table(right, bare_source, keep);
            prune_with_ctes(&mut i.with, bare_source, keep);
            return Expression::Intersect(i);
        }
        Expression::Except(mut e) => {
            let left = std::mem::replace(&mut e.left, Expression::Null(Null));
            let right = std::mem::replace(&mut e.right, Expression::Null(Null));
            e.left = prune_derived_table(left, bare_source, keep);
            e.right = prune_derived_table(right, bare_source, keep);
            prune_with_ctes(&mut e.with, bare_source, keep);
            return Expression::Except(e);
        }
        _ => {}
    }

    let Expression::Select(mut select) = expr else {
        return expr;
    };

    descend_into_from_positions(&mut select, bare_source, keep);

    if is_single_unjoined_source(&select, bare_source) {
        select
            .expressions
            .retain(|e| column_output_name(e).map(|n| keep.contains(&n)).unwrap_or(true));
        if select.expressions.is_empty() {
            // Safety net: never produce an empty SELECT list.
            select.expressions.push(Expression::Star(
                polyglot_sql::expressions::Star {
                    table: None,
                    except: None,
                    replace: None,
                    rename: None,
                    trailing_comments: vec![],
                    span: None,
                },
            ));
        }
    }

    Expression::Select(select)
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
fn inline_view_ast(
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
        if !t.name.name.eq_ignore_ascii_case(&bare_view) {
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

/// Rewrite `sql` so that every *nested* (non-outermost) SELECT statement
/// scanning `source` directly (with no join, no `*`, no DISTINCT) has its
/// projection list pruned to `keep_cols`. The outermost statement's own
/// projection list is never touched — see the module-level doc comment.
///
/// Returns `None` if `sql` doesn't parse, or regenerating it fails; callers
/// treat that as "leave this query's text unchanged" (safe — just misses an
/// opportunity to prune, never breaks anything).
fn prune_dead_source_columns(
    sql: &str,
    source_id: &str,
    keep_cols: &[String],
    dialect: DialectType,
) -> Option<String> {
    if keep_cols.is_empty() {
        return None;
    }
    let bare_source = bare_table_name(source_id);
    let keep: HashSet<String> = keep_cols.iter().cloned().collect();

    let root = polyglot_sql::parse_one(sql, dialect).ok()?;
    let pruned = prune_root_query(root, &bare_source, &keep)?;
    polyglot_sql::generate(&pruned, dialect).ok()
}

/// Prune the FROM/JOIN/CTE positions of the outermost query `expr` — a
/// `Select`, or a `UNION`/`INTERSECT`/`EXCEPT` of such — without ever
/// mutating any top-level branch's own projection list, matching
/// [`collect_top_level_select_ptrs`]'s notion of which selects are eligible
/// for pruning. Returns `None` if `expr` is some other statement shape this
/// pass doesn't know how to descend into (e.g. a bare `VALUES` list).
fn prune_root_query(
    expr: Expression,
    bare_source: &str,
    keep: &HashSet<String>,
) -> Option<Expression> {
    match expr {
        Expression::Select(mut select) => {
            descend_into_from_positions(&mut select, bare_source, keep);
            Some(Expression::Select(select))
        }
        // See the `mem::replace` note in `prune_derived_table` — `Union`'s
        // `Drop` impl forbids partially moving out of `.left`/`.right`.
        Expression::Union(mut u) => {
            let left = std::mem::replace(&mut u.left, Expression::Null(Null));
            let right = std::mem::replace(&mut u.right, Expression::Null(Null));
            u.left = prune_root_query(left, bare_source, keep)?;
            u.right = prune_root_query(right, bare_source, keep)?;
            prune_with_ctes(&mut u.with, bare_source, keep);
            Some(Expression::Union(u))
        }
        Expression::Intersect(mut i) => {
            let left = std::mem::replace(&mut i.left, Expression::Null(Null));
            let right = std::mem::replace(&mut i.right, Expression::Null(Null));
            i.left = prune_root_query(left, bare_source, keep)?;
            i.right = prune_root_query(right, bare_source, keep)?;
            prune_with_ctes(&mut i.with, bare_source, keep);
            Some(Expression::Intersect(i))
        }
        Expression::Except(mut e) => {
            let left = std::mem::replace(&mut e.left, Expression::Null(Null));
            let right = std::mem::replace(&mut e.right, Expression::Null(Null));
            e.left = prune_root_query(left, bare_source, keep)?;
            e.right = prune_root_query(right, bare_source, keep)?;
            prune_with_ctes(&mut e.with, bare_source, keep);
            Some(Expression::Except(e))
        }
        _ => None,
    }
}

/// Extract the column names referenced by a raw SQL predicate string (as
/// returned by [`Connector::pushdown`]'s `filters`), by parsing it as a
/// standalone `WHERE` clause. Used to make sure a column needed only by a
/// pushed-down filter (never itself selected) still survives projection
/// pruning of `source`.
///
/// Returns `None` if `filter` doesn't parse — callers must treat that as
/// "this predicate's columns are unknown," i.e. fail closed (assume all
/// columns needed), never silently drop the predicate's columns from the
/// keep-list.
fn filter_referenced_columns(filter: &str, dialect: DialectType) -> Option<Vec<String>> {
    let wrapped = format!("SELECT 1 WHERE {filter}");
    match polyglot_sql::parse_one(&wrapped, dialect) {
        Ok(expr) => Some(polyglot_sql::ast_transforms::get_column_names(&expr)),
        Err(_) => None,
    }
}

// ---------------------------------------------------------------------------
// scratch DAG materialization
// ---------------------------------------------------------------------------

/// Materialize every node of `dag` as a real table under an isolated scratch
/// name, in topological order, with every cross-node reference in each
/// node's query text rewritten to the corresponding scratch name — so the
/// connector's own query planner (e.g. DuckDB's `EXPLAIN`) can analyze any
/// node's query in isolation via [`Connector::pushdown`], regardless of
/// which other not-yet-materialized DAG nodes it references.
///
/// This must materialize *every* node, not just the one(s) about to be
/// analyzed: a TempTable's own query text (or a frontier's) can reference
/// another TempTable that isn't a real relation under its original name
/// either — e.g. two landing-pad TempTables where one transitively depends
/// on the other. Analyzing them one at a time, each with its own ad hoc
/// scratch table, breaks the moment one references the other.
///
/// Each scratch relation is a real TABLE regardless of the source node's own
/// `MaterializeMode` (even for what will eventually be a View) — DuckDB's
/// planner sees straight through views (inlining them), so a scratch VIEW
/// would attribute a scan to whatever's underneath it rather than to the
/// scratch relation itself. In practice `dag` here is always a
/// `graph_minor`-reduced DAG with no View nodes left, so this is moot, but
/// materializing as a table is the correct choice either way.
///
/// Returns a map from every original node ID to its scratch relation name;
/// pass it to [`pushdown`] and, when done, to [`cleanup_scratch_dag`].
pub async fn materialize_scratch_dag<C>(
    dag: &Dag,
    conn: &C,
) -> Result<HashMap<String, String>, OptimizerError>
where
    C: Connector + Send + Sync,
{
    let topo = dag.nodes.topological_sort();
    let scratch_names: HashMap<String, String> = topo
        .iter()
        .enumerate()
        .map(|(i, id)| (id.clone(), format!("dee_tmp_pushdown_all_{i}")))
        .collect();

    let mut sorted_ids: Vec<&String> = scratch_names.keys().collect();
    sorted_ids.sort_by_key(|id| std::cmp::Reverse(id.len()));

    for id in &topo {
        let node = dag.nodes.get(id.clone()).ok_or_else(|| {
            OptimizerError::Exec(format!("materialize_scratch_dag: node '{id}' not found"))
        })?;

        let mut query = node.query_text.clone();
        for other_id in &sorted_ids {
            query = query.replace(other_id.as_str(), &scratch_names[*other_id]);
        }

        conn.new_relation(MaterializeMode::Table, scratch_names[id].clone(), query)
            .await
            .map_err(|e| {
                OptimizerError::Exec(format!(
                    "materialize_scratch_dag: failed to create scratch relation for '{id}': {e}"
                ))
            })?;
    }

    Ok(scratch_names)
}

/// Drop every scratch relation created by [`materialize_scratch_dag`], in
/// reverse topological order (sinks first) so nothing is dropped while
/// something else still (transitively) depends on it. Best-effort — a
/// failure to drop one relation is logged but doesn't stop the rest.
pub async fn cleanup_scratch_dag<C>(dag: &Dag, scratch_names: &HashMap<String, String>, conn: &C)
where
    C: Connector + Send + Sync,
{
    let mut topo = dag.nodes.topological_sort();
    topo.reverse();
    for id in &topo {
        let Some(scratch_name) = scratch_names.get(id) else {
            continue;
        };
        if let Err(e) = conn
            .drop_relation(MaterializeMode::Table, scratch_name.clone())
            .await
        {
            warn!("cleanup_scratch_dag: failed to drop scratch relation '{scratch_name}': {e}");
        }
    }
}

// ---------------------------------------------------------------------------
// pushdown
// ---------------------------------------------------------------------------

/// Return type for [`pushdown`]: the rewritten source SQL and schema, plus
/// dead-column-pruned SQL for any frontier node whose nested references to
/// `source` needed adjusting to match the source's new, narrower schema.
pub struct PushdownResult {
    pub source_sql: String,
    pub source_schema: SchemaRef,
    /// Pruned SQL for each frontier node whose text changed, keyed by node ID.
    pub frontier_sql: HashMap<String, String>,
}

/// Pushes down predicates and projections from the frontier materializing
/// nodes into the TempTable node `source`, returning the rewritten SQL for
/// `source`.
///
/// `scratch_names` must map every node ID in `dag` to an already-materialized
/// scratch relation (see [`materialize_scratch_dag`]) — every reference to
/// any DAG node, not just `source`, is rewritten to its scratch counterpart
/// before being handed to the connector, since `source`'s own query text (or
/// a frontier's) may reference *other* TempTables that aren't real relations
/// under their original names either.
///
/// For each node in `frontier_materializes(source)`:
/// 1. The frontier node's query text (rewritten so every DAG node reference
///    resolves to its scratch relation) is analyzed via [`Connector::pushdown`].
/// 2. Filter predicates are combined across all frontier nodes with a
///    logical **OR** (each frontier's own predicates are AND-ed together
///    first, matching how DuckDB's `Filters` already reports each predicate
///    as a separate conjunct).
/// 3. Projected columns are collected and **unioned** across all frontier
///    nodes, plus any column referenced only by a pushed-down filter, so
///    every consumer's required columns are present.
///
/// The combined filter and projection are then applied to `source`'s own
/// query by wrapping it as a subquery — the original query text is preserved
/// verbatim, only the outermost `SELECT`/`WHERE` are added.
pub async fn pushdown<C>(
    dag: &Dag,
    source: &str,
    conn: &C,
    scratch_names: &HashMap<String, String>,
) -> Result<PushdownResult, OptimizerError>
where
    C: Connector + Send + Sync,
{
    let source_node = dag
        .nodes
        .get(source.to_string())
        .ok_or_else(|| OptimizerError::Exec(format!("source node '{source}' not found")))?;

    if !matches!(source_node.materialize, MaterializeMode::TempTable) {
        return Err(OptimizerError::Exec(format!(
            "pushdown: '{source}' is not a TempTable"
        )));
    }

    // Use the pre-resolved schema from resolve_schemas.
    let source_schema: SchemaRef = source_node
        .schema
        .as_ref()
        .ok_or_else(|| {
            OptimizerError::Exec(format!(
                "pushdown: node '{source}' has no resolved schema; \
                 call resolve_schemas before running PushdownPass"
            ))
        })?
        .clone();

    let dialect = dialect_for_db(&dag.db);
    let frontier: HashSet<String> = dag.nodes.frontier_materializes(source);

    if frontier.is_empty() {
        debug!("pushdown '{source}': no materializing frontier, nothing to push down");
        return Ok(PushdownResult {
            source_sql: source_node.query_text.clone(),
            source_schema: Arc::clone(&source_schema),
            frontier_sql: HashMap::new(),
        });
    }

    trace!(
        "  source '{}' query text =\n{}",
        source_node.id, source_node.query_text,
    );

    let scratch_name = scratch_names.get(source).cloned().ok_or_else(|| {
        OptimizerError::Exec(format!(
            "pushdown: no scratch relation registered for '{source}'; \
             call materialize_scratch_dag before running pushdown"
        ))
    })?;

    // Every DAG node reference in a query, sorted longest-first so a shorter
    // ID that happens to be a substring of a longer one is never matched
    // first (mirrors the same trick used in `executor::resolve_schemas` and
    // `graph_minor`).
    let mut sorted_ids: Vec<&String> = scratch_names.keys().collect();
    sorted_ids.sort_by_key(|id| std::cmp::Reverse(id.len()));

    // per-frontier-node filter predicates (each entry = the filters for one node)
    let mut per_node_filters: Vec<Vec<String>> = Vec::new();
    // union of projected columns across all frontier nodes
    let mut required_cols: HashSet<String> = HashSet::new();
    let mut any_node_needs_all_cols = false;

    for n_id in &frontier {
        trace!("trying to pushdown frontier node {} into {}", n_id, source);

        let n_node = match dag.nodes.get(n_id.clone()) {
            Some(n) => n,
            None => continue,
        };

        trace!(
            "  frontier node '{}' query text =\n{}",
            n_node.id, n_node.query_text,
        );

        let mut rewritten_query = n_node.query_text.clone();
        for id in &sorted_ids {
            rewritten_query = rewritten_query.replace(id.as_str(), &scratch_names[*id]);
        }

        let pushdown_map = match conn.pushdown(&rewritten_query).await {
            Ok(m) => m,
            Err(e) => {
                warn!(
                    "pushdown: connector pushdown failed for frontier '{n_id}': {e}, \
                     assuming all columns needed"
                );
                any_node_needs_all_cols = true;
                continue;
            }
        };

        match pushdown_map.and_then(|m| m.get(&scratch_name).cloned()) {
            Some(info) => {
                if info.projections.is_empty() {
                    trace!("  frontier '{n_id}': needs all columns");
                    any_node_needs_all_cols = true;
                } else {
                    trace!(
                        "  frontier '{n_id}': projected columns = [{}]",
                        info.projections.join(", ")
                    );
                    required_cols.extend(info.projections.iter().cloned());
                }

                // Filter-only columns (e.g. `WHERE is_active` when
                // `is_active` is never selected) must survive projection
                // pruning even though DuckDB's own `Projections` list
                // excludes them.
                for f in &info.filters {
                    match filter_referenced_columns(f, dialect) {
                        Some(cols) => required_cols.extend(cols),
                        None => {
                            warn!(
                                "pushdown: couldn't parse filter predicate '{f}' for frontier \
                                 '{n_id}', assuming all columns needed"
                            );
                            any_node_needs_all_cols = true;
                        }
                    }
                }

                if info.filters.is_empty() {
                    trace!("  frontier '{n_id}': no filter predicates found");
                } else {
                    trace!("  frontier '{n_id}': predicates = [{}]", info.filters.join(", "));
                    per_node_filters.push(info.filters);
                }
            }
            None => {
                trace!(
                    "  frontier '{n_id}': scratch scan not found in connector pushdown result, \
                     assuming all columns needed"
                );
                any_node_needs_all_cols = true;
            }
        }

        // Safety net: the connector's own pushdown analysis is (correctly)
        // dead-column-aware — it may legitimately report a column as unused
        // even though it's still selected inside a *joined* (or DISTINCT/`*`)
        // nested scan of `source`, one `prune_dead_source_columns` won't
        // rewrite later. Any such column must still survive pruning of
        // `source`'s schema, or that untouched text will fail to bind.
        match collect_unprunable_source_columns(&n_node.query_text, source, dialect) {
            Some(unprunable) if !unprunable.is_empty() => {
                trace!(
                    "  frontier '{n_id}': columns kept regardless (referenced by an un-prunable scan) = [{}]",
                    unprunable.iter().cloned().collect::<Vec<_>>().join(", ")
                );
                required_cols.extend(unprunable);
            }
            Some(_) => {}
            None => {
                trace!(
                    "  frontier '{n_id}': a `SELECT *`/`table.*` touches a scan of '{source}', \
                     assuming all columns needed"
                );
                any_node_needs_all_cols = true;
            }
        }
    }

    // Build the combined filter: OR of each frontier node's AND-conjunction.
    let combined_filter: Option<String> = {
        let per_node_conjunctions: Vec<String> = per_node_filters
            .into_iter()
            .map(|fs| fs.join(" AND "))
            .collect();
        match per_node_conjunctions.len() {
            0 => None,
            1 => per_node_conjunctions.into_iter().next(),
            _ => Some(
                per_node_conjunctions
                    .into_iter()
                    .map(|f| format!("({f})"))
                    .collect::<Vec<_>>()
                    .join(" OR "),
            ),
        }
    };

    // Build the combined projection: union of required columns.
    // If any node needed all columns, skip projection pruning.
    let projection_cols: Option<Vec<String>> =
        if any_node_needs_all_cols || required_cols.is_empty() {
            None
        } else {
            // Preserve schema order so the output is deterministic.
            let schema_cols: Vec<String> = source_schema
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .filter(|name| required_cols.contains(name))
                .collect();
            Some(schema_cols)
        };

    // If there is nothing to push down, return the original query unchanged.
    if combined_filter.is_none() && projection_cols.is_none() {
        debug!("pushdown '{source}': nothing to push down");
        return Ok(PushdownResult {
            source_sql: source_node.query_text.clone(),
            source_schema: Arc::clone(&source_schema),
            frontier_sql: HashMap::new(),
        });
    }

    // Prune dead columns from every frontier query's nested references to
    // `source` so their SQL text stays consistent with source's new,
    // narrower materialized schema (see the module-level doc comment).
    let mut frontier_sql: HashMap<String, String> = HashMap::new();
    if let Some(cols) = &projection_cols {
        for n_id in &frontier {
            let Some(n_node) = dag.nodes.get(n_id.clone()) else {
                continue;
            };
            if let Some(pruned) = prune_dead_source_columns(&n_node.query_text, source, cols, dialect)
            {
                if pruned != n_node.query_text {
                    frontier_sql.insert(n_id.clone(), pruned);
                }
            }
        }
    }

    // Construct the final SQL by wrapping the original query as a subquery and
    // applying the pushed-down filter and projection as an outer SELECT.
    // This is dialect-agnostic: the original query is preserved verbatim, and
    // only the outermost SELECT and WHERE are added by us.
    let alias = bare_table_name(source);

    let col_list = match &projection_cols {
        Some(cols) if !cols.is_empty() => cols
            .iter()
            .map(|c| format!("\"{c}\""))
            .collect::<Vec<_>>()
            .join(", "),
        _ => "*".to_string(),
    };

    let where_clause = match &combined_filter {
        Some(filter_str) => {
            debug!("pushdown '{source}': pushing filter  → {filter_str}");
            format!(" WHERE {filter_str}")
        }
        None => {
            debug!("pushdown '{source}': no filter to push");
            String::new()
        }
    };

    match &projection_cols {
        Some(cols) if !cols.is_empty() => {
            debug!(
                "pushdown '{source}': pushing projection → [{}]",
                cols.join(", ")
            );
        }
        _ => {
            debug!("pushdown '{source}': no projection pruning (all columns needed)");
        }
    }

    // Compute the output schema for the rewritten node.  When columns were
    // pruned, the new schema is the projection subset (in original field order);
    // otherwise it is identical to the pre-rewrite schema.
    let new_schema: SchemaRef = match &projection_cols {
        Some(cols) if !cols.is_empty() => {
            use duckdb::arrow::datatypes::Schema;
            let fields: Vec<_> = cols
                .iter()
                .filter_map(|name| source_schema.field_with_name(name).ok().cloned())
                .collect();
            Arc::new(Schema::new(fields))
        }
        _ => Arc::clone(&source_schema),
    };

    Ok(PushdownResult {
        source_sql: format!(
            "SELECT {col_list} FROM ({inner}) AS \"{alias}\"{where_clause}",
            inner = source_node.query_text,
        ),
        source_schema: new_schema,
        frontier_sql,
    })
}

/// Produces a copy of `dag` with every `View` node eliminated by inlining
/// its SQL query into every downstream `Table` or `TempTable` that reads from
/// it.  The returned DAG contains no `View` nodes.
///
/// Algorithm (repeated until no views remain):
/// 1. Find all `(view v, non-view table t)` edges where `t` directly depends
///    on `v`.
/// 2. For each such pair: inline `v`'s query into `t` by substituting the
///    view's name for a parenthesized subquery, update `t.query_text`, drop
///    the edge `t → v`, and add edges from `t` to every node that `v` itself
///    depends on (so `t` retains `v`'s transitive deps).
/// 3. Any `View` that has become a sink (nothing depends on it any more) is
///    removed from the graph together with all of its in-edges.
/// 4. Repeat until no `View` nodes are left.
pub async fn graph_minor(dag: &Dag) -> Result<Dag, OptimizerError> {
    let mut minor = dag.clone();
    let dialect = dialect_for_db(&dag.db);

    loop {
        // Collect all (view_id, table_id) pairs where a non-view node
        // directly depends on a view node.
        let pairs: Vec<(String, String)> = minor
            .nodes
            .nodes()
            .filter(|t| !matches!(t.materialize, MaterializeMode::View))
            .flat_map(|t| {
                t.depends_on
                    .iter()
                    .filter(|dep_id| {
                        minor
                            .nodes
                            .get((*dep_id).clone())
                            .map(|v| matches!(v.materialize, MaterializeMode::View))
                            .unwrap_or(false)
                    })
                    .map(|v_id| (v_id.clone(), t.id.clone()))
                    .collect::<Vec<_>>()
            })
            .collect();

        if pairs.is_empty() {
            // No more views reachable from a table — we are done if there are
            // also no view-to-view edges left (handled in the next check).
            break;
        }

        for (view_id, table_id) in &pairs {
            // 1. Inline the view's SQL into the table's query_text by replacing
            //    every occurrence of the view's name with a subquery alias.
            //    This is pure text substitution and works regardless of which
            //    SQL dialect the queries are written in.  The only way this can
            //    legitimately fail is if the view is not actually referenced in
            //    the table's query, which indicates a malformed DAG.
            let view_sql = minor
                .nodes
                .get(view_id.clone())
                .ok_or_else(|| {
                    OptimizerError::Exec(format!("graph_minor: view node '{view_id}' not found"))
                })?
                .query_text
                .clone();
            let current_table_sql = minor
                .nodes
                .get(table_id.clone())
                .ok_or_else(|| {
                    OptimizerError::Exec(format!("graph_minor: table node '{table_id}' not found"))
                })?
                .query_text
                .clone();

            if !current_table_sql.contains(view_id.as_str()) {
                return Err(OptimizerError::Exec(format!(
                    "graph_minor: DAG is malformed — '{table_id}' depends on '{view_id}' \
                     but does not reference it in its query_text"
                )));
            }

            // Wrap the view's SQL in a subquery and replace the view reference
            // at the AST level (see `inline_view_ast`), falling back to plain
            // string substitution if the SQL doesn't parse under polyglot-sql
            // (e.g. an exotic dialect-specific construct) — the fallback
            // preserves whatever alias followed the view's name in the
            // original text (or none, if it had none), matching the prior
            // behavior of this pass exactly.
            let new_sql = inline_view_ast(&current_table_sql, view_id, &view_sql, dialect)
                .unwrap_or_else(|| {
                    current_table_sql.replace(view_id.as_str(), &format!("({view_sql})"))
                });

            // 2a. Update the table's query text.
            {
                let t = minor
                    .nodes
                    .get_mut(table_id.clone())
                    .ok_or_else(|| OptimizerError::Exec(format!("node '{table_id}' vanished")))?;
                t.query_text = new_sql;

                // 2b. Drop the edge t → v.
                t.depends_on.remove(view_id);

                // 2c. Inherit v's dependencies: add edges from t to every
                //     node that v itself depended on.
                let view_deps: Vec<String> = minor
                    .nodes
                    .get(view_id.clone())
                    .map(|v| v.depends_on.iter().cloned().collect())
                    .unwrap_or_default();

                for dep in view_deps {
                    minor
                        .nodes
                        .get_mut(table_id.clone())
                        .ok_or_else(|| OptimizerError::Exec(format!("node '{table_id}' vanished")))?
                        .depends_on
                        .insert(dep);
                }
            }
        }

        // 3. Remove any View nodes that have become sinks (out-degree == 0):
        //    nothing depends on them any more so they are unreachable.
        let orphaned_views: Vec<String> = minor
            .nodes
            .nodes()
            .filter(|n| {
                matches!(n.materialize, MaterializeMode::View) && minor.nodes.out_degree(&n.id) == 0
            })
            .map(|n| n.id.clone())
            .collect();

        for view_id in orphaned_views {
            minor.nodes.remove(view_id);
        }
    }

    // Final sweep: remove any zombie view islands — views that are only
    // referenced by other views (no Table/TempTable consumer anywhere in their
    // downstream).  These arise when make_temp rebases a frontier node onto an
    // lp_* TempTable, leaving the original view chain (and its upstream nodes)
    // with no non-view consumer.  Iteratively remove out-degree-0 views until
    // none remain; each removal cascades to upstream views whose out-degree may
    // now also drop to 0.
    loop {
        let dead: Vec<String> = minor
            .nodes
            .nodes()
            .filter(|n| {
                matches!(n.materialize, MaterializeMode::View) && minor.nodes.out_degree(&n.id) == 0
            })
            .map(|n| n.id.clone())
            .collect();
        if dead.is_empty() {
            break;
        }
        for view_id in dead {
            minor.nodes.remove(view_id);
        }
    }

    // Sanity check: no View nodes should remain.
    let remaining_views: Vec<_> = minor
        .nodes
        .nodes()
        .filter(|n| matches!(n.materialize, MaterializeMode::View))
        .map(|n| n.id.clone())
        .collect();

    if !remaining_views.is_empty() {
        return Err(OptimizerError::Exec(format!(
            "graph_minor: views still present after reduction: {:?}",
            remaining_views
        )));
    }

    Ok(minor)
}

#[derive(Debug, Clone)]
pub struct PushdownPass<C, E>
where
    C: Connector + Send + 'static + Sync,
    E: Executor<C> + Send + Sync,
{
    conn: Arc<C>,
    engine: Arc<E>,
    /// Data collected during the last `run()`, used by `Explain::explain`.
    explain_data: Option<PushdownExplainData>,
}

/// Everything `Explain::explain` needs to describe what the last `run()`
/// did and why, retained from otherwise-local data computed during `run()`.
#[derive(Debug, Clone)]
struct PushdownExplainData {
    /// `(node_id, outcome)` for every TempTable considered, deepest-first.
    outcomes: Vec<(String, String)>,
    rewrites_applied: usize,
}

impl<C, E> PushdownPass<C, E>
where
    C: Connector + Send + 'static + Sync,
    E: Executor<C> + Send + Sync,
{
    pub fn new(conn: Arc<C>, engine: Arc<E>) -> Self {
        Self {
            conn,
            engine,
            explain_data: None,
        }
    }
}

#[async_trait]
impl<C, E> OptimizerPass<C, E> for PushdownPass<C, E>
where
    C: Connector + Send + 'static + Sync,
    E: Executor<C> + Send + Sync,
{
    async fn run(&mut self, dag: &mut Dag) -> Result<PassOutcome, OptimizerError> {
        debug!("PushdownPass: starting");

        // Prerequisite — resolve the Arrow output schema of every node in the
        // DAG by running the whole DAG as views and calling get_schema on each.
        // This is the only schema resolution strategy used by this pass.
        // If it fails the pass errors out immediately.
        debug!("PushdownPass: running resolve_schemas");
        let mut resolved_dag = dag.clone();
        self.engine
            .resolve_schemas(&mut resolved_dag)
            .await
            .map_err(|e| {
                OptimizerError::Exec(format!("PushdownPass: resolve_schemas failed: {e}"))
            })?;
        debug!("PushdownPass: schemas resolved for all nodes");

        // Step 1 — compute the graph minor (all Views inlined into their
        // downstream Tables/TempTables) and work on that copy.
        debug!("PushdownPass: computing graph minor");
        let mut minor = graph_minor(&resolved_dag).await?;
        debug!("PushdownPass: running resolve_schemas for the graph minor");
        self.engine.resolve_schemas(&mut minor).await.map_err(|e| {
            OptimizerError::Exec(format!(
                "PushdownPass: resolve_schemas failed for the graph minor: {e}"
            ))
        })?;
        debug!("PushdownPass: schemas resolved for all nodes in the graph minor");
        debug!(
            "PushdownPass: graph minor has {} nodes",
            minor.nodes.num_nodes()
        );

        // Step 2 — topological order, keep only TempTables, then reverse so
        // that nodes deeper in the DAG (closer to the sinks) come first.
        // Processing deeper nodes first means each pushdown sees the most
        // specific filter/projection context before shallower nodes are updated.
        let mut temp_table_ids: Vec<String> = minor
            .nodes
            .topological_sort()
            .into_iter()
            .filter(|id| {
                minor
                    .nodes
                    .get(id.clone())
                    .map(|n| matches!(n.materialize, MaterializeMode::TempTable))
                    .unwrap_or(false)
            })
            .collect();
        temp_table_ids.reverse();

        debug!(
            "PushdownPass: {} TempTable(s) to process (deepest-first): {:?}",
            temp_table_ids.len(),
            temp_table_ids
        );
        let temp_tables_count = temp_table_ids.len();

        // Materialize every node of the graph minor as a real scratch table
        // up front, once, so pushdown() can analyze any node's query in
        // isolation regardless of which other (not-yet-materialized)
        // TempTables it references — see `materialize_scratch_dag`'s doc
        // comment for why this can't be done one TempTable at a time.
        debug!("PushdownPass: materializing scratch DAG for analysis");
        let scratch_names = materialize_scratch_dag(&minor, self.conn.as_ref())
            .await
            .map_err(|e| {
                OptimizerError::Exec(format!(
                    "PushdownPass: failed to materialize scratch DAG: {e}"
                ))
            })?;

        // Step 3+4 — for each TempTable, run pushdown on the minor DAG.
        // pushdown() returns the rewritten source SQL *and* dead-column-pruned
        // SQL for any frontier node whose text needed adjusting. Apply all of
        // them to both the working minor and the original DAG.
        let mut rewrites: usize = 0;
        let mut outcomes: Vec<(String, String)> = Vec::new();
        for node_id in &temp_table_ids {
            if minor.nodes.frontier_materializes(node_id).is_empty() {
                debug!("PushdownPass: '{node_id}' has no materializing frontier, skipping");
                outcomes.push((
                    node_id.clone(),
                    "skipped (no materializing frontier)".to_string(),
                ));
                continue;
            }

            debug!("PushdownPass: running pushdown on '{node_id}'");

            let result = match pushdown(&minor, node_id, self.conn.as_ref(), &scratch_names).await {
                Ok(r) => r,
                Err(e) => {
                    warn!("PushdownPass: skipping '{node_id}', pushdown failed: {e}");
                    outcomes.push((node_id.clone(), format!("skipped (pushdown failed: {e})")));
                    continue;
                }
            };

            // Apply dead-column-pruned SQL for frontier nodes first so that
            // both DAGs reflect the clean, canonical SQL before the source node
            // is updated.
            for (frontier_id, pruned_sql) in &result.frontier_sql {
                debug!("PushdownPass: updating frontier '{frontier_id}' with pruned SQL");
                if let Some(n) = dag.nodes.get_mut(frontier_id.clone()) {
                    n.query_text = pruned_sql.clone();
                }
                if let Some(n) = minor.nodes.get_mut(frontier_id.clone()) {
                    n.query_text = pruned_sql.clone();
                }
            }

            let original_sql = minor
                .nodes
                .get(node_id.clone())
                .map(|n| n.query_text.as_str())
                .unwrap_or("");

            if result.source_sql == original_sql {
                debug!("PushdownPass: '{node_id}' unchanged (nothing pushed down)");
                outcomes.push((node_id.clone(), "unchanged (nothing pushed down)".to_string()));
                continue;
            }

            debug!(
                "PushdownPass: '{node_id}' rewritten ({} chars)",
                result.source_sql.len()
            );
            outcomes.push((
                node_id.clone(),
                format!("rewritten ({} chars)", result.source_sql.len()),
            ));

            {
                let node = dag.nodes.get_mut(node_id.clone()).ok_or_else(|| {
                    OptimizerError::Exec(format!(
                        "PushdownPass: node '{node_id}' missing from original DAG"
                    ))
                })?;
                node.query_text = result.source_sql.clone();
                node.schema = Some(Arc::clone(&result.source_schema));
            }
            if let Some(n) = minor.nodes.get_mut(node_id.clone()) {
                n.query_text = result.source_sql;
                n.schema = Some(result.source_schema);
            }

            rewrites += 1;
        }

        debug!("PushdownPass: cleaning up scratch DAG");
        cleanup_scratch_dag(&minor, &scratch_names, self.conn.as_ref()).await;

        debug!("PushdownPass: complete — {rewrites} rewrite(s) applied");

        let outcome = PassOutcome {
            // Pushdown analyses each TempTable's plan in place; it never
            // executes a candidate DAG of its own.
            dag_runs_used: 0,
            changes_applied: rewrites as u32,
            candidates_considered: temp_tables_count as u32,
            working_set_size: temp_tables_count as u32,
            iterations: Vec::new(),
            detail: PassDetail::Pushdown(PushdownDetail {
                temp_tables_count,
                rewrites_applied: rewrites,
                outcomes: outcomes
                    .iter()
                    .map(|(node_id, outcome)| PushdownOutcome {
                        node_id: node_id.clone(),
                        outcome: outcome.clone(),
                    })
                    .collect(),
            }),
        };

        self.explain_data = Some(PushdownExplainData {
            outcomes,
            rewrites_applied: rewrites,
        });

        Ok(outcome)
    }
}

impl<C, E> Explain for PushdownPass<C, E>
where
    C: Connector + Send + 'static + Sync,
    E: Executor<C> + Send + Sync,
{
    fn explain_label(&self) -> String {
        "PushdownPass".to_string()
    }

    fn explain(&self) -> String {
        let Some(data) = &self.explain_data else {
            return r#"<div class="panel"><p class="subtle">PushdownPass did not run.</p></div>"#
                .to_string();
        };

        let cards = render_card_grid(&[
            ("TempTables considered", data.outcomes.len().to_string()),
            ("Rewrites applied", data.rewrites_applied.to_string()),
        ]);

        let rows: Vec<Vec<String>> = data
            .outcomes
            .iter()
            .map(|(node_id, outcome)| vec![node_id.clone(), outcome.clone()])
            .collect();
        let table = render_ranked_table(&["TempTable", "Outcome"], &rows);

        format!(
            r#"<div class="section-stack">
        {cards}
        <div class="panel">
          <h2>Per-node outcome</h2>
          <div class="subtle">Each TempTable is processed deepest-first, so nodes closer to the sinks see the most specific filter/projection context before shallower nodes are updated. A node is skipped if it has no materializing frontier to push predicates/projections into, or if pushdown found nothing to change.</div>
          {table}
        </div>
      </div>"#
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectors::duckdb::{DuckDBConfig, DuckDBConnection};
    use crate::dag::{Dag, MaterializeMode, SourceNode, TransformNode};
    use crate::executor::{Executor, SimpleEngine};
    use crate::graph::Graph;
    use std::collections::HashMap;

    // ------------------------------------------------------------------
    // Helpers — real in-memory DuckDB connector + the real SimpleEngine, so
    // these tests exercise the exact same code path production runs.
    // ------------------------------------------------------------------

    async fn in_memory_conn() -> Arc<DuckDBConnection> {
        let config = DuckDBConfig::new_from_path(":memory:".to_string());
        DuckDBConnection::new(config).await.unwrap()
    }

    fn make_dag(nodes: Vec<TransformNode>) -> Dag {
        let mut graph = Graph::new(HashMap::new());
        for node in nodes {
            graph.add_node(node).unwrap();
        }
        Dag {
            db: "DuckDB".to_string(),
            nodes: graph,
            sources: vec![],
        }
    }

    fn node(id: &str, query: &str, mode: MaterializeMode, deps: &[&str]) -> TransformNode {
        TransformNode {
            id: id.to_string(),
            query_text: query.to_string(),
            materialize: mode,
            depends_on: deps.iter().map(|s| s.to_string()).collect::<HashSet<_>>(),
            schema: None,
        }
    }

    /// Create `orders` as a real table in `conn`, then run PushdownPass on a
    /// DAG whose only source is `orders`.
    async fn setup_orders_table(conn: &DuckDBConnection) {
        conn.execute(
            "CREATE TABLE orders AS SELECT \
                range AS order_id, \
                CASE WHEN range % 2 = 0 THEN 'US' ELSE 'EU' END AS region, \
                range * 1.5 AS amount, \
                'open' AS status \
             FROM range(20)"
                .to_string(),
        )
        .await
        .unwrap();
    }

    // ------------------------------------------------------------------
    // Integration tests
    // ------------------------------------------------------------------

    // DAG layout:
    //
    //   orders (source, real table)
    //       │
    //   staging (TempTable)   ← no TABLE downstream, no optimization needed
    //       ├──► summary (View)
    //       └──► report  (View)
    //
    // A TempTable whose only consumers are Views requires no pushdown — the
    // Views are not materialized, so there is nothing to optimize against.
    // The pass must leave the TempTable query unchanged.
    #[tokio::test]
    async fn test_temp_table_with_only_view_consumers_is_unchanged() {
        let conn = in_memory_conn().await;
        setup_orders_table(&conn).await;
        let engine = SimpleEngine::new(Arc::clone(&conn)).unwrap();

        let staging = node(
            "staging",
            "SELECT order_id, region, amount, status FROM orders",
            MaterializeMode::TempTable,
            &[],
        );
        let summary = node(
            "summary",
            "SELECT amount FROM staging WHERE amount > 0",
            MaterializeMode::View,
            &["staging"],
        );
        let report = node(
            "report",
            "SELECT amount FROM staging WHERE amount > 0",
            MaterializeMode::View,
            &["staging"],
        );

        let mut dag = make_dag(vec![staging, summary, report]);
        dag.sources = vec![SourceNode {
            name: "orders".to_string(),
            schema: Arc::new(duckdb::arrow::datatypes::Schema::empty()),
        }];
        let original = dag
            .nodes
            .get("staging".to_string())
            .unwrap()
            .query_text
            .clone();

        let mut pass = PushdownPass::new(Arc::clone(&conn), Arc::new(engine));
        pass.run(&mut dag).await.expect("pass should succeed");

        assert_eq!(
            dag.nodes.get("staging".to_string()).unwrap().query_text,
            original,
            "TempTable with only View consumers must not be rewritten"
        );
    }

    // DAG layout:
    //
    //   orders (source, real table)
    //       │
    //   staging (TempTable)   SELECT order_id, region, amount, status FROM orders
    //       │
    //   final_table (Table)   SELECT region, amount FROM staging WHERE region = 'US'
    //
    // There is a TABLE downstream, so the pass should push down the filter
    // `region = 'US'` that the Table applies, and prune `status` (never
    // referenced downstream) from staging's projection.
    #[tokio::test]
    async fn test_filter_and_projection_pushed_into_temp_table() {
        let conn = in_memory_conn().await;
        setup_orders_table(&conn).await;
        let engine = SimpleEngine::new(Arc::clone(&conn)).unwrap();

        let staging = node(
            "staging",
            "SELECT order_id, region, amount, status FROM orders",
            MaterializeMode::TempTable,
            &[],
        );
        let sink = node(
            "final_table",
            "SELECT region, amount FROM staging WHERE region = 'US'",
            MaterializeMode::Table,
            &["staging"],
        );

        let mut dag = make_dag(vec![staging, sink]);
        dag.sources = vec![SourceNode {
            name: "orders".to_string(),
            schema: Arc::new(duckdb::arrow::datatypes::Schema::empty()),
        }];

        let mut pass = PushdownPass::new(Arc::clone(&conn), Arc::new(engine));
        pass.run(&mut dag).await.expect("pass should succeed");

        let rewritten = dag
            .nodes
            .get("staging".to_string())
            .unwrap()
            .query_text
            .clone();

        assert!(
            rewritten.contains("'US'"),
            "filter predicate 'US' should be pushed into the TempTable; got: {}",
            rewritten
        );

        let outer_select = rewritten.split("FROM (").next().unwrap_or("");
        assert!(
            !outer_select.contains("status"),
            "column `status` should be absent from the outer SELECT projection; got: {}",
            rewritten
        );
    }

    // DAG layout:
    //
    //   orders (source, real table)
    //       │
    //   staging (TempTable)
    //       ├──► table_a (Table)   SELECT order_id, amount FROM staging WHERE region = 'US'
    //       └──► table_b (Table)   SELECT order_id, amount FROM staging WHERE region = 'EU'
    //
    // Two frontier Tables with *different* region filters.  The pass must OR
    // the two predicates so that both consumers' rows survive.
    #[tokio::test]
    async fn test_different_filters_across_table_consumers_are_pushed_as_or() {
        let conn = in_memory_conn().await;
        setup_orders_table(&conn).await;
        let engine = SimpleEngine::new(Arc::clone(&conn)).unwrap();

        let staging = node(
            "staging",
            "SELECT order_id, region, amount, status FROM orders",
            MaterializeMode::TempTable,
            &[],
        );
        let table_a = node(
            "table_a",
            "SELECT order_id, amount FROM staging WHERE region = 'US'",
            MaterializeMode::Table,
            &["staging"],
        );
        let table_b = node(
            "table_b",
            "SELECT order_id, amount FROM staging WHERE region = 'EU'",
            MaterializeMode::Table,
            &["staging"],
        );

        let mut dag = make_dag(vec![staging, table_a, table_b]);
        dag.sources = vec![SourceNode {
            name: "orders".to_string(),
            schema: Arc::new(duckdb::arrow::datatypes::Schema::empty()),
        }];

        let mut pass = PushdownPass::new(Arc::clone(&conn), Arc::new(engine));
        pass.run(&mut dag).await.expect("pass should succeed");

        let rewritten = dag
            .nodes
            .get("staging".to_string())
            .unwrap()
            .query_text
            .clone();

        assert!(
            rewritten.contains("'US'") && rewritten.contains("'EU'"),
            "both filter predicates should appear in the rewritten TempTable query; got: {}",
            rewritten
        );
        assert!(
            rewritten.contains("OR"),
            "filters from different Table consumers must be combined with OR; got: {}",
            rewritten
        );

        let outer_select = rewritten.split("FROM (").next().unwrap_or("");
        assert!(
            !outer_select.contains("status"),
            "column `status` should be absent from the outer SELECT projection; got: {}",
            rewritten
        );
    }

    // Regression test for a real-world failure: two landing-pad TempTables
    // where one's own query text references the *other* (not just its
    // frontier consumer's). `temp_b` is upstream of `temp_a`, and `temp_a`'s
    // query scans `temp_b` directly — neither is a real relation under its
    // original name at analysis time, so pushdown() must materialize the
    // *whole* graph minor as scratch tables up front (not one TempTable at a
    // time) or creating temp_a's own scratch table fails outright.
    //
    // DAG layout:
    //
    //   orders (source, real table)
    //       │
    //   temp_b (TempTable)   SELECT order_id, region, amount FROM orders
    //       │
    //   temp_a (TempTable)   SELECT order_id, region, amount FROM temp_b
    //       │
    //   sink (Table)         SELECT amount FROM temp_a WHERE region = 'US'
    #[tokio::test]
    async fn test_pushdown_pass_handles_temp_table_referencing_another_temp_table() {
        let conn = in_memory_conn().await;
        setup_orders_table(&conn).await;
        let engine = SimpleEngine::new(Arc::clone(&conn)).unwrap();

        let temp_b = node(
            "temp_b",
            "SELECT order_id, region, amount FROM orders",
            MaterializeMode::TempTable,
            &[],
        );
        let temp_a = node(
            "temp_a",
            "SELECT order_id, region, amount FROM temp_b",
            MaterializeMode::TempTable,
            &["temp_b"],
        );
        let sink = node(
            "sink",
            "SELECT amount FROM temp_a WHERE region = 'US'",
            MaterializeMode::Table,
            &["temp_a"],
        );

        let mut dag = make_dag(vec![temp_b, temp_a, sink]);
        dag.sources = vec![SourceNode {
            name: "orders".to_string(),
            schema: Arc::new(duckdb::arrow::datatypes::Schema::empty()),
        }];

        let mut pass = PushdownPass::new(Arc::clone(&conn), Arc::new(engine));
        pass.run(&mut dag)
            .await
            .expect("pass should succeed even when one TempTable's query references another");

        let rewritten = dag.nodes.get("temp_a".to_string()).unwrap().query_text.clone();
        assert!(
            rewritten.contains("'US'"),
            "filter should be pushed into temp_a; got: {rewritten}"
        );
    }

    // DAG layout:
    //
    //   orders (source, real table)
    //       │
    //   staging (TempTable)   SELECT order_id, region, amount, status FROM orders
    //       │
    //   table_a (Table)   SELECT region, amount FROM staging WHERE status = 'open'
    //
    // `status` is referenced only in the filter, never selected downstream.
    // The pass must keep `status` in staging's projection since the pushed
    // filter still needs it, even though DuckDB's own scan-level `Projections`
    // doesn't count it as "selected".
    #[tokio::test]
    async fn test_filter_only_column_survives_projection_pruning() {
        let conn = in_memory_conn().await;
        setup_orders_table(&conn).await;
        let engine = SimpleEngine::new(Arc::clone(&conn)).unwrap();

        let staging = node(
            "staging",
            "SELECT order_id, region, amount, status FROM orders",
            MaterializeMode::TempTable,
            &[],
        );
        let table_a = node(
            "table_a",
            "SELECT region, amount FROM staging WHERE status = 'open'",
            MaterializeMode::Table,
            &["staging"],
        );

        let mut dag = make_dag(vec![staging, table_a]);
        dag.sources = vec![SourceNode {
            name: "orders".to_string(),
            schema: Arc::new(duckdb::arrow::datatypes::Schema::empty()),
        }];

        let mut pass = PushdownPass::new(Arc::clone(&conn), Arc::new(engine));
        pass.run(&mut dag).await.expect("pass should succeed");

        let rewritten_node = dag.nodes.get("staging".to_string()).unwrap();
        let rewritten = rewritten_node.query_text.clone();

        assert!(
            rewritten.contains("status"),
            "column `status` (referenced only by a pushed-down filter) must \
             survive projection pruning in the rewritten TempTable query; got: {}",
            rewritten
        );

        let schema = rewritten_node
            .schema
            .as_ref()
            .expect("pushdown should have recorded a schema for staging");
        assert!(
            schema.field_with_name("status").is_ok(),
            "recorded schema for staging must include `status`; got fields: {:?}",
            schema.fields().iter().map(|f| f.name()).collect::<Vec<_>>()
        );
    }

    // ------------------------------------------------------------------
    // graph_minor tests
    // ------------------------------------------------------------------

    // DAG layout:
    //
    //   raw (source)
    //    │
    //   cleaned (View)   SELECT id, amount FROM raw WHERE amount > 0
    //    │
    //   summary (Table)  SELECT id, amount FROM cleaned
    //
    // After graph_minor:
    //   - `cleaned` should be gone (inlined into `summary`)
    //   - `summary` should still exist as a Table
    //   - The result DAG must contain no View nodes
    //   - `summary`'s query_text should no longer reference `cleaned` by name
    #[tokio::test]
    async fn test_graph_minor_single_view_inlined_into_table() {
        let source = SourceNode {
            name: "raw".to_string(),
            schema: Arc::new(duckdb::arrow::datatypes::Schema::empty()),
        };

        let cleaned = node(
            "cleaned",
            "SELECT id, amount FROM raw WHERE amount > 0",
            MaterializeMode::View,
            &[], // source nodes aren't tracked as graph deps
        );
        let summary = node(
            "summary",
            "SELECT id, amount FROM cleaned",
            MaterializeMode::Table,
            &["cleaned"],
        );

        let mut dag = make_dag(vec![cleaned, summary]);
        dag.sources = vec![source];

        let minor = graph_minor(&dag).await.expect("graph_minor should succeed");

        let view_count = minor
            .nodes
            .nodes()
            .filter(|n| matches!(n.materialize, MaterializeMode::View))
            .count();
        assert_eq!(view_count, 0, "result DAG must contain no View nodes");

        let summary_node = minor
            .nodes
            .get("summary".to_string())
            .expect("summary Table must still exist");
        assert!(
            matches!(summary_node.materialize, MaterializeMode::Table),
            "summary must remain a Table"
        );

        assert!(
            summary_node.query_text.contains("amount > 0"),
            "inlined query must contain the view's filter predicate; got: {}",
            summary_node.query_text
        );
        assert!(
            !summary_node
                .query_text
                .to_lowercase()
                .contains("from cleaned"),
            "inlined query must not reference the view as a bare FROM target; got: {}",
            summary_node.query_text
        );
    }

    // DAG layout:
    //
    //   raw (source)
    //    │
    //   step_one (View)   SELECT id, val FROM raw
    //    │
    //   step_two (View)   SELECT id, val FROM step_one WHERE val > 0
    //    │
    //   output (Table)    SELECT id, val FROM step_two
    //
    // graph_minor must iteratively inline both views.  After reduction:
    //   - Neither `step_one` nor `step_two` should remain.
    //   - `output` should be the only node and reference neither view by name.
    #[tokio::test]
    async fn test_graph_minor_chained_views_all_inlined() {
        let source = SourceNode {
            name: "raw".to_string(),
            schema: Arc::new(duckdb::arrow::datatypes::Schema::empty()),
        };

        let step_one = node(
            "step_one",
            "SELECT id, val FROM raw",
            MaterializeMode::View,
            &[],
        );
        let step_two = node(
            "step_two",
            "SELECT id, val FROM step_one WHERE val > 0",
            MaterializeMode::View,
            &["step_one"],
        );
        let output = node(
            "output",
            "SELECT id, val FROM step_two",
            MaterializeMode::Table,
            &["step_two"],
        );

        let mut dag = make_dag(vec![step_one, step_two, output]);
        dag.sources = vec![source];

        let minor = graph_minor(&dag).await.expect("graph_minor should succeed");

        let view_count = minor
            .nodes
            .nodes()
            .filter(|n| matches!(n.materialize, MaterializeMode::View))
            .count();
        assert_eq!(view_count, 0, "all View nodes must be eliminated");

        let output_node = minor
            .nodes
            .get("output".to_string())
            .expect("output Table must survive");

        assert!(
            output_node.query_text.contains("val > 0"),
            "inlined query must contain the chained filter predicate; got: {}",
            output_node.query_text
        );
        assert!(
            !output_node
                .query_text
                .to_lowercase()
                .contains("from step_one"),
            "inlined query must not reference step_one as a bare FROM target; got: {}",
            output_node.query_text
        );
        assert!(
            !output_node
                .query_text
                .to_lowercase()
                .contains("from step_two"),
            "inlined query must not reference step_two as a bare FROM target; got: {}",
            output_node.query_text
        );
    }

    // DAG layout:
    //
    //   raw (source)
    //    │
    //   orders (View)   SELECT id, amount FROM raw WHERE amount > 0
    //    │
    //   summary (Table)  SELECT o.id, o.amount, x.tag
    //                    FROM orders o
    //                    JOIN customer_orders x ON x.order_id = o.id
    //
    // `summary` references a view named `orders` AND an unrelated table whose
    // name merely *contains* "orders" as a substring (`customer_orders`). A
    // naive `str::replace("orders", ...)` would corrupt `customer_orders` by
    // replacing the embedded substring too. The AST-based inlining must only
    // touch the actual `orders` table reference.
    #[tokio::test]
    async fn test_graph_minor_does_not_corrupt_substring_matching_table_name() {
        let source = SourceNode {
            name: "raw".to_string(),
            schema: Arc::new(duckdb::arrow::datatypes::Schema::empty()),
        };

        let orders = node(
            "orders",
            "SELECT id, amount FROM raw WHERE amount > 0",
            MaterializeMode::View,
            &[],
        );
        let summary = node(
            "summary",
            "SELECT o.id, o.amount, x.tag FROM orders o \
             JOIN customer_orders x ON x.order_id = o.id",
            MaterializeMode::Table,
            &["orders"],
        );

        let mut dag = make_dag(vec![orders, summary]);
        dag.sources = vec![source];

        let minor = graph_minor(&dag).await.expect("graph_minor should succeed");

        let summary_node = minor
            .nodes
            .get("summary".to_string())
            .expect("summary Table must still exist");

        // The unrelated table must survive completely intact — not have its
        // name mangled by a substring match against "orders".
        assert!(
            summary_node.query_text.contains("customer_orders"),
            "unrelated table 'customer_orders' must not be corrupted; got: {}",
            summary_node.query_text
        );
        // The view's own filter must be inlined.
        assert!(
            summary_node.query_text.contains("amount > 0"),
            "inlined query must contain the view's filter predicate; got: {}",
            summary_node.query_text
        );
    }

    // ------------------------------------------------------------------
    // pushdown() unit tests (called directly, not through the whole pass)
    // ------------------------------------------------------------------

    /// Build a minimal DAG with one raw source, one TempTable, and one or more
    /// Table sinks, with schemas resolved against a real in-memory DuckDB
    /// connection so `pushdown` can be called directly.
    async fn orders_dag(
        conn: &Arc<DuckDBConnection>,
        staging_query: &str,
        sinks: Vec<(&str, &str)>, // (node_id, query_text)
    ) -> Dag {
        let staging = node("staging", staging_query, MaterializeMode::TempTable, &[]);

        let mut all_nodes = vec![staging];
        for (id, query) in &sinks {
            all_nodes.push(node(id, query, MaterializeMode::Table, &["staging"]));
        }

        let mut dag = make_dag(all_nodes);
        dag.sources = vec![SourceNode {
            name: "orders".to_string(),
            schema: Arc::new(duckdb::arrow::datatypes::Schema::empty()),
        }];
        let engine: SimpleEngine<DuckDBConnection> =
            SimpleEngine::new(Arc::clone(conn)).expect("SimpleEngine::new should not fail");
        engine
            .resolve_schemas(&mut dag)
            .await
            .expect("resolve_schemas should succeed in test");
        dag
    }

    // DAG layout:
    //
    //   orders (raw source, real table)
    //       │
    //   staging (TempTable)   SELECT * FROM orders
    //       │
    //   us_orders (Table)     SELECT order_id, amount FROM staging WHERE region = 'US'
    //
    // The single frontier Table filters on `region = 'US'` and projects only
    // `order_id` and `amount`.  The `pushdown` function should produce a plan
    // for `staging` that applies that filter and prunes to those columns
    // (`region` survives for the filter; `status` is unreferenced).
    #[tokio::test]
    async fn test_pushdown_single_table_filter_and_projection() {
        let conn = in_memory_conn().await;
        setup_orders_table(&conn).await;

        let dag = orders_dag(
            &conn,
            "SELECT order_id, region, amount, status FROM orders",
            vec![(
                "us_orders",
                "SELECT order_id, amount FROM staging WHERE region = 'US'",
            )],
        )
        .await;

        let scratch_names = materialize_scratch_dag(&dag, conn.as_ref())
            .await
            .expect("materialize_scratch_dag should succeed");

        let PushdownResult {
            source_sql: sql, ..
        } = pushdown(&dag, "staging", conn.as_ref(), &scratch_names)
            .await
            .expect("pushdown should succeed");

        cleanup_scratch_dag(&dag, &scratch_names, conn.as_ref()).await;

        assert!(
            sql.contains("US"),
            "optimized SQL must contain the filter predicate 'US'; got:\n{sql}"
        );
        let outer_select = sql.split("FROM (").next().unwrap_or("");
        assert!(
            !outer_select.contains("status"),
            "column `status` should be absent from the outer SELECT projection; got:\n{sql}"
        );
    }

    // DAG layout:
    //
    //   orders (raw source, real table)
    //       │
    //   staging (TempTable)   SELECT * FROM orders
    //       ├──► us_orders (Table)   SELECT order_id, amount FROM staging WHERE region = 'US'
    //       └──► eu_orders (Table)   SELECT order_id, amount FROM staging WHERE region = 'EU'
    //
    // Two frontier Tables with *different* region filters.  The `pushdown`
    // function must OR the two predicates so that both consumers' rows survive.
    #[tokio::test]
    async fn test_pushdown_multiple_tables_filters_combined_with_or() {
        let conn = in_memory_conn().await;
        setup_orders_table(&conn).await;

        let dag = orders_dag(
            &conn,
            "SELECT order_id, region, amount, status FROM orders",
            vec![
                (
                    "us_orders",
                    "SELECT order_id, amount FROM staging WHERE region = 'US'",
                ),
                (
                    "eu_orders",
                    "SELECT order_id, amount FROM staging WHERE region = 'EU'",
                ),
            ],
        )
        .await;

        let scratch_names = materialize_scratch_dag(&dag, conn.as_ref())
            .await
            .expect("materialize_scratch_dag should succeed");

        let PushdownResult {
            source_sql: sql, ..
        } = pushdown(&dag, "staging", conn.as_ref(), &scratch_names)
            .await
            .expect("pushdown should succeed");

        cleanup_scratch_dag(&dag, &scratch_names, conn.as_ref()).await;

        assert!(sql.contains("US"), "SQL must contain the 'US' filter arm; got:\n{sql}");
        assert!(sql.contains("EU"), "SQL must contain the 'EU' filter arm; got:\n{sql}");
        let outer_select = sql.split("FROM (").next().unwrap_or("");
        assert!(
            !outer_select.contains("status"),
            "column `status` should be absent from the outer SELECT projection; got:\n{sql}"
        );
    }

    // ------------------------------------------------------------------
    // Dead-column pruning unit tests
    // ------------------------------------------------------------------

    #[test]
    fn test_prune_dead_source_columns_removes_unused_nested_column() {
        let sql = r#"SELECT x."a" FROM (SELECT "a", "b", "c" FROM "staging") AS x"#;
        let pruned = prune_dead_source_columns(
            sql,
            "staging",
            &["a".to_string()],
            DialectType::DuckDB,
        )
        .expect("should prune");
        assert!(pruned.contains('a'));
        assert!(!pruned.contains('b'), "got: {pruned}");
        assert!(!pruned.contains('c'), "got: {pruned}");
    }

    // Regression test for a real benchmark failure: a nested SELECT that
    // computes an *aggregate* over a source column (not a plain pass-through)
    // must never be pruned based on the aggregate's own output alias, since
    // that alias (`mean_temp`) will never match any of `staging`'s physical
    // column names (`avg_temp`) — a naive alias-based check would wrongly
    // drop it even though it's the CTE's entire reason for existing.
    #[test]
    fn test_prune_dead_source_columns_never_drops_computed_aggregate_columns() {
        let sql = r#"WITH stats AS (
            SELECT device_id, AVG(avg_temp) AS mean_temp, STDDEV(avg_temp) AS std_temp
            FROM "staging"
            GROUP BY device_id
        )
        SELECT h.device_id, s.mean_temp, s.std_temp
        FROM "staging" AS h JOIN stats AS s USING (device_id)"#;

        // `keep_cols` only contains `device_id` and `avg_temp` — the CTE's
        // computed `mean_temp`/`std_temp` outputs are correctly absent from
        // this list (they aren't physical columns of `staging` at all), but
        // that must not cause them to be pruned from the CTE's SELECT list.
        let pruned = prune_dead_source_columns(
            sql,
            "staging",
            &["device_id".to_string(), "avg_temp".to_string()],
            DialectType::DuckDB,
        )
        .expect("should parse/regenerate");

        assert!(
            pruned.contains("mean_temp") && pruned.contains("std_temp"),
            "computed aggregate columns must never be pruned; got: {pruned}"
        );
    }

    // Regression test for a real benchmark failure: `SELECT pp.*, ...` needs
    // every column of `staging` (via the star), but `get_column_names` never
    // enumerates the physical columns a star expands to, so a naive
    // safety-net that only looks at `Expression::Column` nodes would miss
    // this entirely and let genuinely-needed columns get pruned away.
    #[test]
    fn test_collect_unprunable_source_columns_bails_out_on_star() {
        let sql = r#"SELECT pp.*, ROW_NUMBER() OVER (ORDER BY x) AS rn FROM "staging" AS pp"#;
        let result = collect_unprunable_source_columns(sql, "staging", DialectType::DuckDB);
        assert!(
            result.is_none(),
            "a star touching a scan of source must force 'need all columns'; got: {result:?}"
        );
    }

    // Regression test for a real benchmark failure: `avg(avg_voltage) AS
    // avg_voltage` is a single, unjoined (i.e. "prunable"-shaped) scan of
    // `staging`, but the select-list item computes an aggregate rather than
    // passing a column straight through — `avg_voltage` (the argument) must
    // still be treated as needed even though the select's *shape* would
    // otherwise be eligible for pruning, and even though the output alias
    // happens to share its name with the underlying physical column.
    #[test]
    fn test_collect_unprunable_source_columns_keeps_aggregate_arguments_in_prunable_select() {
        let sql = r#"SELECT region, AVG(avg_voltage) AS avg_voltage FROM "staging" GROUP BY region"#;
        let cols = collect_unprunable_source_columns(sql, "staging", DialectType::DuckDB)
            .expect("no star present, should return a concrete set");
        assert!(
            cols.contains("avg_voltage"),
            "aggregate argument must be kept regardless; got: {cols:?}"
        );
        assert!(
            cols.contains("region"),
            "GROUP BY column must be kept regardless; got: {cols:?}"
        );
    }

    #[test]
    fn test_prune_dead_source_columns_never_touches_outermost_select() {
        // staging is scanned directly at the top level here — nothing "above"
        // this statement to prune against, so it must be left untouched even
        // though `keep_cols` is narrower than what's selected.
        let sql = r#"SELECT "a", "b" FROM "staging""#;
        let pruned = prune_dead_source_columns(
            sql,
            "staging",
            &["a".to_string()],
            DialectType::DuckDB,
        )
        .expect("should parse/regenerate");
        assert!(pruned.contains('b'), "outermost SELECT must be untouched; got: {pruned}");
    }

    #[test]
    fn test_prune_dead_source_columns_skips_joined_scan() {
        // staging appears in a JOIN here — the conservative pruning pass must
        // leave it alone rather than risk ambiguous column attribution.
        let sql = r#"SELECT x."a" FROM (SELECT "a", "b" FROM "staging" JOIN "other" ON "a" = "id") AS x"#;
        let pruned = prune_dead_source_columns(
            sql,
            "staging",
            &["a".to_string()],
            DialectType::DuckDB,
        )
        .expect("should parse/regenerate");
        assert!(pruned.contains('b'), "joined scan must be left untouched; got: {pruned}");
    }

    // Regression test for a real benchmark failure: a bare pass-through
    // column selected inside a UNION ALL branch (itself the root statement)
    // must be treated exactly like the plain-root-Select case — never
    // classified as "prunable" — since `prune_dead_source_columns` never
    // rewrites a top-level branch's own projection list.
    #[test]
    fn test_collect_unprunable_source_columns_keeps_union_branch_passthrough_columns() {
        let sql = r#"SELECT "a", "b" FROM "staging"
                     UNION ALL
                     SELECT "a", "b" FROM "other""#;
        let cols = collect_unprunable_source_columns(sql, "staging", DialectType::DuckDB)
            .expect("no star present, should return a concrete set");
        assert!(
            cols.contains("b"),
            "a bare pass-through column in a top-level UNION branch must survive; got: {cols:?}"
        );
    }

    // Same shape but as the actual rewriter: since the branch is top-level,
    // pruning must leave its projection list untouched even though
    // `keep_cols` is narrower than what's selected.
    #[test]
    fn test_prune_dead_source_columns_never_touches_union_branch() {
        let sql = r#"SELECT "a", "b" FROM "staging"
                     UNION ALL
                     SELECT "a", "b" FROM "other""#;
        let pruned = prune_dead_source_columns(
            sql,
            "staging",
            &["a".to_string()],
            DialectType::DuckDB,
        )
        .expect("should parse/regenerate");
        assert!(pruned.contains('b'), "top-level UNION branch must be untouched; got: {pruned}");
    }

    // A nested (non-top-level) single-unjoined scan of `source` inside a
    // UNION branch that itself lives inside a CTE must still get pruned —
    // this is the "real" pruning opportunity a UNION-aware rewriter unlocks,
    // not just a safety net.
    #[test]
    fn test_prune_dead_source_columns_prunes_nested_scan_inside_cte_union() {
        let sql = r#"WITH combined AS (
            SELECT "a", "b" FROM "staging"
            UNION ALL
            SELECT "a", "b" FROM "other"
        )
        SELECT c."a" FROM combined AS c"#;
        let pruned = prune_dead_source_columns(
            sql,
            "staging",
            &["a".to_string()],
            DialectType::DuckDB,
        )
        .expect("should parse/regenerate");
        assert!(
            pruned.contains(r#"SELECT "a" FROM "staging""#),
            "nested scan inside a CTE's UNION branch must be pruned to just 'a' \
             (the 'other' branch's own 'b' column is untouched); got: {pruned}"
        );
    }

    // The connector's own pushdown analysis (`DuckDB` EXPLAIN-based
    // projection discovery) sees through this exact CTE/UNION nesting and
    // will correctly report `b` as unused overall. The classifier must not
    // re-flag `b` as unprunable here (only the two tests above's top-level
    // branches deserve that), or genuine pruning opportunities regress.
    #[test]
    fn test_collect_unprunable_source_columns_treats_nested_cte_union_scan_as_prunable() {
        let sql = r#"WITH combined AS (
            SELECT "a", "b" FROM "staging"
            UNION ALL
            SELECT "a", "b" FROM "other"
        )
        SELECT c."a" FROM combined AS c"#;
        let cols = collect_unprunable_source_columns(sql, "staging", DialectType::DuckDB)
            .expect("no star present, should return a concrete set");
        assert!(
            !cols.contains("b"),
            "nested scan inside a CTE's UNION branch is prunable; got: {cols:?}"
        );
    }

    // Regression test: a filter predicate the connector can't attribute
    // cleanly (e.g. because it references a construct our bundled SQL parser
    // can't parse) must not silently vanish from the keep-list — the caller
    // must be told to fail closed instead.
    #[test]
    fn test_filter_referenced_columns_none_on_parse_failure() {
        assert!(filter_referenced_columns("((((", DialectType::DuckDB).is_none());
        assert_eq!(
            filter_referenced_columns("is_active", DialectType::DuckDB),
            Some(vec!["is_active".to_string()])
        );
    }
}
