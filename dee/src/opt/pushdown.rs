use async_trait::async_trait;
use duckdb::arrow::datatypes::SchemaRef;
use log::{debug, trace, warn};
use polyglot_sql::{
    dialects::DialectType,
    expressions::{Expression, JoinKind, Null, Select},
    traversal::ExpressionWalk,
};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use crate::{
    connectors::{Connector, PushdownInfo},
    dag::MaterializeMode,
    executor::Executor,
    opt::common::dialect_for_db,
    opt::{
        Dag, Optimization, OptimizerError,
        explain::{render_card_grid, render_ranked_table},
        report::{PassDetail, PassOutcome, PushdownDetail, PushdownOutcome},
        step::{
            OptimizationType, RegisterContext, StepContext, StepOutcome, StepPhase,
        },
        store::Registration,
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
// A TempTable's schema is pruned to only the columns its downstream frontier
// queries actually need. Deciding *which* columns those are cannot be
// delegated to the connector's plan-level analysis: the planner is free to
// report a column as unread because some computation over it is itself dead
// (an unused aggregate, say), while the frontier's SQL *text* still names
// that column somewhere we don't rewrite. Pruning on that basis produces a
// TempTable whose consumers no longer bind.
//
// So the keep-list is derived from the frontier's own text instead, after
// running the dead-column elimination below over it. The pass rewrites the
// query so that every *nested* relation (a derived table, a CTE body, a
// branch of a `UNION ALL`) projects only the columns something above it
// actually references, propagating that top-down to fixpoint. The outermost
// statement's projection list is never touched — that is the frontier's own
// output contract. Whatever survives in a scan of the TempTable after that is
// exactly what the TempTable must keep, so the rewritten text and the pruned
// schema are consistent by construction.
//
// This matters most after `graph_minor` inlines views: a view exposing
// twenty columns to a consumer that reads three leaves seventeen dead column
// references behind, and eliminating them is what lets the TempTable narrow.
// ---------------------------------------------------------------------------

/// The set of output column names something above this relation references,
/// or `None` when that is unknowable (a `*` is in play, or the relation's own
/// row shape is a contract we must not touch) and every column must survive.
type Needed<'a> = Option<&'a HashSet<String>>;

/// The name a select-list item contributes to its relation's output: the
/// alias if there is one, otherwise the bare column name. `None` for anything
/// whose output name we can't read off the syntax (a `*`, or an unaliased
/// computed expression whose name the dialect derives itself) — those items
/// are never dropped.
fn output_name(expr: &Expression) -> Option<String> {
    match expr {
        Expression::Alias(alias) => Some(alias.alias.name.clone()),
        Expression::Column(col) => Some(col.name.name.clone()),
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

/// `true` if a star appears anywhere inside `expr`.
fn contains_star(expr: &Expression) -> bool {
    expr.dfs().any(|e| matches!(e, Expression::Star(_)))
}

/// `true` if `table` is a reference to the DAG node `node_id`.
///
/// Node IDs can be one-, two-, or three-part quoted identifiers such as
/// `"warehouse"."main"."stg_orders"`. Matching compares from the right and
/// only on the parts both sides actually spell out, so an unqualified
/// `stg_orders` in a query matches the node, while a *different* schema's
/// `"warehouse"."raw"."stg_orders"` does not.
fn table_ref_matches(table: &polyglot_sql::expressions::TableRef, node_id: &str) -> bool {
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

/// `true` if any of `select`'s FROM/JOIN targets is a direct reference to
/// `source_id` (regardless of how many other sources/joins are present).
fn select_scans_source(select: &Select, source_id: &str) -> bool {
    let is_match =
        |e: &Expression| matches!(e, Expression::Table(t) if table_ref_matches(t, source_id));
    select
        .from
        .as_ref()
        .map(|f| f.expressions.iter().any(is_match))
        .unwrap_or(false)
        || select.joins.iter().any(|j| is_match(&j.this))
}

/// Every column name referenced anywhere inside `expr`, plus the identifiers
/// named by any `JOIN ... USING (...)` (which are column references the AST
/// stores as bare identifiers) and by `SELECT * EXCLUDE (...)`.
///
/// This is deliberately over-broad: it does not attribute a name to the
/// relation it came from, so a name belonging to a sibling source keeps a
/// same-named column alive in this one. That costs a little pruning and never
/// correctness — the reverse mistake would drop a live column.
fn names_referenced(expr: &Expression) -> HashSet<String> {
    let mut out: HashSet<String> = HashSet::new();
    for node in expr.dfs() {
        match node {
            Expression::Column(col) => {
                out.insert(col.name.name.clone());
            }
            Expression::Select(select) => {
                for join in &select.joins {
                    out.extend(join.using.iter().map(|i| i.name.clone()));
                }
            }
            Expression::Star(star) => {
                if let Some(except) = &star.except {
                    out.extend(except.iter().map(|i| i.name.clone()));
                }
            }
            _ => {}
        }
    }
    out
}

/// The names `select` references outside of its own select-list items --
/// WHERE, GROUP BY, HAVING, QUALIFY, ORDER BY, join conditions, and so on.
///
/// A select-list item whose output name appears here is never dropped: those
/// clauses may refer to it by its output alias (DuckDB resolves lateral
/// column aliases in WHERE as well as in GROUP BY/HAVING/ORDER BY), and
/// dropping the item would leave the reference dangling.
fn names_referenced_outside_select_list(select: &Select) -> HashSet<String> {
    let mut trimmed = select.clone();
    trimmed.expressions.clear();
    names_referenced_in_own_scope(&trimmed)
}

/// The names `select` references *in its own scope*: its select list, WHERE,
/// GROUP BY, ORDER BY, join conditions and so on, but not the interior of the
/// relations it reads from.
///
/// Those interiors are separate scopes — a derived table's own `SELECT a, b`
/// says nothing about whether the query above it still needs `b`, and folding
/// them in would make every column look live and nothing would ever prune.
/// CTE bodies are excluded for the same reason (they are narrowed against the
/// whole statement instead, see [`dce_ctes`]).
fn names_referenced_in_own_scope(select: &Select) -> HashSet<String> {
    let mut trimmed = select.clone();
    trimmed.with = None;
    if let Some(from) = trimmed.from.as_mut() {
        for e in from.expressions.iter_mut() {
            *e = Expression::Null(Null);
        }
    }
    for join in trimmed.joins.iter_mut() {
        // The join *condition* is part of this scope; the relation isn't.
        join.this = Expression::Null(Null);
    }
    names_referenced(&Expression::Select(Box::new(trimmed)))
}

/// `true` if any source of `select` can see another source's columns — a
/// LATERAL/APPLY join, or a lateral derived table. Sibling relations are
/// otherwise independent scopes, and this is the exception that makes
/// [`names_referenced_in_own_scope`] too narrow to prune against.
fn has_lateral_source(select: &Select) -> bool {
    let lateral_join = select.joins.iter().any(|j| {
        matches!(
            j.kind,
            JoinKind::Lateral | JoinKind::LeftLateral | JoinKind::CrossApply | JoinKind::OuterApply
        )
    });
    let lateral_from = select
        .from
        .as_ref()
        .map(|f| {
            f.expressions
                .iter()
                .any(|e| matches!(e, Expression::Subquery(sub) if sub.lateral))
        })
        .unwrap_or(false);
    let lateral_join_target = select
        .joins
        .iter()
        .any(|j| matches!(&j.this, Expression::Subquery(sub) if sub.lateral));
    lateral_join || lateral_from || lateral_join_target
}

/// `true` if `select` refers to its own output columns *by position* --
/// `GROUP BY 1, 2`, `ORDER BY 2 DESC`, and friends.
///
/// Dropping a select-list item renumbers everything after it, so a select
/// that counts positions has to keep its projection exactly as written even
/// when some of it is dead.
fn has_positional_references(select: &Select) -> bool {
    let is_ordinal = |e: &Expression| {
        matches!(
            e,
            Expression::Literal(lit) if matches!(**lit, polyglot_sql::expressions::Literal::Number(_))
        )
    };
    let group_by = select
        .group_by
        .as_ref()
        .map(|g| g.expressions.iter().any(is_ordinal))
        .unwrap_or(false);
    let order_by = select
        .order_by
        .as_ref()
        .map(|o| o.expressions.iter().any(|ord| is_ordinal(&ord.this)))
        .unwrap_or(false);
    let distinct_on = select
        .distinct_on
        .as_ref()
        .map(|d| d.iter().any(is_ordinal))
        .unwrap_or(false);
    group_by || order_by || distinct_on
}

/// Run dead-column elimination over a whole statement, to fixpoint.
///
/// The statement's own output columns are all live by definition, so only
/// nested relations are ever narrowed. Iterating lets a column dropped from
/// an outer relation make the reference that fed it dead in turn, one level
/// per pass; the loop stops as soon as a pass changes nothing.
fn eliminate_dead_columns(mut expr: Expression) -> Expression {
    const MAX_PASSES: usize = 8;
    for _ in 0..MAX_PASSES {
        let before = expr.clone();
        dce_relation(&mut expr, None);
        if expr == before {
            break;
        }
    }
    expr
}

/// Narrow one relation (a statement, derived table, CTE body, or set-op) to
/// `needed`, then recurse into the relations it reads from.
fn dce_relation(expr: &mut Expression, needed: Needed) {
    match expr {
        Expression::Subquery(sub) => dce_relation(&mut sub.this, needed),
        Expression::Union(_) | Expression::Intersect(_) | Expression::Except(_) => {
            dce_set_operation(expr, needed)
        }
        Expression::Select(select) => {
            dce_select_list(select, needed);
            dce_children(select);
        }
        Expression::JoinedTable(joined) => {
            dce_relation(&mut joined.left, needed);
            for join in joined.joins.iter_mut() {
                dce_relation(&mut join.this, needed);
            }
        }
        _ => {}
    }
}

/// Drop `select`'s select-list items that nothing references.
///
/// Left alone entirely when the output row shape is itself load-bearing: a
/// `*` (whose columns can't be named, so nothing can be proven dead),
/// `DISTINCT`/`DISTINCT ON` (dropping a column changes which rows survive
/// deduplication), a positional `GROUP BY`/`ORDER BY` (whose ordinals would
/// renumber, see [`has_positional_references`]), or a `needed` of `None`.
fn dce_select_list(select: &mut Select, needed: Needed) {
    let Some(needed) = needed else {
        return;
    };
    if select_has_star(select)
        || select.distinct
        || select.distinct_on.is_some()
        || has_positional_references(select)
    {
        return;
    }
    let keep_regardless = names_referenced_outside_select_list(select);
    let original = select.expressions.clone();
    select.expressions.retain(|item| match output_name(item) {
        Some(name) => needed.contains(&name) || keep_regardless.contains(&name),
        // No readable output name — can't prove it dead, so keep it.
        None => true,
    });
    if select.expressions.is_empty() {
        // Never emit an empty select list.
        select.expressions = vec![original.into_iter().next().unwrap_or_else(|| {
            Expression::Literal(Box::new(polyglot_sql::expressions::Literal::Number(
                "1".to_string(),
            )))
        })];
    }
}

/// Recurse into every relation `select` reads from — FROM entries, join
/// targets, and its own CTE bodies — carrying the set of names `select`
/// references.
///
/// The same (over-broad) set goes to every source: it is everything this
/// select could possibly read, so a column outside it is dead in all of them.
fn dce_children(select: &mut Select) {
    let child_needed: Option<HashSet<String>> = if select_has_star(select) {
        // A star's columns can't be named, so nothing under it is provably
        // dead.
        None
    } else if has_lateral_source(select) {
        // A lateral source reads a sibling's columns from inside its own
        // body, where this scope can't see the reference.
        None
    } else {
        Some(names_referenced_in_own_scope(select))
    };
    let child_needed = child_needed.as_ref();

    // A NATURAL join matches on whatever column names the two sides happen to
    // share, so narrowing either side changes which rows join.
    let natural = select.joins.iter().any(|j| {
        matches!(
            j.kind,
            JoinKind::Natural | JoinKind::NaturalLeft | JoinKind::NaturalRight | JoinKind::NaturalFull
        )
    });
    let from_needed = if natural { None } else { child_needed };

    if let Some(from) = select.from.as_mut() {
        for e in from.expressions.iter_mut() {
            dce_relation(e, from_needed);
        }
    }
    for join in select.joins.iter_mut() {
        dce_relation(&mut join.this, from_needed);
    }

    dce_ctes(select);
}

/// Narrow the bodies of `select`'s own CTEs.
///
/// A CTE is referenced by name from anywhere in the statement rather than
/// from one syntactic position, so what it must keep is everything the rest
/// of the statement references — computed with that CTE's own body swapped
/// out, so a name only the body itself uses doesn't keep the body's output
/// alive. Later CTEs are processed first, since a CTE can only be referenced
/// by ones defined after it.
fn dce_ctes(select: &mut Select) {
    let Some(mut with) = select.with.take() else {
        return;
    };
    // A recursive CTE refers to itself from inside its own body, which is
    // exactly the reference the swap below hides, so its columns can't be
    // reasoned about this way at all.
    if with.recursive {
        for cte in with.ctes.iter_mut() {
            let mut body = std::mem::replace(&mut cte.this, Expression::Null(Null));
            dce_relation(&mut body, None);
            cte.this = body;
        }
        select.with = Some(with);
        return;
    }

    // Any star anywhere else in the statement could be selecting the CTE's
    // columns wholesale, and column-aliased CTEs (`cte(a, b) AS ...`) bind
    // positionally, so in either case nothing may be dropped.
    let star_elsewhere = {
        let mut body = select.clone();
        body.with = None;
        contains_star(&Expression::Select(Box::new(body)))
            || with.ctes.iter().any(|c| contains_star(&c.this))
    };

    for i in (0..with.ctes.len()).rev() {
        let mut body = std::mem::replace(&mut with.ctes[i].this, Expression::Null(Null));

        let needed: Option<HashSet<String>> =
            if star_elsewhere || !with.ctes[i].columns.is_empty() {
                None
            } else {
                let mut names = {
                    let mut outer = select.clone();
                    outer.with = None;
                    names_referenced(&Expression::Select(Box::new(outer)))
                };
                for (j, cte) in with.ctes.iter().enumerate() {
                    if j != i {
                        names.extend(names_referenced(&cte.this));
                    }
                }
                Some(names)
            };

        dce_relation(&mut body, needed.as_ref());
        with.ctes[i].this = body;
    }

    select.with = Some(with);
}

/// Narrow a `UNION`/`INTERSECT`/`EXCEPT`.
///
/// Branches line up positionally, so a column can only be dropped from one
/// branch if it is dropped from every branch at the same position — and only
/// when the operation compares rows column-by-column nowhere: a deduplicating
/// `UNION`, `INTERSECT` or `EXCEPT` would change its result if a column
/// vanished, as would DuckDB's `UNION BY NAME`. In those cases the branch
/// projections stay exactly as written and only their own sources are
/// narrowed.
fn dce_set_operation(expr: &mut Expression, needed: Needed) {
    let prunable = matches!(expr, Expression::Union(u) if u.all && !u.distinct && !u.by_name);

    // The set operation's own ORDER BY refers to the branches' output names.
    let mut needed_here: Option<HashSet<String>> = needed.cloned();
    if let (Some(set), Expression::Union(u)) = (needed_here.as_mut(), &*expr)
        && let Some(order_by) = &u.order_by
    {
        for ordered in &order_by.expressions {
            set.extend(names_referenced(&ordered.this));
        }
    }

    let mut branches: Vec<&mut Expression> = Vec::new();
    collect_set_operation_branches(expr, &mut branches);

    let branch_needed = if prunable { needed_here.clone() } else { None };
    if let Some(needed_here) = branch_needed.as_ref() {
        prune_set_operation_branches(&mut branches, needed_here);
    }

    for branch in branches {
        // The branch's own list has already been handled positionally above;
        // recursing with `None` narrows its sources without touching it again.
        dce_relation(branch, None);
    }
}

/// Collect every leaf branch of a chain of set operations, in left-to-right
/// order, unwrapping the parentheses a branch may be written with.
fn collect_set_operation_branches<'a>(
    expr: &'a mut Expression,
    out: &mut Vec<&'a mut Expression>,
) {
    match expr {
        Expression::Union(u) => {
            let (left, right) = (&mut u.left, &mut u.right);
            collect_set_operation_branches(left, out);
            collect_set_operation_branches(right, out);
        }
        Expression::Intersect(i) => {
            let (left, right) = (&mut i.left, &mut i.right);
            collect_set_operation_branches(left, out);
            collect_set_operation_branches(right, out);
        }
        Expression::Except(e) => {
            let (left, right) = (&mut e.left, &mut e.right);
            collect_set_operation_branches(left, out);
            collect_set_operation_branches(right, out);
        }
        Expression::Subquery(sub) => collect_set_operation_branches(&mut sub.this, out),
        other => out.push(other),
    }
}

/// Drop the same select-list positions from every branch of a set operation.
///
/// The output names of the whole operation come from its first branch, so
/// that branch decides which positions are live. Nothing is dropped unless
/// every branch is a plain `SELECT` of the same arity with no star and no
/// `DISTINCT`, since otherwise the positions don't line up or the row shape
/// is load-bearing.
fn prune_set_operation_branches(branches: &mut [&mut Expression], needed: &HashSet<String>) {
    let mut selects: Vec<&Select> = Vec::new();
    for branch in branches.iter() {
        match branch {
            Expression::Select(select) => selects.push(select),
            _ => return,
        }
    }
    let Some(first) = selects.first() else {
        return;
    };
    let arity = first.expressions.len();
    if selects.iter().any(|s| {
        s.expressions.len() != arity
            || select_has_star(s)
            || s.distinct
            || s.distinct_on.is_some()
            || has_positional_references(s)
    }) {
        return;
    }

    let keep_regardless: HashSet<String> = selects
        .iter()
        .flat_map(|s| names_referenced_outside_select_list(s))
        .collect();

    let keep: Vec<bool> = first
        .expressions
        .iter()
        .map(|item| match output_name(item) {
            Some(name) => needed.contains(&name) || keep_regardless.contains(&name),
            None => true,
        })
        .collect();
    if keep.iter().all(|k| *k) || !keep.iter().any(|k| *k) {
        return;
    }

    for branch in branches.iter_mut() {
        if let Expression::Select(select) = branch {
            let mut i = 0;
            select.expressions.retain(|_| {
                let k = keep[i];
                i += 1;
                k
            });
        }
    }
}

/// The predicate a single frontier node applies to `source`, or `None` when
/// that node needs every row.
///
/// `scans` is one entry per scan of `source` in that node's plan. Within one
/// scan the reported filters are conjuncts — the scan keeps a row only if
/// all of them hold. *Across* scans they are alternatives: a self-join whose
/// two sides filter on different regions needs the rows either side reads, so
/// the node's predicate is their disjunction. AND-ing them instead would ask
/// for rows satisfying both at once, which for that example is none at all.
///
/// A scan with no filters reads the relation whole, so the node needs every
/// row and no predicate can be pushed on its behalf; likewise if a predicate
/// can't be parsed, since then we can't tell which columns it needs to keep.
fn node_predicate(scans: &[PushdownInfo], n_id: &str, dialect: DialectType) -> Option<String> {
    if scans.is_empty() {
        return None;
    }
    let mut conjunctions: Vec<String> = Vec::new();
    for scan in scans {
        if scan.filters.is_empty() {
            trace!("  frontier '{n_id}': a scan of the source applies no filter");
            return None;
        }
        for f in &scan.filters {
            if filter_referenced_columns(f, dialect).is_none() {
                warn!(
                    "pushdown: couldn't parse filter predicate '{f}' for frontier '{n_id}', \
                     assuming all rows needed"
                );
                return None;
            }
        }
        conjunctions.push(scan.filters.join(" AND "));
    }
    match conjunctions.len() {
        1 => conjunctions.pop(),
        _ => Some(
            conjunctions
                .into_iter()
                .map(|c| format!("({c})"))
                .collect::<Vec<_>>()
                .join(" OR "),
        ),
    }
}

/// Every column of `source_id` that `root` still references, after dead-column
/// elimination has run over it.
///
/// Returns `None` when a `SELECT *`/`table.*` sits on a scan of `source_id`:
/// a star needs every column and names none of them, so there is nothing a
/// keep-list could represent.
///
/// Like [`names_referenced`], this is over-broad where it can't attribute a
/// name to `source_id` specifically (inside a join, say) — it returns every
/// name in scope, and the caller keeps only those that exist in the source's
/// schema.
fn source_columns_referenced(root: &Expression, source_id: &str) -> Option<HashSet<String>> {
    let mut out = HashSet::new();
    for node in root.dfs() {
        if let Expression::Select(select) = node {
            if !select_scans_source(select, source_id) {
                continue;
            }
            if select_has_star(select) {
                return None;
            }
            out.extend(names_referenced(node));
        }
    }
    Some(out)
}

/// The columns of `source` that `sql` still needs once its dead columns are
/// eliminated, together with `sql` rewritten that way.
///
/// Reading this off the query's own text rather than the connector's plan is
/// what keeps the two consistent: the planner reports what its *plan* reads,
/// which can be narrower than what the text still names, and the text is what
/// has to bind against the narrowed relation. See the dead-column
/// elimination notes at the top of this module.
///
/// `None` means every column of `source` has to survive — the query doesn't
/// parse, can't be regenerated, or a star sits on a scan of `source`.
fn columns_needed_from(
    sql: &str,
    source: &str,
    dialect: DialectType,
) -> Option<(HashSet<String>, String)> {
    let parsed = polyglot_sql::parse_one(sql, dialect)
        .inspect_err(|e| trace!("pushdown: can't parse frontier query ({e})"))
        .ok()?;
    let eliminated = eliminate_dead_columns(parsed);
    let cols = source_columns_referenced(&eliminated, source)?;
    // The keep-list above describes the *eliminated* query, so it is only
    // usable together with that query's text.
    let rewritten = polyglot_sql::generate(&eliminated, dialect)
        .inspect_err(|e| trace!("pushdown: can't regenerate frontier query ({e})"))
        .ok()?;
    Some((cols, rewritten))
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
fn rewrite_node_refs(
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

/// Apply [`rewrite_node_refs`], falling back to longest-first textual
/// substitution when the SQL can't be parsed (an exotic dialect construct);
/// the fallback is exactly what this pass did for every query before.
fn rewrite_node_refs_or_replace(
    sql: &str,
    mapping: &HashMap<String, String>,
    sorted_ids: &[&String],
    dialect: DialectType,
) -> String {
    if let Some(rewritten) = rewrite_node_refs(sql, mapping, dialect) {
        return rewritten;
    }
    warn!("pushdown: couldn't parse query for AST-level scratch rewriting, falling back to text substitution");
    let mut out = sql.to_string();
    for id in sorted_ids {
        out = out.replace(id.as_str(), &mapping[*id]);
    }
    out
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

    // Longest-first, so that if a query has to fall back to textual
    // substitution a shorter ID that is a substring of a longer one is never
    // matched first.
    let mut sorted_ids: Vec<&String> = scratch_names.keys().collect();
    sorted_ids.sort_by_key(|id| std::cmp::Reverse(id.len()));

    let dialect = dialect_for_db(&dag.db);

    for id in &topo {
        let node = dag.nodes.get(id.clone()).ok_or_else(|| {
            OptimizerError::Exec(format!("materialize_scratch_dag: node '{id}' not found"))
        })?;

        let query =
            rewrite_node_refs_or_replace(&node.query_text, &scratch_names, &sorted_ids, dialect);

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
/// the dead-column-eliminated SQL of any frontier node whose text had to
/// change for the source's new, narrower schema to bind.
pub struct PushdownResult {
    pub source_sql: String,
    pub source_schema: SchemaRef,
    /// Rewritten SQL for each frontier node whose text changed, keyed by node
    /// ID. Applied together with `source_schema` or not at all: the narrowed
    /// schema is only consistent with the rewritten text.
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
/// **Rows.** Each node in `frontier_materializes(source)` is analyzed via
/// [`Connector::pushdown`], which reports what the planner pushed into each
/// scan of `source`. One node's predicate is the disjunction of its scans'
/// own conjunctions (see [`node_predicate`]), and the pushed filter is the
/// disjunction of every node's predicate: `source` has to keep a row that
/// *any* consumer reads. That only holds if every consumer's rows are
/// accounted for, so a node that reads `source` unfiltered — or that we
/// can't analyze at all — means no filter is pushed.
///
/// **Columns.** Dead columns are eliminated from each frontier node's own SQL
/// (see the notes at the top of this module), and what its scans of `source`
/// still reference afterwards is what `source` must keep; the union across
/// frontier nodes, plus any column the pushed filter needs, is the keep-list.
/// Deriving it from the text rather than from the planner's own projection
/// list is what keeps the narrowed schema and the frontier SQL consistent.
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

    // One entry per frontier node: the predicate that node's scans of
    // `source` apply, already combined across those scans.
    let mut per_node_filters: Vec<String> = Vec::new();
    // Union, across frontier nodes, of the columns of `source` their SQL
    // still references once dead columns have been eliminated from it.
    let mut required_cols: HashSet<String> = HashSet::new();
    // Dead-column-eliminated SQL for each frontier node whose text changed.
    let mut dce_sql: HashMap<String, String> = HashMap::new();
    let mut any_node_needs_all_cols = false;
    // Set whenever we cannot account for every row some frontier node reads.
    // Pushing a filter then would delete rows that node still needs, so the
    // filter is dropped entirely — a predicate is only sound to push when
    // *every* consumer's row requirement is known and expressible.
    let mut any_node_needs_all_rows = false;

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

        // ---- which rows this node needs ----

        let rewritten_query = rewrite_node_refs_or_replace(
            &n_node.query_text,
            scratch_names,
            &sorted_ids,
            dialect,
        );

        match conn.pushdown(&rewritten_query).await {
            Err(e) => {
                warn!(
                    "pushdown: connector pushdown failed for frontier '{n_id}': {e}, \
                     assuming all rows needed"
                );
                any_node_needs_all_rows = true;
            }
            Ok(pushdown_map) => match pushdown_map.and_then(|m| m.get(&scratch_name).cloned()) {
                None => {
                    trace!(
                        "  frontier '{n_id}': scratch scan not found in connector pushdown \
                         result, assuming all rows needed"
                    );
                    any_node_needs_all_rows = true;
                }
                Some(scans) => match node_predicate(&scans, n_id, dialect) {
                    Some(predicate) => {
                        trace!("  frontier '{n_id}': predicate = {predicate}");
                        per_node_filters.push(predicate);
                    }
                    None => {
                        trace!("  frontier '{n_id}': needs all rows");
                        any_node_needs_all_rows = true;
                    }
                },
            },
        }

        // ---- which columns this node needs ----

        match columns_needed_from(&n_node.query_text, source, dialect) {
            None => {
                trace!("  frontier '{n_id}': needs every column of '{source}'");
                any_node_needs_all_cols = true;
            }
            Some((cols, rewritten)) => {
                trace!(
                    "  frontier '{n_id}': columns referenced from '{source}' = [{}]",
                    cols.iter().cloned().collect::<Vec<_>>().join(", ")
                );
                required_cols.extend(cols);
                if rewritten != n_node.query_text {
                    dce_sql.insert(n_id.clone(), rewritten);
                }
            }
        }
    }

    // Build the combined filter: the disjunction of every frontier node's own
    // predicate. `source` must keep a row that *any* consumer reads, so the
    // combination is a union of row sets, never an intersection — and it is
    // only sound at all if every consumer accounted for its rows.
    let combined_filter: Option<String> = if any_node_needs_all_rows {
        debug!(
            "pushdown '{source}': at least one frontier node needs every row, \
             not pushing any filter"
        );
        None
    } else {
        match per_node_filters.len() {
            0 => None,
            1 => per_node_filters.into_iter().next(),
            _ => Some(
                per_node_filters
                    .into_iter()
                    .map(|f| format!("({f})"))
                    .collect::<Vec<_>>()
                    .join(" OR "),
            ),
        }
    };

    // Columns a pushed-down filter references must survive projection pruning
    // even when nothing selects them (`WHERE is_active`, say).
    if let Some(filter) = &combined_filter {
        match filter_referenced_columns(filter, dialect) {
            Some(cols) => required_cols.extend(cols),
            None => {
                warn!(
                    "pushdown: couldn't parse the combined filter '{filter}' for '{source}', \
                     assuming all columns needed"
                );
                any_node_needs_all_cols = true;
            }
        }
    }

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

    // The narrowed schema is only consistent with frontier queries that have
    // had their dead columns eliminated — that elimination is exactly what
    // proved those columns dead — so the two are applied together or not at
    // all.
    let frontier_sql: HashMap<String, String> = if projection_cols.is_some() {
        dce_sql
    } else {
        HashMap::new()
    };

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
    /// Which side of an execution to step on. A rewrite is only useful before
    /// the DAG runs, so `Before` is the author's default — but the setting is
    /// still honoured, because a `Once` optimization is stepped explicitly and
    /// never around a run, which makes the value moot rather than wrong.
    step_phase: StepPhase,
    /// Data collected during the last `step()`, used by `explain`.
    explain_data: Option<PushdownExplainData>,
}

/// Everything `explain` needs to describe what the last `step()`
/// did and why, retained from otherwise-local data computed during `step()`.
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
            step_phase: StepPhase::Before,
            explain_data: None,
        }
    }

    /// Rewrite `dag` in place, reporting what it changed.
    ///
    /// Split out from `step` because HMP and OMP call pushdown on each
    /// candidate before measuring it (`hmp_use_pushdown`, `omp_use_pushdown`).
    /// That is a rewrite inside another optimization's step, not an
    /// optimization registered on the DAG, so it needs the work without the
    /// step machinery around it.
    pub async fn rewrite(&mut self, dag: &mut Dag) -> Result<PassOutcome, OptimizerError> {

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

#[async_trait]
impl<C, E> Optimization<C, E> for PushdownPass<C, E>
where
    C: Connector + Send + 'static + Sync,
    E: Executor<C> + Send + Sync,
{
    fn name(&self) -> &'static str {
        "pushdown"
    }

    /// Pushdown decides everything from the DAG in front of it. There is no
    /// measurement to wait for and nothing a later run could teach it, so it
    /// runs once and is finished.
    fn optimization_type(&self) -> OptimizationType {
        OptimizationType::Once
    }

    fn step_phase(&self) -> StepPhase {
        self.step_phase
    }

    fn set_step_phase(&mut self, phase: StepPhase) {
        self.step_phase = phase;
    }

    /// Nothing to set up. Pushdown keeps no state between steps — it reads
    /// the DAG and the warehouse's own schemas each time — so it owns no
    /// tables, and says so rather than creating an empty one.
    async fn register(
        &self,
        _ctx: &RegisterContext<'_>,
    ) -> Result<Option<Registration>, OptimizerError> {
        Ok(None)
    }

    /// Nothing to tear down, for the same reason.
    async fn deregister(
        &self,
        _ctx: &RegisterContext<'_>,
    ) -> Result<Option<Registration>, OptimizerError> {
        Ok(None)
    }

    async fn step(
        &mut self,
        ctx: &mut StepContext<'_, C, E>,
    ) -> Result<StepOutcome, OptimizerError> {
        let record = self.rewrite(ctx.dag).await?;
        Ok(StepOutcome::Rewrote {
            record: Box::new(record),
        })
    }

    fn explain(&self) -> Option<(String, String)> {
        Some(("PushdownPass".to_string(), self.explain_html()))
    }
}

impl<C, E> PushdownPass<C, E>
where
    C: Connector + Send + 'static + Sync,
    E: Executor<C> + Send + Sync,
{
    fn explain_html(&self) -> String {
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


    fn scalar_i64(conn: &DuckDBConnection, sql: &str) -> i64 {
        conn.pool
            .get()
            .unwrap()
            .query_row(sql, [], |r| r.get(0))
            .unwrap()
    }

    fn orders_source() -> Vec<SourceNode> {
        vec![SourceNode {
            name: "orders".to_string(),
            schema: Arc::new(duckdb::arrow::datatypes::Schema::empty()),
        }]
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
        pass.rewrite(&mut dag).await.expect("pass should succeed");

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
        pass.rewrite(&mut dag).await.expect("pass should succeed");

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
        pass.rewrite(&mut dag).await.expect("pass should succeed");

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
        pass.rewrite(&mut dag)
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
        pass.rewrite(&mut dag).await.expect("pass should succeed");

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


    // DAG layout:
    //
    //   orders (source, real table)
    //       │
    //   staging (TempTable)
    //       ├──► t_us   (Table)  SELECT ... FROM staging WHERE region = 'US'
    //       └──► t_all  (Table)  SELECT ... FROM staging          — every row
    //
    // A consumer that reads the TempTable unfiltered contributes no predicate
    // to OR against, so there is no predicate the TempTable can be narrowed
    // by at all. Pushing the *other* consumer's filter anyway silently
    // deletes rows `t_all` needs.
    //
    // Regression test for the dominant dag-bench failure: `lp_stg_employees`
    // was narrowed to `WHERE (is_active) OR (is_active) OR ...` across five
    // consumers that agreed, dropping the rows a sixth still counted.
    #[tokio::test]
    async fn test_unfiltered_consumer_blocks_filter_pushdown() {
        let conn = in_memory_conn().await;
        setup_orders_table(&conn).await;
        let engine = Arc::new(SimpleEngine::new(Arc::clone(&conn)).unwrap());

        let staging = node(
            "staging",
            "SELECT order_id, region, amount, status FROM orders",
            MaterializeMode::TempTable,
            &[],
        );
        let t_us = node(
            "t_us",
            "SELECT order_id, amount FROM staging WHERE region = 'US'",
            MaterializeMode::Table,
            &["staging"],
        );
        let t_all = node(
            "t_all",
            "SELECT order_id, region FROM staging",
            MaterializeMode::Table,
            &["staging"],
        );

        let mut dag = make_dag(vec![staging, t_us, t_all]);
        dag.sources = orders_source();

        let mut pass = PushdownPass::new(Arc::clone(&conn), Arc::clone(&engine));
        pass.rewrite(&mut dag).await.expect("pass should succeed");
        let rewritten = dag
            .nodes
            .get("staging".to_string())
            .unwrap()
            .query_text
            .clone();
        assert!(
            !rewritten.contains("WHERE"),
            "no filter may be pushed while a consumer reads every row; got: {rewritten}"
        );

        engine.run(&dag).await.expect("rewritten dag should run");
        assert_eq!(
            scalar_i64(&conn, "SELECT count(*) FROM t_all"),
            20,
            "the unfiltered consumer must still see every row"
        );
        assert_eq!(scalar_i64(&conn, "SELECT count(*) FROM t_us"), 10);
    }

    // One consumer scanning the TempTable twice with a different predicate on
    // each side. The two scans are alternatives — the TempTable must keep the
    // rows either side reads — so they combine with OR. AND-ing them (which
    // is what merging the two scans into one filter list did) asks for rows
    // that are both 'US' and 'EU', i.e. none.
    #[tokio::test]
    async fn test_self_join_scans_combine_with_or() {
        let conn = in_memory_conn().await;
        setup_orders_table(&conn).await;
        let engine = Arc::new(SimpleEngine::new(Arc::clone(&conn)).unwrap());

        let staging = node(
            "staging",
            "SELECT order_id, region, amount, status FROM orders",
            MaterializeMode::TempTable,
            &[],
        );
        let sink = node(
            "sink",
            "SELECT a.order_id AS a_id, b.order_id AS b_id \
             FROM staging a JOIN staging b ON a.order_id = b.order_id + 1 \
             WHERE a.region = 'US' AND b.region = 'EU'",
            MaterializeMode::Table,
            &["staging"],
        );
        let mut dag = make_dag(vec![staging, sink]);
        dag.sources = orders_source();

        let expected = scalar_i64(
            &conn,
            "SELECT count(*) FROM orders a JOIN orders b ON a.order_id = b.order_id + 1 \
             WHERE a.region = 'US' AND b.region = 'EU'",
        );
        assert!(expected > 0, "baseline should be non-empty");

        let mut pass = PushdownPass::new(Arc::clone(&conn), Arc::clone(&engine));
        pass.rewrite(&mut dag).await.expect("pass should succeed");
        let rewritten = dag
            .nodes
            .get("staging".to_string())
            .unwrap()
            .query_text
            .clone();

        engine.run(&dag).await.expect("rewritten dag should run");
        assert_eq!(
            scalar_i64(&conn, "SELECT count(*) FROM sink"),
            expected,
            "both scans' rows must survive; staging rewritten to: {rewritten}"
        );
    }

    // A node ID that also appears inside a string literal must not be
    // rewritten when the query is handed to the connector for analysis: the
    // planner would report a predicate over a scratch table name, and that
    // nonsense predicate would then be pushed into the TempTable for real.
    #[tokio::test]
    async fn test_node_id_inside_a_string_literal_is_not_rewritten() {
        let conn = in_memory_conn().await;
        conn.execute(
            "CREATE TABLE events AS SELECT range AS id, \
             CASE range % 4 WHEN 0 THEN 'staging' WHEN 1 THEN 'prod' \
                            WHEN 2 THEN 'aaa' ELSE 'zzz' END AS env FROM range(20)"
                .to_string(),
        )
        .await
        .unwrap();
        let engine = Arc::new(SimpleEngine::new(Arc::clone(&conn)).unwrap());

        let staging = node(
            "staging",
            "SELECT id, env FROM events",
            MaterializeMode::TempTable,
            &[],
        );
        let sink = node(
            "sink",
            "SELECT id FROM staging WHERE env = 'staging'",
            MaterializeMode::Table,
            &["staging"],
        );
        let mut dag = make_dag(vec![staging, sink]);
        dag.sources = vec![SourceNode {
            name: "events".to_string(),
            schema: Arc::new(duckdb::arrow::datatypes::Schema::empty()),
        }];

        let mut pass = PushdownPass::new(Arc::clone(&conn), Arc::clone(&engine));
        pass.rewrite(&mut dag).await.expect("pass should succeed");
        let rewritten = dag
            .nodes
            .get("staging".to_string())
            .unwrap()
            .query_text
            .clone();
        assert!(
            !rewritten.contains("dee_tmp_pushdown"),
            "a scratch relation name must never reach the rewritten DAG; got: {rewritten}"
        );

        engine.run(&dag).await.expect("rewritten dag should run");
        assert_eq!(
            scalar_i64(&conn, "SELECT count(*) FROM sink"),
            5,
            "staging rewritten to: {rewritten}"
        );
    }

    // End-to-end version of `test_dce_keeps_columns_a_later_cte_reads`: the
    // rewritten DAG has to bind.
    #[tokio::test]
    async fn test_frontier_reading_a_temp_table_through_chained_ctes_still_binds() {
        let conn = in_memory_conn().await;
        setup_orders_table(&conn).await;
        let engine = Arc::new(SimpleEngine::new(Arc::clone(&conn)).unwrap());

        let staging = node(
            "staging",
            "SELECT order_id, region, amount, status FROM orders",
            MaterializeMode::TempTable,
            &[],
        );
        let sink = node(
            "sink",
            "WITH a AS (SELECT order_id, region, amount, status FROM staging), \
             b AS (SELECT order_id, region, amount, status FROM a) \
             SELECT region, count(*) AS n FROM b GROUP BY region",
            MaterializeMode::Table,
            &["staging"],
        );
        let mut dag = make_dag(vec![staging, sink]);
        dag.sources = orders_source();

        let mut pass = PushdownPass::new(Arc::clone(&conn), Arc::clone(&engine));
        pass.rewrite(&mut dag).await.expect("pass should succeed");
        let staging_sql = dag
            .nodes
            .get("staging".to_string())
            .unwrap()
            .query_text
            .clone();
        let sink_sql = dag.nodes.get("sink".to_string()).unwrap().query_text.clone();

        engine.run(&dag).await.unwrap_or_else(|e| {
            panic!("rewritten dag should run: {e}\nstaging = {staging_sql}\nsink = {sink_sql}")
        });
        assert_eq!(scalar_i64(&conn, "SELECT count(*) FROM sink"), 2);
    }

    // ------------------------------------------------------------------
    // Dead-column elimination unit tests
    // ------------------------------------------------------------------

    /// Run dead-column elimination over `sql` and print it back out.
    fn dce(sql: &str) -> String {
        let parsed = polyglot_sql::parse_one(sql, DialectType::DuckDB).expect("parses");
        polyglot_sql::generate(&eliminate_dead_columns(parsed), DialectType::DuckDB)
            .expect("regenerates")
    }

    /// The columns of `source` that survive dead-column elimination of `sql`.
    fn surviving_source_columns(sql: &str, source: &str) -> Option<HashSet<String>> {
        let parsed = polyglot_sql::parse_one(sql, DialectType::DuckDB).expect("parses");
        source_columns_referenced(&eliminate_dead_columns(parsed), source)
    }

    #[test]
    fn test_dce_removes_unused_nested_column() {
        let pruned = dce(r#"SELECT x."a" FROM (SELECT "a", "b", "c" FROM "staging") AS x"#);
        assert!(pruned.contains('a'));
        assert!(!pruned.contains('b'), "got: {pruned}");
        assert!(!pruned.contains('c'), "got: {pruned}");
    }

    // A nested select's output is read by *other parts of the same query*,
    // not just by whatever encloses it syntactically. Here the CTE `b` reads
    // columns from the CTE `a`, so narrowing `a` to what the final SELECT
    // uses would leave `b` referencing columns that no longer exist.
    //
    // Regression test for four dag-bench DAGs (p07_saas) that stopped
    // binding: `Referenced column "event_id" not found in FROM clause`.
    #[test]
    fn test_dce_keeps_columns_a_later_cte_reads() {
        let sql = r#"WITH a AS (SELECT order_id, region, status FROM staging),
                          b AS (SELECT order_id, region, status FROM a)
                     SELECT region FROM b"#;
        let pruned = dce(sql);
        assert!(
            pruned.contains("order_id") && pruned.contains("status"),
            "columns a later CTE reads must survive; got: {pruned}"
        );
    }

    // ...but a column nothing else in the statement mentions is still dead,
    // even inside a CTE — otherwise CTE-shaped queries would never prune.
    #[test]
    fn test_dce_prunes_a_cte_column_nothing_reads() {
        let sql = r#"WITH a AS (SELECT order_id, region, source_system FROM staging)
                     SELECT order_id, region FROM a"#;
        let pruned = dce(sql);
        assert!(
            !pruned.contains("source_system"),
            "a CTE column nothing references is dead; got: {pruned}"
        );
    }

    // Dropping a live intermediate makes what fed it dead in turn, so the
    // elimination has to run to fixpoint rather than one level deep.
    #[test]
    fn test_dce_propagates_through_nested_derived_tables() {
        let sql = r#"SELECT z."a"
                     FROM (SELECT y."a", y."b" FROM (SELECT "a", "b", "c" FROM "staging") AS y) AS z"#;
        let pruned = dce(sql);
        assert!(!pruned.contains('b'), "got: {pruned}");
        assert!(!pruned.contains('c'), "got: {pruned}");
    }

    // A nested SELECT that computes an *aggregate* over a source column must
    // keep it: `mean_temp` is what the outer query reads, and `avg_temp` is
    // what `staging` must therefore still provide.
    #[test]
    fn test_dce_never_drops_computed_aggregate_columns() {
        let sql = r#"WITH stats AS (
            SELECT device_id, AVG(avg_temp) AS mean_temp, STDDEV(avg_temp) AS std_temp
            FROM "staging"
            GROUP BY device_id
        )
        SELECT h.device_id, s.mean_temp, s.std_temp
        FROM "staging" AS h JOIN stats AS s USING (device_id)"#;

        let pruned = dce(sql);
        assert!(
            pruned.contains("mean_temp") && pruned.contains("std_temp"),
            "computed aggregate columns the outer query reads must survive; got: {pruned}"
        );
        let cols = surviving_source_columns(sql, "staging").expect("no star");
        assert!(
            cols.contains("avg_temp") && cols.contains("device_id"),
            "the aggregate's argument is still needed from the source; got: {cols:?}"
        );
    }

    // `SELECT pp.*` needs every column of `staging` and names none of them,
    // so there is nothing a keep-list could represent.
    #[test]
    fn test_source_columns_bail_out_on_star() {
        let sql = r#"SELECT pp.*, ROW_NUMBER() OVER (ORDER BY x) AS rn FROM "staging" AS pp"#;
        let result = surviving_source_columns(sql, "staging");
        assert!(
            result.is_none(),
            "a star touching a scan of source must force 'need all columns'; got: {result:?}"
        );
    }

    #[test]
    fn test_source_columns_keep_aggregate_arguments_and_group_keys() {
        let sql = r#"SELECT region, AVG(avg_voltage) AS avg_voltage FROM "staging" GROUP BY region"#;
        let cols = surviving_source_columns(sql, "staging").expect("no star present");
        assert!(
            cols.contains("avg_voltage"),
            "aggregate argument must be kept; got: {cols:?}"
        );
        assert!(
            cols.contains("region"),
            "GROUP BY column must be kept; got: {cols:?}"
        );
    }

    #[test]
    fn test_dce_never_touches_the_outermost_select() {
        // Nothing sits above this statement, so its projection list is its
        // own output contract.
        let pruned = dce(r#"SELECT "a", "b" FROM "staging""#);
        assert!(
            pruned.contains('b'),
            "outermost SELECT must be untouched; got: {pruned}"
        );
    }

    // A joined nested scan is prunable too — its output feeds only the outer
    // query — but the column set attributed to `staging` stays over-broad
    // (the join key from the *other* side is kept as well), which costs a
    // little pruning and never correctness.
    #[test]
    fn test_dce_prunes_joined_nested_scan_but_keeps_join_keys() {
        let sql =
            r#"SELECT x."a" FROM (SELECT "a", "b" FROM "staging" JOIN "other" ON "a" = "id") AS x"#;
        let pruned = dce(sql);
        assert!(!pruned.contains('b'), "got: {pruned}");
        let cols = surviving_source_columns(sql, "staging").expect("no star");
        assert!(cols.contains("a") && cols.contains("id"), "got: {cols:?}");
    }

    #[test]
    fn test_dce_never_touches_a_top_level_union_branch() {
        let sql = r#"SELECT "a", "b" FROM "staging"
                     UNION ALL
                     SELECT "a", "b" FROM "other""#;
        let pruned = dce(sql);
        assert!(
            pruned.contains('b'),
            "top-level UNION branch must be untouched; got: {pruned}"
        );
        let cols = surviving_source_columns(sql, "staging").expect("no star");
        assert!(
            cols.contains("b"),
            "a top-level branch's own output columns all survive; got: {cols:?}"
        );
    }

    // A nested UNION ALL is prunable, but only *positionally* — branches line
    // up by position, so a column dropped from one has to be dropped from all
    // of them or the arity stops matching.
    #[test]
    fn test_dce_prunes_every_branch_of_a_nested_union_together() {
        let sql = r#"WITH combined AS (
            SELECT "a", "b" FROM "staging"
            UNION ALL
            SELECT "a", "b" FROM "other"
        )
        SELECT c."a" FROM combined AS c"#;
        let pruned = dce(sql);
        assert!(
            !pruned.contains(r#""b""#),
            "'b' is dead in both branches; got: {pruned}"
        );
        assert_eq!(
            pruned.matches("SELECT \"a\" FROM").count(),
            2,
            "both branches must be pruned to the same arity; got: {pruned}"
        );
        let cols = surviving_source_columns(sql, "staging").expect("no star");
        assert!(!cols.contains("b"), "got: {cols:?}");
    }

    // A deduplicating set operation compares whole rows, so dropping a column
    // from its branches changes which rows come out.
    #[test]
    fn test_dce_leaves_deduplicating_set_operations_alone() {
        for op in ["UNION", "INTERSECT", "EXCEPT"] {
            let sql = format!(
                r#"WITH combined AS (
                    SELECT "a", "b" FROM "staging" {op} SELECT "a", "b" FROM "other"
                )
                SELECT c."a" FROM combined AS c"#
            );
            let pruned = dce(&sql);
            assert!(
                pruned.contains('b'),
                "{op} dedups on every column, so none may be dropped; got: {pruned}"
            );
        }
    }

    // DISTINCT is the same argument one level down.
    #[test]
    fn test_dce_leaves_distinct_selects_alone() {
        let sql = r#"SELECT x."a" FROM (SELECT DISTINCT "a", "b" FROM "staging") AS x"#;
        let pruned = dce(sql);
        assert!(
            pruned.contains('b'),
            "dropping a column changes what DISTINCT deduplicates; got: {pruned}"
        );
    }

    // An output column referenced only by the select's own ORDER BY/GROUP BY
    // (never by anything above it) still has a live reference to satisfy.
    #[test]
    fn test_dce_keeps_columns_the_select_itself_orders_by() {
        let sql = r#"SELECT x."a" FROM (SELECT "a", "b" FROM "staging" ORDER BY "b") AS x"#;
        let pruned = dce(sql);
        assert!(
            pruned.contains('b'),
            "ORDER BY may reference the output alias; got: {pruned}"
        );
    }


    // `GROUP BY 1, 2` counts select-list positions, so dropping an item ahead
    // of them would silently regroup the query.
    #[test]
    fn test_dce_leaves_selects_with_positional_group_by_alone() {
        let sql = r#"SELECT x."a" FROM (
            SELECT "a", "b", COUNT(*) AS n FROM "staging" GROUP BY 1, 2
        ) AS x"#;
        let pruned = dce(sql);
        assert!(
            pruned.contains(r#""b""#),
            "a positional GROUP BY pins every position; got: {pruned}"
        );
    }

    #[test]
    fn test_dce_leaves_selects_with_positional_order_by_alone() {
        let sql = r#"SELECT x."a" FROM (SELECT "a", "b" FROM "staging" ORDER BY 2 DESC) AS x"#;
        let pruned = dce(sql);
        assert!(
            pruned.contains(r#""b""#),
            "a positional ORDER BY pins every position; got: {pruned}"
        );
    }


    // A recursive CTE's own body references it by name, which is the one
    // reference the "everything else in the statement" rule can't see.
    #[test]
    fn test_dce_leaves_recursive_ctes_alone() {
        let sql = r#"WITH RECURSIVE walk AS (
            SELECT "id", "parent_id" FROM "staging" WHERE "parent_id" IS NULL
            UNION ALL
            SELECT s."id", s."parent_id" FROM "staging" AS s JOIN walk AS w ON s."parent_id" = w."id"
        )
        SELECT "id" FROM walk"#;
        let pruned = dce(sql);
        assert!(
            pruned.contains("parent_id"),
            "a recursive CTE's own columns must survive; got: {pruned}"
        );
    }

    // ------------------------------------------------------------------
    // Table-reference matching / rewriting unit tests
    // ------------------------------------------------------------------

    // A node ID that also occurs inside a string literal must not be
    // rewritten: the connector would then be handed `WHERE env =
    // '<scratch table>'` and report a predicate that means something else.
    #[test]
    fn test_rewrite_node_refs_leaves_string_literals_alone() {
        let mapping = HashMap::from([("staging".to_string(), "scratch_0".to_string())]);
        let rewritten = rewrite_node_refs(
            "SELECT id FROM staging WHERE env = 'staging'",
            &mapping,
            DialectType::DuckDB,
        )
        .expect("rewrites");
        assert!(rewritten.contains("FROM scratch_0"), "got: {rewritten}");
        assert!(
            rewritten.contains("'staging'"),
            "the literal must survive verbatim; got: {rewritten}"
        );
    }

    #[test]
    fn test_rewrite_node_refs_leaves_longer_identifiers_alone() {
        let mapping = HashMap::from([("orders".to_string(), "scratch_0".to_string())]);
        let rewritten = rewrite_node_refs(
            "SELECT orders_total FROM orders",
            &mapping,
            DialectType::DuckDB,
        )
        .expect("rewrites");
        assert!(rewritten.contains("orders_total"), "got: {rewritten}");
        assert!(rewritten.contains("FROM scratch_0"), "got: {rewritten}");
    }

    #[test]
    fn test_rewrite_node_refs_handles_qualified_node_ids() {
        let mapping = HashMap::from([(
            r#""warehouse"."main"."stg_orders""#.to_string(),
            "scratch_0".to_string(),
        )]);
        let rewritten = rewrite_node_refs(
            r#"SELECT a FROM "warehouse"."main"."stg_orders" AS o"#,
            &mapping,
            DialectType::DuckDB,
        )
        .expect("rewrites");
        assert!(rewritten.contains("scratch_0"), "got: {rewritten}");
        assert!(
            rewritten.contains(" AS o") || rewritten.contains(" o"),
            "the alias must survive; got: {rewritten}"
        );
    }

    #[test]
    fn test_table_ref_matches_respects_a_differing_schema() {
        let expr = polyglot_sql::parse_one(
            r#"SELECT a FROM "warehouse"."raw"."orders""#,
            DialectType::DuckDB,
        )
        .unwrap();
        let table = expr
            .dfs()
            .find_map(|e| match e {
                Expression::Table(t) => Some(t.clone()),
                _ => None,
            })
            .unwrap();
        assert!(table_ref_matches(&table, r#""warehouse"."raw"."orders""#));
        assert!(table_ref_matches(&table, "orders"), "unqualified ID matches");
        assert!(
            !table_ref_matches(&table, r#""warehouse"."main"."orders""#),
            "a different schema is a different relation"
        );
    }

    // ------------------------------------------------------------------
    // Predicate combination unit tests
    // ------------------------------------------------------------------

    #[test]
    fn test_node_predicate_ands_within_a_scan_and_ors_across_scans() {
        let scans = vec![
            PushdownInfo {
                projections: vec!["a".to_string()],
                filters: vec!["a<10".to_string(), "b>1".to_string()],
            },
            PushdownInfo {
                projections: vec!["a".to_string()],
                filters: vec!["a>90".to_string()],
            },
        ];
        assert_eq!(
            node_predicate(&scans, "n", DialectType::DuckDB),
            Some("(a<10 AND b>1) OR (a>90)".to_string())
        );
    }

    #[test]
    fn test_node_predicate_is_none_when_any_scan_is_unfiltered() {
        let scans = vec![
            PushdownInfo {
                projections: vec!["a".to_string()],
                filters: vec!["a<10".to_string()],
            },
            PushdownInfo::default(),
        ];
        assert_eq!(
            node_predicate(&scans, "n", DialectType::DuckDB),
            None,
            "an unfiltered scan reads the relation whole"
        );
    }

    #[test]
    fn test_node_predicate_is_none_when_a_predicate_cannot_be_parsed() {
        let scans = vec![PushdownInfo {
            projections: vec!["a".to_string()],
            filters: vec!["((((".to_string()],
        }];
        assert_eq!(node_predicate(&scans, "n", DialectType::DuckDB), None);
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
