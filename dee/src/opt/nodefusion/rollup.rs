//! The rollup: which nodes are shared, and how every consumer reads them back.
//!
//! This implements the rollup spec (`SRS.md`). [`select_rollup`] is steps 2
//! and 3: count how many times each View's SQL runs across the stored builds,
//! choose the set `S` of models that become CTEs, and name the *kinds* -- the
//! members of `S` something outside the rollup reads. [`emit`] is steps 4 to
//! 6: the rollup's own query, and the rewrite of every stored node that reads
//! it.
//!
//! Departures from the spec's wording, all forced by dee's DAG rather than
//! chosen:
//!
//! * A `TempTable` is a stored build like a `Table`. It counts toward `exec`,
//!   and a View that reads one stays outside the rollup for the same reason a
//!   View that reads a Table does.
//! * A repeated View that reads a stored node is excluded at step 1 as well as
//!   step 2. The spec's claim that `S` is closed upstream holds for a View's
//!   View ancestors, not for a Table above it, and letting such a View in would
//!   build the rollup from a table that is itself built from the rollup.
//! * `S` is closed upstream over Views explicitly ([`CteReason::Upstream`]). A
//!   member reading a View outside `S` would read the real view, so the rollup
//!   would depend on something other than the warehouse's sources.
//! * There is no `ref()` marker, so references are rewritten on the parsed
//!   query and regenerated rather than substituted into the original bytes.
//!   CTEs dee adds are prefixed (`n_`, `dee_k<kind>_`, `dee_v_`) rather than
//!   named after the model, so no reference has to rely on shadowing.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use polyglot_sql::{
    dialects::DialectType,
    expressions::{Expression, Identifier, With},
    traversal::{
        ExpressionWalk, contains_aggregate, contains_subquery, contains_window_function, transform,
    },
};

use super::{FUSED_BASE, KIND_COLUMN, NotFusable, stable_topological_order};
use crate::{
    connectors::Connector,
    dag::{Dag, MaterializeMode, TransformNode},
    opt::common::{
        bare_table_name, dialect_for_db, fused_node_name, supports_materialized_hint,
        table_ref_matches,
    },
};

// ---------------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------------

/// Why a model is in `S`, for the log the spec asks for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum CteReason {
    /// A View whose SQL runs this many times, at least twice.
    Repeated { exec: usize },
    /// A View on a path from `S` to a stored build.
    OnPath,
    /// A View a member reads, pulled in so the rollup reads only sources.
    Upstream,
    /// A Table's own query, moved in because `parent` is shared inside `S`.
    TableQuery { parent: String },
}

/// Who reads a kind back out of the rollup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum KindConsumer {
    /// The Table whose own query is this kind; it becomes a projection.
    Own(String),
    /// A stored node that reads the kind's model directly and keeps its SQL.
    Direct(String),
    /// A View outside the rollup that reads the kind; it is pasted into the
    /// stored builds downstream of it.
    Pasted(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RollupKind {
    pub kind: usize,
    pub cte: String,
    pub consumers: Vec<KindConsumer>,
}

#[derive(Debug, Clone)]
pub(super) struct RollupSelection {
    /// `exec(V)` for every View in the DAG, members or not.
    pub exec: HashMap<String, usize>,
    pub members: BTreeMap<String, CteReason>,
    /// The members in the order their CTEs are emitted.
    pub order: Vec<String>,
    /// In `kind` order, numbered from 1.
    pub kinds: Vec<RollupKind>,
    /// Direct readers of each member inside the rollup: the members that
    /// reference it, plus one for its output branch when it is a kind.
    pub readers: HashMap<String, usize>,
}

impl RollupSelection {
    /// The spec's rule: `AS MATERIALIZED` on exactly the CTEs with two or more
    /// readers inside the rollup.
    pub fn default_materialized(&self) -> BTreeSet<String> {
        self.readers
            .iter()
            .filter(|(_, n)| **n >= 2)
            .map(|(id, _)| id.clone())
            .collect()
    }
}

/// A node whose rows are stored, and so what a View's SQL is re-executed for.
fn is_stored(node: &TransformNode) -> bool {
    matches!(
        node.materialize,
        MaterializeMode::Table | MaterializeMode::TempTable
    )
}

/// Every node's readers, sorted so that walks over them depend only on the DAG.
fn children_of(dag: &Dag) -> HashMap<String, Vec<String>> {
    let mut children: HashMap<String, Vec<String>> = HashMap::new();
    for node in dag.nodes.nodes() {
        for dep in &node.depends_on {
            if dag.nodes.get(dep.clone()).is_some() {
                children
                    .entry(dep.clone())
                    .or_default()
                    .push(node.id.clone());
            }
        }
    }
    for kids in children.values_mut() {
        kids.sort();
    }
    children
}

/// `exec(V)` for every View:
///
/// ```text
/// exec(V) = Σ over each child C of V:  1 if C is stored, otherwise exec(C)
/// ```
///
/// Paths, not references. [`crate::graph::Graph::paths_to_sinks`] is the same
/// walk except that it carries on past a Table, which would count a View once
/// more for every Table built from a Table it feeds -- and that Table reads the
/// stored rows, not the View's SQL.
pub(super) fn exec_counts(dag: &Dag) -> HashMap<String, usize> {
    fn exec(
        id: &str,
        dag: &Dag,
        children: &HashMap<String, Vec<String>>,
        memo: &mut HashMap<String, usize>,
    ) -> usize {
        if let Some(&n) = memo.get(id) {
            return n;
        }
        let n = children
            .get(id)
            .map(|kids| {
                kids.iter()
                    .map(|kid| match dag.nodes.get(kid.clone()) {
                        Some(node) if is_stored(node) => 1,
                        Some(_) => exec(kid, dag, children, memo),
                        None => 0,
                    })
                    .sum()
            })
            .unwrap_or(0);
        memo.insert(id.to_string(), n);
        n
    }

    let children = children_of(dag);
    let mut memo = HashMap::new();
    for node in dag.nodes.nodes() {
        if node.materialize == MaterializeMode::View {
            exec(&node.id, dag, &children, &mut memo);
        }
    }
    memo
}

/// Functions whose value depends on when or how many times they run. Matched
/// on the bare, lowercased name, for the ones the parser leaves as plain
/// function calls rather than giving a variant of their own.
const VOLATILE_FUNCTIONS: &[&str] = &[
    "now",
    "current_timestamp",
    "current_date",
    "current_time",
    "localtimestamp",
    "localtime",
    "transaction_timestamp",
    "statement_timestamp",
    "clock_timestamp",
    "timeofday",
    "get_current_time",
    "today",
    "random",
    "rand",
    "setseed",
    "uuid",
    "gen_random_uuid",
    "uuid_generate_v4",
    "nextval",
    "currval",
    "txid_current",
];

fn is_volatile(expr: &Expression) -> bool {
    match expr {
        Expression::CurrentDate(_)
        | Expression::CurrentTime(_)
        | Expression::CurrentTimestamp(_)
        | Expression::CurrentTimestampLTZ(_)
        | Expression::CurrentDatetime(_)
        | Expression::Localtime(_)
        | Expression::Localtimestamp(_)
        | Expression::Random(_)
        | Expression::Rand(_)
        | Expression::Randn(_)
        | Expression::Randstr(_)
        | Expression::Uuid(_)
        | Expression::NextValueFor(_) => true,
        Expression::Function(f) => {
            let bare = f.name.rsplit('.').next().unwrap_or(&f.name);
            VOLATILE_FUNCTIONS.contains(&bare.trim_matches('"').to_ascii_lowercase().as_str())
        }
        _ => false,
    }
}

/// Whether a Table's query has to execute in the Table itself (spec step 3.3).
///
/// A top-level `ORDER BY`, `LIMIT`, `OFFSET` or `FETCH` decides the stored
/// order or which rows are kept, and a volatile function anywhere gives a
/// value that belongs to the Table's own build. An `ORDER BY` inside a window
/// or a subquery is neither, and does not count.
///
/// A query that does not parse counts as sensitive: nothing about it can be
/// checked, so it stays where it was written.
pub(super) fn must_run_in_itself(sql: &str, dialect: DialectType) -> bool {
    let Ok(expr) = polyglot_sql::parse_one(sql, dialect) else {
        return true;
    };
    let decides_stored_rows = match &expr {
        Expression::Select(s) => {
            s.order_by.is_some() || s.limit.is_some() || s.offset.is_some() || s.fetch.is_some()
        }
        Expression::Union(u) => u.order_by.is_some() || u.limit.is_some(),
        _ => false,
    };
    decides_stored_rows || expr.dfs().any(is_volatile)
}

/// Choose the CTE set `S` and its kinds, or say why there is nothing to share.
pub(super) fn select_rollup(dag: &Dag) -> Result<RollupSelection, NotFusable> {
    let dialect = dialect_for_db(&dag.db);
    let exec = exec_counts(dag);
    let children = children_of(dag);
    let is_view = |id: &str| {
        dag.nodes
            .get(id.to_string())
            .is_some_and(|n| n.materialize == MaterializeMode::View)
    };
    // The stored node would be built from the rollup, so a View above it
    // inside the rollup is a cycle.
    let reads_stored = |id: &str| !dag.nodes.upstream_frontier(id, is_stored).is_empty();

    let mut members: BTreeMap<String, CteReason> = BTreeMap::new();

    // 1. Repeated Views.
    for (id, &n) in &exec {
        if n >= 2 && !reads_stored(id) {
            members.insert(id.clone(), CteReason::Repeated { exec: n });
        }
    }
    // 4. Nothing repeated.
    if members.is_empty() {
        return Err(NotFusable(
            "no shareable View executes more than once across the stored builds, so there \
             is nothing for a rollup to share"
                .into(),
        ));
    }

    // 2. Views between S and a stored build. A View whose SQL runs nowhere
    // (exec 0) is on no such path.
    let mut stack: Vec<String> = members.keys().cloned().collect();
    while let Some(id) = stack.pop() {
        for kid in children.get(&id).into_iter().flatten() {
            if members.contains_key(kid)
                || !is_view(kid)
                || exec.get(kid).copied().unwrap_or(0) == 0
                || reads_stored(kid)
            {
                continue;
            }
            members.insert(kid.clone(), CteReason::OnPath);
            stack.push(kid.clone());
        }
    }

    // Closed upstream. A View above a member reads no stored node either,
    // since the member would then read one too.
    let mut stack: Vec<String> = members.keys().cloned().collect();
    while let Some(id) = stack.pop() {
        let Some(node) = dag.nodes.get(id) else {
            continue;
        };
        let mut deps: Vec<&String> = node.depends_on.iter().collect();
        deps.sort();
        for dep in deps {
            if !members.contains_key(dep) && is_view(dep) {
                members.insert(dep.clone(), CteReason::Upstream);
                stack.push(dep.clone());
            }
        }
    }

    // 3. Table queries. "Other readers inside S" counts the Views chosen
    // above and not the Table queries added here, so the answer does not
    // depend on which Table is looked at first.
    let view_members: HashSet<String> = members.keys().cloned().collect();
    let mut tables: Vec<&TransformNode> = dag
        .nodes
        .nodes()
        .filter(|n| n.materialize == MaterializeMode::Table)
        .collect();
    tables.sort_by(|a, b| a.id.cmp(&b.id));
    for table in tables {
        let mut deps: Vec<&String> = table
            .depends_on
            .iter()
            .filter(|d| dag.nodes.get((*d).clone()).is_some())
            .collect();
        // Every model it reads has to be a CTE, or its query cannot run
        // inside the rollup.
        if deps.is_empty() || !deps.iter().all(|d| view_members.contains(*d)) {
            continue;
        }
        deps.sort();
        let shared_parent = deps.into_iter().find(|parent| {
            children
                .get(*parent)
                .is_some_and(|kids| kids.iter().any(|k| view_members.contains(k)))
        });
        let Some(parent) = shared_parent else {
            continue;
        };
        if must_run_in_itself(&table.query_text, dialect) {
            continue;
        }
        members.insert(
            table.id.clone(),
            CteReason::TableQuery {
                parent: parent.clone(),
            },
        );
    }

    let member_set: HashSet<String> = members.keys().cloned().collect();

    // Kinds: the members something outside the rollup reads.
    let mut kinds: Vec<RollupKind> = Vec::new();
    for (id, reason) in &members {
        let is_table_query = matches!(reason, CteReason::TableQuery { .. });
        let mut consumers = Vec::new();
        if is_table_query {
            consumers.push(KindConsumer::Own(id.clone()));
        }
        for kid in children.get(id).into_iter().flatten() {
            if member_set.contains(kid) {
                continue;
            }
            let Some(node) = dag.nodes.get(kid.clone()) else {
                continue;
            };
            if is_stored(node) {
                // A stored node reading a Table query reads that Table's
                // stored rows, not its kind.
                if !is_table_query {
                    consumers.push(KindConsumer::Direct(kid.clone()));
                }
            } else if exec.get(kid).copied().unwrap_or(0) >= 1 {
                consumers.push(KindConsumer::Pasted(kid.clone()));
            }
        }
        if !consumers.is_empty() {
            kinds.push(RollupKind {
                kind: kinds.len() + 1,
                cte: id.clone(),
                consumers,
            });
        }
    }

    let kind_ctes: HashSet<&String> = kinds.iter().map(|k| &k.cte).collect();
    let readers: HashMap<String, usize> = members
        .keys()
        .map(|id| {
            let inside = children
                .get(id)
                .map(|kids| kids.iter().filter(|k| member_set.contains(*k)).count())
                .unwrap_or(0);
            (id.clone(), inside + usize::from(kind_ctes.contains(id)))
        })
        .collect();

    Ok(RollupSelection {
        exec,
        order: stable_topological_order(dag, &member_set),
        members,
        kinds,
        readers,
    })
}

// ---------------------------------------------------------------------------
// Planning names
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub(super) struct RollupPlan {
    pub fused_id: String,
    pub selection: RollupSelection,
    /// Each member's CTE name inside the rollup.
    pub cte_names: BTreeMap<String, String>,
}

/// A bare node name with any quoting removed, for building generated names.
fn plain(id: &str) -> String {
    bare_table_name(id).trim_matches('"').to_string()
}

fn quote(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

/// Select `S` and name the rollup node and its CTEs.
pub(super) fn plan_rollup(dag: &Dag) -> Result<RollupPlan, NotFusable> {
    let selection = select_rollup(dag)?;

    // Beside the first node that will read it, in that node's catalog and
    // schema.
    let sibling = selection
        .kinds
        .iter()
        .flat_map(|k| &k.consumers)
        .map(|c| match c {
            KindConsumer::Own(id) | KindConsumer::Direct(id) | KindConsumer::Pasted(id) => id,
        })
        .min()
        .cloned()
        .ok_or_else(|| NotFusable("nothing outside the rollup would read it".into()))?;
    let mut fused_id = fused_node_name(&sibling, FUSED_BASE);
    let mut n = 2;
    while dag.nodes.get(fused_id.clone()).is_some() {
        fused_id = fused_node_name(&sibling, &format!("{FUSED_BASE}_{n}"));
        n += 1;
        if n > 64 {
            return Err(NotFusable(
                "could not find a free node ID for the rollup".into(),
            ));
        }
    }

    // Two schemas may spell the same bare name, hence the dedupe.
    let mut taken: HashSet<String> = HashSet::new();
    let mut cte_names = BTreeMap::new();
    for id in &selection.order {
        let base = format!("n_{}", plain(id));
        let mut name = base.clone();
        let mut n = 2;
        while !taken.insert(name.to_ascii_lowercase()) {
            name = format!("{base}_{n}");
            n += 1;
        }
        cte_names.insert(id.clone(), name);
    }

    Ok(RollupPlan {
        fused_id,
        selection,
        cte_names,
    })
}

// ---------------------------------------------------------------------------
// Reference rewriting
// ---------------------------------------------------------------------------

/// The names of the CTEs `expr` defines, at any depth, lowercased.
fn local_cte_names(expr: &Expression) -> HashSet<String> {
    let mut names = HashSet::new();
    for node in expr.dfs() {
        let with = match node {
            Expression::Select(s) => s.with.as_ref(),
            Expression::Union(u) => u.with.as_ref(),
            _ => None,
        };
        if let Some(w) = with {
            names.extend(w.ctes.iter().map(|c| c.alias.name.to_ascii_lowercase()));
        }
    }
    names
}

struct Rewritten {
    expr: Expression,
    /// The members of the `deps` passed in that the query still references.
    reads: HashSet<String>,
}

/// Point references to the nodes in `mapping` at their new names, on the
/// parsed query.
///
/// An unqualified reference whose name is a CTE the statement defines itself
/// is left alone. Inside that statement the name may mean its own CTE, and
/// rewriting it would skip the CTE's logic -- `WITH x AS (SELECT * FROM x
/// WHERE ...) SELECT * FROM x` reads node `x` once and its own `x` once, and
/// only scoping can tell the two apart. Leaving both alone is exact: the query
/// goes on reading the real relation, which is still created.
///
/// A rewritten reference keeps the name it was written under as its alias, so
/// a column qualified by that name still resolves.
///
/// `reads` is what the result still references out of `deps`, the node's own
/// dependency set, so a rewrite can only ever remove edges and never add one.
fn rewrite_refs(
    sql: &str,
    mapping: &HashMap<String, String>,
    deps: &HashSet<String>,
    dialect: DialectType,
) -> Option<Rewritten> {
    let parsed = polyglot_sql::parse_one(sql, dialect).ok()?;
    let local = local_cte_names(&parsed);
    let expr = transform(parsed, &|node| {
        let Expression::Table(table) = &node else {
            return Ok(Some(node));
        };
        if table.schema.is_none()
            && table.catalog.is_none()
            && local.contains(&table.name.name.to_ascii_lowercase())
        {
            return Ok(Some(node));
        }
        let Some(new_name) = mapping
            .iter()
            .filter(|(id, _)| table_ref_matches(table, id))
            .map(|(_, name)| name)
            .min()
        else {
            return Ok(Some(node));
        };
        let mut table = table.clone();
        if table.alias.is_none() {
            table.alias = Some(table.name.clone());
            table.alias_explicit_as = true;
        }
        table.name = Identifier::new(new_name.clone());
        table.schema = None;
        table.catalog = None;
        Ok(Some(Expression::Table(table)))
    })
    .ok()?;
    let reads = expr
        .dfs()
        .filter_map(|e| match e {
            Expression::Table(t) => Some(t),
            _ => None,
        })
        .flat_map(|t| deps.iter().filter(move |d| table_ref_matches(t, d)))
        .cloned()
        .collect();
    Some(Rewritten { expr, reads })
}

/// Put `ctes` in front of whatever `WITH` the statement already has.
fn prepend_ctes(expr: &mut Expression, mut ctes: With) -> bool {
    let with = match expr {
        Expression::Select(s) => &mut s.with,
        Expression::Union(u) => &mut u.with,
        _ => return false,
    };
    match with {
        Some(existing) => {
            ctes.ctes.append(&mut existing.ctes);
            existing.ctes = ctes.ctes;
        }
        None => *with = Some(ctes),
    }
    true
}

/// A comma-separated CTE list, parsed into a `With` so it can be prepended to
/// a statement's own.
fn parse_with(ctes: &[String], dialect: DialectType) -> Option<With> {
    let sql = format!("WITH {}\nSELECT 1", ctes.join(",\n"));
    match polyglot_sql::parse_one(&sql, dialect).ok()? {
        Expression::Select(mut s) => s.with.take(),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// The rollup's query
// ---------------------------------------------------------------------------

struct Chain {
    /// `name AS [MATERIALIZED] (body)`, in emission order.
    ctes: Vec<String>,
    reads: HashSet<String>,
    verbatim: usize,
}

fn body_text(sql: &str) -> &str {
    sql.trim_end().trim_end_matches(';').trim_end()
}

fn rollup_chain(
    dag: &Dag,
    plan: &RollupPlan,
    dialect: DialectType,
    materialized: &BTreeSet<String>,
) -> Result<Chain, NotFusable> {
    let hint = supports_materialized_hint(dialect);
    let mapping: HashMap<String, String> = plan
        .cte_names
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let mut chain = Chain {
        ctes: Vec::with_capacity(plan.selection.order.len()),
        reads: HashSet::new(),
        verbatim: 0,
    };
    for id in &plan.selection.order {
        let node = dag
            .nodes
            .get(id.clone())
            .ok_or_else(|| NotFusable(format!("'{id}' vanished from the graph")))?;
        // A body that names no other member is used exactly as written, which
        // is also what lets a leaf staging view use syntax the parser does not
        // cover.
        let body = if node.depends_on.iter().any(|d| mapping.contains_key(d)) {
            let rewritten = rewrite_refs(&node.query_text, &mapping, &node.depends_on, dialect)
                .ok_or_else(|| {
                    NotFusable(format!(
                        "'{id}' reads another shared model, but its query cannot be rewritten \
                         at the AST level"
                    ))
                })?;
            chain.reads.extend(rewritten.reads);
            polyglot_sql::generate(&rewritten.expr, dialect)
                .map_err(|e| NotFusable(format!("regenerating '{id}': {e}")))?
        } else {
            chain.verbatim += 1;
            body_text(&node.query_text).to_string()
        };
        let m = if hint && materialized.contains(id) {
            " MATERIALIZED"
        } else {
            ""
        };
        chain
            .ctes
            .push(format!("{} AS{m} (\n{body}\n)", plan.cte_names[id]));
    }
    Ok(chain)
}

/// Each kind's columns as `(name, type)`, the type spelled as the engine's own
/// `CAST` accepts it.
#[derive(Debug, Clone, Default)]
pub(super) struct KindColumns(pub BTreeMap<usize, Vec<(String, String)>>);

/// Ask the engine what each kind produces.
///
/// From the engine rather than the nodes' Arrow schemas, because that round
/// trip is lossy -- DuckDB's HUGEINT arrives as Decimal128(38, 0) -- and a NULL
/// cast to the lossy type would change the column it fills. Each kind is asked
/// through the rollup's own CTE chain, so only the warehouse's sources have to
/// exist.
pub(super) async fn resolve_kind_columns<C>(
    conn: &C,
    dag: &Dag,
    plan: &RollupPlan,
) -> Result<KindColumns, NotFusable>
where
    C: Connector + Send + Sync,
{
    let dialect = dialect_for_db(&dag.db);
    let chain = rollup_chain(dag, plan, dialect, &BTreeSet::new())?;
    let with = format!("WITH {}", chain.ctes.join(",\n"));
    let mut out = KindColumns::default();
    for kind in &plan.selection.kinds {
        let query = format!("{with}\nSELECT * FROM {}", plan.cte_names[&kind.cte]);
        match conn.column_types(&query).await {
            Ok(Some(columns)) if !columns.is_empty() => {
                out.0.insert(kind.kind, columns);
            }
            Ok(Some(_)) => {
                return Err(NotFusable(format!(
                    "kind {} ('{}') has no columns",
                    kind.kind, kind.cte
                )));
            }
            Ok(None) => {
                return Err(NotFusable(
                    "this connector cannot report column types, and the rollup's branches \
                     need them to fill each other's columns with typed NULLs"
                        .into(),
                ));
            }
            Err(e) => {
                return Err(NotFusable(format!(
                    "could not read the column types of kind {} ('{}'): {e}",
                    kind.kind, kind.cte
                )));
            }
        }
    }
    Ok(out)
}

/// One column of the rollup after `kind`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct FusedColumn {
    pub name: String,
    pub ty: String,
}

struct Layout {
    columns: Vec<FusedColumn>,
    /// Per kind: each of its columns, and the rollup column it lands in.
    per_kind: BTreeMap<usize, Vec<(String, usize)>>,
}

/// The rollup's columns: a name shared with an identical type appears once;
/// on a type clash, or a clash with `kind`, the later kind's column is renamed
/// `<model>__<column>`.
fn layout(plan: &RollupPlan, columns: &KindColumns) -> Result<Layout, NotFusable> {
    let mut out: Vec<FusedColumn> = Vec::new();
    let mut by_name: HashMap<String, usize> = HashMap::new();
    let mut per_kind = BTreeMap::new();
    for kind in &plan.selection.kinds {
        let own = columns.0.get(&kind.kind).ok_or_else(|| {
            NotFusable(format!("no column types were resolved for kind {}", kind.kind))
        })?;
        let mut used: HashSet<usize> = HashSet::new();
        let mut mine = Vec::with_capacity(own.len());
        for (name, ty) in own {
            let lower = name.to_ascii_lowercase();
            let shared = by_name
                .get(&lower)
                .copied()
                .filter(|&i| out[i].ty == *ty && !used.contains(&i));
            let idx = match shared {
                Some(i) => i,
                None => {
                    let base = if lower == KIND_COLUMN || by_name.contains_key(&lower) {
                        format!("{}__{name}", plain(&kind.cte))
                    } else {
                        name.clone()
                    };
                    let mut candidate = base.clone();
                    let mut n = 2;
                    while by_name.contains_key(&candidate.to_ascii_lowercase())
                        || candidate.eq_ignore_ascii_case(KIND_COLUMN)
                    {
                        candidate = format!("{base}_{n}");
                        n += 1;
                    }
                    by_name.insert(candidate.to_ascii_lowercase(), out.len());
                    out.push(FusedColumn {
                        name: candidate,
                        ty: ty.clone(),
                    });
                    out.len() - 1
                }
            };
            used.insert(idx);
            mine.push((name.clone(), idx));
        }
        per_kind.insert(kind.kind, mine);
    }
    Ok(Layout {
        columns: out,
        per_kind,
    })
}

/// The filter a kind's only consumer applies to it, when it can be applied in
/// the kind's branch instead: `SELECT … FROM P WHERE <pred>`, reading nothing
/// else, with a predicate over P's own columns that neither aggregates, looks
/// at a window, runs a subquery nor calls a volatile function.
///
/// Exact because nothing else reads the kind, and the consumer keeps its
/// `WHERE` and applies it again. Returns the name the predicate qualifies the
/// relation by, and the predicate.
fn pushable_filter(
    dag: &Dag,
    kind: &RollupKind,
    columns: &[(String, String)],
    dialect: DialectType,
) -> Option<(String, String)> {
    let [KindConsumer::Direct(consumer)] = kind.consumers.as_slice() else {
        return None;
    };
    let node = dag.nodes.get(consumer.clone())?;
    let Expression::Select(select) = polyglot_sql::parse_one(&node.query_text, dialect).ok()?
    else {
        return None;
    };
    if select.with.is_some()
        || !select.joins.is_empty()
        || !select.lateral_views.is_empty()
        || select.prewhere.is_some()
        || select.group_by.is_some()
        || select.having.is_some()
        || select.qualify.is_some()
        || select.distinct
        || select.distinct_on.is_some()
        || select.windows.is_some()
        || select.sample.is_some()
    {
        return None;
    }
    let [Expression::Table(table)] = select.from.as_ref()?.expressions.as_slice() else {
        return None;
    };
    if !table_ref_matches(table, &kind.cte) {
        return None;
    }
    let pred = &select.where_clause.as_ref()?.this;
    if contains_subquery(pred)
        || contains_aggregate(pred)
        || contains_window_function(pred)
        || pred.dfs().any(is_volatile)
    {
        return None;
    }
    let alias = table.alias.as_ref().unwrap_or(&table.name).name.clone();
    let known: HashSet<String> = columns.iter().map(|(n, _)| n.to_ascii_lowercase()).collect();
    for e in pred.dfs() {
        if let Expression::Column(c) = e {
            if c.table.as_ref().is_some_and(|q| !q.name.eq_ignore_ascii_case(&alias)) {
                return None;
            }
            if !known.contains(&c.name.name.to_ascii_lowercase()) {
                return None;
            }
        }
    }
    let pred = polyglot_sql::generate(pred, dialect).ok()?;
    Some((alias, pred))
}

// ---------------------------------------------------------------------------
// Emission
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct NodeRewrite {
    pub id: String,
    pub query_text: String,
    pub depends_on: HashSet<String>,
}

#[derive(Debug, Clone)]
pub(super) struct Emitted {
    pub fused_id: String,
    pub fused_sql: String,
    /// Normally empty: a member reads a real relation only where scoping kept
    /// a reference from being rewritten.
    pub fused_depends_on: HashSet<String>,
    /// Sorted by node ID.
    pub rewrites: Vec<NodeRewrite>,
    pub columns: Vec<FusedColumn>,
    /// `(kind, predicate)` for every filter applied in a kind's branch.
    pub pushed: Vec<(usize, String)>,
    /// `(node, why)` for every consumer left reading what it read before.
    pub untouched: Vec<(String, String)>,
    pub verbatim_bodies: usize,
    pub materialized: BTreeSet<String>,
}

/// What a kind's consumer reads in place of the model: the kind's rows, under
/// the model's own column names and in its column order.
fn kind_projection(fused_id: &str, layout: &Layout, kind: usize) -> String {
    let columns = layout.per_kind[&kind]
        .iter()
        .map(|(name, i)| format!("{} AS {}", quote(&layout.columns[*i].name), quote(name)))
        .collect::<Vec<_>>()
        .join(", ");
    format!("SELECT {columns} FROM {fused_id} WHERE {KIND_COLUMN} = {kind}")
}

/// Build the rollup's query and every consumer's rewrite, with exactly
/// `materialized` hinted.
pub(super) fn emit(
    dag: &Dag,
    plan: &RollupPlan,
    columns: &KindColumns,
    materialized: &BTreeSet<String>,
) -> Result<Emitted, NotFusable> {
    let dialect = dialect_for_db(&dag.db);
    let sel = &plan.selection;
    let chain = rollup_chain(dag, plan, dialect, materialized)?;
    let layout = layout(plan, columns)?;

    // Step 5: one branch per kind.
    let mut branches = Vec::with_capacity(sel.kinds.len());
    let mut pushed = Vec::new();
    for kind in &sel.kinds {
        let own: HashMap<usize, &str> = layout.per_kind[&kind.kind]
            .iter()
            .map(|(name, i)| (*i, name.as_str()))
            .collect();
        let mut projection = vec![format!("{} AS {KIND_COLUMN}", kind.kind)];
        for (i, column) in layout.columns.iter().enumerate() {
            projection.push(match own.get(&i) {
                Some(name) => format!("{} AS {}", quote(name), quote(&column.name)),
                None => format!("CAST(NULL AS {}) AS {}", column.ty, quote(&column.name)),
            });
        }
        let cte = &plan.cte_names[&kind.cte];
        let from = match pushable_filter(dag, kind, &columns.0[&kind.kind], dialect) {
            Some((alias, pred)) => {
                let from = format!("{cte} AS {} WHERE {pred}", quote(&alias));
                pushed.push((kind.kind, pred));
                from
            }
            None => cte.clone(),
        };
        branches.push(format!("SELECT {} FROM {from}", projection.join(", ")));
    }
    let fused_sql = format!(
        "WITH {}\n{}\nORDER BY {KIND_COLUMN}",
        chain.ctes.join(",\n"),
        branches.join("\nUNION ALL\n")
    );

    // Step 6.
    let kind_by_id: HashMap<&String, &RollupKind> = sel.kinds.iter().map(|k| (&k.cte, k)).collect();
    let is_member = |id: &String| sel.members.contains_key(id);
    let table_queries: HashSet<&String> = sel
        .members
        .iter()
        .filter(|(_, r)| matches!(r, CteReason::TableQuery { .. }))
        .map(|(id, _)| id)
        .collect();
    let kind_name = |k: &RollupKind| format!("dee_k{}_{}", k.kind, plain(&k.cte));
    let view_name = |id: &str| format!("dee_v_{}", plain(id));

    // Views outside the rollup that read a kind through other Views outside
    // it. A stored node downstream gets them pasted in.
    let all_ids: HashSet<String> = dag.nodes.nodes().map(|n| n.id.clone()).collect();
    let mut pasteable: HashSet<String> = HashSet::new();
    for id in stable_topological_order(dag, &all_ids) {
        let Some(node) = dag.nodes.get(id.clone()) else {
            continue;
        };
        if node.materialize == MaterializeMode::View
            && !is_member(&id)
            && node
                .depends_on
                .iter()
                .any(|d| kind_by_id.contains_key(d) || pasteable.contains(d))
        {
            pasteable.insert(id);
        }
    }

    let mut rewrites = Vec::new();
    let mut untouched = Vec::new();
    let mut stored: Vec<&TransformNode> = dag.nodes.nodes().filter(|n| is_stored(n)).collect();
    stored.sort_by(|a, b| a.id.cmp(&b.id));
    'consumers: for node in stored {
        if table_queries.contains(&node.id) {
            let kind = kind_by_id[&node.id];
            rewrites.push(NodeRewrite {
                id: node.id.clone(),
                query_text: kind_projection(&plan.fused_id, &layout, kind.kind),
                depends_on: HashSet::from([plan.fused_id.clone()]),
            });
            continue;
        }

        // A kind read directly -- except a Table query's, whose stored rows
        // are read instead -- and the pasteable Views read directly.
        let direct: Vec<&RollupKind> = node
            .depends_on
            .iter()
            .filter(|d| !table_queries.contains(d))
            .filter_map(|d| kind_by_id.get(d).copied())
            .collect();
        let roots: Vec<&String> = node
            .depends_on
            .iter()
            .filter(|d| pasteable.contains(*d))
            .collect();
        if direct.is_empty() && roots.is_empty() {
            continue;
        }

        let mut views: HashSet<String> = HashSet::new();
        let mut stack: Vec<String> = roots.into_iter().cloned().collect();
        while let Some(id) = stack.pop() {
            if !views.insert(id.clone()) {
                continue;
            }
            if let Some(v) = dag.nodes.get(id) {
                stack.extend(v.depends_on.iter().filter(|d| pasteable.contains(*d)).cloned());
            }
        }
        let view_order = stable_topological_order(dag, &views);

        let mut kinds: BTreeMap<usize, &RollupKind> =
            direct.iter().map(|k| (k.kind, *k)).collect();
        for v in &view_order {
            for d in &dag.nodes.get(v.clone()).expect("from this graph").depends_on {
                if let Some(k) = kind_by_id.get(d) {
                    kinds.insert(k.kind, k);
                }
            }
        }

        // The node's own references: only the kinds it reads directly. The
        // pasted Views': every kind, a Table query's included (spec step 6).
        let mut own_map: HashMap<String, String> =
            direct.iter().map(|k| (k.cte.clone(), kind_name(k))).collect();
        let mut view_map: HashMap<String, String> =
            kinds.values().map(|k| (k.cte.clone(), kind_name(k))).collect();
        for v in &view_order {
            own_map.insert(v.clone(), view_name(v));
            view_map.insert(v.clone(), view_name(v));
        }

        let Some(mut rewritten) = rewrite_refs(&node.query_text, &own_map, &node.depends_on, dialect)
        else {
            untouched.push((node.id.clone(), "its query cannot be rewritten at the AST level".into()));
            continue;
        };
        let added: HashSet<String> = own_map.values().map(|n| n.to_ascii_lowercase()).collect();
        if local_cte_names(&rewritten.expr).iter().any(|n| added.contains(n)) {
            untouched.push((node.id.clone(), "it already defines a CTE with a name dee would add".into()));
            continue;
        }

        let mut ctes: Vec<String> = kinds
            .values()
            .map(|k| format!("{} AS ({})", kind_name(k), kind_projection(&plan.fused_id, &layout, k.kind)))
            .collect();
        let mut reads = std::mem::take(&mut rewritten.reads);
        for v in &view_order {
            let view = dag.nodes.get(v.clone()).expect("from this graph");
            let Some(pasted) = rewrite_refs(&view.query_text, &view_map, &view.depends_on, dialect)
            else {
                untouched.push((
                    node.id.clone(),
                    format!("the View '{v}' it would paste cannot be rewritten at the AST level"),
                ));
                continue 'consumers;
            };
            let Ok(body) = polyglot_sql::generate(&pasted.expr, dialect) else {
                untouched.push((node.id.clone(), format!("the View '{v}' could not be regenerated")));
                continue 'consumers;
            };
            reads.extend(pasted.reads);
            ctes.push(format!("{} AS (\n{body}\n)", view_name(v)));
        }

        let Some(with) = parse_with(&ctes, dialect) else {
            untouched.push((node.id.clone(), "the CTEs dee would add do not parse".into()));
            continue;
        };
        if !prepend_ctes(&mut rewritten.expr, with) {
            untouched.push((node.id.clone(), "its query is not a SELECT or a set operation".into()));
            continue;
        }
        let Ok(query_text) = polyglot_sql::generate(&rewritten.expr, dialect) else {
            untouched.push((node.id.clone(), "its rewritten query could not be regenerated".into()));
            continue;
        };
        reads.insert(plan.fused_id.clone());
        rewrites.push(NodeRewrite {
            id: node.id.clone(),
            query_text,
            depends_on: reads,
        });
    }

    Ok(Emitted {
        fused_id: plan.fused_id.clone(),
        fused_sql,
        fused_depends_on: chain.reads,
        rewrites,
        columns: layout.columns,
        pushed,
        untouched,
        verbatim_bodies: chain.verbatim,
        materialized: materialized.clone(),
    })
}

/// Add the rollup node and apply every rewrite.
pub(super) fn install(dag: &mut Dag, emitted: &Emitted) -> Result<(), String> {
    dag.nodes
        .add_node(TransformNode {
            id: emitted.fused_id.clone(),
            query_text: emitted.fused_sql.clone(),
            materialize: MaterializeMode::TempTable,
            depends_on: emitted.fused_depends_on.clone(),
            schema: None,
        })
        .map_err(|e| format!("adding the rollup node: {e}"))?;
    for rewrite in &emitted.rewrites {
        let node = dag
            .nodes
            .get_mut(rewrite.id.clone())
            .ok_or_else(|| format!("'{}' vanished from the graph", rewrite.id))?;
        node.query_text = rewrite.query_text.clone();
        node.depends_on = rewrite.depends_on.clone();
    }
    dag.nodes
        .check()
        .map_err(|e| format!("the rewrite left an inconsistent graph: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectors::duckdb::{DuckDBConfig, DuckDBConnection};
    use crate::executor::{Executor, SimpleEngine};
    use crate::graph::Graph;
    use std::sync::Arc;

    fn node(id: &str, query: &str, mode: MaterializeMode, deps: &[&str]) -> TransformNode {
        TransformNode {
            id: id.to_string(),
            query_text: query.to_string(),
            materialize: mode,
            depends_on: deps.iter().map(|s| s.to_string()).collect(),
            schema: None,
        }
    }

    fn make_dag(nodes: Vec<TransformNode>) -> Dag {
        let mut graph = Graph::new(HashMap::new());
        for n in nodes {
            graph.add_node_unchecked(n);
        }
        graph.check().unwrap();
        Dag {
            db: "DuckDB".to_string(),
            nodes: graph,
            sources: vec![],
            max_parallelism: None,
        }
    }

    use MaterializeMode::{Table, TempTable, View};

    /// The worked example from `SRS.md`.
    fn spec_example() -> Dag {
        make_dag(vec![
            node("stg_orders", "SELECT order_id, customer_id, region, amount FROM orders", View, &[]),
            node("stg_customers", "SELECT customer_id, name FROM customers", View, &[]),
            node(
                "order_facts",
                "SELECT o.order_id, o.customer_id, c.name, o.region, o.amount \
                 FROM stg_orders o JOIN stg_customers c ON o.customer_id = c.customer_id",
                View,
                &["stg_orders", "stg_customers"],
            ),
            node(
                "region_sales",
                "SELECT region, sum(amount) AS sales FROM order_facts GROUP BY region",
                View,
                &["order_facts"],
            ),
            node(
                "top_customers",
                "SELECT customer_id, name, sum(amount) AS total, \
                 rank() OVER (ORDER BY sum(amount) DESC) AS rnk \
                 FROM order_facts GROUP BY customer_id, name",
                View,
                &["order_facts"],
            ),
            node(
                "customer_totals",
                "SELECT customer_id, sum(amount) AS total FROM order_facts GROUP BY customer_id",
                Table,
                &["order_facts"],
            ),
            node(
                "region_summary",
                "SELECT r.region, r.sales, (SELECT count(*) FROM customer_totals) AS customers \
                 FROM region_sales r",
                View,
                &["region_sales", "customer_totals"],
            ),
            node("rpt_region", "SELECT * FROM region_sales", Table, &["region_sales"]),
            node(
                "rpt_top",
                "SELECT customer_id, name, total FROM top_customers WHERE rnk <= 10",
                Table,
                &["top_customers"],
            ),
            node("rpt_overview", "SELECT * FROM region_summary", Table, &["region_summary"]),
        ])
    }

    /// Base, a View on it, and three Tables: one that can move into the
    /// rollup, one that orders, one that reads the clock.
    fn order_and_clock_dag() -> Dag {
        make_dag(vec![
            node("base", "SELECT k, v FROM src", View, &[]),
            node("derived", "SELECT k, v * 2 AS v2 FROM base", View, &["base"]),
            node("plain", "SELECT k, sum(v) AS total FROM base GROUP BY k", Table, &["base"]),
            node("ordered", "SELECT k, v FROM base ORDER BY k", Table, &["base"]),
            node("stamped", "SELECT k, v, current_timestamp AS at FROM base", Table, &["base"]),
            node("rpt", "SELECT * FROM derived", Table, &["derived"]),
        ])
    }

    fn s(x: &str) -> String {
        x.to_string()
    }

    // -- selection ---------------------------------------------------------

    #[test]
    fn the_spec_example_counts_paths_not_references() {
        let exec = exec_counts(&spec_example());
        let expected: HashMap<String, usize> = [
            ("stg_orders", 4),
            ("stg_customers", 4),
            ("order_facts", 4),
            ("region_sales", 2),
            ("top_customers", 1),
            ("region_summary", 1),
        ]
        .into_iter()
        .map(|(k, v)| (s(k), v))
        .collect();
        assert_eq!(exec, expected);
    }

    #[test]
    fn the_spec_example_selects_the_spec_cte_set_and_kinds() {
        let sel = select_rollup(&spec_example()).unwrap();

        let members: Vec<(&str, CteReason)> = sel
            .members
            .iter()
            .map(|(k, v)| (k.as_str(), v.clone()))
            .collect();
        assert_eq!(
            members,
            vec![
                ("customer_totals", CteReason::TableQuery { parent: s("order_facts") }),
                ("order_facts", CteReason::Repeated { exec: 4 }),
                ("region_sales", CteReason::Repeated { exec: 2 }),
                ("stg_customers", CteReason::Repeated { exec: 4 }),
                ("stg_orders", CteReason::Repeated { exec: 4 }),
                ("top_customers", CteReason::OnPath),
            ],
            "region_summary stays outside because it reads the table customer_totals"
        );

        let kinds: Vec<(&str, Vec<KindConsumer>)> = sel
            .kinds
            .iter()
            .map(|k| (k.cte.as_str(), k.consumers.clone()))
            .collect();
        assert_eq!(
            kinds,
            vec![
                (
                    "customer_totals",
                    vec![
                        KindConsumer::Own(s("customer_totals")),
                        KindConsumer::Pasted(s("region_summary")),
                    ]
                ),
                (
                    "region_sales",
                    vec![
                        KindConsumer::Pasted(s("region_summary")),
                        KindConsumer::Direct(s("rpt_region")),
                    ]
                ),
                ("top_customers", vec![KindConsumer::Direct(s("rpt_top"))]),
            ]
        );
        assert_eq!(
            sel.kinds.iter().map(|k| k.kind).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );

        assert_eq!(
            sel.default_materialized(),
            BTreeSet::from([s("order_facts")]),
            "only order_facts has two or more readers inside the rollup"
        );

        // Every CTE is emitted after everything it reads.
        let pos = |id: &str| sel.order.iter().position(|x| x == id).unwrap();
        assert_eq!(sel.order.len(), sel.members.len());
        assert!(pos("stg_orders") < pos("order_facts"));
        assert!(pos("stg_customers") < pos("order_facts"));
        assert!(pos("order_facts") < pos("customer_totals"));
        assert!(pos("order_facts") < pos("top_customers"));
    }

    #[test]
    fn two_paths_into_one_table_run_a_view_twice() {
        // v ─┬─ w1 ─┐
        //    └─ w2 ─┴─ t
        let dag = make_dag(vec![
            node("v", "SELECT a FROM src", View, &[]),
            node("w1", "SELECT a FROM v WHERE a > 0", View, &["v"]),
            node("w2", "SELECT a FROM v WHERE a < 0", View, &["v"]),
            node(
                "t",
                "SELECT a FROM w1 UNION ALL SELECT a FROM w2",
                Table,
                &["w1", "w2"],
            ),
        ]);
        let sel = select_rollup(&dag).unwrap();
        assert_eq!(sel.exec["v"], 2, "one Table, but v's SQL is inlined into it twice");
        assert_eq!(sel.members["v"], CteReason::Repeated { exec: 2 });
        assert_eq!(sel.members["w1"], CteReason::OnPath);
        assert_eq!(sel.members["w2"], CteReason::OnPath);
        assert!(!sel.members.contains_key("t"), "t's parents are not shared inside S");
        assert_eq!(sel.default_materialized(), BTreeSet::from([s("v")]));
    }

    #[test]
    fn nothing_repeated_is_not_fusable() {
        let dag = make_dag(vec![
            node("a", "SELECT x FROM src_a", View, &[]),
            node("b", "SELECT y FROM src_b", View, &[]),
            node("ta", "SELECT x FROM a", Table, &["a"]),
            node("tb", "SELECT y FROM b", Table, &["b"]),
        ]);
        assert!(select_rollup(&dag).is_err());
    }

    #[test]
    fn a_table_that_orders_or_reads_the_clock_keeps_its_own_query() {
        let sel = select_rollup(&order_and_clock_dag()).unwrap();
        assert_eq!(
            sel.members.get("plain"),
            Some(&CteReason::TableQuery { parent: s("base") })
        );
        assert!(!sel.members.contains_key("ordered"));
        assert!(!sel.members.contains_key("stamped"));

        let base = sel.kinds.iter().find(|k| k.cte == "base").unwrap();
        assert_eq!(
            base.consumers,
            vec![
                KindConsumer::Direct(s("ordered")),
                KindConsumer::Direct(s("stamped")),
            ]
        );
    }

    #[test]
    fn a_view_reading_a_temp_table_stays_outside() {
        let dag = make_dag(vec![
            node("v", "SELECT a FROM src", View, &[]),
            node("tmp", "SELECT a FROM v", TempTable, &["v"]),
            node("w", "SELECT a FROM tmp", View, &["tmp"]),
            node("t1", "SELECT a FROM v", Table, &["v"]),
            node("t2", "SELECT a FROM w", Table, &["w"]),
            node("t3", "SELECT count(*) AS n FROM w", Table, &["w"]),
        ]);
        let sel = select_rollup(&dag).unwrap();
        assert_eq!(sel.exec["w"], 2);
        assert!(sel.members.contains_key("v"), "a TempTable counts as a stored build");
        assert!(!sel.members.contains_key("w"), "w reads a TempTable built from the rollup");
    }

    #[test]
    fn what_must_run_in_the_table_itself() {
        for dialect in [DialectType::DuckDB, DialectType::PostgreSQL] {
            for sql in [
                "SELECT a FROM t ORDER BY a",
                "SELECT a FROM t LIMIT 10",
                "SELECT a FROM t UNION ALL SELECT a FROM u ORDER BY 1",
                "SELECT a, now() AS at FROM t",
                "SELECT a, current_timestamp AS at FROM t",
                "SELECT a, current_date AS d FROM t",
                "SELECT a FROM t WHERE random() < 0.5",
                "SELECT gen_random_uuid() AS id, a FROM t",
                "SELECT a FROM (SELECT a, now() AS n FROM t) s",
            ] {
                assert!(must_run_in_itself(sql, dialect), "{dialect:?}: {sql}");
            }
            for sql in [
                "SELECT a, rank() OVER (ORDER BY a) AS r FROM t",
                "SELECT a FROM (SELECT a FROM t ORDER BY a LIMIT 5) s",
                "SELECT a, count(*) AS n FROM t GROUP BY a",
            ] {
                assert!(!must_run_in_itself(sql, dialect), "{dialect:?}: {sql}");
            }
        }
    }

    // -- emission, end to end on DuckDB ------------------------------------

    async fn duck() -> Arc<DuckDBConnection> {
        DuckDBConnection::new(DuckDBConfig::new_from_path(":memory:".to_string()))
            .await
            .unwrap()
    }

    async fn exec_sql(conn: &DuckDBConnection, sql: &str) {
        conn.execute(sql.to_string()).await.unwrap();
    }

    type Fingerprint = (String, String, Vec<(String, String)>, i64, String);

    /// Relation type, columns and types in order, row count, and an
    /// order-independent content hash over every column not in `skip`.
    fn fingerprint(conn: &DuckDBConnection, relation: &str, skip: &[&str]) -> Fingerprint {
        let c = conn.pool.get().unwrap();
        let ty: String = c
            .query_row(
                &format!("SELECT table_type FROM information_schema.tables WHERE table_name = '{relation}'"),
                [],
                |r| r.get(0),
            )
            .unwrap();
        let mut stmt = c.prepare(&format!("DESCRIBE SELECT * FROM {relation}")).unwrap();
        let cols: Vec<(String, String)> = stmt
            .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))
            .unwrap()
            .map(|r| r.unwrap())
            .collect();
        let keep: Vec<String> = cols
            .iter()
            .filter(|(n, _)| !skip.contains(&n.as_str()))
            .map(|(n, _)| quote(n))
            .collect();
        let n: i64 = c
            .query_row(&format!("SELECT count(*) FROM {relation}"), [], |r| r.get(0))
            .unwrap();
        let h: String = c
            .query_row(
                &format!(
                    "SELECT coalesce(sum(hash(t))::VARCHAR, '0') FROM (SELECT {} FROM {relation}) AS t",
                    keep.join(", ")
                ),
                [],
                |r| r.get(0),
            )
            .unwrap();
        (relation.to_string(), ty, cols, n, h)
    }

    /// The sort key's values in stored order.
    fn stored_order(conn: &DuckDBConnection, relation: &str, key: &str) -> Vec<String> {
        let c = conn.pool.get().unwrap();
        let mut stmt = c
            .prepare(&format!("SELECT {key}::VARCHAR FROM {relation} ORDER BY rowid"))
            .unwrap();
        stmt.query_map([], |r| r.get::<_, String>(0))
            .unwrap()
            .map(|r| r.unwrap())
            .collect()
    }

    /// Run `dag`, fingerprint every node it had before any rewrite, read the
    /// stored order of `ordered`, then drop what it made.
    async fn run(
        conn: &Arc<DuckDBConnection>,
        dag: &Dag,
        relations: &[String],
        skip: &[&str],
        ordered: &[(&str, &str)],
    ) -> (Vec<Fingerprint>, Vec<Vec<String>>) {
        let engine = SimpleEngine::new(Arc::clone(conn)).unwrap();
        engine.run(dag).await.expect("the DAG runs");
        let fps = relations.iter().map(|r| fingerprint(conn, r, skip)).collect();
        let orders = ordered
            .iter()
            .map(|(r, k)| stored_order(conn, r, k))
            .collect();
        engine.cleanup(dag).await.unwrap();
        (fps, orders)
    }

    fn ids(dag: &Dag) -> Vec<String> {
        let mut v: Vec<String> = dag.nodes.nodes().map(|n| n.id.clone()).collect();
        v.sort();
        v
    }

    async fn rollup(conn: &DuckDBConnection, dag: &mut Dag) -> Emitted {
        let plan = plan_rollup(dag).expect("the DAG has something to share");
        let columns = resolve_kind_columns(conn, dag, &plan).await.expect("column types resolve");
        let emitted = emit(dag, &plan, &columns, &plan.selection.default_materialized())
            .expect("the rollup emits");
        install(dag, &emitted).expect("the rewrite installs");
        emitted
    }

    #[tokio::test]
    async fn the_spec_example_rewrite_delivers_the_same_relations() {
        let conn = duck().await;
        exec_sql(
            &conn,
            "CREATE TABLE orders AS SELECT range AS order_id, range % 7 AS customer_id, \
             CASE range % 3 WHEN 0 THEN 'US' WHEN 1 THEN 'EU' ELSE 'APAC' END AS region, \
             CAST(range * 1.25 AS DECIMAL(10, 2)) AS amount FROM range(200)",
        )
        .await;
        // Customer 6 has orders and no row here: an orphan foreign key.
        exec_sql(
            &conn,
            "CREATE TABLE customers AS SELECT range AS customer_id, 'c' || range AS name FROM range(6)",
        )
        .await;

        let original = spec_example();
        let relations = ids(&original);
        let (before, _) = run(&conn, &original, &relations, &[], &[]).await;

        let mut dag = spec_example();
        let emitted = rollup(&conn, &mut dag).await;
        let (after, _) = run(&conn, &dag, &relations, &[], &[]).await;
        assert_eq!(before, after, "every relation keeps its type, columns and rows");

        // Views are untouched.
        for n in original.nodes.nodes().filter(|n| n.materialize == View) {
            let now = dag.nodes.get(n.id.clone()).unwrap();
            assert_eq!(now.query_text, n.query_text, "{}", n.id);
            assert_eq!(now.depends_on, n.depends_on, "{}", n.id);
        }
        // Every table now reads the rollup and nothing else.
        let fused = HashSet::from([emitted.fused_id.clone()]);
        for t in ["customer_totals", "rpt_region", "rpt_top", "rpt_overview"] {
            assert_eq!(dag.nodes.get(s(t)).unwrap().depends_on, fused, "{t}");
        }
        assert!(emitted.fused_depends_on.is_empty());
        assert!(emitted.untouched.is_empty(), "{:?}", emitted.untouched);

        assert_eq!(emitted.fused_sql.matches("MATERIALIZED").count(), 1);
        assert!(emitted.fused_sql.contains("n_order_facts AS MATERIALIZED"));
        assert!(emitted.fused_sql.trim_end().ends_with("ORDER BY kind"));

        assert_eq!(emitted.pushed.len(), 1, "{:?}", emitted.pushed);
        assert_eq!(emitted.pushed[0].0, 3);
        assert!(emitted.pushed[0].1.contains("rnk"));

        assert_eq!(
            emitted.columns.iter().filter(|c| c.name == "total").count(),
            1,
            "customer_totals and top_customers share `total` at the same type: {:?}",
            emitted.columns
        );
    }

    #[tokio::test]
    async fn stored_order_and_the_clock_stay_with_their_table() {
        let conn = duck().await;
        exec_sql(
            &conn,
            "CREATE TABLE src AS SELECT (range * 7) % 5 AS k, range AS v FROM range(60)",
        )
        .await;

        let original = order_and_clock_dag();
        let relations = ids(&original);
        let (before, before_order) =
            run(&conn, &original, &relations, &["at"], &[("ordered", "k")]).await;

        let mut dag = order_and_clock_dag();
        let emitted = rollup(&conn, &mut dag).await;
        let (after, after_order) =
            run(&conn, &dag, &relations, &["at"], &[("ordered", "k")]).await;

        assert_eq!(before, after);
        assert_eq!(before_order, after_order, "the ORDER BY table keeps its stored order");

        let ordered = dag.nodes.get(s("ordered")).unwrap();
        assert!(ordered.query_text.contains("ORDER BY"), "{}", ordered.query_text);
        let stamped = dag.nodes.get(s("stamped")).unwrap();
        assert!(
            stamped.query_text.to_ascii_uppercase().contains("CURRENT_TIMESTAMP"),
            "{}",
            stamped.query_text
        );
        assert!(!emitted.fused_sql.to_ascii_uppercase().contains("CURRENT_TIMESTAMP"));
        assert_eq!(
            dag.nodes.get(s("plain")).unwrap().query_text,
            format!("SELECT \"k\" AS \"k\", \"total\" AS \"total\" FROM {} WHERE kind = 3", emitted.fused_id),
            "a Table query becomes a projection of its kind"
        );
    }

    #[tokio::test]
    async fn clashing_types_are_renamed_and_a_shadowing_cte_is_left_alone() {
        let conn = duck().await;
        exec_sql(&conn, "CREATE TABLE src AS SELECT range % 4 AS k, range AS v FROM range(40)").await;

        let build = || {
            make_dag(vec![
                node("base", "SELECT k, v FROM src", View, &[]),
                node("p1", "SELECT k, v FROM base", View, &["base"]),
                node("p2", "SELECT k, CAST(v AS VARCHAR) AS v FROM base", View, &["base"]),
                node("t1", "SELECT k, v FROM p1", Table, &["p1"]),
                node("t2", "SELECT k, v FROM p2", Table, &["p2"]),
                node(
                    "t3",
                    "WITH p1 AS (SELECT k, v FROM p1 WHERE k > 1) SELECT k, v FROM p1",
                    Table,
                    &["p1"],
                ),
            ])
        };
        let original = build();
        let relations = ids(&original);
        let (before, _) = run(&conn, &original, &relations, &[], &[]).await;

        let mut dag = build();
        let emitted = rollup(&conn, &mut dag).await;
        let (after, _) = run(&conn, &dag, &relations, &[], &[]).await;
        assert_eq!(before, after);

        let names: Vec<&str> = emitted.columns.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, vec!["k", "v", "p2__v"], "{:?}", emitted.columns);

        let t3 = dag.nodes.get(s("t3")).unwrap();
        assert!(
            t3.depends_on.contains("p1"),
            "t3's own CTE shadows p1, so its reference to the real p1 is kept: {}",
            t3.query_text
        );
    }

    #[tokio::test]
    async fn a_dag_the_rollup_already_serves_has_nothing_left_to_share() {
        let conn = duck().await;
        exec_sql(&conn, "CREATE TABLE src AS SELECT range % 5 AS k, range AS v FROM range(20)").await;
        let mut dag = order_and_clock_dag();
        rollup(&conn, &mut dag).await;
        assert!(plan_rollup(&dag).is_err());
    }

    // -- emission, end to end on PostgreSQL --------------------------------

    /// Against a live server, so ignored by default:
    ///
    /// ```bash
    /// cargo test -p dee --lib nodefusion::rollup -- --ignored
    /// ```
    ///
    /// Each test works in a schema of its own, created and dropped here, so no
    /// existing relation in the database is touched. Connection settings come
    /// from the same `DEE_PG_*` variables as `tests/pg_cost_key.rs`.
    mod postgres {
        use super::*;
        use crate::connectors::postgres::{PostgresConfig, PostgresConnection};
        use sqlx::{
            PgPool, Row,
            postgres::{PgConnectOptions, PgPoolOptions},
        };

        fn env_or(key: &str, default: &str) -> String {
            std::env::var(key).unwrap_or_else(|_| default.to_string())
        }

        async fn connect() -> (Arc<PostgresConnection>, PgPool) {
            let host = env_or("DEE_PG_HOST", "127.0.0.1");
            let port: u16 = env_or("DEE_PG_PORT", "5432").parse().unwrap();
            let user = env_or("DEE_PG_USER", "runner");
            let password = env_or("DEE_PG_PASSWORD", "password");
            let database = env_or("DEE_PG_DB", "benchmark");
            let config: PostgresConfig = serde_json::from_value(serde_json::json!({
                "host": host,
                "port": port as i32,
                "user": user,
                "password": password,
                "database": database,
                "num_connections": 4,
            }))
            .unwrap();
            let conn = PostgresConnection::new(config).await.expect("postgres");
            let pool = PgPoolOptions::new()
                .max_connections(2)
                .connect_with(
                    PgConnectOptions::new()
                        .host(&host)
                        .port(port)
                        .username(&user)
                        .password(&password)
                        .database(&database),
                )
                .await
                .expect("postgres");
            (conn, pool)
        }

        async fn sql(pool: &PgPool, q: &str) {
            sqlx::raw_sql(q)
                .execute(pool)
                .await
                .unwrap_or_else(|e| panic!("{e}: {q}"));
        }

        /// The spec's PostgreSQL fingerprint: relation type, columns with
        /// their exact types, row count, and a sum of per-row hashes.
        async fn fingerprint(pool: &PgPool, schema: &str, relation: &str, skip: &[&str]) -> Fingerprint {
            let bare = relation.rsplit('.').next().unwrap().trim_matches('"');
            let ty: String = sqlx::query_scalar(
                "SELECT table_type::text FROM information_schema.tables \
                 WHERE table_schema = $1 AND table_name = $2",
            )
            .bind(schema)
            .bind(bare)
            .fetch_one(pool)
            .await
            .unwrap();
            let cols: Vec<(String, String)> = sqlx::query(
                "SELECT attname::text, format_type(atttypid, atttypmod) FROM pg_attribute \
                 WHERE attrelid = $1::regclass AND attnum > 0 AND NOT attisdropped ORDER BY attnum",
            )
            .bind(relation)
            .fetch_all(pool)
            .await
            .unwrap()
            .iter()
            .map(|r| (r.get::<String, _>(0), r.get::<String, _>(1)))
            .collect();
            let keep: Vec<String> = cols
                .iter()
                .filter(|(n, _)| !skip.contains(&n.as_str()))
                .map(|(n, _)| quote(n))
                .collect();
            let n: i64 = sqlx::query_scalar(&format!("SELECT count(*) FROM {relation}"))
                .fetch_one(pool)
                .await
                .unwrap();
            let h: String = sqlx::query_scalar(&format!(
                "SELECT coalesce(sum(('x' || substr(md5(t::text), 1, 16))::bit(64)::bigint::numeric), 0)::text \
                 FROM (SELECT {} FROM {relation}) AS t",
                keep.join(", ")
            ))
            .fetch_one(pool)
            .await
            .unwrap();
            (relation.to_string(), ty, cols, n, h)
        }

        async fn stored_order(pool: &PgPool, relation: &str, key: &str) -> Vec<String> {
            sqlx::query_scalar(&format!("SELECT {key}::text FROM {relation} ORDER BY ctid"))
                .fetch_all(pool)
                .await
                .unwrap()
        }

        /// Run `dag` and fingerprint `relations`, then drop what it made.
        /// Returns the run's error rather than panicking, so a caller can still
        /// drop its schema.
        async fn run(
            conn: &Arc<PostgresConnection>,
            pool: &PgPool,
            schema: &str,
            dag: &Dag,
            relations: &[String],
            skip: &[&str],
            ordered: &[(&str, &str)],
        ) -> Result<(Vec<Fingerprint>, Vec<Vec<String>>), String> {
            let engine = SimpleEngine::new(Arc::clone(conn)).unwrap();
            let result = engine.run(dag).await.map_err(|e| format!("{e:?}"));
            let mut fps = Vec::new();
            let mut orders = Vec::new();
            if result.is_ok() {
                for r in relations {
                    fps.push(fingerprint(pool, schema, r, skip).await);
                }
                for (r, k) in ordered {
                    orders.push(stored_order(pool, &format!("{schema}.{r}"), k).await);
                }
            }
            let _ = engine.cleanup(dag).await;
            result.map(|_| (fps, orders))
        }

        fn pg(mut dag: Dag) -> Dag {
            dag.db = "postgres".to_string();
            dag
        }

        fn spec_example_in(sc: &str) -> Dag {
            // Quoted and qualified, the way dee spells node IDs.
            let q = |n: &str| format!("\"{sc}\".\"{n}\"");
            pg(make_dag(vec![
                node(
                    &q("stg_orders"),
                    &format!("SELECT order_id, customer_id, region, amount FROM {}", q("orders")),
                    View,
                    &[],
                ),
                node(
                    &q("stg_customers"),
                    &format!("SELECT customer_id, name FROM {}", q("customers")),
                    View,
                    &[],
                ),
                node(
                    &q("order_facts"),
                    &format!(
                        "SELECT o.order_id, o.customer_id, c.name, o.region, o.amount \
                         FROM {} o JOIN {} c ON o.customer_id = c.customer_id",
                        q("stg_orders"),
                        q("stg_customers")
                    ),
                    View,
                    &[&q("stg_orders"), &q("stg_customers")],
                ),
                node(
                    &q("region_sales"),
                    &format!("SELECT region, sum(amount) AS sales FROM {} GROUP BY region", q("order_facts")),
                    View,
                    &[&q("order_facts")],
                ),
                node(
                    &q("top_customers"),
                    &format!(
                        "SELECT customer_id, name, sum(amount) AS total, \
                         rank() OVER (ORDER BY sum(amount) DESC) AS rnk \
                         FROM {} GROUP BY customer_id, name",
                        q("order_facts")
                    ),
                    View,
                    &[&q("order_facts")],
                ),
                node(
                    &q("customer_totals"),
                    &format!(
                        "SELECT customer_id, sum(amount) AS total FROM {} GROUP BY customer_id",
                        q("order_facts")
                    ),
                    Table,
                    &[&q("order_facts")],
                ),
                node(
                    &q("region_summary"),
                    &format!(
                        "SELECT r.region, r.sales, (SELECT count(*) FROM {}) AS customers FROM {} r",
                        q("customer_totals"),
                        q("region_sales")
                    ),
                    View,
                    &[&q("region_sales"), &q("customer_totals")],
                ),
                node(&q("rpt_region"), &format!("SELECT * FROM {}", q("region_sales")), Table, &[&q("region_sales")]),
                node(
                    &q("rpt_top"),
                    &format!("SELECT customer_id, name, total FROM {} WHERE rnk <= 10", q("top_customers")),
                    Table,
                    &[&q("top_customers")],
                ),
                node(
                    &q("rpt_overview"),
                    &format!("SELECT * FROM {}", q("region_summary")),
                    Table,
                    &[&q("region_summary")],
                ),
            ]))
        }

        fn order_and_clock_in(sc: &str) -> Dag {
            // Quoted and qualified, the way dee spells node IDs.
            let q = |n: &str| format!("\"{sc}\".\"{n}\"");
            pg(make_dag(vec![
                node(&q("base"), &format!("SELECT k, v FROM {}", q("src")), View, &[]),
                node(&q("derived"), &format!("SELECT k, v * 2 AS v2 FROM {}", q("base")), View, &[&q("base")]),
                node(
                    &q("plain"),
                    &format!("SELECT k, sum(v) AS total FROM {} GROUP BY k", q("base")),
                    Table,
                    &[&q("base")],
                ),
                node(&q("ordered"), &format!("SELECT k, v FROM {} ORDER BY k", q("base")), Table, &[&q("base")]),
                node(
                    &q("stamped"),
                    &format!("SELECT k, v, current_timestamp AS at FROM {}", q("base")),
                    Table,
                    &[&q("base")],
                ),
                node(&q("rpt"), &format!("SELECT * FROM {}", q("derived")), Table, &[&q("derived")]),
            ]))
        }

        /// Run the original and the rollup on one schema, drop the schema, then
        /// compare.
        async fn check(
            sc: &str,
            setup: &str,
            build: impl Fn(&str) -> Dag,
            skip: &[&str],
            ordered: &[(&str, &str)],
        ) -> Emitted {
            let (conn, pool) = connect().await;
            sql(&pool, &format!("DROP SCHEMA IF EXISTS {sc} CASCADE; CREATE SCHEMA {sc}; {setup}")).await;

            let original = build(sc);
            let relations = ids(&original);
            let before = run(&conn, &pool, sc, &original, &relations, skip, ordered).await;

            let mut dag = build(sc);
            let emitted = async {
                let plan = plan_rollup(&dag).map_err(|e| e.to_string())?;
                let columns = resolve_kind_columns(conn.as_ref(), &dag, &plan)
                    .await
                    .map_err(|e| e.to_string())?;
                let emitted = emit(&dag, &plan, &columns, &plan.selection.default_materialized())
                    .map_err(|e| e.to_string())?;
                install(&mut dag, &emitted)?;
                Ok::<_, String>(emitted)
            }
            .await;
            let after = match &emitted {
                Ok(_) => Some(run(&conn, &pool, sc, &dag, &relations, skip, ordered).await),
                Err(_) => None,
            };

            sql(&pool, &format!("DROP SCHEMA {sc} CASCADE")).await;

            let emitted = emitted.expect("the rollup plans, resolves and emits");
            let before = before.expect("the original DAG runs");
            let after = after
                .unwrap()
                .unwrap_or_else(|e| panic!("the rewritten DAG runs: {e}\n\n{}", emitted.fused_sql));
            assert_eq!(before, after, "every relation keeps its type, columns, rows and order");
            assert!(emitted.untouched.is_empty(), "{:?}", emitted.untouched);
            assert!(
                emitted.fused_id.starts_with(&format!("\"{sc}\".")),
                "the rollup lands in the schema of the nodes reading it: {}",
                emitted.fused_id
            );
            emitted
        }

        #[tokio::test]
        #[ignore = "needs a live Postgres"]
        async fn the_spec_example_rewrite_delivers_the_same_relations_on_postgres() {
            let sc = "dee_nf_rollup_spec";
            let emitted = check(
                sc,
                &format!(
                    "CREATE TABLE {sc}.orders AS SELECT g AS order_id, g % 7 AS customer_id, \
                       CASE g % 3 WHEN 0 THEN 'US' WHEN 1 THEN 'EU' ELSE 'APAC' END AS region, \
                       CAST(g * 1.25 AS numeric(10, 2)) AS amount \
                     FROM generate_series(0, 199) g; \
                     CREATE TABLE {sc}.customers AS SELECT g AS customer_id, 'c' || g AS name \
                     FROM generate_series(0, 5) g;"
                ),
                spec_example_in,
                &[],
                &[],
            )
            .await;
            // Three branches is what untyped NULL fills could not survive:
            // PostgreSQL resolves UNION types pairwise, and two leading NULLs
            // resolve to text.
            assert_eq!(emitted.fused_sql.matches("UNION ALL").count(), 2);
            assert_eq!(emitted.pushed.len(), 1, "{:?}", emitted.pushed);
            assert_eq!(emitted.fused_sql.matches("MATERIALIZED").count(), 1);
        }

        #[tokio::test]
        #[ignore = "needs a live Postgres"]
        async fn stored_order_and_the_clock_stay_with_their_table_on_postgres() {
            let sc = "dee_nf_rollup_order";
            check(
                sc,
                &format!(
                    "CREATE TABLE {sc}.src AS SELECT (g * 7) % 5 AS k, g AS v FROM generate_series(0, 59) g;"
                ),
                order_and_clock_in,
                &["at"],
                &[("ordered", "k")],
            )
            .await;
        }
    }
}
