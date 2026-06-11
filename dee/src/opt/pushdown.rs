use async_trait::async_trait;
use datafusion::{
    arrow::datatypes::SchemaRef,
    catalog::Session,
    common::{
        Column, TableReference,
        tree_node::{Transformed, TreeNode},
    },
    datasource::{TableProvider, empty::EmptyTable, view::ViewTable},
    logical_expr::{
        Expr, LogicalPlan, TableProviderFilterPushDown, TableType,
        utils::{conjunction, disjunction},
    },
    physical_plan::ExecutionPlan,
    prelude::SessionContext,
    sql::unparser::expr_to_sql,
};
use log::{debug, trace, warn};
use std::{
    collections::{HashMap, HashSet},
    marker::PhantomData,
    sync::Arc,
};

use crate::{
    connectors::Connector,
    dag::MaterializeMode,
    executor::Executor,
    opt::common::{
        build_opaque_context, create_logical_plan_with_stubs, is_transitive_dep, register_table_any,
    },
    opt::{Dag, OptimizerError, OptimizerPass},
};

/// A schema-only [`TableProvider`] that declares `Exact` filter pushdown
/// support for every predicate.
///
/// Registering the TempTable under analysis with this provider causes
/// DataFusion's optimizer to push both filter predicates AND column projections
/// directly into the [`TableScan`](datafusion::logical_expr::LogicalPlan::TableScan)
/// node rather than leaving them as separate plan nodes above the scan.  We
/// can then read `TableScan.filters` and `TableScan.projection` to recover
/// exactly what would be pushed down to the TempTable.
#[derive(Debug)]
pub(crate) struct OpaqueScanTable {
    schema: SchemaRef,
}

impl OpaqueScanTable {
    pub(crate) fn new(schema: SchemaRef) -> Self {
        Self { schema }
    }
}

#[async_trait]
impl TableProvider for OpaqueScanTable {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> datafusion::common::Result<Vec<TableProviderFilterPushDown>> {
        // Declare every predicate as Exact so DataFusion moves filters fully
        // into TableScan.filters and removes the Filter nodes above the scan.
        Ok(vec![TableProviderFilterPushDown::Exact; filters.len()])
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
        // Schema-only table — delegate to EmptyTable for the execution plan.
        EmptyTable::new(Arc::clone(&self.schema))
            .scan(_state, projection, _filters, _limit)
            .await
    }
}

/// Inline the SQL query of `source` (a View node) into `target` (a Table or
/// TempTable node), returning the DataFusion [`LogicalPlan`] for `target` with
/// every reference to `source` replaced by `source`'s own query.
///
/// Steps:
/// 1. Build a [`SessionContext`] and register each DAG source table with its
///    Arrow schema so the planner has type information.
/// 2. Walk the DAG in topological order, registering every node that `target`
///    transitively depends on as a [`ViewTable`] (backed by the node's planned
///    [`LogicalPlan`]).  This means `source` is registered as a view whose
///    definition is its own `query_text`.
/// 3. Parse and plan `target`'s `query_text`.  DataFusion resolves the
///    `source` reference through the catalog and folds the view definition
///    inline, giving back a fully-inlined [`LogicalPlan`].
pub async fn inline<C>(
    dag: &Dag,
    source: &str,
    target: &str,
    conn: &C,
) -> Result<LogicalPlan, OptimizerError>
where
    C: Connector + Send + Sync,
{
    let target_node = dag
        .nodes
        .get(target.to_string())
        .ok_or_else(|| OptimizerError::Exec(format!("target node '{target}' not found")))?;

    // Verify that `source` exists and is reachable from `target`.
    if dag.nodes.get(source.to_string()).is_none() {
        return Err(OptimizerError::Exec(format!(
            "source node '{source}' not found"
        )));
    }
    if !is_transitive_dep(dag, target, source) {
        return Err(OptimizerError::Exec(format!(
            "source node '{source}' is not a transitive dependency of target '{target}'"
        )));
    }

    let ctx = SessionContext::new();

    // Register every DAG source (raw tables) with its schema so that the
    // planner can resolve column types when building logical plans.
    for src in &dag.sources {
        register_table_any(
            &ctx,
            &src.name,
            Arc::new(EmptyTable::new(Arc::clone(&src.schema))),
        )?;
    }

    // Walk nodes in topological order and register each one that `target`
    // transitively depends on as a ViewTable.  We stop after registering
    // `source` — nodes that `target` doesn't depend on are irrelevant.
    let topo = dag.nodes.topological_sort();
    for node_id in &topo {
        if node_id == target {
            break;
        }

        let node = match dag.nodes.get(node_id.clone()) {
            Some(n) => n,
            None => continue,
        };

        // Skip nodes that are not in `target`'s transitive dependency set.
        if !is_transitive_dep(dag, target, node_id) {
            continue;
        }

        // If the connector can provide a schema for this node (it has already
        // been materialized as a table/temp table), prefer that so the planner
        // gets accurate types.  Otherwise fall back to planning the SQL text.
        let plan = if matches!(
            node.materialize,
            MaterializeMode::Table | MaterializeMode::TempTable
        ) {
            if let Some(Ok(schema)) = conn.get_schema(node_id.clone()).await {
                register_table_any(&ctx, node_id, Arc::new(EmptyTable::new(schema)))?;
                continue;
            } else {
                create_logical_plan_with_stubs(&ctx, &node.query_text)
                    .await
                    .map_err(|e| {
                        OptimizerError::Exec(format!("failed to plan node '{node_id}': {e}"))
                    })?
            }
        } else {
            create_logical_plan_with_stubs(&ctx, &node.query_text)
                .await
                .map_err(|e| {
                    OptimizerError::Exec(format!("failed to plan node '{node_id}': {e}"))
                })?
        };

        register_table_any(
            &ctx,
            node_id,
            Arc::new(ViewTable::new(plan, Some(node.query_text.clone()))),
        )?;
    }

    // Plan `target`'s query.  DataFusion resolves the `source` reference
    // through the catalog and folds the view definition inline automatically.
    create_logical_plan_with_stubs(&ctx, &target_node.query_text)
        .await
        .map_err(|e| OptimizerError::Exec(format!("failed to plan target '{target}': {e}")))
}

/// Extract the bare (unquoted, unqualified) table name from a node ID that may
/// be a one-, two-, or three-part quoted identifier such as
/// `"warehouse"."main"."stg_accounts"`.
///
/// Used to produce a safe SQL alias when falling back to text-based subquery
/// inlining inside [`graph_minor`].
fn bare_table_name(node_id: &str) -> String {
    // Split on `.` and take the last segment, then strip surrounding `"`.
    node_id
        .split('.')
        .last()
        .unwrap_or(node_id)
        .trim_matches('"')
        .to_string()
}

// ---------------------------------------------------------------------------
// pushdown helpers
// ---------------------------------------------------------------------------

/// Walk an optimized [`LogicalPlan`] tree and extract the filter predicates and
/// projected-column names from the [`TableScan`] node for `source_id`.
///
/// Because the opaque TempTable is registered with [`OpaqueScanTable`] which
/// declares `Exact` filter pushdown support, DataFusion's optimizer moves all
/// predicate and projection information directly into the `TableScan` node.
/// We therefore read only from there — no need to collect filters from
/// surrounding `Filter` nodes in the tree.
///
/// Returns `Some((filters, projected_column_names))` when the scan is found,
/// `None` otherwise.
fn extract_pushdowns(plan: &LogicalPlan, source_id: &str) -> Option<(Vec<Expr>, Vec<String>)> {
    match plan {
        // Target scan found — read filters and projection directly from it.
        // Use resolved_eq so that a fully-qualified node ID like
        // `"warehouse"."main"."account_health"` matches a TableScan whose
        // table_name was parsed as Full { catalog, schema, table }.
        LogicalPlan::TableScan(ts)
            if ts.table_name.resolved_eq(&TableReference::from(source_id)) =>
        {
            let filters: Vec<Expr> = ts
                .filters
                .iter()
                .flat_map(|f| split_conjunction(f))
                .collect();

            // Translate the projection (column index list) into column names.
            let cols: Vec<String> = match &ts.projection {
                Some(indices) => {
                    let full_fields = ts.source.schema().fields().clone();
                    indices
                        .iter()
                        .filter_map(|&i| full_fields.get(i).map(|f| f.name().clone()))
                        .collect()
                }
                // No explicit projection → all columns needed.
                None => vec![],
            };

            Some((filters, cols))
        }

        // Recurse into every input branch, accumulating across all matches.
        // A single query can reference the same table more than once (self-join,
        // UNION, CTE expanded twice), producing multiple TableScan nodes for
        // the same source_id.  find_map would silently drop all but the first.
        other => {
            let mut all_filters: Vec<Expr> = vec![];
            let mut all_cols: Vec<String> = vec![];
            let mut found = false;
            for child in other.inputs() {
                if let Some((filters, cols)) = extract_pushdowns(child, source_id) {
                    all_filters.extend(filters);
                    all_cols.extend(cols);
                    found = true;
                }
            }
            if found {
                Some((all_filters, all_cols))
            } else {
                None
            }
        }
    }
}

/// Split a possibly-conjunctive [`Expr`] into its individual AND clauses.
fn split_conjunction(expr: &Expr) -> Vec<Expr> {
    match expr {
        Expr::BinaryExpr(b) if matches!(b.op, datafusion::logical_expr::Operator::And) => {
            let mut parts = split_conjunction(&b.left);
            parts.extend(split_conjunction(&b.right));
            parts
        }
        other => vec![other.clone()],
    }
}

/// Strips table qualifiers from every `Column` reference inside `expr`.
///
/// The optimizer emits predicates like `staging.region = 'US'`; when we
/// re-apply those predicates to `source`'s own base plan (whose output columns
/// are unqualified or carry a different qualifier) the planner would reject the
/// expression.  Removing the qualifier makes the column name resolve against
/// whatever relation is in scope.
fn strip_table_qualifier(expr: Expr) -> Expr {
    expr.transform_down(|e| match e {
        Expr::Column(col) => Ok(Transformed::yes(Expr::Column(Column::from_name(col.name)))),
        other => Ok(Transformed::no(other)),
    })
    // transform_down is infallible here; unwrap is safe.
    .unwrap()
    .data
}

/// Pushes down predicates and projections from the frontier materializing nodes
/// into the TempTable node `source`, returning the rewritten SQL for `source`.
///
/// For each node in `frontier_materializes(source)`:
/// 1. The DataFusion logical optimizer is run on that node's plan (with
///    `source` registered as an opaque scan so the optimizer surfaces any
///    predicates/projections it can push into `source`).
/// 2. Filter predicates adjacent to the `source` scan are collected and
///    combined across all frontier nodes with a logical **OR**.
/// 3. Projected columns are collected and **unioned** across all frontier
///    nodes (so every consumer's required columns are present).
///
/// The combined filter and projection are then applied to `source`'s own
/// query.  When DataFusion can plan the source query, the result is a fully
/// optimized [`LogicalPlan`] unparsed back to SQL.  When the source query
/// uses dialect-specific functions DataFusion cannot plan (e.g. DuckDB's
/// `date_diff`), the wrapper is constructed directly as a SQL string so the
/// original query is preserved verbatim.
pub async fn pushdown(dag: &Dag, source: &str) -> Result<(String, SchemaRef), OptimizerError> {
    let source_node = dag
        .nodes
        .get(source.to_string())
        .ok_or_else(|| OptimizerError::Exec(format!("source node '{source}' not found")))?;

    if !matches!(source_node.materialize, MaterializeMode::TempTable) {
        return Err(OptimizerError::Exec(format!(
            "pushdown: '{source}' is not a TempTable"
        )));
    }

    // Use the pre-resolved schema from resolve_schemas — no DataFusion planning
    // or connector calls needed here.
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

    // Collect filter predicates (one list per frontier node) and the union of
    // required columns across all frontier nodes.
    let frontier: HashSet<String> = dag.nodes.frontier_materializes(source);

    // per-frontier-node filter predicates (each entry = the filters for one node)
    let mut per_node_filters: Vec<Vec<Expr>> = Vec::new();
    // union of projected columns across all frontier nodes
    let mut required_cols: HashSet<String> = HashSet::new();
    let mut any_node_needs_all_cols = false;

    trace!(
        "  source '{}' query text =\n{}",
        source_node.id, source_node.query_text,
    );

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

        let ctx = build_opaque_context(dag, n_id, source, Arc::clone(&source_schema))?;

        let raw_plan = create_logical_plan_with_stubs(&ctx, &n_node.query_text)
            .await
            .map_err(|e| {
                OptimizerError::Exec(format!(
                    "pushdown: failed to plan frontier node '{n_id}': {e}"
                ))
            })?;

        let opt_plan = ctx.state().optimize(&raw_plan).map_err(|e| {
            OptimizerError::Exec(format!(
                "pushdown: failed to optimize frontier node '{n_id}': {e}"
            ))
        })?;

        trace!(
            "  frontier '{n_id}': optimized plan =\n{}",
            opt_plan.display_indent()
        );

        match extract_pushdowns(&opt_plan, source) {
            Some((filters, cols)) => {
                let filter_strs: Vec<String> = filters
                    .iter()
                    .filter_map(|f| expr_to_sql(&strip_table_qualifier(f.clone())).ok())
                    .map(|e| e.to_string())
                    .collect();
                if filter_strs.is_empty() {
                    trace!("  frontier '{n_id}': no filter predicates found");
                } else {
                    trace!(
                        "  frontier '{n_id}': predicates = [{}]",
                        filter_strs.join(", ")
                    );
                }
                if cols.is_empty() {
                    trace!("  frontier '{n_id}': needs all columns");
                    any_node_needs_all_cols = true;
                } else {
                    trace!(
                        "  frontier '{n_id}': projected columns = [{}]",
                        cols.join(", ")
                    );
                    required_cols.extend(cols);
                }
                if !filters.is_empty() {
                    per_node_filters.push(filters);
                }
            }
            // Scan for source not found in this frontier node's plan — treat
            // conservatively: assume all columns needed, no filter extractable.
            None => {
                trace!(
                    "  frontier '{n_id}': source scan not found in plan, assuming all columns needed"
                );
                any_node_needs_all_cols = true;
            }
        }
    }

    // Build the combined filter: OR of each frontier node's AND-conjunction.
    // Strip table qualifiers so the expressions resolve against the source's
    // own output schema rather than the opaque scan's qualified columns.
    let combined_filter: Option<Expr> = {
        let per_node_conjunctions: Vec<Expr> = per_node_filters
            .into_iter()
            .filter_map(|fs| conjunction(fs.into_iter().map(strip_table_qualifier)))
            .collect();
        disjunction(per_node_conjunctions)
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
        return Ok((source_node.query_text.clone(), Arc::clone(&source_schema)));
    }

    // Construct the final SQL by wrapping the original query as a subquery and
    // applying the pushed-down filter and projection as an outer SELECT.
    // This is dialect-agnostic: the original query is preserved verbatim, and
    // only the outermost SELECT and WHERE are added by us.
    let alias = bare_table_name(source);

    let col_list = match &projection_cols {
        Some(cols) if !cols.is_empty() => cols.join(", "),
        _ => "*".to_string(),
    };

    let where_clause = match combined_filter {
        Some(expr) => {
            let sql_expr = expr_to_sql(&strip_table_qualifier(expr)).map_err(|e| {
                OptimizerError::Exec(format!("pushdown: expr_to_sql for '{source}': {e}"))
            })?;
            let filter_str = sql_expr.to_string();
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
            use datafusion::arrow::datatypes::Schema;
            let fields: Vec<_> = cols
                .iter()
                .filter_map(|name| source_schema.field_with_name(name).ok().cloned())
                .collect();
            Arc::new(Schema::new(fields))
        }
        _ => Arc::clone(&source_schema),
    };

    Ok((
        format!(
            "SELECT {col_list} FROM ({inner}) AS \"{alias}\"{where_clause}",
            inner = source_node.query_text,
        ),
        new_schema,
    ))
}

/// Produces a copy of `dag` with every `View` node eliminated by inlining
/// its SQL query into every downstream `Table` or `TempTable` that reads from
/// it.  The returned DAG contains no `View` nodes.
///
/// Algorithm (repeated until no views remain):
/// 1. Find all `(view v, non-view table t)` edges where `t` directly depends
///    on `v`.
/// 2. For each such pair: inline `v`'s query into `t` with [`inline`], update
///    `t.query_text`, drop the edge `t → v`, and add edges from `t` to every
///    node that `v` itself depends on (so `t` retains `v`'s transitive deps).
/// 3. Any `View` that has become a sink (nothing depends on it any more) is
///    removed from the graph together with all of its in-edges.
/// 4. Repeat until no `View` nodes are left.
pub async fn graph_minor(dag: &Dag) -> Result<Dag, OptimizerError> {
    let mut minor = dag.clone();

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
            let alias = bare_table_name(view_id);
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

            // Wrap the view's SQL in a subquery and replace the view reference.
            // We do NOT add an explicit AS alias here: if the original SQL uses
            // a table alias (e.g. `FROM "warehouse"."main"."stg_accounts" a`),
            // that alias `a` is preserved in-place after the substitution,
            // giving `FROM (view_sql) a`.  Adding an extra `AS "view_name"`
            // would create two consecutive aliases, which is invalid SQL.
            let _ = alias; // alias derived but not used in substitution
            let new_sql = current_table_sql.replace(view_id.as_str(), &format!("({view_sql})"));

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
    _conn: Arc<C>,
    engine: Arc<E>,
    _phantom: PhantomData<C>,
}

impl<C, E> PushdownPass<C, E>
where
    C: Connector + Send + 'static + Sync,
    E: Executor<C> + Send + Sync,
{
    pub fn new(conn: Arc<C>, engine: Arc<E>) -> Self {
        Self {
            _conn: conn,
            engine,
            _phantom: PhantomData,
        }
    }
}

#[async_trait]
impl<C, E> OptimizerPass<C, E> for PushdownPass<C, E>
where
    C: Connector + Send + 'static + Sync,
    E: Executor<C> + Send + Sync,
{
    async fn run(&mut self, dag: &mut Dag) -> Result<HashMap<String, String>, OptimizerError> {
        debug!("PushdownPass: starting");
        let mut stats = HashMap::new();

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
        stats.insert("temp_tables_count".into(), temp_table_ids.len().to_string());

        // Step 3+4 — for each TempTable, run pushdown on the minor DAG and
        // write the resulting SQL back to the *original* DAG.
        let mut rewrites: usize = 0;
        for node_id in &temp_table_ids {
            // If no materializing nodes sit downstream of this TempTable there
            // is nothing to push down — skip it entirely.
            if minor.nodes.frontier_materializes(node_id).is_empty() {
                debug!("PushdownPass: '{node_id}' has no materializing frontier, skipping");
                continue;
            }

            debug!("PushdownPass: running pushdown on '{node_id}'");

            // If pushdown or unparsing fails for this node (e.g. the SQL uses
            // dialect-specific constructs DataFusion cannot plan), skip it and
            // leave its query_text unchanged.  The pass is best-effort.
            let (new_sql, new_schema) = match pushdown(&minor, node_id).await {
                Ok(result) => result,
                Err(e) => {
                    warn!("PushdownPass: skipping '{node_id}', pushdown failed: {e}");
                    continue;
                }
            };

            let original_sql = minor
                .nodes
                .get(node_id.clone())
                .map(|n| n.query_text.as_str())
                .unwrap_or("");

            if new_sql == original_sql {
                debug!("PushdownPass: '{node_id}' unchanged (nothing pushed down)");
                continue;
            }

            debug!(
                "PushdownPass: '{node_id}' rewritten ({} chars)",
                new_sql.len()
            );

            let node = dag.nodes.get_mut(node_id.clone()).ok_or_else(|| {
                OptimizerError::Exec(format!(
                    "PushdownPass: node '{node_id}' missing from original DAG"
                ))
            })?;
            node.query_text = new_sql;
            node.schema = Some(new_schema);

            rewrites += 1;
        }

        stats.insert("rewrites_applied".into(), rewrites.to_string());
        debug!("PushdownPass: complete — {rewrites} rewrite(s) applied");
        Ok(stats)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectors::{Connector, ConnectorError};
    use crate::dag::{Dag, MaterializeMode, TransformNode};
    use crate::executor::{ExecStats, Executor, ExecutorError};
    use crate::graph::Graph;
    use async_trait::async_trait;
    use chrono::Utc;
    use datafusion::arrow::datatypes::SchemaRef;
    use std::collections::{HashMap, HashSet};
    use tokio::sync::watch;

    // ------------------------------------------------------------------
    // Stub connector/executor — PushdownPass stores them behind PhantomData
    // and never calls their methods.
    // ------------------------------------------------------------------

    #[derive(Debug, Default)]
    struct StubConnector;

    #[async_trait]
    impl Connector for StubConnector {
        type Config = ();
        type Connection = StubConnector;

        async fn new(_: ()) -> Result<Arc<Self::Connection>, ConnectorError> {
            Ok(Arc::new(StubConnector))
        }
        async fn execute(&self, _: String) -> Result<usize, ConnectorError> {
            Ok(0)
        }
        async fn new_relation(
            &self,
            _: MaterializeMode,
            _: String,
            _: String,
        ) -> Result<usize, ConnectorError> {
            Ok(0)
        }
        async fn drop_relation(
            &self,
            _: MaterializeMode,
            _: String,
        ) -> Result<usize, ConnectorError> {
            Ok(0)
        }
        async fn get_schema(&self, _: String) -> Option<Result<SchemaRef, ConnectorError>> {
            None
        }
    }

    struct StubExecutor;

    #[async_trait]
    impl Executor<StubConnector> for StubExecutor {
        type ExecutionEngine = StubExecutor;

        fn new(_: Arc<StubConnector>) -> Result<StubExecutor, ExecutorError> {
            Ok(StubExecutor)
        }
        async fn run(&self, _: &Dag) -> Result<ExecStats, ExecutorError> {
            let now = Utc::now();
            Ok(ExecStats {
                start: now,
                finish: now,
                duration: chrono::TimeDelta::zero(),
                node_stats: Default::default(),
                system_samples: vec![],
            })
        }
        async fn cleanup(&self, _: &Dag) -> Result<usize, ExecutorError> {
            Ok(0)
        }
        fn cancel_sender(&self) -> Arc<watch::Sender<bool>> {
            Arc::new(watch::channel(false).0)
        }
        async fn resolve_schemas(&self, dag: &mut Dag) -> Result<(), ExecutorError> {
            // In tests we use DataFusion to derive schemas (no live DB).
            // Walk nodes in topological order, building a SessionContext as we go.
            use crate::opt::common::{create_logical_plan_with_stubs, register_table_any};
            use datafusion::datasource::empty::EmptyTable;
            use datafusion::datasource::view::ViewTable;
            use datafusion::prelude::SessionContext;

            let ctx = SessionContext::new();
            for src in &dag.sources {
                register_table_any(
                    &ctx,
                    &src.name,
                    Arc::new(EmptyTable::new(Arc::clone(&src.schema))),
                )
                .map_err(|e| ExecutorError::Exec(e.to_string()))?;
            }

            let topo = dag.nodes.topological_sort();
            let mut planned: Vec<(String, datafusion::arrow::datatypes::SchemaRef)> = Vec::new();
            for node_id in &topo {
                let node = dag.nodes.get(node_id.clone()).unwrap();
                let plan = create_logical_plan_with_stubs(&ctx, &node.query_text)
                    .await
                    .map_err(|e| ExecutorError::Exec(format!("resolve_schemas test stub: {e}")))?;
                let schema = Arc::new(plan.schema().as_arrow().clone());
                register_table_any(&ctx, node_id, Arc::new(ViewTable::new(plan, None)))
                    .map_err(|e| ExecutorError::Exec(e.to_string()))?;
                planned.push((node_id.clone(), schema));
            }
            for (node_id, schema) in planned {
                if let Some(n) = dag.nodes.get_mut(node_id) {
                    n.schema = Some(schema);
                }
            }
            Ok(())
        }
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

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

    fn pass() -> PushdownPass<StubConnector, StubExecutor> {
        PushdownPass::new(Arc::new(StubConnector), Arc::new(StubExecutor))
    }

    // ------------------------------------------------------------------
    // Integration tests
    // ------------------------------------------------------------------

    // DAG layout:
    //
    //   raw (View)
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
        use crate::dag::SourceNode;
        use datafusion::arrow::datatypes::{DataType, Field, Schema};

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("amount", DataType::Float64, false),
            Field::new("status", DataType::Utf8, false),
        ]));

        let raw = node(
            "raw",
            "SELECT id, amount, status FROM source_table",
            MaterializeMode::View,
            &[],
        );
        let temp = node(
            "staging",
            "SELECT id, amount, status FROM raw",
            MaterializeMode::TempTable,
            &["raw"],
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

        let mut dag = make_dag(vec![raw, temp, summary, report]);
        dag.sources = vec![SourceNode {
            name: "source_table".to_string(),
            schema,
        }];
        let original = dag
            .nodes
            .get("staging".to_string())
            .unwrap()
            .query_text
            .clone();

        pass().run(&mut dag).await.expect("pass should succeed");

        assert_eq!(
            dag.nodes.get("staging".to_string()).unwrap().query_text,
            original,
            "TempTable with only View consumers must not be rewritten"
        );
    }

    // DAG layout:
    //
    //   source (View)
    //       │
    //   staging (TempTable)
    //       │
    //   final_table (Table)   SELECT region, total FROM staging WHERE region = 'US'
    //
    // There is a TABLE downstream, so the pass should push down the filter
    // `region = 'US'` that the Table applies.  Projection pruning is governed
    // by what the Table's query actually selects.
    #[tokio::test]
    async fn test_filter_pushed_into_temp_table_with_single_table_downstream() {
        use crate::dag::SourceNode;
        use datafusion::arrow::datatypes::{DataType, Field, Schema};

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("region", DataType::Utf8, false),
            Field::new("total", DataType::Float64, false),
        ]));

        let source = node("source", "SELECT * FROM raw", MaterializeMode::View, &[]);
        let staging = node(
            "staging",
            "SELECT id, region, total FROM source",
            MaterializeMode::TempTable,
            &["source"],
        );
        let sink = node(
            "final_table",
            "SELECT region, total FROM staging WHERE region = 'US'",
            MaterializeMode::Table,
            &["staging"],
        );

        let mut dag = make_dag(vec![source, staging, sink]);
        dag.sources = vec![SourceNode {
            name: "raw".to_string(),
            schema,
        }];
        pass().run(&mut dag).await.expect("pass should succeed");

        let rewritten = dag
            .nodes
            .get("staging".to_string())
            .unwrap()
            .query_text
            .clone();

        // plan_to_sql qualifies column names (e.g. "raw"."region") so we check
        // for the literal value that must appear in the WHERE predicate.
        assert!(
            rewritten.contains("'US'"),
            "filter predicate 'US' should be pushed into the TempTable; got: {}",
            rewritten
        );
    }

    // DAG layout:
    //
    //   source (View)
    //       │
    //   staging (TempTable)
    //       ├──► table_a (Table)   SELECT amount FROM staging WHERE region = 'US'
    //       └──► table_b (Table)   SELECT amount FROM staging WHERE region = 'US'
    //
    // Both Table consumers share the same filter.  The pass should push it
    // into the TempTable.
    #[tokio::test]
    async fn test_common_filter_pushed_when_multiple_table_consumers_agree() {
        use crate::dag::SourceNode;
        use datafusion::arrow::datatypes::{DataType, Field, Schema};

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("region", DataType::Utf8, false),
            Field::new("amount", DataType::Float64, false),
        ]));

        let source = node("source", "SELECT * FROM raw", MaterializeMode::View, &[]);
        let staging = node(
            "staging",
            "SELECT id, region, amount FROM source",
            MaterializeMode::TempTable,
            &["source"],
        );
        let table_a = node(
            "table_a",
            "SELECT amount FROM staging WHERE region = 'US'",
            MaterializeMode::Table,
            &["staging"],
        );
        let table_b = node(
            "table_b",
            "SELECT amount FROM staging WHERE region = 'US'",
            MaterializeMode::Table,
            &["staging"],
        );

        let mut dag = make_dag(vec![source, staging, table_a, table_b]);
        dag.sources = vec![SourceNode {
            name: "raw".to_string(),
            schema,
        }];
        pass().run(&mut dag).await.expect("pass should succeed");

        let rewritten = dag
            .nodes
            .get("staging".to_string())
            .unwrap()
            .query_text
            .clone();

        assert!(
            rewritten.contains("'US'"),
            "filter predicate 'US' should be pushed when all Table consumers agree; got: {}",
            rewritten
        );
    }

    // DAG layout:
    //
    //   source (View)
    //       │
    //   staging (TempTable)
    //       ├──► table_a (Table)   SELECT amount FROM staging WHERE region = 'US'
    //       └──► table_b (Table)   SELECT amount FROM staging WHERE region = 'EU'
    //
    // The two Table consumers apply different filters.  The pass should push
    // down the logical OR of all consumer filters so that every row any Table
    // needs is still present in the TempTable.
    #[tokio::test]
    async fn test_different_filters_across_table_consumers_are_pushed_as_or() {
        use crate::dag::SourceNode;
        use datafusion::arrow::datatypes::{DataType, Field, Schema};

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("region", DataType::Utf8, false),
            Field::new("amount", DataType::Float64, false),
        ]));

        let source = node("source", "SELECT * FROM raw", MaterializeMode::View, &[]);
        let staging = node(
            "staging",
            "SELECT id, region, amount FROM source",
            MaterializeMode::TempTable,
            &["source"],
        );
        let table_a = node(
            "table_a",
            "SELECT amount FROM staging WHERE region = 'US'",
            MaterializeMode::Table,
            &["staging"],
        );
        let table_b = node(
            "table_b",
            "SELECT amount FROM staging WHERE region = 'EU'",
            MaterializeMode::Table,
            &["staging"],
        );

        let mut dag = make_dag(vec![source, staging, table_a, table_b]);
        dag.sources = vec![SourceNode {
            name: "raw".to_string(),
            schema,
        }];
        pass().run(&mut dag).await.expect("pass should succeed");

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
    }

    // DAG layout:
    //
    //   source (View)
    //       │
    //   staging (TempTable)
    //       ├──► table_a (Table)   SELECT region, amount FROM staging WHERE region = 'US'
    //       └──► table_b (Table)   SELECT region, amount FROM staging WHERE region = 'EU'
    //
    // Both Table consumers select the same column subset `region, amount` —
    // the TempTable can be pruned to only those columns (plus whatever the OR
    // filter requires, which is already covered).  The unused column `id`
    // should not appear in the rewritten query.
    #[tokio::test]
    async fn test_projection_pruned_to_union_of_columns_needed_by_table_consumers() {
        use crate::dag::SourceNode;
        use datafusion::arrow::datatypes::{DataType, Field, Schema};

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("region", DataType::Utf8, false),
            Field::new("amount", DataType::Float64, false),
        ]));

        let source = node("source", "SELECT * FROM raw", MaterializeMode::View, &[]);
        let staging = node(
            "staging",
            "SELECT id, region, amount FROM source",
            MaterializeMode::TempTable,
            &["source"],
        );
        let table_a = node(
            "table_a",
            "SELECT region, amount FROM staging WHERE region = 'US'",
            MaterializeMode::Table,
            &["staging"],
        );
        let table_b = node(
            "table_b",
            "SELECT region, amount FROM staging WHERE region = 'EU'",
            MaterializeMode::Table,
            &["staging"],
        );

        let mut dag = make_dag(vec![source, staging, table_a, table_b]);
        dag.sources = vec![SourceNode {
            name: "raw".to_string(),
            schema: Arc::clone(&schema),
        }];
        pass().run(&mut dag).await.expect("pass should succeed");

        let rewritten = dag
            .nodes
            .get("staging".to_string())
            .unwrap()
            .query_text
            .clone();

        // The combined OR filter must be present.
        assert!(
            rewritten.contains("'US'") && rewritten.contains("'EU'"),
            "OR filter predicates must be pushed into the TempTable query; got: {}",
            rewritten
        );
        assert!(
            rewritten.contains("OR"),
            "filter predicates must be combined with OR; got: {}",
            rewritten
        );

        // `id` is not needed by either consumer — it must not appear in the
        // outer SELECT projection.  It may still appear inside the preserved
        // inner subquery, but the outer SELECT determines what gets materialised.
        let outer_proj = rewritten.split("FROM (").next().unwrap_or("");
        assert!(
            !outer_proj.contains("\"id\"")
                && !outer_proj
                    .split_whitespace()
                    .any(|t| t.trim_matches(',') == "id"),
            "column `id` must not appear in the outer SELECT projection; got: {}",
            rewritten
        );
    }

    // ------------------------------------------------------------------
    // graph_minor tests
    // ------------------------------------------------------------------

    // DAG layout:
    //
    //   raw (source, arrow schema)
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
        use crate::dag::SourceNode;
        use datafusion::arrow::datatypes::{DataType, Field, Schema};

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("amount", DataType::Float64, false),
        ]));
        let source = SourceNode {
            name: "raw".to_string(),
            schema,
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

        // No views should remain.
        let view_count = minor
            .nodes
            .nodes()
            .filter(|n| matches!(n.materialize, MaterializeMode::View))
            .count();
        assert_eq!(view_count, 0, "result DAG must contain no View nodes");

        // summary must still be present as a Table.
        let summary_node = minor
            .nodes
            .get("summary".to_string())
            .expect("summary Table must still exist");
        assert!(
            matches!(summary_node.materialize, MaterializeMode::Table),
            "summary must remain a Table"
        );

        // The view's filter logic must be present in the inlined query.
        assert!(
            summary_node.query_text.contains("amount > 0")
                || summary_node.query_text.contains("amount > 0.0"),
            "inlined query must contain the view's filter predicate; got: {}",
            summary_node.query_text
        );
        // `cleaned` must no longer exist as an independent table reference —
        // it should appear only as a subquery alias (if at all), meaning `FROM
        // cleaned` without a surrounding subquery is gone.
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
        use crate::dag::SourceNode;
        use datafusion::arrow::datatypes::{DataType, Field, Schema};

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("val", DataType::Int64, false),
        ]));
        let source = SourceNode {
            name: "raw".to_string(),
            schema,
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

        // The chained filter must be present in the fully-inlined query.
        assert!(
            output_node.query_text.contains("val > 0"),
            "inlined query must contain the chained filter predicate; got: {}",
            output_node.query_text
        );
        // Neither view should appear as a bare FROM target any more.
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

    // ------------------------------------------------------------------
    // pushdown tests
    // ------------------------------------------------------------------

    use crate::dag::SourceNode;
    use datafusion::arrow::datatypes::{DataType, Field, Schema};

    /// Build a minimal DAG with one raw source, one TempTable, and one or more
    /// Table sinks, with schemas resolved via `StubExecutor` so `pushdown` can
    /// be called directly.
    async fn orders_dag(
        staging_query: &str,
        sinks: Vec<(&str, &str)>, // (node_id, query_text)
    ) -> Dag {
        let schema = Arc::new(Schema::new(vec![
            Field::new("order_id", DataType::Int64, false),
            Field::new("region", DataType::Utf8, false),
            Field::new("amount", DataType::Float64, false),
            Field::new("status", DataType::Utf8, false),
        ]));
        let source = SourceNode {
            name: "orders".to_string(),
            schema,
        };

        let staging = node(
            "staging",
            staging_query,
            MaterializeMode::TempTable,
            &[], // raw source is not a graph node
        );

        let mut all_nodes = vec![staging];
        for (id, query) in &sinks {
            all_nodes.push(node(id, query, MaterializeMode::Table, &["staging"]));
        }

        let mut dag = make_dag(all_nodes);
        dag.sources = vec![source];
        StubExecutor
            .resolve_schemas(&mut dag)
            .await
            .expect("resolve_schemas should succeed in test");
        dag
    }

    // DAG layout:
    //
    //   orders (raw source, schema: order_id, region, amount, status)
    //       │
    //   staging (TempTable)   SELECT * FROM orders
    //       │
    //   us_orders (Table)     SELECT order_id, amount FROM staging WHERE region = 'US'
    //
    // The single frontier Table filters on `region = 'US'` and projects only
    // `order_id` and `amount`.  The `pushdown` function should produce a plan
    // for `staging` that:
    //   - Applies `region = 'US'` as a filter (the optimizer has no reason to
    //     weaken it when there is only one consumer).
    //   - Projects only the columns actually needed: `order_id` and `amount`.
    //     (`region` is needed for the filter itself; `status` is unreferenced.)
    #[tokio::test]
    async fn test_pushdown_single_table_filter_and_projection() {
        let dag = orders_dag(
            "SELECT order_id, region, amount, status FROM orders",
            vec![(
                "us_orders",
                "SELECT order_id, amount FROM staging WHERE region = 'US'",
            )],
        )
        .await;

        let (sql, _schema) = pushdown(&dag, "staging")
            .await
            .expect("pushdown should succeed");

        assert!(
            sql.contains("US"),
            "optimized SQL must contain the filter predicate 'US'; got:\n{sql}"
        );
        // `status` must not appear in the outer SELECT projection.  It will
        // still be present in the preserved inner subquery, but the outer
        // SELECT determines what the TempTable actually materialises.
        let outer_select = sql.split("FROM (").next().unwrap_or("");
        assert!(
            !outer_select.contains("status"),
            "column `status` should be absent from the outer SELECT projection; got:\n{sql}"
        );
    }

    // DAG layout:
    //
    //   orders (raw source)
    //       │
    //   staging (TempTable)   SELECT * FROM orders
    //       ├──► us_orders (Table)   SELECT order_id, amount FROM staging WHERE region = 'US'
    //       └──► eu_orders (Table)   SELECT order_id, amount FROM staging WHERE region = 'EU'
    //
    // Two frontier Tables with *different* region filters.  The `pushdown`
    // function must OR the two predicates so that both consumers' rows survive.
    // Both consumers select the same columns (`order_id`, `amount`), so the
    // projection can still be pruned — `status` is not needed by either.
    #[tokio::test]
    async fn test_pushdown_multiple_tables_filters_combined_with_or() {
        let dag = orders_dag(
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

        let (sql, _schema) = pushdown(&dag, "staging")
            .await
            .expect("pushdown should succeed");

        // Both filter arms must appear in the SQL.
        assert!(
            sql.contains("US"),
            "SQL must contain the 'US' filter arm; got:\n{sql}"
        );
        assert!(
            sql.contains("EU"),
            "SQL must contain the 'EU' filter arm; got:\n{sql}"
        );
        // `status` must not appear in the outer SELECT projection.
        let outer_select = sql.split("FROM (").next().unwrap_or("");
        assert!(
            !outer_select.contains("status"),
            "column `status` should be absent from the outer SELECT projection; got:\n{sql}"
        );
    }
}
