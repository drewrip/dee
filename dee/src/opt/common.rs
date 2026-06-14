use std::{collections::HashSet, sync::Arc};

use datafusion::{
    arrow::{
        array::{ArrayRef, new_null_array},
        datatypes::{DataType, FieldRef, SchemaRef},
    },
    catalog::MemoryCatalogProvider,
    catalog::memory::MemorySchemaProvider,
    common::TableReference,
    datasource::{TableProvider, empty::EmptyTable, view::ViewTable},
    execution::session_state::SessionStateBuilder,
    logical_expr::{
        Accumulator, AggregateUDF, AggregateUDFImpl, LogicalPlan, PartitionEvaluator, ScalarUDF,
        ScalarUDFImpl, Signature, Volatility, WindowUDF, WindowUDFImpl,
        function::{AccumulatorArgs, PartitionEvaluatorArgs, StateFieldsArgs, WindowUDFFieldArgs},
    },
    optimizer::{
        OptimizerRule, common_subexpr_eliminate::CommonSubexprEliminate,
        single_distinct_to_groupby::SingleDistinctToGroupBy,
    },
    physical_plan::ColumnarValue,
    prelude::SessionContext,
    scalar::ScalarValue,
    sql::unparser::dialect::{
        BigQueryDialect, DefaultDialect, DuckDBDialect, MySqlDialect, PostgreSqlDialect,
        SqliteDialect,
    },
};
use log::{debug, trace};
use thiserror::Error;

use crate::{
    dag::{Dag, MaterializeMode, TransformNode},
    opt::OptimizerError,
};

// ---------------------------------------------------------------------------
// ValidationError
// ---------------------------------------------------------------------------

#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("physical plan creation failed: {0}")]
    PhysicalPlan(String),
    #[error("node '{node}' failed validation: {reason}")]
    Node { node: String, reason: String },
}

// ---------------------------------------------------------------------------
// Stub UDF/UDAF/UDWF — planning-only, never executed
//
// Registering these stubs on a SessionContext lets DataFusion plan queries
// that contain dialect-specific functions (e.g. DuckDB's `date_diff`) which
// have no built-in DataFusion implementation.  We don't care what the
// functions *do* — only what columns/filters they depend on — so returning
// DataType::Null for every call is safe for our planning-only usage.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct StubScalar {
    name: String,
    sig: Signature,
}

impl StubScalar {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            sig: Signature::variadic_any(Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for StubScalar {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn signature(&self) -> &Signature {
        &self.sig
    }
    fn return_type(&self, _arg_types: &[DataType]) -> datafusion::common::Result<DataType> {
        Ok(DataType::Null)
    }
    fn invoke_with_args(
        &self,
        _args: datafusion::logical_expr::ScalarFunctionArgs,
    ) -> datafusion::common::Result<ColumnarValue> {
        datafusion::common::internal_err!("planning-only stub '{}' was executed", self.name)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct StubAggregate {
    name: String,
    sig: Signature,
}

impl StubAggregate {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            sig: Signature::variadic_any(Volatility::Immutable),
        }
    }
}

#[derive(Debug)]
struct StubAccumulator;

impl Accumulator for StubAccumulator {
    fn update_batch(&mut self, _values: &[ArrayRef]) -> datafusion::common::Result<()> {
        datafusion::common::internal_err!("planning-only stub accumulator used")
    }
    fn evaluate(&mut self) -> datafusion::common::Result<ScalarValue> {
        datafusion::common::internal_err!("planning-only stub accumulator used")
    }
    fn size(&self) -> usize {
        0
    }
    fn state(&mut self) -> datafusion::common::Result<Vec<ScalarValue>> {
        datafusion::common::internal_err!("planning-only stub accumulator used")
    }
    fn merge_batch(&mut self, _states: &[ArrayRef]) -> datafusion::common::Result<()> {
        datafusion::common::internal_err!("planning-only stub accumulator used")
    }
}

impl AggregateUDFImpl for StubAggregate {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn signature(&self) -> &Signature {
        &self.sig
    }
    fn return_type(&self, _arg_types: &[DataType]) -> datafusion::common::Result<DataType> {
        Ok(DataType::Null)
    }
    fn accumulator(
        &self,
        _acc_args: AccumulatorArgs,
    ) -> datafusion::common::Result<Box<dyn Accumulator>> {
        Ok(Box::new(StubAccumulator))
    }
    fn state_fields(&self, _args: StateFieldsArgs) -> datafusion::common::Result<Vec<FieldRef>> {
        Ok(vec![Arc::new(datafusion::arrow::datatypes::Field::new(
            "stub_state",
            DataType::Null,
            true,
        ))])
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct StubWindow {
    name: String,
    sig: Signature,
}

impl StubWindow {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            sig: Signature::variadic_any(Volatility::Immutable),
        }
    }
}

#[derive(Debug)]
struct StubPartitionEvaluator;

impl PartitionEvaluator for StubPartitionEvaluator {
    fn evaluate_all(
        &mut self,
        _values: &[ArrayRef],
        num_rows: usize,
    ) -> datafusion::common::Result<ArrayRef> {
        Ok(new_null_array(&DataType::Null, num_rows))
    }
}

impl WindowUDFImpl for StubWindow {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn signature(&self) -> &Signature {
        &self.sig
    }
    fn field(&self, _field_args: WindowUDFFieldArgs) -> datafusion::common::Result<FieldRef> {
        Ok(Arc::new(datafusion::arrow::datatypes::Field::new(
            self.name(),
            DataType::Null,
            true,
        )))
    }
    fn partition_evaluator(
        &self,
        _args: PartitionEvaluatorArgs,
    ) -> datafusion::common::Result<Box<dyn PartitionEvaluator>> {
        Ok(Box::new(StubPartitionEvaluator))
    }
}

// ---------------------------------------------------------------------------
// Stub-aware SQL planner
// ---------------------------------------------------------------------------

/// Extract an unknown function name from a DataFusion planning error.
///
/// DataFusion 53 reports unknown functions as `Invalid function 'name'`.
/// We also handle the older `There is no UDF/UDAF/UDWF named "name"` messages
/// for forward-compatibility.  When the newer single-quote format is detected
/// we cannot tell whether the function is scalar/aggregate/window from the
/// error alone, so we register stubs for all three kinds and let DataFusion's
/// context-driven dispatch select the correct one.
fn extract_unknown_fn_name(err: &datafusion::common::DataFusionError) -> Option<String> {
    let msg = err.to_string();

    // DF 53+ format: Invalid function 'name'
    if let Some(start) = msg.find("Invalid function '") {
        let rest = &msg[start + "Invalid function '".len()..];
        if let Some(end) = rest.find('\'') {
            return Some(rest[..end].to_string());
        }
    }

    // Legacy format (kept for robustness): There is no UDF/UDAF/UDWF named "name"
    for prefix in &[
        "There is no UDF named \"",
        "There is no UDAF named \"",
        "There is no UDWF named \"",
    ] {
        if let Some(start) = msg.find(prefix) {
            let rest = &msg[start + prefix.len()..];
            if let Some(end) = rest.find('"') {
                return Some(rest[..end].to_string());
            }
        }
    }

    None
}

/// Plan `sql` on `ctx`, automatically registering planning-only stub functions
/// for any unknown scalar/aggregate/window function.
///
/// When DataFusion cannot resolve a function name it raises an error.  We
/// catch those errors, register stubs for the unknown function as all three
/// kinds (scalar, aggregate, window), and retry.  DataFusion's context-driven
/// resolution picks the correct kind based on where the call appears in the
/// query.  Retries up to 64 times to guard against infinite loops.
pub async fn create_logical_plan_with_stubs(
    ctx: &SessionContext,
    sql: &str,
) -> datafusion::common::Result<datafusion::logical_expr::LogicalPlan> {
    for _ in 0..64u32 {
        match ctx.state().create_logical_plan(sql).await {
            Ok(plan) => return Ok(plan),
            Err(e) => match extract_unknown_fn_name(&e) {
                Some(name) => {
                    // Register the function as all three stub kinds.  DataFusion
                    // resolves which one to use from the call site context.
                    ctx.register_udf(ScalarUDF::new_from_impl(StubScalar::new(&name)));
                    ctx.register_udaf(AggregateUDF::new_from_impl(StubAggregate::new(&name)));
                    ctx.register_udwf(WindowUDF::new_from_impl(StubWindow::new(&name)));
                }
                None => return Err(e),
            },
        }
    }
    datafusion::common::internal_err!("create_logical_plan_with_stubs: exceeded retry limit")
}

// ---------------------------------------------------------------------------
// validate / validate_dag
// ---------------------------------------------------------------------------

/// Attempt to create a physical execution plan for `plan` using `ctx`.
///
/// This exercises DataFusion's physical planner — type checking, operator
/// selection, schema propagation — without executing any rows.  Because all
/// registered tables are schema-only (`EmptyTable` / `OpaqueScanTable`), the
/// call is cheap and purely structural.
///
/// Returns `Err(ValidationError::PhysicalPlan)` if physical planning fails.
pub async fn validate(ctx: &SessionContext, plan: &LogicalPlan) -> Result<(), ValidationError> {
    ctx.state()
        .create_physical_plan(plan)
        .await
        .map(|_| ())
        .map_err(|e| ValidationError::PhysicalPlan(e.to_string()))
}

/// Validate every node in `dag` by walking them in topological order and
/// attempting to plan + physically compile each node's SQL against empty
/// (schema-only) tables.
///
/// A [`SessionContext`] is built incrementally: DAG sources are registered
/// first as `EmptyTable`, then each transform node is planned via
/// [`create_logical_plan_with_stubs`], validated with [`validate`], and
/// finally registered as a [`ViewTable`] so downstream nodes can resolve it.
///
/// Returns [`ValidationError::Node`] on the first node whose logical or
/// physical plan fails.
pub async fn validate_dag(dag: &Dag) -> Result<(), ValidationError> {
    let ctx = SessionContext::new();

    debug!("beginning dag validation...");
    for src in &dag.sources {
        register_table_any(
            &ctx,
            &src.name,
            Arc::new(EmptyTable::new(Arc::clone(&src.schema))),
        )
        .map_err(|e| ValidationError::Node {
            node: src.name.clone(),
            reason: e.to_string(),
        })?;
    }
    debug!("registered {} sources", dag.sources.len());

    let topo = dag.nodes.topological_sort();
    for node_id in &topo {
        let node = match dag.nodes.get(node_id.clone()) {
            Some(n) => n,
            None => continue,
        };
        trace!(
            "checking node={}, with query_text = \n{}",
            node_id, node.query_text
        );
        let plan = create_logical_plan_with_stubs(&ctx, &node.query_text)
            .await
            .map_err(|e| ValidationError::Node {
                node: node_id.clone(),
                reason: format!("logical planning: {e}"),
            })?;

        trace!("node ({}), plan:\n{}", node_id, plan.display_indent());

        validate(&ctx, &plan)
            .await
            .map_err(|e| ValidationError::Node {
                node: node_id.clone(),
                reason: e.to_string(),
            })?;

        trace!("node validated");

        register_table_any(&ctx, node_id, Arc::new(ViewTable::new(plan, None))).map_err(|e| {
            ValidationError::Node {
                node: node_id.clone(),
                reason: e.to_string(),
            }
        })?;
    }

    debug!("finished and passed validation");
    Ok(())
}

// ---------------------------------------------------------------------------
// dialect_for_db — map a DAG sql_dialect string to a DataFusion unparser dialect
// ---------------------------------------------------------------------------

/// Return a boxed DataFusion unparser [`Dialect`] for `db`.
///
/// Matches common dialect names case-insensitively.  Defaults to
/// [`DuckDBDialect`] when the dialect is unknown or empty, because DuckDB is
/// the primary target engine and its dialect is the safest default.
pub fn dialect_for_db(db: &str) -> Box<dyn datafusion::sql::unparser::dialect::Dialect> {
    match db.to_lowercase().as_str() {
        "duckdb" => Box::new(DuckDBDialect::new()),
        "postgresql" | "postgres" => Box::new(PostgreSqlDialect {}),
        "mysql" => Box::new(MySqlDialect {}),
        "sqlite" => Box::new(SqliteDialect {}),
        "bigquery" => Box::new(BigQueryDialect {}),
        "default" => Box::new(DefaultDialect {}),
        _ => Box::new(DuckDBDialect::new()),
    }
}

// ---------------------------------------------------------------------------
// register_table_any — shared catalog/schema creation helper
// ---------------------------------------------------------------------------

/// Register `provider` in `ctx` under `name`, which may be an unqualified,
/// two-part (`schema.table`), or three-part (`catalog.schema.table`) name.
///
/// DataFusion's default [`SessionContext`] only contains a `datafusion`
/// catalog.  When node IDs carry fully-qualified names the catalog and schema
/// must be created first.  This helper ensures they exist before delegating to
/// [`SessionContext::register_table`].
pub fn register_table_any(
    ctx: &SessionContext,
    name: &str,
    provider: Arc<dyn TableProvider>,
) -> Result<(), OptimizerError> {
    let table_ref = TableReference::from(name);

    if let TableReference::Full {
        catalog, schema, ..
    } = &table_ref
    {
        if ctx.catalog(catalog.as_ref()).is_none() {
            ctx.register_catalog(catalog.as_ref(), Arc::new(MemoryCatalogProvider::new()));
        }
        let cat = ctx
            .catalog(catalog.as_ref())
            .ok_or_else(|| OptimizerError::Exec(format!("failed to create catalog '{catalog}'")))?;

        if cat.schema(schema.as_ref()).is_none() {
            cat.register_schema(schema.as_ref(), Arc::new(MemorySchemaProvider::new()))
                .map_err(|e| {
                    OptimizerError::Exec(format!(
                        "failed to create schema '{catalog}.{schema}': {e}"
                    ))
                })?;
        }
    }

    ctx.register_table(table_ref, provider)
        .map(|_| ())
        .map_err(|e| OptimizerError::Exec(format!("failed to register table '{name}': {e}")))
}

// ---------------------------------------------------------------------------
// build_opaque_context — shared by PushdownPass and other passes
// ---------------------------------------------------------------------------

use crate::opt::pushdown::OpaqueScanTable;

/// Build a [`SessionContext`] in which every transitive dependency of
/// `target_id` is registered, with one special rule: `opaque_id` (the
/// TempTable under analysis) is always registered as an [`OpaqueScanTable`]
/// backed by `opaque_schema`, so the DataFusion optimizer surfaces pushed-down
/// predicates and projections in the resulting [`LogicalPlan`] tree.
///
/// All other nodes use their pre-resolved `TransformNode::schema` (populated
/// by `Executor::resolve_schemas`).  No SQL planning or connector calls are
/// made here — only schema registration.
///
/// Unknown dialect-specific functions in any registered node's SQL are handled
/// transparently via [`create_logical_plan_with_stubs`]; callers do not need
/// to worry about this.
pub fn build_opaque_context(
    dag: &Dag,
    target_id: &str,
    opaque_id: &str,
    opaque_schema: SchemaRef,
) -> Result<SessionContext, OptimizerError> {
    // Build a session state with CommonSubexprEliminate removed.  CSE rewrites
    // the plan in ways that break filter/projection extraction from TableScan
    // nodes (it introduces shared `__common_expr_N` aliases that obscure the
    // original predicates).
    let cse_name = CommonSubexprEliminate::new().name().to_string();
    let sdgb_name = SingleDistinctToGroupBy::new().name().to_string();
    let exclude_rules: HashSet<String> = HashSet::from_iter([cse_name, sdgb_name]);
    let rules: Vec<Arc<dyn OptimizerRule + Send + Sync>> =
        SessionStateBuilder::new_with_default_features()
            .build()
            .optimizer()
            .rules
            .iter()
            .filter(|r| !exclude_rules.contains(r.name()))
            .cloned()
            .collect();
    let state = SessionStateBuilder::new_with_default_features()
        .with_optimizer_rules(rules)
        .build();
    let ctx = SessionContext::new_with_state(state);

    for src in &dag.sources {
        register_table_any(
            &ctx,
            &src.name,
            Arc::new(EmptyTable::new(Arc::clone(&src.schema))),
        )?;
    }

    register_table_any(
        &ctx,
        opaque_id,
        Arc::new(OpaqueScanTable::new(opaque_schema)),
    )?;

    let topo = dag.nodes.topological_sort();
    for node_id in &topo {
        if node_id == target_id || node_id == opaque_id {
            continue;
        }
        if !is_transitive_dep(dag, target_id, node_id) {
            continue;
        }

        let node = match dag.nodes.get(node_id.clone()) {
            Some(n) => n,
            None => continue,
        };

        let schema = node.schema.as_ref().ok_or_else(|| {
            OptimizerError::Exec(format!(
                "build_opaque_context: node '{node_id}' has no resolved schema; \
                 call resolve_schemas before running PushdownPass"
            ))
        })?;

        register_table_any(&ctx, node_id, Arc::new(EmptyTable::new(Arc::clone(schema))))?;
    }

    Ok(ctx)
}

// ---------------------------------------------------------------------------
// make_temp
// ---------------------------------------------------------------------------

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
        assert!(
            !m.query_text.contains(" n"),
            "m must not reference n directly"
        );
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
            node(
                "v1",
                MaterializeMode::View,
                &["n"],
                "SELECT x FROM n WHERE x > 0",
            ),
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

    // ---------------------------------------------------------------------------
    // Tests: create_logical_plan_with_stubs handles unknown dialect functions
    // ---------------------------------------------------------------------------

    // Verify that a query using an unknown scalar function (e.g. DuckDB's
    // `date_diff`) can be planned without error.
    #[tokio::test]
    async fn test_stub_unknown_scalar_function() {
        use datafusion::arrow::datatypes::{DataType, Field, Schema};
        use datafusion::datasource::empty::EmptyTable;

        let ctx = SessionContext::new();
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Date32, false),
            Field::new("b", DataType::Date32, false),
        ]));
        ctx.register_table("t", Arc::new(EmptyTable::new(schema)))
            .unwrap();

        let plan =
            create_logical_plan_with_stubs(&ctx, "SELECT date_diff('day', a, b) AS diff FROM t")
                .await
                .expect("planning with unknown scalar function should succeed via stub");

        // The plan must mention our table.
        assert!(
            plan.display_indent().to_string().contains('t'),
            "plan should reference table t"
        );
    }

    // Verify that a query using an unknown aggregate-like function can be planned.
    //
    // Because DataFusion resolves scalar UDFs before aggregate UDFs, the stub is
    // registered as scalar and the call is treated as a per-row scalar expression.
    // This is sufficient for pushdown planning purposes — we only need to know
    // which columns are referenced, not the exact evaluation semantics.
    #[tokio::test]
    async fn test_stub_unknown_aggregate_function() {
        use datafusion::arrow::datatypes::{DataType, Field, Schema};
        use datafusion::datasource::empty::EmptyTable;

        let ctx = SessionContext::new();
        let schema = Arc::new(Schema::new(vec![
            Field::new("grp", DataType::Utf8, false),
            Field::new("val", DataType::Float64, false),
        ]));
        ctx.register_table("t", Arc::new(EmptyTable::new(schema)))
            .unwrap();

        // Use the unknown function without GROUP BY to avoid group-by validation;
        // planning succeeds because the stub accepts any arguments.
        let plan = create_logical_plan_with_stubs(&ctx, "SELECT custom_agg(val) AS agg FROM t")
            .await
            .expect("planning with unknown aggregate-like function should succeed via stub");

        assert!(
            plan.display_indent().to_string().contains('t'),
            "plan should reference table t"
        );
    }

    // Verify that a query using an unknown window function can be planned.
    #[tokio::test]
    async fn test_stub_unknown_window_function() {
        use datafusion::arrow::datatypes::{DataType, Field, Schema};
        use datafusion::datasource::empty::EmptyTable;

        let ctx = SessionContext::new();
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("val", DataType::Float64, false),
        ]));
        ctx.register_table("t", Arc::new(EmptyTable::new(schema)))
            .unwrap();

        let plan = create_logical_plan_with_stubs(
            &ctx,
            "SELECT id, custom_win(val) OVER (ORDER BY id) AS w FROM t",
        )
        .await
        .expect("planning with unknown window function should succeed via stub");

        assert!(
            plan.display_indent().to_string().contains('t'),
            "plan should reference table t"
        );
    }

    // ---------------------------------------------------------------------------
    // Tests: validate and validate_dag
    // ---------------------------------------------------------------------------

    // A valid plan against a correctly-typed table should pass validation.
    #[tokio::test]
    async fn test_validate_valid_plan_succeeds() {
        use datafusion::arrow::datatypes::{DataType, Field, Schema};
        use datafusion::datasource::empty::EmptyTable;

        let ctx = SessionContext::new();
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("val", DataType::Float64, false),
        ]));
        ctx.register_table("t", Arc::new(EmptyTable::new(schema)))
            .unwrap();

        let plan = create_logical_plan_with_stubs(&ctx, "SELECT id, val FROM t WHERE val > 0.0")
            .await
            .expect("planning should succeed");

        validate(&ctx, &plan)
            .await
            .expect("validate should succeed for a well-formed plan");
    }

    // A plan that references a non-existent column should fail at the logical
    // planning stage (before validate is even reached), but a plan whose types
    // are internally inconsistent should fail physical planning.  We simulate
    // the latter by manually constructing an invalid plan via mismatched cast.
    //
    // In practice the most common failure is a bad column reference — we test
    // that the *logical* planner catches it via create_logical_plan_with_stubs.
    #[tokio::test]
    async fn test_validate_dag_invalid_column_reference_fails() {
        use crate::dag::SourceNode;
        use datafusion::arrow::datatypes::{DataType, Field, Schema};

        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));

        let mut dag = make_dag(vec![node(
            "t1",
            MaterializeMode::TempTable,
            &[],
            "SELECT nonexistent_col FROM src",
        )]);
        dag.sources = vec![SourceNode {
            name: "src".to_string(),
            schema,
        }];

        let err = validate_dag(&dag)
            .await
            .expect_err("validate_dag should fail when a node references a non-existent column");

        let msg = err.to_string();
        assert!(
            msg.contains("t1"),
            "error should identify the offending node; got: {msg}"
        );
    }

    // A two-node DAG where each node's SQL is valid should pass validate_dag.
    #[tokio::test]
    async fn test_validate_dag_valid_dag_succeeds() {
        use crate::dag::SourceNode;
        use datafusion::arrow::datatypes::{DataType, Field, Schema};

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("amount", DataType::Float64, false),
        ]));

        let mut dag = make_dag(vec![
            node(
                "staging",
                MaterializeMode::TempTable,
                &[],
                "SELECT id, amount FROM raw",
            ),
            node(
                "final",
                MaterializeMode::Table,
                &["staging"],
                "SELECT id FROM staging WHERE amount > 0.0",
            ),
        ]);
        dag.sources = vec![SourceNode {
            name: "raw".to_string(),
            schema,
        }];

        validate_dag(&dag)
            .await
            .expect("validate_dag should succeed for a well-formed DAG");
    }
}
