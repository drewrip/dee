use async_trait::async_trait;
use datafusion::{
    arrow::datatypes::SchemaRef,
    common::{
        TableReference,
        tree_node::{Transformed, TreeNode},
    },
    datasource::{TableProvider, empty::EmptyTable, provider_as_source},
    logical_expr::{LogicalPlan, TableScanBuilder},
    prelude::SessionContext,
    sql::unparser::Unparser,
};
use log::{debug, warn};
use std::{
    collections::{HashMap, HashSet},
    marker::PhantomData,
    sync::Arc,
};

use crate::{
    connectors::Connector,
    dag::{Dag, MaterializeMode, TransformNode},
    executor::Executor,
    opt::{
        OptimizerError, OptimizerPass,
        common::{create_logical_plan_with_stubs, dialect_for_db, register_table_any, schema_prefix},
        pushdown::graph_minor,
    },
};

// ---------------------------------------------------------------------------
// Plan flattening — parent-linked subtree index
// ---------------------------------------------------------------------------

/// One node of a flattened [`LogicalPlan`] tree.  `parent` points back to the
/// index of the enclosing node in the same `Vec<PlanEntry>`, `None` for the
/// tree's root.  This lets us start at any subtree and walk *up* toward the
/// root — the traversal direction the largest-common-subtree search needs.
struct PlanEntry {
    plan: LogicalPlan,
    parent: Option<usize>,
}

/// Flatten `plan` (and all its descendants) into `entries`, recording parent
/// links as we go.  Returns the index at which `plan` itself was stored.
fn flatten_plan(plan: &LogicalPlan, parent: Option<usize>, entries: &mut Vec<PlanEntry>) -> usize {
    let idx = entries.len();
    entries.push(PlanEntry {
        plan: plan.clone(),
        parent,
    });
    for child in plan.inputs() {
        flatten_plan(child, Some(idx), entries);
    }
    idx
}

/// A leaf plan node (no inputs — e.g. a bare `TableScan` or `EmptyRelation`)
/// is never worth factoring out on its own: replacing one leaf with a scan of
/// another node adds indirection without eliminating any computation.
fn is_trivial(plan: &LogicalPlan) -> bool {
    plan.inputs().is_empty()
}

/// A location where a candidate subtree occurs: `(node_id, index into that
/// node's flattened entries)`.
type Occurrence = (String, usize);

fn distinct_node_count(occurrences: &[Occurrence]) -> usize {
    occurrences
        .iter()
        .map(|(id, _)| id.as_str())
        .collect::<HashSet<_>>()
        .len()
}

/// Every distinct subtree appearing anywhere across all planned nodes, keyed
/// by the subtree itself (`LogicalPlan` is `Hash + Eq`, so structurally equal
/// subtrees collide into the same bucket regardless of which node they came
/// from) and mapping to every place that subtree occurs.
fn build_global_map(
    node_entries: &HashMap<String, Vec<PlanEntry>>,
) -> HashMap<LogicalPlan, Vec<Occurrence>> {
    let mut global: HashMap<LogicalPlan, Vec<Occurrence>> = HashMap::new();
    for (node_id, entries) in node_entries {
        for (idx, entry) in entries.iter().enumerate() {
            if is_trivial(&entry.plan) {
                continue;
            }
            global
                .entry(entry.plan.clone())
                .or_default()
                .push((node_id.clone(), idx));
        }
    }
    global
}

/// An occurrence is "climbable" when its immediate parent subtree is *also*
/// shared across two or more distinct nodes — meaning there is a strictly
/// larger common subtree one level up that already covers this one.  Walking
/// up stops (the occurrence is maximal) the first time the parent is either
/// absent (this occurrence is already a whole-plan root) or not shared.
fn is_climbable(
    occ: &Occurrence,
    node_entries: &HashMap<String, Vec<PlanEntry>>,
    global: &HashMap<LogicalPlan, Vec<Occurrence>>,
) -> bool {
    let (node_id, idx) = occ;
    let entries = match node_entries.get(node_id) {
        Some(e) => e,
        None => return false,
    };
    let parent_idx = match entries[*idx].parent {
        Some(p) => p,
        None => return false,
    };
    let parent_plan = &entries[parent_idx].plan;
    match global.get(parent_plan) {
        Some(parent_occurrences) => distinct_node_count(parent_occurrences) >= 2,
        None => false,
    }
}

/// A maximal common subplan: the shared subtree itself, plus the set of
/// planned-node ids whose plan contains it at a maximal (non-climbable)
/// position.
struct Candidate {
    plan: LogicalPlan,
    node_ids: HashSet<String>,
}

/// Find every maximal common subplan across `node_entries`.
///
/// A bucket in the global subtree map qualifies as a candidate when it is
/// shared by two or more distinct nodes *and*, after dropping every occurrence
/// that is climbable (i.e. already subsumed by a larger shared subtree one
/// level up — see [`is_climbable`]), at least two distinct nodes remain.
fn find_maximal_common_subplans(node_entries: &HashMap<String, Vec<PlanEntry>>) -> Vec<Candidate> {
    let global = build_global_map(node_entries);

    let mut candidates: Vec<Candidate> = global
        .iter()
        .filter(|(_, occurrences)| distinct_node_count(occurrences) >= 2)
        .filter_map(|(plan, occurrences)| {
            let node_ids: HashSet<String> = occurrences
                .iter()
                .filter(|occ| !is_climbable(occ, node_entries, &global))
                .map(|(id, _)| id.clone())
                .collect();
            if node_ids.len() >= 2 {
                Some(Candidate {
                    plan: plan.clone(),
                    node_ids,
                })
            } else {
                None
            }
        })
        .collect();

    // Larger / more widely shared subplans first; break ties on a stable
    // textual key so results don't depend on HashMap iteration order.
    candidates.sort_by(|a, b| {
        b.node_ids
            .len()
            .cmp(&a.node_ids.len())
            .then_with(|| a.plan.display_indent().to_string().cmp(&b.plan.display_indent().to_string()))
    });

    candidates
}

// ---------------------------------------------------------------------------
// Building optimized plans, one per Table/TempTable node of the graph minor
// ---------------------------------------------------------------------------

/// Plan and optimize every node of `minor` (a view-free DAG — see
/// [`graph_minor`]), registering each node as a schema-only table in the
/// shared [`SessionContext`] as we go so downstream nodes' plans terminate at
/// a `TableScan` referencing it rather than being expanded further. Keeping
/// DAG node boundaries intact is what lets us later recognize a shared
/// subtree as ending at (or containing) a scan of a real node.
async fn build_optimized_plans(minor: &Dag) -> Result<HashMap<String, LogicalPlan>, OptimizerError> {
    let ctx = SessionContext::new();

    for src in &minor.sources {
        register_table_any(
            &ctx,
            &src.name,
            Arc::new(EmptyTable::new(Arc::clone(&src.schema))),
        )?;
    }

    let mut plans = HashMap::new();
    for node_id in minor.nodes.topological_sort() {
        let node = match minor.nodes.get(node_id.clone()) {
            Some(n) => n,
            None => continue,
        };

        let raw_plan = create_logical_plan_with_stubs(&ctx, &node.query_text)
            .await
            .map_err(|e| OptimizerError::Exec(format!("CSPE: failed to plan node '{node_id}': {e}")))?;

        let opt_plan = ctx.state().optimize(&raw_plan).map_err(|e| {
            OptimizerError::Exec(format!("CSPE: failed to optimize node '{node_id}': {e}"))
        })?;

        let schema = Arc::new(opt_plan.schema().as_arrow().clone());
        register_table_any(&ctx, &node_id, Arc::new(EmptyTable::new(schema)))?;

        plans.insert(node_id, opt_plan);
    }

    Ok(plans)
}

/// Build a `TableScan` over a freshly factored-out node, used to replace every
/// occurrence of a common subplan.
fn table_scan_for(node_id: &str, schema: SchemaRef) -> Result<LogicalPlan, OptimizerError> {
    let provider: Arc<dyn TableProvider> = Arc::new(EmptyTable::new(schema));
    let source = provider_as_source(provider);
    TableScanBuilder::new(TableReference::from(node_id), source)
        .build()
        .map(LogicalPlan::TableScan)
        .map_err(|e| OptimizerError::Exec(format!("CSPE: failed to build TableScan for '{node_id}': {e}")))
}

/// Collect every DAG node id referenced as a `TableScan` inside `plan`, used
/// to compute the `depends_on` set for a newly factored node.
fn collect_dep_ids(plan: &LogicalPlan, minor: &Dag, deps: &mut HashSet<String>) {
    if let LogicalPlan::TableScan(ts) = plan {
        for id in minor.nodes.nodes().map(|n| n.id.clone()) {
            if ts.table_name.resolved_eq(&TableReference::from(id.as_str())) {
                deps.insert(id);
                break;
            }
        }
    }
    for child in plan.inputs() {
        collect_dep_ids(child, minor, deps);
    }
}

// ---------------------------------------------------------------------------
// CSPEPass
// ---------------------------------------------------------------------------

/// Common Subplan Elimination pass.
///
/// Finds `LogicalPlan` subtrees that are structurally identical across two or
/// more DAG nodes and factors each one out into its own `TempTable` node, with
/// every participating node rewritten to scan the new node instead of
/// recomputing the shared logic.
///
/// Algorithm:
/// 1. Compute the [`graph_minor`] of the DAG (View nodes inlined away, leaving
///    only `Table`/`TempTable` nodes).
/// 2. Plan and run the DataFusion optimizer over every node's SQL, producing
///    one optimized [`LogicalPlan`] per node.
/// 3. Flatten each plan into a parent-linked subtree index and group every
///    subtree across all nodes by structural equality. A subtree shared by
///    two or more nodes is a candidate; it is *maximal* if its immediate
///    parent is not itself shared, since climbing higher would find no wider
///    match (per [`find_maximal_common_subplans`]).
/// 4. For each maximal candidate: unparse it to SQL, create a new `TempTable`
///    node for it (wiring up dependencies from the `TableScan`s inside it),
///    and rewrite every participating node's plan to scan the new node in
///    place of the shared subtree, then regenerate that node's `query_text`.
#[derive(Debug, Clone)]
pub struct CSPEPass<C, E>
where
    C: Connector + Send + 'static + Sync,
    E: Executor<C> + Send + Sync,
{
    _conn: Arc<C>,
    _engine: Arc<E>,
    _phantom: PhantomData<C>,
}

impl<C, E> CSPEPass<C, E>
where
    C: Connector + Send + 'static + Sync,
    E: Executor<C> + Send + Sync,
{
    pub fn new(conn: Arc<C>, engine: Arc<E>) -> Self {
        Self {
            _conn: conn,
            _engine: engine,
            _phantom: PhantomData,
        }
    }
}

#[async_trait]
impl<C, E> OptimizerPass<C, E> for CSPEPass<C, E>
where
    C: Connector + Send + 'static + Sync,
    E: Executor<C> + Send + Sync,
{
    async fn run(&mut self, dag: &mut Dag) -> Result<HashMap<String, String>, OptimizerError> {
        debug!("CSPEPass: starting");
        let mut stats = HashMap::new();

        debug!("CSPEPass: computing graph minor");
        let mut minor = graph_minor(dag).await?;
        debug!("CSPEPass: graph minor has {} nodes", minor.nodes.num_nodes());

        debug!("CSPEPass: building optimized plans");
        let plans = build_optimized_plans(&minor).await?;

        let mut node_entries: HashMap<String, Vec<PlanEntry>> = HashMap::new();
        for (node_id, plan) in &plans {
            let mut entries = Vec::new();
            flatten_plan(plan, None, &mut entries);
            node_entries.insert(node_id.clone(), entries);
        }

        let candidates = find_maximal_common_subplans(&node_entries);
        debug!(
            "CSPEPass: found {} maximal common subplan candidate(s)",
            candidates.len()
        );
        stats.insert("common_subplans_found".into(), candidates.len().to_string());

        let dialect = dialect_for_db(&minor.db);
        let mut current_plans = plans.clone();
        let mut rewritten_nodes: HashSet<String> = HashSet::new();
        let mut factored: usize = 0;

        for (i, candidate) in candidates.iter().enumerate() {
            // Place the factored node in the same catalog/schema as one of its
            // participating nodes (deterministically: the lexicographically
            // smallest id) so it lands alongside the queries that use it.
            let sample_id = candidate.node_ids.iter().min().cloned().unwrap_or_default();
            let prefix = schema_prefix(&sample_id);
            let new_id = if prefix.is_empty() {
                format!("cspe_{i}")
            } else {
                format!("{prefix}\"cspe_{i}\"")
            };

            let unparser = Unparser::new(dialect.as_ref());
            let sql = match unparser.plan_to_sql(&candidate.plan) {
                Ok(stmt) => stmt.to_string(),
                Err(e) => {
                    warn!("CSPEPass: skipping candidate, failed to unparse common subplan: {e}");
                    continue;
                }
            };

            let mut deps = HashSet::new();
            collect_dep_ids(&candidate.plan, &minor, &mut deps);
            let new_schema: SchemaRef = Arc::new(candidate.plan.schema().as_arrow().clone());

            let replacement = match table_scan_for(&new_id, Arc::clone(&new_schema)) {
                Ok(p) => p,
                Err(e) => {
                    warn!("CSPEPass: {e}");
                    continue;
                }
            };

            if let Err(e) = minor.nodes.add_node(TransformNode {
                id: new_id.clone(),
                query_text: sql.clone(),
                materialize: MaterializeMode::TempTable,
                depends_on: deps.clone(),
                schema: Some(Arc::clone(&new_schema)),
            }) {
                warn!("CSPEPass: failed to add factored node '{new_id}' to graph minor: {e}");
                continue;
            }
            if let Err(e) = dag.nodes.add_node(TransformNode {
                id: new_id.clone(),
                query_text: sql,
                materialize: MaterializeMode::TempTable,
                depends_on: deps,
                schema: Some(Arc::clone(&new_schema)),
            }) {
                warn!("CSPEPass: failed to add factored node '{new_id}' to DAG: {e}");
                minor.nodes.remove(new_id.clone());
                continue;
            }

            for node_id in &candidate.node_ids {
                let plan = match current_plans.get(node_id) {
                    Some(p) => p.clone(),
                    None => continue,
                };
                let target = candidate.plan.clone();
                let repl = replacement.clone();
                let rewritten = match plan.transform_down(|node| {
                    if node == target {
                        Ok(Transformed::yes(repl.clone()))
                    } else {
                        Ok(Transformed::no(node))
                    }
                }) {
                    Ok(t) => t.data,
                    Err(e) => {
                        warn!("CSPEPass: failed to rewrite '{node_id}': {e}");
                        continue;
                    }
                };
                current_plans.insert(node_id.clone(), rewritten);
                rewritten_nodes.insert(node_id.clone());
            }

            factored += 1;
        }

        for node_id in &rewritten_nodes {
            let plan = &current_plans[node_id];
            let unparser = Unparser::new(dialect.as_ref());
            let sql = match unparser.plan_to_sql(plan) {
                Ok(stmt) => stmt.to_string(),
                Err(e) => {
                    warn!("CSPEPass: could not regenerate SQL for '{node_id}': {e}");
                    continue;
                }
            };

            // The rewritten plan's `TableScan`s are the ground truth for what
            // this node now reads from: it always gains an edge to the new
            // factored node(s) it now scans, and loses edges to any old
            // dependency that no longer appears anywhere in the plan (because
            // every reference to it was inside the replaced subtree).
            let mut new_deps = HashSet::new();
            collect_dep_ids(plan, &minor, &mut new_deps);

            if let Some(n) = dag.nodes.get_mut(node_id.clone()) {
                n.query_text = sql.clone();
                n.depends_on = new_deps.clone();
            }
            if let Some(n) = minor.nodes.get_mut(node_id.clone()) {
                n.query_text = sql;
                n.depends_on = new_deps;
            }
        }

        stats.insert("factored_nodes".into(), factored.to_string());
        stats.insert("nodes_rewritten".into(), rewritten_nodes.len().to_string());
        debug!(
            "CSPEPass: complete — {factored} factored node(s), {} node(s) rewritten",
            rewritten_nodes.len()
        );
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
    use crate::dag::{Dag, MaterializeMode, SourceNode, TransformNode};
    use crate::executor::{ExecStats, Executor, ExecutorError};
    use crate::graph::Graph;
    use async_trait::async_trait;
    use chrono::Utc;
    use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
    use std::collections::HashMap;
    use tokio::sync::watch;

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
        async fn resolve_schemas(&self, _dag: &mut Dag) -> Result<(), ExecutorError> {
            Ok(())
        }
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

    fn pass() -> CSPEPass<StubConnector, StubExecutor> {
        CSPEPass::new(Arc::new(StubConnector), Arc::new(StubExecutor))
    }

    fn raw_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("region", DataType::Utf8, false),
            Field::new("amount", DataType::Float64, false),
        ]))
    }

    // DAG layout:
    //
    //   staging (TempTable)   SELECT id, region, amount FROM raw
    //       ├──► table_a (Table)   SELECT id, amount FROM staging WHERE region = 'US'
    //       └──► table_b (Table)   SELECT id, amount FROM staging WHERE region = 'US'
    //
    // table_a and table_b are identical queries, so their entire optimized
    // plan (Filter → TableScan) is one big common subplan — the maximal match
    // is the whole plan.  Both should be rewritten to scan a new factored node.
    #[tokio::test]
    async fn test_identical_sibling_queries_factored_into_shared_node() {
        let staging = node(
            "staging",
            "SELECT id, region, amount FROM raw",
            MaterializeMode::TempTable,
            &[],
        );
        let table_a = node(
            "table_a",
            "SELECT id, amount FROM staging WHERE region = 'US'",
            MaterializeMode::Table,
            &["staging"],
        );
        let table_b = node(
            "table_b",
            "SELECT id, amount FROM staging WHERE region = 'US'",
            MaterializeMode::Table,
            &["staging"],
        );

        let mut dag = make_dag(vec![staging, table_a, table_b]);
        dag.sources = vec![SourceNode {
            name: "raw".to_string(),
            schema: raw_schema(),
        }];

        let stats = pass().run(&mut dag).await.expect("pass should succeed");
        assert_eq!(stats.get("factored_nodes").map(String::as_str), Some("1"));

        let a = dag.nodes.get("table_a".to_string()).unwrap();
        let b = dag.nodes.get("table_b".to_string()).unwrap();

        assert!(
            a.query_text.to_lowercase().contains("cspe_"),
            "table_a must be rewritten to scan the factored node; got: {}",
            a.query_text
        );
        assert!(
            b.query_text.to_lowercase().contains("cspe_"),
            "table_b must be rewritten to scan the factored node; got: {}",
            b.query_text
        );

        // Exactly one new factored node should exist and it should carry the
        // shared filter predicate.
        let factored_ids: Vec<_> = dag
            .nodes
            .nodes()
            .filter(|n| n.id.starts_with("cspe_"))
            .collect();
        assert_eq!(factored_ids.len(), 1, "exactly one node should be factored out");
        assert!(
            factored_ids[0].query_text.contains("US"),
            "factored node must contain the shared filter; got: {}",
            factored_ids[0].query_text
        );
        assert!(factored_ids[0].depends_on.contains("staging"));

        // The rewritten nodes' edges must be corrected too: they now depend on
        // the factored node, and no longer directly on `staging` (their only
        // reference to it was inside the replaced subtree).
        let factored_id = factored_ids[0].id.clone();
        assert!(
            a.depends_on.contains(&factored_id),
            "table_a must gain an edge to the factored node; got: {:?}",
            a.depends_on
        );
        assert!(
            !a.depends_on.contains("staging"),
            "table_a must drop its now-stale edge to staging; got: {:?}",
            a.depends_on
        );
        assert!(
            b.depends_on.contains(&factored_id),
            "table_b must gain an edge to the factored node; got: {:?}",
            b.depends_on
        );
        assert!(
            !b.depends_on.contains("staging"),
            "table_b must drop its now-stale edge to staging; got: {:?}",
            b.depends_on
        );
    }

    // DAG layout:
    //
    //   staging (TempTable)   SELECT id, region, amount FROM raw
    //       ├──► table_a (Table)   SELECT amount, amount * 2 AS doubled FROM staging WHERE region = 'US'
    //       └──► table_b (Table)   SELECT amount FROM staging WHERE region = 'US'
    //
    // table_a and table_b diverge at the outermost projection, but share the
    // same `Filter(region = 'US') -> TableScan(staging)` subtree beneath it.
    // That subtree — not the whole plan — is the maximal common subplan.
    #[tokio::test]
    async fn test_partial_subtree_shared_beneath_differing_projections() {
        let staging = node(
            "staging",
            "SELECT id, region, amount FROM raw",
            MaterializeMode::TempTable,
            &[],
        );
        let table_a = node(
            "table_a",
            "SELECT amount, amount * 2 AS doubled FROM staging WHERE region = 'US'",
            MaterializeMode::Table,
            &["staging"],
        );
        let table_b = node(
            "table_b",
            "SELECT amount FROM staging WHERE region = 'US'",
            MaterializeMode::Table,
            &["staging"],
        );

        let mut dag = make_dag(vec![staging, table_a, table_b]);
        dag.sources = vec![SourceNode {
            name: "raw".to_string(),
            schema: raw_schema(),
        }];

        let stats = pass().run(&mut dag).await.expect("pass should succeed");
        assert_eq!(stats.get("factored_nodes").map(String::as_str), Some("1"));

        let a = dag.nodes.get("table_a".to_string()).unwrap();
        let b = dag.nodes.get("table_b".to_string()).unwrap();

        assert!(
            a.query_text.to_lowercase().contains("cspe_"),
            "table_a must reference the factored node; got: {}",
            a.query_text
        );
        assert!(
            b.query_text.to_lowercase().contains("cspe_"),
            "table_b must reference the factored node; got: {}",
            b.query_text
        );
        // table_a's distinguishing computation must survive the rewrite.
        assert!(
            a.query_text.to_lowercase().contains("doubled") || a.query_text.contains('*'),
            "table_a must retain its own projection on top of the shared subplan; got: {}",
            a.query_text
        );

        // Edges must be corrected the same way as the whole-plan case: both
        // nodes' only reference to `staging` was inside the replaced
        // Filter->TableScan subtree, so both must drop that edge and gain one
        // to the factored node instead.
        let factored_id = dag
            .nodes
            .nodes()
            .find(|n| n.id.starts_with("cspe_"))
            .expect("a factored node must exist")
            .id
            .clone();
        assert!(a.depends_on.contains(&factored_id));
        assert!(!a.depends_on.contains("staging"));
        assert!(b.depends_on.contains(&factored_id));
        assert!(!b.depends_on.contains("staging"));
    }

    // DAG layout:
    //
    //   staging (TempTable)   SELECT id, region, amount FROM raw
    //       ├──► table_a (Table)   SELECT amount FROM staging WHERE region = 'US'
    //       └──► table_b (Table)   SELECT amount FROM staging WHERE region = 'EU'
    //
    // No two nodes share a common subplan (different filters, no shared
    // Filter/TableScan subtree), so the pass must leave the DAG untouched.
    #[tokio::test]
    async fn test_no_common_subplan_leaves_dag_unchanged() {
        let staging = node(
            "staging",
            "SELECT id, region, amount FROM raw",
            MaterializeMode::TempTable,
            &[],
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

        let mut dag = make_dag(vec![staging, table_a, table_b]);
        dag.sources = vec![SourceNode {
            name: "raw".to_string(),
            schema: raw_schema(),
        }];

        let stats = pass().run(&mut dag).await.expect("pass should succeed");
        assert_eq!(stats.get("factored_nodes").map(String::as_str), Some("0"));

        let factored: Vec<_> = dag.nodes.nodes().filter(|n| n.id.starts_with("cspe_")).collect();
        assert!(factored.is_empty(), "no node should be factored out");
    }
}
