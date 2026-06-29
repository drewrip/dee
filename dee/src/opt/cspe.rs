use std::{
    collections::{HashMap, HashSet},
    hash::{DefaultHasher, Hash, Hasher},
    marker::PhantomData,
    sync::Arc,
};

use async_trait::async_trait;
use datafusion::{
    common::{TableReference, tree_node::{Transformed, TreeNode}},
    datasource::{empty::EmptyTable, provider_as_source},
    logical_expr::{LogicalPlan, LogicalPlanBuilder},
    sql::unparser::Unparser,
};
use log::{debug, warn};

use crate::{
    connectors::Connector,
    dag::{Dag, MaterializeMode, TransformNode},
    executor::Executor,
    opt::{
        OptimizerError, OptimizerPass,
        common::{dialect_for_db, optimized_plan},
        pushdown::graph_minor,
    },
};

// ---------------------------------------------------------------------------
// Schema prefix helper
// ---------------------------------------------------------------------------

/// Extract the catalog+schema prefix from a qualified node ID.
///
/// `"warehouse"."main"."foo"` → `"warehouse"."main".`
/// `"foo"`                    → `` (empty)
fn schema_prefix(node_id: &str) -> String {
    if let Some(pos) = node_id.rfind("\".\"") {
        format!("{}\".", &node_id[..pos])
    } else {
        String::new()
    }
}

// ---------------------------------------------------------------------------
// Plan hashing
// ---------------------------------------------------------------------------

fn hash_plan(plan: &LogicalPlan) -> u64 {
    let mut hasher = DefaultHasher::new();
    plan.hash(&mut hasher);
    hasher.finish()
}

// ---------------------------------------------------------------------------
// Subtree collection
// ---------------------------------------------------------------------------

/// Walk `plan` top-down, inserting every unique subtree (keyed by hash) into
/// `map`.  When two nodes hash identically the first one wins (they are
/// structurally equal, so either clone suffices).
fn collect_subtrees(plan: &LogicalPlan) -> HashMap<u64, LogicalPlan> {
    let mut map = HashMap::new();
    collect_subtrees_inner(plan, &mut map);
    map
}

fn collect_subtrees_inner(plan: &LogicalPlan, map: &mut HashMap<u64, LogicalPlan>) {
    map.entry(hash_plan(plan)).or_insert_with(|| plan.clone());
    for child in plan.inputs() {
        collect_subtrees_inner(child, map);
    }
}

// ---------------------------------------------------------------------------
// Maximal common subtree search
// ---------------------------------------------------------------------------

/// Returns true for leaf `TableScan` nodes — a bare scan carries no
/// computation worth factoring out.
fn is_trivial(plan: &LogicalPlan) -> bool {
    matches!(plan, LogicalPlan::TableScan(_))
}

/// Walk `plan` top-down, collecting every **maximal** non-trivial node whose
/// hash appears in `common_hashes`.
///
/// "Maximal" means no ancestor was itself a match: once a matching root is
/// found the subtree below it is not explored, so each common subplan is
/// reported exactly once at its highest point.
fn find_maximal_common_subtrees(
    plan: &LogicalPlan,
    common_hashes: &HashSet<u64>,
    result: &mut Vec<LogicalPlan>,
) {
    let h = hash_plan(plan);
    if !is_trivial(plan) && common_hashes.contains(&h) {
        result.push(plan.clone());
        // Do NOT recurse — everything below belongs to this common subtree.
        return;
    }
    for child in plan.inputs() {
        find_maximal_common_subtrees(child, common_hashes, result);
    }
}

// ---------------------------------------------------------------------------
// Subplan replacement
// ---------------------------------------------------------------------------

/// Replace every occurrence of `target` (matched by structural equality via
/// hash + `PartialEq`) in `plan` with `replacement`, returning the rewritten
/// plan.
fn replace_subplan(
    plan: LogicalPlan,
    target_hash: u64,
    target: &LogicalPlan,
    replacement: &LogicalPlan,
) -> datafusion::common::Result<LogicalPlan> {
    plan.transform_down(|node| {
        if hash_plan(&node) == target_hash && &node == target {
            Ok(Transformed::yes(replacement.clone()))
        } else {
            Ok(Transformed::no(node))
        }
    })
    .map(|t| t.data)
}

// ---------------------------------------------------------------------------
// DAG node reference extraction
// ---------------------------------------------------------------------------

/// Collect all `TableScan` table names in `plan` that match a known DAG node
/// ID using `TableReference::resolved_eq` (handles qualified vs. unqualified
/// names correctly).
fn extract_dag_node_refs(plan: &LogicalPlan, known_ids: &[&str]) -> HashSet<String> {
    let mut names = HashSet::new();
    extract_refs_inner(plan, known_ids, &mut names);
    names
}

fn extract_refs_inner(plan: &LogicalPlan, known_ids: &[&str], names: &mut HashSet<String>) {
    if let LogicalPlan::TableScan(ts) = plan {
        for &id in known_ids {
            if ts.table_name.resolved_eq(&TableReference::from(id)) {
                names.insert(id.to_string());
                break;
            }
        }
    }
    for child in plan.inputs() {
        extract_refs_inner(child, known_ids, names);
    }
}

// ---------------------------------------------------------------------------
// CSPEPass
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CSPEPass<C, E>
where
    C: Connector + Send + 'static,
    E: Executor<C> + Send,
{
    _conn: Arc<C>,
    _engine: Arc<E>,
    _phantom: PhantomData<C>,
}

impl<C, E> CSPEPass<C, E>
where
    C: Connector + Send + 'static,
    E: Executor<C> + Send,
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

        // Step 1: Compute graph minor — eliminate all View nodes by inlining
        // their SQL into downstream Table/TempTable nodes.
        debug!("CSPEPass: computing graph minor");
        let minor = graph_minor(dag).await?;
        let node_count = minor.nodes.num_nodes();
        debug!("CSPEPass: graph minor has {node_count} node(s)");
        stats.insert("graph_minor_nodes".into(), node_count.to_string());

        if node_count < 2 {
            stats.insert("common_subplans_found".into(), "0".into());
            return Ok(stats);
        }

        // Step 2: For each node in the minor DAG, build an optimized LogicalPlan
        // and collect a hash → subtree map.
        let all_ids: Vec<String> = minor.nodes.topological_sort();

        let mut plans: HashMap<String, LogicalPlan> = HashMap::new();
        let mut subtree_maps: HashMap<String, HashMap<u64, LogicalPlan>> = HashMap::new();

        for node_id in &all_ids {
            match optimized_plan(&minor, node_id).await {
                Ok(plan) => {
                    let subtrees = collect_subtrees(&plan);
                    debug!(
                        "CSPEPass: '{node_id}' — {} unique subtrees",
                        subtrees.len()
                    );
subtree_maps.insert(node_id.clone(), subtrees);
                    plans.insert(node_id.clone(), plan);
                }
                Err(e) => {
                    warn!("CSPEPass: failed to plan '{node_id}', skipping: {e}");
                }
            }
        }

        let all_dag_ids: Vec<String> = dag.nodes.nodes().map(|n| n.id.clone()).collect();
        let dialect = dialect_for_db(&dag.db);
        let mut cspe_counter = 0usize;
        let mut subplans_found = 0usize;

        // Iterate over pairs.  We snapshot the key list once; after each
        // factoring the plans and subtree_maps are updated in-place so
        // subsequent pairs see the latest versions.
        let planned_ids: Vec<String> = plans.keys().cloned().collect();

        // Step 3 & 4: Find and factor out the largest maximal common subplan
        // for each (A, B) pair.
        for i in 0..planned_ids.len() {
            for j in (i + 1)..planned_ids.len() {
                let id_a = &planned_ids[i];
                let id_b = &planned_ids[j];

                let common_hashes: HashSet<u64> = {
                    let subtrees_a = match subtree_maps.get(id_a) {
                        Some(m) => m,
                        None => continue,
                    };
                    let subtrees_b = match subtree_maps.get(id_b) {
                        Some(m) => m,
                        None => continue,
                    };
                    subtrees_a
                        .keys()
                        .filter(|h| subtrees_b.contains_key(h))
                        .copied()
                        .collect()
                };

                if common_hashes.is_empty() {
                    continue;
                }

                let plan_a = match plans.get(id_a) {
                    Some(p) => p.clone(),
                    None => continue,
                };
                let plan_b = match plans.get(id_b) {
                    Some(p) => p.clone(),
                    None => continue,
                };

                // Walk plan_a top-down to find maximal common subtrees.
                let mut maximal: Vec<LogicalPlan> = Vec::new();
                find_maximal_common_subtrees(&plan_a, &common_hashes, &mut maximal);

                if maximal.is_empty() {
                    continue;
                }

                // Pick the largest candidate (most nodes in its subtree).
                let best = maximal
                    .iter()
                    .max_by_key(|p| collect_subtrees(p).len())
                    .unwrap()
                    .clone();

                debug!(
                    "CSPEPass: common subplan between '{id_a}' and '{id_b}' — \
                     {} maximal candidate(s), best has {} subtree node(s)",
                    maximal.len(),
                    collect_subtrees(&best).len()
                );

                // Generate SQL for the new TempTable using the Unparser.
                let cspe_sql = match Unparser::new(dialect.as_ref()).plan_to_sql(&best) {
                    Ok(stmt) => stmt.to_string(),
                    Err(e) => {
                        warn!(
                            "CSPEPass: cannot unparse common subplan for \
                             '{id_a}'/'{id_b}': {e}"
                        );
                        continue;
                    }
                };

                // Name the new node using the same catalog/schema prefix as A.
                let prefix = schema_prefix(id_a);
                let cspe_name = if prefix.is_empty() {
                    format!("cspe_{cspe_counter}")
                } else {
                    format!("{prefix}\"cspe_{cspe_counter}\"")
                };
                cspe_counter += 1;

                debug!("CSPEPass: materializing as '{cspe_name}': {cspe_sql}");

                // Build a scan LogicalPlan for cspe_name backed by the common
                // subplan's output schema.
                let arrow_schema = Arc::new(best.schema().as_arrow().clone());
                let replacement = match LogicalPlanBuilder::scan(
                    TableReference::from(cspe_name.as_str()),
                    provider_as_source(Arc::new(EmptyTable::new(Arc::clone(&arrow_schema)))),
                    None,
                ) {
                    Ok(b) => match b.build() {
                        Ok(p) => p,
                        Err(e) => {
                            warn!("CSPEPass: failed to build scan plan for '{cspe_name}': {e}");
                            continue;
                        }
                    },
                    Err(e) => {
                        warn!("CSPEPass: failed to create scan for '{cspe_name}': {e}");
                        continue;
                    }
                };

                let target_hash = hash_plan(&best);

                // Rewrite A's and B's plans by replacing the common subplan
                // with a scan of cspe_name.
                let new_plan_a =
                    match replace_subplan(plan_a, target_hash, &best, &replacement) {
                        Ok(p) => p,
                        Err(e) => {
                            warn!("CSPEPass: replace_subplan failed for '{id_a}': {e}");
                            continue;
                        }
                    };
                let new_plan_b =
                    match replace_subplan(plan_b, target_hash, &best, &replacement) {
                        Ok(p) => p,
                        Err(e) => {
                            warn!("CSPEPass: replace_subplan failed for '{id_b}': {e}");
                            continue;
                        }
                    };

                // Unparse the rewritten plans to SQL strings.
                let new_sql_a =
                    match Unparser::new(dialect.as_ref()).plan_to_sql(&new_plan_a) {
                        Ok(stmt) => stmt.to_string(),
                        Err(e) => {
                            warn!("CSPEPass: cannot unparse modified plan for '{id_a}': {e}");
                            continue;
                        }
                    };
                let new_sql_b =
                    match Unparser::new(dialect.as_ref()).plan_to_sql(&new_plan_b) {
                        Ok(stmt) => stmt.to_string(),
                        Err(e) => {
                            warn!("CSPEPass: cannot unparse modified plan for '{id_b}': {e}");
                            continue;
                        }
                    };

                // Compute depends_on sets from the TableScan references in
                // each plan (only references to known DAG nodes count).
                let id_slice: Vec<&str> = all_dag_ids.iter().map(|s| s.as_str()).collect();

                let cspe_deps: HashSet<String> = extract_dag_node_refs(&best, &id_slice);

                let mut new_deps_a: HashSet<String> =
                    extract_dag_node_refs(&new_plan_a, &id_slice);
                new_deps_a.insert(cspe_name.clone());

                let mut new_deps_b: HashSet<String> =
                    extract_dag_node_refs(&new_plan_b, &id_slice);
                new_deps_b.insert(cspe_name.clone());

                // ------------------------------------------------------------------
                // Apply all changes to the original DAG.
                // ------------------------------------------------------------------

                // 1. Add the new cspe TempTable.
                dag.nodes.add_node_unchecked(TransformNode {
                    id: cspe_name.clone(),
                    query_text: cspe_sql,
                    materialize: MaterializeMode::TempTable,
                    depends_on: cspe_deps,
                    schema: None,
                });

                // 2. Update node A.
                if let Some(node) = dag.nodes.get_mut(id_a.clone()) {
                    node.query_text = new_sql_a;
                    node.depends_on = new_deps_a;
                    node.schema = None;
                }

                // 3. Update node B.
                if let Some(node) = dag.nodes.get_mut(id_b.clone()) {
                    node.query_text = new_sql_b;
                    node.depends_on = new_deps_b;
                    node.schema = None;
                }

                // Update the in-memory plan maps so subsequent pairs compare
                // against the already-rewritten plans.
                plans.insert(id_a.clone(), new_plan_a.clone());
                plans.insert(id_b.clone(), new_plan_b.clone());
                subtree_maps.insert(id_a.clone(), collect_subtrees(&new_plan_a));
                subtree_maps.insert(id_b.clone(), collect_subtrees(&new_plan_b));

                subplans_found += 1;
                debug!("CSPEPass: factored '{cspe_name}' from '{id_a}' and '{id_b}'");
            }
        }

        stats.insert("common_subplans_found".into(), subplans_found.to_string());
        debug!("CSPEPass: complete — {subplans_found} common subplan(s) factored out");
        Ok(stats)
    }
}
