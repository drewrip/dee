//! NodeFusion -- collapse the DAG's Table nodes into one fused query.
//!
//! A dee DAG executes as one relation per node: every View is created as a
//! view and every Table as its own `CREATE TABLE ... AS`. Work shared between
//! two Tables is therefore planned -- and usually scanned -- once per Table
//! that reaches it. That duplication is what HMP and OMP spend measured DAG
//! runs deciding how to materialize away.
//!
//! This pass attacks the same duplication structurally, and for free. It
//! builds one new node `f` that reproduces the whole upstream DAG as a `WITH`
//! chain and emits every Table's rows in a single `UNION ALL`, so the engine
//! sees one query and can share a CTE across every consumer inside it. Each
//! original Table node becomes a cheap projection out of `f`:
//!
//! ```sql
//! -- node f (TempTable, depends on nothing)
//! WITH n_v1 AS (<v1 sql>),
//!      n_t0 AS MATERIALIZED (<t0 sql, refs to v1 rewritten to n_v1>),
//!      n_v2 AS (<v2 sql, refs to t0 rewritten to n_t0>),
//!      n_t1 AS (<t1 sql, refs to v2 rewritten to n_v2>)
//! SELECT 0 AS kind, a AS "k0_a", b AS "k0_b", CAST(NULL AS VARCHAR) AS "k1_c" FROM n_t0
//! UNION ALL
//! SELECT 1 AS kind, CAST(NULL AS INTEGER) AS "k0_a", CAST(NULL AS VARCHAR) AS "k0_b", c AS "k1_c" FROM n_t1
//!
//! -- node t0                              -- node t1
//! SELECT "k0_a" AS a, "k0_b" AS b         SELECT "k1_c" AS c
//! FROM <f> WHERE kind = 0                 FROM <f> WHERE kind = 1
//! ```
//!
//! The branches of a `UNION ALL` must agree on their row shape and the Tables
//! do not, so the fused relation's schema is a `kind` discriminator followed
//! by every Table's columns concatenated in `kind` order, each branch filling
//! the columns that are not its own with NULL.
//!
//! Like Pushdown, this is a pure rewrite: it measures nothing, runs the DAG
//! zero times, and decides everything from the DAG in front of it.
//!
//! # What is fused, and what is not
//!
//! Every `Table` node is fused, *including* a Table that feeds another Table.
//! An intermediate Table gets a CTE as well as a `kind` branch, so it is
//! computed once inside `f`, read from that CTE by whatever is downstream of
//! it, and still delivered as its own relation. Its CTE is materialized by
//! default, because it is read by both its own branch and whatever is
//! downstream of it; a Table that feeds nothing is read once and is not. That is also what keeps the
//! graph acyclic: everything upstream of any Table ends up inside `f`, so `f`
//! itself depends on nothing but the warehouse's own source tables.
//!
//! `TempTable` nodes are not fused. They are the landing pads HMP and OMP
//! create, and their whole purpose is to be a materialization barrier the
//! search placed deliberately; folding one into a CTE would silently undo the
//! decision that put it there. A TempTable anywhere upstream of a Table makes
//! the DAG unfusable rather than partly fused, because a partial fusion would
//! have to leave that Table out and the point is to share work across all of
//! them.
//!
//! # What fusion changes
//!
//! The rows a Table holds, its columns and their types are all preserved
//! exactly -- that is the contract, and the tests check it against a real
//! engine. What is not preserved is the *order* those rows are stored in: a
//! Table whose query ends in `ORDER BY` gets that ordering applied inside its
//! CTE and then loses it passing through the `UNION ALL`. A SQL table is an
//! unordered relation and a consumer that cares must order for itself, so this
//! breaks nothing that was guaranteed -- but a DAG that was quietly relying on
//! `CREATE TABLE AS ... ORDER BY` laying rows down in order will notice.
//!
//! View nodes are left in the graph and are still created as views. A view is
//! a definition rather than a computation, so the duplication costs nothing,
//! and keeping them is what lets a sink view -- or any consumer that was not
//! fused -- go on binding to the name it was written against.

use std::collections::{HashMap, HashSet};

use async_trait::async_trait;
use log::debug;
use polyglot_sql::dialects::DialectType;

use crate::{
    connectors::Connector,
    dag::{Dag, MaterializeMode, TransformNode},
    executor::Executor,
    opt::{
        Optimization, OptimizerConfig, OptimizerError,
        common::{
            bare_table_name, dialect_for_db, fused_node_name, rewrite_node_refs,
            supports_materialized_hint,
        },
        explain::{render_card_grid, render_ranked_table},
        report::{NodeFusionDetail, PassDetail, PassOutcome},
        step::{OptimizationType, RegisterContext, StepContext, StepOutcome, StepPhase},
        store::Registration,
    },
};

/// The bare name of the fused node, before the schema prefix it inherits from
/// the nodes it stands in front of.
const FUSED_BASE: &str = "dee_fused";

/// The discriminator column every branch of the fused `UNION ALL` leads with.
const KIND_COLUMN: &str = "kind";

// ---------------------------------------------------------------------------
// The pass
// ---------------------------------------------------------------------------

pub struct NodeFusionPass {
    /// Emit inlined *View* CTEs `AS MATERIALIZED`. Never applies to an inlined
    /// Table's CTE, which is materialized on its own account.
    materialize_ctes: bool,
    /// Materialize an inlined *View* CTE that more than one Table reads.
    ///
    /// Naive on purpose: it counts how many Table nodes reach the View through
    /// the fused graph and materializes it above one, without pricing what the
    /// View costs or what sharing it would save. That is the whole point of
    /// having it -- a floor to measure a cost-based rule against.
    naive_materialize_ctes: bool,
    /// When set, the exact set of node IDs whose CTEs are materialized --
    /// overriding both `materialize_ctes` and the Table default, so this is
    /// also how an intermediate Table's CTE is made plain.
    materialize_override: Option<Vec<String>>,
    step_phase: StepPhase,
    explain_data: Option<ExplainData>,
}

struct ExplainData {
    outcome: String,
    fused_id: Option<String>,
    /// `(node id, role, CTE name, materialized)` in emission order.
    ctes: Vec<(String, &'static str, String, bool)>,
    /// `(kind, node id, column count)` in kind order.
    tables: Vec<(usize, String, usize)>,
    fused_columns: usize,
}

impl NodeFusionPass {
    pub fn new(
        materialize_ctes: bool,
        naive_materialize_ctes: bool,
        materialize_override: Option<Vec<String>>,
    ) -> Self {
        Self {
            materialize_ctes,
            naive_materialize_ctes,
            materialize_override,
            step_phase: StepPhase::Before,
            explain_data: None,
        }
    }

    pub fn from_config(config: &OptimizerConfig) -> Self {
        Self::new(
            config.nodefusion_materialize_ctes,
            config.nodefusion_naive_materialize_ctes,
            config.nodefusion_materialize_ctes_override.clone(),
        )
    }
}

/// Why a DAG cannot be fused.
///
/// Not an error: an unfusable DAG is a normal thing to hand this pass, and the
/// honest response is to leave it exactly as it was and say why. An
/// [`OptimizerError`] is reserved for a DAG that *is* fusable and whose
/// rewrite then failed, which is a bug rather than a shape.
#[derive(Debug, Clone)]
struct NotFusable(String);

impl std::fmt::Display for NotFusable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// ---------------------------------------------------------------------------
// Planning
// ---------------------------------------------------------------------------

/// One CTE in the fused query's `WITH` chain.
#[derive(Debug, Clone)]
struct CtePlan {
    node_id: String,
    cte_name: String,
    materialized: bool,
    /// A Table's CTE also becomes a branch of the `UNION ALL`; a View's does not.
    is_table: bool,
}

/// One Table node's branch of the fused `UNION ALL`, and the projection that
/// reads it back out.
#[derive(Debug, Clone)]
struct TablePlan {
    node_id: String,
    kind: usize,
    cte_name: String,
    /// `(column name in the node's own schema, its name in the fused
    /// relation)`, in the node's schema order.
    columns: Vec<(String, String)>,
}

#[derive(Debug, Clone)]
struct FusionPlan {
    fused_id: String,
    /// The `WITH` chain, in a topological order of the fused nodes.
    ctes: Vec<CtePlan>,
    /// The Tables, in `kind` order.
    tables: Vec<TablePlan>,
    /// Every column of the fused relation after `kind`, in `kind` order.
    fused_columns: Vec<String>,
}

impl NodeFusionPass {
    /// Decide what to fuse, or why this DAG cannot be.
    fn plan(&self, dag: &Dag) -> Result<FusionPlan, NotFusable> {
        let dialect = dialect_for_db(&dag.db);
        let topo = dag.nodes.topological_sort();
        if topo.len() < dag.nodes.num_nodes() {
            return Err(NotFusable(
                "the graph does not topologically sort, so there is no order to emit CTEs in"
                    .into(),
            ));
        }

        // Sorted by ID, not by topological position. A `kind` is a label
        // rather than an ordering, and the fused query's text has to be a
        // function of the DAG alone: `Graph::topological_sort` breaks ties
        // between independent nodes in `HashMap` iteration order, which varies
        // per process, so kinds taken from it would renumber run to run. dee's
        // DAGs are content-addressed, so a rewrite that is not byte-stable
        // mints a new version every time it runs over the same definition.
        let mut tables: Vec<String> = dag
            .nodes
            .nodes()
            .filter(|n| n.materialize == MaterializeMode::Table)
            .map(|n| n.id.clone())
            .collect();
        tables.sort();
        // One Table fused with itself is strictly worse than not fusing: the
        // fused node computes exactly what the node did, and the node then
        // copies it back out.
        if tables.len() < 2 {
            return Err(NotFusable(format!(
                "{} Table node(s); fusion needs at least two to share anything",
                tables.len()
            )));
        }

        // Everything the Tables read, transitively -- the Tables included,
        // since each one's own query becomes a CTE too.
        let closure = upstream_closure(dag, &tables);
        for id in &closure {
            let Some(node) = dag.nodes.get(id.clone()) else {
                return Err(NotFusable(format!("node '{id}' is not in the graph")));
            };
            if node.materialize == MaterializeMode::TempTable {
                return Err(NotFusable(format!(
                    "'{id}' is a TempTable upstream of a Table. A materialization search put it \
                     there on purpose, and folding it into a CTE would undo that decision"
                )));
            }
        }

        // A name for the fused node in the same catalog/schema as the nodes
        // that will read it, and not one already taken.
        let fused_id = {
            let sibling = &tables[0];
            let mut candidate = fused_node_name(sibling, FUSED_BASE);
            let mut n = 2;
            while dag.nodes.get(candidate.clone()).is_some() {
                candidate = fused_node_name(sibling, &format!("{FUSED_BASE}_{n}"));
                n += 1;
                if n > 64 {
                    return Err(NotFusable(
                        "could not find a free node ID for the fused node".into(),
                    ));
                }
            }
            candidate
        };

        // CTE names. Node IDs are qualified identifiers and a CTE name is a
        // single one, so the bare names are what is available -- and two
        // schemas may spell the same bare name, hence the dedupe.
        let mut taken: HashSet<String> = HashSet::new();
        let mut cte_names: HashMap<String, String> = HashMap::new();
        for id in &closure {
            let base = format!("n_{}", bare_table_name(id));
            let mut name = base.clone();
            let mut n = 2;
            while !taken.insert(name.clone()) {
                name = format!("{base}_{n}");
                n += 1;
            }
            cte_names.insert(id.clone(), name);
        }

        // The fused relation's columns: every Table's schema, concatenated in
        // kind order, each prefixed by its kind so two Tables spelling the
        // same column name stay distinct.
        let mut fused_columns: Vec<String> = Vec::new();
        let mut fused_taken: HashSet<String> = HashSet::new();
        let mut table_plans: Vec<TablePlan> = Vec::new();
        for (kind, id) in tables.iter().enumerate() {
            let node = dag.nodes.get(id.clone()).expect("checked above");
            let Some(schema) = node.schema.as_ref() else {
                return Err(NotFusable(format!(
                    "'{id}' has no resolved schema; call resolve_schemas before NodeFusion"
                )));
            };
            let fields = schema.flattened_fields();
            if fields.is_empty() {
                return Err(NotFusable(format!("'{id}' resolved to a schema with no columns")));
            }
            let mut columns = Vec::with_capacity(fields.len());
            for field in fields {
                let base = format!("k{kind}_{}", field.name());
                let mut name = base.clone();
                let mut n = 2;
                while !fused_taken.insert(name.clone()) {
                    name = format!("{base}_{n}");
                    n += 1;
                }
                fused_columns.push(name.clone());
                columns.push((field.name().clone(), name));
            }
            table_plans.push(TablePlan {
                node_id: id.clone(),
                kind,
                cte_name: cte_names[id].clone(),
                columns,
            });
        }

        // Which fused nodes another fused node reads. A Table in here is an
        // *intermediate* Table: something downstream of it was fused too, so
        // its CTE is read by more than its own `UNION ALL` branch.
        let mut feeds_another: HashSet<&String> = HashSet::new();
        for id in &closure {
            if let Some(node) = dag.nodes.get(id.clone()) {
                for dep in &node.depends_on {
                    if let Some(dep) = closure.get(dep) {
                        feeds_another.insert(dep);
                    }
                }
            }
        }

        // How many Tables reach each fused node. The naive materialization
        // rule reads off this: a View more than one Table reaches is a View
        // whose work the fused query would otherwise do more than once.
        let readers = table_readers(dag, &tables, &closure);

        // The `WITH` chain is emitted in a topological order, so a CTE is
        // always defined before the CTE that reads it -- and in a *stable*
        // one, so the same DAG always produces the same text.
        let table_set: HashSet<&String> = tables.iter().collect();
        let hint = supports_materialized_hint(dialect);
        let ctes: Vec<CtePlan> = stable_topological_order(dag, &closure)
            .iter()
            .map(|id| {
                let is_table = table_set.contains(id);
                let intermediate_table = is_table && feeds_another.contains(id);
                let shared_view = !is_table && readers.get(id).copied().unwrap_or(0) > 1;
                CtePlan {
                    node_id: id.clone(),
                    cte_name: cte_names[id].clone(),
                    materialized: hint
                        && self.wants_materialized(id, intermediate_table, shared_view),
                    is_table,
                }
            })
            .collect();

        Ok(FusionPlan {
            fused_id,
            ctes,
            tables: table_plans,
            fused_columns,
        })
    }

    /// Whether `node_id`'s CTE is emitted `AS MATERIALIZED`.
    ///
    /// With no override, an *intermediate* Table's CTE is materialized -- a
    /// Table that feeds another fused node -- and everything else follows the
    /// global default. That Table was authored as a materialization barrier,
    /// and a plain CTE read by both its own `UNION ALL` branch and whatever is
    /// downstream of it is free to unfold a copy into each, which is the
    /// duplication this pass exists to remove. A Table that feeds nothing is
    /// read exactly once, by its own branch, so materializing it would buy
    /// nothing and is left to the global default like any other CTE.
    ///
    /// `naive_materialize_ctes` adds the Views more than one Table reaches --
    /// `shared_view`. It is the narrower of the two global switches and is
    /// subsumed by `materialize_ctes`, which turns on every View regardless.
    ///
    /// An override names the exact set instead, so it is also the way to make
    /// an intermediate Table's CTE plain. A name matches either as the full
    /// node ID or as its bare table name, so `"wh"."main"."orders"` can be
    /// asked for as `orders`.
    fn wants_materialized(
        &self,
        node_id: &str,
        intermediate_table: bool,
        shared_view: bool,
    ) -> bool {
        match &self.materialize_override {
            Some(list) => list
                .iter()
                .any(|n| n == node_id || bare_table_name(n) == bare_table_name(node_id)),
            None => {
                intermediate_table
                    || self.materialize_ctes
                    || (self.naive_materialize_ctes && shared_view)
            }
        }
    }

    /// Rewrite `dag` in place. Returns what happened, for the report.
    pub fn rewrite(&mut self, dag: &mut Dag) -> Result<PassOutcome, OptimizerError> {
        let plan = match self.plan(dag) {
            Ok(plan) => plan,
            Err(why) => return Ok(self.not_fused(why)),
        };

        let dialect = dialect_for_db(&dag.db);
        let FusedSql {
            sql: fused_sql,
            verbatim_bodies,
        } = match build_fused_sql(dag, &plan, dialect) {
            Ok(built) => built,
            Err(why) => return Ok(self.not_fused(why)),
        };

        // Parse what we assembled before handing it to the executor, so a
        // malformed fused query is a failed optimization rather than a DAG
        // that only breaks at run time. This can only be asked when every body
        // was rewritten -- each of those round-tripped through the parser, so a
        // failure here is the frame. A body carried through verbatim may be
        // one the parser cannot read at all (that is why it was carried
        // through), and it would fail this check while saying nothing about
        // the frame. The engine is the authority on those either way.
        if verbatim_bodies == 0 {
            polyglot_sql::parse_one(&fused_sql, dialect).map_err(|e| {
                OptimizerError::Exec(format!(
                    "nodefusion built a fused query that does not parse ({e}); \
                     refusing to install it"
                ))
            })?;
        } else {
            debug!(
                "nodefusion: {verbatim_bodies} CTE bod(ies) are used as authored, so the \
                 assembled query was not parse-checked"
            );
        }

        // The fused node reads nothing but the warehouse's own source tables:
        // everything upstream of a Table is inside it now.
        dag.nodes
            .add_node(TransformNode {
                id: plan.fused_id.clone(),
                query_text: fused_sql,
                materialize: MaterializeMode::TempTable,
                depends_on: HashSet::new(),
                schema: None,
            })
            .map_err(|e| OptimizerError::Exec(format!("nodefusion: adding the fused node: {e}")))?;

        for table in &plan.tables {
            let projection = table
                .columns
                .iter()
                .map(|(original, fused)| format!("\"{fused}\" AS \"{original}\""))
                .collect::<Vec<_>>()
                .join(", ");
            let node = dag
                .nodes
                .get_mut(table.node_id.clone())
                .expect("planned from this graph");
            node.query_text = format!(
                "SELECT {projection} FROM {} WHERE {KIND_COLUMN} = {}",
                plan.fused_id, table.kind
            );
            // The node's own schema is unchanged -- the projection puts every
            // column back under its original name, in its original order -- so
            // it is deliberately left alone for whatever reads it next.
            node.depends_on = HashSet::from([plan.fused_id.clone()]);
        }

        dag.nodes.check().map_err(|e| {
            OptimizerError::Exec(format!("nodefusion produced an inconsistent graph: {e}"))
        })?;

        let views_inlined = plan.ctes.iter().filter(|c| !c.is_table).count();
        let materialized = plan.ctes.iter().filter(|c| c.materialized).count();
        let outcome = format!(
            "fused {} Table node(s) and {views_inlined} inlined View(s) into '{}'",
            plan.tables.len(),
            plan.fused_id
        );
        debug!("nodefusion: {outcome}");

        self.explain_data = Some(ExplainData {
            outcome: outcome.clone(),
            fused_id: Some(plan.fused_id.clone()),
            ctes: plan
                .ctes
                .iter()
                .map(|c| {
                    (
                        c.node_id.clone(),
                        if c.is_table { "Table" } else { "View" },
                        c.cte_name.clone(),
                        c.materialized,
                    )
                })
                .collect(),
            tables: plan
                .tables
                .iter()
                .map(|t| (t.kind, t.node_id.clone(), t.columns.len()))
                .collect(),
            fused_columns: plan.fused_columns.len(),
        });

        let mut record = PassOutcome::empty().with_detail(PassDetail::NodeFusion(
            NodeFusionDetail {
                fused_node: Some(plan.fused_id.clone()),
                tables_fused: plan.tables.len(),
                views_inlined,
                materialized_ctes: materialized,
                fused_columns: plan.fused_columns.len(),
                outcome,
            },
        ));
        // One change: the DAG became one fused node. The Tables rewritten to
        // read it are that change's consequence, not separate decisions.
        record.changes_applied = 1;
        record.candidates_considered = 1;
        record.working_set_size = plan.ctes.len() as u32;
        Ok(record)
    }

    /// Record that this DAG was left alone, and why. The DAG itself is
    /// untouched -- everything up to here only read it.
    fn not_fused(&mut self, why: NotFusable) -> PassOutcome {
        debug!("nodefusion: not fusing this DAG -- {why}");
        let outcome = format!("not fused: {why}");
        self.explain_data = Some(ExplainData {
            outcome: outcome.clone(),
            fused_id: None,
            ctes: Vec::new(),
            tables: Vec::new(),
            fused_columns: 0,
        });
        PassOutcome::empty().with_detail(PassDetail::NodeFusion(NodeFusionDetail {
            fused_node: None,
            tables_fused: 0,
            views_inlined: 0,
            materialized_ctes: 0,
            fused_columns: 0,
            outcome,
        }))
    }

    fn explain_html(&self) -> String {
        let Some(data) = &self.explain_data else {
            return r#"<div class="panel"><p class="subtle">NodeFusionPass did not run.</p></div>"#
                .to_string();
        };

        let cards = render_card_grid(&[
            ("Outcome", data.outcome.clone()),
            (
                "Fused node",
                data.fused_id.clone().unwrap_or_else(|| "-".into()),
            ),
            ("CTEs", data.ctes.len().to_string()),
            (
                "Materialized CTEs",
                data.ctes.iter().filter(|c| c.3).count().to_string(),
            ),
            ("Fused columns", data.fused_columns.to_string()),
        ]);

        let cte_rows: Vec<Vec<String>> = data
            .ctes
            .iter()
            .enumerate()
            .map(|(i, (node_id, role, cte, materialized))| {
                vec![
                    (i + 1).to_string(),
                    node_id.clone(),
                    role.to_string(),
                    cte.clone(),
                    if *materialized { "MATERIALIZED" } else { "plain" }.to_string(),
                ]
            })
            .collect();
        let cte_table = render_ranked_table(&["#", "Node", "Role", "CTE", "Hint"], &cte_rows);

        let kind_rows: Vec<Vec<String>> = data
            .tables
            .iter()
            .map(|(kind, node_id, cols)| {
                vec![kind.to_string(), node_id.clone(), cols.to_string()]
            })
            .collect();
        let kind_table = render_ranked_table(&["kind", "Table", "Columns"], &kind_rows);

        format!(
            r#"<div class="section-stack">
        {cards}
        <div class="panel">
          <h2>The WITH chain</h2>
          <div class="subtle">Emitted in the graph's own topological order, so every CTE is defined before the CTE that reads it. A Table's CTE is materialized by default -- it was authored as a materialization barrier, and a plain CTE read by several downstream CTEs may be unfolded into each of them, which is the duplication this pass removes.</div>
          {cte_table}
        </div>
        <div class="panel">
          <h2>The UNION ALL</h2>
          <div class="subtle">One branch per Table, discriminated by <code>kind</code>. The fused relation's columns are these schemas concatenated in kind order; each branch fills the columns that are not its own with NULL, and each Table reads its own back out under their original names.</div>
          {kind_table}
        </div>
      </div>"#
        )
    }
}

/// Every node `tables` read, transitively, plus `tables` themselves.
fn upstream_closure(dag: &Dag, tables: &[String]) -> HashSet<String> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut stack: Vec<String> = tables.to_vec();
    while let Some(id) = stack.pop() {
        if !seen.insert(id.clone()) {
            continue;
        }
        if let Some(node) = dag.nodes.get(id) {
            // A dependency that is not a node is one of the warehouse's own
            // source tables, which the fused query reads by name like any
            // other query does.
            for dep in &node.depends_on {
                if dag.nodes.get(dep.clone()).is_some() {
                    stack.push(dep.clone());
                }
            }
        }
    }
    seen
}

/// A topological order of `closure` that depends only on the graph, never on
/// how a `HashMap` happened to iterate.
///
/// [`Graph::topological_sort`] peels off source nodes in hash order, so two
/// independent nodes come back in either order and the fused query's CTE chain
/// would be spelled differently run to run. Taking the lexicographically
/// smallest ready node at each step instead makes the whole rewrite a function
/// of the DAG, which is what dee's content-addressed versioning needs: a
/// version is minted when the definition changes, not when the optimizer is
/// re-run over an unchanged one.
///
/// A node whose dependencies cannot all be emitted -- which a cycle would
/// cause, and [`NodeFusionPass::plan`] rejects before this is reached -- is
/// left out rather than emitted out of order.
fn stable_topological_order(dag: &Dag, closure: &HashSet<String>) -> Vec<String> {
    let mut remaining: Vec<String> = closure.iter().cloned().collect();
    remaining.sort();
    let mut emitted: HashSet<String> = HashSet::new();
    let mut order: Vec<String> = Vec::with_capacity(remaining.len());

    while !remaining.is_empty() {
        let ready: Option<usize> = remaining.iter().position(|id| {
            dag.nodes.get(id.clone()).is_some_and(|node| {
                node.depends_on
                    .iter()
                    .all(|dep| !closure.contains(dep) || emitted.contains(dep))
            })
        });
        match ready {
            Some(i) => {
                let id = remaining.remove(i);
                emitted.insert(id.clone());
                order.push(id);
            }
            None => break,
        }
    }
    order
}

/// How many of `tables` reach each node of `closure`, following dependencies
/// transitively.
///
/// "Reach" is deliberately plain: the walk does not stop at an intermediate
/// Table, so a View above one is counted for every Table downstream of it even
/// though that Table's own CTE, being materialized, already shares the View
/// once. Modelling that is what a cost-based rule would do; this is the naive
/// floor it gets measured against.
///
/// A Table reaches itself, which is why the naive rule is only ever consulted
/// for Views -- every Table would otherwise have at least one reader and the
/// rule would say nothing.
fn table_readers(
    dag: &Dag,
    tables: &[String],
    closure: &HashSet<String>,
) -> HashMap<String, usize> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for table in tables {
        let mut seen: HashSet<&String> = HashSet::new();
        let mut stack: Vec<&String> = vec![table];
        while let Some(id) = stack.pop() {
            if !seen.insert(id) {
                continue;
            }
            *counts.entry(id.clone()).or_insert(0) += 1;
            if let Some(node) = dag.nodes.get(id.clone()) {
                for dep in &node.depends_on {
                    if let Some(dep) = closure.get(dep) {
                        stack.push(dep);
                    }
                }
            }
        }
    }
    counts
}

/// Assemble the fused node's query text.
///
/// Each CTE body is the node's own query with its references to other fused
/// nodes rewritten to their CTE names. That rewrite happens on the parsed AST
/// ([`rewrite_node_refs`]), and a body that cannot be rewritten there fails the
/// pass rather than falling back to text substitution -- the same rule
/// [`make_temp`](crate::opt::common::make_temp) follows, and for the same
/// reason: a substring match inside a string literal would hand the engine a
/// query that means something else.
///
/// The frame around those bodies -- the `WITH` chain, the branches, their
/// projections -- is assembled as text. Nothing in it comes from user SQL: the
/// identifiers are generated here and the literals are the integers this pass
/// chose, so there is no substring to match wrongly. The result is parsed
/// before it is installed.
fn build_fused_sql(
    dag: &Dag,
    plan: &FusionPlan,
    dialect: DialectType,
) -> Result<FusedSql, NotFusable> {
    // Only fused nodes are rewritten; a reference to anything else is a
    // reference to a real relation and must survive untouched.
    let mapping: HashMap<String, String> = plan
        .ctes
        .iter()
        .map(|c| (c.node_id.clone(), c.cte_name.clone()))
        .collect();

    // Bodies used exactly as authored, never round-tripped through the parser.
    // Counted because the assembled query can only be parse-checked when there
    // are none: a body the parser could not read makes the whole query
    // unreadable to it too, and that says nothing about the frame.
    let mut verbatim = 0usize;
    let mut cte_sql: Vec<String> = Vec::with_capacity(plan.ctes.len());
    for cte in &plan.ctes {
        let node = dag
            .nodes
            .get(cte.node_id.clone())
            .ok_or_else(|| NotFusable(format!("'{}' vanished from the graph", cte.node_id)))?;
        // A body that names no fused node needs no rewriting, and is used
        // exactly as authored. That is not only cheaper than a parse and
        // regenerate that would change nothing -- it is what lets a DAG fuse
        // when a leaf staging view uses syntax the parser does not cover. Most
        // nodes with no fused dependencies are exactly those.
        let names_a_fused_node = node.depends_on.iter().any(|dep| mapping.contains_key(dep));
        if !names_a_fused_node {
            verbatim += 1;
        }
        let body = if names_a_fused_node {
            rewrite_node_refs(&node.query_text, &mapping, dialect).ok_or_else(|| {
                // Refusing to fall back to text substitution, which would
                // silently change what the query means. Not an error: a node
                // the parser cannot read is a fact about this DAG, the same
                // kind of fact as a TempTable in the closure, and the honest
                // response is to leave the DAG alone and name the node.
                NotFusable(format!(
                    "'{}' reads a node that would become a CTE, but its query cannot be \
                     rewritten at the AST level -- the parser does not cover it, and \
                     substituting the reference textually could change what the query means",
                    cte.node_id
                ))
            })?
        } else {
            node.query_text.clone()
        };
        let hint = if cte.materialized { " MATERIALIZED" } else { "" };
        let name = &cte.cte_name;
        cte_sql.push(format!("{name} AS{hint} ({body})"));
    }

    // One branch per Table: its own columns read from its CTE, every other
    // Table's columns filled with a NULL of the right type.
    let mut branches: Vec<String> = Vec::with_capacity(plan.tables.len());
    for table in &plan.tables {
        let own: HashMap<&str, &str> = table
            .columns
            .iter()
            .map(|(original, fused)| (fused.as_str(), original.as_str()))
            .collect();
        let mut projection: Vec<String> = vec![format!("{} AS {KIND_COLUMN}", table.kind)];
        for fused in &plan.fused_columns {
            match own.get(fused.as_str()) {
                Some(original) => projection.push(format!("\"{original}\" AS \"{fused}\"")),
                // An untyped NULL, deliberately. Every fused column belongs to
                // exactly one Table and is selected from that Table's CTE in
                // exactly one branch, so set-operation type resolution gives
                // the column that branch's type -- which is the type the
                // unfused node produced, exactly.
                //
                // Casting instead is what this used to do, and it was worse:
                // the only type available to cast to is the one read back off
                // the node's *Arrow* schema, and that round trip is lossy.
                // DuckDB's HUGEINT arrives as Decimal128(38, 0) and would have
                // gone back as DECIMAL(38,0), quietly changing the delivered
                // table's column type.
                None => projection.push(format!("NULL AS \"{fused}\"")),
            }
        }
        branches.push(format!(
            "SELECT {} FROM {}",
            projection.join(", "),
            table.cte_name
        ));
    }

    Ok(FusedSql {
        sql: format!(
            "WITH {}\n{}",
            cte_sql.join(",\n     "),
            branches.join("\nUNION ALL\n")
        ),
        verbatim_bodies: verbatim,
    })
}

/// The assembled fused query, and how many of its CTE bodies were used exactly
/// as authored rather than rewritten.
struct FusedSql {
    sql: String,
    verbatim_bodies: usize,
}

// ---------------------------------------------------------------------------
// The Optimization interface
// ---------------------------------------------------------------------------

#[async_trait]
impl<C, E> Optimization<C, E> for NodeFusionPass
where
    C: Connector + Send + 'static + Sync,
    E: Executor<C> + Send + Sync,
{
    fn name(&self) -> &'static str {
        "nodefusion"
    }

    /// NodeFusion decides everything from the DAG in front of it. There is no
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

    /// Nothing to set up -- it keeps no state between steps.
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
        let record = self.rewrite(ctx.dag)?;
        Ok(StepOutcome::Rewrote {
            record: Box::new(record),
        })
    }

    fn explain(&self) -> Option<(String, String)> {
        Some(("NodeFusionPass".to_string(), self.explain_html()))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectors::duckdb::{DuckDBConfig, DuckDBConnection};
    use crate::dag::SourceNode;
    use crate::executor::{Executor, SimpleEngine};
    use crate::graph::Graph;
    use std::sync::Arc;

    // ------------------------------------------------------------------
    // Helpers -- a real in-memory DuckDB and the real SimpleEngine, so these
    // tests exercise the same path a run does. A fused query that only looks
    // right is not evidence of anything; one the engine executes to the same
    // rows is.
    // ------------------------------------------------------------------

    async fn in_memory_conn() -> Arc<DuckDBConnection> {
        let config = DuckDBConfig::new_from_path(":memory:".to_string());
        DuckDBConnection::new(config).await.unwrap()
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

    fn make_dag(nodes: Vec<TransformNode>) -> Dag {
        let mut graph = Graph::new(HashMap::new());
        for n in nodes {
            graph.add_node(n).unwrap();
        }
        Dag {
            db: "DuckDB".to_string(),
            nodes: graph,
            sources: vec![SourceNode {
                name: "orders".to_string(),
                schema: Arc::new(duckdb::arrow::datatypes::Schema::empty()),
            }],
            max_parallelism: None,
        }
    }

    async fn setup_orders(conn: &DuckDBConnection) {
        conn.execute(
            "CREATE TABLE orders AS SELECT \
                range AS order_id, \
                CASE WHEN range % 2 = 0 THEN 'US' ELSE 'EU' END AS region, \
                range * 1.5 AS amount \
             FROM range(20)"
                .to_string(),
        )
        .await
        .unwrap();
    }

    /// The DAG the happy-path tests use:
    ///
    ///   orders (a real table)
    ///     └─ stg (View) ──┬──► by_region (Table)
    ///                     └──► totals    (Table)
    ///
    /// `stg` is read by both Tables, which is exactly the duplication fusion
    /// is meant to collapse into one shared CTE.
    fn two_table_dag() -> Dag {
        make_dag(vec![
            node(
                "stg",
                "SELECT order_id, region, amount FROM orders WHERE amount > 0",
                MaterializeMode::View,
                &[],
            ),
            node(
                "by_region",
                "SELECT region, count(*) AS n, sum(amount) AS total FROM stg GROUP BY region",
                MaterializeMode::Table,
                &["stg"],
            ),
            node(
                "totals",
                "SELECT count(*) AS orders, sum(amount) AS amount_sum FROM stg",
                MaterializeMode::Table,
                &["stg"],
            ),
        ])
    }

    /// An order-independent fingerprint of a relation: its row count and the
    /// sum of the hashes of its rows, so a rewrite that reorders rows is not
    /// mistaken for one that changes them.
    fn fingerprint(conn: &DuckDBConnection, relation: &str) -> (i64, i64) {
        let c = conn.pool.get().unwrap();
        let n: i64 = c
            .query_row(&format!("SELECT count(*) FROM {relation}"), [], |r| r.get(0))
            .unwrap();
        let h: i64 = c
            .query_row(
                &format!(
                    "SELECT coalesce((sum(hash(t)::HUGEINT) % 1000000007)::BIGINT, 0) \
                     FROM {relation} AS t"
                ),
                [],
                |r| r.get(0),
            )
            .unwrap();
        (n, h)
    }

    /// The relation's columns as `name type`, in order.
    ///
    /// The types matter as much as the rows: a fused branch that filled a
    /// column with a *cast* NULL rather than an untyped one used to turn
    /// DuckDB's HUGEINT into DECIMAL(38,0), because the only type available to
    /// cast to is the one read back off the node's Arrow schema and that round
    /// trip is lossy. The rows were identical; the delivered table was not.
    fn columns(conn: &DuckDBConnection, relation: &str) -> Vec<String> {
        let c = conn.pool.get().unwrap();
        let mut stmt = c
            .prepare(&format!(
                "SELECT column_name, column_type FROM (DESCRIBE SELECT * FROM {relation})"
            ))
            .unwrap();
        let rows = stmt
            .query_map([], |r| {
                Ok(format!("{} {}", r.get::<_, String>(0)?, r.get::<_, String>(1)?))
            })
            .unwrap();
        rows.map(|r| r.unwrap()).collect()
    }

    /// Run `dag`, fingerprint each of `relations`, then drop everything it made.
    async fn run_and_fingerprint(
        conn: &Arc<DuckDBConnection>,
        dag: &Dag,
        relations: &[&str],
    ) -> Vec<(Vec<String>, (i64, i64))> {
        let engine = SimpleEngine::new(Arc::clone(conn)).unwrap();
        engine.run(dag).await.expect("the DAG should run");
        let out = relations
            .iter()
            .map(|r| (columns(conn, r), fingerprint(conn, r)))
            .collect();
        engine.cleanup(dag).await.unwrap();
        out
    }

    async fn resolved(conn: &Arc<DuckDBConnection>, dag: &mut Dag) {
        let engine = SimpleEngine::new(Arc::clone(conn)).unwrap();
        engine.resolve_schemas(dag).await.expect("schemas resolve");
    }

    // ------------------------------------------------------------------
    // End to end: the fused DAG produces the same relations
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn test_fused_dag_produces_the_same_tables() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;

        let baseline_dag = two_table_dag();
        let baseline = run_and_fingerprint(&conn, &baseline_dag, &["by_region", "totals"]).await;

        let mut dag = two_table_dag();
        resolved(&conn, &mut dag).await;
        let mut pass = NodeFusionPass::new(false, false, None);
        pass.rewrite(&mut dag).expect("the DAG should fuse");

        let fused = run_and_fingerprint(&conn, &dag, &["by_region", "totals"]).await;

        assert_eq!(
            baseline, fused,
            "fusion must not change a Table's columns or its rows"
        );
    }

    #[tokio::test]
    async fn test_the_rewritten_dag_has_the_shape_fusion_promises() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;
        let mut dag = two_table_dag();
        resolved(&conn, &mut dag).await;

        NodeFusionPass::new(false, false, None)
            .rewrite(&mut dag)
            .expect("the DAG should fuse");

        let fused = dag.nodes.get("dee_fused".to_string()).expect("fused node");
        assert_eq!(fused.materialize, MaterializeMode::TempTable);
        assert!(
            fused.depends_on.is_empty(),
            "everything upstream of a Table is inside the fused node, so it depends on nothing"
        );
        assert!(fused.query_text.contains("UNION ALL"));

        for table in ["by_region", "totals"] {
            let node = dag.nodes.get(table.to_string()).unwrap();
            assert_eq!(node.materialize, MaterializeMode::Table);
            assert_eq!(
                node.depends_on,
                HashSet::from(["dee_fused".to_string()]),
                "a fused Table reads the fused node and nothing else"
            );
            assert!(node.query_text.contains("WHERE kind = "));
        }

        // The View stays in the graph, and stays a View.
        let stg = dag.nodes.get("stg".to_string()).expect("the View is kept");
        assert_eq!(stg.materialize, MaterializeMode::View);
    }

    #[tokio::test]
    async fn test_fusion_preserves_column_types() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;

        // `sum(order_id)` is a HUGEINT, which reaches an Arrow schema as
        // Decimal128(38, 0) and cannot be told apart from a real
        // DECIMAL(38,0) on the way back. The branches fill a foreign column
        // with an untyped NULL precisely so that round trip never happens.
        let dag_of = || {
            make_dag(vec![
                node(
                    "stg",
                    "SELECT order_id, region, amount FROM orders",
                    MaterializeMode::View,
                    &[],
                ),
                node(
                    "wide",
                    "SELECT sum(order_id) AS id_sum, region, \
                     count(*) > 0 AS any_rows, max(amount) AS biggest \
                     FROM stg GROUP BY region",
                    MaterializeMode::Table,
                    &["stg"],
                ),
                node(
                    "narrow",
                    "SELECT count(*) AS n FROM stg",
                    MaterializeMode::Table,
                    &["stg"],
                ),
            ])
        };

        let baseline = run_and_fingerprint(&conn, &dag_of(), &["wide", "narrow"]).await;

        let mut dag = dag_of();
        resolved(&conn, &mut dag).await;
        NodeFusionPass::new(false, false, None).rewrite(&mut dag).unwrap();
        let fused = run_and_fingerprint(&conn, &dag, &["wide", "narrow"]).await;

        assert_eq!(
            baseline, fused,
            "a fused Table must keep its column types, not only its rows"
        );
        assert!(
            baseline[0].0.iter().any(|c| c.contains("HUGEINT")),
            "the fixture must actually exercise HUGEINT -- got {:?}",
            baseline[0].0
        );
    }

    // ------------------------------------------------------------------
    // A Table above a Table
    // ------------------------------------------------------------------

    /// DAG layout:
    ///
    ///   orders ─► stg (View) ─► base (Table) ─► enriched (View) ─► rollup (Table)
    ///
    /// `base` feeds `rollup` through a View, so it is inlined as a CTE *and*
    /// still delivered as its own relation.
    fn layered_dag() -> Dag {
        make_dag(vec![
            node(
                "stg",
                "SELECT order_id, region, amount FROM orders",
                MaterializeMode::View,
                &[],
            ),
            node(
                "base",
                "SELECT region, sum(amount) AS total FROM stg GROUP BY region",
                MaterializeMode::Table,
                &["stg"],
            ),
            node(
                "enriched",
                "SELECT region, total, total * 2 AS doubled FROM base",
                MaterializeMode::View,
                &["base"],
            ),
            node(
                "rollup",
                "SELECT sum(doubled) AS all_doubled FROM enriched",
                MaterializeMode::Table,
                &["enriched"],
            ),
        ])
    }

    #[tokio::test]
    async fn test_a_table_feeding_a_table_is_inlined_and_still_delivered() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;

        let baseline_dag = layered_dag();
        let baseline = run_and_fingerprint(&conn, &baseline_dag, &["base", "rollup"]).await;

        let mut dag = layered_dag();
        resolved(&conn, &mut dag).await;
        let mut pass = NodeFusionPass::new(false, false, None);
        pass.rewrite(&mut dag).expect("the DAG should fuse");

        let fused_sql = dag
            .nodes
            .get("dee_fused".to_string())
            .unwrap()
            .query_text
            .clone();
        assert!(
            fused_sql.contains("n_base AS MATERIALIZED ("),
            "an inlined Table's CTE is materialized by default -- got:\n{fused_sql}"
        );
        assert!(
            fused_sql.contains("n_stg AS ("),
            "an inlined View's CTE is plain by default -- got:\n{fused_sql}"
        );
        assert!(
            dag.nodes
                .get("base".to_string())
                .unwrap()
                .query_text
                .contains("WHERE kind = "),
            "the intermediate Table still gets a kind of its own"
        );

        let fused = run_and_fingerprint(&conn, &dag, &["base", "rollup"]).await;
        assert_eq!(baseline, fused);
    }

    #[tokio::test]
    async fn test_the_rewrite_is_a_function_of_the_dag_alone() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;

        // The same logical DAG, built by inserting its nodes in two different
        // orders. `Graph` is a `HashMap`, so that is enough to change how it
        // iterates -- and the fused query must not notice. dee's DAGs are
        // content-addressed: a rewrite that is not byte-stable mints a new
        // version every time it is re-run over an unchanged definition.
        let build = |reverse: bool| {
            let mut nodes = vec![
                node(
                    "stg",
                    "SELECT order_id, region, amount FROM orders",
                    MaterializeMode::View,
                    &[],
                ),
                node(
                    "by_region",
                    "SELECT region, count(*) AS n FROM stg GROUP BY region",
                    MaterializeMode::Table,
                    &["stg"],
                ),
                node(
                    "totals",
                    "SELECT count(*) AS n FROM stg",
                    MaterializeMode::Table,
                    &["stg"],
                ),
                node(
                    "biggest",
                    "SELECT max(amount) AS m FROM stg",
                    MaterializeMode::Table,
                    &["stg"],
                ),
            ];
            if reverse {
                nodes.reverse();
            }
            // Inserted in the given order and only checked afterwards:
            // `add_node` refuses a node whose dependency is not in the graph
            // yet, which is the very ordering this test needs to vary.
            let mut graph = Graph::new(HashMap::new());
            for n in nodes {
                graph.add_node_unchecked(n);
            }
            graph.check().unwrap();
            Dag {
                db: "DuckDB".to_string(),
                nodes: graph,
                sources: vec![SourceNode {
                    name: "orders".to_string(),
                    schema: Arc::new(duckdb::arrow::datatypes::Schema::empty()),
                }],
                max_parallelism: None,
            }
        };

        async fn fuse(conn: &Arc<DuckDBConnection>, mut dag: Dag) -> Vec<(String, String)> {
            resolved(conn, &mut dag).await;
            NodeFusionPass::new(false, false, None)
                .rewrite(&mut dag)
                .unwrap();
            let mut texts: Vec<(String, String)> = dag
                .nodes
                .nodes()
                .map(|n| (n.id.clone(), n.query_text.clone()))
                .collect();
            texts.sort();
            texts
        }

        assert_eq!(
            fuse(&conn, build(false)).await,
            fuse(&conn, build(true)).await,
            "the fused query and every rewritten Table must be byte-identical"
        );
    }

    #[tokio::test]
    async fn test_kinds_are_assigned_in_sorted_node_id_order() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;
        let mut dag = layered_dag();
        resolved(&conn, &mut dag).await;

        let plan = NodeFusionPass::new(false, false, None).plan(&dag).unwrap();
        let by_kind: Vec<&str> = plan.tables.iter().map(|t| t.node_id.as_str()).collect();
        assert_eq!(
            by_kind,
            vec!["base", "rollup"],
            "a kind is a label, not an ordering, so it follows the one total order \
             the DAG has: its node IDs"
        );
        assert_eq!(plan.tables[0].kind, 0);
        assert_eq!(plan.tables[1].kind, 1);
    }

    #[tokio::test]
    async fn test_ctes_are_emitted_in_a_topological_order() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;
        let mut dag = layered_dag();
        resolved(&conn, &mut dag).await;

        let plan = NodeFusionPass::new(false, false, None).plan(&dag).unwrap();
        let order: Vec<&str> = plan.ctes.iter().map(|c| c.node_id.as_str()).collect();
        assert_eq!(order, vec!["stg", "base", "enriched", "rollup"]);
    }

    // ------------------------------------------------------------------
    // The materialization knobs
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn test_the_global_default_materializes_views_and_not_only_tables() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;
        let mut dag = layered_dag();
        resolved(&conn, &mut dag).await;

        let plan = NodeFusionPass::new(true, false, None).plan(&dag).unwrap();
        let materialized: HashSet<&str> = plan
            .ctes
            .iter()
            .filter(|c| c.materialized)
            .map(|c| c.node_id.as_str())
            .collect();
        assert_eq!(
            materialized,
            HashSet::from(["stg", "base", "enriched", "rollup"]),
            "the global default turns on everything the intermediate-Table rule did not"
        );
    }

    #[tokio::test]
    async fn test_only_a_table_that_feeds_another_is_materialized_by_default() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;

        // `base` feeds `rollup`, so its CTE is read by its own UNION ALL
        // branch and by `enriched` -- two readers, and a plain CTE may be
        // unfolded into each. `rollup` feeds nothing and is read once.
        let mut layered = layered_dag();
        resolved(&conn, &mut layered).await;
        let plan = NodeFusionPass::new(false, false, None).plan(&layered).unwrap();
        let materialized: Vec<&str> = plan
            .ctes
            .iter()
            .filter(|c| c.materialized)
            .map(|c| c.node_id.as_str())
            .collect();
        assert_eq!(materialized, vec!["base"]);

        // And where no Table feeds another, nothing is materialized at all.
        let mut flat = two_table_dag();
        resolved(&conn, &mut flat).await;
        let plan = NodeFusionPass::new(false, false, None).plan(&flat).unwrap();
        assert!(
            plan.ctes.iter().all(|c| !c.materialized),
            "a Table read only by its own branch gains nothing from MATERIALIZED"
        );
    }

    #[tokio::test]
    async fn test_an_override_is_the_exact_set_and_can_demote_a_table() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;
        let mut dag = layered_dag();
        resolved(&conn, &mut dag).await;

        // `base` is a Table and would be materialized by default; naming only
        // `stg` takes it back off.
        let plan = NodeFusionPass::new(true, false, Some(vec!["stg".to_string()]))
            .plan(&dag)
            .unwrap();
        let materialized: Vec<&str> = plan
            .ctes
            .iter()
            .filter(|c| c.materialized)
            .map(|c| c.node_id.as_str())
            .collect();
        assert_eq!(materialized, vec!["stg"]);
    }

    /// DAG layout, built so the naive rule has both answers to give:
    ///
    ///   orders ─► shared (View) ─┬─► mid (View) ─► tbl_a (Table)
    ///                            └────────────────► tbl_b (Table)
    ///          └─► lonely (View) ──────────────────► tbl_b (Table)
    ///
    /// `shared` is reached by both Tables; `mid` by only `tbl_a` and `lonely`
    /// by only `tbl_b`.
    ///
    fn shared_view_dag() -> Dag {
        make_dag(vec![
            node(
                "shared",
                "SELECT order_id, region, amount FROM orders",
                MaterializeMode::View,
                &[],
            ),
            node(
                "mid",
                "SELECT region, amount * 2 AS doubled FROM shared",
                MaterializeMode::View,
                &["shared"],
            ),
            node(
                "lonely",
                "SELECT order_id, amount FROM orders WHERE amount > 1",
                MaterializeMode::View,
                &[],
            ),
            node(
                "tbl_a",
                "SELECT region, sum(doubled) AS d FROM mid GROUP BY region",
                MaterializeMode::Table,
                &["mid"],
            ),
            node(
                "tbl_b",
                "SELECT count(*) AS n, (SELECT count(*) FROM lonely) AS m FROM shared",
                MaterializeMode::Table,
                &["shared", "lonely"],
            ),
        ])
    }

    #[tokio::test]
    async fn test_the_naive_rule_materializes_only_views_more_than_one_table_reads() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;
        let mut dag = shared_view_dag();
        resolved(&conn, &mut dag).await;

        let plan = NodeFusionPass::new(false, true, None).plan(&dag).unwrap();
        let materialized: HashSet<&str> = plan
            .ctes
            .iter()
            .filter(|c| c.materialized)
            .map(|c| c.node_id.as_str())
            .collect();
        assert_eq!(
            materialized,
            HashSet::from(["shared"]),
            "only the View both Tables reach; `mid` and `lonely` have one reader each"
        );

        // Off, it materializes nothing here: no Table feeds another.
        let plan = NodeFusionPass::new(false, false, None).plan(&dag).unwrap();
        assert!(plan.ctes.iter().all(|c| !c.materialized));
    }

    #[tokio::test]
    async fn test_the_naive_rule_counts_readers_through_intermediate_views() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;
        // In `layered_dag`, `stg` is read by `base` and -- through `base` and
        // `enriched` -- by `rollup`. Reachability is transitive and does not
        // stop at the intermediate Table, which is the naive part.
        let mut dag = layered_dag();
        resolved(&conn, &mut dag).await;

        let plan = NodeFusionPass::new(false, true, None).plan(&dag).unwrap();
        let materialized: HashSet<&str> = plan
            .ctes
            .iter()
            .filter(|c| c.materialized)
            .map(|c| c.node_id.as_str())
            .collect();
        assert_eq!(
            materialized,
            HashSet::from(["stg", "base"]),
            "`stg` by the naive rule, `base` because it feeds another Table; \
             `enriched` is reached by `rollup` alone"
        );
    }

    #[tokio::test]
    async fn test_the_naive_rule_does_not_change_what_the_dag_produces() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;

        let baseline = run_and_fingerprint(&conn, &shared_view_dag(), &["tbl_a", "tbl_b"]).await;

        let mut dag = shared_view_dag();
        resolved(&conn, &mut dag).await;
        NodeFusionPass::new(false, true, None).rewrite(&mut dag).unwrap();
        assert!(
            dag.nodes
                .get("dee_fused".to_string())
                .unwrap()
                .query_text
                .contains("n_shared AS MATERIALIZED ("),
            "the shared View should carry the hint"
        );

        let fused = run_and_fingerprint(&conn, &dag, &["tbl_a", "tbl_b"]).await;
        assert_eq!(baseline, fused, "a materialization hint changes plans, not rows");
    }

    #[test]
    fn test_the_global_switch_subsumes_the_naive_one() {
        let all = NodeFusionPass::new(true, false, None);
        let naive = NodeFusionPass::new(false, true, None);
        // A View no second Table reaches: on under the global switch, off
        // under the naive rule.
        assert!(all.wants_materialized("v", false, false));
        assert!(!naive.wants_materialized("v", false, false));
        // And an override still names the exact set, ignoring both.
        let overridden = NodeFusionPass::new(true, true, Some(vec!["other".into()]));
        assert!(!overridden.wants_materialized("v", true, true));
    }

    #[test]
    fn test_an_override_matches_a_bare_name_against_a_qualified_id() {
        let pass = NodeFusionPass::new(false, false, Some(vec!["orders".to_string()]));
        assert!(pass.wants_materialized("\"wh\".\"main\".\"orders\"", false, false));
        assert!(!pass.wants_materialized("\"wh\".\"main\".\"other\"", true, true));
    }

    // ------------------------------------------------------------------
    // The DAGs that are left alone
    // ------------------------------------------------------------------

    async fn assert_not_fused(dag: &mut Dag, expect: &str) {
        let before: Vec<(String, String)> = dag
            .nodes
            .nodes()
            .map(|n| (n.id.clone(), n.query_text.clone()))
            .collect();
        let n_before = dag.nodes.num_nodes();

        let mut pass = NodeFusionPass::new(false, false, None);
        let record = pass.rewrite(dag).expect("an unfusable DAG is not an error");

        let PassDetail::NodeFusion(detail) = &record.detail else {
            panic!("expected a NodeFusion detail");
        };
        assert!(
            detail.outcome.contains(expect),
            "expected an outcome mentioning '{expect}', got '{}'",
            detail.outcome
        );
        assert_eq!(detail.tables_fused, 0);
        assert_eq!(record.changes_applied, 0);
        assert_eq!(dag.nodes.num_nodes(), n_before, "no node was added");
        let after: Vec<(String, String)> = dag
            .nodes
            .nodes()
            .map(|n| (n.id.clone(), n.query_text.clone()))
            .collect();
        let mut before = before;
        let mut after = after;
        before.sort();
        after.sort();
        assert_eq!(before, after, "an unfusable DAG must be left exactly as it was");
    }

    #[tokio::test]
    async fn test_a_single_table_is_not_worth_fusing() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;
        let mut dag = make_dag(vec![
            node("stg", "SELECT * FROM orders", MaterializeMode::View, &[]),
            node(
                "only",
                "SELECT count(*) AS n FROM stg",
                MaterializeMode::Table,
                &["stg"],
            ),
        ]);
        resolved(&conn, &mut dag).await;
        assert_not_fused(&mut dag, "at least two").await;
    }

    #[tokio::test]
    async fn test_a_temp_table_upstream_of_a_table_blocks_fusion() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;
        let mut dag = make_dag(vec![
            node("pad", "SELECT * FROM orders", MaterializeMode::TempTable, &[]),
            node(
                "count_pad",
                "SELECT count(*) AS n FROM pad",
                MaterializeMode::Table,
                &["pad"],
            ),
            node(
                "sum_pad",
                "SELECT sum(amount) AS s FROM pad",
                MaterializeMode::Table,
                &["pad"],
            ),
        ]);
        resolved(&conn, &mut dag).await;
        assert_not_fused(&mut dag, "TempTable").await;
    }

    #[tokio::test]
    async fn test_a_node_the_parser_cannot_read_blocks_fusion_only_if_it_reads_a_cte() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;

        // `extract('year' from d)` is DuckDB-valid and polyglot-sql cannot
        // parse it. Here it sits in a leaf staging view that names no fused
        // node, so nothing needs rewriting and it is carried through exactly
        // as authored.
        let mut dag = make_dag(vec![
            node(
                "stg",
                "SELECT order_id, region, amount, \
                 extract('year' from DATE '2024-03-01') AS yr FROM orders",
                MaterializeMode::View,
                &[],
            ),
            node(
                "by_region",
                "SELECT region, count(*) AS n FROM stg GROUP BY region",
                MaterializeMode::Table,
                &["stg"],
            ),
            node(
                "totals",
                "SELECT count(*) AS n, max(yr) AS yr FROM stg",
                MaterializeMode::Table,
                &["stg"],
            ),
        ]);
        resolved(&conn, &mut dag).await;
        NodeFusionPass::new(false, false, None)
            .rewrite(&mut dag)
            .expect("an unparseable leaf is carried through verbatim");
        let fused = dag.nodes.get("dee_fused".to_string()).unwrap();
        assert!(
            fused.query_text.contains("extract('year'"),
            "the body should be used as authored, not regenerated -- got:\n{}",
            fused.query_text
        );
        let engine = SimpleEngine::new(Arc::clone(&conn)).unwrap();
        engine.run(&dag).await.expect("and the fused DAG still runs");
        engine.cleanup(&dag).await.unwrap();

        // The same syntax in a node that *does* read a CTE cannot be rewritten,
        // and blocks fusion rather than being substituted textually.
        let mut dag = make_dag(vec![
            node(
                "stg",
                "SELECT order_id, region, amount FROM orders",
                MaterializeMode::View,
                &[],
            ),
            node(
                "by_region",
                "SELECT region, count(*) AS n, \
                 extract('year' from DATE '2024-03-01') AS yr FROM stg GROUP BY region",
                MaterializeMode::Table,
                &["stg"],
            ),
            node(
                "totals",
                "SELECT count(*) AS n FROM stg",
                MaterializeMode::Table,
                &["stg"],
            ),
        ]);
        resolved(&conn, &mut dag).await;
        assert_not_fused(&mut dag, "cannot be rewritten at the AST level").await;
    }

    #[tokio::test]
    async fn test_an_unresolved_schema_blocks_fusion() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;
        // Deliberately not resolved: the fused relation's columns come from
        // these schemas, and guessing them would be worse than not fusing.
        let mut dag = two_table_dag();
        assert_not_fused(&mut dag, "no resolved schema").await;
    }

    #[tokio::test]
    async fn test_fusing_an_already_fused_dag_is_refused() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;
        let mut dag = two_table_dag();
        resolved(&conn, &mut dag).await;
        NodeFusionPass::new(false, false, None).rewrite(&mut dag).unwrap();

        // The fused node is a TempTable and every Table now reads it, so the
        // second pass sees a TempTable upstream of a Table and declines. It
        // must decline rather than fuse again: a second fusion would wrap the
        // projections in another UNION ALL and gain nothing.
        assert_not_fused(&mut dag, "TempTable").await;
    }

    // ------------------------------------------------------------------
    // Naming
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn test_the_fused_node_lands_in_the_same_schema_as_its_tables() {
        let conn = in_memory_conn().await;
        setup_orders(&conn).await;
        conn.execute("CREATE SCHEMA wh".to_string()).await.unwrap();

        let mut dag = make_dag(vec![
            node(
                "\"wh\".\"stg\"",
                "SELECT order_id, region, amount FROM orders",
                MaterializeMode::View,
                &[],
            ),
            node(
                "\"wh\".\"count_stg\"",
                "SELECT count(*) AS n FROM \"wh\".\"stg\"",
                MaterializeMode::Table,
                &["\"wh\".\"stg\""],
            ),
            node(
                "\"wh\".\"sum_stg\"",
                "SELECT sum(amount) AS s FROM \"wh\".\"stg\"",
                MaterializeMode::Table,
                &["\"wh\".\"stg\""],
            ),
        ]);
        resolved(&conn, &mut dag).await;

        NodeFusionPass::new(false, false, None).rewrite(&mut dag).unwrap();
        assert!(
            dag.nodes.get("\"wh\".\"dee_fused\"".to_string()).is_some(),
            "the fused node inherits the schema prefix of the Tables reading it"
        );

        let engine = SimpleEngine::new(Arc::clone(&conn)).unwrap();
        engine.run(&dag).await.expect("the fused DAG runs");
        assert_eq!(fingerprint(&conn, "\"wh\".\"count_stg\"").0, 1);
        engine.cleanup(&dag).await.unwrap();
    }
}
