use async_trait::async_trait;
use datafusion::{
    common::tree_node::{Transformed, TreeNode, TreeNodeRecursion},
    datasource::empty::EmptyTable,
    logical_expr::{LogicalPlan, table_scan},
    prelude::SessionContext,
    sql::unparser::plan_to_sql,
};
use std::{
    collections::{HashMap, HashSet},
    marker::PhantomData,
    sync::Arc,
};

use log::debug;

use crate::{
    connectors::Connector,
    dag::{Dag, MaterializeMode, TransformNode},
    executor::Executor,
    opt::{OptimizerError, OptimizerPass},
};

#[derive(Debug, Clone)]
pub struct CSEPass<C, E>
where
    C: Connector + Send + 'static,
    E: Executor<C> + Send,
{
    a: PhantomData<C>,
    b: PhantomData<E>,
}

impl<C, E> CSEPass<C, E>
where
    C: Connector + Send + 'static,
    E: Executor<C> + Send,
{
    pub fn new() -> Self {
        Self {
            a: PhantomData,
            b: PhantomData,
        }
    }
}

#[async_trait]
impl<C, E> OptimizerPass<C, E> for CSEPass<C, E>
where
    C: Connector + Send + 'static,
    E: Executor<C> + Send,
{
    async fn run(&mut self, dag: &mut Dag) -> Result<HashMap<String, String>, OptimizerError> {
        debug!("Running CSEPass");
        let ctx = SessionContext::new();
        for s in &dag.sources {
            let et = Arc::new(EmptyTable::new(s.schema.clone()));
            ctx.register_table(s.name.clone(), et).map_err(|_| {
                OptimizerError::Exec(format!(
                    "couldn't register empty table for source {}",
                    s.name.clone()
                ))
            })?;
        }
        let mut lps = Vec::new();
        let mut subtrees: Vec<Vec<LogicalPlan>> = vec![vec![]; dag.nodes.num_nodes()];
        let df_optimizer = datafusion::optimizer::Optimizer::new();
        let config = datafusion::optimizer::OptimizerContext::new().with_max_passes(16);
        for (i, node) in dag.nodes.nodes().enumerate() {
            let df = ctx.sql(&node.query_text.clone()).await.unwrap();
            let mut lp = df.logical_plan().clone();
            lp = df_optimizer.optimize(lp, &config, |_, _| ()).unwrap();
            lp.apply(|node| {
                subtrees[i].push(node.clone());
                Ok(TreeNodeRecursion::Continue)
            })
            .unwrap();
            lps.push(lp);
        }
        let mut lookup = vec![];
        for st in &subtrees {
            let mut table = HashMap::new();
            for (j, n) in st.iter().enumerate() {
                table.insert(n, j);
            }
            lookup.push(table);
        }

        let mut common = vec![];
        for (i, lp) in subtrees.get(0).unwrap().iter().enumerate() {
            let mut matches_all = true;
            common = vec![i];
            for (dim, _) in subtrees.iter().enumerate().skip(1) {
                let new_idx = lookup[dim].get(lp);
                match new_idx {
                    Some(idx) => common.push(*idx),
                    None => matches_all = false,
                };
            }
            if matches_all {
                break;
            }
        }
        let cs = &subtrees[0][common[0]];
        let out_refs = cs.expressions();
        println!("exprs = {:?}", out_refs);
        let common_schema = cs.schema().as_arrow();
        let common_table = Arc::new(EmptyTable::new(Arc::new(common_schema.clone())));
        ctx.register_table("cse_1".to_string(), common_table)
            .unwrap();
        let sql = plan_to_sql(cs).unwrap();
        dag.nodes.add_node_unchecked(TransformNode {
            id: "cse_1".to_string(),
            query_text: sql.to_string(),
            materialize: MaterializeMode::View,
            depends_on: HashSet::new(),
        });
        let new_table_scan = table_scan(Some("cse_1"), common_schema, None).unwrap();

        let new_scan_plan = new_table_scan.plan();

        let mut new_edges = Vec::new();
        for (node, lp) in dag.nodes.nodes_mut().zip(lps) {
            let new_lp = lp
                .transform_down(|expr| {
                    if expr == *cs {
                        Ok(Transformed::yes(new_scan_plan.clone()))
                    } else {
                        Ok(Transformed::no(expr))
                    }
                })
                .unwrap();
            let new_sql = plan_to_sql(&new_lp.data).unwrap().to_string();
            node.query_text = new_sql;
            new_edges.push(("cse_1", node.id.clone()));
        }

        for (src, dst) in new_edges {
            dag.nodes.add_edge(&src.to_string(), &dst).unwrap();
        }

        Ok(HashMap::new())
    }
}
