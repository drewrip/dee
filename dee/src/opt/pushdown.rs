use async_trait::async_trait;
use std::{collections::HashMap, marker::PhantomData, sync::Arc};

use crate::{
    connectors::Connector,
    executor::Executor,
    opt::{Dag, OptimizerError, OptimizerPass},
};

#[derive(Debug, Clone)]
pub struct PushdownPass<C, E>
where
    C: Connector + Send + 'static + Sync,
    E: Executor<C> + Send + Sync,
{
    _conn: Arc<C>,
    _phantom: PhantomData<E>,
}

impl<C, E> PushdownPass<C, E>
where
    C: Connector + Send + 'static + Sync,
    E: Executor<C> + Send + Sync,
{
    pub fn new(conn: Arc<C>) -> Self {
        Self {
            _conn: conn,
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
    async fn run(&mut self, _dag: &mut Dag) -> Result<HashMap<String, String>, OptimizerError> {
        Err(OptimizerError::NotImplemented("PushdownPass".to_string()))
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

    fn node(
        id: &str,
        query: &str,
        mode: MaterializeMode,
        deps: &[&str],
    ) -> TransformNode {
        TransformNode {
            id: id.to_string(),
            query_text: query.to_string(),
            materialize: mode,
            depends_on: deps.iter().map(|s| s.to_string()).collect::<HashSet<_>>(),
        }
    }

    fn pass() -> PushdownPass<StubConnector, StubExecutor> {
        PushdownPass::new(Arc::new(StubConnector))
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
        let raw = node("raw", "SELECT id, amount, status FROM source_table", MaterializeMode::View, &[]);
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
        let original = dag.nodes.get("staging".to_string()).unwrap().query_text.clone();

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
        pass().run(&mut dag).await.expect("pass should succeed");

        let rewritten = dag.nodes.get("staging".to_string()).unwrap().query_text.clone();

        assert!(
            rewritten.contains("region = 'US'"),
            "filter from the Table consumer should be pushed into the TempTable; got: {}",
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
        pass().run(&mut dag).await.expect("pass should succeed");

        let rewritten = dag.nodes.get("staging".to_string()).unwrap().query_text.clone();

        assert!(
            rewritten.contains("region = 'US'"),
            "common filter should be pushed when all Table consumers agree; got: {}",
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
        pass().run(&mut dag).await.expect("pass should succeed");

        let rewritten = dag.nodes.get("staging".to_string()).unwrap().query_text.clone();

        assert!(
            rewritten.contains("region = 'US'") && rewritten.contains("region = 'EU'"),
            "both filter predicates should appear in the rewritten TempTable query as an OR; got: {}",
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
        pass().run(&mut dag).await.expect("pass should succeed");

        let rewritten = dag.nodes.get("staging".to_string()).unwrap().query_text.clone();

        assert!(
            rewritten.contains("region") && rewritten.contains("amount"),
            "columns needed by Table consumers must be present; got: {}",
            rewritten
        );
        assert!(
            !rewritten.contains("id"),
            "column `id` not needed by any Table consumer should be pruned; got: {}",
            rewritten
        );
    }
}
