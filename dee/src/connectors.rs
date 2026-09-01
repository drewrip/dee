use async_trait::async_trait;
use ::duckdb::arrow::datatypes::SchemaRef;
use std::{collections::HashMap, sync::Arc};
use thiserror::Error;

use crate::dag::MaterializeMode;

/// All pre-implemented connectors
pub mod duckdb;
pub mod postgres;

#[derive(Error, Debug)]
pub enum ConnectorError {
    #[error("couldn't create a connection to the DB - {0}")]
    Create(String),
    #[error("couldn't execute query against connector - {0}")]
    Execute(String),
}

/// What can be pushed down into **one scan** of a relation, as reported by a
/// connector's native query planner (e.g. DuckDB's `EXPLAIN (FORMAT JSON)`).
///
/// `projections` are the column names that scan actually reads; `filters` are
/// raw SQL predicate strings (in the connector's own dialect) that the plan
/// applies directly against that scan, and are **conjuncts** — the scan keeps
/// a row only if every one of them holds.
///
/// One query can scan the same relation more than once (a self-join, or two
/// branches of a UNION), and each such scan gets its own `PushdownInfo`,
/// which is why [`Connector::pushdown`] reports a `Vec` per relation. Merging
/// them would be unsound: two scans' filters are alternatives (the relation
/// must keep every row either scan reads), never conjuncts.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PushdownInfo {
    pub projections: Vec<String>,
    pub filters: Vec<String>,
}

#[async_trait]
pub trait Connector {
    type Config;
    type Connection;

    async fn new(config: Self::Config) -> Result<Arc<Self::Connection>, ConnectorError>;

    async fn execute(&self, query_text: String) -> Result<usize, ConnectorError>;

    async fn new_relation(
        &self,
        relation_type: MaterializeMode,
        name: String,
        query_text: String,
    ) -> Result<usize, ConnectorError>;

    async fn new_relation_and_explain(
        &self,
        relation_type: MaterializeMode,
        name: String,
        query_text: String,
    ) -> Result<(usize, Option<String>), ConnectorError> {
        let res = self.new_relation(relation_type, name, query_text).await?;
        Ok((res, None))
    }

    async fn drop_relation(
        &self,
        relation_type: MaterializeMode,
        name: String,
    ) -> Result<usize, ConnectorError>;

    async fn get_schema(&self, name: String) -> Option<Result<SchemaRef, ConnectorError>>;

    /// Ask the connector's own query planner what can be pushed down into
    /// each relation `query_text` scans, keyed by relation name — one
    /// [`PushdownInfo`] per scan of that relation, in plan order, since a
    /// query can scan the same relation several times with different
    /// predicates each time.
    ///
    /// Returns `Ok(None)` when the connector has no native way to answer
    /// this (e.g. Postgres today). Returns `Ok(Some(map))` — possibly with
    /// an empty map if the query scans no relations directly (e.g. a
    /// constant-only `SELECT`) — when the connector could analyze the query.
    async fn pushdown(
        &self,
        _query_text: &str,
    ) -> Result<Option<HashMap<String, Vec<PushdownInfo>>>, ConnectorError> {
        Ok(None)
    }

    /// Parse this backend's plan JSON into the optimizer's neutral plan form.
    ///
    /// Returns `None` when the text is not a plan this backend recognizes.
    fn parse_plan(&self, _json: &str) -> Option<Vec<crate::plan::PlanNode>> {
        None
    }

    /// What this backend's per-operator plan timings physically measure.
    ///
    /// DuckDB reports CPU time, Postgres wall time, so a cost ranking built
    /// from them is optimizing for different things on each. Results record
    /// this so the two are never silently compared.
    fn time_basis(&self) -> crate::plan::TimeBasis {
        crate::plan::TimeBasis::CpuTime
    }

    async fn sample_system_cpu_usage(&self) -> Result<Option<f64>, ConnectorError> {
        Ok(None)
    }

    async fn sample_system_memory_usage(&self) -> Result<Option<u64>, ConnectorError> {
        Ok(None)
    }

    async fn sample_system_disk_usage(&self) -> Result<DiskUsageSample, ConnectorError> {
        Ok(DiskUsageSample::default())
    }
}

/// A single point of process/DB-reported disk activity. All fields are
/// independently optional since not every connector can report every field
/// (e.g. Postgres runs out-of-process and can only report DB-side sizes,
/// not this process's own read/write bytes).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DiskUsageSample {
    /// On-disk size of the connector's working file/database, in bytes.
    pub disk_bytes: Option<u64>,
    /// Cumulative bytes read by this process since it started.
    pub read_bytes: Option<u64>,
    /// Cumulative bytes written by this process since it started.
    pub written_bytes: Option<u64>,
}
