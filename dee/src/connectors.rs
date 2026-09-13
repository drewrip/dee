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

    /// Stop the queries this connector currently has in flight, returning how
    /// many were signalled.
    ///
    /// Cancelling a DAG means cancelling the SQL. Dropping or aborting the task
    /// that awaits a query does not stop the engine executing it: the driver
    /// call blocks, and the statement runs to completion regardless. Anything
    /// that then drops or rebuilds the relation races a write still in flight,
    /// which is how a cancelled run corrupts the warehouse it was supposed to
    /// leave alone.
    ///
    /// **When this returns, nothing is executing.** Signalling alone is not
    /// enough: the task that issued a statement can be aborted while the
    /// statement itself is still unwinding, so this signals and then waits for
    /// the engine to actually go quiet, re-signalling as it waits. Callers rely
    /// on that to drop and rebuild those relations safely.
    async fn interrupt_inflight(&self) -> usize;

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

    /// The plan this backend would use for `query_text`, in the backend's own
    /// JSON, without executing it.
    ///
    /// A plain EXPLAIN, never an ANALYZE: the caller is asking what a query
    /// *would* do, and the query may be an expensive one it has no intention of
    /// running. The text is whatever [`Connector::parse_plan`] on the same
    /// connector accepts.
    ///
    /// `Ok(None)` when the connector cannot answer, which a caller must treat
    /// as "unknown" rather than "no plan".
    async fn explain(&self, _query_text: &str) -> Result<Option<String>, ConnectorError> {
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

    /// Which write path this engine would persist `query_text`'s result
    /// through, asked of the engine itself.
    ///
    /// An engine that has more than one way to write a relation decides which
    /// during planning, and will say so if the statement it is asked to plan is
    /// the write rather than the `SELECT` underneath it. On DuckDB, `EXPLAIN` of
    /// a `CREATE OR REPLACE TABLE ... AS ...` names the sink at the root of the
    /// plan --- exactly, including the cases the shape of the `SELECT` cannot
    /// distinguish --- without executing anything and without creating or
    /// replacing the table.
    ///
    /// The engine's answer or nothing. There was briefly a fallback that
    /// inferred the sink from the shape of the `SELECT`, and it was removed: it
    /// was right 83 times in 86 against the engine's 86, and the three it missed
    /// were a case the `SELECT` plan genuinely cannot express (a window's
    /// `PARTITION BY`). A cost model that silently substitutes a guess for a
    /// measurement is worse than one that declines to answer, because the guess
    /// is indistinguishable from the fact downstream.
    ///
    /// `Ok(None)` means *this engine will not say*, and the caller must treat
    /// the write as unpriced rather than free.
    ///
    /// Defaults to [`SINGLE_WRITE_PATH`](crate::plan::SINGLE_WRITE_PATH) --- one
    /// way of writing, which is the truth for Postgres and for any engine that
    /// names no write operator, and is what
    /// [`crate::plan::observed_write_path`] files those engines' measurements
    /// under, so the two agree by construction.
    async fn write_path_for(&self, _query_text: &str) -> Result<Option<String>, ConnectorError> {
        Ok(Some(crate::plan::SINGLE_WRITE_PATH.to_string()))
    }

    /// A stable identity for the engine that learned cost constants belong to.
    ///
    /// Seconds per byte is a property of an engine on a machine, not of the
    /// pipeline that happened to measure it, so what one DAG observes should
    /// price the next one's Views. But it is a property of *that* engine.
    /// DuckDB appending to a local file and Postgres writing through WAL to a
    /// different disk do not share a write constant, and neither do two
    /// databases whose storage settings differ --- so constants are stored
    /// under this key, and shared exactly as far as they transfer.
    ///
    /// Carry what plausibly changes the cost of a byte and nothing else. A key
    /// that carries an irrelevant setting splits one engine's samples into two
    /// half-learned models, and a constant fitted on too few writes is the
    /// failure this exists to avoid.
    ///
    /// `Ok(None)` where the connector cannot identify itself. The caller then
    /// keeps its constants in memory rather than pooling them with an unknown
    /// engine's, which is the whole point of the key.
    async fn cost_backend_key(&self) -> Result<Option<String>, ConnectorError> {
        Ok(None)
    }

    /// How much parallelism the engine as a whole can bring to bear, in
    /// cores or worker slots.
    ///
    /// The denominator for "does one node leave capacity another node could
    /// use". It is the engine's budget rather than the machine's core count
    /// because the two differ, and it is the *global* budget rather than the
    /// per-query one because that is what a second concurrent node competes
    /// for: DuckDB shares one thread pool across every query, so a second node
    /// gets no threads of its own, while Postgres hands each backend its own
    /// workers up to a server-wide ceiling. `None` where the engine will not
    /// say, and the caller falls back to the machine.
    async fn parallelism_budget(&self) -> Result<Option<usize>, ConnectorError> {
        Ok(None)
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
