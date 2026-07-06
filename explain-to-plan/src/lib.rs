//! Translates database `EXPLAIN` / `EXPLAIN ANALYZE` output into a DataFusion
//! [`ExecutionPlan`], so that plans produced by other query engines can be
//! manipulated and reasoned about with DataFusion's plan APIs.

use std::sync::Arc;

use datafusion::physical_plan::ExecutionPlan;

pub mod duckdb;
pub mod raise;

pub use duckdb::{DuckDBTranslateConfig, DuckDBTranslateError};
pub use raise::{RaiseToLogicalError, raise_to_logical};

/// Arrow schema metadata key a [`ExplainToPlan`] implementation should set on
/// the schema of every scan leaf (e.g. `EmptyExec`) it produces, containing
/// the name of the table that leaf scans.
///
/// A translated [`ExecutionPlan`] otherwise carries no table identity at its
/// scan leaves — [`EmptyExec`](datafusion::physical_plan::empty::EmptyExec)
/// stores only a schema. [`raise_to_logical`] reads this key back off each
/// leaf's schema to reconstruct a `LogicalPlan::TableScan` referencing the
/// right table.
pub const TABLE_NAME_METADATA_KEY: &str = "explain_to_plan.table_name";

/// Implemented for each database system whose `EXPLAIN` output we know how to
/// translate into a DataFusion [`ExecutionPlan`].
pub trait ExplainToPlan {
    type Error: std::error::Error;

    /// Parse `explain`, the textual output of that database's `EXPLAIN` (or
    /// `EXPLAIN ANALYZE`) statement, and translate it into an equivalent
    /// DataFusion [`ExecutionPlan`].
    fn explain_to_plan(&self, explain: String) -> Result<Arc<dyn ExecutionPlan>, Self::Error>;
}
