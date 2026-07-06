//! Translates database `EXPLAIN` / `EXPLAIN ANALYZE` output into a DataFusion
//! [`ExecutionPlan`], so that plans produced by other query engines can be
//! manipulated and reasoned about with DataFusion's plan APIs.

use std::sync::Arc;

use datafusion::physical_plan::ExecutionPlan;

pub mod duckdb;

pub use duckdb::{DuckDBTranslateConfig, DuckDBTranslateError};

/// Implemented for each database system whose `EXPLAIN` output we know how to
/// translate into a DataFusion [`ExecutionPlan`].
pub trait ExplainToPlan {
    type Error: std::error::Error;

    /// Parse `explain`, the textual output of that database's `EXPLAIN` (or
    /// `EXPLAIN ANALYZE`) statement, and translate it into an equivalent
    /// DataFusion [`ExecutionPlan`].
    fn explain_to_plan(&self, explain: String) -> Result<Arc<dyn ExecutionPlan>, Self::Error>;
}
