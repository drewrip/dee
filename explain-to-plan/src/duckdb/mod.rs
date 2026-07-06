mod error;
mod expr;
mod model;
mod translate;

pub use error::DuckDBTranslateError;
pub use translate::Catalog;

use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::SessionContext;

use crate::ExplainToPlan;

/// Configuration for translating DuckDB `EXPLAIN` / `EXPLAIN ANALYZE`
/// (`FORMAT JSON`) output into a DataFusion [`ExecutionPlan`].
///
/// DuckDB's `EXPLAIN` output does not carry column types, so base table
/// schemas must be registered up front via [`Self::with_table`] /
/// [`Self::with_catalog`] using the same table name DuckDB reports (e.g.
/// `"memory.main.orders"`, or just `"orders"`).
pub struct DuckDBTranslateConfig {
    ctx: SessionContext,
    catalog: Catalog,
}

impl Default for DuckDBTranslateConfig {
    fn default() -> Self {
        Self { ctx: SessionContext::new(), catalog: Catalog::new() }
    }
}

impl DuckDBTranslateConfig {
    pub fn new() -> Self {
        Self::default()
    }

    /// Use a caller-supplied [`SessionContext`] to resolve scalar/aggregate
    /// function names (e.g. one with extra UDFs registered).
    pub fn with_session_context(ctx: SessionContext) -> Self {
        Self { ctx, catalog: Catalog::new() }
    }

    /// Registers the schema for a table DuckDB may reference in a
    /// `SEQ_SCAN`/`TABLE_SCAN` node.
    pub fn with_table(mut self, name: impl Into<String>, schema: SchemaRef) -> Self {
        self.catalog.insert(name.into(), schema);
        self
    }

    /// Registers many table schemas at once.
    pub fn with_catalog(mut self, catalog: Catalog) -> Self {
        self.catalog.extend(catalog);
        self
    }
}

impl ExplainToPlan for DuckDBTranslateConfig {
    type Error = DuckDBTranslateError;

    fn explain_to_plan(&self, explain: String) -> Result<Arc<dyn ExecutionPlan>, Self::Error> {
        let root = model::parse_root(&explain)?;
        translate::translate_with_catalog(&root, &self.ctx, &self.catalog)
    }
}
