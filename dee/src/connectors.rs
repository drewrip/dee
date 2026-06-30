use async_trait::async_trait;
use datafusion::arrow::datatypes::SchemaRef;
use std::sync::Arc;
use thiserror::Error;
use crate::dag::MaterializeMode;

// All pre-implemented connectors
pub mod duckdb;
pub mod postgres;

// ---------------------------------------------------------------------------
// Capability traits — compile-time discoverable connector features
// ---------------------------------------------------------------------------

/// Schema retrieval capability.
///
/// Connectors implementing this trait can resolve the Arrow output schema for
/// a named relation by executing `SELECT * FROM <name> LIMIT 0` and reading
/// the schema pointer.
#[async_trait]
pub trait SchemaSupport: Send + Sync {
    async fn get_schema(&self, name: &str) -> Result<SchemaRef, ConnectorError>;
}

/// Query plan explanation capability.
///
/// Connectors implementing this trait can produce a JSON query plan string
/// for an arbitrary SQL query via `EXPLAIN (FORMAT JSON) <query>`.
#[async_trait]
pub trait ExplainSupport: Send + Sync {
    async fn explain(&self, query: &str) -> Result<String, ConnectorError>;
}

/// Query profiling capability.
///
/// Connectors implementing this trait can capture execution profiling data
/// (e.g. DuckDB's `enable_profiling = 'json'`) and return it alongside the
/// row count.
#[async_trait]
pub trait ProfilingSupport: Send + Sync {
    type PlanData: Send + Sync;

    async fn execute_with_profile(
        &self,
        query: &str,
    ) -> Result<(usize, Option<Self::PlanData>), ConnectorError>;
}

/// Relation creation and deletion capability.
///
/// Connectors implementing this trait can create and drop relations
/// (tables, views, temp tables) in a dialect-specific way.
#[async_trait]
pub trait RelationOps: Send + Sync {
    async fn create_relation(
        &self,
        relation_type: MaterializeMode,
        name: String,
        query: String,
    ) -> Result<usize, ConnectorError>;

    async fn drop_relation(
        &self,
        relation_type: MaterializeMode,
        name: String,
    ) -> Result<usize, ConnectorError>;
}

/// System metrics sampling capability.
///
/// Connectors implementing this trait can report CPU and memory usage
/// samples from the database process.
#[async_trait]
pub trait SystemMetrics: Send + Sync {
    async fn sample_cpu(&self) -> Result<Option<f64>, ConnectorError>;
    async fn sample_memory(&self) -> Result<Option<u64>, ConnectorError>;
}
// Blanket impl: Arc<T> delegates to T for all capability traits.
// This lets us use SchemaSupport, ExplainSupport, etc. on Arc<Connector>
// without requiring explicit impls on Arc.
#[async_trait]
impl<T: SchemaSupport + ?Sized> SchemaSupport for Arc<T> {
    async fn get_schema(&self, name: &str) -> Result<datafusion::arrow::datatypes::SchemaRef, ConnectorError> {
        (**self).get_schema(name).await
    }
}

#[async_trait]
impl<T: ExplainSupport + ?Sized> ExplainSupport for Arc<T> {
    async fn explain(&self, query: &str) -> Result<String, ConnectorError> {
        (**self).explain(query).await
    }
}

#[async_trait]
impl<T: ProfilingSupport + ?Sized> ProfilingSupport for Arc<T> {
    type PlanData = T::PlanData;

    async fn execute_with_profile(&self, query: &str) -> Result<(usize, Option<Self::PlanData>), ConnectorError> {
        (**self).execute_with_profile(query).await
    }
}

#[async_trait]
impl<T: RelationOps + ?Sized> RelationOps for Arc<T> {
    async fn create_relation(&self, relation_type: MaterializeMode, name: String, query: String) -> Result<usize, ConnectorError> {
        (**self).create_relation(relation_type, name, query).await
    }

    async fn drop_relation(&self, relation_type: MaterializeMode, name: String) -> Result<usize, ConnectorError> {
        (**self).drop_relation(relation_type, name).await
    }
}

#[async_trait]
impl<T: SystemMetrics + ?Sized> SystemMetrics for Arc<T> {
    async fn sample_cpu(&self) -> Result<Option<f64>, ConnectorError> {
        (**self).sample_cpu().await
    }

    async fn sample_memory(&self) -> Result<Option<u64>, ConnectorError> {
        (**self).sample_memory().await
    }
}

// ---------------------------------------------------------------------------
// ConnectorError
// ---------------------------------------------------------------------------

#[derive(Error, Debug)]
pub enum ConnectorError {
    #[error("couldn't create a connection to the DB - {0}")]
    Create(String),
    #[error("couldn't execute query against connector - {err} in query: {query}")]
    Execute { err: String, query: String },
    #[error("schema error: {0}")]
    Schema(String),
    #[error("explain not supported")]
    ExplainNotSupported,
    #[error("relation not found: {name} in mode {mode}")]
    RelationNotFound { name: String, mode: String },
    #[error("permission error: {0}")]
    Permission(String),
    #[error("timeout: {0}")]
    Timeout(String),
    #[error("internal error: {0}")]
    Internal(String),
}

// ---------------------------------------------------------------------------
// Helper: detect "relation not found" from error strings
// ---------------------------------------------------------------------------

/// Return `true` if `err` looks like a "relation / table / view not found"
/// error from a SQL database.  Covers common patterns across PostgreSQL,
/// DuckDB, and other engines.
fn is_relation_not_found(err: &str) -> bool {
    let lower = err.to_lowercase();
    lower.contains("relation not found")
        || lower.contains("does not exist")
        || lower.contains("table not found")
        || lower.contains("table doesn't exist")
        || lower.contains("view not found")
        || lower.contains("view doesn't exist")
        || lower.contains("no such table")
        || lower.contains("no such view")
}

// ---------------------------------------------------------------------------
// Connector — base trait with capability-based shims
// ---------------------------------------------------------------------------

#[async_trait]
pub trait Connector: Send + Sync {
    type Config;
    type Connection;

    async fn new(config: Self::Config) -> Result<Arc<Self::Connection>, ConnectorError>;

    async fn execute(&self, query_text: String) -> Result<usize, ConnectorError>;

    // -- Relation lifecycle --

    /// Create a relation.
    ///
    /// *Deprecated*: Use [`RelationOps::create_relation`] where available.
    #[deprecated(
        since = "0.2.0",
        note = "Use RelationOps::create_relation"
    )]
    async fn new_relation(
        &self,
        relation_type: MaterializeMode,
        name: String,
        query_text: String,
    ) -> Result<usize, ConnectorError>;

    /// Create a relation and optionally return an explain string.
    ///
    /// *Deprecated*: Use [`RelationOps::create_relation`] combined with
    /// [`ExplainSupport::explain`] where available.
    #[deprecated(
        since = "0.2.0",
        note = "Use RelationOps::create_relation + ExplainSupport::explain"
    )]
    #[allow(deprecated)]
    async fn new_relation_and_explain(
        &self,
        relation_type: MaterializeMode,
        name: String,
        query_text: String,
    ) -> Result<(usize, Option<String>), ConnectorError> {
        let res = self.new_relation(relation_type, name, query_text).await?;
        Ok((res, None))
    }

    /// Drop a relation.
    ///
    /// *Deprecated*: Use [`RelationOps::drop_relation`] where available.
    #[deprecated(
        since = "0.2.0",
        note = "Use RelationOps::drop_relation"
    )]
    async fn drop_relation(
        &self,
        relation_type: MaterializeMode,
        name: String,
    ) -> Result<usize, ConnectorError>;

    // -- Schema resolution --

    /// Resolve the Arrow schema for a named relation.
    ///
    /// Returns `Some(Ok(schema))` when the connector supports schema
    /// resolution, `Some(Err(e))` when it does not (capability absent),
    /// or `None` when the relation simply doesn't exist.
    ///
    /// *Deprecated*: Use [`SchemaSupport::get_schema`] where available.
    #[deprecated(
        since = "0.2.0",
        note = "Use SchemaSupport::get_schema"
    )]
    async fn get_schema(
        &self,
        name: String,
    ) -> Option<Result<SchemaRef, ConnectorError>>;

    // -- Dialect --

    /// Return the SQL dialect identifier (e.g. `"duckdb"`, `"postgresql"`).
    async fn dialect(&self) -> &'static str {
        "unknown"
    }

    // -- Relation existence --

    /// Check whether a relation exists in the database.
    ///
    /// Default implementation executes `SELECT 1 FROM <name> LIMIT 0` and
    /// interprets the result: `true` on success, `false` on
    /// `RelationNotFound`, error otherwise.
    async fn relation_exists(&self, name: &str) -> Result<bool, ConnectorError> {
        match self
            .execute(format!("SELECT 1 FROM {} LIMIT 0", name))
            .await
        {
            Ok(_) => Ok(true),
            Err(e) if is_relation_not_found(&e.to_string()) => Ok(false),
            Err(e) => Err(e),
        }
    }

    // -- System metrics --

    async fn sample_system_cpu_usage(&self) -> Result<Option<f64>, ConnectorError> {
        Ok(None)
    }

    async fn sample_system_memory_usage(&self) -> Result<Option<u64>, ConnectorError> {
        Ok(None)
    }
}
