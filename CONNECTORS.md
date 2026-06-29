# Connector Interface Design & Extension Plan

## 1. Current State

### 1.1. The trait as-is

```rust
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
    ) -> Result<(usize, Option<String>), ConnectorError>;

    async fn drop_relation(
        &self,
        relation_type: MaterializeMode,
        name: String,
    ) -> Result<usize, ConnectorError>;

    async fn get_schema(&self, name: String) -> Option<Result<SchemaRef, ConnectorError>>;

    async fn sample_system_cpu_usage(&self) -> Result<Option<f64>, ConnectorError>;
    async fn sample_system_memory_usage(&self) -> Result<Option<u64>, ConnectorError>;
}
```

### 1.2. Two implementations

| Method | DuckDB | Postgres |
|--------|--------|----------|
| `execute` | `conn.execute(query, params!)` | `conn.execute(query)` |
| `new_relation` | `CREATE OR REPLACE <type> name AS (query)` | `CREATE [OR REPLACE] VIEW/TABLE name AS (query)` |
| `new_relation_and_explain` | EXPLAIN JSON for Views; `SET enable_profiling` for Tables | **default impl** → `(0, None)` |
| `drop_relation` | `DROP <type> IF EXISTS name` | `DROP <type> IF EXISTS name CASCADE` |
| `get_schema` | `SELECT * FROM name LIMIT 0` via `query_arrow()` | **always `None`** |
| `sample_cpu` | `ps -o %cpu= -p <pid>` | **default → `Ok(None)`** |
| `sample_memory` | `SELECT memory_usage FROM pragma_database_size()` | `SELECT SUM(total_bytes) FROM pg_backend_memory_contexts` |

### 1.3. Pain points

**Return-type issues:**
1. `get_schema` returns `Option<Result<SchemaRef, ConnectorError>>`. The `Option` is used for "not supported" (Postgres returns `None`), but callers (`executor.rs:486–506`) must match three branches: `Some(Ok)`, `Some(Err)`, and `None`. This is a classic anti-pattern — the `None` case is indistinguishable from "connector doesn't support schema introspection" vs "schema introspection failed but returned None instead of Err".
2. `sample_system_cpu_usage` and `sample_system_memory_usage` return `Result<Option<T>, ConnectorError>`. The `Option` means "not supported" but callers treat `None` as "no data collected" — the distinction is lost.
3. `new_relation_and_explain` returns `(usize, Option<String>)`. The `Option` default is `None` meaning "explain not supported", which is encoded as a return value rather than an error or explicit capability flag.
4. `execute` returns `usize` (rows affected). This is meaningless for SELECT queries and conflates DDL execution with data query results.

**Syntax assumptions:**
5. `new_relation` wraps the query in `AS (query)`. DuckDB accepts this; Postgres also accepts it for views but `CREATE TABLE ... AS (query)` may behave differently across dialects. A connector should generate its own DDL, not rely on a generic wrapper.
6. `new_relation` uses `CREATE OR REPLACE` for views but not for tables — DuckDB supports `CREATE OR REPLACE TABLE`, Postgres does not (needs `DROP + CREATE`). The connector needs dialect-specific DDL generation.
7. `drop_relation` uses `DROP TABLE IF EXISTS` — Postgres adds `CASCADE` but DuckDB does not.

**Missing capabilities:**
8. No way to execute arbitrary DDL that doesn't create/drop a relation (e.g., DuckDB's `SET enable_profiling`, `INSTALL icu; LOAD icu`). `execute()` exists but only returns `usize` — no result data is ever returned.
9. No way to execute a query and get results back (SELECT). The entire codebase only ever calls DDL methods; there is no `execute_query` that returns rows.
10. No way to check if a relation exists before operations. Both connectors unconditionally create/drop, which is fine for `CREATE OR REPLACE` / `DROP IF EXISTS` but prevents conditional logic (e.g., "skip if exists").
11. No way to run parameterized queries. Both `execute` and `new_relation` take a raw `String` for the query text. No parameter binding, which would be useful for safe queries and query plans.
12. No connector-level capability discovery. There's no way to ask a connector "do you support explain?" or "do you support schema introspection?" — callers must fall through error handling or check for `None`/`None` returns.

**Error handling:**
13. `ConnectorError` has only two variants: `Create(String)` and `Execute(String)`. There's no distinction for "schema not supported", "explain not supported", "relation not found", "permission denied", "timeout", etc.
14. `PostgresConfig` has no builder methods, no `new()`, no validation. `DuckDBConfig` has a builder but no `new()` constructor.
15. Connection pools use a 2-hour timeout (`Duration::from_hours(2)` for both DuckDB's r2d2 and Postgres's `log_slow_statements`). This means hung connections persist indefinitely.

**Architecture:**
16. `type Connection = Self` — both implementations use `Self` as their connection type. The associated type is therefore redundant and adds noise.
17. `type Config` has no trait bounds (`Send`, `Sync`, `Clone` must be added by each impl manually). If `Config` needs to be serialized (as it is in `connections.json`), `Serialize`/`Deserialize` must be derived per-connector.
18. No factory/registry pattern. CLI code (`run.rs:32–40`, `opt.rs:55–71`) uses a `Connection` enum with exhaustive `match` to construct the concrete connector. Adding a new connector requires touching four files: `connections.rs`, `run.rs`, `opt.rs`, and adding the connector module.
19. `resolve_schemas` in `executor.rs:353–554` creates temporary views using `new_relation` and then calls `get_schema`. This assumes all connectors support both operations — Postgres fails because `get_schema` returns `None`.
20. `materialize_mode_in_duckdb` and `materialize_mode_in_pg` are standalone functions mapping `MaterializeMode` to SQL strings. DuckDB treats `Table` and `TempTable` the same; Postgres does the same. This duplication is unnecessary but also indicates that the `MaterializeMode` enum may not capture the full semantic difference between connectors.

---

## 2. Proposed Trait Design

### 2.1. Capability-based trait hierarchy

Instead of one monolithic trait with optional methods (defaults returning `None`/`Ok(None)`), use a capability-based trait hierarchy where each method is opt-in via a supertrait.

```rust
// core.rs — the minimal connector trait
#[async_trait]
pub trait Connector: Send + Sync {
    /// Configuration type. Must be Serialize + Deserialize + Clone + Send + 'static.
    type Config: serde::Serialize + serde::de::DeserializeOwned + Clone + Send + 'static;

    /// Create a new connector instance.
    async fn new(config: Self::Config) -> Result<Self, ConnectorError>
    where
        Self: Sized;

    /// Execute a raw SQL query (DDL or DML).
    /// Returns the number of rows affected, or 0 for statements that don't produce a count.
    async fn execute(&self, query: &str) -> Result<usize, ConnectorError>;

    /// Dialect identifier (e.g., "duckdb", "postgresql", "mysql").
    fn dialect(&self) -> &str;

    /// Check if a relation exists.
    fn relation_exists(&self, name: &str) -> impl Future<Output = Result<bool, ConnectorError>> + Send;
}

// schema_support.rs — opt-in schema introspection
#[async_trait]
pub trait SchemaSupport: Connector {
    /// Resolve the Arrow schema of a relation.
    /// Returns an error (not None) if the relation doesn't exist or schema is unreadable.
    fn get_schema(&self, name: &str) -> impl Future<Output = Result<SchemaRef, ConnectorError>> + Send;
}

// explain_support.rs — opt-in query plan introspection
#[async_trait]
pub trait ExplainSupport: Connector {
    /// Run EXPLAIN on a query and return the plan string.
    fn explain(&self, query: &str) -> impl Future<Output = Result<String, ConnectorError>> + Send;
}

// profiling_support.rs — opt-in query profiling
#[async_trait]
pub trait ProfilingSupport: Connector {
    type PlanData: Send;

    /// Execute a query with profiling enabled and return the plan/profile data.
    /// The connector decides the best mechanism (EXPLAIN, SET enable_profiling, etc.).
    fn execute_with_profile(
        &self,
        query: &str,
    ) -> impl Future<Output = Result<(usize, Self::PlanData), ConnectorError>> + Send;
}

// system_metrics.rs — opt-in system resource monitoring
#[async_trait]
pub trait SystemMetrics: Connector {
    fn sample_cpu(&self) -> impl Future<Output = Result<Option<f64>, ConnectorError>> + Send;
    fn sample_memory(&self) -> impl Future<Output = Result<Option<u64>, ConnectorError>> + Send;
}

// relation_ops.rs — opt-in relation management (DML/DDL with typed operations)
#[async_trait]
pub trait RelationOps: Connector {
    /// Create a relation with the given materialization mode.
    fn create_relation(
        &self,
        mode: MaterializeMode,
        name: &str,
        query: &str,
    ) -> impl Future<Output = Result<usize, ConnectorError>> + Send;

    /// Drop a relation.
    fn drop_relation(
        &self,
        mode: MaterializeMode,
        name: &str,
    ) -> impl Future<Output = Result<usize, ConnectorError>> + Send;
}
```

### 2.2. Composed trait for full-featured connectors

```rust
// The trait that real connectors like DuckDBConnection and PostgresConnection implement.
pub type FullConnector = Connector
    + SchemaSupport
    + ExplainSupport
    + ProfilingSupport<PlanData = String>
    + SystemMetrics
    + RelationOps;
```

### 2.3. Trait alias for the Executor interface

The `Executor` trait currently takes a generic `C: Connector`. After the refactor:

```rust
#[async_trait]
pub trait Executor<C: Connector + Send + Sync> {
    type Engine;
    fn new(conn: Arc<C>) -> Result<Self::Engine, ExecutorError>;
    async fn run(&self, dag: &Dag) -> Result<ExecStats, ExecutorError>;
    async fn cleanup(&self, dag: &Dag) -> Result<usize, ExecutorError>;
    fn cancel_sender(&self) -> Arc<watch::Sender<bool>>;
    async fn resolve_schemas(&self, dag: &mut Dag) -> Result<(), ExecutorError>;
}
```

Where `resolve_schemas` is reworked to use `SchemaSupport` when available:
- If `C: SchemaSupport`, call `get_schema` directly.
- If not, fall back to creating temporary views and attempting `get_schema` (which will fail gracefully).

### 2.4. Error enum expansion

```rust
#[derive(Error, Debug)]
pub enum ConnectorError {
    #[error("connection failed: {0}")]
    Create(String),

    #[error("execution failed: {err} | query: {query}")]
    Execute { err: String, query: String },

    #[error("schema introspection failed for '{name}': {0}")]
    Schema(String),

    #[error("explain/profiling not supported by this connector")]
    ExplainNotSupported,

    #[error("relation '{name}' of type '{mode}' not found")]
    RelationNotFound { name: String, mode: String },

    #[error("permission denied: {0}")]
    Permission(String),

    #[error("timeout: {0}")]
    Timeout(String),

    #[error("internal error: {0}")]
    Internal(String),
}
```

---

## 3. Refactoring Steps

### Phase 1: Foundation (non-breaking)

#### Step 1.1: Expand `ConnectorError`
- Add `Schema(String)`, `ExplainNotSupported`, `RelationNotFound`, `Permission`, `Timeout`, `Internal` variants.
- Change `Execute(String)` to `Execute { err: String, query: String }` to always include the query text in errors.

#### Step 1.2: Add `dialect()` to `Connector`
- Add `fn dialect(&self) -> &str` with a default implementation returning `"unknown"`.
- Implement in DuckDB (`"duckdb"`) and Postgres (`"postgresql"`).

#### Step 1.3: Add `relation_exists()` to `Connector`
- Provide a default implementation that attempts a lightweight query (e.g., `SELECT 1 FROM name LIMIT 0`) and returns `true` if successful, `false` on `RelationNotFound`.
- This default is connector-agnostic and works for any SQL system.

### Phase 2: Extract capabilities into supertraits

#### Step 2.1: Create `schema_support` module
- Define the `SchemaSupport` trait with `get_schema(&self, name: &str) -> Result<SchemaRef, ConnectorError>`.
- Remove the `Option<...>` wrapper from the base trait's `get_schema`.
- The base trait's `get_schema` becomes deprecated and forwards to `SchemaSupport::get_schema`.

#### Step 2.2: Create `explain_support` module
- Define `ExplainSupport::explain(&self, query: &str) -> Result<String, ConnectorError>`.
- Remove the `new_relation_and_explain` method; replace with `create_relation_with_plan`.
- DuckDB implements both `explain()` and `create_relation_with_profile()` separately.

#### Step 2.3: Create `profiling_support` module
- Define `ProfilingSupport::execute_with_profile(&self, query: &str) -> Result<(usize, Self::PlanData), ConnectorError>`.
- DuckDB's PlanData = `String` (JSON from `enable_profiling`).
- Postgres's PlanData = `None` (not supported).

#### Step 2.4: Create `system_metrics` module
- Define `SystemMetrics::sample_cpu()` and `sample_memory()`.
- Remove from base `Connector`; keep defaults returning `Ok(None)`.

#### Step 2.5: Create `relation_ops` module
- Define `RelationOps::create_relation()` and `drop_relation()`.
- This replaces `new_relation` and `drop_relation` in the base trait.
- The base trait's methods become deprecated shims.

### Phase 3: Refactor implementations

#### Step 3.1: DuckDB connector
- Implement `SchemaSupport` — already works via `SELECT * FROM name LIMIT 0` + `query_arrow()`.
- Implement `ExplainSupport` — uses `EXPLAIN (FORMAT JSON) query` for Views.
- Implement `ProfilingSupport` — uses `SET enable_profiling = 'json'` for Tables.
- Implement `SystemMetrics` — already works via `pragma_database_size()` and `ps`.
- Implement `RelationOps` — use `CREATE OR REPLACE VIEW`, `CREATE TABLE`, `CREATE TEMPORARY TABLE`.
- Implement `dialect()` → `"duckdb"`.
- Implement `relation_exists()` → default (works via `SELECT 1 FROM name LIMIT 0`).

#### Step 3.2: Postgres connector
- Implement `SchemaSupport` — use `information_schema.columns` query:
  ```sql
  SELECT column_name, data_type
  FROM information_schema.columns
  WHERE table_name = $1 AND table_schema = $2
  ORDER BY ordinal_position
  ```
  Then build an Arrow `SchemaRef` from the results.
- Implement `ExplainSupport` — use `EXPLAIN (FORMAT JSON) query`.
- Implement `ProfilingSupport` — return `ExplainNotSupported` (Postgres doesn't have a simple profiling mechanism).
- Implement `SystemMetrics` — `sample_memory` already works; `sample_cpu` → `Ok(None)`.
- Implement `RelationOps` — use `CREATE OR REPLACE VIEW`, `CREATE TABLE AS`, `CREATE TEMPORARY TABLE`.
- Implement `dialect()` → `"postgresql"`.
- Implement `relation_exists()` → default.

### Phase 4: Update callers

#### Step 4.1: `resolve_schemas` in `executor.rs`
- Check `C: SchemaSupport` at compile time via trait bounds.
- If available, call `conn.get_schema(name)` directly.
- If not, fall back to the current temporary-view approach (which calls `get_schema` and handles `ConnectorError::Schema(...)` as "schema not available").

#### Step 4.2: `SimpleEngine::run`
- Replace `new_relation_and_explain` with `create_relation_with_profile` when profiling is enabled.
- Use `create_relation` / `drop_relation` instead of `new_relation` / `drop_relation`.

#### Step 4.3: `spawn_sampler`
- Check `C: SystemMetrics` at compile time.
- If available, call `conn.sample_cpu()` / `conn.sample_memory()`.
- If not, skip profiling entirely (or log at debug level).

#### Step 4.4: Optimizer passes
- `build_opaque_context` and `build_ctx_for_node` already use `conn.get_schema()`. They need to handle `ConnectorError::Schema(...)` gracefully.

### Phase 5: Connection factory pattern

#### Step 5.1: `ConnectionRegistry`
```rust
pub struct ConnectionRegistry {
    registry: HashMap<&'static str, Box<dyn Fn(serde_json::Value) -> Result<Arc<dyn FullConnector>, ConnectorError>>>,
}

impl ConnectionRegistry {
    pub fn register(&mut self, name: &'static str, factory: Box<dyn Fn(serde_json::Value) -> Result<Arc<dyn FullConnector>, ConnectorError>>) {
        self.registry.insert(name, factory);
    }

    pub fn connect(&self, config: serde_json::Value) -> Result<Arc<dyn FullConnector>, ConnectorError> {
        // ... match on config["type"], deserialize, call factory
    }
}
```

#### Step 5.2: Register built-in connectors
- `ConnectionRegistry::register("duckdb", box |v| { ... })`
- `ConnectionRegistry::register("postgresql", box |v| { ... })`

#### Step 5.3: Update CLI
- Replace the `match &target_connection { Connection::DuckDB(_) => ... Connection::Postgres(_) => ... }` pattern with `registry.connect(connection_json)?`.
- `connections.rs` `Connection` enum becomes an internal type for serialization only.

---

## 4. Adding a New Connector: Step-by-Step

To add a new connector (e.g., `clickhouse`), follow this checklist:

### 4.1. Create the connector module
```
dee/src/connectors/clickhouse.rs
```

### 4.2. Implement the types
```rust
#[derive(Serialize, Deserialize, Clone)]
pub struct ClickHouseConfig {
    pub host: String,
    pub port: u16,
    pub user: String,
    pub password: String,
    pub database: String,
}

pub struct ClickHouseConnection { /* pool/client */ }
```

### 4.3. Implement the base trait
```rust
#[async_trait]
impl Connector for ClickHouseConnection {
    type Config = ClickHouseConfig;

    async fn new(config: Self::Config) -> Result<Self, ConnectorError> { ... }

    async fn execute(&self, query: &str) -> Result<usize, ConnectorError> { ... }

    fn dialect(&self) -> &str { "clickhouse" }

    fn relation_exists(&self, name: &str) -> impl Future<Output = Result<bool, ConnectorError>> + Send {
        // default impl works
        async { todo!() }
    }
}
```

### 4.4. Implement optional capability traits (as needed)
```rust
#[async_trait]
impl SchemaSupport for ClickHouseConnection { ... }
#[async_trait]
impl ExplainSupport for ClickHouseConnection { ... }
#[async_trait]
impl ProfilingSupport for ClickHouseConnection {
    type PlanData = String;
    async fn execute_with_profile(&self, query: &str) -> Result<(usize, String), ConnectorError> { ... }
}
#[async_trait]
impl SystemMetrics for ClickHouseConnection { ... }
#[async_trait]
impl RelationOps for ClickHouseConnection { ... }
```

### 4.5. Register the connector
In `mod.rs`:
```rust
pub mod duckdb;
pub mod postgres;
pub mod clickhouse;  // new
```

### 4.6. Register in the connection registry
```rust
// In the connector module's init or a central registry setup:
registry.register("clickhouse", Box::new(|v: serde_json::Value| {
    let config: ClickHouseConfig = serde_json::from_value(v)?;
    Ok(Arc::new(ClickHouseConnection::new(config).await?))
}));
```

### 4.7. Add to `connections.json` format
Users can now specify `"type": "clickhouse"` in their connections file.

---

## 5. Migration Compatibility

### 5.1. Backward compatibility
The base `Connector` trait retains its existing method names (`new_relation`, `drop_relation`, `get_schema`, etc.) as deprecated shims that forward to the new trait methods. This allows existing code to compile during the transition.

### 5.2. Deprecation path
```rust
#[deprecated(since = "0.2.0", note = "use RelationOps::create_relation instead")]
async fn new_relation(...) -> ... { ... }

#[deprecated(since = "0.2.0", note = "use SchemaSupport::get_schema instead")]
fn get_schema(...) -> ... { ... }
```

### 5.3. No breaking changes in data format
The `DagFile`, `MaterializeMode`, and `connections.json` format remain unchanged. Only the internal trait structure evolves.

---

## 6. Design Decisions

### 6.1. Why trait hierarchy instead of optional methods?
The current approach (default methods returning `Ok(None)` or `None`) conflates three distinct concepts:
- **Capability:** "Does this connector support this feature?"
- **Error:** "The feature failed for a recoverable reason."
- **Absence:** "The feature returned no meaningful data."

A trait hierarchy makes the first two explicit at the type level (you can't call `get_schema` unless `C: SchemaSupport`), and the third as a `Result` error.

### 6.2. Why not use `Box<dyn Connector>` for dynamic dispatch?
The codebase already uses `Arc<C>` with static dispatch via generics. Dynamic dispatch via `Box<dyn Connector>` would:
- Inhibit inlining and monomorphization
- Add heap allocation per call
- Break `Send + Sync` bounds in async contexts

Static dispatch is preferred; the `ConnectionRegistry` returns `Arc<dyn FullConnector>` only at the boundary between CLI and library.

### 6.3. Why `PlanData = String` for profiling?
DuckDB's profiling output is JSON text. Postgres has no simple profiling mechanism. Using `String` as the plan data type is simple and avoids serialization overhead in the trait. Consumers can deserialize the JSON as needed.

### 6.4. Why keep `MaterializeMode` in the base trait?
`MaterializeMode` is a DAG-level concept, not a connector-level concept. The trait accepts it because `resolve_schemas` and `SimpleEngine::run` need to know how to materialize nodes. This is appropriate — the connector translates the abstract mode into concrete DDL.

### 6.5. Why `relation_exists` as a separate method?
Checking existence before `DROP` or `CREATE` is a common pattern. The default implementation (try `SELECT 1 FROM name LIMIT 0`) works for all SQL connectors. Connectors with catalog introspection (e.g., Postgres's `information_schema.tables`) can override for better performance.

### 6.6. Why remove `type Connection` associated type?
Both existing implementations use `Self` for `Connection`. The associated type adds zero value and forces every new connector to declare `type Connection = Self`. Removing it simplifies the trait: `Self` is always the connection type.

### 6.7. Why not add `execute_query` (SELECT support)?
The current codebase never executes SELECT queries — only DDL (CREATE/DROP). Adding SELECT support would require:
- A result type (rows, columns, Arrow arrays)
- Pagination support
- Streaming vs. buffering decisions
- Type mapping (SQL types → Rust/Arrow types)

This is a significant addition beyond the scope of the connector interface refinement. If SELECT support is needed later, it should be a separate `QueryResult` trait that returns an Arrow `RecordBatchReader`.

---

## 7. File Structure After Refactor

```
dee/src/connectors/
├── mod.rs              # pub mod duckdb; pub mod postgres; + ConnectionRegistry
├── core.rs             # Connector trait (execute, dialect, relation_exists)
├── schema_support.rs   # SchemaSupport trait
├── explain_support.rs  # ExplainSupport trait
├── profiling_support.rs# ProfilingSupport trait
├── system_metrics.rs   # SystemMetrics trait
├── relation_ops.rs     # RelationOps trait (create_relation, drop_relation)
├── error.rs            # ConnectorError enum (expanded)
├── duckdb.rs           # DuckDBConnection impl (all traits)
├── postgres.rs         # PostgresConnection impl (all traits)
└── registry.rs         # ConnectionRegistry for dynamic connector selection
```

---

## 8. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Breaking changes to existing code | Phase 1 is additive (new methods, deprecated old). Phase 2 deprecates, doesn't remove. |
| `SchemaSupport` default impl for `relation_exists` may be slow | Allow connectors to override; use catalog queries where available. |
| Postgres `get_schema` implementation is complex (building Arrow Schema from `information_schema`) | Start with a simple column-name-only schema; refine with data types later. |
| `ProfilingSupport::PlanData` is `String` — consumers must parse JSON | Document the expected format per connector; add a `PlanData` trait with `to_json()` if needed. |
| Connection registry adds indirection | Registry is only used at the CLI boundary; library code still uses generics. |
| `execute` returns `usize` which is misleading for DDL | Accept as-is for now; a future `QueryResult` trait would handle SELECT properly. |

---

## 9. Checklist: What Changes Where

| File | Changes |
|------|---------|
| `connectors.rs` | Split into `core.rs`, `schema_support.rs`, `explain_support.rs`, `profiling_support.rs`, `system_metrics.rs`, `relation_ops.rs`, `error.rs`, `registry.rs` |
| `connectors/duckdb.rs` | Re-implement all trait methods; add `SchemaSupport`, `ExplainSupport`, `ProfilingSupport`, `SystemMetrics`, `RelationOps` |
| `connectors/postgres.rs` | Re-implement all trait methods; **add `SchemaSupport`** (currently missing); add `ExplainSupport`; add `RelationOps` |
| `executor.rs` | Update `resolve_schemas` to use `SchemaSupport`; update `SimpleEngine::run` to use new trait methods |
| `opt/common.rs` | Update `build_opaque_context` to handle `SchemaSupport` |
| `opt/pushdown.rs` | Update `StubConnector` test impl to implement all new traits |
| `connections.rs` | Keep as-is for serialization; add registry integration |
| `dee-cli/src/run.rs` | Replace `match` with `ConnectionRegistry::connect()` |
| `dee-cli/src/opt.rs` | Replace `match` with `ConnectionRegistry::connect()` |

---
