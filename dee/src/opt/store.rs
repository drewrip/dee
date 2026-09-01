//! The metadata-database port an optimization keeps its state through.
//!
//! An optimization decides what to do on a DAG run by reading state it wrote
//! on earlier runs, and that state has to outlive the process. The store that
//! holds it belongs to `dee-server`, but the optimizations that need it live
//! here, so the library defines the port and the server implements it.
//!
//! The interface is deliberately SQL-and-JSON rather than a typed row API.
//! Every optimization owns its own tables, created by
//! [`Optimization::register`](crate::opt::Optimization::register), and no two
//! of them share a shape -- a typed API would have to grow a variant per pass
//! and would put the library's schema in the server's migrations. Rows come
//! back as JSON objects keyed by column name, which every backing store can
//! produce and which `serde_json` already knows how to take apart.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use duckdb::Connection;
use serde_json::Value;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum OptStoreError {
    #[error("metadata store: {0}")]
    Backend(String),
    /// The statement touched a table outside the optimization's namespace.
    /// See [`OptStore::execute`].
    #[error("optimization '{optimization}' may only touch tables named '{prefix}*', not '{table}'")]
    OutsideNamespace {
        optimization: String,
        prefix: String,
        table: String,
    },
    #[error("a row could not be decoded: {0}")]
    Decode(String),
}

/// Read/write access to the metadata database, scoped to one optimization's
/// own tables.
///
/// Implementations are handed out per optimization: the namespace an
/// [`execute`](OptStore::execute) is checked against is fixed when the handle
/// is built, so a pass cannot widen it by asking.
#[async_trait]
pub trait OptStore: Send + Sync {
    /// The table-name prefix this handle permits, e.g. `opt_hmp_`.
    fn table_prefix(&self) -> &str;

    /// Run a statement that changes something -- DDL from `register`, or an
    /// insert/update from `step`. Returns rows affected.
    ///
    /// Every table the statement names must start with
    /// [`table_prefix`](OptStore::table_prefix). An optimization is arbitrary
    /// code writing arbitrary SQL against the database that also holds every
    /// run, plan and connection credential dee has ever recorded, and nothing
    /// about a materialization search requires reaching any of it.
    async fn execute(&self, sql: &str, params: &[Value]) -> Result<usize, OptStoreError>;

    /// Run a query, one JSON object per row keyed by column name. Subject to
    /// the same namespace rule as [`execute`](OptStore::execute).
    async fn query(&self, sql: &str, params: &[Value]) -> Result<Vec<Value>, OptStoreError>;
}

impl dyn OptStore {
    /// The first row of `sql`, or `None` when it returned nothing.
    pub async fn query_one(
        &self,
        sql: &str,
        params: &[Value],
    ) -> Result<Option<Value>, OptStoreError> {
        Ok(self.query(sql, params).await?.into_iter().next())
    }
}

/// The tables an [`Optimization`](crate::opt::Optimization) created for one
/// DAG.
///
/// Returned by `register` so the server can record what exists, and by
/// `deregister` so it can confirm what went away. An optimization that keeps
/// no state -- Pushdown -- returns `None` from both rather than an empty
/// `Registration`, so "nothing to set up" stays distinguishable from "set up
/// nothing".
#[derive(Clone, Debug, PartialEq)]
pub struct Registration {
    pub tables: Vec<String>,
}

impl Registration {
    pub fn new(tables: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            tables: tables.into_iter().map(Into::into).collect(),
        }
    }
}

/// The namespace prefix an optimization's tables must live under.
pub fn table_prefix(optimization: &str) -> String {
    format!("opt_{optimization}_")
}

/// Table names a statement refers to, for the namespace check.
///
/// This is a lexical scan for the identifier following `FROM`, `JOIN`,
/// `INTO`, `UPDATE`, `TABLE` or `DELETE FROM`, not a parse. It is
/// deliberately conservative: anything it cannot confidently read as a table
/// reference is still returned, so the check fails closed. Comments and
/// string literals are skipped so a table name inside one is not mistaken for
/// a reference.
pub fn referenced_tables(sql: &str) -> Vec<String> {
    let stripped = strip_literals(sql);
    let tokens: Vec<&str> = stripped.split_whitespace().collect();
    let mut tables = Vec::new();

    let mut i = 0;
    while i < tokens.len() {
        let keyword = tokens[i].to_ascii_uppercase();
        let names_a_table = matches!(
            keyword.as_str(),
            "FROM" | "JOIN" | "INTO" | "UPDATE" | "TABLE"
        );
        if names_a_table {
            // `CREATE TABLE IF NOT EXISTS x` and `DROP TABLE IF EXISTS x`
            // put three words between the keyword and the name.
            let mut j = i + 1;
            while j < tokens.len()
                && matches!(
                    tokens[j].to_ascii_uppercase().as_str(),
                    "IF" | "NOT" | "EXISTS"
                )
            {
                j += 1;
            }
            if let Some(name) = tokens.get(j) {
                let name = name.trim_matches(|c: char| !c.is_alphanumeric() && c != '_' && c != '.');
                if !name.is_empty() {
                    tables.push(name.to_ascii_lowercase());
                }
            }
        }
        i += 1;
    }

    tables.sort();
    tables.dedup();
    tables
}

/// Blank out string literals and comments so `referenced_tables` cannot be
/// fooled by `SELECT 'from runs'`.
fn strip_literals(sql: &str) -> String {
    let mut out = String::with_capacity(sql.len());
    let mut chars = sql.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\'' => {
                out.push(' ');
                while let Some(c) = chars.next() {
                    if c == '\'' {
                        if chars.peek() == Some(&'\'') {
                            chars.next();
                            continue;
                        }
                        break;
                    }
                }
                out.push(' ');
            }
            '-' if chars.peek() == Some(&'-') => {
                for c in chars.by_ref() {
                    if c == '\n' {
                        break;
                    }
                }
                out.push(' ');
            }
            '/' if chars.peek() == Some(&'*') => {
                chars.next();
                let mut prev = ' ';
                for c in chars.by_ref() {
                    if prev == '*' && c == '/' {
                        break;
                    }
                    prev = c;
                }
                out.push(' ');
            }
            // Punctuation that can abut an identifier becomes a separator, so
            // `from(runs)` and `from runs` tokenize the same.
            '(' | ')' | ',' | ';' => out.push(' '),
            _ => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_finds_the_table_a_statement_reads() {
        assert_eq!(referenced_tables("SELECT * FROM opt_hmp_state"), ["opt_hmp_state"]);
    }

    #[test]
    fn test_finds_tables_behind_create_and_drop_guards() {
        // `IF NOT EXISTS` sits between the keyword and the name, and a check
        // that stopped at the first word after `TABLE` would read "if" as the
        // table and let the real one through unchecked.
        assert_eq!(
            referenced_tables("CREATE TABLE IF NOT EXISTS opt_omp_state (a INT)"),
            ["opt_omp_state"]
        );
        assert_eq!(
            referenced_tables("DROP TABLE IF EXISTS opt_omp_state"),
            ["opt_omp_state"]
        );
    }

    #[test]
    fn test_finds_every_table_in_a_join() {
        assert_eq!(
            referenced_tables(
                "SELECT * FROM opt_hmp_state s JOIN opt_hmp_trials t ON s.dag_id = t.dag_id"
            ),
            ["opt_hmp_state", "opt_hmp_trials"]
        );
    }

    #[test]
    fn test_a_table_name_inside_a_literal_is_not_a_reference() {
        // Otherwise the namespace check could be tripped by data rather than
        // by an actual reference -- a false positive that would block a
        // legitimate write.
        assert_eq!(
            referenced_tables("INSERT INTO opt_hmp_trials (note) VALUES ('from runs')"),
            ["opt_hmp_trials"]
        );
    }

    #[test]
    fn test_a_comment_hides_nothing_and_names_nothing() {
        assert_eq!(
            referenced_tables("SELECT * FROM opt_hmp_state -- FROM runs\n"),
            ["opt_hmp_state"]
        );
        assert_eq!(
            referenced_tables("/* FROM connections */ SELECT * FROM opt_hmp_state"),
            ["opt_hmp_state"]
        );
    }

    #[test]
    fn test_a_reach_outside_the_namespace_is_visible() {
        // The point of the whole module: this is what `execute` refuses.
        let tables = referenced_tables("SELECT config FROM connections");
        assert_eq!(tables, ["connections"]);
        assert!(!tables.iter().all(|t| t.starts_with("opt_hmp_")));
    }

    #[test]
    fn test_prefix_is_derived_from_the_optimization_name() {
        assert_eq!(table_prefix("hmp"), "opt_hmp_");
    }
}

/// An [`OptStore`] that remembers nothing.
///
/// Used where there is no metadata database to reach. Writes succeed and are
/// discarded; reads return nothing. A continuous optimization against this
/// sees an empty state on every step, so it behaves as if each run were its
/// first -- which is the truthful reading of "nothing was remembered", and is
/// why the server never hands one out.
pub struct NullStore {
    prefix: String,
}

impl NullStore {
    pub fn new(optimization: &str) -> Self {
        Self {
            prefix: table_prefix(optimization),
        }
    }
}

#[async_trait]
impl OptStore for NullStore {
    fn table_prefix(&self) -> &str {
        &self.prefix
    }

    async fn execute(&self, _sql: &str, _params: &[Value]) -> Result<usize, OptStoreError> {
        Ok(0)
    }

    async fn query(&self, _sql: &str, _params: &[Value]) -> Result<Vec<Value>, OptStoreError> {
        Ok(Vec::new())
    }
}


// ---------------------------------------------------------------------------
// The DuckDB bridge
//
// The metadata database is DuckDB, and both the server's store and the
// in-process one below have to turn an optimization's SQL into rows of JSON
// the same way. Doing it once here keeps the namespace check, the parameter
// binding and the row encoding from drifting apart between the two.
// ---------------------------------------------------------------------------

/// Refuse a statement that touches anything outside `prefix`.
pub fn check_namespace(
    optimization: &str,
    prefix: &str,
    sql: &str,
) -> Result<(), OptStoreError> {
    for table in referenced_tables(sql) {
        if !table.starts_with(prefix) {
            return Err(OptStoreError::OutsideNamespace {
                optimization: optimization.to_string(),
                prefix: prefix.to_string(),
                table,
            });
        }
    }
    Ok(())
}

/// Bind a JSON value as a DuckDB parameter.
///
/// Only the shapes an optimization's state actually uses. A nested object or
/// array arrives as its JSON text, which is how the passes store their search
/// state in the first place.
fn to_duck(value: &Value) -> duckdb::types::Value {
    use duckdb::types::Value as D;
    match value {
        Value::Null => D::Null,
        Value::Bool(b) => D::Boolean(*b),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                D::BigInt(i)
            } else {
                D::Double(n.as_f64().unwrap_or(f64::NAN))
            }
        }
        Value::String(s) => D::Text(s.clone()),
        other => D::Text(other.to_string()),
    }
}

/// Run a statement, after checking it stays inside `prefix`.
pub fn execute_on(
    conn: &Connection,
    optimization: &str,
    prefix: &str,
    sql: &str,
    params: &[Value],
) -> Result<usize, OptStoreError> {
    check_namespace(optimization, prefix, sql)?;
    let bound: Vec<duckdb::types::Value> = params.iter().map(to_duck).collect();
    let refs: Vec<&dyn duckdb::ToSql> = bound.iter().map(|v| v as &dyn duckdb::ToSql).collect();
    conn.execute(sql, refs.as_slice())
        .map_err(|e| OptStoreError::Backend(e.to_string()))
}

/// Run a query, one JSON object per row.
///
/// The statement is wrapped in `SELECT to_json(t) FROM (<sql>) t` so DuckDB
/// does the row-to-JSON conversion itself. That is what lets every
/// optimization keep its own table shape without the store needing to know
/// any of them.
pub fn query_on(
    conn: &Connection,
    optimization: &str,
    prefix: &str,
    sql: &str,
    params: &[Value],
) -> Result<Vec<Value>, OptStoreError> {
    check_namespace(optimization, prefix, sql)?;
    let wrapped = format!("SELECT to_json(t) AS row FROM ({sql}) t");
    let bound: Vec<duckdb::types::Value> = params.iter().map(to_duck).collect();
    let refs: Vec<&dyn duckdb::ToSql> = bound.iter().map(|v| v as &dyn duckdb::ToSql).collect();

    let mut stmt = conn
        .prepare(&wrapped)
        .map_err(|e| OptStoreError::Backend(e.to_string()))?;
    let rows = stmt
        .query_map(refs.as_slice(), |row| row.get::<_, String>(0))
        .map_err(|e| OptStoreError::Backend(e.to_string()))?;

    let mut out = Vec::new();
    for row in rows {
        let text = row.map_err(|e| OptStoreError::Backend(e.to_string()))?;
        out.push(
            serde_json::from_str(&text).map_err(|e| OptStoreError::Decode(e.to_string()))?,
        );
    }
    Ok(out)
}

/// Whether an error means "that table is not there".
///
/// An optimization's tables exist only while it is registered, and a step can
/// race a deregistration -- a run already in flight when the optimization is
/// removed will still take its turn. Reading that as "no state", rather than
/// as a failure, is what keeps removing an optimization from failing the run
/// that happened to be underway. The backend reports it in the error text
/// rather than as a distinct kind, so this matches on the message; being wrong
/// costs a step that does nothing.
pub fn is_missing_table(error: &OptStoreError) -> bool {
    let OptStoreError::Backend(message) = error else {
        return false;
    };
    let message = message.to_ascii_lowercase();
    message.contains("does not exist") || message.contains("no such table")
}

/// An [`OptStore`] over an in-process DuckDB database.
///
/// The server hands out its own, backed by the metadata database. This one is
/// for tests and for tooling that has no server: it is a real store with real
/// tables, so an optimization stepped against it behaves exactly as it would
/// in production, only without outliving the process.
pub struct MemoryStore {
    conn: Arc<Mutex<Connection>>,
    optimization: String,
    prefix: String,
}

impl MemoryStore {
    /// A store over a fresh in-memory database.
    pub fn open(optimization: &str) -> Result<Self, OptStoreError> {
        let conn = Connection::open_in_memory()
            .map_err(|e| OptStoreError::Backend(e.to_string()))?;
        Ok(Self::over(Arc::new(Mutex::new(conn)), optimization))
    }

    /// A store over an existing database, so several optimizations can share
    /// one -- as they do in the metadata database.
    pub fn over(conn: Arc<Mutex<Connection>>, optimization: &str) -> Self {
        Self {
            conn,
            optimization: optimization.to_string(),
            prefix: table_prefix(optimization),
        }
    }

    pub fn database(&self) -> Arc<Mutex<Connection>> {
        Arc::clone(&self.conn)
    }
}

#[async_trait]
impl OptStore for MemoryStore {
    fn table_prefix(&self) -> &str {
        &self.prefix
    }

    async fn execute(&self, sql: &str, params: &[Value]) -> Result<usize, OptStoreError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OptStoreError::Backend(format!("store lock poisoned: {e}")))?;
        execute_on(&conn, &self.optimization, &self.prefix, sql, params)
    }

    async fn query(&self, sql: &str, params: &[Value]) -> Result<Vec<Value>, OptStoreError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OptStoreError::Backend(format!("store lock poisoned: {e}")))?;
        query_on(&conn, &self.optimization, &self.prefix, sql, params)
    }
}

/// An [`OptStoreFactory`](crate::opt::OptStoreFactory) over one in-memory
/// database shared by every optimization, each scoped to its own namespace.
pub struct MemoryStoreFactory {
    conn: Arc<Mutex<Connection>>,
}

impl MemoryStoreFactory {
    pub fn open() -> Result<Self, OptStoreError> {
        let conn = Connection::open_in_memory()
            .map_err(|e| OptStoreError::Backend(e.to_string()))?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }
}

impl crate::opt::OptStoreFactory for MemoryStoreFactory {
    fn store_for(&self, optimization: &str) -> Arc<dyn OptStore> {
        Arc::new(MemoryStore::over(Arc::clone(&self.conn), optimization))
    }
}
