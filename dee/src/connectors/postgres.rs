use crate::{
    connectors::{Connector, ConnectorError, PushdownInfo},
    dag::MaterializeMode,
    plan::{PlanNode, TimeBasis, parse_postgres_plan},
};
use async_trait::async_trait;
use duckdb::arrow::datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit};
use log::{debug, trace};
use serde::{Deserialize, Serialize};
use sqlx::{
    Column, ConnectOptions, Executor, PgPool, Row, Statement, TypeInfo,
    postgres::{PgConnectOptions, PgPoolOptions},
};
use std::{collections::HashMap, sync::Arc, time::Duration};

#[derive(Serialize, Deserialize, Clone)]
pub struct PostgresConfig {
    host: String,
    port: Option<i32>,
    user: String,
    password: String,
    database: String,
    num_connections: Option<u32>,
}

impl PostgresConfig {}

pub struct PostgresConnection {
    pool: PgPool,
}

fn materialize_mode_in_pg(mode: MaterializeMode) -> String {
    match mode {
        MaterializeMode::Table | MaterializeMode::TempTable => "TABLE".to_string(),
        MaterializeMode::View => "VIEW".to_string(),
    }
}

impl PostgresConnection {
    /// Run an EXPLAIN that was asked for `FORMAT JSON` and return its text.
    ///
    /// Postgres returns the document as a single row of a single column.
    async fn explain_json(&self, sql: &str) -> Result<String, ConnectorError> {
        let row = sqlx::query(sql).fetch_one(&self.pool).await.map_err(|e| {
            ConnectorError::Execute(format!("{} - query_text:\n{}", e, sql))
        })?;
        // `EXPLAIN (FORMAT JSON)` comes back as a `json`-typed column, so it
        // has to be decoded as JSON and re-serialized. Older servers and the
        // TEXT format return text, so both are accepted.
        if let Ok(value) = row.try_get::<serde_json::Value, _>(0) {
            return Ok(value.to_string());
        }
        row.try_get::<String, _>(0).map_err(|e| {
            ConnectorError::Execute(format!("Failed to read explain output: {}", e))
        })
    }
}

/// Map a Postgres type name onto an Arrow type.
///
/// The optimizer uses schemas to know which columns a node produces, so
/// column identity is what matters here; an unrecognized type falls back to
/// `Utf8` rather than failing, since losing a column entirely would silently
/// change what pushdown prunes.
fn pg_type_to_arrow(name: &str) -> DataType {
    match name.to_ascii_uppercase().as_str() {
        "BOOL" => DataType::Boolean,
        "INT2" | "SMALLINT" => DataType::Int16,
        "INT4" | "INT" | "INTEGER" | "SERIAL" => DataType::Int32,
        "INT8" | "BIGINT" | "BIGSERIAL" => DataType::Int64,
        "FLOAT4" | "REAL" => DataType::Float32,
        "FLOAT8" | "DOUBLE PRECISION" => DataType::Float64,
        // Postgres NUMERIC is arbitrary precision; Arrow's widest fixed
        // decimal is the closest faithful representation.
        "NUMERIC" | "DECIMAL" => DataType::Decimal128(38, 10),
        "DATE" => DataType::Date32,
        "TIMESTAMP" => DataType::Timestamp(TimeUnit::Microsecond, None),
        "TIMESTAMPTZ" => DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
        "TIME" => DataType::Time64(TimeUnit::Microsecond),
        "BYTEA" => DataType::Binary,
        "UUID" | "TEXT" | "VARCHAR" | "BPCHAR" | "CHAR" | "NAME" | "JSON" | "JSONB" => {
            DataType::Utf8
        }
        _ => DataType::Utf8,
    }
}

/// Rows written by an `EXPLAIN ANALYZE CREATE TABLE ... AS`, read off the plan.
///
/// The root node's actual row count is the number of rows the statement
/// produced, which for a CTAS is the number of rows stored.
fn rows_written_from_pg_plan(json: &str) -> Option<usize> {
    let plans = parse_postgres_plan(json)?;
    plans.first()?.cardinality.map(|c| c as usize)
}

/// Whether an expression refers to something only the running plan can supply.
///
/// Postgres writes runtime artifacts into scan predicates and output lists:
/// `(InitPlan 1).col1` for a subquery evaluated once, `(SubPlan 2)` for a
/// correlated one, and `$1` for a parameter. None of these are reproducible as
/// standalone SQL, so a predicate mentioning one cannot be pushed into another
/// query -- it has to be left where Postgres computes it.
fn references_runtime_plan(expr: &str) -> bool {
    expr.contains("InitPlan")
        || expr.contains("SubPlan")
        || expr.contains("$0")
        || expr.contains("$1")
        || expr.contains("$2")
}

/// Whether `expr` is a plain column reference rather than a computed one.
///
/// `Output` lists arbitrary expressions, but the pushdown pass uses
/// projections as a set of column names to prune by, so anything that is not
/// a bare identifier is dropped rather than misinterpreted as a column.
fn is_bare_column(expr: &str) -> bool {
    !expr.is_empty()
        && expr
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '"')
}

/// Strip a leading relation qualifier from a column reference.
///
/// Postgres reports `Output` and filter expressions fully qualified
/// (`readings.temperature_c`), while the pushdown pass rewrites queries where
/// that alias may not exist. Dropping the qualifier matches what DuckDB
/// reports and keeps the two backends' pushdown info interchangeable.
fn strip_qualifier(expr: &str, relation: &str) -> String {
    let prefix = format!("{}.", relation);
    expr.replace(&prefix, "")
}

/// Record the projections and filters Postgres pushed into each scan.
///
/// Any node naming a `Relation Name` is a scan; `Output` is the set of columns
/// it must produce and `Filter` / `Index Cond` / `Recheck Cond` are the
/// predicates evaluated at the scan itself. Each scan is reported separately
/// (see [`PushdownInfo`]) -- one query can scan the same relation several
/// times with entirely different predicates.
fn collect_pg_scan_pushdowns(
    node: &serde_json::Value,
    out: &mut HashMap<String, Vec<PushdownInfo>>,
) {
    if let Some(relation) = node.get("Relation Name").and_then(|v| v.as_str()) {
        let alias = node
            .get("Alias")
            .and_then(|v| v.as_str())
            .unwrap_or(relation);
        let mut info = PushdownInfo::default();

        if let Some(outputs) = node.get("Output").and_then(|v| v.as_array()) {
            for o in outputs.iter().filter_map(|v| v.as_str()) {
                let col = strip_qualifier(o, alias);
                if is_bare_column(&col) && !info.projections.contains(&col) {
                    info.projections.push(col);
                }
            }
        }
        for key in ["Filter", "Index Cond", "Recheck Cond"] {
            if let Some(f) = node.get(key).and_then(|v| v.as_str()) {
                if references_runtime_plan(f) {
                    trace!("skipping non-reproducible predicate: {}", f);
                    continue;
                }
                let pred = strip_qualifier(f, alias);
                if !info.filters.contains(&pred) {
                    info.filters.push(pred);
                }
            }
        }
        out.entry(relation.to_string()).or_default().push(info);
    }
    if let Some(children) = node.get("Plans").and_then(|v| v.as_array()) {
        for child in children {
            collect_pg_scan_pushdowns(child, out);
        }
    }
}

#[async_trait]
impl Connector for PostgresConnection {
    type Config = PostgresConfig;
    type Connection = PostgresConnection;

    /// `max_parallel_workers`: the server-wide pool every backend draws its
    /// parallel workers from. The per-gather limit bounds one query; this
    /// bounds the engine, which is what a second concurrent node contends for.
    async fn parallelism_budget(&self) -> Result<Option<usize>, ConnectorError> {
        let row: (String,) = sqlx::query_as("SHOW max_parallel_workers")
            .fetch_one(&self.pool)
            .await
            .map_err(|e| ConnectorError::Execute(format!("reading max_parallel_workers - {e}")))?;
        // Workers are additional to the backend that requested them, so the
        // engine can keep one more thing busy than it has workers.
        Ok(row.0.trim().parse::<usize>().ok().map(|w| w + 1))
    }

    async fn new(config: Self::Config) -> Result<Arc<Self::Connection>, ConnectorError> {
        let conn_options = PgConnectOptions::new_without_pgpass()
            .host(&config.host)
            .port(config.port.unwrap_or(5432) as u16)
            .username(&config.user)
            .password(&config.password)
            .database(&config.database)
            .log_slow_statements(log::LevelFilter::Off, Duration::from_hours(2));

        let pool = PgPoolOptions::new()
            .max_connections(config.num_connections.unwrap_or(4))
            .connect_with(conn_options)
            .await
            .map_err(|_| ConnectorError::Create("couldn't create PgPool".into()))?;
        let pg_conn = PostgresConnection { pool };
        Ok(Arc::new(pg_conn))
    }

    async fn execute(&self, query_text: String) -> Result<usize, ConnectorError> {
        let mut conn = self.pool.acquire().await.map_err(|e| {
            ConnectorError::Execute(format!("couldn't retrieve connection from pool - {}", e))
        })?;
        let temp_q: &str = &query_text;
        let rows = conn
            .execute(temp_q)
            .await
            .map_err(|e| ConnectorError::Execute(format!("couldn't execute SQL - {}", e)))?;
        Ok(rows.rows_affected() as usize)
    }

    async fn new_relation(
        &self,
        relation_type: MaterializeMode,
        name: String,
        query_text: String,
    ) -> Result<usize, ConnectorError> {
        let ddl_text = match relation_type {
            MaterializeMode::View => format!("CREATE OR REPLACE VIEW {} AS ({})", name, query_text),
            MaterializeMode::Table | MaterializeMode::TempTable => {
                format!("CREATE TABLE {} AS ({})", name, query_text)
            }
        };
        self.execute(ddl_text).await
    }

    async fn drop_relation(
        &self,
        relation_type: MaterializeMode,
        name: String,
    ) -> Result<usize, ConnectorError> {
        let text_rel_type = materialize_mode_in_pg(relation_type);
        let ddl_text = format!("DROP {} IF EXISTS {} CASCADE", text_rel_type, name);
        self.execute(ddl_text).await
    }

    async fn get_schema(&self, name: String) -> Option<Result<SchemaRef, ConnectorError>> {
        trace!("attempt to fetch arrow schema for {}", name);
        // Prepare `SELECT * ... LIMIT 0` and read the column metadata Postgres
        // returns for the prepared statement. This needs no parsing of the
        // (possibly quoted, possibly three-part) relation name, and never
        // transfers a row.
        let sql = format!("SELECT * FROM {} LIMIT 0", name);
        let stmt = match self.pool.prepare(&sql).await {
            Ok(s) => s,
            Err(e) => {
                return Some(Err(ConnectorError::Execute(format!(
                    "couldn't prepare schema query for {name}: {e}"
                ))));
            }
        };
        let fields: Vec<Field> = stmt
            .columns()
            .iter()
            .map(|c| Field::new(c.name(), pg_type_to_arrow(c.type_info().name()), true))
            .collect();
        Some(Ok(Arc::new(Schema::new(fields))))
    }

    async fn new_relation_and_explain(
        &self,
        relation_type: MaterializeMode,
        name: String,
        query_text: String,
    ) -> Result<(usize, Option<String>), ConnectorError> {
        let rel_type = materialize_mode_in_pg(relation_type);
        match relation_type {
            MaterializeMode::View => {
                // A view materializes nothing, so there is no execution to
                // analyze; the un-executed plan is what the optimizer traces
                // operators through.
                let plan = self
                    .explain_json(&format!("EXPLAIN (VERBOSE, FORMAT JSON) {}", query_text))
                    .await?;
                let ddl = format!("CREATE OR REPLACE {} {} AS ({})", rel_type, name, query_text);
                let rows = self.execute(ddl).await?;
                Ok((rows, Some(plan)))
            }
            MaterializeMode::Table | MaterializeMode::TempTable => {
                // EXPLAIN ANALYZE on a CTAS both creates the relation and
                // reports what it actually cost, so one statement does the work
                // and the measurement.
                let ddl = format!("CREATE {} {} AS ({})", rel_type, name, query_text);
                let plan = self
                    .explain_json(&format!(
                        "EXPLAIN (ANALYZE, VERBOSE, BUFFERS, FORMAT JSON) {}",
                        ddl
                    ))
                    .await?;
                let rows = rows_written_from_pg_plan(&plan).unwrap_or(0);
                Ok((rows, Some(plan)))
            }
        }
    }

    async fn pushdown(
        &self,
        query_text: &str,
    ) -> Result<Option<HashMap<String, Vec<PushdownInfo>>>, ConnectorError> {
        let json = self
            .explain_json(&format!("EXPLAIN (VERBOSE, FORMAT JSON) {}", query_text))
            .await?;
        let wrappers: serde_json::Value = serde_json::from_str(&json).map_err(|e| {
            ConnectorError::Execute(format!("Failed to parse explain JSON: {e} - json:\n{json}"))
        })?;

        let mut out: HashMap<String, Vec<PushdownInfo>> = HashMap::new();
        if let Some(items) = wrappers.as_array() {
            for item in items {
                if let Some(plan) = item.get("Plan") {
                    collect_pg_scan_pushdowns(plan, &mut out);
                }
            }
        }
        debug!("postgres pushdown: {} relation(s) analyzed", out.len());
        Ok(Some(out))
    }

    fn parse_plan(&self, json: &str) -> Option<Vec<PlanNode>> {
        parse_postgres_plan(json)
    }

    fn time_basis(&self) -> TimeBasis {
        // Postgres reports `Actual Total Time`, which is wall clock, not the
        // CPU time DuckDB reports.
        TimeBasis::WallTime
    }

    async fn sample_system_memory_usage(&self) -> Result<Option<u64>, ConnectorError> {
        let row = sqlx::query(
            "SELECT COALESCE(SUM(total_bytes), 0)::BIGINT AS memory_bytes FROM pg_backend_memory_contexts",
        )
        .fetch_one(&self.pool)
        .await
        .map_err(|e| ConnectorError::Execute(format!("Failed to sample memory usage: {}", e)))?;

        let memory_bytes: i64 = row.try_get("memory_bytes").map_err(|e| {
            ConnectorError::Execute(format!("Failed to decode memory usage sample: {}", e))
        })?;

        Ok(Some(memory_bytes.max(0) as u64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scan_pushdown_collects_projections_and_filters() {
        let plan: serde_json::Value = serde_json::from_str(
            r#"{"Node Type":"Seq Scan","Relation Name":"readings","Alias":"readings",
                "Output":["readings.device_id","readings.temperature_c"],
                "Filter":"(readings.temperature_c > 20)","Plans":[]}"#,
        )
        .unwrap();
        let mut out = HashMap::new();
        collect_pg_scan_pushdowns(&plan, &mut out);
        let info = &out["readings"][0];
        // Qualifiers are stripped so the info matches what DuckDB reports and
        // the pushdown pass can rewrite with it.
        assert_eq!(info.projections, vec!["device_id", "temperature_c"]);
        assert_eq!(info.filters, vec!["(temperature_c > 20)"]);
    }

    #[test]
    fn scan_pushdown_uses_the_alias_not_the_relation_when_they_differ() {
        let plan: serde_json::Value = serde_json::from_str(
            r#"{"Node Type":"Seq Scan","Relation Name":"readings","Alias":"r",
                "Output":["r.device_id"],"Filter":"(r.battery_pct < 15)","Plans":[]}"#,
        )
        .unwrap();
        let mut out = HashMap::new();
        collect_pg_scan_pushdowns(&plan, &mut out);
        assert_eq!(out["readings"][0].projections, vec!["device_id"]);
        assert_eq!(out["readings"][0].filters, vec!["(battery_pct < 15)"]);
    }

    #[test]
    fn scan_pushdown_descends_into_child_plans() {
        let plan: serde_json::Value = serde_json::from_str(
            r#"{"Node Type":"Hash Join","Plans":[
                {"Node Type":"Seq Scan","Relation Name":"a","Output":["a.x"],"Plans":[]},
                {"Node Type":"Seq Scan","Relation Name":"b","Output":["b.y"],"Plans":[]}]}"#,
        )
        .unwrap();
        let mut out = HashMap::new();
        collect_pg_scan_pushdowns(&plan, &mut out);
        assert_eq!(out.len(), 2);
        assert_eq!(out["a"][0].projections, vec!["x"]);
        assert_eq!(out["b"][0].projections, vec!["y"]);
    }

    // Two scans of the same relation are two independent sets of predicates:
    // merging them would turn alternatives into a conjunction.
    #[test]
    fn scan_pushdown_keeps_each_scan_of_a_relation_separate() {
        let plan: serde_json::Value = serde_json::from_str(
            r#"{"Node Type":"Hash Join","Plans":[
                {"Node Type":"Seq Scan","Relation Name":"t","Alias":"x","Output":["x.a"],
                 "Filter":"(x.a < 10)","Plans":[]},
                {"Node Type":"Seq Scan","Relation Name":"t","Alias":"y","Output":["y.a"],
                 "Filter":"(y.a > 90)","Plans":[]}]}"#,
        )
        .unwrap();
        let mut out = HashMap::new();
        collect_pg_scan_pushdowns(&plan, &mut out);
        assert_eq!(out["t"].len(), 2);
        assert_eq!(out["t"][0].filters, vec!["(a < 10)"]);
        assert_eq!(out["t"][1].filters, vec!["(a > 90)"]);
    }

    #[test]
    fn scan_pushdown_collects_index_conditions() {
        let plan: serde_json::Value = serde_json::from_str(
            r#"{"Node Type":"Index Scan","Relation Name":"t","Alias":"t","Output":["t.id"],
                "Index Cond":"(t.id = 42)","Plans":[]}"#,
        )
        .unwrap();
        let mut out = HashMap::new();
        collect_pg_scan_pushdowns(&plan, &mut out);
        assert_eq!(out["t"][0].filters, vec!["(id = 42)"]);
    }

    #[test]
    fn ctas_row_count_comes_from_the_plan_root() {
        let json = r#"[{"Plan":{"Node Type":"Seq Scan","Actual Rows":541,"Actual Loops":1,
            "Actual Total Time":12.0,"Plan Rows":500,"Plans":[]}}]"#;
        assert_eq!(rows_written_from_pg_plan(json), Some(541));
    }

    #[test]
    fn pg_types_map_onto_arrow() {
        assert_eq!(pg_type_to_arrow("INT4"), DataType::Int32);
        assert_eq!(pg_type_to_arrow("int8"), DataType::Int64);
        assert_eq!(pg_type_to_arrow("FLOAT8"), DataType::Float64);
        assert_eq!(pg_type_to_arrow("TEXT"), DataType::Utf8);
        assert_eq!(pg_type_to_arrow("BOOL"), DataType::Boolean);
        // An unknown type keeps the column rather than dropping it.
        assert_eq!(pg_type_to_arrow("SOMETHING_NEW"), DataType::Utf8);
    }
}

#[cfg(test)]
mod pushdown_guard_tests {
    use super::*;

    #[test]
    fn predicates_referencing_a_subplan_are_not_pushed() {
        // Postgres computes these while running; reproducing them as
        // standalone SQL yields `syntax error at or near "1"`.
        let plan: serde_json::Value = serde_json::from_str(
            r#"{"Node Type":"Seq Scan","Relation Name":"stats","Alias":"stats",
                "Output":["stats.ts_hour"],
                "Filter":"(ts_hour >= ((InitPlan 1).col1 - '30 days'::interval))","Plans":[]}"#,
        )
        .unwrap();
        let mut out = HashMap::new();
        collect_pg_scan_pushdowns(&plan, &mut out);
        assert!(out["stats"][0].filters.is_empty());
        // The projection is still usable even though the filter is not.
        assert_eq!(out["stats"][0].projections, vec!["ts_hour"]);
    }

    #[test]
    fn ordinary_predicates_are_still_pushed() {
        let plan: serde_json::Value = serde_json::from_str(
            r#"{"Node Type":"Seq Scan","Relation Name":"t","Alias":"t","Output":["t.a"],
                "Filter":"(t.a > 5)","Plans":[]}"#,
        )
        .unwrap();
        let mut out = HashMap::new();
        collect_pg_scan_pushdowns(&plan, &mut out);
        assert_eq!(out["t"][0].filters, vec!["(a > 5)"]);
    }

    #[test]
    fn computed_output_expressions_are_not_treated_as_columns() {
        let plan: serde_json::Value = serde_json::from_str(
            r#"{"Node Type":"Seq Scan","Relation Name":"t","Alias":"t",
                "Output":["t.a","(t.b + t.c)","date_trunc('hour'::text, t.ts)"],"Plans":[]}"#,
        )
        .unwrap();
        let mut out = HashMap::new();
        collect_pg_scan_pushdowns(&plan, &mut out);
        assert_eq!(out["t"][0].projections, vec!["a"]);
    }

    #[test]
    fn parameter_references_are_not_pushed() {
        let plan: serde_json::Value = serde_json::from_str(
            r#"{"Node Type":"Index Scan","Relation Name":"t","Alias":"t","Output":["t.a"],
                "Index Cond":"(t.id = $1)","Plans":[]}"#,
        )
        .unwrap();
        let mut out = HashMap::new();
        collect_pg_scan_pushdowns(&plan, &mut out);
        assert!(out["t"][0].filters.is_empty());
    }
}
