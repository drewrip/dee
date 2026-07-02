use crate::{
    connectors::{Connector, ConnectorError},
    dag::MaterializeMode,
};
use async_trait::async_trait;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::display::{DisplayAs, DisplayFormatType};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType, SchedulingType};
use datafusion::physical_plan::{ExecutionPlan, Partitioning, PlanProperties};
use duckdb::{Config, DuckdbConnectionManager, params};
use log::{info, trace};
use r2d2::Pool;
use serde::{Deserialize, Serialize};
use std::{path::PathBuf, process::Command, sync::Arc, time::Duration};
use tempfile;

#[derive(Serialize, Deserialize, Clone)]
pub struct DuckDBConfig {
    pub database: PathBuf,
    pub num_connections: u32,
    pub threads: Option<i64>,
    pub max_memory: Option<String>,
}

impl DuckDBConfig {
    pub fn new_from_path(path: String) -> Self {
        Self {
            database: PathBuf::from(path),
            num_connections: 1,
            threads: None,
            max_memory: None,
        }
    }

    pub fn with_num_connections(mut self, num_connections: u32) -> Self {
        self.num_connections = num_connections;
        self
    }

    pub fn with_threads(mut self, num_threads: i64) -> Self {
        self.threads = Some(num_threads);
        self
    }

    pub fn with_max_memory(mut self, mem_str: String) -> Self {
        self.max_memory = Some(mem_str);
        self
    }
}

pub struct DuckDBConnection {
    pub pool: Pool<DuckdbConnectionManager>,
}

// ---------------------------------------------------------------------------
// DuckDBPlan — a DataFusion ExecutionPlan built from DuckDB EXPLAIN JSON
// ---------------------------------------------------------------------------

/// A single node in a DuckDB execution plan tree.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct DuckDBPlanNode {
    name: String,
    extra_info: serde_json::Value,
    children: Vec<Arc<DuckDBPlanNode>>,
}

/// A DataFusion [`ExecutionPlan`] built from a DuckDB
/// `EXPLAIN (FORMAT JSON)` output.
#[derive(Debug)]
pub struct DuckDBPlan {
    root: Arc<DuckDBPlanNode>,
    schema: SchemaRef,
    cache: Arc<PlanProperties>,
    children: Vec<Arc<dyn ExecutionPlan>>,
}

impl DuckDBPlan {
    pub fn try_from_json(json_plan: &str, schema: SchemaRef) -> Option<Self> {
        let value: serde_json::Value = serde_json::from_str(json_plan).ok()?;
        
        // Handle both EXPLAIN (FORMAT JSON) arrays and profiling output objects
        let root_node = if let Some(arr) = value.as_array() {
            if arr.is_empty() {
                return None;
            }
            Arc::new(DuckDBPlanNode::from_json(&arr[0])?)
        } else if let Some(obj) = value.as_object() {
            // Profiling output format: use operator_name as the node name
            let name = obj.get("operator_name")
                .and_then(|v| v.as_str())
                .unwrap_or("UNKNOWN")
                .to_string();
            let extra_info = obj.get("extra_info").cloned().unwrap_or(serde_json::Value::Null);
            let children: Vec<Arc<DuckDBPlanNode>> = obj.get("children")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| DuckDBPlanNode::from_profiling_json(v))
                        .map(Arc::new)
                        .collect()
                })
                .unwrap_or_default();
            Arc::new(DuckDBPlanNode { name, extra_info, children })
        } else {
            return None;
        };
        
        let (root, children) = Self::build_tree(&root_node, &schema);
        let cache = Self::compute_properties(Arc::clone(&schema), 1);
        Some(Self {
            root,
            schema,
            cache: Arc::new(cache),
            children,
        })
    }

    pub fn root_name(&self) -> &str {
        &self.root.name
    }

    fn build_tree(
        node: &Arc<DuckDBPlanNode>,
        schema: &SchemaRef,
    ) -> (Arc<DuckDBPlanNode>, Vec<Arc<dyn ExecutionPlan>>) {
        let mut children = Vec::new();
        for child_node in &node.children {
            let (child_root, child_children) = Self::build_tree(child_node, schema);
            let child_cache = Self::compute_properties(Arc::clone(schema), 1);
            let child_plan: Arc<dyn ExecutionPlan> = Arc::new(DuckDBPlan {
                root: child_root,
                schema: Arc::clone(schema),
                cache: Arc::new(child_cache),
                children: child_children,
            });
            children.push(child_plan);
        }
        (Arc::clone(node), children)
    }

    fn compute_properties(schema: SchemaRef, n_partitions: usize) -> PlanProperties {
        PlanProperties::new(
            EquivalenceProperties::new(schema),
            Partitioning::UnknownPartitioning(n_partitions),
            EmissionType::Incremental,
            Boundedness::Bounded,
        )
        .with_scheduling_type(SchedulingType::Cooperative)
    }
}

impl DuckDBPlanNode {
    fn from_json(value: &serde_json::Value) -> Option<Self> {
        let name = value
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("UNKNOWN")
            .to_string();
        let extra_info = value.get("extra_info").cloned().unwrap_or(serde_json::Value::Null);
        let children: Vec<Arc<DuckDBPlanNode>> = value
            .get("children")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| DuckDBPlanNode::from_json(v))
                    .map(Arc::new)
                    .collect()
            })
            .unwrap_or_default();
        Some(Self {
            name,
            extra_info,
            children,
        })
    }

    fn from_profiling_json(value: &serde_json::Value) -> Option<Self> {
        let name = value
            .get("operator_name")
            .and_then(|v| v.as_str())
            .and_then(|n| if n.is_empty() { None } else { Some(n) })
            .or_else(|| value.get("operator_type").and_then(|v| v.as_str()))
            .unwrap_or("UNKNOWN")
            .to_string();
        let extra_info = value.get("extra_info").cloned().unwrap_or(serde_json::Value::Null);
        let children: Vec<Arc<DuckDBPlanNode>> = value
            .get("children")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| DuckDBPlanNode::from_profiling_json(v))
                    .map(Arc::new)
                    .collect()
            })
            .unwrap_or_default();
        Some(Self {
            name,
            extra_info,
            children,
        })
    }
}

impl DisplayAs for DuckDBPlan {
    fn fmt_as(
        &self,
        t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "DuckDBPlan({})", self.root.name)
            }
            DisplayFormatType::TreeRender => {
                write!(f, "{}", self.root.name)
            }
        }
    }
}

impl std::fmt::Display for DuckDBPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.fmt_as(DisplayFormatType::Default, f)
    }
}

impl ExecutionPlan for DuckDBPlan {
    fn name(&self) -> &'static str {
        "DuckDBPlan"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        self.children.iter().collect()
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<datafusion::execution::TaskContext>,
    ) -> datafusion::common::Result<datafusion::execution::SendableRecordBatchStream> {
        use datafusion::physical_plan::memory::MemoryStream;
        use datafusion::arrow::record_batch::RecordBatch;
        let batch = RecordBatch::new_empty(Arc::clone(&self.schema));
        Ok(Box::pin(MemoryStream::try_new(
            vec![batch],
            Arc::clone(&self.schema),
            None,
        )?))
    }
}

fn materialize_mode_in_duckdb(mode: MaterializeMode) -> String {
    match mode {
        MaterializeMode::Table | MaterializeMode::TempTable => "TABLE".to_string(),
        MaterializeMode::View => "VIEW".to_string(),
    }
}

fn parse_duckdb_size_bytes(value: &str) -> Option<u64> {
    let trimmed = value.trim();
    let mut parts = trimmed.split_whitespace();
    let quantity = parts.next()?.parse::<f64>().ok()?;
    let unit = parts.next().unwrap_or("B").to_ascii_uppercase();
    let multiplier = match unit.as_str() {
        "B" => 1.0,
        "KB" | "KIB" => 1024.0,
        "MB" | "MIB" => 1024.0 * 1024.0,
        "GB" | "GIB" => 1024.0 * 1024.0 * 1024.0,
        "TB" | "TIB" => 1024.0 * 1024.0 * 1024.0 * 1024.0,
        _ => return None,
    };
    Some((quantity * multiplier).round() as u64)
}

fn sample_process_cpu_usage(pid: u32) -> Result<Option<f64>, ConnectorError> {
    let output = Command::new("ps")
        .args(["-o", "%cpu=", "-p", &pid.to_string()])
        .output()
        .map_err(|e| ConnectorError::Execute(format!("Failed to run ps for cpu usage: {}", e)))?;

    if !output.status.success() {
        return Err(ConnectorError::Execute(format!(
            "ps exited with status {} while sampling cpu usage",
            output.status
        )));
    }

    let stdout = String::from_utf8(output.stdout).map_err(|e| {
        ConnectorError::Execute(format!("Failed to decode ps cpu usage output: {}", e))
    })?;

    Ok(stdout.trim().parse::<f64>().ok())
}

#[async_trait]
impl Connector for DuckDBConnection {
    type Config = DuckDBConfig;
    type Connection = DuckDBConnection;

    async fn new(config: Self::Config) -> Result<Arc<Self::Connection>, ConnectorError> {
        let mut conf = Config::default();
        if let Some(max_mem) = config.max_memory {
            conf = conf
                .max_memory(&max_mem)
                .map_err(|_| ConnectorError::Create("set max memory problem".to_string()))?;
        }
        if let Some(threads) = config.threads {
            conf = conf
                .threads(threads)
                .map_err(|_| ConnectorError::Create("set threads problem".to_string()))?;
        }

        conf = conf
            .access_mode(duckdb::AccessMode::ReadWrite)
            .map_err(|_| ConnectorError::Create("set access_mode".to_string()))?;

        let manager = DuckdbConnectionManager::file_with_flags(config.database, conf)
            .map_err(|e| ConnectorError::Create(format!("connection manager - {}", e)))?;
        let pool = Pool::builder()
            .connection_timeout(Duration::from_hours(2))
            .max_size(config.num_connections)
            .build(manager)
            .map_err(|_| ConnectorError::Create("r2d2 pool".to_string()))?;

        {
            let conn = pool.get().map_err(|_| {
                ConnectorError::Create("couldn't get connection for ICU setup".to_string())
            })?;
            conn.execute_batch("INSTALL icu; LOAD icu;").map_err(|e| {
                ConnectorError::Create(format!("failed to install/load ICU: {}", e))
            })?;
        }

        Ok(Arc::new(Self { pool }))
    }

    async fn execute(&self, query_text: String) -> Result<usize, ConnectorError> {
        let conn = self
            .pool
            .get()
            .map_err(|_| ConnectorError::Execute("didn't get connection from pool".to_string()))?;
        conn.execute(&query_text.clone(), params![]).map_err(|e| {
            ConnectorError::Execute(format!("{} - query_text:\n{}", e.to_string(), query_text))
        })
    }

    async fn new_relation(
        &self,
        relation_type: MaterializeMode,
        name: String,
        query_text: String,
    ) -> Result<usize, ConnectorError> {
        let rel_type = materialize_mode_in_duckdb(relation_type);
        trace!("creating new_relation ({}, {})", rel_type, name);
        let tmpl_query = format!(
            "CREATE OR REPLACE {} {} AS ({})",
            rel_type, name, query_text
        );
        self.execute(tmpl_query).await
    }

    async fn new_relation_and_explain(
        &self,
        relation_type: MaterializeMode,
        name: String,
        query_text: String,
    ) -> Result<(usize, Option<String>), ConnectorError> {
        let conn = self
            .pool
            .get()
            .map_err(|_| ConnectorError::Execute("didn't get connection from pool".to_string()))?;

        match relation_type {
            MaterializeMode::View => {
                let explain_query = format!("EXPLAIN (FORMAT JSON) {}", query_text);
                let mut stmt = conn.prepare(&explain_query).map_err(|e| {
                    ConnectorError::Execute(format!("Failed to prepare explain: {}", e))
                })?;

                let json_str: String = stmt
                    .query_row([], |row| {
                        let col_count = row.as_ref().column_count();
                        if col_count >= 2 {
                            row.get(1)
                        } else {
                            row.get(0)
                        }
                    })
                    .map_err(|e| {
                        ConnectorError::Execute(format!("Failed to execute explain: {}", e))
                    })?;

                let rel_type = materialize_mode_in_duckdb(relation_type);
                let tmpl_query = format!("CREATE {} {} AS ({})", rel_type, name, query_text);
                let res = conn.execute(&tmpl_query, params![]).map_err(|e| {
                    ConnectorError::Execute(format!(
                        "{} - query_text:\n{}",
                        e.to_string(),
                        tmpl_query
                    ))
                })?;
                Ok((res, Some(json_str)))
            }
            MaterializeMode::Table | MaterializeMode::TempTable => {
                let temp_file = tempfile::Builder::new()
                    .suffix(".json")
                    .tempfile()
                    .map_err(|e| {
                        ConnectorError::Execute(format!("Failed to create temp file: {}", e))
                    })?;
                let temp_path = temp_file
                    .path()
                    .to_str()
                    .ok_or(ConnectorError::Execute("Invalid temp path".to_string()))?;

                conn.execute("SET enable_profiling = 'json';", [])
                    .map_err(|e| {
                        ConnectorError::Execute(format!("Failed to enable profiling: {}", e))
                    })?;
                conn.execute(&format!("SET profiling_output = '{}';", temp_path), [])
                    .map_err(|e| {
                        ConnectorError::Execute(format!("Failed to set profiling output: {}", e))
                    })?;

                let rel_type = materialize_mode_in_duckdb(relation_type);
                let tmpl_query = format!("CREATE {} {} AS ({})", rel_type, name, query_text);
                let res = conn.execute(&tmpl_query, params![]).map_err(|e| {
                    ConnectorError::Execute(format!(
                        "{} - query_text:\n{}",
                        e.to_string(),
                        tmpl_query
                    ))
                })?;

                conn.execute("RESET enable_profiling;", []).map_err(|e| {
                    ConnectorError::Execute(format!("Failed to disable profiling: {}", e))
                })?;
                conn.execute("RESET profiling_output;", []).map_err(|e| {
                    ConnectorError::Execute(format!("Failed to reset profiling output: {}", e))
                })?;

                let json_str = std::fs::read_to_string(temp_path).map_err(|e| {
                    ConnectorError::Execute(format!("Failed to read profiling output: {}", e))
                })?;

                Ok((res, Some(json_str)))
            }
        }
    }

    async fn drop_relation(
        &self,
        relation_type: MaterializeMode,
        name: String,
    ) -> Result<usize, ConnectorError> {
        let rel_type = materialize_mode_in_duckdb(relation_type);
        trace!("attempt drop_relation ({}, {})", rel_type, name);
        let tmpl_query = format!("DROP {} IF EXISTS {}", rel_type, name);
        self.execute(tmpl_query).await
    }

    async fn get_schema(&self, name: String) -> Option<Result<SchemaRef, ConnectorError>> {
        info!("attempt to fetch arrow schema for {}", name);
        let conn = match self.pool.get() {
            Ok(c) => c,
            Err(e) => {
                return Some(Err(ConnectorError::Execute(format!(
                    "couldn't get connection from pool: {e}"
                ))));
            }
        };
        let tmpl_query = format!("SELECT * FROM {} LIMIT 0", name);
        let mut stmt = match conn.prepare(&tmpl_query) {
            Ok(s) => s,
            Err(e) => {
                return Some(Err(ConnectorError::Execute(format!(
                    "couldn't prepare schema query for {name}: {e}"
                ))));
            }
        };
        match stmt.query_arrow([]) {
            Ok(arrow) => Some(Ok(arrow.get_schema())),
            Err(e) => Some(Err(ConnectorError::Execute(format!(
                "couldn't execute schema query for {name}: {e}"
            )))),
        }
    }

    async fn sample_system_memory_usage(&self) -> Result<Option<u64>, ConnectorError> {
        let conn = self
            .pool
            .get()
            .map_err(|_| ConnectorError::Execute("didn't get connection from pool".to_string()))?;

        let mut stmt = conn
            .prepare("SELECT memory_usage FROM pragma_database_size()")
            .map_err(|e| {
                ConnectorError::Execute(format!("Failed to prepare memory usage sample: {}", e))
            })?;

        let memory_usage: String = stmt
            .query_row([], |row| row.get(0))
            .map_err(|e| ConnectorError::Execute(format!("Failed to query memory usage: {}", e)))?;

        Ok(parse_duckdb_size_bytes(&memory_usage))
    }

    async fn sample_system_cpu_usage(&self) -> Result<Option<f64>, ConnectorError> {
        sample_process_cpu_usage(std::process::id())
    }

    async fn explain_to_logical_plan(
        &self,
        json_plan: &str,
        schema: SchemaRef,
    ) -> Option<Arc<dyn ExecutionPlan>> {
        DuckDBPlan::try_from_json(json_plan, schema).map(|p| Arc::new(p) as Arc<dyn ExecutionPlan>)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use std::fmt::Write;

    #[test]
    fn test_parse_duckdb_size_bytes() {
        assert_eq!(parse_duckdb_size_bytes("44.0 KiB"), Some(45056));
        assert_eq!(parse_duckdb_size_bytes("1.0 B"), Some(1));
    }

    #[test]
    fn test_sample_process_cpu_usage() {
        let cpu = sample_process_cpu_usage(std::process::id()).unwrap();
        assert!(cpu.is_some());
        assert!(cpu.unwrap() >= 0.0);
    }

    fn make_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
        ]))
    }

    #[test]
    fn test_parse_dummy_scan() {
        let json = r#"[{"name":"PROJECTION","children":[{"name":"DUMMY_SCAN","children":[],"extra_info":{}}],"extra_info":{"Projections":"1","Estimated Cardinality":"1"}}]"#;
        let plan = DuckDBPlan::try_from_json(json, make_schema()).expect("parse failed");
        assert_eq!(plan.root_name(), "PROJECTION");
        assert_eq!(plan.children.len(), 1);
    }

    #[test]
    fn test_parse_seq_scan_with_filter() {
        let json = r#"[{"name":"PROJECTION","children":[{"name":"SEQ_SCAN","children":[],"extra_info":{"Table":"memory.main.test_table","Type":"Sequential Scan","Projections":["id","name"],"Filters":"id>50","Estimated Cardinality":"20"}}],"extra_info":{"Projections":["id","name"]}}]"#;
        let plan = DuckDBPlan::try_from_json(json, make_schema()).expect("parse failed");
        assert_eq!(plan.root_name(), "PROJECTION");
        assert_eq!(plan.children.len(), 1);
    }

    #[test]
    fn test_parse_hash_join() {
        let json = r#"[{"name":"HASH_JOIN","children":[{"name":"SEQ_SCAN","children":[],"extra_info":{"Table":"memory.main.t1","Type":"Sequential Scan","Projections":["id","name"],"Filters":"id>50","Estimated Cardinality":"20"}},{"name":"SEQ_SCAN","children":[],"extra_info":{"Table":"memory.main.t2","Type":"Sequential Scan","Projections":["id","triple"],"Filters":"id>50","Estimated Cardinality":"20"}}],"extra_info":{"Join Type":"INNER","Conditions":"id = id","Estimated Cardinality":"4"}}]"#;
        let plan = DuckDBPlan::try_from_json(json, make_schema()).expect("parse failed");
        assert_eq!(plan.root_name(), "HASH_JOIN");
        assert_eq!(plan.children.len(), 2);
    }

    #[test]
    fn test_parse_hash_group_by() {
        let json = r##"[{"name":"PROJECTION","children":[{"name":"HASH_GROUP_BY","children":[{"name":"PROJECTION","children":[{"name":"SEQ_SCAN","children":[],"extra_info":{"Table":"memory.main.t1","Type":"Sequential Scan","Projections":["name","double_id"],"Estimated Cardinality":"100"}}],"extra_info":{"Projections":["name","double_id"]}}],"extra_info":{"Groups":"#0","Aggregates":["count_star()","sum_no_overflow(#1)"]}}],"extra_info":{"Projections":["__internal_decompress_string(#0)","#1","#2"]}}]"##;
        let plan = DuckDBPlan::try_from_json(json, make_schema()).expect("parse failed");
        assert_eq!(plan.root_name(), "PROJECTION");
        assert_eq!(plan.children.len(), 1);
    }

    #[test]
    fn test_parse_union() {
        let json = r#"[{"name":"UNION","children":[{"name":"SEQ_SCAN","children":[],"extra_info":{"Table":"memory.main.t1","Type":"Sequential Scan","Projections":["id","label"],"Estimated Cardinality":"10"}},{"name":"SEQ_SCAN","children":[],"extra_info":{"Table":"memory.main.t2","Type":"Sequential Scan","Projections":["id","label"],"Estimated Cardinality":"10"}}],"extra_info":{}}]"#;
        let plan = DuckDBPlan::try_from_json(json, make_schema()).expect("parse failed");
        assert_eq!(plan.root_name(), "UNION");
        assert_eq!(plan.children.len(), 2);
    }

    #[test]
    fn test_parse_order_by() {
        let json = r#"[{"name":"PROJECTION","children":[{"name":"ORDER_BY","children":[{"name":"SEQ_SCAN","children":[],"extra_info":{"Table":"memory.main.test_table","Type":"Sequential Scan","Projections":["id","name"],"Estimated Cardinality":"100"}}],"extra_info":{"Order By":"memory.main.test_table.id ASC"}}],"extra_info":{"Projections":["id","name"]}}]"#;
        let plan = DuckDBPlan::try_from_json(json, make_schema()).expect("parse failed");
        assert_eq!(plan.root_name(), "PROJECTION");
        assert_eq!(plan.children.len(), 1);
    }

    #[test]
    fn test_parse_window() {
        let json = r##"[{"name":"PROJECTION","children":[{"name":"WINDOW","children":[{"name":"SEQ_SCAN","children":[],"extra_info":{"Table":"memory.main.t1","Type":"Sequential Scan","Projections":["id","val"],"Estimated Cardinality":"100"}}],"extra_info":{"Projections":"sum(val) OVER (ORDER BY id ASC NULLS LAST)"}}],"extra_info":{"Projections":["#0","#1","#2"]}}]"##;
        let plan = DuckDBPlan::try_from_json(json, make_schema()).expect("parse failed");
        assert_eq!(plan.root_name(), "PROJECTION");
        assert_eq!(plan.children.len(), 1);
    }

    #[test]
    fn test_parse_nested_projections() {
        let json = r##"[{"name":"PROJECTION","children":[{"name":"PROJECTION","children":[{"name":"PROJECTION","children":[{"name":"SEQ_SCAN","children":[],"extra_info":{"Table":"memory.main.t1","Type":"Sequential Scan","Projections":["name","double_id"],"Estimated Cardinality":"100"}}],"extra_info":{"Projections":["__internal_compress_string_uhugeint(#0)","#1"]}}],"extra_info":{"Projections":["name","double_id"]}}],"extra_info":{"Projections":["__internal_decompress_string(#0)","#1","#2"]}}]"##;
        let plan = DuckDBPlan::try_from_json(json, make_schema()).expect("parse failed");
        assert_eq!(plan.root_name(), "PROJECTION");
        assert_eq!(plan.children.len(), 1);
        let child = plan.children[0].as_any().downcast_ref::<DuckDBPlan>().unwrap();
        assert_eq!(child.root_name(), "PROJECTION");
        assert_eq!(child.children.len(), 1);
        let grandchild = child.children[0].as_any().downcast_ref::<DuckDBPlan>().unwrap();
        assert_eq!(grandchild.root_name(), "PROJECTION");
        assert_eq!(grandchild.children.len(), 1);
        assert_eq!(grandchild.children[0].as_any().downcast_ref::<DuckDBPlan>().unwrap().root_name(), "SEQ_SCAN");
    }

    #[test]
    fn test_parse_invalid_json_returns_none() {
        let plan = DuckDBPlan::try_from_json("not json", make_schema());
        assert!(plan.is_none());
    }

    #[test]
    fn test_parse_empty_array_returns_none() {
        let plan = DuckDBPlan::try_from_json("[]", make_schema());
        assert!(plan.is_none());
    }

    #[test]
    fn test_execution_plan_properties() {
        let json = r#"[{"name":"SEQ_SCAN","children":[],"extra_info":{"Table":"memory.main.t1"}}]"#;
        let plan = DuckDBPlan::try_from_json(json, make_schema()).expect("parse failed");
        assert!(plan.schema().fields().len() >= 2);
    }

    #[test]
    fn test_execution_plan_children() {
        let json = r#"[{"name":"HASH_JOIN","children":[{"name":"SEQ_SCAN","children":[],"extra_info":{}},{"name":"SEQ_SCAN","children":[],"extra_info":{}}],"extra_info":{}}]"#;
        let plan = DuckDBPlan::try_from_json(json, make_schema()).expect("parse failed");
        let children = plan.children();
        assert!(children.len() >= 1);
    }

    #[test]
    fn test_execution_plan_display() {
        let json = r#"[{"name":"SEQ_SCAN","children":[],"extra_info":{}}]"#;
        let plan = DuckDBPlan::try_from_json(json, make_schema()).expect("parse failed");
        let mut s = String::new();
        write!(s, "{}", plan).unwrap();
        assert!(s.contains("DuckDBPlan"));
    }

    #[test]
    fn test_execution_plan_schema() {
        let json = r#"[{"name":"SEQ_SCAN","children":[],"extra_info":{}}]"#;
        let schema = make_schema();
        let plan = DuckDBPlan::try_from_json(json, schema).expect("parse failed");
        assert!(plan.schema().fields().len() >= 2);
        assert_eq!(plan.schema().field(0).name(), "id");
        assert_eq!(plan.schema().field(1).name(), "name");
    }

    #[test]
    fn test_execution_plan_with_new_children() {
        let json = r#"[{"name":"SEQ_SCAN","children":[],"extra_info":{}}]"#;
        let plan = DuckDBPlan::try_from_json(json, make_schema()).expect("parse failed");
        let arc = Arc::new(plan);
        let result = arc.with_new_children(vec![]);
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_explain_to_logical_plan_integration() {
        let config = DuckDBConfig::new_from_path(":memory:".to_string());
        let conn = DuckDBConnection::new(config).await.unwrap();

        conn.execute(
            "CREATE TABLE test_people AS SELECT generate_series AS id, 'name_' || cast(generate_series AS VARCHAR) AS name, generate_series * 2 AS double_id FROM generate_series(1, 100)".to_string(),
        ).await.unwrap();

        let schema = conn.get_schema("test_people".to_string()).await.unwrap().unwrap();

        let (rows, explain_json) = conn
            .new_relation_and_explain(
                MaterializeMode::TempTable,
                "explain_test".to_string(),
                "SELECT id, name FROM test_people WHERE id > 50 ORDER BY id".to_string(),
            )
            .await
            .unwrap();

        assert_eq!(rows, 0);
        assert!(explain_json.is_some());

        let plan = conn
            .explain_to_logical_plan(&explain_json.unwrap(), schema)
            .await
            .expect("should convert explain to ExecutionPlan");

        assert!(plan.schema().fields().len() >= 2);
        let children = plan.children();
        assert!(!children.is_empty());
    }

    #[tokio::test]
    async fn test_explain_to_logical_plan_join() {
        let config = DuckDBConfig::new_from_path(":memory:".to_string());
        let conn = DuckDBConnection::new(config).await.unwrap();

        conn.execute(
            "CREATE TABLE t1 AS SELECT generate_series AS id, 'name_' || cast(generate_series AS VARCHAR) AS name FROM generate_series(1, 100)".to_string(),
        ).await.unwrap();
        conn.execute(
            "CREATE TABLE t2 AS SELECT generate_series AS id, generate_series * 3 AS triple FROM generate_series(1, 100)".to_string(),
        ).await.unwrap();

        let schema = conn.get_schema("t1".to_string()).await.unwrap().unwrap();

        let (rows, explain_json) = conn
            .new_relation_and_explain(
                MaterializeMode::TempTable,
                "join_test".to_string(),
                "SELECT t1.id, t1.name, t2.triple FROM t1 JOIN t2 ON t1.id = t2.id WHERE t1.id > 50".to_string(),
            )
            .await
            .unwrap();

        assert_eq!(rows, 0);
        assert!(explain_json.is_some());

        let plan = conn
            .explain_to_logical_plan(&explain_json.unwrap(), schema)
            .await
            .expect("should convert explain to ExecutionPlan");

        // Plan was successfully created
    }

    #[tokio::test]
    async fn test_explain_to_logical_plan_aggregate() {
        let config = DuckDBConfig::new_from_path(":memory:".to_string());
        let conn = DuckDBConnection::new(config).await.unwrap();

        conn.execute(
            "CREATE TABLE t1 AS SELECT generate_series AS id, 'name_' || cast(generate_series AS VARCHAR) AS name, generate_series * 2 AS double_id FROM generate_series(1, 100)".to_string(),
        ).await.unwrap();

        let schema = conn.get_schema("t1".to_string()).await.unwrap().unwrap();

        let (rows, explain_json) = conn
            .new_relation_and_explain(
                MaterializeMode::TempTable,
                "agg_test".to_string(),
                "SELECT name, count(*) as cnt, sum(double_id) as total FROM t1 GROUP BY name".to_string(),
            )
            .await
            .unwrap();

        assert_eq!(rows, 0);
        assert!(explain_json.is_some());

        let plan = conn
            .explain_to_logical_plan(&explain_json.unwrap(), schema)
            .await
            .expect("should convert explain to ExecutionPlan");

        let children = plan.children();
        assert!(!children.is_empty());
    }

    #[tokio::test]
    async fn test_explain_to_logical_plan_union() {
        let config = DuckDBConfig::new_from_path(":memory:".to_string());
        let conn = DuckDBConnection::new(config).await.unwrap();

        conn.execute(
            "CREATE TABLE t1 AS SELECT generate_series AS id, 'a' AS label FROM generate_series(1, 10)".to_string(),
        ).await.unwrap();
        conn.execute(
            "CREATE TABLE t2 AS SELECT generate_series + 10 AS id, 'b' AS label FROM generate_series(1, 10)".to_string(),
        ).await.unwrap();

        let schema = conn.get_schema("t1".to_string()).await.unwrap().unwrap();

        let (rows, explain_json) = conn
            .new_relation_and_explain(
                MaterializeMode::TempTable,
                "union_test".to_string(),
                "SELECT * FROM t1 UNION ALL SELECT * FROM t2".to_string(),
            )
            .await
            .unwrap();

        assert_eq!(rows, 0);
        assert!(explain_json.is_some());

        let plan = conn
            .explain_to_logical_plan(&explain_json.unwrap(), schema)
            .await
            .expect("should convert explain to ExecutionPlan");

        let children = plan.children();
        assert!(children.len() >= 1);
    }

    #[tokio::test]
    async fn test_explain_to_logical_plan_view() {
        let config = DuckDBConfig::new_from_path(":memory:".to_string());
        let conn = DuckDBConnection::new(config).await.unwrap();

        conn.execute(
            "CREATE TABLE t1 AS SELECT generate_series AS id, generate_series * 2 AS val FROM generate_series(1, 100)".to_string(),
        ).await.unwrap();

        let schema = conn.get_schema("t1".to_string()).await.unwrap().unwrap();

        let (rows, explain_json) = conn
            .new_relation_and_explain(
                MaterializeMode::View,
                "view_test".to_string(),
                "SELECT id, val, sum(val) OVER (ORDER BY id) as running_sum FROM t1".to_string(),
            )
            .await
            .unwrap();

        assert_eq!(rows, 0);
        assert!(explain_json.is_some());

        let plan = conn
            .explain_to_logical_plan(&explain_json.unwrap(), schema)
            .await
            .expect("should convert explain to ExecutionPlan");

        assert!(!plan.children().is_empty());
    }

    #[tokio::test]
    async fn test_explain_to_logical_plan_returns_some() {
        let config = DuckDBConfig::new_from_path(":memory:".to_string());
        let conn = DuckDBConnection::new(config).await.unwrap();

        let json = r#"[{"name":"SEQ_SCAN","children":[],"extra_info":{}}]"#;
        let result = conn.explain_to_logical_plan(json, make_schema()).await;
        assert!(result.is_some());
    }
}
