use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::sql::unparser::{Unparser, dialect::DefaultDialect};
use explain_to_plan::{DuckDBTranslateConfig, ExplainToPlan, RaiseToLogicalError, raise_to_logical};

fn t1_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("grp", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]))
}

fn t2_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("cat", DataType::Int64, false),
    ]))
}

fn config() -> DuckDBTranslateConfig {
    DuckDBTranslateConfig::new()
        .with_table("test.main.t1", t1_schema())
        .with_table("test.main.t2", t2_schema())
}

fn fixture(name: &str) -> String {
    std::fs::read_to_string(format!("{}/tests/fixtures/{name}.json", env!("CARGO_MANIFEST_DIR")))
        .unwrap_or_else(|e| panic!("failed to read fixture {name}: {e}"))
}

/// Runs the full pipeline this module exists for: DuckDB `EXPLAIN ANALYZE`
/// JSON -> `ExecutionPlan` (via [`explain_to_plan`]) -> `LogicalPlan` (via
/// [`raise_to_logical`]) -> SQL text (via DataFusion's own `Unparser`).
fn raise_fixture_to_sql(name: &str) -> String {
    let physical = config()
        .explain_to_plan(fixture(name))
        .unwrap_or_else(|e| panic!("failed to translate fixture {name}: {e}"));
    let logical = raise_to_logical(&physical)
        .unwrap_or_else(|e| panic!("failed to raise fixture {name} to a LogicalPlan: {e}"));
    Unparser::new(&DefaultDialect {})
        .plan_to_sql(&logical)
        .unwrap_or_else(|e| panic!("failed to unparse fixture {name}: {e}"))
        .to_string()
}

#[test]
fn scan_raises_to_a_bare_table_scan() {
    let sql = raise_fixture_to_sql("scan");
    assert!(sql.to_uppercase().contains("SELECT"), "{sql}");
    assert!(sql.contains("t1"), "{sql}");
}

#[test]
fn scan_with_filter_raises_to_a_where_clause() {
    let sql = raise_fixture_to_sql("scan_filter");
    assert!(sql.to_uppercase().contains("WHERE"), "{sql}");
    assert!(sql.contains("t1"), "{sql}");
}

#[test]
fn projection_with_arithmetic_expression_round_trips() {
    let sql = raise_fixture_to_sql("projection_expr");
    assert!(sql.to_uppercase().contains("SELECT"), "{sql}");
    assert!(sql.contains("t1"), "{sql}");
}

#[test]
fn group_by_raises_to_group_by_clause() {
    let sql = raise_fixture_to_sql("agg_single_group");
    assert!(sql.to_uppercase().contains("GROUP BY"), "{sql}");
}

#[test]
fn inner_hash_join_raises_to_join_on() {
    let sql = raise_fixture_to_sql("join_inner");
    assert!(sql.to_uppercase().contains("JOIN"), "{sql}");
    assert!(sql.contains("t1"), "{sql}");
    assert!(sql.contains("t2"), "{sql}");
}

#[test]
fn hash_join_with_mixed_equi_and_nonequi_conditions_keeps_residual_filter() {
    let sql = raise_fixture_to_sql("join_mixed_equi_nonequi");
    assert!(sql.to_uppercase().contains("JOIN"), "{sql}");
}

#[test]
fn nonequi_join_raises_to_nested_loop_join_with_full_condition() {
    let sql = raise_fixture_to_sql("join_nonequi");
    assert!(sql.to_uppercase().contains("JOIN"), "{sql}");
}

#[test]
fn cross_join_raises_without_a_join_condition() {
    let sql = raise_fixture_to_sql("cross_join");
    assert!(sql.contains("t1"), "{sql}");
    assert!(sql.contains("t2"), "{sql}");
}

#[test]
fn union_all_raises_to_a_sql_union() {
    let sql = raise_fixture_to_sql("union_all");
    assert!(sql.to_uppercase().contains("UNION"), "{sql}");
}

#[test]
fn order_by_raises_to_order_by_clause() {
    let sql = raise_fixture_to_sql("order_by");
    assert!(sql.to_uppercase().contains("ORDER BY"), "{sql}");
}

#[test]
fn top_n_raises_to_order_by_with_limit() {
    let sql = raise_fixture_to_sql("top_n");
    assert!(sql.to_uppercase().contains("ORDER BY"), "{sql}");
    assert!(sql.to_uppercase().contains("LIMIT"), "{sql}");
}

#[test]
fn plain_limit_raises_to_a_limit_clause() {
    let sql = raise_fixture_to_sql("limit_simple");
    assert!(sql.to_uppercase().contains("LIMIT"), "{sql}");
}

#[test]
fn aggregate_window_function_raises_to_sql() {
    // Every window fixture on disk (window_multi, window_frame,
    // window_rank_nopartition, window_row_number) includes a ranking
    // function alongside/instead of an aggregate one, so a synthetic plain
    // EXPLAIN JSON isolates the aggregate-backed case
    // (`PlainAggregateWindowExpr`), which raise_to_logical does support.
    let plain = r#"[{"name": "WINDOW", "extra_info": {"Projections": "sum(id) OVER (PARTITION BY grp ORDER BY id ASC NULLS LAST)"}, "children": [
        {"name": "SEQ_SCAN", "extra_info": {"Table": "test.main.t1", "Projections": ["id", "grp"]}, "children": []}
    ]}]"#;
    let physical = config()
        .explain_to_plan(plain.to_string())
        .expect("translation to ExecutionPlan should succeed");
    let logical = raise_to_logical(&physical).expect("raising an aggregate-backed window function should succeed");
    let sql = Unparser::new(&DefaultDialect {})
        .plan_to_sql(&logical)
        .expect("unparsing should succeed")
        .to_string();
    assert!(sql.to_uppercase().contains("OVER"), "{sql}");
}

#[test]
fn ranking_window_function_is_unsupported_not_silently_wrong() {
    // ROW_NUMBER() is backed by StandardWindowExpr, whose underlying function
    // definition raise_to_logical cannot recover generically -- it must
    // error rather than produce an incorrect plan.
    let physical = config()
        .explain_to_plan(fixture("window_row_number"))
        .expect("translation to ExecutionPlan should still succeed");
    let err = raise_to_logical(&physical).unwrap_err();
    assert!(matches!(err, RaiseToLogicalError::UnsupportedOperator(_)), "{err}");
}

#[test]
fn scan_leaf_without_table_name_tag_errors_clearly() {
    use arrow::datatypes::Schema as ArrowSchema;
    use datafusion::physical_plan::empty::EmptyExec;

    // Construct an EmptyExec directly (bypassing the duckdb translator, which
    // always tags the schema) to exercise the missing-tag error path.
    let untagged_schema: SchemaRef = Arc::new(ArrowSchema::new(vec![Field::new("x", DataType::Int64, false)]));
    let plan: Arc<dyn datafusion::physical_plan::ExecutionPlan> = Arc::new(EmptyExec::new(untagged_schema));

    let err = raise_to_logical(&plan).unwrap_err();
    assert!(matches!(err, RaiseToLogicalError::MissingTableName), "{err}");
}
