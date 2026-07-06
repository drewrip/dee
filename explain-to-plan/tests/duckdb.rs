use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::physical_plan::ExecutionPlan;
use explain_to_plan::{DuckDBTranslateConfig, ExplainToPlan};

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

fn translate(name: &str) -> Arc<dyn ExecutionPlan> {
    config()
        .explain_to_plan(fixture(name))
        .unwrap_or_else(|e| panic!("failed to translate fixture {name}: {e}"))
}

/// Renders a plan tree as `OperatorName [col1, col2]` lines, indented by
/// depth, so tests can assert on overall plan *shape* without depending on
/// verbose `Debug` output.
fn shape(plan: &Arc<dyn ExecutionPlan>) -> String {
    fn walk(plan: &Arc<dyn ExecutionPlan>, depth: usize, out: &mut String) {
        let schema = plan.schema();
        let cols: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        out.push_str(&"  ".repeat(depth));
        out.push_str(plan.name());
        out.push_str(" [");
        out.push_str(&cols.join(", "));
        out.push_str("]\n");
        for child in plan.children() {
            walk(child, depth + 1, out);
        }
    }
    let mut out = String::new();
    walk(plan, 0, &mut out);
    out
}

/// Depth-first search for the first node of type `T` anywhere in the plan tree.
fn find_plan<T: ExecutionPlan>(plan: &Arc<dyn ExecutionPlan>) -> Option<&T> {
    if let Some(found) = plan.downcast_ref::<T>() {
        return Some(found);
    }
    for child in plan.children() {
        if let Some(found) = find_plan::<T>(child) {
            return Some(found);
        }
    }
    None
}

#[test]
fn scan_translates_to_projection_over_empty_exec() {
    let plan = translate("scan");
    let s = shape(&plan);
    assert!(s.contains("ProjectionExec"), "{s}");
    assert!(s.contains("EmptyExec"), "{s}");
    assert_eq!(
        plan.schema().fields().iter().map(|f| f.name().as_str()).collect::<Vec<_>>(),
        vec!["id", "grp"]
    );
}

#[test]
fn scan_with_filter_pushes_filter_below_projection() {
    let plan = translate("scan_filter");
    let s = shape(&plan);
    assert!(s.contains("FilterExec"), "{s}");
    assert_eq!(plan.schema().fields().iter().map(|f| f.name().as_str()).collect::<Vec<_>>(), vec!["id"]);
}

#[test]
fn projection_with_arithmetic_expression() {
    let plan = translate("projection_expr");
    let s = shape(&plan);
    assert!(s.contains("ProjectionExec"), "{s}");
    // The arithmetic projection column is named after its DuckDB source text.
    let schema = plan.schema();
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert!(names.iter().any(|n| n.contains('%')), "{names:?}");
}

#[test]
fn single_column_group_by_produces_aggregate_exec() {
    let plan = translate("agg_single_group");
    let s = shape(&plan);
    assert!(s.contains("AggregateExec"), "{s}");
}

#[test]
fn multi_column_group_by_with_multiple_aggregates() {
    let plan = translate("agg_multi_group");
    let s = shape(&plan);
    assert!(s.contains("AggregateExec"), "{s}");
    // grp, m, count, min, max, avg after the outer decompress projection.
    assert_eq!(plan.schema().fields().len(), 6);
}

#[test]
fn inner_hash_join() {
    let plan = translate("join_inner");
    let s = shape(&plan);
    assert!(s.contains("HashJoinExec"), "{s}");
}

#[test]
fn left_hash_join() {
    let plan = translate("join_left");
    let s = shape(&plan);
    assert!(s.contains("HashJoinExec"), "{s}");
}

#[test]
fn semi_join_from_exists_subquery() {
    let plan = translate("join_semi");
    let s = shape(&plan);
    assert!(s.contains("HashJoinExec"), "{s}");
}

#[test]
fn anti_join_from_not_exists_subquery() {
    let plan = translate("join_anti");
    let s = shape(&plan);
    assert!(s.contains("HashJoinExec"), "{s}");
}

#[test]
fn cross_join_has_no_condition() {
    let plan = translate("cross_join");
    let s = shape(&plan);
    assert!(s.contains("CrossJoinExec"), "{s}");
}

#[test]
fn union_all_combines_two_scans() {
    let plan = translate("union_all");
    let s = shape(&plan);
    assert!(s.contains("UnionExec"), "{s}");
    assert_eq!(plan.children().len(), 2);
}

#[test]
fn order_by_without_limit_is_a_full_sort() {
    let plan = translate("order_by");
    let s = shape(&plan);
    assert!(s.contains("SortExec"), "{s}");
}

#[test]
fn top_n_produces_sort_with_fetch() {
    let plan = translate("top_n");
    // Downcast to confirm the fetch limit made it onto the SortExec itself
    // (not a separate LimitExec).
    let sort = plan
        .downcast_ref::<datafusion::physical_plan::sorts::sort::SortExec>()
        .unwrap_or_else(|| panic!("expected root to be SortExec, got: {}", shape(&plan)));
    assert_eq!(sort.fetch(), Some(5));
}

#[test]
fn plain_limit_uses_observed_cardinality_as_fetch() {
    let plan = translate("limit_simple");
    let limit = plan
        .downcast_ref::<datafusion::physical_plan::limit::GlobalLimitExec>()
        .unwrap_or_else(|| panic!("expected root to be GlobalLimitExec, got: {}", shape(&plan)));
    assert_eq!(limit.fetch(), Some(5));
}

#[test]
fn single_window_function_produces_window_agg_exec() {
    let plan = translate("window_row_number");
    let s = shape(&plan);
    assert!(s.contains("WindowAggExec"), "{s}");
}

#[test]
fn multiple_window_functions_in_one_node() {
    let plan = translate("window_multi");
    let s = shape(&plan);
    assert!(s.contains("WindowAggExec"), "{s}");
    let window = find_plan::<datafusion::physical_plan::windows::WindowAggExec>(&plan)
        .unwrap_or_else(|| panic!("expected a WindowAggExec in: {s}"));
    assert_eq!(window.window_expr().len(), 2);
}

#[test]
fn window_with_explicit_rows_frame() {
    let plan = translate("window_frame");
    let s = shape(&plan);
    assert!(s.contains("WindowAggExec"), "{s}");
    let window = find_plan::<datafusion::physical_plan::windows::WindowAggExec>(&plan)
        .unwrap_or_else(|| panic!("expected a WindowAggExec in: {s}"));
    let frame = window.window_expr()[0].get_window_frame();
    assert_eq!(frame.units, datafusion::logical_expr::WindowFrameUnits::Rows);
}

#[test]
fn window_without_partition_by_still_translates() {
    let plan = translate("window_rank_nopartition");
    let s = shape(&plan);
    assert!(s.contains("WindowAggExec"), "{s}");
}

#[test]
fn non_equi_join_produces_nested_loop_join_with_filter() {
    let plan = translate("join_nonequi");
    let s = shape(&plan);
    assert!(s.contains("NestedLoopJoinExec"), "{s}");
}

#[test]
fn hash_join_with_mixed_equi_and_nonequi_conditions() {
    let plan = translate("join_mixed_equi_nonequi");
    let s = shape(&plan);
    assert!(s.contains("HashJoinExec"), "{s}");
    let join = find_plan::<datafusion::physical_plan::joins::HashJoinExec>(&plan)
        .unwrap_or_else(|| panic!("expected a HashJoinExec in: {s}"));
    assert_eq!(join.on().len(), 1, "exactly one equi condition should drive the hash join");
    assert!(join.filter().is_some(), "the non-equi condition should become a residual JoinFilter");
}

#[test]
fn unknown_table_gives_clear_error_instead_of_panicking() {
    let plan_json = fixture("scan");
    let err = DuckDBTranslateConfig::new().explain_to_plan(plan_json).unwrap_err();
    assert!(err.to_string().contains("catalog entry"), "{err}");
}

#[test]
fn plain_explain_without_analyze_also_parses() {
    // Plain `EXPLAIN (FORMAT JSON)` (no ANALYZE) never has `operator_cardinality`,
    // so scans/filters/projections/joins/aggregates should all still translate;
    // only nodes that need runtime stats (STREAMING_LIMIT) would fail.
    let plain = r#"[{"name": "PROJECTION", "extra_info": {"Projections": "id"}, "children": [
        {"name": "SEQ_SCAN", "extra_info": {"Table": "test.main.t1", "Projections": "id"}, "children": []}
    ]}]"#;
    let plan = config().explain_to_plan(plain.to_string()).unwrap();
    assert_eq!(plan.schema().fields().len(), 1);
}

#[test]
fn nonequi_join_with_unresolvable_condition_errors_instead_of_silently_wrong() {
    // A condition that references columns from both sides on a single
    // operand (e.g. `grp + cat < id`) isn't a supported shape; it must be
    // rejected rather than mistranslated.
    let plain = r#"[{"name": "PIECEWISE_MERGE_JOIN", "extra_info": {"Join Type": "INNER", "Conditions": "grp + cat < id"}, "children": [
        {"name": "SEQ_SCAN", "extra_info": {"Table": "test.main.t1", "Projections": ["grp", "id"]}, "children": []},
        {"name": "SEQ_SCAN", "extra_info": {"Table": "test.main.t2", "Projections": "cat"}, "children": []}
    ]}]"#;
    let err = config().explain_to_plan(plain.to_string()).unwrap_err();
    assert!(matches!(err, explain_to_plan::DuckDBTranslateError::UnsupportedOperator(_)), "{err}");
}
