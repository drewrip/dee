//! Recursively lowers a [`DuckDBNode`] tree into a DataFusion
//! [`ExecutionPlan`] tree, bottom-up.
//!
//! Supported operators: `SEQ_SCAN`/`TABLE_SCAN`, `PROJECTION`, `FILTER`,
//! `HASH_JOIN` (INNER/LEFT/RIGHT/FULL/SEMI/ANTI, equi- or mixed equi/non-equi
//! conditions), `PIECEWISE_MERGE_JOIN`/`NESTED_LOOP_JOIN` (non-equi
//! conditions), `CROSS_PRODUCT`, `PERFECT_HASH_GROUP_BY`/`HASH_GROUP_BY`,
//! `ORDER_BY`, `TOP_N`, `STREAMING_LIMIT`/`LIMIT`, `UNION`, `WINDOW`.
//! Anything else (e.g. mark joins, recursive CTEs) returns
//! [`DuckDBTranslateError::UnsupportedOperator`].
//!
//! DuckDB's `EXPLAIN` output carries no column type information, so base
//! table schemas must be supplied by the caller via a catalog lookup rather
//! than inferred.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::datatypes::{Schema, SchemaRef};
use datafusion::common::{JoinSide, JoinType, NullEquality};
use datafusion::logical_expr::{Operator, WindowFunctionDefinition};
use datafusion::physical_expr::aggregate::AggregateExprBuilder;
use datafusion::physical_expr::expressions::Column as PhysColumn;
use datafusion::physical_expr::{LexOrdering, PhysicalSortExpr};
use datafusion::physical_plan::joins::utils::{ColumnIndex, JoinFilter};
use datafusion::physical_plan::joins::{CrossJoinExec, HashJoinExec, NestedLoopJoinExec, PartitionMode};
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::union::UnionExec;
use datafusion::physical_plan::windows::{WindowAggExec, create_window_expr};
use datafusion::physical_plan::{
    ExecutionPlan, PhysicalExpr, aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy},
    empty::EmptyExec, filter::FilterExec, projection::ProjectionExec,
};
use datafusion::prelude::SessionContext;

use super::DuckDBTranslateError;
use super::expr;
use super::model::DuckDBNode;

type Result<T> = std::result::Result<T, DuckDBTranslateError>;

/// Table schemas known to the translator, keyed by table name as it appears
/// in DuckDB's `"Table"` extra_info field (typically `catalog.schema.table`).
/// A bare table name (last dotted segment) is also accepted as a lookup key.
pub type Catalog = HashMap<String, SchemaRef>;

pub struct Translator<'a> {
    ctx: &'a SessionContext,
    catalog: &'a Catalog,
}

pub fn translate_with_catalog(
    node: &DuckDBNode,
    ctx: &SessionContext,
    catalog: &Catalog,
) -> Result<Arc<dyn ExecutionPlan>> {
    let translator = Translator { ctx, catalog };
    translator.translate(node)
}

fn extra<'n>(node: &'n DuckDBNode, field: &str) -> Result<&'n super::model::ExtraInfoValue> {
    node.extra_info.get(field).ok_or_else(|| DuckDBTranslateError::MissingField {
        operator: node.operator_name.clone(),
        field: field.to_string(),
    })
}

impl<'a> Translator<'a> {
    fn translate(&self, node: &DuckDBNode) -> Result<Arc<dyn ExecutionPlan>> {
        match node.operator_name.as_str() {
            "SEQ_SCAN" | "TABLE_SCAN" => self.translate_scan(node),
            "PROJECTION" => self.translate_projection(node),
            "FILTER" => self.translate_filter(node),
            "HASH_JOIN" => self.translate_hash_join(node),
            "PIECEWISE_MERGE_JOIN" | "NESTED_LOOP_JOIN" => self.translate_nonequi_join(node),
            "CROSS_PRODUCT" => self.translate_cross_product(node),
            "PERFECT_HASH_GROUP_BY" | "HASH_GROUP_BY" => self.translate_group_by(node),
            "ORDER_BY" => self.translate_order_by(node, None),
            "TOP_N" => self.translate_top_n(node),
            "STREAMING_LIMIT" | "LIMIT" => self.translate_limit(node),
            "UNION" => self.translate_union(node),
            "WINDOW" => self.translate_window(node),
            other => Err(DuckDBTranslateError::UnsupportedOperator(other.to_string())),
        }
    }

    fn only_child(&self, node: &DuckDBNode) -> Result<Arc<dyn ExecutionPlan>> {
        match node.children.as_slice() {
            [child] => self.translate(child),
            other => Err(DuckDBTranslateError::ExprParse(format!(
                "operator {} expected exactly one child, found {}",
                node.operator_name,
                other.len()
            ))),
        }
    }

    fn translate_scan(&self, node: &DuckDBNode) -> Result<Arc<dyn ExecutionPlan>> {
        let table = extra(node, "Table")?.first().unwrap_or_default().to_string();
        let full_schema = self.catalog.get(&table).cloned().or_else(|| {
            let bare = table.rsplit('.').next().unwrap_or(&table);
            self.catalog.get(bare).cloned()
        });
        let full_schema = full_schema.ok_or_else(|| DuckDBTranslateError::MissingField {
            operator: node.operator_name.clone(),
            field: format!("catalog entry for table '{table}'"),
        })?;

        let mut plan: Arc<dyn ExecutionPlan> = Arc::new(EmptyExec::new(full_schema.clone()));

        if let Some(filters) = node.extra_info.get("Filters") {
            for f in filters.iter() {
                if f.starts_with("optional:") {
                    // Speculative/dynamic filters DuckDB may not actually apply
                    // (e.g. runtime bloom filters); skip, they aren't guaranteed.
                    continue;
                }
                let predicate = expr::parse_expr(f, plan.schema().as_ref(), self.ctx)?;
                let physical = expr::to_physical(&predicate, plan.schema().as_ref())?;
                plan = Arc::new(FilterExec::try_new(physical, plan)?);
            }
        }

        if let Some(projections) = node.extra_info.get("Projections") {
            let cols: Vec<String> = projections.iter().map(str::to_string).collect();
            if !cols.is_empty() {
                let input_schema = plan.schema();
                let exprs: Result<Vec<(Arc<dyn PhysicalExpr>, String)>> = cols
                    .iter()
                    .map(|name| {
                        let idx = input_schema.index_of(name).map_err(|_| {
                            DuckDBTranslateError::ExprParse(format!(
                                "scan projects unknown column '{name}' from table '{table}'"
                            ))
                        })?;
                        Ok((
                            Arc::new(PhysColumn::new(name, idx)) as Arc<dyn PhysicalExpr>,
                            name.clone(),
                        ))
                    })
                    .collect();
                plan = Arc::new(ProjectionExec::try_new(exprs?, plan)?);
            }
        }

        Ok(plan)
    }

    fn translate_projection(&self, node: &DuckDBNode) -> Result<Arc<dyn ExecutionPlan>> {
        let input = self.only_child(node)?;
        let projections = extra(node, "Projections")?;
        let input_schema = input.schema();

        let mut exprs = Vec::new();
        for text in projections.iter() {
            let logical = expr::parse_expr(text, input_schema.as_ref(), self.ctx)?;
            let physical = expr::to_physical(&logical, input_schema.as_ref())?;
            // DuckDB's `__internal_{de,}compress_integral_*` wrappers (and plain
            // positional passthroughs) are storage-format artifacts, not real
            // renames -- downstream operators keep referring to the *original*
            // column name, so preserve it rather than naming the output column
            // after the wrapper's literal text.
            let name = passthrough_column_name(&logical)
                .map(str::to_string)
                .unwrap_or_else(|| text.to_string());
            exprs.push((physical, name));
        }
        Ok(Arc::new(ProjectionExec::try_new(exprs, input)?))
    }

    fn translate_filter(&self, node: &DuckDBNode) -> Result<Arc<dyn ExecutionPlan>> {
        let input = self.only_child(node)?;
        let text = extra(node, "Expression")?.first().ok_or_else(|| DuckDBTranslateError::MissingField {
            operator: node.operator_name.clone(),
            field: "Expression".to_string(),
        })?;
        let input_schema = input.schema();
        let logical = expr::parse_expr(text, input_schema.as_ref(), self.ctx)?;
        let physical = expr::to_physical(&logical, input_schema.as_ref())?;
        Ok(Arc::new(FilterExec::try_new(physical, input)?))
    }

    fn translate_cross_product(&self, node: &DuckDBNode) -> Result<Arc<dyn ExecutionPlan>> {
        let [left_node, right_node] = node.children.as_slice() else {
            return Err(DuckDBTranslateError::ExprParse(
                "CROSS_PRODUCT expects exactly two children".to_string(),
            ));
        };
        let left = self.translate(left_node)?;
        let right = self.translate(right_node)?;
        Ok(Arc::new(CrossJoinExec::new(left, right)))
    }

    fn translate_hash_join(&self, node: &DuckDBNode) -> Result<Arc<dyn ExecutionPlan>> {
        let [left_node, right_node] = node.children.as_slice() else {
            return Err(DuckDBTranslateError::ExprParse(
                "HASH_JOIN expects exactly two children".to_string(),
            ));
        };
        let left = self.translate(left_node)?;
        let right = self.translate(right_node)?;
        let left_schema = left.schema();
        let right_schema = right.schema();

        let join_type_text = extra(node, "Join Type")?.first().unwrap_or("INNER");
        let join_type = map_join_type(join_type_text)?;

        // Each entry in `Conditions` is a standalone clause (implicitly
        // AND-ed together); a HASH_JOIN needs at least one equality clause to
        // drive the hash build/probe, but may carry additional non-equi
        // clauses that DataFusion evaluates as a post-join residual filter.
        let mut on = Vec::new();
        let mut non_equi = Vec::new();
        for clause in extra(node, "Conditions")?.iter() {
            match split_top_level_eq(clause) {
                Some((lhs, rhs)) => {
                    let pair = self.resolve_join_pair(&lhs, &rhs, left_schema.as_ref(), right_schema.as_ref())?;
                    on.push(pair);
                }
                None => non_equi.push(clause.to_string()),
            }
        }
        if on.is_empty() {
            return Err(DuckDBTranslateError::UnsupportedOperator(
                "HASH_JOIN with no equi-join condition".to_string(),
            ));
        }
        let filter = if non_equi.is_empty() {
            None
        } else {
            Some(self.build_join_filter(&non_equi, left_schema.as_ref(), right_schema.as_ref())?)
        };

        Ok(Arc::new(HashJoinExec::try_new(
            left,
            right,
            on,
            filter,
            &join_type,
            None,
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            false,
        )?))
    }

    /// Translates `PIECEWISE_MERGE_JOIN`/`NESTED_LOOP_JOIN`, DuckDB's
    /// operators for joins with no usable equi-join key (e.g. `t1.a < t2.b`).
    /// Both are lowered to [`NestedLoopJoinExec`]: the merge-join's sortedness
    /// is a runtime optimization that doesn't change plan semantics.
    fn translate_nonequi_join(&self, node: &DuckDBNode) -> Result<Arc<dyn ExecutionPlan>> {
        let [left_node, right_node] = node.children.as_slice() else {
            return Err(DuckDBTranslateError::ExprParse(format!(
                "{} expects exactly two children",
                node.operator_name
            )));
        };
        let left = self.translate(left_node)?;
        let right = self.translate(right_node)?;
        let left_schema = left.schema();
        let right_schema = right.schema();

        let join_type_text = extra(node, "Join Type")?.first().unwrap_or("INNER");
        let join_type = map_join_type(join_type_text)?;

        let clauses: Vec<String> = extra(node, "Conditions")?.iter().map(str::to_string).collect();
        let filter = self.build_join_filter(&clauses, left_schema.as_ref(), right_schema.as_ref())?;

        Ok(Arc::new(NestedLoopJoinExec::try_new(left, right, Some(filter), &join_type, None)?))
    }

    /// Builds a [`JoinFilter`] evaluating the AND of `clauses` over the
    /// concatenation of `left_schema` and `right_schema`. Each clause must be
    /// a simple `<left-side expr> <op> <right-side expr>` comparison (the
    /// shape DuckDB actually emits for non-equi join conditions); arbitrary
    /// expressions spanning both sides within a single operand are not
    /// supported.
    fn build_join_filter(
        &self,
        clauses: &[String],
        left_schema: &Schema,
        right_schema: &Schema,
    ) -> Result<JoinFilter> {
        let left_len = left_schema.fields().len();
        let mut combined: Option<Arc<dyn PhysicalExpr>> = None;
        for clause in clauses {
            let (lhs, op, rhs) = split_top_level_cmp(clause).ok_or_else(|| {
                DuckDBTranslateError::UnsupportedOperator(format!(
                    "unsupported non-equi join condition shape: {clause}"
                ))
            })?;
            let (left_expr, right_expr) = self.resolve_join_pair_for_op(&lhs, &rhs, left_schema, right_schema)?;
            let right_expr_shifted = shift_columns(right_expr, left_len)?;
            let merged_schema = concat_schema(left_schema, right_schema);
            let clause_expr =
                datafusion::physical_expr::expressions::binary(left_expr, op, right_expr_shifted, &merged_schema)?;
            combined = Some(match combined {
                None => clause_expr,
                Some(acc) => datafusion::physical_expr::expressions::binary(
                    acc,
                    Operator::And,
                    clause_expr,
                    &merged_schema,
                )?,
            });
        }
        let expression = combined.ok_or_else(|| {
            DuckDBTranslateError::UnsupportedOperator("join with no conditions".to_string())
        })?;

        let column_indices = (0..left_len)
            .map(|i| ColumnIndex { index: i, side: JoinSide::Left })
            .chain(
                (0..right_schema.fields().len()).map(|i| ColumnIndex { index: i, side: JoinSide::Right }),
            )
            .collect();
        let merged_schema = Arc::new(concat_schema(left_schema, right_schema));
        Ok(JoinFilter::new(expression, column_indices, merged_schema))
    }

    /// Like [`Self::resolve_join_pair`], but for a comparison operator other
    /// than equality (no "same name printed on both sides" fallback, since
    /// that quirk is specific to DuckDB's equi-join decorrelation).
    fn resolve_join_pair_for_op(
        &self,
        lhs: &str,
        rhs: &str,
        left_schema: &Schema,
        right_schema: &Schema,
    ) -> Result<(Arc<dyn PhysicalExpr>, Arc<dyn PhysicalExpr>)> {
        let lhs_on_left = expr::parse_expr(lhs, left_schema, self.ctx);
        let rhs_on_right = expr::parse_expr(rhs, right_schema, self.ctx);
        if let (Ok(l), Ok(r)) = (&lhs_on_left, &rhs_on_right) {
            return Ok((expr::to_physical(l, left_schema)?, expr::to_physical(r, right_schema)?));
        }
        let lhs_on_right = expr::parse_expr(lhs, right_schema, self.ctx);
        let rhs_on_left = expr::parse_expr(rhs, left_schema, self.ctx);
        if let (Ok(r), Ok(l)) = (&lhs_on_right, &rhs_on_left) {
            return Ok((expr::to_physical(l, left_schema)?, expr::to_physical(r, right_schema)?));
        }
        Err(DuckDBTranslateError::UnsupportedOperator(format!(
            "could not resolve join condition '{lhs} ? {rhs}' against left/right schemas"
        )))
    }

    fn resolve_join_pair(
        &self,
        lhs: &str,
        rhs: &str,
        left_schema: &Schema,
        right_schema: &Schema,
    ) -> Result<(Arc<dyn PhysicalExpr>, Arc<dyn PhysicalExpr>)> {
        let lhs_on_left = expr::parse_expr(lhs, left_schema, self.ctx);
        let rhs_on_right = expr::parse_expr(rhs, right_schema, self.ctx);
        if let (Ok(l), Ok(r)) = (&lhs_on_left, &rhs_on_right) {
            return Ok((expr::to_physical(l, left_schema)?, expr::to_physical(r, right_schema)?));
        }
        let lhs_on_right = expr::parse_expr(lhs, right_schema, self.ctx);
        let rhs_on_left = expr::parse_expr(rhs, left_schema, self.ctx);
        if let (Ok(r), Ok(l)) = (&lhs_on_right, &rhs_on_left) {
            return Ok((expr::to_physical(l, left_schema)?, expr::to_physical(r, right_schema)?));
        }

        // DuckDB's `EXPLAIN` text for SEMI/ANTI joins produced by decorrelating
        // an EXISTS/NOT EXISTS subquery sometimes prints the *same* column
        // name on both sides of the condition (a cosmetic quirk of its
        // planner, not an actual self-join): e.g. `"grp IS NOT DISTINCT FROM
        // grp"` even though the right side's only column is named `cat`. Fall
        // back to positional resolution (this side's sole column) for
        // whichever side couldn't be resolved by name but has exactly one
        // candidate column.
        let left_expr = lhs_on_left
            .ok()
            .or_else(|| rhs_on_left.ok())
            .map(|e| expr::to_physical(&e, left_schema))
            .or_else(|| single_column(left_schema).map(Ok));
        let right_expr = rhs_on_right
            .ok()
            .or_else(|| lhs_on_right.ok())
            .map(|e| expr::to_physical(&e, right_schema))
            .or_else(|| single_column(right_schema).map(Ok));

        if let (Some(l), Some(r)) = (left_expr, right_expr) {
            return Ok((l?, r?));
        }

        Err(DuckDBTranslateError::UnsupportedOperator(format!(
            "could not resolve join condition '{lhs} = {rhs}' against left/right schemas"
        )))
    }

    fn translate_group_by(&self, node: &DuckDBNode) -> Result<Arc<dyn ExecutionPlan>> {
        let input = self.only_child(node)?;
        let input_schema = input.schema();

        let mut group_exprs = Vec::new();
        if let Some(groups) = node.extra_info.get("Groups") {
            for text in groups.iter() {
                let logical = expr::parse_expr(text, input_schema.as_ref(), self.ctx)?;
                let physical = expr::to_physical(&logical, input_schema.as_ref())?;
                group_exprs.push((physical, text.to_string()));
            }
        }
        let group_by = PhysicalGroupBy::new_single(group_exprs);

        let mut agg_exprs = Vec::new();
        if let Some(aggregates) = node.extra_info.get("Aggregates") {
            for text in aggregates.iter() {
                agg_exprs.push(self.translate_aggregate_call(text, input_schema.as_ref())?);
            }
        }

        let filter_exprs = vec![None; agg_exprs.len()];

        Ok(Arc::new(AggregateExec::try_new(
            AggregateMode::Single,
            group_by,
            agg_exprs,
            filter_exprs,
            input,
            input_schema,
        )?))
    }

    fn translate_aggregate_call(
        &self,
        text: &str,
        input_schema: &Schema,
    ) -> Result<Arc<datafusion::physical_expr::aggregate::AggregateFunctionExpr>> {
        let call = expr::parse_call(text, input_schema, self.ctx)?;
        let mapped_name = map_aggregate_fn_name(&call.name);

        // `count_star()` takes no arguments; DataFusion's `count` UDAF expects
        // one, conventionally `COUNT(*) -> count(1)`.
        let args: Vec<Arc<dyn PhysicalExpr>> = if call.args.is_empty() {
            vec![Arc::new(datafusion::physical_expr::expressions::Literal::new(
                datafusion::common::ScalarValue::Int64(Some(1)),
            ))]
        } else {
            call.args
                .iter()
                .map(|e| expr::to_physical(e, input_schema))
                .collect::<Result<Vec<_>>>()?
        };

        use datafusion::execution::FunctionRegistry;
        let udaf = self.ctx.udaf(mapped_name).map_err(|_| {
            DuckDBTranslateError::UnsupportedFunction(call.name.clone())
        })?;

        let mut builder = AggregateExprBuilder::new(udaf, args)
            .schema(Arc::new(input_schema.clone()))
            .alias(text.to_string());
        if call.distinct {
            builder = builder.distinct();
        }
        Ok(Arc::new(builder.build()?))
    }

    fn translate_order_by(
        &self,
        node: &DuckDBNode,
        fetch: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let input = self.only_child(node)?;
        let input_schema = input.schema();
        let order_by = extra(node, "Order By")?;

        let mut sort_exprs = Vec::new();
        for text in order_by.iter() {
            let (logical, asc, nulls_first) = expr::parse_order_by(text, input_schema.as_ref(), self.ctx)?;
            let physical = expr::to_physical(&logical, input_schema.as_ref())?;
            sort_exprs.push(PhysicalSortExpr {
                expr: physical,
                options: arrow::compute::SortOptions { descending: !asc, nulls_first },
            });
        }
        let ordering = LexOrdering::new(sort_exprs).ok_or_else(|| {
            DuckDBTranslateError::MissingField {
                operator: node.operator_name.clone(),
                field: "Order By".to_string(),
            }
        })?;

        let sort = SortExec::new(ordering, input);
        let sort = if let Some(n) = fetch { sort.with_fetch(Some(n)) } else { sort };
        Ok(Arc::new(sort))
    }

    fn translate_top_n(&self, node: &DuckDBNode) -> Result<Arc<dyn ExecutionPlan>> {
        let top = extra(node, "Top")?.first().ok_or_else(|| DuckDBTranslateError::MissingField {
            operator: node.operator_name.clone(),
            field: "Top".to_string(),
        })?;
        let n: usize = top.parse().map_err(|_| DuckDBTranslateError::ExprParse(format!(
            "expected integer TOP_N count, found '{top}'"
        )))?;
        self.translate_order_by(node, Some(n))
    }

    fn translate_limit(&self, node: &DuckDBNode) -> Result<Arc<dyn ExecutionPlan>> {
        let input = self.only_child(node)?;
        // DuckDB's EXPLAIN output does not carry the literal LIMIT/OFFSET
        // count in `extra_info` for STREAMING_LIMIT; the best available
        // signal is the *actual* row count observed under EXPLAIN ANALYZE,
        // which is exact only when the input has at least `fetch` rows and
        // there is no OFFSET (DuckDB rewrites LIMIT+OFFSET into a different
        // plan shape entirely).
        let fetch = node.operator_cardinality.ok_or_else(|| DuckDBTranslateError::MissingField {
            operator: node.operator_name.clone(),
            field: "operator_cardinality (requires EXPLAIN ANALYZE; LIMIT count is not present in plain EXPLAIN output)".to_string(),
        })? as usize;
        Ok(Arc::new(datafusion::physical_plan::limit::GlobalLimitExec::new(input, 0, Some(fetch))))
    }

    fn translate_union(&self, node: &DuckDBNode) -> Result<Arc<dyn ExecutionPlan>> {
        let children: Result<Vec<Arc<dyn ExecutionPlan>>> =
            node.children.iter().map(|c| self.translate(c)).collect();
        Ok(UnionExec::try_new(children?)?)
    }

    fn translate_window(&self, node: &DuckDBNode) -> Result<Arc<dyn ExecutionPlan>> {
        let input = self.only_child(node)?;
        let input_schema = input.schema();
        let calls = extra(node, "Projections")?;

        let mut window_exprs = Vec::new();
        for text in calls.iter() {
            let parsed = expr::parse_window_call(text, input_schema.as_ref(), self.ctx)?;
            let fun = self.resolve_window_fn(&parsed.func_name)?;

            // `count_star()` (no args) used as a window function still needs
            // DataFusion's `count` UDAF to receive an argument.
            let args: Vec<Arc<dyn PhysicalExpr>> = if parsed.args.is_empty() {
                vec![Arc::new(datafusion::physical_expr::expressions::Literal::new(
                    datafusion::common::ScalarValue::Int64(Some(1)),
                ))]
            } else {
                parsed
                    .args
                    .iter()
                    .map(|e| expr::to_physical(e, input_schema.as_ref()))
                    .collect::<Result<Vec<_>>>()?
            };
            let partition_by: Vec<Arc<dyn PhysicalExpr>> = parsed
                .partition_by
                .iter()
                .map(|e| expr::to_physical(e, input_schema.as_ref()))
                .collect::<Result<Vec<_>>>()?;
            let order_by: Vec<PhysicalSortExpr> = parsed
                .order_by
                .iter()
                .map(|(e, asc, nulls_first)| {
                    Ok(PhysicalSortExpr {
                        expr: expr::to_physical(e, input_schema.as_ref())?,
                        options: arrow::compute::SortOptions { descending: !asc, nulls_first: *nulls_first },
                    })
                })
                .collect::<Result<Vec<_>>>()?;

            let window_expr = create_window_expr(
                &fun,
                text.to_string(),
                &args,
                &partition_by,
                &order_by,
                Arc::new(parsed.window_frame),
                input_schema.clone(),
                false,
                parsed.distinct,
                None,
            )?;
            window_exprs.push(window_expr);
        }

        Ok(Arc::new(WindowAggExec::try_new(window_exprs, input, false)?))
    }

    fn resolve_window_fn(&self, name: &str) -> Result<WindowFunctionDefinition> {
        let mapped = map_aggregate_fn_name(name);
        use datafusion::execution::FunctionRegistry;
        if let Ok(udwf) = self.ctx.udwf(mapped) {
            return Ok(WindowFunctionDefinition::WindowUDF(udwf));
        }
        if let Ok(udaf) = self.ctx.udaf(mapped) {
            return Ok(WindowFunctionDefinition::AggregateUDF(udaf));
        }
        Err(DuckDBTranslateError::UnsupportedFunction(name.to_string()))
    }
}

fn single_column(schema: &Schema) -> Option<Arc<dyn PhysicalExpr>> {
    match schema.fields().as_ref() {
        [only] => Some(Arc::new(PhysColumn::new(only.name(), 0))),
        _ => None,
    }
}

fn map_join_type(text: &str) -> Result<JoinType> {
    Ok(match text.to_ascii_uppercase().as_str() {
        "INNER" => JoinType::Inner,
        "LEFT" => JoinType::Left,
        "RIGHT" => JoinType::Right,
        "FULL" | "OUTER" => JoinType::Full,
        "SEMI" => JoinType::LeftSemi,
        "ANTI" => JoinType::LeftAnti,
        other => return Err(DuckDBTranslateError::UnsupportedOperator(format!("join type {other}"))),
    })
}

/// If `expr` is (transitively, through identity casts) a plain column
/// reference, returns that column's name.
fn passthrough_column_name(expr: &datafusion::logical_expr::Expr) -> Option<&str> {
    match expr {
        datafusion::logical_expr::Expr::Column(c) => Some(c.name.as_str()),
        datafusion::logical_expr::Expr::Cast(c) => passthrough_column_name(&c.expr),
        _ => None,
    }
}

fn map_aggregate_fn_name(name: &str) -> &str {
    match name.to_ascii_lowercase().as_str() {
        "count_star" => "count",
        "sum_no_overflow" => "sum",
        other => Box::leak(other.to_string().into_boxed_str()),
    }
}

/// Splits `"lhs = rhs"` on a top-level (depth 0) `=` sign, or on
/// `IS [NOT] DISTINCT FROM` (DuckDB's null-safe equality, which it uses for
/// join keys derived from `IN`/`EXISTS` subqueries). Returns `None` if
/// neither is found at depth 0 (e.g. a non-equality condition).
fn split_top_level_eq(text: &str) -> Option<(String, String)> {
    for marker in [" IS NOT DISTINCT FROM ", " IS DISTINCT FROM "] {
        if let Some(idx) = text.find(marker) {
            let lhs = text[..idx].trim().to_string();
            let rhs = text[idx + marker.len()..].trim().to_string();
            return Some((lhs, rhs));
        }
    }

    let chars: Vec<char> = text.chars().collect();
    let mut depth = 0i32;
    for (i, &c) in chars.iter().enumerate() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            '=' if depth == 0 => {
                // avoid splitting on <=, >=, !=, <>
                let prev = chars.get(i.wrapping_sub(1)).copied();
                if matches!(prev, Some('<') | Some('>') | Some('!')) {
                    continue;
                }
                let lhs: String = chars[..i].iter().collect();
                let rhs: String = chars[i + 1..].iter().collect();
                return Some((lhs.trim().to_string(), rhs.trim().to_string()));
            }
            _ => {}
        }
    }
    None
}

/// Splits `"lhs OP rhs"` on a top-level (depth 0) comparison operator
/// (`<=`, `>=`, `<>`, `!=`, `=`, `<`, `>`, or `IS [NOT] DISTINCT FROM`),
/// returning the two operand texts and the matched [`Operator`]. Returns
/// `None` if no top-level comparison operator is found.
fn split_top_level_cmp(text: &str) -> Option<(String, Operator, String)> {
    for (marker, op) in [
        (" IS NOT DISTINCT FROM ", Operator::IsNotDistinctFrom),
        (" IS DISTINCT FROM ", Operator::IsDistinctFrom),
    ] {
        if let Some(idx) = text.find(marker) {
            let lhs = text[..idx].trim().to_string();
            let rhs = text[idx + marker.len()..].trim().to_string();
            return Some((lhs, op, rhs));
        }
    }

    let chars: Vec<char> = text.chars().collect();
    let mut depth = 0i32;
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            '(' => depth += 1,
            ')' => depth -= 1,
            c if depth == 0 => {
                let two: Option<Operator> = chars.get(i + 1).and_then(|&next| match (c, next) {
                    ('<', '=') => Some(Operator::LtEq),
                    ('>', '=') => Some(Operator::GtEq),
                    ('<', '>') => Some(Operator::NotEq),
                    ('!', '=') => Some(Operator::NotEq),
                    _ => None,
                });
                let (op, width) = if let Some(op) = two {
                    (Some(op), 2)
                } else {
                    (
                        match c {
                            '=' => Some(Operator::Eq),
                            '<' => Some(Operator::Lt),
                            '>' => Some(Operator::Gt),
                            _ => None,
                        },
                        1,
                    )
                };
                if let Some(op) = op {
                    let lhs: String = chars[..i].iter().collect();
                    let rhs: String = chars[i + width..].iter().collect();
                    return Some((lhs.trim().to_string(), op, rhs.trim().to_string()));
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// Concatenates two schemas' fields into one (used to build a join filter's
/// intermediate batch schema; field names may collide, which is fine since
/// resolution happens by index, not by name).
fn concat_schema(left: &Schema, right: &Schema) -> Schema {
    let fields: Vec<_> =
        left.fields().iter().cloned().chain(right.fields().iter().cloned()).collect();
    Schema::new(fields)
}

/// Rewrites every [`PhysColumn`] leaf in `expr`'s tree, adding `offset` to its
/// index. Used to re-point a `PhysicalExpr` computed against a lone (left or
/// right) join side onto the combined intermediate schema used by a
/// [`JoinFilter`].
fn shift_columns(expr: Arc<dyn PhysicalExpr>, offset: usize) -> Result<Arc<dyn PhysicalExpr>> {
    if offset == 0 {
        return Ok(expr);
    }
    if let Some(col) = expr.downcast_ref::<PhysColumn>() {
        return Ok(Arc::new(PhysColumn::new(col.name(), col.index() + offset)));
    }
    let children: Vec<Arc<dyn PhysicalExpr>> = expr.children().into_iter().cloned().collect();
    if children.is_empty() {
        return Ok(expr);
    }
    let new_children = children
        .into_iter()
        .map(|c| shift_columns(c, offset))
        .collect::<Result<Vec<_>>>()?;
    Ok(Arc::clone(&expr).with_new_children(new_children)?)
}
