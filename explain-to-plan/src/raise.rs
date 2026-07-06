//! Raises a DataFusion physical [`ExecutionPlan`] back into an equivalent
//! [`LogicalPlan`], by converting each physical operator into its logical
//! counterpart (e.g. a `HashJoinExec` becomes a `Join`). This is the mirror
//! image of [`crate::duckdb`], which lowers a database's `EXPLAIN` output
//! into an `ExecutionPlan` in the first place. Once a plan has been raised,
//! it can be turned into SQL with DataFusion's own
//! [`Unparser`](datafusion::sql::unparser::Unparser).
//!
//! # Scope
//!
//! This module covers exactly the operator/expression surface
//! [`crate::duckdb::translate`] produces: `EmptyExec` (scan leaf),
//! `FilterExec`, `ProjectionExec`, `HashJoinExec`, `NestedLoopJoinExec`,
//! `CrossJoinExec`, `AggregateExec` (single/ungrouped-or-simple-grouped
//! only — no grouping sets), `SortExec`, `GlobalLimitExec`, `UnionExec`, and
//! `WindowAggExec` backed by an aggregate window function
//! (`PlainAggregateWindowExpr`/`SlidingAggregateWindowExpr`).
//!
//! Not supported, and reported as [`RaiseToLogicalError::UnsupportedOperator`]
//! or [`RaiseToLogicalError::UnsupportedExpr`] rather than silently producing
//! an incorrect plan:
//! - Any other physical operator (e.g. `RepartitionExec`, `CoalesceBatchesExec`
//!   — these never appear in a plan lowered by [`crate::duckdb::translate`],
//!   but could in a plan built some other way).
//! - Window functions backed by `StandardWindowExpr` (`ROW_NUMBER`, `RANK`,
//!   `LAG`/`LEAD`, ...) — DataFusion's physical layer does not expose the
//!   underlying function definition for these generically.
//! - A `HashJoinExec` equi-join key that isn't a plain column reference (an
//!   "expression join").
//! - A scan leaf whose schema is missing the
//!   [`crate::TABLE_NAME_METADATA_KEY`] tag.

use std::sync::Arc;

use datafusion::common::{Column as LogicalColumn, DFSchema, NullEquality};
use datafusion::datasource::{empty::EmptyTable, provider_as_source};
use datafusion::logical_expr::{
    Expr, LogicalPlan, LogicalPlanBuilder,
    expr::{AggregateFunction, ScalarFunction, Sort as LogicalSort, WindowFunction, WindowFunctionParams},
};
use datafusion::physical_expr::{
    ScalarFunctionExpr,
    aggregate::AggregateFunctionExpr,
    expressions::{
        BinaryExpr as PhysBinaryExpr, CaseExpr, CastExpr, Column as PhysColumn, InListExpr,
        IsNotNullExpr, IsNullExpr, LikeExpr, Literal as PhysLiteral, NegativeExpr, NotExpr,
    },
    window::{PlainAggregateWindowExpr, SlidingAggregateWindowExpr},
};
use datafusion::physical_plan::{
    ExecutionPlan, PhysicalExpr,
    aggregates::AggregateExec,
    empty::EmptyExec,
    filter::FilterExec,
    joins::{CrossJoinExec, HashJoinExec, NestedLoopJoinExec},
    limit::GlobalLimitExec,
    projection::ProjectionExec,
    sorts::sort::SortExec,
    union::UnionExec,
    windows::WindowAggExec,
};
use thiserror::Error;

use crate::TABLE_NAME_METADATA_KEY;

#[derive(Debug, Error)]
pub enum RaiseToLogicalError {
    #[error("unsupported physical operator: {0}")]
    UnsupportedOperator(String),
    #[error("unsupported physical expression: {0}")]
    UnsupportedExpr(String),
    #[error(
        "scan leaf schema is missing the '{TABLE_NAME_METADATA_KEY}' metadata key needed to recover its table name"
    )]
    MissingTableName,
    #[error("DataFusion error while building plan: {0}")]
    DataFusion(#[from] datafusion::error::DataFusionError),
}

type Result<T> = std::result::Result<T, RaiseToLogicalError>;

/// Raise `plan` into an equivalent [`LogicalPlan`]. See the module
/// documentation for exactly which physical operators/expressions this
/// supports.
pub fn raise_to_logical(plan: &Arc<dyn ExecutionPlan>) -> Result<LogicalPlan> {
    if let Some(scan) = plan.downcast_ref::<EmptyExec>() {
        return raise_scan(scan, plan);
    }
    if let Some(f) = plan.downcast_ref::<FilterExec>() {
        return raise_filter(f);
    }
    if let Some(p) = plan.downcast_ref::<ProjectionExec>() {
        return raise_projection(p);
    }
    if let Some(j) = plan.downcast_ref::<HashJoinExec>() {
        return raise_hash_join(j);
    }
    if let Some(j) = plan.downcast_ref::<NestedLoopJoinExec>() {
        return raise_nested_loop_join(j);
    }
    if let Some(j) = plan.downcast_ref::<CrossJoinExec>() {
        return raise_cross_join(j);
    }
    if let Some(a) = plan.downcast_ref::<AggregateExec>() {
        return raise_aggregate(a);
    }
    if let Some(s) = plan.downcast_ref::<SortExec>() {
        return raise_sort(s);
    }
    if let Some(l) = plan.downcast_ref::<GlobalLimitExec>() {
        return raise_limit(l);
    }
    if let Some(u) = plan.downcast_ref::<UnionExec>() {
        return raise_union(u);
    }
    if let Some(w) = plan.downcast_ref::<WindowAggExec>() {
        return raise_window(w);
    }

    Err(RaiseToLogicalError::UnsupportedOperator(plan.name().to_string()))
}

fn raise_scan(_scan: &EmptyExec, plan: &Arc<dyn ExecutionPlan>) -> Result<LogicalPlan> {
    let schema = plan.schema();
    let table_name = schema
        .metadata()
        .get(TABLE_NAME_METADATA_KEY)
        .cloned()
        .ok_or(RaiseToLogicalError::MissingTableName)?;

    let provider = Arc::new(EmptyTable::new(schema));
    let source = provider_as_source(provider);
    Ok(LogicalPlanBuilder::scan(table_name.as_str(), source, None)?.build()?)
}

fn raise_filter(f: &FilterExec) -> Result<LogicalPlan> {
    let input = raise_to_logical(f.input())?;
    let predicate = to_logical_expr(f.predicate(), input.schema())?;
    Ok(LogicalPlanBuilder::from(input).filter(predicate)?.build()?)
}

fn raise_projection(p: &ProjectionExec) -> Result<LogicalPlan> {
    let input = raise_to_logical(p.input())?;
    let schema = input.schema().clone();
    let mut name_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let exprs: Vec<Expr> = p
        .expr()
        .iter()
        .map(|pe| {
            let e = to_logical_expr(&pe.expr, &schema)?;
            let count = name_counts.entry(pe.alias.clone()).or_insert(0);
            *count += 1;
            if *count > 1 {
                // A query can genuinely select the very same column (or two
                // expressions that happen to share DuckDB's derived name)
                // twice, e.g. `SELECT id, id FROM t`. Left alone this would
                // produce two identically-named output fields — even mixing
                // a qualified and an unqualified field with the same bare
                // name is rejected as ambiguous — so every repeat past the
                // first gets a disambiguating suffix.
                return Ok(e.alias(format!("{}_{count}", pe.alias)));
            }
            // A plain column reference already named `alias` is a pure
            // passthrough (e.g. DuckDB's positional `#N` projections, which
            // very commonly just re-select an input column unchanged).
            // Re-aliasing it anyway would strip its table qualifier even
            // though nothing about it actually changed, which then makes two
            // differently-qualified inputs with the same bare column name
            // (e.g. both sides of a join projecting their own "id") collide
            // into a single ambiguous unqualified field once each side drops
            // its qualifier. Leave true passthroughs alone; only alias
            // genuinely computed expressions or renames.
            let is_passthrough = matches!(&e, Expr::Column(c) if c.name == pe.alias);
            Ok(if is_passthrough { e } else { e.alias(pe.alias.clone()) })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(LogicalPlanBuilder::from(input).project(exprs)?.build()?)
}

/// Resolve an equi-join key to a [`LogicalColumn`] by position against
/// `schema` (one side's already-raised logical schema). We resolve by
/// *position*, not by the physical column's own reported name, because a
/// physical child's output field name can be a synthetic artifact (e.g.
/// DuckDB positional projections translate to a physical schema field
/// literally named `"#0"`) that has no meaning in the schema we just
/// rebuilt — only its position lines up.
///
/// Only plain column references are supported — an "expression join" key
/// would need an extra `Projection` materializing the computed key ahead of
/// the join, which [`crate::duckdb::translate`] never produces, so we don't
/// handle it here.
fn join_key_column(expr: &Arc<dyn PhysicalExpr>, schema: &DFSchema) -> Result<LogicalColumn> {
    let c = expr.downcast_ref::<PhysColumn>().ok_or_else(|| {
        RaiseToLogicalError::UnsupportedExpr("join key is not a plain column reference".to_string())
    })?;
    let (qualifier, field) = schema.qualified_field(c.index());
    Ok(LogicalColumn::new(qualifier.cloned(), field.name().clone()))
}

fn raise_hash_join(j: &HashJoinExec) -> Result<LogicalPlan> {
    let left = raise_to_logical(j.left())?;
    let right = raise_to_logical(j.right())?;

    let mut left_keys = Vec::new();
    let mut right_keys = Vec::new();
    for (l, r) in j.on() {
        left_keys.push(join_key_column(l, left.schema())?);
        right_keys.push(join_key_column(r, right.schema())?);
    }

    let filter = match j.filter() {
        Some(jf) => {
            let merged = left.schema().join(right.schema())?;
            Some(to_logical_expr(jf.expression(), &merged)?)
        }
        None => None,
    };

    Ok(LogicalPlanBuilder::from(left)
        .join_detailed(
            right,
            *j.join_type(),
            (left_keys, right_keys),
            filter,
            NullEquality::NullEqualsNothing,
        )?
        .build()?)
}

fn raise_nested_loop_join(j: &NestedLoopJoinExec) -> Result<LogicalPlan> {
    let left = raise_to_logical(j.left())?;
    let right = raise_to_logical(j.right())?;

    let filter = match j.filter() {
        Some(jf) => {
            let merged = left.schema().join(right.schema())?;
            to_logical_expr(jf.expression(), &merged)?
        }
        None => {
            return Err(RaiseToLogicalError::UnsupportedOperator(
                "NestedLoopJoinExec with no filter".to_string(),
            ));
        }
    };

    Ok(LogicalPlanBuilder::from(left)
        .join_detailed(
            right,
            *j.join_type(),
            (Vec::<LogicalColumn>::new(), Vec::<LogicalColumn>::new()),
            Some(filter),
            NullEquality::NullEqualsNothing,
        )?
        .build()?)
}

fn raise_cross_join(j: &CrossJoinExec) -> Result<LogicalPlan> {
    let left = raise_to_logical(j.left())?;
    let right = raise_to_logical(j.right())?;
    Ok(LogicalPlanBuilder::from(left).cross_join(right)?.build()?)
}

fn aggregate_call(agg: &Arc<AggregateFunctionExpr>, schema: &DFSchema) -> Result<Expr> {
    let args = agg
        .expressions()
        .iter()
        .map(|e| to_logical_expr(e, schema))
        .collect::<Result<Vec<_>>>()?;
    Ok(Expr::AggregateFunction(AggregateFunction::new_udf(
        Arc::new(agg.fun().clone()),
        args,
        agg.is_distinct(),
        None,
        vec![],
        None,
    )))
}

fn raise_aggregate(a: &AggregateExec) -> Result<LogicalPlan> {
    if !a.group_expr().is_single() {
        return Err(RaiseToLogicalError::UnsupportedOperator(
            "AggregateExec with grouping sets".to_string(),
        ));
    }
    let input = raise_to_logical(a.input())?;
    let schema = input.schema().clone();

    let group_exprs: Vec<Expr> = a
        .group_expr()
        .expr()
        .iter()
        .map(|(e, _alias)| to_logical_expr(e, &schema))
        .collect::<Result<Vec<_>>>()?;

    let aggr_exprs: Vec<Expr> = a
        .aggr_expr()
        .iter()
        .map(|agg| aggregate_call(agg, &schema))
        .collect::<Result<Vec<_>>>()?;

    Ok(LogicalPlanBuilder::from(input)
        .aggregate(group_exprs, aggr_exprs)?
        .build()?)
}

fn raise_sort(s: &SortExec) -> Result<LogicalPlan> {
    let input = raise_to_logical(s.input())?;
    let schema = input.schema().clone();
    let sort_exprs: Vec<LogicalSort> = s
        .expr()
        .iter()
        .map(|pse| {
            Ok(LogicalSort {
                expr: to_logical_expr(&pse.expr, &schema)?,
                asc: !pse.options.descending,
                nulls_first: pse.options.nulls_first,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(LogicalPlanBuilder::from(input)
        .sort_with_limit(sort_exprs, s.fetch())?
        .build()?)
}

fn raise_limit(l: &GlobalLimitExec) -> Result<LogicalPlan> {
    let input = raise_to_logical(l.input())?;
    Ok(LogicalPlanBuilder::from(input).limit(l.skip(), l.fetch())?.build()?)
}

fn raise_union(u: &UnionExec) -> Result<LogicalPlan> {
    let mut inputs = u.inputs().iter();
    let first = inputs
        .next()
        .ok_or_else(|| RaiseToLogicalError::UnsupportedOperator("UnionExec with no inputs".to_string()))?;
    let mut builder = LogicalPlanBuilder::from(raise_to_logical(first)?);
    for input in inputs {
        builder = builder.union(raise_to_logical(input)?)?;
    }
    Ok(builder.build()?)
}

/// Raise a `WindowAggExec` whose window expressions are aggregate-backed
/// (`SUM(x) OVER (...)`, `COUNT(x) OVER (...)`, ...). Ranking/value window
/// functions (`ROW_NUMBER`, `RANK`, `LAG`/`LEAD`, ...) are backed by
/// `StandardWindowExpr`, whose underlying function definition DataFusion's
/// physical layer does not expose generically — those return
/// [`RaiseToLogicalError::UnsupportedOperator`].
fn raise_window(w: &WindowAggExec) -> Result<LogicalPlan> {
    let input = raise_to_logical(w.input())?;
    let schema = input.schema().clone();

    let window_exprs: Vec<Expr> = w
        .window_expr()
        .iter()
        .map(|we| {
            let any = we.as_any();
            let (fun, args, distinct) = if let Some(plain) = any.downcast_ref::<PlainAggregateWindowExpr>() {
                let agg = plain.get_aggregate_expr();
                (Arc::new(agg.fun().clone()).into(), agg.expressions(), agg.is_distinct())
            } else if let Some(sliding) = any.downcast_ref::<SlidingAggregateWindowExpr>() {
                let agg = sliding.get_aggregate_expr();
                (Arc::new(agg.fun().clone()).into(), agg.expressions(), agg.is_distinct())
            } else {
                return Err(RaiseToLogicalError::UnsupportedOperator(format!(
                    "window function '{}' is not backed by an aggregate (ranking/value window \
                     functions like ROW_NUMBER/RANK/LAG/LEAD are not supported)",
                    we.name()
                )));
            };

            let args: Vec<Expr> = args.iter().map(|e| to_logical_expr(e, &schema)).collect::<Result<Vec<_>>>()?;
            let partition_by: Vec<Expr> = we
                .partition_by()
                .iter()
                .map(|e| to_logical_expr(e, &schema))
                .collect::<Result<Vec<_>>>()?;
            let order_by: Vec<LogicalSort> = we
                .order_by()
                .iter()
                .map(|pse| {
                    Ok(LogicalSort {
                        expr: to_logical_expr(&pse.expr, &schema)?,
                        asc: !pse.options.descending,
                        nulls_first: pse.options.nulls_first,
                    })
                })
                .collect::<Result<Vec<_>>>()?;

            Ok(Expr::WindowFunction(Box::new(WindowFunction {
                fun,
                params: WindowFunctionParams {
                    args,
                    partition_by,
                    order_by,
                    window_frame: (**we.get_window_frame()).clone(),
                    filter: None,
                    null_treatment: None,
                    distinct,
                },
            }))
            .alias(we.name().to_string()))
        })
        .collect::<Result<Vec<_>>>()?;

    // `WindowAggExec` appends its window columns to the input's schema, so a
    // plain logical `Window` (which does the same) is a direct equivalent.
    Ok(LogicalPlanBuilder::from(input).window(window_exprs)?.build()?)
}

/// Converts a physical expression back into a logical [`Expr`]. Covers
/// exactly the expression node types [`crate::duckdb::expr::to_physical`] can
/// produce; anything else returns [`RaiseToLogicalError::UnsupportedExpr`].
///
/// `schema` is the already-raised logical schema of whatever plan node this
/// expression is evaluated against (its input, for a `Filter`/`Projection`/
/// `Aggregate`/etc., or the concatenation of both sides' schemas for a join
/// filter). A physical [`PhysColumn`] is resolved by *position* against it,
/// not by its own reported name — a physical child's field name can be a
/// synthetic artifact of how it was produced (e.g. DuckDB positional
/// projections translate to a physical schema field literally named `"#0"`)
/// with no meaning in the schema we've just rebuilt; only its position
/// reliably lines up.
fn to_logical_expr(expr: &Arc<dyn PhysicalExpr>, schema: &DFSchema) -> Result<Expr> {
    if let Some(c) = expr.downcast_ref::<PhysColumn>() {
        let (qualifier, field) = schema.qualified_field(c.index());
        return Ok(Expr::Column(LogicalColumn::new(qualifier.cloned(), field.name().clone())));
    }
    if let Some(l) = expr.downcast_ref::<PhysLiteral>() {
        return Ok(Expr::Literal(l.value().clone(), None));
    }
    if let Some(b) = expr.downcast_ref::<PhysBinaryExpr>() {
        let left = to_logical_expr(b.left(), schema)?;
        let right = to_logical_expr(b.right(), schema)?;
        return Ok(Expr::BinaryExpr(datafusion::logical_expr::BinaryExpr::new(
            Box::new(left),
            *b.op(),
            Box::new(right),
        )));
    }
    if let Some(n) = expr.downcast_ref::<NotExpr>() {
        return Ok(Expr::Not(Box::new(to_logical_expr(n.arg(), schema)?)));
    }
    if let Some(n) = expr.downcast_ref::<IsNullExpr>() {
        return Ok(Expr::IsNull(Box::new(to_logical_expr(n.arg(), schema)?)));
    }
    if let Some(n) = expr.downcast_ref::<IsNotNullExpr>() {
        return Ok(Expr::IsNotNull(Box::new(to_logical_expr(n.arg(), schema)?)));
    }
    if let Some(n) = expr.downcast_ref::<NegativeExpr>() {
        return Ok(Expr::Negative(Box::new(to_logical_expr(n.arg(), schema)?)));
    }
    if let Some(c) = expr.downcast_ref::<CastExpr>() {
        let inner = to_logical_expr(c.expr(), schema)?;
        return Ok(Expr::Cast(datafusion::logical_expr::Cast::new(
            Box::new(inner),
            c.cast_type().clone(),
        )));
    }
    if let Some(l) = expr.downcast_ref::<InListExpr>() {
        let inner = to_logical_expr(l.expr(), schema)?;
        let list = l
            .list()
            .iter()
            .map(|e| to_logical_expr(e, schema))
            .collect::<Result<Vec<_>>>()?;
        return Ok(Expr::InList(datafusion::logical_expr::expr::InList {
            expr: Box::new(inner),
            list,
            negated: l.negated(),
        }));
    }
    if let Some(c) = expr.downcast_ref::<CaseExpr>() {
        let operand = c.expr().map(|e| to_logical_expr(e, schema)).transpose()?.map(Box::new);
        let when_then_expr = c
            .when_then_expr()
            .iter()
            .map(|(when, then)| {
                Ok((Box::new(to_logical_expr(when, schema)?), Box::new(to_logical_expr(then, schema)?)))
            })
            .collect::<Result<Vec<_>>>()?;
        let else_expr = c.else_expr().map(|e| to_logical_expr(e, schema)).transpose()?.map(Box::new);
        return Ok(Expr::Case(datafusion::logical_expr::Case {
            expr: operand,
            when_then_expr,
            else_expr,
        }));
    }
    if let Some(l) = expr.downcast_ref::<LikeExpr>() {
        let inner = to_logical_expr(l.expr(), schema)?;
        let pattern = to_logical_expr(l.pattern(), schema)?;
        return Ok(Expr::Like(datafusion::logical_expr::Like {
            negated: l.negated(),
            expr: Box::new(inner),
            pattern: Box::new(pattern),
            escape_char: None,
            case_insensitive: l.case_insensitive(),
        }));
    }
    if let Some(f) = expr.downcast_ref::<ScalarFunctionExpr>() {
        let args = f.args().iter().map(|e| to_logical_expr(e, schema)).collect::<Result<Vec<_>>>()?;
        return Ok(Expr::ScalarFunction(ScalarFunction {
            func: Arc::new(f.fun().clone()),
            args,
        }));
    }

    Err(RaiseToLogicalError::UnsupportedExpr(expr.to_string()))
}
