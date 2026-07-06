use thiserror::Error;

#[derive(Debug, Error)]
pub enum DuckDBTranslateError {
    #[error("failed to parse DuckDB EXPLAIN JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("failed to parse DuckDB expression: {0}")]
    ExprParse(String),
    #[error("unsupported DuckDB function: {0}")]
    UnsupportedFunction(String),
    #[error("unsupported DuckDB operator: {0}")]
    UnsupportedOperator(String),
    #[error("missing required extra_info field '{field}' on operator {operator}")]
    MissingField { operator: String, field: String },
    #[error("DataFusion error while building plan: {0}")]
    DataFusion(#[from] datafusion::error::DataFusionError),
}
