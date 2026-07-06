//! Serde model for DuckDB's JSON `EXPLAIN` / `EXPLAIN ANALYZE` output.
//!
//! DuckDB emits two distinct shapes depending on the statement:
//!
//! - `EXPLAIN (FORMAT JSON) <query>` returns a JSON array of one root node
//!   (field `name` for the operator), e.g. `[{"name": "PROJECTION", ...}]`.
//! - `EXPLAIN (ANALYZE, FORMAT JSON) <query>` returns a single top-level
//!   object with query-wide profiling stats and a `children` array; the
//!   first (and only) child is always an `EXPLAIN_ANALYZE` wrapper node
//!   whose own child is the real plan root. Operator nodes here use
//!   `operator_name` rather than `name`.
//!
//! Both shapes nest plan nodes identically otherwise, so [`DuckDBNode`]
//! models a single node and accepts either key.

use std::collections::HashMap;

use serde::Deserialize;

/// A single node in a DuckDB physical plan tree, as emitted by `EXPLAIN` or
/// `EXPLAIN ANALYZE` with `FORMAT JSON`.
#[derive(Debug, Clone, Deserialize)]
pub struct DuckDBNode {
    #[serde(alias = "name")]
    pub operator_name: String,
    #[serde(default)]
    pub extra_info: HashMap<String, ExtraInfoValue>,
    #[serde(default)]
    pub children: Vec<DuckDBNode>,
    #[serde(default)]
    pub operator_cardinality: Option<u64>,
}

/// `extra_info` values are heterogeneous: DuckDB emits a bare string when an
/// entry has a single item (e.g. one projection expression) and a JSON array
/// when it has several. This normalizes both into a `Vec<String>`.
#[derive(Debug, Clone)]
pub struct ExtraInfoValue(pub Vec<String>);

impl ExtraInfoValue {
    pub fn first(&self) -> Option<&str> {
        self.0.first().map(String::as_str)
    }

    pub fn iter(&self) -> impl Iterator<Item = &str> {
        self.0.iter().map(String::as_str)
    }
}

impl<'de> Deserialize<'de> for ExtraInfoValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        fn value_to_string(v: &serde_json::Value) -> String {
            match v {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            }
        }

        Ok(match serde_json::Value::deserialize(deserializer)? {
            serde_json::Value::Array(vs) => ExtraInfoValue(vs.iter().map(value_to_string).collect()),
            other => ExtraInfoValue(vec![value_to_string(&other)]),
        })
    }
}

/// The top-level shape produced by `EXPLAIN (ANALYZE, FORMAT JSON) ...`: a
/// query-wide stats object whose only meaningful field for us is `children`,
/// which holds a single `EXPLAIN_ANALYZE` wrapper node.
#[derive(Debug, Clone, Deserialize)]
pub struct DuckDBAnalyzeOutput {
    #[serde(default)]
    pub children: Vec<DuckDBNode>,
}

/// Parses the raw text produced by a DuckDB `EXPLAIN` statement (with
/// `FORMAT JSON`) into a single plan root node, regardless of whether it was
/// produced with or without `ANALYZE`.
pub fn parse_root(explain: &str) -> Result<DuckDBNode, serde_json::Error> {
    let trimmed = explain.trim();

    if trimmed.starts_with('[') {
        // Plain `EXPLAIN (FORMAT JSON)`: a JSON array with a single root node.
        let mut nodes: Vec<DuckDBNode> = serde_json::from_str(trimmed)?;
        return match nodes.pop() {
            Some(node) => Ok(node),
            None => Err(serde::de::Error::custom(
                "EXPLAIN (FORMAT JSON) output was an empty array",
            )),
        };
    }

    // `EXPLAIN (ANALYZE, FORMAT JSON)`: object wrapping an EXPLAIN_ANALYZE
    // node, whose single child is the real plan root.
    let analyze: DuckDBAnalyzeOutput = serde_json::from_str(trimmed)?;
    let wrapper = analyze.children.into_iter().next().ok_or_else(|| {
        serde::de::Error::custom("EXPLAIN ANALYZE output had no EXPLAIN_ANALYZE wrapper node")
    })?;
    if wrapper.children.is_empty() {
        Ok(wrapper)
    } else {
        Ok(wrapper.children.into_iter().next().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_plain_explain_array() {
        let json = r#"[{"name": "PROJECTION", "extra_info": {"Projections": "id"}, "children": []}]"#;
        let node = parse_root(json).unwrap();
        assert_eq!(node.operator_name, "PROJECTION");
        assert_eq!(node.extra_info["Projections"].first(), Some("id"));
    }

    #[test]
    fn parses_analyze_wrapper() {
        let json = r#"{
            "children": [
                {
                    "operator_name": "EXPLAIN_ANALYZE",
                    "extra_info": {},
                    "children": [
                        {"operator_name": "SEQ_SCAN", "extra_info": {"Table": "t1"}, "children": []}
                    ]
                }
            ]
        }"#;
        let node = parse_root(json).unwrap();
        assert_eq!(node.operator_name, "SEQ_SCAN");
    }

    #[test]
    fn extra_info_normalizes_single_and_many() {
        let json = r#"{"operator_name": "PROJECTION", "extra_info": {"Projections": ["a", "b"], "Table": "t"}, "children": []}"#;
        let node: DuckDBNode = serde_json::from_str(json).unwrap();
        assert_eq!(node.extra_info["Projections"].0, vec!["a", "b"]);
        assert_eq!(node.extra_info["Table"].0, vec!["t"]);
    }
}
