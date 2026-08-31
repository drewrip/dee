//! Content hashing.
//!
//! Used for two things with different purposes but the same requirement: a
//! DAG's `content_hash`, which makes resubmission idempotent, and a
//! connection's `config_hash`, which keys the connector cache so an edited
//! connection stops reusing the old pool.
//!
//! Both need a hash that is stable across serializations of the same value,
//! so the input is canonicalized first: object keys sorted, no insignificant
//! whitespace. Serde's map ordering is otherwise whatever the source JSON had.

use dee::file::DagFile;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

/// Recursively sort every object's keys so two equal values that were written
/// with different key order canonicalize identically.
pub fn canonicalize(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut sorted = Map::new();
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            for key in keys {
                sorted.insert(key.clone(), canonicalize(&map[key]));
            }
            Value::Object(sorted)
        }
        Value::Array(items) => Value::Array(items.iter().map(canonicalize).collect()),
        other => other.clone(),
    }
}

/// Canonical JSON text for `value`.
pub fn canonical_json(value: &Value) -> String {
    canonicalize(value).to_string()
}

/// Lowercase hex sha256 of the canonical form of `value`.
pub fn content_hash(value: &Value) -> String {
    let mut hasher = Sha256::new();
    hasher.update(canonical_json(value).as_bytes());
    format!("{:x}", hasher.finalize())
}

/// The canonical JSON of a DAG definition, for content addressing.
///
/// Beyond sorting object keys, this imposes an order on the parts of a
/// `DagFile` whose order carries no meaning: nodes by id, sources by name, and
/// each node's `depends_on`. That matters more than it looks. `impl From<Dag>
/// for DagFile` iterates a `HashMap`, so the optimizer emits nodes in a
/// different order every run -- without this, re-submitting an unchanged DAG
/// would look like a new version each time.
pub fn canonical_dag_json(dag: &DagFile) -> Result<String, serde_json::Error> {
    let mut value = serde_json::to_value(dag)?;

    if let Some(nodes) = value.get_mut("nodes").and_then(Value::as_array_mut) {
        for node in nodes.iter_mut() {
            if let Some(deps) = node.get_mut("depends_on").and_then(Value::as_array_mut) {
                deps.sort_by(|a, b| a.as_str().unwrap_or("").cmp(b.as_str().unwrap_or("")));
            }
        }
        nodes.sort_by(|a, b| sort_key(a, "id").cmp(&sort_key(b, "id")));
    }
    if let Some(sources) = value.get_mut("sources").and_then(Value::as_array_mut) {
        sources.sort_by(|a, b| sort_key(a, "name").cmp(&sort_key(b, "name")));
    }

    Ok(canonical_json(&value))
}

fn sort_key<'a>(value: &'a Value, field: &str) -> &'a str {
    value.get(field).and_then(Value::as_str).unwrap_or("")
}

/// Lowercase hex sha256 of a DAG definition's canonical form.
pub fn dag_hash(dag: &DagFile) -> Result<String, serde_json::Error> {
    let mut hasher = Sha256::new();
    hasher.update(canonical_dag_json(dag)?.as_bytes());
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_key_order_does_not_change_the_hash() {
        let a: Value = serde_json::from_str(r#"{"b": 1, "a": {"d": 2, "c": 3}}"#).unwrap();
        let b: Value = serde_json::from_str(r#"{"a": {"c": 3, "d": 2}, "b": 1}"#).unwrap();
        assert_eq!(content_hash(&a), content_hash(&b));
    }

    #[test]
    fn test_whitespace_does_not_change_the_hash() {
        let a: Value = serde_json::from_str(r#"{"a":1}"#).unwrap();
        let b: Value = serde_json::from_str("{\n  \"a\" : 1\n}").unwrap();
        assert_eq!(content_hash(&a), content_hash(&b));
    }

    #[test]
    fn test_a_changed_value_changes_the_hash() {
        assert_ne!(
            content_hash(&json!({"query": "select 1"})),
            content_hash(&json!({"query": "select 2"}))
        );
    }

    #[test]
    fn test_array_order_is_significant() {
        // Arrays carry meaning that objects' key order does not: a DAG's node
        // list order is incidental, but this function must not assume that for
        // every array it sees. Callers that want order-insensitivity sort
        // before hashing.
        assert_ne!(content_hash(&json!(["a", "b"])), content_hash(&json!(["b", "a"])));
    }
    fn dag(json: &str) -> DagFile {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn test_node_order_does_not_change_a_dags_hash() {
        // `impl From<Dag> for DagFile` walks a HashMap, so the optimizer emits
        // nodes in an arbitrary order. If that changed the hash, every
        // resubmission would look like a new version.
        let a = dag(r#"{"nodes":[
            {"id":"a","query_text":"select 1","depends_on":[],"materialize":"view"},
            {"id":"b","query_text":"select 2","depends_on":["a"],"materialize":"table"}
        ],"sources":[]}"#);
        let b = dag(r#"{"nodes":[
            {"id":"b","query_text":"select 2","depends_on":["a"],"materialize":"table"},
            {"id":"a","query_text":"select 1","depends_on":[],"materialize":"view"}
        ],"sources":[]}"#);
        assert_eq!(dag_hash(&a).unwrap(), dag_hash(&b).unwrap());
    }

    #[test]
    fn test_dependency_order_does_not_change_a_dags_hash() {
        // depends_on becomes a HashSet inside `Dag`, so its order out is not
        // its order in.
        let a = dag(r#"{"nodes":[
            {"id":"c","query_text":"q","depends_on":["a","b"],"materialize":"view"}
        ],"sources":[]}"#);
        let b = dag(r#"{"nodes":[
            {"id":"c","query_text":"q","depends_on":["b","a"],"materialize":"view"}
        ],"sources":[]}"#);
        assert_eq!(dag_hash(&a).unwrap(), dag_hash(&b).unwrap());
    }

    #[test]
    fn test_source_order_does_not_change_a_dags_hash() {
        let a = dag(r#"{"nodes":[],"sources":[{"name":"x","columns":[]},{"name":"y","columns":[]}]}"#);
        let b = dag(r#"{"nodes":[],"sources":[{"name":"y","columns":[]},{"name":"x","columns":[]}]}"#);
        assert_eq!(dag_hash(&a).unwrap(), dag_hash(&b).unwrap());
    }

    #[test]
    fn test_a_changed_query_changes_a_dags_hash() {
        let a = dag(r#"{"nodes":[{"id":"a","query_text":"select 1","depends_on":[],"materialize":"view"}],"sources":[]}"#);
        let b = dag(r#"{"nodes":[{"id":"a","query_text":"select 2","depends_on":[],"materialize":"view"}],"sources":[]}"#);
        assert_ne!(dag_hash(&a).unwrap(), dag_hash(&b).unwrap());
    }

    #[test]
    fn test_a_changed_materialization_changes_a_dags_hash() {
        // This is the optimizer's whole output, so it must never be mistaken
        // for an unchanged DAG.
        let a = dag(r#"{"nodes":[{"id":"a","query_text":"q","depends_on":[],"materialize":"view"}],"sources":[]}"#);
        let b = dag(r#"{"nodes":[{"id":"a","query_text":"q","depends_on":[],"materialize":"temp_table"}],"sources":[]}"#);
        assert_ne!(dag_hash(&a).unwrap(), dag_hash(&b).unwrap());
    }

    #[test]
    fn test_a_dropped_dependency_changes_a_dags_hash() {
        let a = dag(r#"{"nodes":[{"id":"c","query_text":"q","depends_on":["a","b"],"materialize":"view"}],"sources":[]}"#);
        let b = dag(r#"{"nodes":[{"id":"c","query_text":"q","depends_on":["a"],"materialize":"view"}],"sources":[]}"#);
        assert_ne!(dag_hash(&a).unwrap(), dag_hash(&b).unwrap());
    }
}
