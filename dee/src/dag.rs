use std::{collections::HashMap, sync::Arc};

use std::collections::HashSet;

use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use thiserror::Error;

use crate::{
    file::{DagFile, DagFileNode, DagFileSource},
    graph::Graph,
};

#[derive(Error, Debug)]
pub enum FormatError {
    #[error("problem with parsing Dag file - {0}")]
    Parser(String),
}

/// Interal DAG representation

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaterializeMode {
    View,
    Table,
    TempTable,
}

impl MaterializeMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            MaterializeMode::View => "view",
            MaterializeMode::Table => "table",
            MaterializeMode::TempTable => "temp_table",
        }
    }
}

impl From<String> for MaterializeMode {
    fn from(value: String) -> Self {
        match value.to_lowercase().as_str() {
            "table" => MaterializeMode::Table,
            "temp_table" => MaterializeMode::TempTable,
            _ => MaterializeMode::View,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SourceNode {
    pub name: String,
    pub schema: SchemaRef,
}

impl TryFrom<DagFileSource> for SourceNode {
    type Error = FormatError;
    fn try_from(value: DagFileSource) -> Result<Self, Self::Error> {
        let name = value.name;
        let fields: Result<Vec<Field>, FormatError> = value
            .columns
            .iter()
            .map(|c| {
                c.data_type
                    .parse::<DataType>()
                    .map_err(|_| {
                        FormatError::Parser(format!(
                            "can't parse data type {}",
                            c.data_type.clone()
                        ))
                    })
                    .and_then(|dt| Ok(Field::new(c.name.clone(), dt, false)))
            })
            .collect();
        let schema = Arc::new(Schema::new(fields?));
        Ok(Self { name, schema })
    }
}

#[derive(Clone, Debug)]
pub struct TransformNode {
    pub id: String,
    pub query_text: String,
    pub materialize: MaterializeMode,
    pub depends_on: HashSet<String>,
    /// Arrow schema of this node's output, populated by `Executor::resolve_schemas`.
    /// `None` when parsed from a file or not yet resolved.
    pub schema: Option<SchemaRef>,
}

impl From<DagFileNode> for TransformNode {
    fn from(value: DagFileNode) -> Self {
        // If materialize strategy isn't provided, default to view
        let materialize = match value.materialize {
            Some(s) => MaterializeMode::from(s),
            None => MaterializeMode::View,
        };

        Self {
            id: value.id,
            query_text: value.query_text,
            materialize,
            depends_on: HashSet::from_iter(value.depends_on),
            schema: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Dag {
    pub db: String,
    pub nodes: Graph,
    pub sources: Vec<SourceNode>,
}

impl TryFrom<DagFile> for Dag {
    type Error = FormatError;
    fn try_from(value: DagFile) -> Result<Self, Self::Error> {
        let dialect = match value.metadata {
            Some(meta) => meta.sql_dialect.unwrap_or("Unknown".into()),
            None => "Unknown".into(),
        };
        let sources: Vec<SourceNode> = value
            .sources
            .iter()
            .cloned()
            .map(TryFrom::try_from)
            .collect::<Result<Vec<SourceNode>, FormatError>>()?;

        let nodes: Vec<TransformNode> = value.nodes.iter().cloned().map(From::from).collect();
        let mut node_map = HashMap::new();
        for node in nodes {
            node_map.insert(node.id.clone(), node);
        }
        let graph = Graph::new(node_map);
        graph
            .check()
            .map_err(|e| FormatError::Parser(format!("bad graph - {}", e)))?;
        for sink_id in graph.sinks() {
            if let Some(node) = graph.get(sink_id.clone()) {
                if matches!(node.materialize, MaterializeMode::TempTable) {
                    log::warn!(
                        "Node '{}' uses 'temp_table' materialization but has no dependents. Temp tables are intended as intermediate nodes and a sink temp_table is probably not desired.",
                        sink_id
                    );
                }
            }
        }
        Ok(Self {
            db: dialect,
            nodes: graph,
            sources,
        })
    }
}

fn transform_to_file_node(value: &TransformNode) -> DagFileNode {
    DagFileNode {
        id: value.id.clone(),
        query_text: value.query_text.clone(),
        depends_on: value.depends_on.clone().into_iter().collect(),
        materialize: Some(value.materialize.as_str().to_string()),
    }
}

impl From<Dag> for DagFile {
    fn from(value: Dag) -> DagFile {
        let nodes = value
            .nodes
            .nodes()
            .map(|n| transform_to_file_node(n))
            .collect();

        let sources = value
            .sources
            .iter()
            .map(|s| DagFileSource {
                name: s.name.clone(),
                columns: s
                    .schema
                    .flattened_fields()
                    .iter()
                    .map(|f| crate::file::DagColumn {
                        name: f.name().clone(),
                        data_type: f.data_type().to_string(),
                    })
                    .collect(),
            })
            .collect();
        DagFile {
            metadata: Some(crate::file::DagFileMetadata {
                sql_dialect: Some(value.db.clone()),
            }),
            sources,
            nodes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::file::DagFileNode;

    #[test]
    fn test_materialize_mode_from_string() {
        assert_eq!(
            MaterializeMode::from("table".to_string()),
            MaterializeMode::Table
        );
        assert_eq!(
            MaterializeMode::from("TABLE".to_string()),
            MaterializeMode::Table
        );
        assert_eq!(
            MaterializeMode::from("temp_table".to_string()),
            MaterializeMode::TempTable
        );
        assert_eq!(
            MaterializeMode::from("view".to_string()),
            MaterializeMode::View
        );
        assert_eq!(
            MaterializeMode::from("unknown".to_string()),
            MaterializeMode::View
        );
    }

    #[test]
    fn test_transform_node_from_dag_file_node() {
        let dfn = DagFileNode {
            id: "test".to_string(),
            query_text: "SELECT 1".to_string(),
            depends_on: vec![],
            materialize: Some("temp_table".to_string()),
        };
        let tn = TransformNode::from(dfn);
        assert_eq!(tn.materialize, MaterializeMode::TempTable);
    }
}
