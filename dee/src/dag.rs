use std::{collections::HashMap, sync::Arc};

use std::collections::HashSet;

use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use thiserror::Error;

use crate::{
    file::{DagColumn, DagFile, DagFileMetadata, DagFileNode, DagFileSource},
    graph::Graph,
};

#[derive(Error, Debug)]
pub enum FormatError {
    #[error("problem with parsing Dag file - {0}")]
    Parser(String),
}

/// Interal DAG representation

#[derive(Clone, Copy, Debug)]
pub enum MaterializeMode {
    View,
    Table,
    Incremental,
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
}

impl From<DagFileNode> for TransformNode {
    fn from(value: DagFileNode) -> Self {
        // If materialize strategy isn't provided, default to view
        let materialize = match value.materialize {
            Some(should_materialize) => {
                if should_materialize {
                    MaterializeMode::Table
                } else {
                    MaterializeMode::View
                }
            }
            None => MaterializeMode::View,
        };

        Self {
            id: value.id,
            query_text: value.query_text,
            materialize,
            depends_on: HashSet::from_iter(value.depends_on),
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
        Ok(Self {
            db: dialect,
            nodes: graph,
            sources,
        })
    }
}

fn transform_to_file_node(value: &TransformNode) -> DagFileNode {
    let materialize = match value.materialize {
        MaterializeMode::View => false,
        MaterializeMode::Table => true,
        MaterializeMode::Incremental => true,
    };
    DagFileNode {
        id: value.id.clone(),
        query_text: value.query_text.clone(),
        depends_on: value.depends_on.clone().into_iter().collect(),
        materialize: Some(materialize),
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
                    .map(|f| DagColumn {
                        name: f.name().clone(),
                        data_type: f.data_type().to_string(),
                    })
                    .collect(),
            })
            .collect();
        DagFile {
            metadata: Some(DagFileMetadata {
                sql_dialect: Some(value.db.clone()),
            }),
            sources,
            nodes,
        }
    }
}
