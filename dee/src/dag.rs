use petgraph::{Direction::Incoming, graph::NodeIndex, prelude::StableDiGraph};
use std::{collections::HashMap, sync::Arc};

use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use thiserror::Error;

use crate::file::{DagColumn, DagFile, DagFileMetadata, DagFileNode, DagFileSource};

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

#[derive(Clone)]
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

#[derive(Clone)]
pub struct TransformNode {
    pub id: String,
    pub query_text: String,
    pub materialize: MaterializeMode,
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
        }
    }
}

pub struct Dag {
    pub db: String,
    pub graph: StableDiGraph<u32, ()>,
    pub nodes: Vec<TransformNode>,
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
        let mut node_index: HashMap<String, u32> = HashMap::with_capacity(nodes.len());
        let mut n = 0;
        let mut graph = StableDiGraph::new();
        for node in &nodes {
            node_index.insert(node.id.clone(), n);
            graph.add_node(n);
            n += 1;
        }
        for node in &value.nodes {
            match node_index.get(&node.id) {
                Some(dst) => {
                    for src_node in &node.depends_on {
                        match node_index.get(src_node) {
                            Some(src) => {
                                graph.add_edge((*src).into(), (*dst).into(), ());
                            }
                            None => (),
                        }
                    }
                }
                None => (),
            }
        }
        Ok(Self {
            db: dialect,
            graph,
            nodes,
            sources,
        })
    }
}

fn transform_to_file_node(
    idx: u32,
    dag: &Dag,
    nidx_map: &HashMap<u32, NodeIndex>,
    value: &TransformNode,
) -> DagFileNode {
    let parents: Vec<NodeIndex> = dag
        .graph
        .neighbors_directed(*nidx_map.get(&idx).unwrap(), Incoming)
        .collect();
    let p: Vec<u32> = parents
        .iter()
        .map(|ancestor| *dag.graph.node_weight(*ancestor).unwrap())
        .collect();

    let depends: Vec<String> = p
        .iter()
        .map(|tidx| dag.nodes.get(*tidx as usize).unwrap().id.clone())
        .collect();

    let materialize = match value.materialize {
        MaterializeMode::View => false,
        MaterializeMode::Table => true,
        MaterializeMode::Incremental => true,
    };
    DagFileNode {
        id: value.id.clone(),
        query_text: value.query_text.clone(),
        depends_on: depends,
        materialize: Some(materialize),
    }
}

impl From<Dag> for DagFile {
    fn from(value: Dag) -> DagFile {
        let map: HashMap<u32, NodeIndex> = value
            .graph
            .node_indices()
            .map(|nidx| (*value.graph.node_weight(nidx).unwrap(), nidx))
            .collect();
        let nodes = value
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| transform_to_file_node(i as u32, &value, &map, n))
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
