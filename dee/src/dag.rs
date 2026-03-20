use std::collections::HashMap;

use petgraph::{Direction::Incoming, data::DataMap, graph::NodeIndex, prelude::StableDiGraph};

use crate::file::{DagFile, DagFileMetadata, DagFileNode};

/// Interal DAG representation

#[derive(Clone, Copy, Debug)]
pub enum MaterializeMode {
    View,
    Table,
    Incremental,
}

#[derive(Clone)]
pub struct TransformNode {
    pub id: String,
    pub query_text: String,
    pub materialize: MaterializeMode,
    pub no_mangle: bool,
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
        // If no_mangle isn't specified then allow mangling
        let no_mangle = match value.no_mangle {
            Some(mangle) => mangle,
            None => false,
        };

        Self {
            id: value.id,
            query_text: value.query_text,
            materialize,
            no_mangle,
        }
    }
}

pub struct Dag {
    pub db: String,
    pub graph: StableDiGraph<u32, ()>,
    pub nodes: Vec<TransformNode>,
}

impl From<DagFile> for Dag {
    fn from(value: DagFile) -> Self {
        let dialect = match value.metadata {
            Some(meta) => meta.sql_dialect.unwrap_or("Unknown".into()),
            None => "Unknown".into(),
        };
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
        Self {
            db: dialect,
            graph,
            nodes,
        }
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
        no_mangle: Some(false),
    }
}

impl From<Dag> for DagFile {
    fn from(value: Dag) -> DagFile {
        let map: HashMap<u32, NodeIndex> = value
            .graph
            .node_indices()
            .map(|nidx| (*value.graph.node_weight(nidx).unwrap(), nidx))
            .collect();

        DagFile {
            metadata: Some(DagFileMetadata {
                sql_dialect: Some(value.db.clone()),
            }),
            nodes: value
                .nodes
                .iter()
                .enumerate()
                .map(|(i, n)| transform_to_file_node(i as u32, &value, &map, n))
                .collect(),
        }
    }
}
