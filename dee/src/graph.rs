use crate::dag::TransformNode;
use std::collections::{HashMap, HashSet};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum GraphError {
    #[error("node in graph points to nonexistent node - {0} -> {1}")]
    BadGraph(String, String),
    #[error("node in graph doesn't exist - {0}")]
    MissingNode(String),
}

pub type GraphType = HashMap<String, TransformNode>;

#[derive(Debug, Clone)]
pub struct Graph {
    g: GraphType,
}

impl Graph {
    pub fn new(g: GraphType) -> Self {
        Self { g }
    }

    pub fn get(&self, node: String) -> Option<&TransformNode> {
        self.g.get(&node)
    }

    pub fn get_mut(&mut self, node: String) -> Option<&mut TransformNode> {
        self.g.get_mut(&node)
    }

    pub fn check(&self) -> Result<(), GraphError> {
        for (id, node) in self.g.iter() {
            for parent in &node.depends_on {
                if !self.g.contains_key(parent) {
                    return Err(GraphError::BadGraph(id.clone(), parent.clone()));
                }
            }
        }
        Ok(())
    }

    pub fn check_nodes(&self, nodes: Vec<String>) -> Result<(), GraphError> {
        for node in nodes {
            let child = self
                .g
                .get(&node)
                .ok_or(GraphError::MissingNode(node.clone()))?;
            for parent in &child.depends_on {
                if !self.g.contains_key(parent) {
                    return Err(GraphError::BadGraph(node.clone(), parent.clone()));
                }
            }
        }
        Ok(())
    }

    pub fn remove(&mut self, node_to_remove: String) -> Option<usize> {
        self.g.remove(&node_to_remove)?;
        let mut rem_count = 0;
        for node in &mut self.g {
            rem_count += node.1.depends_on.remove(&node_to_remove) as usize;
        }
        Some(rem_count)
    }

    pub fn add_node(&mut self, new_node: TransformNode) -> Result<(), GraphError> {
        let node_name = new_node.id.clone();
        self.g.insert(node_name.clone(), new_node);
        self.check_nodes(vec![node_name])?;
        Ok(())
    }

    pub fn add_node_unchecked(&mut self, new_node: TransformNode) {
        self.g.insert(new_node.id.clone(), new_node);
    }

    pub fn add_edge(&mut self, src_node: &String, dst_node: &String) -> Result<(), GraphError> {
        if !self.g.contains_key(dst_node) {
            return Err(GraphError::MissingNode(dst_node.clone()));
        }
        let src = self
            .g
            .get_mut(src_node)
            .ok_or(GraphError::MissingNode(src_node.clone()))?;
        src.depends_on.insert(dst_node.clone());
        Ok(())
    }

    pub fn add_edge_unchecked(&mut self, src_node: &String, dst_node: &String) {
        let src = self.g.get_mut(src_node);
        if let Some(good_src) = src {
            good_src.depends_on.insert(dst_node.clone());
        }
    }

    pub fn nodes(&self) -> impl Iterator<Item = &TransformNode> {
        self.g.values()
    }

    pub fn nodes_mut(&mut self) -> impl Iterator<Item = &mut TransformNode> {
        self.g.values_mut()
    }

    pub fn num_nodes(&self) -> usize {
        self.g.len()
    }

    pub fn num_edges(&self) -> usize {
        self.g.iter().map(|n| n.1.depends_on.len()).sum()
    }

    pub fn in_degree(&self, node: &String) -> Option<usize> {
        self.g.get(node).map(|n| n.depends_on.len())
    }

    pub fn out_degree(&self, node: &String) -> usize {
        self.nodes()
            .map(|n| n.depends_on.contains(node) as usize)
            .sum()
    }

    ///
    /// Nodes that have no dependencies
    ///
    pub fn sources(&self) -> impl Iterator<Item = String> {
        self.g
            .iter()
            .filter(|n| n.1.depends_on.len() == 0)
            .map(|n| n.0.clone())
    }

    ///
    /// Nodes that have no other nodes that depend on them
    ///
    pub fn sinks(&self) -> impl Iterator<Item = String> {
        let mut someone_depends_on = HashSet::new();
        for (_, node) in self.g.iter() {
            someone_depends_on.extend(node.depends_on.clone());
        }

        self.g
            .keys()
            .cloned()
            .filter(move |k| !someone_depends_on.contains(k))
    }

    pub fn draw(&self) -> String {
        let mut lines: Vec<String> = Vec::new();

        for id in self.g.keys() {
            lines.push(format!("\"{}\"", id.clone().replace("\"", "")));
        }

        for (id, node) in self.g.iter() {
            lines.push(format!("// node={}", id));
            for parent in node.depends_on.iter() {
                lines.push(format!(
                    "\"{}\" -> \"{}\"",
                    parent.replace("\"", ""),
                    id.replace("\"", "")
                ));
            }
        }
        let line_section = lines
            .iter()
            .map(|l| format!("\t{}", l))
            .collect::<Vec<String>>()
            .join("\n");
        format!("digraph G {{\n{}\n}}", line_section)
    }

    pub fn topological_sort(&self) -> Vec<String> {
        let mut result = Vec::new();
        let mut work_graph = self.clone();

        while work_graph.num_nodes() > 0 {
            let sources: Vec<String> = work_graph.sources().collect();
            if sources.is_empty() {
                // Cycle detected or something else wrong, but for now just break
                break;
            }
            for source in sources {
                result.push(source.clone());
                work_graph.remove(source);
            }
        }
        result
    }
}
