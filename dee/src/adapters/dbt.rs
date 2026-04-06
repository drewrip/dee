use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::file::{DagFile, DagFileNode, DagFileSource, DagColumn, DagFileMetadata};

#[derive(Serialize, Deserialize, Debug)]
pub struct DbtManifest {
    pub metadata: DbtMetadata,
    pub nodes: HashMap<String, DbtNode>,
    pub sources: HashMap<String, DbtSource>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DbtMetadata {
    pub adapter_type: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DbtNode {
    pub unique_id: String,
    pub depends_on: DbtDependsOn,
    pub compiled_code: Option<String>,
    pub raw_code: Option<String>,
    pub config: DbtConfig,
    pub relation_name: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DbtDependsOn {
    pub nodes: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DbtConfig {
    pub materialized: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DbtSource {
    pub unique_id: String,
    pub name: String,
    pub columns: Option<HashMap<String, DbtColumn>>,
    pub relation_name: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DbtColumn {
    pub name: String,
    pub data_type: Option<String>,
}

impl From<DbtManifest> for DagFile {
    fn from(manifest: DbtManifest) -> Self {
        let mut nodes = Vec::new();
        
        // Map to keep track of unique_id to relation_name mapping for dependency resolution
        let mut id_to_rel = HashMap::new();
        for (id, node) in &manifest.nodes {
            if let Some(rel) = &node.relation_name {
                id_to_rel.insert(id.clone(), rel.clone());
            } else {
                id_to_rel.insert(id.clone(), id.clone());
            }
        }
        for (id, source) in &manifest.sources {
            if let Some(rel) = &source.relation_name {
                id_to_rel.insert(id.clone(), rel.clone());
            } else {
                id_to_rel.insert(id.clone(), id.clone());
            }
        }

        for (id, node) in &manifest.nodes {
            let query_text = node.compiled_code.as_ref()
                .or(node.raw_code.as_ref())
                .cloned()
                .unwrap_or_default();
            
            // Per user request, DO NOT change the query_text.
            // dbt compiled queries already use relation_names.
            let final_query = query_text;
            
            // Filter depends_on to only include nodes that exist in our nodes list,
            // and use their relation_name as the ID.
            let depends_on: Vec<String> = node.depends_on.nodes.iter()
                .filter(|dep_id| manifest.nodes.contains_key(*dep_id))
                .map(|dep_id| id_to_rel.get(dep_id).cloned().unwrap_or_else(|| dep_id.clone()))
                .collect();
                
            let materialize = node.config.materialized.as_deref().map(|m| m == "table" || m == "incremental");
            
            // Use relation_name as the ID so it matches what's used in queries
            let node_id = node.relation_name.clone().unwrap_or_else(|| id.clone());
            
            nodes.push(DagFileNode {
                id: node_id,
                query_text: final_query,
                depends_on,
                materialize,
            });
        }

        let mut sources = Vec::new();
        for (id, source) in manifest.sources {
            let mut columns = Vec::new();
            if let Some(source_cols) = source.columns {
                for (col_name, col) in source_cols {
                    columns.push(DagColumn {
                        name: col_name,
                        data_type: col.data_type.unwrap_or_else(|| "Unknown".to_string()),
                    });
                }
            }
            // Use relation_name for source as well if available
            let source_name = source.relation_name.unwrap_or(id);
            sources.push(DagFileSource {
                name: source_name,
                columns,
            });
        }

        DagFile {
            metadata: Some(DagFileMetadata {
                sql_dialect: Some(manifest.metadata.adapter_type),
            }),
            nodes,
            sources,
        }
    }
}
