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
        
        // Map to keep track of relation_name to unique_id mapping for replacement
        let mut rel_to_id = HashMap::new();
        for (id, node) in &manifest.nodes {
            if let Some(rel) = &node.relation_name {
                rel_to_id.insert(rel.clone(), id.clone());
            }
        }
        for (id, source) in &manifest.sources {
            if let Some(rel) = &source.relation_name {
                rel_to_id.insert(rel.clone(), id.clone());
            }
        }

        for (id, node) in &manifest.nodes {
            let query_text = node.compiled_code.as_ref()
                .or(node.raw_code.as_ref())
                .cloned()
                .unwrap_or_default();
            
            // Replace relation_names with unique_ids to link them in our DAG
            let mut final_query = query_text;
            for (rel, target_id) in &rel_to_id {
                final_query = final_query.replace(rel, target_id);
            }
            
            // Filter depends_on to only include nodes that exist in our nodes list
            let depends_on: Vec<String> = node.depends_on.nodes.iter()
                .filter(|dep_id| manifest.nodes.contains_key(*dep_id))
                .cloned()
                .collect();
                
            let materialize = node.config.materialized.as_deref().map(|m| m == "table" || m == "incremental");
            
            nodes.push(DagFileNode {
                id: id.clone(),
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
            sources.push(DagFileSource {
                name: id,
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
