use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct DagFile {
    pub metadata: Option<DagFileMetadata>,
    pub nodes: Vec<DagFileNode>,
}

#[derive(Serialize, Deserialize)]
pub struct DagFileMetadata {
    pub sql_dialect: Option<String>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct DagFileNode {
    pub id: String,
    pub query_text: String,
    pub depends_on: Vec<String>,
    pub materialize: Option<bool>,
    pub no_mangle: Option<bool>,
}
