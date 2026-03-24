use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct DagFile {
    pub metadata: Option<DagFileMetadata>,
    pub nodes: Vec<DagFileNode>,
    pub sources: Vec<DagFileSource>,
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
}

#[derive(Serialize, Deserialize, Clone)]
pub struct DagFileSource {
    pub name: String,
    pub columns: Vec<DagColumn>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct DagColumn {
    pub name: String,
    pub data_type: String,
}
