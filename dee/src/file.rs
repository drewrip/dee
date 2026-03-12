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

#[derive(Serialize, Deserialize)]
pub struct DagFileNode {
    pub id: String,
    #[serde(flatten)]
    pub query: SQLText,
    pub depends_on: Vec<String>,
    pub no_mangle: Option<bool>,
}

#[derive(Serialize, Deserialize)]
pub enum SQLText {
    #[serde(rename = "query_path")]
    QueryPath(String),
    #[serde(rename = "query_text")]
    QueryText(String),
}
