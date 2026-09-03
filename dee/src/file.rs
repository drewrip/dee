use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct DagFile {
    pub metadata: Option<DagFileMetadata>,
    pub nodes: Vec<DagFileNode>,
    pub sources: Vec<DagFileSource>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct DagFileMetadata {
    pub sql_dialect: Option<String>,
    /// Most nodes the executor may have in flight at once. `None` is
    /// unlimited -- every runnable node is started as soon as its dependencies
    /// are met, which is what dee did before this field existed.
    ///
    /// Skipped when absent so a DAG that was never tuned serializes exactly as
    /// it always has, and its content hash does not move.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_parallelism: Option<usize>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct DagFileNode {
    pub id: String,
    pub query_text: String,
    pub depends_on: Vec<String>,
    pub materialize: Option<String>,
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
