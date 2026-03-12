pub mod connectors;
pub mod file;

#[cfg(test)]
mod tests {
    use crate::{connectors::Connector, file::DagFile};

    use super::*;
    use serde_json::Result;

    #[test]
    fn test_json_parsing() {
        let data = r#"
            {
                "nodes": [
                    {
                        "id": "source",
                        "query_text": "SELECT 1",
                        "depends_on": []
                    },
                    {
                        "id": "sink",
                        "query_path": "test.sql",
                        "depends_on": ["source"]
                    }
                ]
            }"#;

        let df: Result<DagFile> = serde_json::from_str(data);
        assert!(df.is_ok());
        let dag = df.unwrap();
        assert!(dag.nodes.len() == 2);
    }

    #[test]
    fn test_duckdb_connection() {
        let prof = connectors::duckdb::DuckDBProfile::new_in_memory();
        let raw_conn = connectors::duckdb::DuckDBConnection::new(prof);
        assert!(raw_conn.is_ok());
        let mut conn = raw_conn.unwrap();
        let res = conn.execute("SELECT 1".to_string()).unwrap();
        assert!(res == 0);
    }
}
