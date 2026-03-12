pub mod connectors;
pub mod file;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{connectors::Connector, file::DagFile};

    use super::*;
    use futures::stream::{FuturesUnordered, StreamExt};
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

    #[tokio::test]
    async fn test_duckdb_connection() {
        let prof = connectors::duckdb::DuckDBProfile::new_in_memory();
        let raw_conn = connectors::duckdb::DuckDBConnection::new(prof);
        assert!(raw_conn.is_ok());
        let conn = raw_conn.unwrap();
        let res = conn.execute("SELECT 1".to_string()).await.unwrap();
        assert!(res == 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_duckdb_multithreading() {
        let prof = connectors::duckdb::DuckDBProfile::new_in_memory();
        let raw_conn = connectors::duckdb::DuckDBConnection::new(prof);
        let conn = match raw_conn {
            Err(e) => {
                eprintln!("{}", e);
                assert!(false);
                return;
            }
            Ok(c) => c,
        };

        let mut tasks = Vec::new();
        for i in 0..10 {
            let curr_conn = Arc::clone(&conn);
            let task = tokio::spawn(async move {
                let res = curr_conn.execute(format!("SELECT {}", i)).await.unwrap();
                println!("SELECT {}, res = {}", i, res);
                i
            });
            println!("pushed task {}", i);
            tasks.push(task);
        }

        println!("finished pushing");
        let mut stream = FuturesUnordered::from_iter(tasks);
        while let Some(item) = stream.next().await {
            println!("result = {}", item.unwrap());
        }
    }
}
