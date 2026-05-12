use dee::{
    connectors::{Connector, duckdb::DuckDBConnection, postgres::PostgresConnection},
    dag::Dag,
    executor::{Executor, SimpleEngine},
    file::DagFile,
    connections::Connection,
};
use log::info;

use std::fs;
use std::{collections::HashMap, error::Error};

use crate::RunCommand;

pub async fn run(run_cmd: RunCommand) -> Result<(), Box<dyn Error>> {
    info!("Running DAG: {}", run_cmd.dag_file);

    let connections_files: HashMap<String, Connection> =
        serde_json::from_str(&fs::read_to_string(run_cmd.connections)?)?;
    let target_connection = connections_files
        .get(&run_cmd.target)
        .expect("target connection not found");
    let exec_stats = match &target_connection {
        Connection::DuckDB(config) => {
            let conn = DuckDBConnection::new(config.clone()).await?;
            let mut engine = SimpleEngine::new(conn)?;
            if let Some(dump_plans) = &run_cmd.dump_plans {
                engine = engine.with_plans_dir(dump_plans.clone());
            }

            let dag_file: DagFile = serde_json::from_str(&fs::read_to_string(run_cmd.dag_file)?)?;
            let dag = Dag::try_from(dag_file)?;
            engine.cleanup(&dag).await?;
            engine.run(&dag).await?
        }
        Connection::Postgres(config) => {
            let conn = PostgresConnection::new(config.clone()).await?;
            let mut engine = SimpleEngine::new(conn)?;
            if let Some(dump_plans) = &run_cmd.dump_plans {
                engine = engine.with_plans_dir(dump_plans.clone());
            }

            let dag_file: DagFile = serde_json::from_str(&fs::read_to_string(run_cmd.dag_file)?)?;
            let dag = Dag::try_from(dag_file)?;
            engine.cleanup(&dag).await?;
            engine.run(&dag).await?
        }
    };
    info!("stats = {:?}", exec_stats);
    Ok(())
}
