use dee::{
    connectors::{Connector, duckdb::DuckDBConnection, postgres::PostgresConnection},
    dag::Dag,
    executor::{Executor, SimpleEngine},
    file::DagFile,
    profiles::Profile,
};
use log::info;

use std::fs;
use std::{collections::HashMap, error::Error};

use crate::RunCommand;

pub async fn run(run_cmd: RunCommand) -> Result<(), Box<dyn Error>> {
    info!("Running DAG: {}", run_cmd.dag_file);

    let profiles_files: HashMap<String, Profile> =
        serde_json::from_str(&fs::read_to_string(run_cmd.profiles)?)?;
    let target_profile = profiles_files
        .get(&run_cmd.target)
        .expect("target profile not found");
    let exec_stats = match &target_profile {
        Profile::DuckDB(profile) => {
            let conn = DuckDBConnection::new(profile.clone()).await?;
            let mut engine = SimpleEngine::new(conn)?;
            if let Some(dump_plans) = &run_cmd.dump_plans {
                engine = engine.with_plans_dir(dump_plans.clone());
            }

            let dag_file: DagFile = serde_json::from_str(&fs::read_to_string(run_cmd.dag_file)?)?;
            let dag = Dag::try_from(dag_file)?;
            engine.cleanup(&dag).await?;
            engine.run(&dag).await?
        }
        Profile::Postgres(profile) => {
            let conn = PostgresConnection::new(profile.clone()).await?;
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
