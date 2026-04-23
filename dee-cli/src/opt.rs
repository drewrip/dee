use dee::{
    connectors::{Connector, duckdb::DuckDBConnection, postgres::PostgresConnection},
    dag::Dag,
    executor::{Executor, SimpleEngine},
    file::DagFile,
    opt::Optimizer,
    profiles::Profile,
};
use log::info;
use serde::Serialize;

use std::{collections::HashMap, error::Error};
use std::{fs, sync::Arc};

use crate::OptCommand;

pub async fn opt(opt_cmd: OptCommand) -> Result<(), Box<dyn Error>> {
    info!("Optimizing DAG: {}", opt_cmd.dag_file);

    let profiles_files: HashMap<String, Profile> =
        serde_json::from_str(&fs::read_to_string(opt_cmd.profiles)?)?;
    let target_profile = profiles_files
        .get(&opt_cmd.target)
        .expect("target profile not found");
    let dag_file: DagFile = serde_json::from_str(&fs::read_to_string(opt_cmd.dag_file)?)?;
    let mut dag = Dag::try_from(dag_file)?;
    let opt_stats = match &target_profile {
        Profile::DuckDB(profile) => {
            let conn = DuckDBConnection::new(profile.clone()).await?;
            let engine = SimpleEngine::new(conn.clone())?;
            engine.cleanup(&dag).await?;
            let mut optimizer = Optimizer::new(conn, Arc::new(engine)).stats_on_passes(true);
            optimizer.run(&mut dag).await?
        }
        Profile::Postgres(profile) => {
            let conn = PostgresConnection::new(profile.clone()).await?;
            let engine = SimpleEngine::new(conn.clone())?;
            engine.cleanup(&dag).await?;
            let mut optimizer = Optimizer::new(conn, Arc::new(engine)).stats_on_passes(true);
            optimizer.run(&mut dag).await?
        }
    };
    if opt_cmd.stats {
        let mut buf = Vec::new();
        let formatter = serde_json::ser::PrettyFormatter::with_indent(b"    ");
        let mut ser = serde_json::Serializer::with_formatter(&mut buf, formatter);
        opt_stats.serialize(&mut ser).unwrap();
        let stats_str = String::from_utf8(buf).unwrap();
        println!("{}", stats_str);
    }
    let new_dag_file: DagFile = DagFile::from(dag);
    let mut buf = Vec::new();
    let formatter = serde_json::ser::PrettyFormatter::with_indent(b"    ");
    let mut ser = serde_json::Serializer::with_formatter(&mut buf, formatter);
    new_dag_file.serialize(&mut ser).unwrap();
    let out_str = String::from_utf8(buf).unwrap();
    if let Some(output) = opt_cmd.output {
        fs::write(output, out_str)?;
    } else {
        println!("{}", out_str);
    }
    Ok(())
}
