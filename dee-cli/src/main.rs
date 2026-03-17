use clap::{Args, Parser, Subcommand};
use dee::{
    connectors::{
        Connector,
        duckdb::{DuckDBConnection, DuckDBProfile},
    },
    dag::Dag,
    executor::{Executor, SimpleEngine},
    file::DagFile,
};
use log::info;
use std::fs;
use std::{error::Error, path::PathBuf};

#[derive(Parser)]
pub struct CliArgs {
    #[command(subcommand)]
    command: CliCommand,
}

#[derive(Subcommand)]
pub enum CliCommand {
    Run(RunCommand),
    Opt,
}

#[derive(Args)]
pub struct RunCommand {
    #[arg(short, long)]
    db_file: String,
    dag_file: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    let args = CliArgs::parse();
    match args.command {
        CliCommand::Run(run_cmd) => {
            info!("Running DAG: {}", run_cmd.dag_file);

            let prof = DuckDBProfile::new_with_path(PathBuf::from(run_cmd.db_file));
            let conn = DuckDBConnection::new(prof)?;
            let engine = SimpleEngine::new(conn)?;

            let dag_file: DagFile = serde_json::from_str(&fs::read_to_string(run_cmd.dag_file)?)?;
            let dag = Dag::from(dag_file);

            let res = engine.run(dag).await?;
            info!("res = {}", res);
        }
        CliCommand::Opt => {
            info!("Optimizing DAG");
        }
    }

    Ok(())
}
