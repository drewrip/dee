use clap::{Args, Parser, Subcommand};
use dee::{
    connectors::{
        Connector,
        duckdb::{DuckDBConnection, DuckDBProfile},
    },
    dag::Dag,
    executor::{Executor, SimpleEngine},
    file::DagFile,
    opt::Optimizer,
};
use log::info;
use serde::Serialize;

use std::{error::Error, path::PathBuf};
use std::{fs, sync::Arc};

#[derive(Parser)]
pub struct CliArgs {
    #[command(subcommand)]
    command: CliCommand,
}

#[derive(Subcommand)]
pub enum CliCommand {
    Run(RunCommand),
    Opt(OptCommand),
    Draw(DrawCommand),
    Convert(ConvertCommand),
}

#[derive(Args)]
pub struct RunCommand {
    #[arg(short, long)]
    db_file: String,
    dag_file: String,
}

#[derive(Args)]
pub struct DrawCommand {
    dag_file: String,
}

#[derive(Args)]
pub struct OptCommand {
    #[arg(short, long)]
    db_file: String,
    dag_file: String,
    #[arg(short, long)]
    output: Option<String>,
}

#[derive(clap::ValueEnum, Clone, Debug)]
pub enum ConvertFormat {
    Dbt,
}

#[derive(Args)]
pub struct ConvertCommand {
    #[arg(short, long)]
    format: ConvertFormat,
    manifest_file: String,
    #[arg(short, long)]
    output: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    let args = CliArgs::parse();
    match args.command {
        CliCommand::Run(run_cmd) => {
            info!("Running DAG: {}", run_cmd.dag_file);

            let prof = DuckDBProfile::new_with_path(PathBuf::from(run_cmd.db_file))
                .with_num_connections(8)
                .with_threads(16);
            let conn = DuckDBConnection::new(prof)?;
            let engine = SimpleEngine::new(conn)?;

            let dag_file: DagFile = serde_json::from_str(&fs::read_to_string(run_cmd.dag_file)?)?;
            let dag = Dag::try_from(dag_file)?;
            engine.cleanup(&dag).await?;
            let res = engine.run(&dag).await?;
            info!("stats = {:?}", res);
        }
        CliCommand::Opt(opt_cmd) => {
            info!("Optimizing DAG: {}", opt_cmd.dag_file);

            let prof = DuckDBProfile::new_with_path(PathBuf::from(opt_cmd.db_file))
                .with_num_connections(8)
                .with_threads(16);
            let conn = DuckDBConnection::new(prof)?;
            let engine = SimpleEngine::new(conn.clone())?;

            let dag_file: DagFile = serde_json::from_str(&fs::read_to_string(opt_cmd.dag_file)?)?;
            let mut dag = Dag::try_from(dag_file)?;

            let mut optimizer = Optimizer::new(conn, Arc::new(engine));
            optimizer.run(&mut dag).await?;
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
        }
        CliCommand::Draw(draw_cmd) => {
            let dag_file: DagFile = serde_json::from_str(&fs::read_to_string(draw_cmd.dag_file)?)?;
            let dag = Dag::try_from(dag_file)?;
            let dotfile = dag.nodes.draw();
            println!("{}", dotfile);
        }
        CliCommand::Convert(convert_cmd) => match convert_cmd.format {
            ConvertFormat::Dbt => {
                let manifest: dee::adapters::dbt::DbtManifest =
                    serde_json::from_str(&fs::read_to_string(convert_cmd.manifest_file)?)?;
                let dag_file = DagFile::from(manifest);
                let mut buf = Vec::new();
                let formatter = serde_json::ser::PrettyFormatter::with_indent(b"    ");
                let mut ser = serde_json::Serializer::with_formatter(&mut buf, formatter);
                dag_file.serialize(&mut ser).unwrap();
                let out_str = String::from_utf8(buf).unwrap();
                if let Some(output) = convert_cmd.output {
                    fs::write(output, out_str)?;
                } else {
                    println!("{}", out_str);
                }
            }
        },
    }

    Ok(())
}
