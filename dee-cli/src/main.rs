use clap::{Args, Parser, Subcommand};
use dee::{dag::Dag, file::DagFile};
use serde::Serialize;

use std::error::Error;
use std::fs;

pub mod client;
pub mod connection;
pub mod dag;
pub mod optimize;
pub mod runs;
pub mod schedule;
pub mod serve;

#[derive(Parser)]
pub struct CliArgs {
    /// Base URL of the dee server.
    #[arg(long, global = true, env = "DEE_SERVER", default_value = "http://127.0.0.1:8471")]
    server: String,

    #[command(subcommand)]
    command: CliCommand,
}

#[derive(Subcommand)]
pub enum CliCommand {
    /// Run the dee server: the DAG registry, scheduler and run history.
    Serve(ServeCommand),
    /// Manage the server's named connection targets.
    #[command(subcommand_value_name = "SUBCOMMAND")]
    Connection(connection::ConnectionCommand),
    /// Submit and inspect DAGs in the server's registry.
    #[command(subcommand_value_name = "SUBCOMMAND")]
    Dag(dag::DagCommand),
    /// Run a DAG now.
    Trigger(runs::TriggerCommand),
    /// Inspect run history.
    #[command(subcommand_value_name = "SUBCOMMAND")]
    Runs(runs::RunsCommand),
    /// Cancel a run or run group.
    Cancel {
        /// A run id or a run group id.
        id: String,
    },
    /// Optimize a registered DAG.
    Optimize(optimize::OptimizeCommand),
    /// Put DAGs on a cron schedule.
    #[command(subcommand_value_name = "SUBCOMMAND")]
    Schedule(schedule::ScheduleCommand),
    /// Render a DAG definition file as SVG or graphviz DOT. Runs locally.
    Draw(DrawCommand),
    /// Convert a dbt manifest into a dee DAG definition. Runs locally.
    Convert(ConvertCommand),
}

#[derive(Args)]
pub struct ServeCommand {
    /// Address to listen on. Use port 0 to let the OS choose; the chosen
    /// address is printed on stdout.
    #[arg(long)]
    bind: Option<String>,
    /// Metadata database file. Defaults to `$DEE_HOME/dee.duckdb`, else
    /// `~/.dee/dee.duckdb`. Must not be a warehouse database.
    #[arg(long)]
    metadata_db: Option<String>,
    /// How often the scheduler checks for due DAGs.
    #[arg(long)]
    tick_interval_ms: Option<u64>,
    /// Maximum DAG runs executing at once across all DAGs.
    #[arg(long)]
    max_concurrent_runs: Option<usize>,
}

#[derive(Args)]
pub struct DrawCommand {
    dag_file: String,
    #[arg(long, action)]
    dot: bool,
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
async fn main() {
    // Default to `info` so the server narrates what it is doing -- crash
    // recovery, schedule fires, released pools. RUST_LOG still wins.
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info,duckdb=warn,sqlx=warn"),
    )
    .init();
    let args = CliArgs::parse();
    if let Err(e) = run(args).await {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

async fn run(args: CliArgs) -> Result<(), Box<dyn Error>> {
    let server = args.server.clone();
    match args.command {
        CliCommand::Serve(serve_cmd) => serve::serve(serve_cmd).await?,
        CliCommand::Connection(cmd) => {
            connection::run(&client::Client::new(&server), cmd).await?
        }
        CliCommand::Dag(cmd) => dag::run(&client::Client::new(&server), cmd).await?,
        CliCommand::Trigger(cmd) => runs::trigger(&client::Client::new(&server), cmd).await?,
        CliCommand::Runs(cmd) => runs::run(&client::Client::new(&server), cmd).await?,
        CliCommand::Cancel { id } => runs::cancel(&client::Client::new(&server), id).await?,
        CliCommand::Optimize(cmd) => {
            optimize::optimize(&client::Client::new(&server), cmd).await?
        }
        CliCommand::Schedule(cmd) => schedule::run(&client::Client::new(&server), cmd).await?,
        CliCommand::Draw(draw_cmd) => {
            let dag_file: DagFile = serde_json::from_str(&fs::read_to_string(draw_cmd.dag_file)?)?;
            let dag = Dag::try_from(dag_file)?;
            if draw_cmd.dot {
                println!("{}", dag.nodes.draw());
            } else {
                let source_names: Vec<String> =
                    dag.sources.iter().map(|s| s.name.clone()).collect();
                let svg = dag.nodes.draw_svg(&source_names);
                match draw_cmd.output {
                    Some(path) => fs::write(path, svg)?,
                    None => print!("{}", svg),
                }
            }
        }
        CliCommand::Convert(convert_cmd) => match convert_cmd.format {
            ConvertFormat::Dbt => {
                let manifest: dee::adapters::dbt::DbtManifest =
                    serde_json::from_str(&fs::read_to_string(convert_cmd.manifest_file)?)?;
                let dag_file = DagFile::from(manifest);
                let mut buf = Vec::new();
                let formatter = serde_json::ser::PrettyFormatter::with_indent(b"    ");
                let mut ser = serde_json::Serializer::with_formatter(&mut buf, formatter);
                dag_file.serialize(&mut ser)?;
                let out_str = String::from_utf8(buf)?;
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
