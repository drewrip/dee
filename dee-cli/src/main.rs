use clap::{Args, Parser, Subcommand};
use dee::{dag::Dag, file::DagFile};
use serde::Serialize;

use std::error::Error;
use std::fs;

pub mod opt;
pub mod run;

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
    profiles: String,
    #[arg(short, long)]
    target: String,
    #[arg(long)]
    get_plans: Option<String>,

    dag_file: String,
}

#[derive(Args)]
pub struct DrawCommand {
    dag_file: String,
}

#[derive(Args)]
pub struct OptCommand {
    #[arg(short, long)]
    profiles: String,
    #[arg(short, long)]
    target: String,
    #[arg(short, long)]
    output: Option<String>,
    #[arg(short, long, action)]
    stats: bool,
    #[arg(short, long, default_value = "cost")]
    pub metric: Metric,
    #[arg(long, default_value = "exhaustive")]
    pub strategy: Strategy,

    dag_file: String,
}

#[derive(clap::ValueEnum, Clone, Debug)]
pub enum Strategy {
    Exhaustive,
    Heuristic,
}

impl From<Strategy> for dee::opt::OMPStrategy {
    fn from(s: Strategy) -> Self {
        match s {
            Strategy::Exhaustive => dee::opt::OMPStrategy::Exhaustive,
            Strategy::Heuristic => dee::opt::OMPStrategy::Heuristic,
        }
    }
}

#[derive(clap::ValueEnum, Clone, Debug)]
pub enum Metric {
    Runtime,
    Cost,
}

impl From<Metric> for dee::opt::OptimizationMetric {
    fn from(m: Metric) -> Self {
        match m {
            Metric::Runtime => dee::opt::OptimizationMetric::Runtime,
            Metric::Cost => dee::opt::OptimizationMetric::Cost,
        }
    }
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
        CliCommand::Run(run_cmd) => run::run(run_cmd).await?,
        CliCommand::Opt(opt_cmd) => opt::opt(opt_cmd).await?,
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
