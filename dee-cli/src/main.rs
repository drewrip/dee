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
    connections: String,
    #[arg(short, long)]
    target: String,
    #[arg(long, action)]
    profile: bool,
    #[arg(long)]
    profile_dump: Option<String>,
    #[arg(long)]
    profile_viz: Option<String>,
    #[arg(long)]
    dump_plans: Option<String>,

    #[arg(required = true)]
    dag_files: Vec<String>,
}

#[derive(Args)]
pub struct DrawCommand {
    dag_file: String,
}

#[derive(clap::ValueEnum, Clone, Debug)]
pub enum CliOMPCostMetric {
    Actual,
    Estimate,
}

#[derive(clap::ValueEnum, Clone, Debug)]
pub enum CliOMPCentrality {
    Outdegree,
    Paths,
}

#[derive(Args)]
pub struct OptCommand {
    #[arg(short, long)]
    connections: String,
    #[arg(short, long)]
    target: String,
    #[arg(short, long)]
    output: Option<String>,
    #[arg(short, long, action)]
    stats: bool,
    #[arg(long)]
    omp_top: Option<usize>,
    #[arg(long)]
    omp_cost: Option<CliOMPCostMetric>,
    #[arg(long, default_value = "outdegree")]
    omp_node_centrality: CliOMPCentrality,

    #[arg(long, value_delimiter = ',', conflicts_with = "disable")]
    enable: Option<Vec<String>>,
    #[arg(long, value_delimiter = ',', conflicts_with = "enable")]
    disable: Option<Vec<String>>,

    dag_file: String,
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
