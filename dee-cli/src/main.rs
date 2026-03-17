use clap::{Args, Parser, Subcommand};
use log::info;

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
async fn main() {
    env_logger::init();
    let args = CliArgs::parse();
    match args.command {
        CliCommand::Run(run_cmd) => {
            info!("Running DAG: {}", run_cmd.dag_file);
        }
        CliCommand::Opt => {
            info!("Optimizing DAG");
        }
    }
}
