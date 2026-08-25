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
    profile_interval_ms: Option<u64>,
    #[arg(long)]
    dump_plans: Option<String>,
    /// Execute each DAG this many times inside a single process, so
    /// per-repetition timings exclude process startup, connection-pool
    /// creation and the initial cleanup. Each repetition is reported
    /// separately.
    #[arg(long, default_value_t = 1)]
    repeat: usize,
    /// Untimed repetitions run before the measured ones, to warm the page
    /// cache and the engine. Reported with `phase = "warmup"`.
    #[arg(long, default_value_t = 0)]
    warmups: usize,
    /// Write the machine-readable `ProfileReport` JSON here. Implies
    /// profiling. This is what the benchmarking harness consumes.
    #[arg(long)]
    report_json: Option<String>,

    #[arg(required = true)]
    dag_files: Vec<String>,
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
pub enum CliOMPCentrality {
    Outdegree,
    Paths,
}

#[derive(clap::ValueEnum, Clone, Debug)]
pub enum CliHMPStrategy {
    Breadth,
    Greedy,
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
    #[arg(long, default_value = "outdegree")]
    omp_node_centrality: CliOMPCentrality,

    #[arg(long, value_delimiter = ',', conflicts_with = "disable")]
    enable: Option<Vec<String>>,
    #[arg(long, value_delimiter = ',', conflicts_with = "enable")]
    disable: Option<Vec<String>>,

    /// Rank HMP VIEW candidates by the total cost of the duplicate
    /// computation they introduce downstream, instead of an estimated cost
    /// to run the VIEW itself: for each operator in a materialized TABLE's
    /// EXPLAIN ANALYZE plan, add its CPU cost to every candidate VIEW whose
    /// own EXPLAIN plan contains that operator.
    #[arg(long, action)]
    hmp_downstream_cost: bool,
    #[arg(long, default_value_t = 1)]
    hmp_max_runs: usize,
    #[arg(long, default_value_t = 0.5)]
    hmp_top_cpu_time: f64,
    /// Log a table of HMP operator rankings after the baseline run. Pass a
    /// path to also write the table there, e.g. `--hmp-show-operators=out.txt`.
    #[arg(long, num_args = 0..=1, default_missing_value = "")]
    hmp_show_operators: Option<String>,
    /// Log a table of HMP node rankings after the baseline run. Pass a path
    /// to also write the table there, e.g. `--hmp-show-nodes=out.txt`.
    #[arg(long, num_args = 0..=1, default_missing_value = "")]
    hmp_show_nodes: Option<String>,
    /// Rank HMP VIEW candidates by total CPU time divided by the View's
    /// estimated cardinality (from its EXPLAIN plan), instead of raw total
    /// CPU time.
    #[arg(long, action)]
    hmp_normalize_with_cardinality: bool,
    /// Choose the search strategy HMP uses to select which VIEWs to
    /// materialize. `breadth` (default) tries all k-sized combinations
    /// smallest-first. `greedy` walks the node ranking and commits each
    /// materialization that improves performance.
    #[arg(long, default_value = "breadth")]
    hmp_strategy: CliHMPStrategy,
    /// Number of hypotheses the `greedy` strategy's beam search keeps alive
    /// at each step. Unused by the `breadth` strategy.
    #[arg(long, default_value_t = 2)]
    hmp_beam_width: usize,
    /// Disable running the PushdownPass before evaluating each HMP
    /// materialization candidate. Enabled by default for more accurate cost
    /// measurements.
    #[arg(long, action)]
    hmp_no_pushdown: bool,
    #[arg(long, action)]
    omp_exhaust: bool,
    /// Disable running the PushdownPass before evaluating each OMP
    /// materialization candidate. Enabled by default for more accurate cost
    /// measurements.
    #[arg(long, action)]
    omp_no_pushdown: bool,
    /// Capture a CPU/memory/disk timeseries for every HMP/OMP candidate run
    /// and include it in each iteration's `--stats` output.
    #[arg(long, action)]
    profile_iterations: bool,
    /// Write an HTML report explaining what each enabled pass did and why.
    /// Pass a path to choose the output file, e.g. `--explain=out.html`;
    /// bare `--explain` writes to `explain.html`.
    #[arg(long, num_args = 0..=1, default_missing_value = "explain.html")]
    explain: Option<String>,
    /// Write the machine-readable `OptimizeReport` JSON here. This is what
    /// the benchmarking harness consumes; `--stats` remains the
    /// human-oriented stdout dump.
    #[arg(long)]
    report_json: Option<String>,

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
async fn main() {
    env_logger::init();
    let args = CliArgs::parse();
    if let Err(e) = run(args).await {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

async fn run(args: CliArgs) -> Result<(), Box<dyn Error>> {
    match args.command {
        CliCommand::Run(run_cmd) => run::run(run_cmd).await?,
        CliCommand::Opt(opt_cmd) => opt::opt(opt_cmd).await?,
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
