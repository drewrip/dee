//! `dee queue`: schedule N runs of a DAG to execute one after another.

use clap::{Args, Subcommand};
use serde_json::{Value, json};

use crate::client::Client;

#[derive(Args)]
pub struct QueueCommand {
    #[command(subcommand)]
    pub command: QueueSubcommand,
}

#[derive(Subcommand)]
pub enum QueueSubcommand {
    /// Put N runs of a DAG on the queue, to execute back to back.
    Add(AddArgs),
    /// Show what is waiting, front of the queue first.
    #[command(alias = "ls")]
    List(ListArgs),
    /// Remove one entry that has not started yet.
    Drop {
        /// The run group id printed by `dee queue add` or `dee queue list`.
        run_group_id: String,
    },
    /// Remove every entry that has not started yet.
    Clear(ClearArgs),
}

#[derive(Args)]
pub struct AddArgs {
    /// DAG to run.
    pub name: String,
    /// How many times to run it. Each is a separate run group, started only
    /// once the one before it has finished.
    #[arg(short = 'n', long, default_value_t = 1)]
    pub count: i32,
    /// Version to run. Left unset, every entry runs whatever version is
    /// current when its turn comes -- so an optimization landing mid-queue is
    /// picked up by the runs behind it.
    #[arg(long)]
    pub version: Option<i32>,
    /// Connection to run against. Defaults to the DAG's target.
    #[arg(short, long)]
    pub target: Option<String>,
    /// Timed repetitions within each entry, executed against one warm pool.
    #[arg(long, default_value_t = 1)]
    pub repeat: i32,
    /// Untimed repetitions run first, to warm the page cache and the engine.
    #[arg(long, default_value_t = 0)]
    pub warmups: i32,
    /// Keep the DAG's existing relations instead of dropping them first.
    #[arg(long)]
    pub no_cleanup: bool,
    /// Capture each node's EXPLAIN plan.
    #[arg(long)]
    pub collect_plans: bool,
    #[arg(long)]
    pub sample_interval_ms: Option<i32>,
    /// Wait for the whole queue of entries to drain before returning.
    #[arg(long)]
    pub wait: bool,
    /// Seconds to wait for each entry with --wait.
    #[arg(long, default_value_t = 3600)]
    pub timeout: u64,
}

#[derive(Args)]
pub struct ListArgs {
    #[arg(long)]
    pub dag: Option<String>,
    /// Include entries that have already finished.
    #[arg(long)]
    pub all: bool,
    #[arg(long, default_value_t = 50)]
    pub limit: usize,
}

#[derive(Args)]
pub struct ClearArgs {
    /// Only clear this DAG's entries.
    #[arg(long)]
    pub dag: Option<String>,
}

pub async fn run(client: &Client, cmd: QueueCommand) -> Result<(), Box<dyn std::error::Error>> {
    match cmd.command {
        QueueSubcommand::Add(args) => add(client, args).await,
        QueueSubcommand::List(args) => list(client, args).await,
        QueueSubcommand::Drop { run_group_id } => {
            client.delete(&format!("/v1/queue/{run_group_id}")).await?;
            println!("dropped {run_group_id} from the queue");
            Ok(())
        }
        QueueSubcommand::Clear(args) => {
            let path = match &args.dag {
                Some(dag) => format!("/v1/queue?dag={dag}"),
                None => "/v1/queue".to_string(),
            };
            client.delete(&path).await?;
            println!("cleared the queue");
            Ok(())
        }
    }
}

async fn add(client: &Client, args: AddArgs) -> Result<(), Box<dyn std::error::Error>> {
    let body = json!({
        "count": args.count,
        "version": args.version,
        "target": args.target,
        "warmups": args.warmups,
        "repetitions": args.repeat,
        "cleanup_before": !args.no_cleanup,
        "collect_plans": args.collect_plans,
        "sample_interval_ms": args.sample_interval_ms,
    });
    let path = if args.wait {
        format!(
            "/v1/dags/{}/queue?wait=true&timeout_s={}",
            args.name, args.timeout
        )
    } else {
        format!("/v1/dags/{}/queue", args.name)
    };

    let result: Value = client.post(&path, &body).await?;
    let entries = result["entries"].as_array().cloned().unwrap_or_default();
    let count = entries.len();
    let status = result["status"].as_str().unwrap_or("queued");

    if args.wait {
        println!("{count} queued run(s) of {} finished: {status}", args.name);
        for entry in &entries {
            let group_id = entry["run_group_id"].as_str().unwrap_or("");
            let runs: Vec<Value> = client
                .get(&format!("/v1/runs?group={group_id}&limit=1000"))
                .await?;
            crate::runs::print_runs(&runs);
        }
        if status != "succeeded" {
            return Err(format!("the queue finished with status '{status}'").into());
        }
    } else {
        let version = if result["pinned_version"].as_bool().unwrap_or(false) {
            format!("v{}", result["version"].as_i64().unwrap_or(0))
        } else {
            "the current version at dispatch".to_string()
        };
        println!(
            "queued {count} run(s) of {} on '{}', running {version}",
            args.name,
            result["target"].as_str().unwrap_or(""),
        );
        println!("watch them with: dee queue list --dag {}", args.name);
    }
    Ok(())
}

async fn list(client: &Client, args: ListArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut path = format!("/v1/queue?limit={}", args.limit);
    if args.all {
        path.push_str("&all=true");
    }
    if let Some(dag) = &args.dag {
        path.push_str(&format!("&dag={dag}"));
    }

    let entries: Vec<Value> = client.get(&path).await?;
    if entries.is_empty() {
        println!("the queue is empty");
        return Ok(());
    }

    println!(
        "{:>3}  {:<38} {:<12} {:<10} {}",
        "#", "RUN GROUP ID", "STATUS", "QUEUE", "DAG"
    );
    for entry in entries {
        // An entry that has started has no place in the queue any more; what
        // matters about it is its status.
        let position = entry["position"]
            .as_i64()
            .map(|p| p.to_string())
            .unwrap_or_else(|| "-".into());
        println!(
            "{:>3}  {:<38} {:<12} {:<10} {}@v{}",
            position,
            entry["run_group_id"].as_str().unwrap_or(""),
            entry["status"].as_str().unwrap_or(""),
            entry["queue_state"].as_str().unwrap_or(""),
            entry["dag_name"].as_str().unwrap_or(""),
            entry["dag_version"].as_i64().unwrap_or(0),
        );
    }
    Ok(())
}
