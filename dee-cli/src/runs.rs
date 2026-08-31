//! `dee trigger`, `dee runs ...` and `dee cancel`.

use std::fs;
use std::path::PathBuf;

use clap::{Args, Subcommand};
use serde_json::{Value, json};

use crate::client::Client;

#[derive(Args)]
pub struct TriggerCommand {
    /// DAG to run.
    pub name: String,
    /// Version to run. Defaults to the DAG's current version.
    #[arg(long)]
    pub version: Option<i32>,
    /// Connection to run against. Defaults to the DAG's target.
    #[arg(short, long)]
    pub target: Option<String>,
    /// Timed repetitions, executed back to back against one warm pool.
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
    /// Wait for the run to finish before returning.
    #[arg(long)]
    pub wait: bool,
    /// Seconds to wait with --wait.
    #[arg(long, default_value_t = 3600)]
    pub timeout: u64,
    /// With --wait, write the ProfileReport JSON here.
    #[arg(long)]
    pub report_json: Option<PathBuf>,
}

#[derive(Args)]
pub struct RunsCommand {
    #[command(subcommand)]
    pub command: RunsSubcommand,
}

#[derive(Subcommand)]
pub enum RunsSubcommand {
    /// List runs, most recent first.
    #[command(alias = "ls")]
    List(ListArgs),
    /// Show one run.
    Get { run_id: String },
    /// Show a run's per-node timings.
    Nodes { run_id: String },
    /// Show a run's lifecycle events.
    Logs { run_id: String },
    /// Fetch a run group's profile report.
    Report(ReportArgs),
}

#[derive(Args)]
pub struct ListArgs {
    #[arg(long)]
    pub dag: Option<String>,
    #[arg(long)]
    pub status: Option<String>,
    #[arg(long)]
    pub group: Option<String>,
    /// Show only warmup or only measured repetitions.
    #[arg(long)]
    pub phase: Option<String>,
    #[arg(long, default_value_t = 20)]
    pub limit: usize,
}

#[derive(Args)]
pub struct ReportArgs {
    /// A run id or a run group id.
    pub id: String,
    /// Write the rendered HTML report instead of JSON.
    #[arg(long)]
    pub html: bool,
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

pub async fn trigger(
    client: &Client,
    cmd: TriggerCommand,
) -> Result<(), Box<dyn std::error::Error>> {
    let body = json!({
        "version": cmd.version,
        "target": cmd.target,
        "warmups": cmd.warmups,
        "repetitions": cmd.repeat,
        "cleanup_before": !cmd.no_cleanup,
        "collect_plans": cmd.collect_plans,
        "sample_interval_ms": cmd.sample_interval_ms,
    });
    let path = if cmd.wait {
        format!(
            "/v1/dags/{}/runs?wait=true&timeout_s={}",
            cmd.name, cmd.timeout
        )
    } else {
        format!("/v1/dags/{}/runs", cmd.name)
    };

    let result: Value = client.post(&path, &body).await?;
    let group_id = result["run_group_id"].as_str().unwrap_or("");
    let status = result["status"].as_str().unwrap_or("queued");

    if cmd.wait {
        println!("run group {group_id} {status}");
        let runs: Vec<Value> = client
            .get(&format!("/v1/runs?group={group_id}&limit=1000"))
            .await?;
        print_runs(&runs);

        if let Some(path) = &cmd.report_json {
            let report: Value = client
                .get(&format!("/v1/run-groups/{group_id}/report"))
                .await?;
            fs::write(path, serde_json::to_string_pretty(&report)?)?;
        }
        if status != "succeeded" {
            return Err(format!("run group finished with status '{status}'").into());
        }
    } else {
        println!("triggered {} as run group {group_id}", cmd.name);
        println!("follow it with: dee runs list --group {group_id}");
    }
    Ok(())
}

pub async fn run(client: &Client, cmd: RunsCommand) -> Result<(), Box<dyn std::error::Error>> {
    match cmd.command {
        RunsSubcommand::List(args) => {
            let mut path = format!("/v1/runs?limit={}", args.limit);
            for (key, value) in [
                ("dag", args.dag),
                ("status", args.status),
                ("group", args.group),
                ("phase", args.phase),
            ] {
                if let Some(value) = value {
                    path.push_str(&format!("&{key}={value}"));
                }
            }
            let runs: Vec<Value> = client.get(&path).await?;
            if runs.is_empty() {
                println!("no runs");
                return Ok(());
            }
            print_runs(&runs);
            Ok(())
        }
        RunsSubcommand::Get { run_id } => {
            let run: Value = client.get(&format!("/v1/runs/{run_id}")).await?;
            println!("{}", serde_json::to_string_pretty(&run)?);
            Ok(())
        }
        RunsSubcommand::Nodes { run_id } => {
            let nodes: Vec<Value> = client.get(&format!("/v1/runs/{run_id}/nodes")).await?;
            if nodes.is_empty() {
                println!("no node executions recorded for this run");
                return Ok(());
            }
            println!("{:>10}  {:<12} {}", "DURATION", "MATERIALIZE", "NODE");
            for node in nodes {
                println!(
                    "{:>8}ms  {:<12} {}",
                    node["duration_ms"].as_i64().unwrap_or(0),
                    node["materialize"].as_str().unwrap_or(""),
                    node["node_id"].as_str().unwrap_or(""),
                );
            }
            Ok(())
        }
        RunsSubcommand::Logs { run_id } => {
            let events: Vec<Value> = client.get(&format!("/v1/runs/{run_id}/logs")).await?;
            for event in events {
                println!(
                    "{}  {:<5} {}",
                    event["ts"].as_str().unwrap_or(""),
                    event["level"].as_str().unwrap_or(""),
                    event["message"].as_str().unwrap_or(""),
                );
            }
            Ok(())
        }
        RunsSubcommand::Report(args) => report(client, args).await,
    }
}

async fn report(client: &Client, args: ReportArgs) -> Result<(), Box<dyn std::error::Error>> {
    // Accept either id: a run id resolves to its group, since the report
    // covers a whole repetition series.
    let group_id = match client
        .get::<Value>(&format!("/v1/runs/{}", args.id))
        .await
    {
        Ok(run) => run["run_group_id"].as_str().unwrap_or(&args.id).to_string(),
        Err(_) => args.id.clone(),
    };

    let text = if args.html {
        client
            .get_text(&format!("/v1/run-groups/{group_id}/report.html"))
            .await?
    } else {
        let report: Value = client
            .get(&format!("/v1/run-groups/{group_id}/report"))
            .await?;
        serde_json::to_string_pretty(&report)?
    };

    match args.output {
        Some(path) => fs::write(path, text)?,
        None => println!("{text}"),
    }
    Ok(())
}

pub async fn cancel(client: &Client, id: String) -> Result<(), Box<dyn std::error::Error>> {
    // Same tolerance as `report`: cancelling a run cancels its whole series.
    let path = match client.get::<Value>(&format!("/v1/runs/{id}")).await {
        Ok(_) => format!("/v1/runs/{id}/cancel"),
        Err(_) => format!("/v1/run-groups/{id}/cancel"),
    };
    let result: Value = client.post(&path, &json!({})).await?;
    println!("{}", result["detail"].as_str().unwrap_or("cancellation requested"));
    Ok(())
}

pub fn print_runs(runs: &[Value]) {
    println!(
        "{:<38} {:<12} {:<8} {:>4} {:>10}  {}",
        "RUN ID", "STATUS", "PHASE", "REP", "DURATION", "DAG"
    );
    for run in runs {
        let duration = run["duration_ms"]
            .as_i64()
            .map(|ms| format!("{ms}ms"))
            .unwrap_or_else(|| "-".into());
        println!(
            "{:<38} {:<12} {:<8} {:>4} {:>10}  {}@v{}",
            run["run_id"].as_str().unwrap_or(""),
            run["status"].as_str().unwrap_or(""),
            run["phase"].as_str().unwrap_or(""),
            run["rep_index"].as_i64().unwrap_or(0),
            duration,
            run["dag_name"].as_str().unwrap_or(""),
            run["dag_version"].as_i64().unwrap_or(0),
        );
    }
}
