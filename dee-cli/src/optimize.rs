//! `dee optimize` -- run the optimizer against a registered DAG.
//!
//! The flags are the ones `dee-cli opt` had, so muscle memory and the
//! benchmark's option table carry over. What changed is where they go: they
//! now build an `OptimizerConfig` that is sent as JSON, rather than an argv.

use std::fs;
use std::path::PathBuf;

use clap::Args;
use serde_json::{Value, json};

use crate::client::Client;

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
pub struct OptimizeCommand {
    /// DAG to optimize.
    pub name: String,
    /// Version to optimize. Defaults to the DAG's current version.
    #[arg(long)]
    pub version: Option<i32>,
    #[arg(short, long)]
    pub target: Option<String>,

    /// Passes to run, starting from everything off.
    #[arg(long, value_delimiter = ',', conflicts_with = "disable")]
    pub enable: Option<Vec<String>>,
    /// Passes to skip, starting from everything on.
    #[arg(long, value_delimiter = ',', conflicts_with = "enable")]
    pub disable: Option<Vec<String>>,

    #[arg(long)]
    pub omp_top: Option<usize>,
    #[arg(long, default_value = "outdegree")]
    pub omp_node_centrality: CliOMPCentrality,
    #[arg(long, action)]
    pub omp_exhaust: bool,
    #[arg(long, action)]
    pub omp_no_pushdown: bool,

    #[arg(long, action)]
    pub hmp_downstream_cost: bool,
    #[arg(long, default_value_t = 1)]
    pub hmp_max_runs: usize,
    #[arg(long, default_value_t = 0.5)]
    pub hmp_top_cpu_time: f64,
    #[arg(long, action)]
    pub hmp_normalize_with_cardinality: bool,
    #[arg(long, default_value = "breadth")]
    pub hmp_strategy: CliHMPStrategy,
    #[arg(long, default_value_t = 2)]
    pub hmp_beam_width: usize,
    #[arg(long, action)]
    pub hmp_no_pushdown: bool,
    /// Log HMP's operator ranking table after the baseline run.
    #[arg(long, action)]
    pub hmp_show_operators: bool,
    /// Log HMP's node ranking table after the baseline run.
    #[arg(long, action)]
    pub hmp_show_nodes: bool,
    #[arg(long, action)]
    pub profile_iterations: bool,

    /// Store the rewritten DAG as a new version.
    #[arg(long, action)]
    pub save: bool,
    /// Write the machine-readable OptimizeReport JSON here.
    #[arg(long)]
    pub report_json: Option<PathBuf>,
    /// Write the HTML report explaining what each pass did.
    #[arg(long, num_args = 0..=1, default_missing_value = "explain.html")]
    pub explain: Option<String>,
    /// Write the resulting DAG definition here. Implies --save.
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    /// Return as soon as the optimization is accepted instead of waiting.
    #[arg(long, action)]
    pub detach: bool,
    #[arg(long, default_value_t = 7200)]
    pub timeout: u64,
}

pub async fn optimize(
    client: &Client,
    cmd: OptimizeCommand,
) -> Result<(), Box<dyn std::error::Error>> {
    let save = cmd.save || cmd.output.is_some();
    let mut config = json!({
        "omp_top": cmd.omp_top,
        "omp_centrality": match cmd.omp_node_centrality {
            CliOMPCentrality::Outdegree => "outdegree",
            CliOMPCentrality::Paths => "paths",
        },
        // These two flags name the behaviour they turn off, so the config
        // value is their negation.
        "omp_early_termination": !cmd.omp_exhaust,
        "omp_use_pushdown": !cmd.omp_no_pushdown,
        "hmp_use_pushdown": !cmd.hmp_no_pushdown,
        "hmp_downstream_cost": cmd.hmp_downstream_cost,
        "hmp_max_runs": cmd.hmp_max_runs,
        "hmp_top_cpu_time": cmd.hmp_top_cpu_time,
        "hmp_normalize_with_cardinality": cmd.hmp_normalize_with_cardinality,
        "hmp_strategy": match cmd.hmp_strategy {
            CliHMPStrategy::Breadth => "breadth",
            CliHMPStrategy::Greedy => "greedy",
        },
        "hmp_beam_width": cmd.hmp_beam_width,
        "profile_iterations": cmd.profile_iterations,
        // The server refuses a path here, so only the log-the-table form is
        // reachable remotely.
        "hmp_show_operators": if cmd.hmp_show_operators { Some("") } else { None },
        "hmp_show_nodes": if cmd.hmp_show_nodes { Some("") } else { None },
    });

    // `--enable` starts from nothing on, so the resulting pass set is exactly
    // what was asked for and does not shift when dee's defaults change.
    let (hmp, omp, pushdown) = match (&cmd.enable, &cmd.disable) {
        (Some(enabled), _) => (
            enabled.iter().any(|p| p == "hmp"),
            enabled.iter().any(|p| p == "omp"),
            enabled.iter().any(|p| p == "pushdown"),
        ),
        (None, Some(disabled)) => (
            !disabled.iter().any(|p| p == "hmp"),
            !disabled.iter().any(|p| p == "omp"),
            !disabled.iter().any(|p| p == "pushdown"),
        ),
        (None, None) => (true, true, true),
    };
    config["run_hmp_pass"] = json!(hmp);
    config["run_omp_pass"] = json!(omp);
    config["run_pushdown_pass"] = json!(pushdown);

    let body = json!({
        "version": cmd.version,
        "target": cmd.target,
        "config": config,
        "save_as_version": save,
        "explain": cmd.explain.is_some(),
    });

    let path = if cmd.detach {
        format!("/v1/dags/{}/optimize", cmd.name)
    } else {
        format!(
            "/v1/dags/{}/optimize?wait=true&timeout_s={}",
            cmd.name, cmd.timeout
        )
    };

    let accepted: Value = client.post(&path, &body).await?;
    let id = accepted["optimization_id"].as_str().unwrap_or("").to_string();

    if cmd.detach {
        println!("started optimization {id}");
        println!("check it with: dee optimize-status {id}");
        return Ok(());
    }

    let status = accepted["status"].as_str().unwrap_or("");
    if status != "succeeded" {
        let detail: Value = client.get(&format!("/v1/optimizations/{id}")).await?;
        return Err(format!(
            "optimization {status}: {}",
            detail["error"].as_str().unwrap_or("no detail recorded")
        )
        .into());
    }

    let detail: Value = client.get(&format!("/v1/optimizations/{id}")).await?;
    summarize(&detail);

    if let Some(path) = &cmd.report_json {
        let report: Value = client.get(&format!("/v1/optimizations/{id}/report")).await?;
        fs::write(path, serde_json::to_string_pretty(&report)?)?;
    }
    if let Some(path) = &cmd.explain {
        let html = client
            .get_text(&format!("/v1/optimizations/{id}/explain.html"))
            .await?;
        fs::write(path, html)?;
    }
    if let Some(path) = &cmd.output {
        let dag: Value = client.get(&format!("/v1/optimizations/{id}/dag")).await?;
        fs::write(path, serde_json::to_string_pretty(&dag)?)?;
    }
    Ok(())
}

fn summarize(detail: &Value) {
    let baseline = detail["baseline_runtime_ms"].as_i64();
    let final_ms = detail["final_runtime_ms"].as_i64();

    println!(
        "optimized {} v{} in {}ms using {} dag run(s)",
        detail["dag_name"].as_str().unwrap_or(""),
        detail["source_version"].as_i64().unwrap_or(0),
        detail["wall_ms"].as_i64().unwrap_or(0),
        detail["dag_runs_used"].as_i64().unwrap_or(0),
    );

    if let (Some(before), Some(after)) = (baseline, final_ms) {
        let delta = before - after;
        let pct = if before > 0 {
            delta.abs() as f64 * 100.0 / before as f64
        } else {
            0.0
        };
        // Say "faster" or "slower" rather than a signed percentage: a signed
        // improvement reads like a regression to half the people who see it.
        let direction = match delta {
            d if d > 0 => format!("{pct:.1}% faster"),
            d if d < 0 => format!("{pct:.1}% slower"),
            _ => "unchanged".to_string(),
        };
        println!("runtime {before}ms -> {after}ms ({direction})");
        if delta > 0 {
            // The cost of optimizing is only meaningful against the saving it
            // buys, so state it as the number of runs that repay it.
            let cost = detail["wall_ms"].as_i64().unwrap_or(0);
            println!("  pays for itself after {} run(s)", (cost + delta - 1) / delta);
        }
    }

    println!(
        "{} change(s) applied; {} node(s) -> {} node(s)",
        detail["total_changes_applied"].as_i64().unwrap_or(0),
        detail["nodes_before"].as_i64().unwrap_or(0),
        detail["nodes_after"].as_i64().unwrap_or(0),
    );

    for pass in detail["passes"].as_array().into_iter().flatten() {
        println!(
            "  {:<14} {:>7}ms  {:>2} change(s), {:>2} candidate(s), {} dag run(s)",
            pass["pass_name"].as_str().unwrap_or(""),
            pass["wall_ms"].as_i64().unwrap_or(0),
            pass["changes_applied"].as_i64().unwrap_or(0),
            pass["candidates_considered"].as_i64().unwrap_or(0),
            pass["dag_runs_used"].as_i64().unwrap_or(0),
        );
    }

    if let Some(version) = detail["result_version"].as_i64() {
        println!("saved as version {version}");
    }
}
