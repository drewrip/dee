//! `dee optimize` -- run the optimizer against a registered DAG.
//!
//! The flags are the ones `dee-cli opt` had, so muscle memory and the
//! benchmark's option table carry over. What changed is where they go: they
//! now build an `OptimizerConfig` that is sent as JSON, rather than an argv.
//!
//! A flag that is not passed is not sent, and the server fills it in from the
//! DAG's own configuration -- so the settings a DAG was submitted with are
//! what it is optimized under, and these flags are overrides for one run.

use std::fs;
use std::path::PathBuf;

use clap::Args;
use serde_json::{Value, json};

use crate::client::Client;
use crate::optconfig::{OptimizerArgs, print_config};

#[derive(Args)]
pub struct OptimizeCommand {
    /// DAG to optimize.
    pub name: String,
    /// Version to optimize. Defaults to the DAG's current version.
    #[arg(long)]
    pub version: Option<i32>,
    #[arg(short, long)]
    pub target: Option<String>,

    #[command(flatten)]
    pub optimizer: OptimizerArgs,

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
    let body = json!({
        "version": cmd.version,
        "target": cmd.target,
        "config": cmd.optimizer.to_json()?,
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

    // With the DAG carrying settings of its own, the flags on this command are
    // no longer the whole story, so say what actually ran.
    println!("optimizing {} with:", cmd.name);
    print_config(&accepted["config"]);

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
