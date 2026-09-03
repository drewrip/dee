//! `dee optimization` -- attach an optimization to a DAG.
//!
//! The counterpart to `dee optimize`. That command means "optimize this DAG
//! now": it performs its own DAG runs, takes the same exclusive claim a run
//! does, and finishes with an answer. This one attaches an optimization to a
//! DAG so it steps around the runs the DAG performs anyway, improving it over
//! its lifetime and costing no runs of its own.
//!
//! Which of the two applies is a property of the optimization, not a choice:
//! `dee optimization available` says which are continuous and which run once.

use clap::{Args, Subcommand};
use serde_json::{Value, json};

use crate::client::Client;
use crate::optconfig::OptimizerArgs;

#[derive(Args)]
pub struct OptimizationCommand {
    #[command(subcommand)]
    pub command: OptimizationSubcommand,
}

#[derive(Subcommand)]
pub enum OptimizationSubcommand {
    /// Attach an optimization to a DAG.
    Register(RegisterCommand),
    /// Show the optimizations a DAG is under.
    #[command(alias = "ls")]
    List { dag: String },
    /// Change when a registered optimization steps.
    Phase(PhaseCommand),
    /// Detach an optimization, tearing down the state it kept.
    #[command(alias = "rm")]
    Deregister { dag: String, optimization: String },
    /// Show every optimization dee can register.
    Available,
}

#[derive(Args)]
pub struct RegisterCommand {
    pub dag: String,
    /// Which optimization: parallelism, hmp, omp, pushdown.
    pub optimization: String,
    /// When it steps: before, after, or both. Defaults to what the
    /// optimization asks for.
    #[arg(long)]
    pub step_phase: Option<String>,

    #[command(flatten)]
    pub optimizer: OptimizerArgs,
}

#[derive(Args)]
pub struct PhaseCommand {
    pub dag: String,
    pub optimization: String,
    /// before, after, or both.
    pub step_phase: String,
}

pub async fn optimization(
    client: &Client,
    cmd: OptimizationCommand,
) -> Result<(), Box<dyn std::error::Error>> {
    match cmd.command {
        OptimizationSubcommand::Register(cmd) => {
            let body = json!({
                "name": cmd.optimization,
                "step_phase": cmd.step_phase,
                "config": cmd.optimizer.to_json()?,
            });
            let created: Value = client
                .post(&format!("/v1/dags/{}/optimizations", cmd.dag), &body)
                .await?;
            print_registration(&created);

            // A continuous optimization does nothing until the DAG runs, which
            // is easy to mistake for it having failed to register.
            if created["optimization_type"].as_str() == Some("continuous") {
                println!(
                    "\nit will step around each run of {}; give it runs to learn from:",
                    cmd.dag
                );
                println!("  dee trigger {}", cmd.dag);
                println!("  dee queue add {} -n 10", cmd.dag);
            }
            Ok(())
        }

        OptimizationSubcommand::List { dag } => {
            let rows: Vec<Value> = client
                .get(&format!("/v1/dags/{dag}/optimizations"))
                .await?;
            if rows.is_empty() {
                println!("{dag} has no optimizations registered");
                println!("attach one with: dee optimization register {dag} hmp");
                return Ok(());
            }
            println!(
                "{:<10} {:<11} {:<7} {:<9} {}",
                "NAME", "TYPE", "PHASE", "STATE", "TABLES"
            );
            for row in &rows {
                println!(
                    "{:<10} {:<11} {:<7} {:<9} {}",
                    row["name"].as_str().unwrap_or(""),
                    row["optimization_type"].as_str().unwrap_or(""),
                    row["step_phase"].as_str().unwrap_or(""),
                    state_of(row),
                    tables_of(row),
                );
            }
            Ok(())
        }

        OptimizationSubcommand::Phase(cmd) => {
            let updated: Value = client
                .patch(
                    &format!("/v1/dags/{}/optimizations/{}", cmd.dag, cmd.optimization),
                    &json!({ "step_phase": cmd.step_phase }),
                )
                .await?;
            print_registration(&updated);
            Ok(())
        }

        OptimizationSubcommand::Deregister { dag, optimization } => {
            let removed: Value = client
                .delete_for(&format!("/v1/dags/{dag}/optimizations/{optimization}"))
                .await?;
            println!("deregistered {optimization} from {dag}");
            let tables = removed["tables"].as_array().cloned().unwrap_or_default();
            if tables.is_empty() {
                println!("  it kept no state, so nothing was torn down");
            } else {
                println!("  tore down {}", tables_of(&removed));
            }
            Ok(())
        }

        OptimizationSubcommand::Available => {
            let rows: Vec<Value> = client.get("/v1/optimizations/available").await?;
            for row in &rows {
                println!(
                    "{}  ({}, steps {} by default)",
                    row["name"].as_str().unwrap_or(""),
                    row["optimization_type"].as_str().unwrap_or(""),
                    row["default_step_phase"].as_str().unwrap_or(""),
                );
                println!("  {}\n", row["doc"].as_str().unwrap_or(""));
            }
            Ok(())
        }
    }
}

fn state_of(row: &Value) -> &'static str {
    if row["active"].as_bool().unwrap_or(false) {
        "stepping"
    } else {
        "converged"
    }
}

fn tables_of(row: &Value) -> String {
    let tables: Vec<&str> = row["tables"]
        .as_array()
        .map(|a| a.iter().filter_map(|t| t.as_str()).collect())
        .unwrap_or_default();
    if tables.is_empty() {
        "-".to_string()
    } else {
        tables.join(", ")
    }
}

fn print_registration(row: &Value) {
    println!(
        "{} registered on {} ({}, steps {})",
        row["name"].as_str().unwrap_or(""),
        row["dag"].as_str().unwrap_or(""),
        row["optimization_type"].as_str().unwrap_or(""),
        row["step_phase"].as_str().unwrap_or(""),
    );
    let tables = tables_of(row);
    if tables == "-" {
        println!("  it keeps no state between runs");
    } else {
        println!("  state in {tables}");
    }
    if let Some(version) = row["result_version"].as_i64() {
        println!("  converged; promoted version {version}");
    }
}
