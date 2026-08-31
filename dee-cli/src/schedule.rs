//! `dee schedule ...` -- put a DAG on a cron schedule.

use clap::{Args, Subcommand};
use serde_json::{Value, json};

use crate::client::Client;

#[derive(Args)]
pub struct ScheduleCommand {
    #[command(subcommand)]
    pub command: ScheduleSubcommand,
}

#[derive(Subcommand)]
pub enum ScheduleSubcommand {
    /// Put a DAG on a schedule, or change the one it has.
    Set(SetArgs),
    /// Show a DAG's schedule.
    Get { name: String },
    /// List every schedule, soonest first.
    #[command(alias = "ls")]
    List,
    /// Stop a schedule firing, keeping its definition.
    Pause { name: String },
    /// Resume a paused schedule. The next firing is computed from now.
    Resume { name: String },
    /// Remove a schedule entirely.
    Unset { name: String },
    /// Show windows that produced no run, and why.
    Skips {
        name: String,
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
}

#[derive(Args)]
pub struct SetArgs {
    pub name: String,
    /// A cron expression, e.g. "0 3 * * *" for 3am daily.
    #[arg(long)]
    pub cron: String,
    /// IANA timezone the expression is interpreted in.
    #[arg(long, default_value = "UTC")]
    pub timezone: String,
    /// Connection to run against. Defaults to the DAG's target.
    #[arg(short, long)]
    pub target: Option<String>,
    /// Create the schedule paused.
    #[arg(long)]
    pub paused: bool,
}

pub async fn run(
    client: &Client,
    cmd: ScheduleCommand,
) -> Result<(), Box<dyn std::error::Error>> {
    match cmd.command {
        ScheduleSubcommand::Set(args) => {
            let body = json!({
                "cron": args.cron,
                "timezone": args.timezone,
                "enabled": !args.paused,
                "target": args.target,
            });
            let view: Value = client
                .put(&format!("/v1/dags/{}/schedule", args.name), &body)
                .await?;
            print_schedule(&args.name, &view);
            Ok(())
        }
        ScheduleSubcommand::Get { name } => {
            let view: Value = client.get(&format!("/v1/dags/{name}/schedule")).await?;
            print_schedule(&name, &view);
            Ok(())
        }
        ScheduleSubcommand::List => {
            let rows: Vec<Value> = client.get("/v1/schedules").await?;
            if rows.is_empty() {
                println!("no schedules");
                return Ok(());
            }
            println!(
                "{:<24} {:<16} {:<18} {:<8} {}",
                "DAG", "CRON", "TIMEZONE", "STATE", "NEXT RUN"
            );
            for row in rows {
                println!(
                    "{:<24} {:<16} {:<18} {:<8} {}",
                    row["dag_name"].as_str().unwrap_or(""),
                    row["cron"].as_str().unwrap_or(""),
                    row["timezone"].as_str().unwrap_or(""),
                    if row["enabled"].as_bool().unwrap_or(false) {
                        "active"
                    } else {
                        "paused"
                    },
                    row["next_fire_at"].as_str().unwrap_or("-"),
                );
            }
            Ok(())
        }
        ScheduleSubcommand::Pause { name } => {
            let view: Value = client
                .post(&format!("/v1/dags/{name}/schedule/pause"), &json!({}))
                .await?;
            print_schedule(&name, &view);
            Ok(())
        }
        ScheduleSubcommand::Resume { name } => {
            let view: Value = client
                .post(&format!("/v1/dags/{name}/schedule/resume"), &json!({}))
                .await?;
            print_schedule(&name, &view);
            Ok(())
        }
        ScheduleSubcommand::Unset { name } => {
            client.delete(&format!("/v1/dags/{name}/schedule")).await?;
            println!("removed the schedule for '{name}'");
            Ok(())
        }
        ScheduleSubcommand::Skips { name, limit } => {
            let rows: Vec<Value> = client
                .get(&format!("/v1/dags/{name}/schedule/skips?limit={limit}"))
                .await?;
            if rows.is_empty() {
                println!("no skipped windows for '{name}'");
                return Ok(());
            }
            println!("{:<28} {:<16} {:>8}  {}", "WINDOW", "REASON", "WINDOWS", "DETAIL");
            for row in rows {
                println!(
                    "{:<28} {:<16} {:>8}  {}",
                    row["scheduled_for"].as_str().unwrap_or(""),
                    row["reason"].as_str().unwrap_or(""),
                    row["windows_skipped"].as_i64().unwrap_or(1),
                    row["detail"].as_str().unwrap_or(""),
                );
            }
            Ok(())
        }
    }
}

fn print_schedule(name: &str, view: &Value) {
    println!(
        "{name}: {} ({})",
        view["cron"].as_str().unwrap_or(""),
        view["timezone"].as_str().unwrap_or("UTC")
    );
    if let Some(description) = view["description"].as_str() {
        println!("  {description}");
    }
    if view["enabled"].as_bool().unwrap_or(false) {
        println!("  next run: {}", view["next_fire_at"].as_str().unwrap_or("-"));
    } else {
        println!("  paused");
    }
    if let Some(target) = view["target"].as_str() {
        println!("  target: {target}");
    }
}
