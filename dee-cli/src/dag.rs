//! `dee dag ...` -- submit and inspect DAGs in the server's registry.

use std::fs;
use std::path::{Path, PathBuf};

use clap::{Args, Subcommand};
use dee::file::DagFile;
use serde_json::{Value, json};

use crate::client::Client;

#[derive(Args)]
pub struct DagCommand {
    #[command(subcommand)]
    pub command: DagSubcommand,
}

#[derive(Subcommand)]
pub enum DagSubcommand {
    /// Submit a DAG definition. Resubmitting unchanged content is a no-op.
    Submit(SubmitArgs),
    /// List registered DAGs.
    #[command(alias = "ls")]
    List,
    /// Show one DAG.
    Get { name: String },
    /// List a DAG's versions, newest first.
    Versions { name: String },
    /// Print a version's definition.
    Show(ShowArgs),
    /// Remove a DAG and its history.
    #[command(alias = "remove")]
    Rm { name: String },
    /// Render a DAG's graph.
    Graph(GraphArgs),
}

#[derive(Args)]
pub struct SubmitArgs {
    /// DAG definition JSON, as produced by `dee convert`.
    pub file: PathBuf,
    /// Name to register under. Defaults to the file's stem.
    #[arg(long)]
    pub name: Option<String>,
    /// Connection this DAG runs against.
    #[arg(short, long)]
    pub target: Option<String>,
    #[arg(long)]
    pub description: Option<String>,
}

#[derive(Args)]
pub struct ShowArgs {
    pub name: String,
    /// Defaults to the DAG's current version.
    #[arg(long)]
    pub version: Option<i32>,
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

#[derive(Args)]
pub struct GraphArgs {
    pub name: String,
    /// Emit graphviz DOT instead of SVG.
    #[arg(long)]
    pub dot: bool,
    #[arg(long)]
    pub version: Option<i32>,
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

pub async fn run(client: &Client, cmd: DagCommand) -> Result<(), Box<dyn std::error::Error>> {
    match cmd.command {
        DagSubcommand::Submit(args) => submit(client, args).await,
        DagSubcommand::List => {
            let rows: Vec<Value> = client.get("/v1/dags").await?;
            if rows.is_empty() {
                println!("no dags registered");
                return Ok(());
            }
            println!("{:<24} {:>8}  {:<16} {}", "NAME", "VERSION", "TARGET", "DESCRIPTION");
            for row in rows {
                println!(
                    "{:<24} {:>8}  {:<16} {}",
                    row["name"].as_str().unwrap_or(""),
                    row["current_version"].as_i64().unwrap_or(0),
                    row["default_target"].as_str().unwrap_or("-"),
                    row["description"].as_str().unwrap_or(""),
                );
            }
            Ok(())
        }
        DagSubcommand::Get { name } => {
            let row: Value = client.get(&format!("/v1/dags/{name}")).await?;
            println!("{}", serde_json::to_string_pretty(&row)?);
            Ok(())
        }
        DagSubcommand::Versions { name } => {
            let rows: Vec<Value> = client.get(&format!("/v1/dags/{name}/versions")).await?;
            println!(
                "{:>8}  {:<11} {:>6} {:>12}  {:<12} {}",
                "VERSION", "ORIGIN", "NODES", "DERIVED FROM", "HASH", "CREATED"
            );
            for row in rows {
                let hash = row["content_hash"].as_str().unwrap_or("");
                println!(
                    "{:>8}  {:<11} {:>6} {:>12}  {:<12} {}",
                    row["version"].as_i64().unwrap_or(0),
                    row["origin"].as_str().unwrap_or(""),
                    row["node_count"].as_i64().unwrap_or(0),
                    row["derived_from_version"]
                        .as_i64()
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| "-".into()),
                    &hash[..hash.len().min(12)],
                    row["created_at"].as_str().unwrap_or(""),
                );
            }
            Ok(())
        }
        DagSubcommand::Show(args) => {
            let path = match args.version {
                Some(v) => format!("/v1/dags/{}/versions/{v}", args.name),
                None => {
                    let dag: Value = client.get(&format!("/v1/dags/{}", args.name)).await?;
                    let v = dag["current_version"].as_i64().unwrap_or(1);
                    format!("/v1/dags/{}/versions/{v}", args.name)
                }
            };
            let detail: Value = client.get(&path).await?;
            let text = serde_json::to_string_pretty(&detail["definition"])?;
            emit(args.output.as_deref(), &text)
        }
        DagSubcommand::Rm { name } => {
            client.delete(&format!("/v1/dags/{name}")).await?;
            println!("removed dag '{name}'");
            Ok(())
        }
        DagSubcommand::Graph(args) => {
            let mut path = format!(
                "/v1/dags/{}/graph?format={}",
                args.name,
                if args.dot { "dot" } else { "svg" }
            );
            if let Some(v) = args.version {
                path.push_str(&format!("&version={v}"));
            }
            let text = client.get_text(&path).await?;
            emit(args.output.as_deref(), &text)
        }
    }
}

async fn submit(client: &Client, args: SubmitArgs) -> Result<(), Box<dyn std::error::Error>> {
    let text = fs::read_to_string(&args.file)
        .map_err(|e| format!("reading {}: {e}", args.file.display()))?;
    let definition: DagFile = serde_json::from_str(&text)
        .map_err(|e| format!("{} is not a dag definition: {e}", args.file.display()))?;

    let name = match &args.name {
        Some(name) => name.clone(),
        None => args
            .file
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or("cannot infer a name from the file; pass --name")?
            .to_string(),
    };

    let body = json!({
        "name": name,
        "definition": definition,
        "target": args.target,
        "description": args.description,
    });
    let result: Value = client.post("/v1/dags", &body).await?;

    let version = result["version"].as_i64().unwrap_or(0);
    if result["created"].as_bool().unwrap_or(false) {
        println!("submitted '{name}' as version {version}");
    } else {
        // Worth saying plainly: the common cause is re-running a pipeline that
        // regenerates an identical definition.
        println!("'{name}' is unchanged; still at version {version}");
    }
    for warning in result["warnings"].as_array().into_iter().flatten() {
        eprintln!("warning: {}", warning.as_str().unwrap_or(""));
    }
    Ok(())
}

fn emit(output: Option<&Path>, text: &str) -> Result<(), Box<dyn std::error::Error>> {
    match output {
        Some(path) => {
            fs::write(path, text)?;
            Ok(())
        }
        None => {
            println!("{text}");
            Ok(())
        }
    }
}
