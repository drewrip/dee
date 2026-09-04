//! `dee connection ...` -- manage the server's named targets.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use clap::{Args, Subcommand};
use dee::connections::Connection;
use dee::connectors::{duckdb::DuckDBConfig, postgres::PostgresConfig};
use serde_json::{Value, json};

use crate::client::Client;

#[derive(Args)]
pub struct ConnectionCommand {
    #[command(subcommand)]
    pub command: ConnectionSubcommand,
}

#[derive(Subcommand)]
pub enum ConnectionSubcommand {
    /// Register a target, or import targets from an existing connections.json.
    Add(AddArgs),
    /// List registered targets. Credentials are redacted by the server.
    #[command(alias = "ls")]
    List,
    /// Show one target.
    Get { name: String },
    /// Remove a target.
    #[command(alias = "remove")]
    Rm { name: String },
    /// Connect to a target and run a trivial query.
    Test { name: String },
}

#[derive(clap::ValueEnum, Clone, Debug, PartialEq)]
pub enum ConnectionKind {
    Duckdb,
    Postgres,
}

#[derive(Args)]
pub struct AddArgs {
    /// Name to register the target under. Required unless --from-file is used.
    pub name: Option<String>,

    #[arg(long = "type", value_name = "KIND")]
    pub kind: Option<ConnectionKind>,

    /// Import from an existing connections.json (a map of name to connection).
    #[arg(short = 'f', long, conflicts_with = "kind")]
    pub from_file: Option<PathBuf>,

    /// Replace the target if it already exists.
    #[arg(long)]
    pub replace: bool,

    // -- duckdb --
    #[arg(long)]
    pub database: Option<String>,
    #[arg(long)]
    pub threads: Option<i64>,
    #[arg(long)]
    pub max_memory: Option<String>,

    // -- postgres --
    #[arg(long)]
    pub host: Option<String>,
    #[arg(long)]
    pub port: Option<i32>,
    #[arg(long)]
    pub user: Option<String>,
    #[arg(long)]
    pub password: Option<String>,

    /// Size of the connection pool the server keeps open for this target.
    /// Postgres only: DuckDB's pool size is the DAG's degree of parallelism,
    /// which is a property of the DAG rather than of the connection.
    #[arg(long)]
    pub num_connections: Option<u32>,
}

pub async fn run(
    client: &Client,
    cmd: ConnectionCommand,
) -> Result<(), Box<dyn std::error::Error>> {
    match cmd.command {
        ConnectionSubcommand::Add(args) => add(client, args).await,
        ConnectionSubcommand::List => {
            let rows: Vec<Value> = client.get("/v1/connections").await?;
            if rows.is_empty() {
                println!("no connections registered");
                return Ok(());
            }
            println!("{:<24} {:<10} {}", "NAME", "KIND", "TARGET");
            for row in rows {
                println!(
                    "{:<24} {:<10} {}",
                    row["name"].as_str().unwrap_or(""),
                    row["kind"].as_str().unwrap_or(""),
                    describe(&row["config"]),
                );
            }
            Ok(())
        }
        ConnectionSubcommand::Get { name } => {
            let row: Value = client.get(&format!("/v1/connections/{name}")).await?;
            println!("{}", serde_json::to_string_pretty(&row)?);
            Ok(())
        }
        ConnectionSubcommand::Rm { name } => {
            client.delete(&format!("/v1/connections/{name}")).await?;
            println!("removed connection '{name}'");
            Ok(())
        }
        ConnectionSubcommand::Test { name } => {
            let result: Value = client
                .post(&format!("/v1/connections/{name}/test"), &json!({}))
                .await?;
            if result["ok"].as_bool().unwrap_or(false) {
                println!(
                    "ok ({}ms, plan timings are {})",
                    result["latency_ms"].as_i64().unwrap_or(0),
                    result["time_basis"].as_str().unwrap_or("unknown"),
                );
                Ok(())
            } else {
                Err(format!(
                    "connection '{name}' failed: {}",
                    result["error"].as_str().unwrap_or("unknown error")
                )
                .into())
            }
        }
    }
}

async fn add(client: &Client, args: AddArgs) -> Result<(), Box<dyn std::error::Error>> {
    let path = if args.replace {
        "/v1/connections?upsert=true"
    } else {
        "/v1/connections"
    };

    // Importing the old file format is how an existing setup moves over, and
    // it is what the benchmark harness already generates per project.
    if let Some(file) = &args.from_file {
        let text = fs::read_to_string(file)
            .map_err(|e| format!("reading {}: {e}", file.display()))?;
        let entries: HashMap<String, Connection> = serde_json::from_str(&text)
            .map_err(|e| format!("{} is not a connections.json map: {e}", file.display()))?;

        let wanted = args.name.as_deref();
        let mut imported = 0;
        for (name, config) in entries {
            if let Some(only) = wanted {
                if name != only {
                    continue;
                }
            }
            let body = json!({"name": name, "config": config});
            post_connection(client, path, &body).await?;
            println!("registered connection '{name}'");
            imported += 1;
        }
        if imported == 0 {
            return Err(match wanted {
                Some(only) => format!("'{only}' is not in {}", file.display()).into(),
                None => format!("{} contains no connections", file.display()).into(),
            });
        }
        return Ok(());
    }

    let name = args
        .name
        .clone()
        .ok_or("a connection name is required (or use --from-file)")?;
    let kind = args
        .kind
        .clone()
        .ok_or("--type duckdb|postgres is required (or use --from-file)")?;
    let config = build_config(&kind, &args)?;

    let body = json!({"name": name, "config": config});
    post_connection(client, path, &body).await?;
    println!("registered connection '{name}'");
    Ok(())
}

/// The server reports a duplicate in HTTP terms; the person at the terminal
/// needs the flag that fixes it, not the query parameter behind it.
async fn post_connection(
    client: &Client,
    path: &str,
    body: &Value,
) -> Result<(), Box<dyn std::error::Error>> {
    match client.post::<Value, Value>(path, body).await {
        Ok(_) => Ok(()),
        Err(e) => {
            if let Some(api) = e.downcast_ref::<crate::client::ApiError>() {
                if api.code == "conflict" {
                    return Err(format!(
                        "{}; pass --replace to overwrite it",
                        api.message.split(';').next().unwrap_or(&api.message)
                    )
                    .into());
                }
            }
            Err(e)
        }
    }
}

/// Build the connection through serde rather than struct literals:
/// `PostgresConfig`'s fields are private, so JSON is the only way to construct
/// one from outside the library.
fn build_config(
    kind: &ConnectionKind,
    args: &AddArgs,
) -> Result<Connection, Box<dyn std::error::Error>> {
    match kind {
        ConnectionKind::Duckdb => {
            let database = args
                .database
                .clone()
                .ok_or("--database is required for a duckdb connection")?;
            if args.num_connections.is_some() {
                return Err("--num-connections does not apply to duckdb: pooled DuckDB \
                            connections share one database, so the pool would only be a \
                            second cap on node concurrency. Set the DAG's max_parallelism \
                            instead, or let ParallelismTuning measure it"
                    .into());
            }
            let mut config = DuckDBConfig::new_from_path(database);
            config.threads = args.threads;
            config.max_memory = args.max_memory.clone();
            Ok(Connection::DuckDB(config))
        }
        ConnectionKind::Postgres => {
            let value = json!({
                "host": args.host.clone().ok_or("--host is required for a postgres connection")?,
                "port": args.port,
                "user": args.user.clone().ok_or("--user is required for a postgres connection")?,
                "password": args.password.clone().unwrap_or_default(),
                "database": args.database.clone()
                    .ok_or("--database is required for a postgres connection")?,
                "num_connections": args.num_connections,
            });
            let config: PostgresConfig = serde_json::from_value(value)?;
            Ok(Connection::Postgres(config))
        }
    }
}

/// A one-line "where does this point" for the list view.
fn describe(config: &Value) -> String {
    // Dispatch on the `type` tag rather than the shape: both configs have a
    // `database` field, so structural matching silently reads a postgres
    // connection as a duckdb one.
    match config.get("type").and_then(Value::as_str) {
        Some("duckdb") => config
            .get("database")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        Some("postgres") => format!(
            "{}:{}/{}",
            config.get("host").and_then(Value::as_str).unwrap_or(""),
            config.get("port").and_then(Value::as_i64).unwrap_or(5432),
            config.get("database").and_then(Value::as_str).unwrap_or(""),
        ),
        _ => String::new(),
    }
}
