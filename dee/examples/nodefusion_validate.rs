//! Scratch harness: run a DAG as authored, then run it under NodeFusion, and
//! check that every Table it produces holds the same rows.
//!
//! NodeFusion rewrites every Table that reads a shared model, so "it compiles
//! and the pass reports success" is not evidence of much. This runs the thing.
//!
//! usage:
//!   cargo run -p dee --example nodefusion_validate -- <dag.json> <warehouse.duckdb>
//!   cargo run -p dee --example nodefusion_validate -- <dag.json> postgres://user:pass@host:port/db
//!
//! On DuckDB each run gets a fresh copy of the warehouse file. On PostgreSQL
//! the sources live in the server, so each run drops what the previous one
//! created before it starts, and the database is otherwise left as it was.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use async_trait::async_trait;
use dee::{
    connectors::Connector,
    connectors::duckdb::{DuckDBConfig, DuckDBConnection},
    connectors::postgres::{PostgresConfig, PostgresConnection},
    dag::{Dag, MaterializeMode},
    executor::{Executor, SimpleEngine},
    file::DagFile,
    opt::nodefusion::NodeFusionPass,
};
use sqlx::{
    PgPool,
    postgres::{PgConnectOptions, PgPoolOptions},
};

/// Where a run happens, and how its tables are read back.
#[async_trait]
trait Target: Send + Sync {
    type Conn: Connector + Send + Sync + 'static;

    /// A connection with the sources in it, ready for one run.
    async fn open(&self, tag: &str) -> Arc<Self::Conn>;

    /// Release what `open` handed out, after `dag`'s relations are dropped.
    async fn close(&self, tag: &str, conn: Arc<Self::Conn>);

    /// Columns and types, in order, then row count and an order-independent
    /// content hash.
    ///
    /// Float and time columns are left out of the hash: parallel float
    /// aggregation and `current_timestamp` both vary run to run, so neither can
    /// tell a rewrite bug from noise. Their names and types are still compared.
    async fn fingerprint(&self, conn: &Self::Conn, table: &str) -> Result<String, String>;
}

fn comparable(ty: &str) -> bool {
    let t = ty.to_ascii_lowercase();
    !(t.contains("time")
        || t.contains("date")
        || t.contains("double")
        || t.contains("float")
        || t.contains("real"))
}

// ---------------------------------------------------------------------------
// DuckDB
// ---------------------------------------------------------------------------

struct DuckTarget {
    src: String,
}

impl DuckTarget {
    fn path(&self, tag: &str) -> String {
        format!(
            "{}/nfv_{}_{}/warehouse.duckdb",
            std::env::var("SCRATCH").unwrap_or("/tmp".into()),
            tag,
            std::process::id()
        )
    }
}

#[async_trait]
impl Target for DuckTarget {
    type Conn = DuckDBConnection;

    async fn open(&self, tag: &str) -> Arc<DuckDBConnection> {
        let path = self.path(tag);
        std::fs::create_dir_all(std::path::Path::new(&path).parent().unwrap()).unwrap();
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(format!("{path}.wal"));
        std::fs::copy(&self.src, &path).unwrap();
        DuckDBConnection::new(DuckDBConfig::new_from_path(path))
            .await
            .unwrap()
    }

    async fn close(&self, tag: &str, conn: Arc<DuckDBConnection>) {
        drop(conn);
        let path = self.path(tag);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(format!("{path}.wal"));
    }

    async fn fingerprint(&self, conn: &DuckDBConnection, table: &str) -> Result<String, String> {
        let c = conn.pool.get().map_err(|e| e.to_string())?;
        let cols: Vec<(String, String)> = {
            let mut stmt = c
                .prepare(&format!(
                    "SELECT column_name, column_type FROM (DESCRIBE SELECT * FROM {table})"
                ))
                .map_err(|e| e.to_string())?;
            let rows = stmt
                .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))
                .map_err(|e| e.to_string())?;
            rows.filter_map(|r| r.ok()).collect()
        };
        let shape = cols.iter().map(|(n, t)| format!("{n} {t}")).collect::<Vec<_>>().join(", ");
        let exprs: Vec<String> = cols
            .iter()
            .filter(|(_, t)| comparable(t))
            .map(|(n, _)| format!("\"{n}\""))
            .collect();
        let n: i64 = c
            .query_row(&format!("SELECT count(*) FROM {table}"), [], |r| r.get(0))
            .map_err(|e| e.to_string())?;
        if exprs.is_empty() {
            return Ok(format!("[{shape}] n={n} (no comparable columns)"));
        }
        let sql = format!(
            "SELECT coalesce(sum(hash(t)::HUGEINT), 0)::VARCHAR FROM (SELECT {} FROM {table}) AS t",
            exprs.join(", ")
        );
        let h: String = c
            .query_row(&sql, [], |r| r.get(0))
            .map_err(|e| format!("{e} - {sql}"))?;
        Ok(format!("[{shape}] n={n} h={h}"))
    }
}

// ---------------------------------------------------------------------------
// PostgreSQL
// ---------------------------------------------------------------------------

struct PgTarget {
    config: serde_json::Value,
    /// For reading tables back: the connector keeps its pool to itself.
    pool: PgPool,
}

impl PgTarget {
    /// `postgres://user:password@host:port/database`
    async fn from_url(url: &str) -> Self {
        let rest = url
            .strip_prefix("postgres://")
            .or_else(|| url.strip_prefix("postgresql://"))
            .expect("a postgres:// URL");
        let (auth, location) = rest.rsplit_once('@').expect("user:password@host:port/db");
        let (user, password) = auth.split_once(':').unwrap_or((auth, ""));
        let (hostport, database) = location.split_once('/').expect("a database name");
        let (host, port) = hostport.split_once(':').unwrap_or((hostport, "5432"));
        let port: u16 = port.parse().expect("a numeric port");
        let config = serde_json::json!({
            "host": host, "port": port as i32, "user": user, "password": password,
            "database": database, "num_connections": 16,
        });
        let pool = PgPoolOptions::new()
            .max_connections(2)
            .connect_with(
                PgConnectOptions::new()
                    .host(host)
                    .port(port)
                    .username(user)
                    .password(password)
                    .database(database),
            )
            .await
            .expect("connect to postgres");
        Self { config, pool }
    }
}

#[async_trait]
impl Target for PgTarget {
    type Conn = PostgresConnection;

    async fn open(&self, _tag: &str) -> Arc<PostgresConnection> {
        let config: PostgresConfig = serde_json::from_value(self.config.clone()).unwrap();
        PostgresConnection::new(config).await.expect("postgres")
    }

    async fn close(&self, _tag: &str, conn: Arc<PostgresConnection>) {
        drop(conn);
    }

    async fn fingerprint(&self, _conn: &PostgresConnection, table: &str) -> Result<String, String> {
        let cols: Vec<(String, String)> = sqlx::query_as(
            "SELECT attname::text, format_type(atttypid, atttypmod) FROM pg_attribute \
             WHERE attrelid = $1::regclass AND attnum > 0 AND NOT attisdropped ORDER BY attnum",
        )
        .bind(table)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| format!("{e} ({table})"))?;
        let shape = cols.iter().map(|(n, t)| format!("{n} {t}")).collect::<Vec<_>>().join(", ");
        let exprs: Vec<String> = cols
            .iter()
            .filter(|(_, t)| comparable(t))
            .map(|(n, _)| format!("\"{}\"", n.replace('"', "\"\"")))
            .collect();
        let n: i64 = sqlx::query_scalar(&format!("SELECT count(*) FROM {table}"))
            .fetch_one(&self.pool)
            .await
            .map_err(|e| format!("{e} ({table})"))?;
        if exprs.is_empty() {
            return Ok(format!("[{shape}] n={n} (no comparable columns)"));
        }
        let sql = format!(
            "SELECT coalesce(sum(('x' || substr(md5(t::text), 1, 16))::bit(64)::bigint::numeric), 0)::text \
             FROM (SELECT {} FROM {table}) AS t",
            exprs.join(", ")
        );
        let h: String = sqlx::query_scalar(&sql)
            .fetch_one(&self.pool)
            .await
            .map_err(|e| format!("{e} - {sql}"))?;
        Ok(format!("[{shape}] n={n} h={h}"))
    }
}

// ---------------------------------------------------------------------------
// The comparison
// ---------------------------------------------------------------------------

fn tables(dag: &Dag) -> Vec<String> {
    let mut v: Vec<String> = dag
        .nodes
        .nodes()
        .filter(|n| matches!(n.materialize, MaterializeMode::Table))
        .map(|n| n.id.clone())
        .collect();
    v.sort();
    v
}

/// Run `dag` and fingerprint each of `tables`, then drop what it made.
async fn run_and_fingerprint<T: Target>(
    target: &T,
    tag: &str,
    dag: &Dag,
    tables: &[String],
) -> Result<HashMap<String, String>, String> {
    let conn = target.open(tag).await;
    let engine = Arc::new(SimpleEngine::new(Arc::clone(&conn)).unwrap());
    // A previous run, or a benchmark that left its relations behind, must not
    // be what gets fingerprinted.
    let _ = engine.cleanup(dag).await;
    let result = engine.run(dag).await.map_err(|e| e.to_string());
    let mut out = HashMap::new();
    let mut read_error = None;
    if result.is_ok() {
        for t in tables {
            match target.fingerprint(&conn, t).await {
                Ok(fp) => {
                    out.insert(t.clone(), fp);
                }
                Err(e) => {
                    read_error = Some(e);
                    break;
                }
            }
        }
    }
    let _ = engine.cleanup(dag).await;
    drop(engine);
    target.close(tag, conn).await;
    result?;
    match read_error {
        Some(e) => Err(e),
        None => Ok(out),
    }
}

/// One configuration to check: either the pass's own rules, or an exact set of
/// node IDs the way the adaptive search installs one.
enum Variant {
    Rule(Option<Vec<String>>),
    Exact(HashSet<String>),
}

/// The bare table name of a qualified node ID, for matching a CTE name.
fn bare(id: &str) -> String {
    id.rsplit('.').next().unwrap_or(id).trim_matches('"').to_string()
}

async fn validate<T: Target>(target: T, base_dag: Dag) -> bool {
    let table_ids = tables(&base_dag);
    println!("{} Table node(s): {}", table_ids.len(), table_ids.join(", "));

    // ---- baseline, and a control to expose a DAG that disagrees with itself
    // before any difference is attributed to the pass ----
    let baseline = run_and_fingerprint(&target, "base", &base_dag, &table_ids)
        .await
        .expect("the baseline DAG should run");
    let control = run_and_fingerprint(&target, "ctrl", &base_dag, &table_ids)
        .await
        .expect("the control DAG should run");
    let noisy: Vec<&String> = table_ids
        .iter()
        .filter(|t| baseline.get(*t) != control.get(*t))
        .collect();
    if noisy.is_empty() {
        println!("control run matches baseline on every Table\n");
    } else {
        println!("NONDETERMINISTIC, excluded from the comparison:");
        for t in &noisy {
            println!("   {t}");
        }
        println!();
    }
    if std::env::var("NFV_FP").is_ok() {
        for t in &table_ids {
            println!("   {t}: {}", baseline[t]);
        }
        println!();
    }

    let mut variants: Vec<(String, Variant)> = vec![
        ("the rule: CTEs with two or more readers (the default)".into(), Variant::Rule(None)),
        ("nothing materialized".into(), Variant::Rule(Some(Vec::new()))),
    ];

    // Every configuration the adaptive search can actually install, through the
    // code path it installs them by.
    //
    // Not the same as the override above: `rewrite_with` takes an exact set of
    // node IDs, while the override also matches bare table names, so checking
    // only the override would leave the search's own path unchecked. These are
    // the single flips -- each decidable CTE turned the other way from the
    // default rule -- which is what a trial installs and what a promotion
    // installs. A search that produced a rollup with different *rows* in it
    // would be a much worse bug than a search that ranked badly.
    {
        let conn = target.open("probe").await;
        let decidable = NodeFusionPass::new(None).decidable_ctes(&base_dag);
        // Whatever the default rule turned on, read back off a default rewrite
        // rather than re-derived here.
        let mut d = base_dag.clone();
        let fused_default = NodeFusionPass::new(None)
            .rewrite(conn.as_ref(), &mut d)
            .await
            .is_ok();
        let fused_sql = d
            .nodes
            .nodes()
            .find(|n| n.id.contains("dee_fused"))
            .map(|n| n.query_text.clone())
            .unwrap_or_default();
        let default_set: HashSet<String> = decidable
            .iter()
            .filter(|id| {
                fused_default && fused_sql.contains(&format!("n_{} AS MATERIALIZED", bare(id)))
            })
            .cloned()
            .collect();
        for id in &decidable {
            let mut set = default_set.clone();
            if !set.remove(id) {
                set.insert(id.clone());
            }
            let mut members: Vec<String> = set.iter().cloned().collect();
            members.sort();
            variants.push((
                format!("the search flipping '{id}' -> {{{}}}", members.join(", ")),
                Variant::Exact(set),
            ));
        }
        target.close("probe", conn).await;
    }

    let mut failures = 0usize;
    let mut ran = 0usize;
    for (label, variant) in variants {
        let label = label.as_str();
        // The rewrite reads each kind's column types off the engine, so it
        // needs a connection with the sources in it.
        let conn = target.open("rewrite").await;
        let mut dag = base_dag.clone();
        let mut pass = match &variant {
            Variant::Rule(over) => NodeFusionPass::new(over.clone()),
            Variant::Exact(_) => NodeFusionPass::new(None),
        };
        let rewritten = match &variant {
            Variant::Rule(..) => pass.rewrite(conn.as_ref(), &mut dag).await,
            Variant::Exact(set) => pass.rewrite_with(conn.as_ref(), &mut dag, set).await,
        };
        target.close("rewrite", conn).await;
        let record = match rewritten {
            Ok(record) => record,
            Err(e) => {
                failures += 1;
                println!("[FAIL] {label}: the pass errored: {e}");
                continue;
            }
        };
        if record.changes_applied == 0 {
            println!("[skip] {label}: this DAG was not fused");
            if let dee::opt::PassDetail::NodeFusion(d) = &record.detail {
                println!("       {}", d.outcome);
            }
            continue;
        }

        if std::env::var("NFV_SQL").is_ok() {
            if let Some(f) = dag.nodes.nodes().find(|n| n.id.contains("dee_fused")) {
                println!("--- rollup query for {label} ---\n{}\n", f.query_text);
            }
        }

        match run_and_fingerprint(&target, "trial", &dag, &table_ids).await {
            Err(e) => {
                failures += 1;
                println!("[FAIL-RUN] {label}\n    {e}");
                if let Some(f) = dag.nodes.nodes().find(|n| n.id.contains("dee_fused")) {
                    println!("    rollup query:\n{}", f.query_text);
                }
            }
            Ok(fused) => {
                let mismatches: Vec<String> = table_ids
                    .iter()
                    .filter(|t| !noisy.contains(t))
                    .filter(|t| baseline.get(*t) != fused.get(*t))
                    .map(|t| {
                        format!(
                            "      {t}: baseline {:?} vs rewritten {:?}",
                            baseline.get(t),
                            fused.get(t)
                        )
                    })
                    .collect();
                if mismatches.is_empty() {
                    ran += 1;
                    println!("[ok]   {label}: every Table matches the baseline, columns and types included");
                } else {
                    failures += 1;
                    println!("[FAIL] {label}:");
                    for m in &mismatches {
                        println!("{m}");
                    }
                }
            }
        }
    }

    if failures > 0 {
        println!("\n{failures} variant(s) failed");
        return false;
    }
    if ran == 0 {
        println!("\nno variant was fused, so nothing was compared");
        return false;
    }
    println!("\nall {ran} rewritten variant(s) agree with the baseline");
    true
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: nodefusion_validate <dag.json> <warehouse.duckdb | postgres://user:pass@host:port/db>"
        );
        std::process::exit(2);
    }
    let file: DagFile = serde_json::from_str(&std::fs::read_to_string(&args[1]).unwrap()).unwrap();
    let base_dag: Dag = Dag::try_from(file).unwrap();

    let ok = if args[2].starts_with("postgres://") || args[2].starts_with("postgresql://") {
        validate(PgTarget::from_url(&args[2]).await, base_dag).await
    } else {
        validate(DuckTarget { src: args[2].clone() }, base_dag).await
    };
    std::process::exit(if ok { 0 } else { 1 });
}
