//! Scratch harness: run a DAG unfused, then run it under NodeFusion, and check
//! that every Table it produces holds the same rows.
//!
//! Fusion is a whole-DAG rewrite -- one query standing in for the entire
//! upstream graph -- so "it compiles and the pass reports success" is not
//! evidence of much. This runs the thing.
//!
//! usage: cargo run -p dee --example nodefusion_validate -- <dag.json> <warehouse.duckdb>

use std::{collections::HashMap, sync::Arc};

use dee::{
    connectors::Connector,
    connectors::duckdb::{DuckDBConfig, DuckDBConnection},
    dag::{Dag, MaterializeMode},
    executor::{Executor, SimpleEngine},
    file::DagFile,
    opt::nodefusion::NodeFusionPass,
};

/// The relation's columns as `name type`, in order. Compared alongside the
/// fingerprint because the fingerprint cannot see a type change: it skips
/// float and decimal columns, so a rewrite that quietly turned a HUGEINT into
/// a DECIMAL(38,0) would pass it. (One did.)
async fn shape(conn: &DuckDBConnection, table: &str) -> Result<String, String> {
    let c = conn.pool.get().map_err(|e| e.to_string())?;
    let mut stmt = c
        .prepare(&format!(
            "SELECT column_name, column_type FROM (DESCRIBE SELECT * FROM {table})"
        ))
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([], |r| Ok(format!("{} {}", r.get::<_, String>(0)?, r.get::<_, String>(1)?)))
        .map_err(|e| e.to_string())?;
    Ok(rows.filter_map(|r| r.ok()).collect::<Vec<_>>().join(", "))
}

/// count(*) plus an order-independent checksum. Timestamp and floating-point
/// columns are skipped: `current_timestamp` and DuckDB's parallel float
/// aggregation both vary run to run, so neither can distinguish a fusion bug
/// from noise. Everything else is compared exactly.
async fn fingerprint(conn: &DuckDBConnection, table: &str) -> Result<String, String> {
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
    let exprs: Vec<String> = cols
        .iter()
        .filter(|(_, t)| {
            !t.contains("TIMESTAMP")
                && !t.contains("TIME")
                && !t.contains("DOUBLE")
                && !t.contains("FLOAT")
                && !t.contains("DECIMAL")
                && !t.contains("NUMERIC")
        })
        .map(|(n, _)| format!("\"{n}\""))
        .collect();
    let n: i64 = c
        .query_row(&format!("SELECT count(*) FROM {table}"), [], |r| r.get(0))
        .map_err(|e| e.to_string())?;
    if exprs.is_empty() {
        return Ok(format!("n={n} (no comparable columns)"));
    }
    let sql = format!(
        "SELECT coalesce(sum(hash(t)::HUGEINT), 0)::VARCHAR FROM (SELECT {} FROM {table}) AS t",
        exprs.join(", ")
    );
    let h: String = c
        .query_row(&sql, [], |r| r.get(0))
        .map_err(|e| format!("{e} - {sql}"))?;
    Ok(format!("n={n} h={h}"))
}

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

async fn fresh_conn(src: &str, tag: &str) -> (Arc<DuckDBConnection>, String) {
    let dir = format!(
        "{}/nfv_{}_{}",
        std::env::var("SCRATCH").unwrap_or("/tmp".into()),
        tag,
        std::process::id()
    );
    std::fs::create_dir_all(&dir).unwrap();
    let path = format!("{dir}/warehouse.duckdb");
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(format!("{path}.wal"));
    std::fs::copy(src, &path).unwrap();
    let conn = DuckDBConnection::new(DuckDBConfig::new_from_path(path.clone()))
        .await
        .unwrap();
    (conn, path)
}

/// Run `dag` on a fresh copy of the warehouse and fingerprint each of `tables`.
async fn run_and_fingerprint(
    warehouse: &str,
    tag: &str,
    dag: &Dag,
    tables: &[String],
) -> Result<HashMap<String, String>, String> {
    let (conn, path) = fresh_conn(warehouse, tag).await;
    let engine = Arc::new(SimpleEngine::new(Arc::clone(&conn)).unwrap());
    let result = engine.run(dag).await.map_err(|e| e.to_string());
    let mut out = HashMap::new();
    if result.is_ok() {
        for t in tables {
            let fp = fingerprint(&conn, t).await?;
            out.insert(t.clone(), format!("[{}] {fp}", shape(&conn, t).await?));
        }
    }
    drop(engine);
    drop(conn);
    let _ = std::fs::remove_file(&path);
    result.map(|_| out)
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: nodefusion_validate <dag.json> <warehouse.duckdb>");
        std::process::exit(2);
    }
    let dag_json = &args[1];
    let warehouse = &args[2];

    let file: DagFile = serde_json::from_str(&std::fs::read_to_string(dag_json).unwrap()).unwrap();
    let base_dag: Dag = Dag::try_from(file).unwrap();
    let table_ids = tables(&base_dag);
    println!("{} Table node(s): {}", table_ids.len(), table_ids.join(", "));

    // ---- baseline, and a control to expose a DAG that disagrees with itself
    // before any difference is attributed to the pass ----
    let baseline = run_and_fingerprint(warehouse, "base", &base_dag, &table_ids)
        .await
        .expect("the baseline DAG should run");
    let control = run_and_fingerprint(warehouse, "ctrl", &base_dag, &table_ids)
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

    // ---- the fused DAG, under each setting of the materialization knob ----
    let variants: Vec<(&str, bool, bool, Option<Vec<String>>)> = vec![
        ("plain View CTEs (the default)", false, false, None),
        ("the naive rule: Views more than one Table reads", false, true, None),
        ("every CTE materialized", true, false, None),
        ("nothing materialized", false, false, Some(Vec::new())),
    ];

    let mut failures = 0usize;
    let mut ran = 0usize;
    for (label, materialize, naive, override_list) in variants {
        let (conn, path) = fresh_conn(warehouse, "trial").await;
        let engine = Arc::new(SimpleEngine::new(Arc::clone(&conn)).unwrap());
        let mut dag = base_dag.clone();
        if let Err(e) = engine.resolve_schemas(&mut dag).await {
            println!("[skip] {label}: resolve_schemas failed: {e}");
            drop(engine);
            drop(conn);
            let _ = std::fs::remove_file(&path);
            continue;
        }
        drop(engine);
        drop(conn);
        let _ = std::fs::remove_file(&path);

        let mut pass = NodeFusionPass::new(materialize, naive, override_list);
        let record = match pass.rewrite(&mut dag) {
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
                println!("--- fused query for {label} ---\n{}\n", f.query_text);
            }
        }

        match run_and_fingerprint(warehouse, "trial", &dag, &table_ids).await {
            Err(e) => {
                failures += 1;
                println!("[FAIL-RUN] {label}\n    {e}");
                if let Some(f) = dag.nodes.nodes().find(|n| n.id.contains("dee_fused")) {
                    println!("    fused query:\n{}", f.query_text);
                }
            }
            Ok(fused) => {
                let mismatches: Vec<String> = table_ids
                    .iter()
                    .filter(|t| !noisy.contains(t))
                    .filter(|t| baseline.get(*t) != fused.get(*t))
                    .map(|t| {
                        format!(
                            "      {t}: baseline {:?} vs fused {:?}",
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
        std::process::exit(1);
    }
    if ran == 0 {
        println!("\nno variant was fused, so nothing was compared");
        std::process::exit(1);
    }
    println!("\nall {ran} fused variant(s) agree with the baseline");
}
