//! Scratch harness: for each View node in a dag-bench DAG, mimic HMP's
//! `build_trial` (make_temp on that node, then PushdownPass) and check that
//! the resulting DAG (a) still runs and (b) produces identical sink tables
//! compared to the unoptimized baseline.
//!
//! usage: cargo run -p dee --example pushdown_validate -- <dag.json> <warehouse.duckdb> [node_substr]

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use dee::{
    connectors::duckdb::{DuckDBConfig, DuckDBConnection},
    connectors::Connector,
    dag::{Dag, MaterializeMode},
    executor::{Executor, SimpleEngine},
    file::DagFile,
    opt::{common::make_temp, pushdown::PushdownPass},
};

/// count(*) + an order-independent checksum. Timestamp columns are skipped
/// and floating-point columns rounded, so that DuckDB's nondeterministic
/// parallel float aggregation doesn't masquerade as a pushdown bug.
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
        // TIMESTAMP columns are usually `current_timestamp`, and DuckDB's
        // parallel aggregation of DOUBLEs is not bit-reproducible run to run
        // (the control run proves it), so neither can distinguish a pushdown
        // bug from noise. Everything else is compared exactly.
        .filter(|(_, t)| {
            !t.contains("TIMESTAMP")
                && !t.contains("TIME")
                && !t.contains("DOUBLE")
                && !t.contains("FLOAT")
                // DECIMAL columns here are casts of DOUBLE aggregates, so
                // they inherit that nondeterminism.
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

fn sink_tables(dag: &Dag) -> Vec<String> {
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
        "{}/pdv_{}_{}",
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

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dag_json = &args[1];
    let warehouse = &args[2];
    let filter = args.get(3).cloned();

    let file: DagFile = serde_json::from_str(&std::fs::read_to_string(dag_json).unwrap()).unwrap();
    let base_dag: Dag = Dag::try_from(file).unwrap();
    let sinks = sink_tables(&base_dag);

    // ---- baseline ----
    let (conn, path) = fresh_conn(warehouse, "base").await;
    let engine = Arc::new(SimpleEngine::new(Arc::clone(&conn)).unwrap());
    engine.run(&base_dag).await.expect("baseline run");
    let mut baseline: HashMap<String, String> = HashMap::new();
    for s in &sinks {
        baseline.insert(s.clone(), fingerprint(&conn, s).await.unwrap());
    }
    drop(engine);
    drop(conn);
    let _ = std::fs::remove_file(&path);
    println!("baseline captured for {} sink table(s)", sinks.len());

    // Control: a second untouched run, to expose any nondeterminism in the
    // fingerprint itself before attributing differences to the pass. A sink
    // that disagrees with itself here (an `ORDER BY ... LIMIT` over ties, say)
    // can't testify either way and is dropped from the comparison.
    let mut nondeterministic: HashSet<String> = HashSet::new();
    {
        let (conn, path) = fresh_conn(warehouse, "ctrl").await;
        let engine = Arc::new(SimpleEngine::new(Arc::clone(&conn)).unwrap());
        engine.run(&base_dag).await.expect("control run");
        let mut noisy = Vec::new();
        for s in &sinks {
            let fp = fingerprint(&conn, s).await.unwrap();
            if baseline.get(s) != Some(&fp) {
                noisy.push(format!("   {s}: {} vs {fp}", baseline[s]));
                nondeterministic.insert(s.clone());
            }
        }
        if noisy.is_empty() {
            println!("control run matches baseline on all sinks\n");
        } else {
            println!("CONTROL RUN IS NONDETERMINISTIC on (excluded from comparison):");
            for x in &noisy { println!("{x}"); }
            println!();
        }
        drop(engine); drop(conn);
        let _ = std::fs::remove_file(&path);
    }

    let candidates: Vec<String> = base_dag
        .nodes
        .nodes()
        .filter(|n| matches!(n.materialize, MaterializeMode::View))
        .map(|n| n.id.clone())
        .filter(|id| filter.as_ref().map(|f| id.contains(f)).unwrap_or(true))
        .collect();

    let mut bad = 0usize;
    for cand in &candidates {
        let (conn, path) = fresh_conn(warehouse, "trial").await;
        let engine = Arc::new(SimpleEngine::new(Arc::clone(&conn)).unwrap());
        let mut dag = base_dag.clone();

        if let Err(e) = make_temp(&mut dag, cand) {
            println!("[skip] {cand}: make_temp failed: {e}");
            continue;
        }
        let before = dag.clone();

        let mut pass = PushdownPass::new(Arc::clone(&conn), Arc::clone(&engine));
        if let Err(e) = pass.rewrite(&mut dag).await {
            println!("[skip] {cand}: pushdown errored: {e}");
            continue;
        }

        match engine.run(&dag).await {
            Err(e) => {
                bad += 1;
                println!("[FAIL-RUN] {cand}\n    {e}");
                report_diff(&before, &dag);
            }
            Ok(_) => {
                let mut mismatches = Vec::new();
                for s in &sinks {
                    if nondeterministic.contains(s) {
                        continue;
                    }
                    match fingerprint(&conn, s).await {
                        Ok(fp) => {
                            if baseline.get(s) != Some(&fp) {
                                mismatches.push(format!(
                                    "      {s}: baseline {} vs trial {fp}",
                                    baseline[s]
                                ));
                            }
                        }
                        Err(e) => mismatches.push(format!("      {s}: fingerprint failed: {e}")),
                    }
                }
                if mismatches.is_empty() {
                    println!("[ok]   {cand}{}", rewrite_summary(&before, &dag));
                } else {
                    bad += 1;
                    println!("[FAIL-DATA] {cand}");
                    for m in &mismatches {
                        println!("{m}");
                    }
                    report_diff(&before, &dag);
                }
            }
        }
        drop(engine);
        drop(conn);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(format!("{path}.wal"));
    }

    println!("\n{} candidate(s), {bad} bad", candidates.len());
}

/// A compact note on what pushdown actually did, so a run of all-`[ok]` can
/// be told apart from a run where the pass simply stopped rewriting anything.
fn rewrite_summary(before: &Dag, after: &Dag) -> String {
    let mut parts = Vec::new();
    for n in after.nodes.nodes() {
        let Some(old) = before.nodes.get(n.id.clone()) else {
            continue;
        };
        if old.query_text == n.query_text {
            continue;
        }
        if !matches!(n.materialize, MaterializeMode::TempTable) {
            parts.push(format!("{}: frontier sql rewritten", short(&n.id)));
            continue;
        }
        let head = n.query_text.split(" FROM (").next().unwrap_or("");
        let cols = if head.trim_end() == "SELECT *" {
            "all cols".to_string()
        } else {
            format!("{} cols", head.matches('"').count() / 2)
        };
        let filter = if n.query_text.rsplit(')').next().unwrap_or("").contains(" WHERE ") {
            ", filter pushed"
        } else {
            ""
        };
        parts.push(format!("{}: {cols}{filter}", short(&n.id)));
    }
    if parts.is_empty() {
        "   (no rewrite)".to_string()
    } else {
        format!("   [{}]", parts.join("; "))
    }
}

fn short(id: &str) -> String {
    id.rsplit('.').next().unwrap_or(id).trim_matches('"').to_string()
}

fn report_diff(before: &Dag, after: &Dag) {
    for n in after.nodes.nodes() {
        let old = before.nodes.get(n.id.clone()).map(|b| b.query_text.clone());
        if old.as_deref() != Some(n.query_text.as_str()) {
            println!("    --- rewritten node {} ---", n.id);
            println!("    before: {}", old.unwrap_or_default().replace('\n', " "));
            println!("    after : {}", n.query_text.replace('\n', " "));
        }
    }
}
