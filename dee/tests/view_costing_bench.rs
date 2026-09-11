//! Microbenchmark: how accurately does each HMP costing method price a VIEW?
//!
//! Not a test -- an experiment, run explicitly:
//!
//! ```text
//! BENCH_BACKEND=duckdb BENCH_CATALOG=catalog.json BENCH_OUT=out.json \
//!   cargo test --release -p dee --test view_costing_bench -- --ignored --nocapture
//! ```
//!
//! For each micro-DAG it runs the DAG as authored (every candidate inlined) with
//! plan collection on, reads a per-View cost off that one run under each costing
//! method, and then measures the ground truth directly: materialize exactly one
//! View as a TABLE, run the DAG again, and record both what building that View
//! cost and what materializing it did to the DAG's makespan.

use std::collections::HashMap;
use std::sync::Arc;

use dee::connectors::Connector;
use dee::connectors::duckdb::{DuckDBConfig, DuckDBConnection};
use dee::connectors::postgres::{PostgresConfig, PostgresConnection};
use dee::dag::Dag;
use dee::executor::{ExecStats, Executor, ProfilingConfig, SimpleEngine};
use dee::file::{DagColumn, DagFile, DagFileMetadata, DagFileNode, DagFileSource};
use dee::opt::OptimizerConfig;
use dee::opt::hmp::{HMPPass, HmpCostMethod};
use dee::opt::leafset::PlanArena;
use serde_json::{Value, json};

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

/// One micro-DAG from the catalog as a `DagFile`, with `overrides` (node id ->
/// materialize mode) applied on top of what the catalog declares.
fn dag_file(
    spec: &Value,
    sources: &[String],
    dialect: &str,
    overrides: &HashMap<String, String>,
) -> DagFile {
    let nodes = spec["nodes"]
        .as_array()
        .expect("nodes")
        .iter()
        .map(|n| {
            let id = n["id"].as_str().unwrap().to_string();
            let mat = overrides
                .get(&id)
                .cloned()
                .unwrap_or_else(|| n["materialize"].as_str().unwrap_or("view").to_string());
            DagFileNode {
                id,
                query_text: n["query_text"].as_str().unwrap().to_string(),
                depends_on: n["depends_on"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|d| d.as_str().unwrap().to_string())
                    .collect(),
                materialize: Some(mat),
            }
        })
        .collect();
    DagFile {
        metadata: Some(DagFileMetadata {
            sql_dialect: Some(dialect.to_string()),
            max_parallelism: None,
        }),
        nodes,
        sources: sources
            .iter()
            .map(|s| DagFileSource {
                name: s.clone(),
                columns: Vec::<DagColumn>::new(),
            })
            .collect(),
    }
}

/// The per-View ranking one costing method reads off `stats`.
fn costs<C>(
    conn: Arc<C>,
    engine: Arc<SimpleEngine<C>>,
    dag: &Dag,
    stats: &ExecStats,
    method: HmpCostMethod,
    downstream: bool,
) -> Vec<Value>
where
    C: Connector + Send + Sync + 'static,
    SimpleEngine<C>: Executor<C, ExecutionEngine = SimpleEngine<C>> + Send + Sync,
{
    let cfg = OptimizerConfig {
        hmp_cost_method: method,
        hmp_downstream_cost: downstream,
        hmp_normalize_with_cardinality: false,
        ..Default::default()
    };
    let pass: HMPPass<C, SimpleEngine<C>> = HMPPass::from_config(conn, engine, &cfg);
    pass.ranking_for(dag, stats)
        .into_iter()
        .map(|r| {
            json!({
                "node": r.node,
                "cost_s": r.total_cpu_time_s,
                "cardinality": r.cardinality,
                "leaves": r.leaves,
                "matched": r.matched,
            })
        })
        .collect()
}

/// Every View HMP would consider a candidate at all.
fn candidates(dag: &Dag) -> Vec<String> {
    let mut out: Vec<String> = dag
        .nodes
        .nodes()
        .filter(|n| n.materialize == dee::dag::MaterializeMode::View)
        .filter(|n| dag.nodes.out_degree(&n.id) > 1 && dag.nodes.paths_to_sinks(&n.id) > 1)
        .map(|n| n.id.clone())
        .collect();
    out.sort();
    out
}

async fn run_once<C>(engine: &SimpleEngine<C>, dag: &Dag) -> ExecStats
where
    C: Connector + Send + Sync + 'static,
    SimpleEngine<C>: Executor<C, ExecutionEngine = SimpleEngine<C>>,
{
    let _ = engine.cleanup(dag).await;
    let stats = engine.run(dag).await.expect("run failed");
    let _ = engine.cleanup(dag).await;
    stats
}

async fn bench<C>(conn: Arc<C>, catalog: &Value, dialect: &str, reps: usize, out_path: &str)
where
    C: Connector + Send + Sync + 'static,
    SimpleEngine<C>: Executor<C, ExecutionEngine = SimpleEngine<C>> + Send + Sync,
{
    for stmt in catalog["setup"][dialect].as_array().expect("setup") {
        let sql = stmt.as_str().unwrap();
        conn.execute(sql.to_string())
            .await
            .unwrap_or_else(|e| panic!("setup failed: {e}\n{sql}"));
    }
    eprintln!("[{dialect}] fixtures ready");

    let sources: Vec<String> = catalog["sources"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s.as_str().unwrap().to_string())
        .collect();

    let engine = Arc::new(
        SimpleEngine::new(conn.clone())
            .expect("engine")
            .with_profiling(ProfilingConfig {
                sample_interval: std::time::Duration::from_millis(500),
                collect_plans: true,
            }),
    );

    let only = std::env::var("BENCH_ONLY").ok();
    let mut results: Vec<Value> = Vec::new();

    for spec in catalog["dags"].as_array().unwrap() {
        let name = spec["name"].as_str().unwrap();
        if let Some(o) = &only
            && !o.split(',').any(|x| x == name)
        {
            continue;
        }
        let no_override = HashMap::new();
        let base: Dag = dag_file(spec, &sources, dialect, &no_override)
            .try_into()
            .expect("dag");
        let cands = candidates(&base);
        eprintln!("[{dialect}] {name}: candidates = {cands:?}");

        // Warm-up, then `reps` measured baseline runs. Costs are read off the
        // last one; the makespans are kept for the saving comparison.
        let _ = run_once(&engine, &base).await;
        let mut base_ms: Vec<i64> = Vec::new();
        let mut leafset: Vec<Vec<Value>> = Vec::new();
        let mut signature: Vec<Vec<Value>> = Vec::new();
        // `downstream_cost` changes what the score means: "what this View costs
        // to build once" vs "what would vanish if it were built once". The
        // second is the quantity the materialization hypothesis is about, so
        // both are recorded and scored.
        let mut leafset_dup: Vec<Vec<Value>> = Vec::new();
        let mut signature_dup: Vec<Vec<Value>> = Vec::new();
        let mut learned: Vec<Vec<Value>> = Vec::new();
        let mut learned_dup: Vec<Vec<Value>> = Vec::new();
        let mut last: Option<ExecStats> = None;
        for _ in 0..reps {
            let s = run_once(&engine, &base).await;
            base_ms.push(s.duration.num_milliseconds());
            // Both methods read the same run, so any disagreement between them
            // is the method and not the measurement.
            for (sink, method, downstream) in [
                (&mut leafset, HmpCostMethod::Leafset, false),
                (&mut signature, HmpCostMethod::Signature, false),
                (&mut leafset_dup, HmpCostMethod::Leafset, true),
                (&mut signature_dup, HmpCostMethod::Signature, true),
                (&mut learned, HmpCostMethod::LearnedCost, false),
                (&mut learned_dup, HmpCostMethod::LearnedCost, true),
            ] {
                sink.push(costs(
                    conn.clone(),
                    engine.clone(),
                    &base,
                    &s,
                    method,
                    downstream,
                ));
            }
            last = Some(s);
        }
        let stats = last.unwrap();

        // Ground truth: materialize one candidate at a time.
        let mut truth: Vec<Value> = Vec::new();
        for cand in &cands {
            let mut ov = HashMap::new();
            ov.insert(cand.clone(), "table".to_string());
            let variant: Dag = dag_file(spec, &sources, dialect, &ov).try_into().expect("dag");
            let _ = run_once(&engine, &variant).await; // warm
            let mut build_ms: Vec<i64> = Vec::new();
            let mut make_ms: Vec<i64> = Vec::new();
            let mut plan_total_s: Vec<f64> = Vec::new();
            let mut plan_compute_s: Vec<f64> = Vec::new();
            let mut rows: Option<u64> = None;
            let mut root_op = String::new();
            let mut build_plan: Option<String> = None;
            for _ in 0..reps {
                let s = run_once(&engine, &variant).await;
                let ns = s.node_stats.get(cand).expect("candidate node stats");
                build_ms.push(ns.duration.num_milliseconds());
                rows = ns.rows_produced.or(rows);
                make_ms.push(s.duration.num_milliseconds());
                // The measured node duration is dominated by writing the table.
                // What both costing methods claim to estimate is the *compute*
                // of running the View once, so read that off the same plan the
                // methods read: everything strictly below the write operator.
                if let Some(plan) = &ns.plan {
                    build_plan = Some(plan.clone());
                }
                if let Some(plan) = &ns.plan
                    && let Some(parsed) = conn.parse_plan(plan)
                {
                    let arena = PlanArena::build(&parsed);
                    plan_total_s.push(arena.total_time());
                    if let Some(root) = arena.nodes.first() {
                        root_op = root.operator.clone();
                        plan_compute_s.push(
                            root.children
                                .iter()
                                .map(|c| arena.nodes[*c].subtree_time)
                                .sum::<f64>(),
                        );
                    }
                }
            }
            truth.push(json!({
                "node": cand,
                "build_ms": build_ms,
                "makespan_ms": make_ms,
                "plan_total_s": plan_total_s,
                "plan_compute_s": plan_compute_s,
                "root_op": root_op,
                "rows": rows,
                "build_plan": build_plan,
            }));
        }

        // Raw plan JSON per node, so a costing model can be prototyped offline
        // against exactly what the pass sees: a VIEW's plain EXPLAIN and a
        // TABLE's EXPLAIN ANALYZE.
        let mut baseline_plans: HashMap<String, &String> = HashMap::new();
        for (id, ns) in &stats.node_stats {
            if let Some(plan) = &ns.plan {
                baseline_plans.insert(id.clone(), plan);
            }
        }
        let materialize: HashMap<String, String> = base
            .nodes
            .nodes()
            .map(|n| (n.id.clone(), n.materialize.as_str().to_string()))
            .collect();

        let mut baseline_plan_s: HashMap<String, f64> = HashMap::new();
        for (id, ns) in &stats.node_stats {
            if let Some(plan) = &ns.plan
                && let Some(parsed) = conn.parse_plan(plan)
            {
                let t = PlanArena::build(&parsed).total_time();
                if t > 0.0 {
                    baseline_plan_s.insert(id.clone(), t);
                }
            }
        }

        let node_ms: HashMap<String, i64> = stats
            .node_stats
            .iter()
            .map(|(k, v)| (k.clone(), v.duration.num_milliseconds()))
            .collect();

        results.push(json!({
            "dag": name,
            "doc": spec["doc"],
            "backend": dialect,
            "candidates": cands,
            "baseline_makespan_ms": base_ms,
            "baseline_node_ms": node_ms,
            "baseline_plan_s": baseline_plan_s,
            "baseline_plans": baseline_plans,
            "materialize": materialize,
            "leafset": leafset,
            "signature": signature,
            "leafset_dup": leafset_dup,
            "signature_dup": signature_dup,
            "learned_cost": learned,
            "learned_cost_dup": learned_dup,
            "truth": truth,
        }));
        std::fs::write(out_path, serde_json::to_string_pretty(&results).unwrap()).unwrap();
        eprintln!("[{dialect}] {name}: done");
    }
    eprintln!("[{dialect}] wrote {out_path}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore]
async fn view_costing_microbenchmark() {
    let catalog: Value = serde_json::from_str(
        &std::fs::read_to_string(env_or("BENCH_CATALOG", "catalog.json")).unwrap(),
    )
    .unwrap();
    let reps: usize = env_or("BENCH_REPS", "3").parse().unwrap();
    let out = env_or("BENCH_OUT", "view_costing.json");

    match env_or("BENCH_BACKEND", "duckdb").as_str() {
        "duckdb" => {
            let cfg = DuckDBConfig::new_from_path(env_or("BENCH_DUCKDB", "bench_view_cost.duckdb"))
                .with_threads(env_or("BENCH_THREADS", "8").parse().unwrap())
                .with_max_memory("8GB".to_string());
            let conn = DuckDBConnection::new(cfg).await.expect("duckdb");
            bench(conn, &catalog, "duckdb", reps, &out).await;
        }
        "postgres" => {
            let cfg: PostgresConfig = serde_json::from_value(json!({
                "host": env_or("BENCH_PG_HOST", "0.0.0.0"),
                "port": env_or("BENCH_PG_PORT", "5432").parse::<i32>().unwrap(),
                "user": env_or("BENCH_PG_USER", "runner"),
                "password": env_or("BENCH_PG_PASSWORD", "password"),
                "database": env_or("BENCH_PG_DB", "benchmark"),
                "num_connections": 16,
            }))
            .unwrap();
            let conn = PostgresConnection::new(cfg).await.expect("postgres");
            bench(conn, &catalog, "postgres", reps, &out).await;
        }
        other => panic!("unknown backend {other}"),
    }
}
