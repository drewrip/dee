use axum::Json;
use axum::extract::State;
use chrono::{DateTime, Utc};
use serde::Serialize;

use crate::error::ServerError;
use crate::state::{AppState, VERSION};
use crate::store;

#[derive(Serialize)]
pub struct Health {
    status: &'static str,
}

/// Deliberately does not touch the database: this is a liveness probe, and a
/// client waiting for the server to come up should not be blocked behind a
/// slow first query.
pub async fn healthz() -> Json<Health> {
    Json(Health { status: "ok" })
}

#[derive(Serialize)]
pub struct Info {
    version: &'static str,
    instance_id: String,
    started_at: DateTime<Utc>,
    bind: String,
    metadata_db: String,
    schema_version: i32,
    schema_applied_at: Option<DateTime<Utc>>,
    max_concurrent_runs: usize,
    tick_interval_ms: u64,
}

pub async fn info(State(state): State<AppState>) -> Result<Json<Info>, ServerError> {
    let schema_applied_at = store::last_migrated_at(&state.store).await?;
    Ok(Json(Info {
        version: VERSION,
        instance_id: state.instance_id.clone(),
        started_at: state.started_at,
        bind: state.config.bind.to_string(),
        metadata_db: state.config.metadata_db.display().to_string(),
        schema_version: store::schema::latest_version(),
        schema_applied_at,
        max_concurrent_runs: state.config.max_concurrent_runs,
        tick_interval_ms: state.config.tick_interval.as_millis() as u64,
    }))
}

#[derive(Serialize)]
pub struct OptimizerOption {
    /// Field name in the `config` object of an optimize request.
    name: &'static str,
    /// The equivalent `dee optimize` flag, for documentation.
    flag: &'static str,
    kind: &'static str,
    /// Passes that actually read this option. The benchmark harness prunes its
    /// experiment matrix by this, so sweeping an option does not multiply
    /// cells for variants that ignore it.
    passes: &'static [&'static str],
    choices: Option<&'static [&'static str]>,
    default: serde_json::Value,
    doc: &'static str,
}

/// A machine-readable description of every optimizer option.
///
/// This exists so a client never has to parse `--help` to discover the option
/// set. `dee-bench`'s `doctor` checks its own table against this, which turns a
/// text-scraping heuristic into a real contract.
pub async fn optimizer_options() -> Json<Vec<OptimizerOption>> {
    use serde_json::json;
    let d = dee::opt::OptimizerConfig::default();

    Json(vec![
        OptimizerOption { name: "run_hmp_pass", flag: "--enable hmp", kind: "bool",
            passes: &["hmp"], choices: None, default: json!(d.run_hmp_pass),
            doc: "Run the heuristic materialization pass." },
        OptimizerOption { name: "run_omp_pass", flag: "--enable omp", kind: "bool",
            passes: &["omp"], choices: None, default: json!(d.run_omp_pass),
            doc: "Run the centrality-based materialization pass." },
        OptimizerOption { name: "run_pushdown_pass", flag: "--enable pushdown", kind: "bool",
            passes: &["pushdown"], choices: None, default: json!(d.run_pushdown_pass),
            doc: "Run the filter and projection pushdown rewrite." },
        OptimizerOption { name: "omp_top", flag: "--omp-top", kind: "int",
            passes: &["omp"], choices: None, default: json!(d.omp_top),
            doc: "Consider only the top N candidate nodes in OMP." },
        OptimizerOption { name: "omp_centrality", flag: "--omp-node-centrality", kind: "str",
            passes: &["omp"], choices: Some(&["outdegree", "paths"]),
            default: json!(d.omp_centrality), doc: "How OMP ranks candidate nodes." },
        OptimizerOption { name: "omp_early_termination", flag: "--omp-exhaust", kind: "bool",
            passes: &["omp"], choices: None, default: json!(d.omp_early_termination),
            doc: "Stop OMP at the first candidate that does not improve. The CLI flag is the negation." },
        OptimizerOption { name: "omp_use_pushdown", flag: "--omp-no-pushdown", kind: "bool",
            passes: &["omp"], choices: None, default: json!(d.omp_use_pushdown),
            doc: "Run pushdown before evaluating each OMP candidate. The CLI flag is the negation." },
        OptimizerOption { name: "hmp_downstream_cost", flag: "--hmp-downstream-cost", kind: "bool",
            passes: &["hmp"], choices: None, default: json!(d.hmp_downstream_cost),
            doc: "Rank HMP candidates by the duplicate downstream work they cause." },
        OptimizerOption { name: "hmp_max_runs", flag: "--hmp-max-runs", kind: "int",
            passes: &["hmp"], choices: None, default: json!(d.hmp_max_runs),
            doc: "Budget of DAG executions HMP may spend searching." },
        OptimizerOption { name: "hmp_top_cpu_time", flag: "--hmp-top-cpu-time", kind: "float",
            passes: &["hmp"], choices: None, default: json!(d.hmp_top_cpu_time),
            doc: "Fraction of total cost the candidate prefix must cover." },
        OptimizerOption { name: "hmp_normalize_with_cardinality",
            flag: "--hmp-normalize-with-cardinality", kind: "bool", passes: &["hmp"],
            choices: None, default: json!(d.hmp_normalize_with_cardinality),
            doc: "Divide candidate cost by estimated cardinality." },
        OptimizerOption { name: "hmp_strategy", flag: "--hmp-strategy", kind: "str",
            passes: &["hmp"], choices: Some(&["breadth", "greedy"]),
            default: json!(d.hmp_strategy), doc: "HMP's search strategy over the candidate ranking." },
        OptimizerOption { name: "hmp_beam_width", flag: "--hmp-beam-width", kind: "int",
            passes: &["hmp"], choices: None, default: json!(d.hmp_beam_width),
            doc: "Beam width for the greedy HMP strategy. Ignored by breadth." },
        OptimizerOption { name: "hmp_use_pushdown", flag: "--hmp-no-pushdown", kind: "bool",
            passes: &["hmp"], choices: None, default: json!(d.hmp_use_pushdown),
            doc: "Run pushdown before evaluating each HMP candidate. The CLI flag is the negation." },
        OptimizerOption { name: "profile_iterations", flag: "--profile-iterations", kind: "bool",
            passes: &["hmp", "omp"], choices: None, default: json!(d.profile_iterations),
            doc: "Capture a resource timeseries for every candidate run." },
    ])
}
