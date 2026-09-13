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
        OptimizerOption { name: "run_nodefusion_pass", flag: "--enable nodefusion", kind: "bool",
            passes: &["nodefusion"], choices: None, default: json!(d.run_nodefusion_pass),
            doc: "Fuse every Table node into one query. A pure rewrite." },
        OptimizerOption { name: "nodefusion_materialize_ctes", flag: "--nodefusion-materialize-ctes", kind: "bool",
            passes: &["nodefusion"], choices: None, default: json!(d.nodefusion_materialize_ctes),
            doc: "Emit the View CTEs NodeFusion inlines as materialized CTEs. An inlined Table's CTE is materialized on its own account and is not affected." },
        OptimizerOption { name: "nodefusion_naive_materialize_ctes", flag: "--nodefusion-naive-materialize-ctes", kind: "bool",
            passes: &["nodefusion"], choices: None, default: json!(d.nodefusion_naive_materialize_ctes),
            doc: "Materialize an inlined View CTE that more than one Table node reads. Naive: it counts readers rather than pricing them. Subsumed by nodefusion_materialize_ctes." },
        OptimizerOption { name: "nodefusion_materialize_ctes_override", flag: "--nodefusion-materialize-ctes-override", kind: "str_list",
            passes: &["nodefusion"], choices: None, default: json!(d.nodefusion_materialize_ctes_override),
            doc: "The exact set of node IDs whose CTEs NodeFusion materializes, overriding every default -- so it is also how an inlined Table's CTE is made plain." },
        OptimizerOption { name: "run_parallelism_pass", flag: "--enable parallelism", kind: "bool",
            passes: &["parallelism"], choices: None, default: json!(d.run_parallelism_pass),
            doc: "Run the node-concurrency ladder." },
        OptimizerOption { name: "parallelism_ladder", flag: "--parallelism-ladder", kind: "int_list",
            passes: &["parallelism"], choices: None, default: json!(d.parallelism_ladder),
            doc: "Node-concurrency caps to measure. Rungs that cannot bind on the DAG are dropped." },
        OptimizerOption { name: "parallelism_seed_repeats", flag: "--parallelism-seed-repeats", kind: "int",
            passes: &["parallelism"], choices: None, default: json!(d.parallelism_seed_repeats),
            doc: "Runs spent measuring the DAG's current setting before the ladder starts." },
        OptimizerOption { name: "parallelism_confirm_runs", flag: "--parallelism-confirm-runs", kind: "int",
            passes: &["parallelism"], choices: None, default: json!(d.parallelism_confirm_runs),
            doc: "Re-measurements a rung must survive after beating the incumbent's best sample." },
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
        OptimizerOption { name: "hmp_cost_method", flag: "--hmp-cost-method", kind: "str",
            passes: &["hmp"],
            choices: Some(&["leafset", "signature", "node_time", "learned_cost",
                            "dup_attribution"]),
            default: json!(d.hmp_cost_method),
            doc: "How a View's cost is read off a run's plans. `leafset` matches a View \
                  against the region of a consumer's plan whose scanned base relations are \
                  contained in the View's own; `signature` matches operators between plans \
                  by name and estimated cardinality; `learned_cost` prices the View's own \
                  plan with per-operator seconds-per-byte constants fitted to the EXPLAIN \
                  ANALYZE plans of every CREATE TABLE run so far; `dup_attribution` inlines \
                  the View into each consumer as a materialized CTE, EXPLAINs them, and \
                  charges it every copy of itself but one." },
        OptimizerOption { name: "hmp_dup_cost_model", flag: "--hmp-dup-cost-model", kind: "str",
            passes: &["hmp"],
            choices: Some(&["learned_cost", "cardinality", "operators"]),
            default: json!(d.hmp_dup_cost_model),
            doc: "What the `dup_attribution` cost method prices a plan region with. \
                  `learned_cost` gives seconds, from constants fitted to executed plans; \
                  `cardinality` sums the region's estimated output rows; `operators` counts \
                  its operators. Ignored by every other cost method." },
        OptimizerOption { name: "hmp_objective", flag: "--hmp-objective", kind: "str",
            passes: &["hmp"],
            choices: Some(&["makespan", "query_time"]),
            default: json!(d.hmp_objective),
            doc: "Which measure the HMP search minimizes. `makespan` orders candidates by \
                  predicted wall clock and promotes a trial that beats the incumbent's \
                  runtime; `query_time` orders them by duplicate computation removed and \
                  promotes a trial that beats the incumbent's node time. Materializing a View \
                  cuts the sum and usually lengthens the critical path, so the two disagree." },
        OptimizerOption { name: "hmp_search_budget", flag: "--hmp-search-budget", kind: "int",
            passes: &["hmp"], choices: None, default: json!(d.hmp_search_budget),
            doc: "Candidate combinations HMP prices before it spends any DAG run on them. \
                  Bounds costing, which is EXPLAIN-only; `hmp_max_runs` bounds executions." },
        OptimizerOption { name: "hmp_use_pushdown", flag: "--hmp-no-pushdown", kind: "bool",
            passes: &["hmp"], choices: None, default: json!(d.hmp_use_pushdown),
            doc: "Run pushdown before evaluating each HMP candidate. The CLI flag is the negation." },
        OptimizerOption { name: "trial_resume", flag: "--trial-resume", kind: "bool",
            passes: &["hmp", "omp", "parallelism"], choices: None, default: json!(d.trial_resume),
            doc: "Cancel a candidate that overruns the best configuration found so far and \
                  finish the run under that configuration, rebuilding only what the cancelled \
                  candidate never got to. Off measures every candidate to completion." },
        OptimizerOption { name: "trial_budget_eps", flag: "--trial-budget-eps", kind: "float",
            passes: &["hmp", "omp", "parallelism"], choices: None,
            default: json!(d.trial_budget_eps),
            doc: "Fraction by which a candidate may overrun the best configuration before it \
                  is cut short. Zero (the default) stops a candidate the moment it can no \
                  longer be faster than the incumbent." },
        OptimizerOption { name: "trial_reuse", flag: "--trial-reuse", kind: "str",
            passes: &["hmp", "omp", "parallelism"], choices: Some(&["equivalent", "strict"]),
            default: json!(d.trial_reuse),
            doc: "How much of a cancelled candidate the resume keeps. `equivalent` relies on \
                  every DAG dee produces holding the same tuples as the one it came from, so a \
                  finished relation is reusable whenever the incumbent has a node of that name, \
                  and the candidate's landing pads are read rather than recomputed. `strict` \
                  keeps only identically-defined nodes." },
        OptimizerOption { name: "profile_iterations", flag: "--profile-iterations", kind: "bool",
            passes: &["hmp", "omp", "parallelism"], choices: None, default: json!(d.profile_iterations),
            doc: "Capture a resource timeseries for every candidate run." },
    ])
}
