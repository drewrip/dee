//! Typed, serializable reports describing what the optimizer did.
//!
//! Every [`OptimizerPass`](crate::opt::OptimizerPass) returns a
//! [`PassOutcome`]; [`Optimizer::run`](crate::opt::Optimizer::run) stamps each
//! one with ordering and timing to produce a [`PassReport`], and collects them
//! into a single [`OptimizeReport`].
//!
//! These types replace the previous stringly-typed `HashMap<String, String>`
//! stats. The benchmarking harness consumes them directly as JSON via
//! `dee-cli opt --report-json`, so every field here is part of a stable
//! machine-readable contract — see `benchmark/src/dee_bench/schema.py`.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::profile::SystemUsageSample;

/// Runtime of a single candidate DAG the pass ran while searching, in the
/// order it was tried. For passes that measure a baseline first, iteration 1
/// is that baseline.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct IterationStat {
    pub iteration: usize,
    /// Measured wall time of this candidate. For a cancelled trial this is
    /// only a lower bound (the cancellation budget), since the run was killed
    /// before finishing.
    pub runtime_ms: i64,
    /// Materialization combo tried at this iteration; empty for a baseline
    /// and for passes that don't vary materializations.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub combo: Vec<String>,
    /// `"ok"`, `"cancelled"`, `"skipped"`, or `"baseline"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome: Option<String>,
    /// CPU/memory/disk timeseries sampled during this iteration's run. Only
    /// populated when `profile_iterations` is enabled.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub system_samples: Vec<SystemUsageSample>,
}

impl IterationStat {
    pub fn new(iteration: usize, runtime_ms: i64) -> Self {
        Self {
            iteration,
            runtime_ms,
            ..Default::default()
        }
    }

    pub fn with_combo(mut self, combo: Vec<String>) -> Self {
        self.combo = combo;
        self
    }

    pub fn with_outcome(mut self, outcome: impl Into<String>) -> Self {
        self.outcome = Some(outcome.into());
        self
    }

    pub fn with_samples(mut self, samples: Vec<SystemUsageSample>) -> Self {
        self.system_samples = samples;
        self
    }
}

/// HMP-specific fields of a [`PassReport`].
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct HmpDetail {
    /// Wall time of the unmodified DAG, measured before any search.
    pub baseline_runtime_ms: i64,
    /// Wall time of the best candidate found.
    pub final_runtime_ms: i64,
    /// The `--hmp-max-runs` budget this pass was given.
    pub max_runs: usize,
    pub top_cpu_time: f64,
    /// `"Breadth"` or `"Greedy"`.
    pub strategy: String,
    pub beam_width: usize,
    pub normalize_with_cardinality: bool,
    pub downstream_cost: bool,
    pub use_pushdown: bool,
    /// The Views this pass converted into materialized TempTables.
    pub new_materializations: Vec<String>,
    /// The ranked candidate Views the search actually explored.
    pub working_set: Vec<String>,
}

/// OMP-specific fields of a [`PassReport`].
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct OmpDetail {
    pub baseline_value: f64,
    pub best_value: f64,
    /// Fractional change from baseline to best; negative is an improvement.
    pub opt_change: f64,
    /// Nodes materialized as TempTables in the winning plan.
    pub best_plan: Vec<String>,
    /// `"OutDegree"` or `"Paths"`.
    pub centrality: String,
    /// Candidate nodes in ranked order, with their centrality score.
    pub candidates_ranked: Vec<CandidateScore>,
    pub early_termination: bool,
    pub use_pushdown: bool,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CandidateScore {
    pub node_id: String,
    pub score: f64,
}

/// Pushdown-specific fields of a [`PassReport`].
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PushdownDetail {
    pub temp_tables_count: usize,
    pub rewrites_applied: usize,
    /// One entry per TempTable considered, deepest-first.
    pub outcomes: Vec<PushdownOutcome>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PushdownOutcome {
    pub node_id: String,
    pub outcome: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PassDetail {
    Hmp(HmpDetail),
    Omp(OmpDetail),
    Pushdown(PushdownDetail),
}

/// What a pass reports about its own run. The [`Optimizer`](crate::opt::Optimizer)
/// stamps this with ordering and timing to build a [`PassReport`].
#[derive(Clone, Debug)]
pub struct PassOutcome {
    /// How many full DAG executions this pass spent.
    pub dag_runs_used: u32,
    /// The single comparable "how many changes did this pass make" number.
    pub changes_applied: u32,
    /// How many candidates the pass evaluated.
    pub candidates_considered: u32,
    /// How many candidates were in the pass's working set to begin with.
    pub working_set_size: u32,
    pub iterations: Vec<IterationStat>,
    pub detail: PassDetail,
}

/// A [`PassOutcome`] plus the ordering and timing the optimizer observed.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PassReport {
    /// `"HMPPass"`, `"OMPPass"` or `"PushdownPass"`.
    pub pass: String,
    /// 0-based position in the pass pipeline, as actually executed.
    pub order: u32,
    pub started_at: DateTime<Utc>,
    pub finished_at: DateTime<Utc>,
    pub wall_ms: i64,
    pub dag_runs_used: u32,
    pub changes_applied: u32,
    pub candidates_considered: u32,
    pub working_set_size: u32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub iterations: Vec<IterationStat>,
    pub detail: PassDetail,
}

impl PassReport {
    pub fn from_outcome(
        pass: impl Into<String>,
        order: u32,
        started_at: DateTime<Utc>,
        finished_at: DateTime<Utc>,
        outcome: PassOutcome,
    ) -> Self {
        Self {
            pass: pass.into(),
            order,
            started_at,
            finished_at,
            wall_ms: (finished_at - started_at).num_milliseconds(),
            dag_runs_used: outcome.dag_runs_used,
            changes_applied: outcome.changes_applied,
            candidates_considered: outcome.candidates_considered,
            working_set_size: outcome.working_set_size,
            iterations: outcome.iterations,
            detail: outcome.detail,
        }
    }
}

/// The complete record of one `Optimizer::run`.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct OptimizeReport {
    pub started_at: DateTime<Utc>,
    pub finished_at: DateTime<Utc>,
    /// Wall time of the whole optimization, including every candidate DAG run
    /// the passes performed. This is the numerator of the payback analysis.
    pub wall_ms: i64,
    /// Runtime of the unoptimized DAG, if any pass measured one.
    pub baseline_runtime_ms: Option<i64>,
    /// Runtime of the optimized DAG as last measured during the search.
    pub final_runtime_ms: Option<i64>,
    /// Total DAG executions spent across all passes.
    pub dag_runs_used: u32,
    pub nodes_before: u32,
    pub nodes_after: u32,
    pub passes: Vec<PassReport>,
}

impl OptimizeReport {
    /// Total changes applied across every pass.
    pub fn total_changes_applied(&self) -> u32 {
        self.passes.iter().map(|p| p.changes_applied).sum()
    }

    /// The report for a named pass, if it ran.
    pub fn pass(&self, name: &str) -> Option<&PassReport> {
        self.passes.iter().find(|p| p.pass == name)
    }
}
