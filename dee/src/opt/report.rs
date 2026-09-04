//! Typed, serializable reports describing what the optimizer did.
//!
//! Every [`OptimizerPass`](crate::opt::OptimizerPass) returns a
//! [`PassOutcome`]; [`Optimizer::run`](crate::opt::Optimizer::run) stamps each
//! one with ordering and timing to produce a [`PassReport`], and collects them
//! into a single [`OptimizeReport`].
//!
//! These types replace the previous stringly-typed `HashMap<String, String>`
//! stats. The server stores an `OptimizeReport` verbatim and serves it from
//! `GET /v1/optimizations/{id}/report`, which the benchmarking harness consumes
//! directly; every field here is therefore part of a stable machine-readable
//! contract — see `benchmark/src/dee_bench/schema.py`.

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

/// ParallelismTuning-specific fields of a [`PassReport`].
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ParallelismDetail {
    /// The setting the DAG arrived with. `None` is unlimited.
    pub baseline_parallelism: Option<usize>,
    /// The setting the search settled on. Equal to `baseline_parallelism`
    /// when nothing beat it.
    pub chosen_parallelism: Option<usize>,
    /// Worst baseline sample -- the runtime the DAG was known to be capable of
    /// before the search, rather than its luckiest draw.
    pub baseline_runtime_ms: f64,
    /// Worst sample of the chosen setting, on the same principle.
    pub best_runtime_ms: f64,
    /// Fractional change from baseline to best; negative is an improvement.
    pub opt_change: f64,
    pub seed_repeats: usize,
    pub confirm_runs: usize,
    /// The configured ladder, before pruning against the DAG.
    pub ladder: Vec<usize>,
    /// Whether each rung was judged against a control measured beside it.
    #[serde(default)]
    pub paired: bool,
    /// Cores one node occupied on its own, from the probe rung: CPU-seconds
    /// divided by wall seconds. `None` when the engine reported no CPU.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub probe_cores: Option<f64>,
    /// What the probe concluded: `"saturated"` (one node already fills the
    /// machine, search upward from the narrowest rung) or `"idle capacity"`
    /// (one node leaves cores unused, try the widest rungs first).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search_direction: Option<String>,
    /// Every setting the search resolved, baseline first, in the order tried.
    pub rungs: Vec<RungResult>,
}

/// One setting the parallelism ladder measured, and what was decided about it.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RungResult {
    /// `None` is unlimited.
    pub parallelism: Option<usize>,
    /// The incumbent measurements taken next to this rung, one per pair.
    /// Empty when the search ran unpaired.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub control_samples: Vec<f64>,
    /// Per-pair `rung / control` wall-time ratios. Below 1 is an improvement,
    /// and being a ratio it is unaffected by drift the pair shared.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pair_ratios: Vec<f64>,
    /// `rung / control` CPU-seconds for the same work. Below 1 means the rung
    /// relieved contention; around 1 means it changed only how the work was
    /// scheduled. `None` when the engine reported no CPU samples.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_ratio: Option<f64>,
    /// Every runtime measured at this setting, in the order measured. Empty
    /// when the trial produced no usable time.
    pub samples: Vec<f64>,
    /// `"baseline"`, `"accepted"`, `"rejected (screen)"`,
    /// `"rejected (rank test)"`, or `"rejected (censored)"`.
    pub verdict: String,
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
    Parallelism(ParallelismDetail),
    /// A step that advanced an optimization's state without reaching a
    /// conclusion worth describing -- a continuous pass recording a
    /// measurement, say. Pass-specific detail is reported when the search
    /// finishes, not on every run it observes.
    None,
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

impl PassOutcome {
    /// An outcome that reports no work and no pass-specific detail.
    ///
    /// The starting point for a step that only advanced its own state; a pass
    /// fills in the counters it actually moved.
    pub fn empty() -> Self {
        Self {
            dag_runs_used: 0,
            changes_applied: 0,
            candidates_considered: 0,
            working_set_size: 0,
            iterations: Vec::new(),
            detail: PassDetail::None,
        }
    }

    pub fn with_detail(mut self, detail: PassDetail) -> Self {
        self.detail = detail;
        self
    }

    pub fn with_iterations(mut self, iterations: Vec<IterationStat>) -> Self {
        self.iterations = iterations;
        self
    }

    pub fn with_changes(mut self, changes_applied: u32) -> Self {
        self.changes_applied = changes_applied;
        self
    }

    pub fn with_dag_runs(mut self, dag_runs_used: u32) -> Self {
        self.dag_runs_used = dag_runs_used;
        self
    }

    pub fn with_candidates(mut self, considered: u32, working_set_size: u32) -> Self {
        self.candidates_considered = considered;
        self.working_set_size = working_set_size;
        self
    }
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
