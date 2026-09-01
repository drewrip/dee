pub mod common;
pub mod explain;
pub mod hmp;
pub mod omp;
pub mod pushdown;
pub mod registry;
pub mod report;
pub mod step;
pub mod store;

use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use log::{debug, error, warn};
use serde::{Deserialize, Serialize};

use thiserror::Error;

use crate::{
    connectors::Connector,
    dag::Dag,
    executor::Executor,
    opt::{hmp::HMPStrategy, omp::OMPCentrality},
};

pub use crate::opt::explain::render_explain_html;
pub use crate::opt::report::{
    CandidateScore, HmpDetail, IterationStat, OmpDetail, OptimizeReport, PassDetail, PassOutcome,
    PassReport, PushdownDetail, PushdownOutcome,
};
pub use crate::opt::step::{
    OptimizationType, RegisterContext, RunContext, StepContext, StepOutcome, StepPhase, run_phase,
};
pub use crate::opt::store::{OptStore, OptStoreError, Registration};

#[derive(Error, Debug)]
pub enum OptimizerError {
    #[error("couldn't execute DAG - {0}")]
    Exec(String),
    #[error("this pass isn't implemented yet, skipping - {0}")]
    NotImplemented(String),
    #[error("optimization state: {0}")]
    Store(#[from] OptStoreError),
    #[error("unknown optimization '{0}'")]
    Unknown(String),
}

/// One optimization dee can apply to a DAG.
///
/// The interface is the same whether an optimization improves the DAG over its
/// whole lifetime or rewrites it once, because the server needs to treat them
/// the same way: register it on a DAG, step it, deregister it. What differs is
/// [`optimization_type`](Optimization::optimization_type), which tells the
/// server *when* to step, and what the optimization keeps in the metadata
/// database, which is entirely its own business.
///
/// The critical property is that an optimization does not run the DAG. The
/// server runs it -- on a schedule, from a trigger, out of the queue -- and an
/// optimization gets a turn on either side of each execution. That is what
/// lets a search amortize across runs the DAG was going to perform anyway,
/// instead of buying its own.
///
/// Generic over `C`/`E` rather than taking them as trait objects: neither
/// appears in a method signature, so this stays object-safe, and the server
/// already dispatches on its concrete connector type.
#[async_trait]
pub trait Optimization<C, E>: Send
where
    C: Connector + Send + Sync + 'static,
    E: Executor<C> + Send + Sync,
{
    /// Stable identifier: `"hmp"`, `"omp"`, `"pushdown"`. Names the
    /// optimization on the wire, in the registry, and in its table prefix.
    fn name(&self) -> &'static str;

    fn optimization_type(&self) -> OptimizationType;

    /// Which side of an execution [`step`](Optimization::step) is called on.
    fn step_phase(&self) -> StepPhase;

    /// Change it. The author's choice is the default; a registration may
    /// override it.
    fn set_step_phase(&mut self, phase: StepPhase);

    /// Create whatever metadata-database tables this optimization keeps its
    /// state in, and record that it is now registered on this DAG.
    ///
    /// `Ok(None)` means the optimization keeps no persisted state -- Pushdown
    /// decides everything from the DAG in front of it -- which is different
    /// from creating no tables by accident.
    ///
    /// Must be idempotent: registering an already-registered optimization is
    /// how a server restart re-establishes one.
    async fn register(
        &self,
        ctx: &RegisterContext<'_>,
    ) -> Result<Option<Registration>, OptimizerError>;

    /// Drop what `register` created, leaving the metadata database as it was.
    /// `Ok(None)` when there was nothing to drop.
    async fn deregister(
        &self,
        ctx: &RegisterContext<'_>,
    ) -> Result<Option<Registration>, OptimizerError>;

    /// Do one unit of work.
    ///
    /// For a `Continuous` optimization this is called around every execution
    /// of the DAG, on the sides named by `step_phase`. For a `Once`
    /// optimization it is called a single time, with no run attached.
    ///
    /// There is no "which run is this" parameter beyond
    /// [`StepContext::run`], and deliberately so: what to do on this run is
    /// decided by reading the state written on previous ones, so the
    /// optimization stays the only thing that understands its own search.
    async fn step(
        &mut self,
        ctx: &mut StepContext<'_, C, E>,
    ) -> Result<StepOutcome, OptimizerError>;

    /// `(tab label, HTML snippet)` explaining what the optimization did and
    /// why, from data it retained while stepping. `None` when it has nothing
    /// to show yet.
    fn explain(&self) -> Option<(String, String)> {
        None
    }
}

/// Hands out an [`OptStore`] scoped to one optimization's table namespace.
///
/// The library never opens the metadata database itself; the server owns it
/// and implements this. Scoping happens here rather than at each call so an
/// optimization cannot ask for a handle wider than its own.
pub trait OptStoreFactory: Send + Sync {
    fn store_for(&self, optimization: &str) -> Arc<dyn OptStore>;
}

/// An [`OptStoreFactory`] whose stores discard everything.
///
/// For callers with no metadata database -- tests, and `dee convert`-style
/// local tooling. A continuous optimization against this makes no progress
/// across calls, which is the honest behaviour when nothing is remembered.
pub struct NullStoreFactory;

impl OptStoreFactory for NullStoreFactory {
    fn store_for(&self, optimization: &str) -> Arc<dyn OptStore> {
        Arc::new(store::NullStore::new(optimization))
    }
}

/// Drives optimizations in batch: register, step to convergence, deregister.
///
/// It holds no per-pass settings of its own any more -- an optimization is
/// built from the [`OptimizerConfig`] by the registry, and everything about
/// how it searches lives in the optimization. What is left here is the loop
/// that supplies executions and the assembly of the [`OptimizeReport`].
#[derive(Debug, Clone)]
pub struct Optimizer<C, E>
where
    C: Connector + Send,
    E: Executor<C>,
{
    conn: Arc<C>,
    engine: Arc<E>,
    config: OptimizerConfig,
    /// Ceiling on executions this driver will perform for one continuous
    /// optimization before giving up on it converging. A search bounds itself
    /// -- `hmp_max_runs`, OMP's plan count -- so reaching this means a bug in
    /// the search, not a legitimately long one; it exists so a batch
    /// `dee optimize` cannot become an unbounded warehouse workload.
    max_batch_iterations: usize,
    /// Result stats
    stats_on_passes: bool,
    /// Collect per-pass `Explain` sections while stepping
    explain_enabled: bool,
    /// `(pass label, explain HTML)` pairs collected during the last `run()`,
    /// in pass execution order. Only populated when `explain_enabled`.
    explain_sections: Vec<(String, String)>,
}

impl<C, E> Optimizer<C, E>
where
    C: Connector + Send + 'static + Sync,
    E: Executor<C> + Send + Sync + 'static,
{
    pub fn new(conn: Arc<C>, engine: Arc<E>) -> Self {
        let config = OptimizerConfig::default();
        Self::new_with_config(conn, engine, config)
    }

    pub fn new_with_config(conn: Arc<C>, engine: Arc<E>, config: OptimizerConfig) -> Self {
        Self {
            conn,
            engine,
            explain_enabled: config.explain,
            config,
            max_batch_iterations: DEFAULT_MAX_BATCH_ITERATIONS,
            stats_on_passes: false,
            explain_sections: Vec::new(),
        }
    }

    pub fn with_max_batch_iterations(mut self, iterations: usize) -> Self {
        self.max_batch_iterations = iterations.max(1);
        self
    }

    pub fn stats_on_passes(mut self, collect_stats: bool) -> Self {
        self.stats_on_passes = collect_stats;
        self
    }

    /// `(pass label, explain HTML)` pairs collected during the last `run()`,
    /// in pass execution order. Empty unless `OptimizerConfig::with_explain`
    /// was enabled.
    pub fn explain_sections(&self) -> &[(String, String)] {
        &self.explain_sections
    }

    /// Drive every enabled optimization to convergence, now.
    ///
    /// This is the batch face of the step interface -- `dee optimize`, and the
    /// benchmark harness. A `Once` optimization is stepped once; a
    /// `Continuous` one is stepped in a loop with this driver supplying the
    /// executions its search would otherwise have to wait for the schedule to
    /// provide. Same trait, same state, same results; the only difference from
    /// running under the server's driver is who performs the runs and how
    /// long they take to arrive.
    ///
    /// Batch registration is transient: each optimization is registered, run
    /// to convergence, and deregistered, so a one-shot `dee optimize` leaves
    /// no continuous optimization attached to the DAG behind it.
    pub async fn run(
        &mut self,
        dag: &mut Dag,
        dag_id: &str,
        dag_name: &str,
        dag_version: i32,
        stores: &dyn OptStoreFactory,
    ) -> Result<OptimizeReport, OptimizerError> {
        let started_at = Utc::now();
        let nodes_before = dag.nodes.num_nodes() as u32;
        let mut passes: Vec<PassReport> = Vec::new();
        let mut order: u32 = 0;

        if let Err(e) = self.engine.resolve_schemas(dag).await {
            error!("couldn't resolve_schemas: {e}")
        }

        for name in self.config.enabled_passes() {
            let Some(mut optimization) =
                registry::build::<C, E>(name, self.conn.clone(), self.engine.clone(), &self.config)
            else {
                warn!("no optimization named '{name}'; skipping");
                continue;
            };

            let store = stores.store_for(name);
            let register_ctx = RegisterContext {
                store: store.as_ref(),
                dag_id,
                dag_name,
            };
            optimization.register(&register_ctx).await?;

            let pass_started = Utc::now();
            let outcome = self
                .converge(
                    optimization.as_mut(),
                    store.as_ref(),
                    dag,
                    dag_id,
                    dag_name,
                    dag_version,
                )
                .await;
            let pass_finished = Utc::now();

            // Deregister whatever happened, so a failed optimization does not
            // leave half-built state that the next one would read as progress.
            if let Err(e) = optimization.deregister(&register_ctx).await {
                warn!("could not deregister '{name}': {e}");
            }

            let outcome = outcome?;
            if self.explain_enabled {
                if let Some(section) = optimization.explain() {
                    self.explain_sections.push(section);
                }
            }
            passes.push(PassReport::from_outcome(
                pass_label(name),
                order,
                pass_started,
                pass_finished,
                outcome,
            ));
            order += 1;
        }

        let finished_at = Utc::now();

        // Baseline/final runtimes come from whichever pass measured them.
        // HMP measures both explicitly; OMP's costs are the equivalent.
        let mut baseline_runtime_ms = None;
        let mut final_runtime_ms = None;
        for pass in &passes {
            match &pass.detail {
                PassDetail::Hmp(d) => {
                    baseline_runtime_ms.get_or_insert(d.baseline_runtime_ms);
                    final_runtime_ms = Some(d.final_runtime_ms);
                }
                PassDetail::Omp(d) => {
                    baseline_runtime_ms.get_or_insert(d.baseline_value.round() as i64);
                    final_runtime_ms = Some(d.best_value.round() as i64);
                }
                PassDetail::Pushdown(_) | PassDetail::None => {}
            }
        }

        Ok(OptimizeReport {
            started_at,
            finished_at,
            wall_ms: (finished_at - started_at).num_milliseconds(),
            baseline_runtime_ms,
            final_runtime_ms,
            dag_runs_used: passes.iter().map(|p| p.dag_runs_used).sum(),
            nodes_before,
            nodes_after: dag.nodes.num_nodes() as u32,
            passes,
        })
    }

    /// Step one optimization until it finishes, performing the executions a
    /// continuous search asks for.
    ///
    /// `dag` is the committed definition and is only replaced when the
    /// optimization promotes a candidate. Each iteration works on a clone, so
    /// a rejected trial leaves nothing behind -- the same reason the server's
    /// driver never persists a trial.
    async fn converge(
        &self,
        optimization: &mut dyn Optimization<C, E>,
        store: &dyn OptStore,
        dag: &mut Dag,
        dag_id: &str,
        dag_name: &str,
        dag_version: i32,
    ) -> Result<PassOutcome, OptimizerError> {
        let phase = optimization.step_phase();
        let mut working = dag.clone();
        let mut last_run: Option<Dag> = None;
        let mut dag_runs_used: u32 = 0;
        let mut record = PassOutcome::empty();

        if optimization.optimization_type() == OptimizationType::Once {
            let mut ctx = StepContext {
                store,
                conn: self.conn.clone(),
                engine: self.engine.clone(),
                dag: &mut working,
                dag_id,
                dag_name,
                dag_version,
                side: StepPhase::Before,
                run: None,
            };
            let outcome = optimization.step(&mut ctx).await?;
            if let Some(r) = outcome.record() {
                record = r.clone();
            }
            if outcome.persists() {
                *dag = working;
            }
            return Ok(record);
        }

        // A continuous search with no measurements yet needs a first run to
        // form an opinion, so the loop always executes -- an `Idle` before-step
        // means "run what we have", not "there is nothing to do".
        for iteration in 0..self.max_batch_iterations {
            working = dag.clone();

            let mut before = None;
            if phase.includes(StepPhase::Before) {
                let mut ctx = StepContext {
                    store,
                    conn: self.conn.clone(),
                    engine: self.engine.clone(),
                    dag: &mut working,
                    dag_id,
                    dag_name,
                    dag_version,
                    side: StepPhase::Before,
                    run: None,
                };
                let outcome = optimization.step(&mut ctx).await?;
                if outcome.is_terminal() {
                    if let Some(r) = outcome.record() {
                        record = r.clone();
                    }
                    if outcome.persists() {
                        *dag = working;
                    }
                    record.dag_runs_used = dag_runs_used;
                    return Ok(record);
                }
                before = Some(outcome);
            }

            // Each candidate must start from a warehouse without the previous
            // one's relations in it, or its timing measures the leftovers.
            if let Some(previous) = &last_run {
                if let Err(e) = self.engine.cleanup(previous).await {
                    debug!("cleanup of the previous candidate at iteration {iteration}: {e}");
                }
            }
            // And without its own, which is not the same thing. A candidate
            // introduces landing-pad relations whose names are not unique to
            // this DAG, so one left behind by an interrupted search -- or by a
            // different DAG in the same warehouse -- collides with the
            // candidate about to be created rather than with the one before it.
            if let Err(e) = self.engine.cleanup(&working).await {
                debug!("cleanup of the candidate at iteration {iteration}: {e}");
            }

            let stats = match self.engine.run(&working).await {
                Ok(stats) => stats,
                Err(e) => return Err(OptimizerError::Exec(e.to_string())),
            };
            dag_runs_used += 1;
            last_run = Some(working.clone());

            let label = match &before {
                Some(StepOutcome::Trial { label, .. }) => label.clone(),
                _ => format!("iteration {iteration}"),
            };
            debug!(
                "{}: {label} ran in {}ms",
                optimization.name(),
                stats.duration.num_milliseconds()
            );

            if phase.includes(StepPhase::After) {
                let mut ctx = StepContext {
                    store,
                    conn: self.conn.clone(),
                    engine: self.engine.clone(),
                    dag: &mut working,
                    dag_id,
                    dag_name,
                    dag_version,
                    side: StepPhase::After,
                    run: Some(RunContext {
                        run_id: format!("batch-{iteration}"),
                        run_group_id: format!("batch-{dag_id}"),
                        run_phase: run_phase::MEASURE.to_string(),
                        rep_index: iteration as i32,
                        stats: Some(stats),
                    }),
                };
                let outcome = optimization.step(&mut ctx).await?;
                if let Some(r) = outcome.record() {
                    record = r.clone();
                }
                if outcome.persists() {
                    *dag = working;
                }
                if outcome.is_terminal() {
                    record.dag_runs_used = dag_runs_used;
                    return Ok(record);
                }
            }
        }

        warn!(
            "'{}' did not converge within {} iterations; keeping the best it found",
            optimization.name(),
            self.max_batch_iterations
        );
        record.dag_runs_used = dag_runs_used;
        Ok(record)
    }
}

/// The report name for an optimization, kept as it was so stored reports and
/// the benchmark's `pass_stats` table do not change meaning.
fn pass_label(name: &str) -> &'static str {
    match name {
        "hmp" => "HMPPass",
        "omp" => "OMPPass",
        "pushdown" => "PushdownPass",
        _ => "UnknownPass",
    }
}

/// See [`Optimizer::max_batch_iterations`].
const DEFAULT_MAX_BATCH_ITERATIONS: usize = 512;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OptimizerConfig {
    pub run_omp_pass: bool,
    pub run_hmp_pass: bool,
    pub omp_top: Option<usize>,
    pub omp_centrality: OMPCentrality,
    pub omp_early_termination: bool,
    pub omp_use_pushdown: bool,
    pub hmp_downstream_cost: bool,
    pub hmp_max_runs: usize,
    pub hmp_top_cpu_time: f64,
    pub hmp_show_operators: Option<String>,
    pub hmp_show_nodes: Option<String>,
    pub hmp_normalize_with_cardinality: bool,
    pub hmp_strategy: HMPStrategy,
    pub hmp_use_pushdown: bool,
    /// HMP: number of hypotheses the `Greedy` strategy's beam search keeps
    /// alive at each step. Unused by the `Breadth` strategy.
    pub hmp_beam_width: usize,
    /// Capture a CPU/memory/disk timeseries for every HMP/OMP candidate run
    /// and attach it to that iteration's stats.
    pub profile_iterations: bool,
    pub run_pushdown_pass: bool,
    /// Collect an `Explain` HTML section from each pass during `run()`.
    pub explain: bool,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        OptimizerConfig {
            run_omp_pass: true,
            run_hmp_pass: true,
            omp_top: None,
            omp_centrality: OMPCentrality::default(),
            omp_early_termination: true,
            omp_use_pushdown: true,
            hmp_downstream_cost: false,
            hmp_max_runs: 1,
            hmp_top_cpu_time: 0.5,
            hmp_show_operators: None,
            hmp_show_nodes: None,
            hmp_normalize_with_cardinality: false,
            hmp_strategy: HMPStrategy::default(),
            hmp_use_pushdown: true,
            hmp_beam_width: 2,
            profile_iterations: false,
            run_pushdown_pass: false,
            explain: false,
        }
    }
}
impl OptimizerConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_all_disabled(mut self) -> Self {
        self.run_omp_pass = false;
        self.run_hmp_pass = false;
        self.run_pushdown_pass = false;
        self
    }

    pub fn with_all_enabled(mut self) -> Self {
        self.run_omp_pass = true;
        self.run_hmp_pass = true;
        self.run_pushdown_pass = true;
        self
    }

    /// The optimizations this config enables, in the order they are applied.
    ///
    /// Order is HMP -> OMP -> Pushdown, as it has always been: the
    /// materialization searches decide what to materialize, and Pushdown then
    /// narrows what those materializations have to read.
    pub fn enabled_passes(&self) -> Vec<&'static str> {
        let mut passes = Vec::new();
        if self.run_hmp_pass {
            passes.push("hmp");
        }
        if self.run_omp_pass {
            passes.push("omp");
        }
        if self.run_pushdown_pass {
            passes.push("pushdown");
        }
        passes
    }

    pub fn set_pass(&mut self, name: &str, enabled: bool) {
        match name.to_lowercase().as_str() {
            "omp" => self.run_omp_pass = enabled,
            "hmp" => self.run_hmp_pass = enabled,
            "pushdown" => self.run_pushdown_pass = enabled,
            _ => warn!("Unknown optimizer pass: {}", name),
        }
    }

    pub fn with_omp_pass(mut self) -> Self {
        self.run_omp_pass = true;
        self
    }

    pub fn with_hmp_pass(mut self) -> Self {
        self.run_hmp_pass = true;
        self
    }

    pub fn with_omp_top(mut self, top: Option<usize>) -> Self {
        self.omp_top = top;
        self
    }

    pub fn with_omp_centrality(mut self, centrality: OMPCentrality) -> Self {
        self.omp_centrality = centrality;
        self
    }

    pub fn with_omp_early_termination(mut self, early_termination: bool) -> Self {
        self.omp_early_termination = early_termination;
        self
    }

    pub fn with_omp_use_pushdown(mut self, use_pushdown: bool) -> Self {
        self.omp_use_pushdown = use_pushdown;
        self
    }

    /// Rank HMP VIEW candidates by the total cost of duplicate computation
    /// they introduce downstream, instead of an estimated cost to run the
    /// VIEW itself.
    pub fn with_hmp_downstream_cost(mut self, downstream_cost: bool) -> Self {
        self.hmp_downstream_cost = downstream_cost;
        self
    }

    pub fn with_hmp_max_runs(mut self, max_runs: usize) -> Self {
        self.hmp_max_runs = max_runs.max(1);
        self
    }

    pub fn with_hmp_top_cpu_time(mut self, top_cpu_time: f64) -> Self {
        self.hmp_top_cpu_time = if top_cpu_time > 0.0 && top_cpu_time <= 1.0 {
            top_cpu_time
        } else {
            warn!(
                "hmp_top_cpu_time must be in (0, 1.0], got {}, falling back to 0.5",
                top_cpu_time
            );
            0.5
        };
        self
    }

    pub fn with_hmp_show_operators(mut self, show_operators: Option<String>) -> Self {
        self.hmp_show_operators = show_operators;
        self
    }

    pub fn with_hmp_show_nodes(mut self, show_nodes: Option<String>) -> Self {
        self.hmp_show_nodes = show_nodes;
        self
    }

    pub fn with_hmp_normalize_with_cardinality(mut self, normalize_with_cardinality: bool) -> Self {
        self.hmp_normalize_with_cardinality = normalize_with_cardinality;
        self
    }

    pub fn with_hmp_strategy(mut self, strategy: HMPStrategy) -> Self {
        self.hmp_strategy = strategy;
        self
    }

    pub fn with_hmp_use_pushdown(mut self, use_pushdown: bool) -> Self {
        self.hmp_use_pushdown = use_pushdown;
        self
    }

    /// Number of hypotheses the `Greedy` strategy's beam search keeps alive
    /// at each step. Unused by the `Breadth` strategy.
    pub fn with_hmp_beam_width(mut self, beam_width: usize) -> Self {
        self.hmp_beam_width = beam_width.max(1);
        self
    }

    /// Capture a CPU/memory/disk timeseries for every HMP/OMP candidate run
    /// and attach it to that iteration's stats.
    pub fn with_profile_iterations(mut self, profile_iterations: bool) -> Self {
        self.profile_iterations = profile_iterations;
        self
    }

    pub fn with_pushdown_pass(mut self) -> Self {
        self.run_pushdown_pass = true;
        self
    }

    pub fn with_explain(mut self, enabled: bool) -> Self {
        self.explain = enabled;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_config_round_trips_through_json() {
        let config = OptimizerConfig {
            hmp_strategy: HMPStrategy::Greedy,
            omp_centrality: OMPCentrality::Paths,
            hmp_max_runs: 7,
            hmp_top_cpu_time: 0.25,
            run_pushdown_pass: false,
            ..OptimizerConfig::default()
        };

        let json = serde_json::to_string(&config).unwrap();
        let back: OptimizerConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(back.hmp_strategy, HMPStrategy::Greedy);
        assert!(matches!(back.omp_centrality, OMPCentrality::Paths));
        assert_eq!(back.hmp_max_runs, 7);
        assert_eq!(back.hmp_top_cpu_time, 0.25);
        assert!(!back.run_pushdown_pass);
    }

    #[test]
    fn test_optimizer_config_fills_missing_fields_from_default() {
        // The API accepts a partial config, so every absent field must fall
        // back to the same default the CLI would have used.
        let partial: OptimizerConfig = serde_json::from_str(r#"{"hmp_max_runs": 4}"#).unwrap();
        let defaults = OptimizerConfig::default();

        assert_eq!(partial.hmp_max_runs, 4);
        assert_eq!(partial.hmp_top_cpu_time, defaults.hmp_top_cpu_time);
        assert_eq!(partial.run_hmp_pass, defaults.run_hmp_pass);
        assert_eq!(partial.hmp_beam_width, defaults.hmp_beam_width);
    }

    #[test]
    fn test_optimizer_config_rejects_unknown_fields() {
        // A misspelled option in an API body must be an error rather than a
        // silently ignored setting.
        let err = serde_json::from_str::<OptimizerConfig>(r#"{"hmp_max_run": 4}"#);
        assert!(err.is_err());
    }

    #[test]
    fn test_enum_names_match_cli_value_names() {
        // The CLI's --hmp-strategy/--omp-node-centrality values and the JSON
        // encoding must agree, or the same setting means two different things
        // depending on how it was supplied.
        assert_eq!(serde_json::to_string(&HMPStrategy::Breadth).unwrap(), "\"breadth\"");
        assert_eq!(serde_json::to_string(&OMPCentrality::OutDegree).unwrap(), "\"outdegree\"");
    }
}
