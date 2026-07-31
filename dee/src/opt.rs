pub mod common;
pub mod explain;
pub mod hmp;
pub mod omp;
pub mod pushdown;

use std::{collections::HashMap, sync::Arc};

use async_trait::async_trait;
use log::{debug, error, warn};

use thiserror::Error;

use crate::{
    connectors::Connector,
    dag::Dag,
    executor::Executor,
    opt::{
        hmp::{HMPPass, HMPStrategy},
        omp::{OMPCentrality, OMPPass},
        pushdown::PushdownPass,
    },
};

pub use crate::opt::explain::render_explain_html;

#[derive(Error, Debug)]
pub enum OptimizerError {
    #[error("couldn't execute DAG - {0}")]
    Exec(String),
    #[error("this pass isn't implemented yet, skipping - {0}")]
    NotImplemented(String),
}

#[async_trait]
pub trait OptimizerPass<C, E>
where
    C: Connector + Send + 'static,
    E: Executor<C> + Send,
{
    async fn run(&mut self, dag: &mut Dag) -> Result<HashMap<String, String>, OptimizerError>;
}

/// Implemented by optimizer passes that can explain, after `run()` has
/// completed, what they did and why. Each pass retains whatever data it
/// collected during `run()` (candidates considered, plans tried, the one
/// chosen) so `explain()` can render it into an HTML snippet.
pub trait Explain {
    /// Tab label for this pass's explain section, e.g. `"HMPPass"`.
    fn explain_label(&self) -> String;

    /// An HTML snippet (a `<section>...</section>`) explaining what this
    /// pass did and why. Only meaningful after `run()` has completed.
    fn explain(&self) -> String;
}

#[derive(Debug, Clone)]
pub struct Optimizer<C, E>
where
    C: Connector + Send,
    E: Executor<C>,
{
    conn: Arc<C>,
    engine: Arc<E>,
    /// Optimal materialization plan
    run_omp_pass: bool,
    /// Heuristic materialization plan
    run_hmp_pass: bool,
    /// OMP top N
    omp_top: Option<usize>,
    /// OMP node centrality metric
    omp_centrality: OMPCentrality,
    /// OMP early termination
    omp_early_termination: bool,
    /// OMP use pushdown before each candidate evaluation
    omp_use_pushdown: bool,
    /// HMP: rank VIEW candidates by the total cost of duplicate computation
    /// they introduce downstream, instead of an estimated cost to run the
    /// VIEW itself.
    hmp_downstream_cost: bool,
    /// HMP max DAG runs to spend searching for materialization candidates
    hmp_max_runs: usize,
    /// HMP fraction of total operator CPU time used to build the working set
    hmp_top_cpu_time: f64,
    /// HMP: when set, log a table of operator rankings after the baseline
    /// run. `Some("")` logs only; `Some(path)` also writes it to `path`.
    hmp_show_operators: Option<String>,
    /// HMP: when set, log a table of node (View) rankings after the
    /// baseline run. `Some("")` logs only; `Some(path)` also writes it to
    /// `path`.
    hmp_show_nodes: Option<String>,
    /// HMP: rank VIEW candidates by total CPU time divided by the View's
    /// estimated cardinality (from its EXPLAIN plan), instead of raw total
    /// CPU time.
    hmp_normalize_with_cardinality: bool,
    /// HMP: strategy for searching through the node ranking.
    hmp_strategy: HMPStrategy,
    /// HMP: use pushdown before each candidate evaluation
    hmp_use_pushdown: bool,
    /// HMP: number of hypotheses the `Greedy` strategy's beam search keeps
    /// alive at each step.
    hmp_beam_width: usize,
    /// Capture a CPU/memory/disk timeseries for every HMP/OMP candidate run
    /// and attach it to that iteration's stats.
    profile_iterations: bool,
    /// Pushdown pass
    run_pushdown_pass: bool,
    /// Result stats
    stats_on_passes: bool,
    /// Collect per-pass `Explain` sections during `run()`
    explain_enabled: bool,
    /// `(pass label, explain HTML)` pairs collected during the last `run()`,
    /// in pass execution order. Only populated when `explain_enabled`.
    explain_sections: Vec<(String, String)>,
}

impl<C, E> Optimizer<C, E>
where
    C: Connector + Send + 'static + Sync,
    E: Executor<C> + Send + Sync,
{
    pub fn new(conn: Arc<C>, engine: Arc<E>) -> Self {
        let config = OptimizerConfig::default();
        Self::new_with_config(conn, engine, config)
    }

    pub fn new_with_config(conn: Arc<C>, engine: Arc<E>, config: OptimizerConfig) -> Self {
        Self {
            conn,
            engine,
            run_omp_pass: config.run_omp_pass,
            run_hmp_pass: config.run_hmp_pass,
            omp_top: config.omp_top,
            omp_centrality: config.omp_centrality,
            omp_early_termination: config.omp_early_termination,
            omp_use_pushdown: config.omp_use_pushdown,
            hmp_downstream_cost: config.hmp_downstream_cost,
            hmp_max_runs: config.hmp_max_runs,
            hmp_top_cpu_time: config.hmp_top_cpu_time,
            hmp_show_operators: config.hmp_show_operators,
            hmp_show_nodes: config.hmp_show_nodes,
            hmp_normalize_with_cardinality: config.hmp_normalize_with_cardinality,
            hmp_strategy: config.hmp_strategy,
            hmp_use_pushdown: config.hmp_use_pushdown,
            hmp_beam_width: config.hmp_beam_width,
            profile_iterations: config.profile_iterations,
            run_pushdown_pass: config.run_pushdown_pass,
            stats_on_passes: false,
            explain_enabled: config.explain,
            explain_sections: Vec::new(),
        }
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

    pub async fn run(
        &mut self,
        dag: &mut Dag,
    ) -> Result<HashMap<String, Arc<HashMap<String, String>>>, OptimizerError> {
        let mut stats = HashMap::new();

        if let Err(e) = self.engine.resolve_schemas(dag).await {
            error!("couldn't resolve_schemas: {e}")
        }

        if self.run_hmp_pass {
            let mut pass: HMPPass<C, E> = HMPPass::new(
                self.conn.clone(),
                self.engine.clone(),
                self.hmp_downstream_cost,
                self.hmp_max_runs,
                self.hmp_top_cpu_time,
                self.hmp_show_operators.clone(),
                self.hmp_show_nodes.clone(),
                self.hmp_normalize_with_cardinality,
                self.hmp_strategy,
                self.hmp_use_pushdown,
                self.hmp_beam_width,
                self.profile_iterations,
            );
            let res = pass.run(dag).await?;
            if self.explain_enabled {
                self.explain_sections
                    .push((pass.explain_label(), pass.explain()));
            }
            if self.stats_on_passes {
                stats.insert("HMPPass".to_string(), Arc::new(res));
            }
        } else {
            debug!("skipping HMP pass");
        }

        if self.run_omp_pass {
            let mut pass: OMPPass<C, E> = OMPPass::new(
                self.conn.clone(),
                self.engine.clone(),
                self.omp_top,
                self.omp_centrality,
                self.omp_early_termination,
                self.omp_use_pushdown,
                self.profile_iterations,
            );
            let res = pass.run(dag).await?;
            if self.explain_enabled {
                self.explain_sections
                    .push((pass.explain_label(), pass.explain()));
            }
            if self.stats_on_passes {
                stats.insert("OMPPass".to_string(), Arc::new(res));
            }
        } else {
            debug!("skipping OMP pass");
        }

        if self.run_pushdown_pass {
            let mut pass: PushdownPass<C, E> =
                PushdownPass::new(self.conn.clone(), self.engine.clone());
            let res = pass.run(dag).await?;
            if self.explain_enabled {
                self.explain_sections
                    .push((pass.explain_label(), pass.explain()));
            }
            if self.stats_on_passes {
                stats.insert("PushdownPass".to_string(), Arc::new(res));
            }
        } else {
            debug!("skipping Pushdown pass");
        }

        Ok(stats)
    }
}

#[derive(Debug, Clone)]
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
        self
    }

    pub fn with_all_enabled(mut self) -> Self {
        self.run_omp_pass = true;
        self.run_hmp_pass = true;
        self
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
