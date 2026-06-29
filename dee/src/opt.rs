pub mod common;
pub mod cspe;
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
        common::validate_dag,
        cspe::CSPEPass,
        hmp::HMPPass,
        omp::{OMPCentrality, OMPPass},
        pushdown::PushdownPass,
    },
};

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
    /// HMP no plan dups
    hmp_no_plan_dups: bool,
    /// Pushdown pass
    run_pushdown_pass: bool,
    pub run_cspe_pass: bool,
    /// Result stats
    stats_on_passes: bool,
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
            hmp_no_plan_dups: config.hmp_no_plan_dups,
            run_pushdown_pass: config.run_pushdown_pass,
            run_cspe_pass: config.run_cspe_pass,
            stats_on_passes: false,
        }
    }

    pub fn stats_on_passes(mut self, collect_stats: bool) -> Self {
        self.stats_on_passes = collect_stats;
        self
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
            let mut pass: HMPPass<C, E> = HMPPass::new(self.conn.clone(), self.hmp_no_plan_dups);
            let res = pass.run(dag).await?;
            if self.stats_on_passes {
                stats.insert("HMPPass".to_string(), Arc::new(res));
            }
            if let Err(e) = validate_dag(dag).await {
                warn!("HMPPass produced an invalid DAG: {e}");
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
            );
            let res = pass.run(dag).await?;
            if self.stats_on_passes {
                stats.insert("OMPPass".to_string(), Arc::new(res));
            }
            if let Err(e) = validate_dag(dag).await {
                warn!("OMPPass produced an invalid DAG: {e}");
            }
        } else {
            debug!("skipping OMP pass");
        }

        if self.run_pushdown_pass {
            let mut pass: PushdownPass<C, E> =
                PushdownPass::new(self.conn.clone(), self.engine.clone());
            let res = pass.run(dag).await?;
            if self.stats_on_passes {
                stats.insert("PushdownPass".to_string(), Arc::new(res));
            }
            if let Err(e) = validate_dag(dag).await {
                warn!("PushdownPass produced an invalid DAG: {e}");
            }
        } else {
            debug!("skipping Pushdown pass");
        }
        if self.run_cspe_pass {
            let mut pass: CSPEPass<C, E> = CSPEPass::new(self.conn.clone(), self.engine.clone());
            let res = pass.run(dag).await?;
            if self.stats_on_passes {
                stats.insert("CSPEPass".to_string(), Arc::new(res));
            }
            if let Err(e) = validate_dag(dag).await {
                warn!("CSPEPass produced an invalid DAG: {e}");
            }
        } else {
            debug!("skipping CSP pass");
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
    pub hmp_no_plan_dups: bool,
    pub run_pushdown_pass: bool,
    pub run_cspe_pass: bool,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        OptimizerConfig {
            run_omp_pass: true,
            run_hmp_pass: true,
            omp_top: None,
            omp_centrality: OMPCentrality::default(),
            omp_early_termination: true,
            omp_use_pushdown: false,
            hmp_no_plan_dups: false,
            run_pushdown_pass: false,
            run_cspe_pass: false,
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
            "cspe" => self.run_cspe_pass = enabled,
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

    pub fn with_hmp_no_plan_dups(mut self, no_dups: bool) -> Self {
        self.hmp_no_plan_dups = no_dups;
        self
    }

    pub fn with_pushdown_pass(mut self) -> Self {
        self.run_pushdown_pass = true;
        self
    }
    pub fn with_cspe_pass(mut self) -> Self {
        self.run_cspe_pass = true;
        self
    }
}
