pub mod cse;
pub mod omp;
pub mod hmp;

use std::{collections::HashMap, sync::Arc};

use async_trait::async_trait;
use log::debug;

use thiserror::Error;

use crate::{
    connectors::Connector,
    dag::Dag,
    executor::Executor,
    opt::{
        cse::CSEPass,
        omp::{OMPCentrality, OMPCostMetric, OMPPass},
        hmp::HMPPass,
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
    /// Common Subexpression elimination
    run_cse_pass: bool,
    /// Optimal materialization plan
    run_omp_pass: bool,
    /// Heuristic materialization plan
    run_hmp_pass: bool,
    /// Logical rewriting
    run_lr_pass: bool,
    /// OMP top N
    omp_top: Option<usize>,
    /// OMP cost metric
    omp_cost: OMPCostMetric,
    /// OMP node centrality metric
    omp_centrality: OMPCentrality,
    /// HMP no plan dups
    hmp_no_plan_dups: bool,
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
            run_cse_pass: config.run_cse_pass,
            run_omp_pass: config.run_omp_pass,
            run_hmp_pass: config.run_hmp_pass,
            run_lr_pass: config.run_lr_pass,
            omp_top: config.omp_top,
            omp_cost: config.omp_cost,
            omp_centrality: config.omp_centrality,
            hmp_no_plan_dups: config.hmp_no_plan_dups,
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
        if self.run_cse_pass {
            let mut pass: CSEPass<C, E> = CSEPass::new();
            let res = pass.run(dag).await?;
            if self.stats_on_passes {
                stats.insert("CSEPass".to_string(), Arc::new(res));
            }
        } else {
            debug!("skipping CSE pass");
        }

        if self.run_hmp_pass {
            let mut pass: HMPPass<C, E> = HMPPass::new(self.conn.clone(), self.hmp_no_plan_dups);
            let res = pass.run(dag).await?;
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
                self.omp_cost,
                self.omp_centrality,
            );
            let res = pass.run(dag).await?;
            if self.stats_on_passes {
                stats.insert("OMPPass".to_string(), Arc::new(res));
            }
        } else {
            debug!("skipping OMP pass");
        }

        if self.run_lr_pass {
            return Err(OptimizerError::NotImplemented("LR".to_string()));
        } else {
            debug!("skipping LR pass");
        }
        Ok(stats)
    }
}

#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    pub run_cse_pass: bool,
    pub run_omp_pass: bool,
    pub run_hmp_pass: bool,
    pub run_lr_pass: bool,
    pub omp_top: Option<usize>,
    pub omp_cost: OMPCostMetric,
    pub omp_centrality: OMPCentrality,
    pub hmp_no_plan_dups: bool,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        OptimizerConfig {
            run_cse_pass: true,
            run_omp_pass: true,
            run_hmp_pass: true,
            run_lr_pass: false,
            omp_top: None,
            omp_cost: OMPCostMetric::default(),
            omp_centrality: OMPCentrality::default(),
            hmp_no_plan_dups: false,
        }
    }
}
impl OptimizerConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_all_disabled(mut self) -> Self {
        self.run_cse_pass = false;
        self.run_omp_pass = false;
        self.run_hmp_pass = false;
        self.run_lr_pass = false;
        self
    }

    pub fn with_all_enabled(mut self) -> Self {
        self.run_cse_pass = true;
        self.run_omp_pass = true;
        self.run_hmp_pass = true;
        self.run_lr_pass = false; // LR is still not implemented
        self
    }

    pub fn set_pass(&mut self, name: &str, enabled: bool) {
        match name.to_lowercase().as_str() {
            "cse" => self.run_cse_pass = enabled,
            "omp" => self.run_omp_pass = enabled,
            "hmp" => self.run_hmp_pass = enabled,
            _ => debug!("Unknown optimizer pass: {}", name),
        }
    }

    pub fn with_cse_pass(mut self) -> Self {
        self.run_cse_pass = true;
        self
    }

    pub fn with_omp_pass(mut self) -> Self {
        self.run_omp_pass = true;
        self
    }

    pub fn with_hmp_pass(mut self) -> Self {
        self.run_hmp_pass = true;
        self
    }

    pub fn with_lr_pass(mut self) -> Self {
        self.run_lr_pass = true;
        self
    }

    pub fn with_omp_top(mut self, top: Option<usize>) -> Self {
        self.omp_top = top;
        self
    }

    pub fn with_omp_cost(mut self, cost: OMPCostMetric) -> Self {
        self.omp_cost = cost;
        self
    }

    pub fn with_omp_centrality(mut self, centrality: OMPCentrality) -> Self {
        self.omp_centrality = centrality;
        self
    }

    pub fn with_hmp_no_plan_dups(mut self, no_dups: bool) -> Self {
        self.hmp_no_plan_dups = no_dups;
        self
    }
}
