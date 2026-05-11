pub mod cse;
pub mod omp;

use std::{collections::HashMap, sync::Arc};

use async_trait::async_trait;
use log::debug;
use serde::{Deserialize, Serialize};

use thiserror::Error;

use crate::{
    connectors::Connector,
    dag::Dag,
    executor::Executor,
    opt::{cse::CSEPass, omp::OMPPass},
};

#[derive(Error, Debug)]
pub enum OptimizerError {
    #[error("couldn't execute DAG - {0}")]
    Exec(String),
    #[error("this pass isn't implemented yet, skipping - {0}")]
    NotImplemented(String),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OptimizationMetric {
    Runtime,
    Cost,
}

#[async_trait]
pub trait OptimizerPass<C, E>
where
    C: Connector + Send + 'static,
    E: Executor<C> + Send,
{
    async fn run(&mut self, dag: &mut Dag) -> Result<HashMap<String, String>, OptimizerError>;
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum OMPStrategy {
    #[default]
    Exhaustive,
    Heuristic,
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
    /// Logical rewriting
    run_lr_pass: bool,
    /// Result stats
    stats_on_passes: bool,
    /// Metric to optimize for
    omp_metric: OptimizationMetric,
    /// Strategy for OMP pass
    omp_strategy: OMPStrategy,
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
            run_lr_pass: config.run_lr_pass,
            stats_on_passes: false,
            omp_metric: config.omp_metric,
            omp_strategy: config.omp_strategy,
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

        if self.run_omp_pass {
            let mut pass: OMPPass<C, E> =
                OMPPass::new(self.conn.clone(), self.engine.clone(), self.omp_metric);
            pass.set_strategy(self.omp_strategy);
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
    pub run_lr_pass: bool,
    pub omp_metric: OptimizationMetric,
    pub omp_strategy: OMPStrategy,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        OptimizerConfig {
            run_cse_pass: false,
            run_omp_pass: true,
            run_lr_pass: false,
            omp_metric: OptimizationMetric::Cost,
            omp_strategy: OMPStrategy::Exhaustive,
        }
    }
}
impl OptimizerConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_cse_pass(mut self) -> Self {
        self.run_cse_pass = true;
        self
    }

    pub fn with_omp_pass(mut self) -> Self {
        self.run_omp_pass = true;
        self
    }

    pub fn with_lr_pass(mut self) -> Self {
        self.run_lr_pass = true;
        self
    }

    pub fn with_omp_metric(mut self, metric: OptimizationMetric) -> Self {
        self.omp_metric = metric;
        self
    }

    pub fn with_omp_strategy(mut self, strategy: OMPStrategy) -> Self {
        self.omp_strategy = strategy;
        self
    }
}
