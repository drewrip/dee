pub mod cse;

use std::sync::Arc;

use crate::{connectors::Connector, dag::Dag, executor::Executor, opt::cse::CSEPass};

use async_trait::async_trait;
use log::debug;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum OptimizerError {
    #[error("couldn't execute DAG - {0}")]
    Exec(String),
    #[error("this pass isn't implemented yet, skipping - {0}")]
    NotImplemented(String),
}

pub trait OptimizerPass<C, E>
where
    C: Connector + Send,
    E: Executor<C>,
{
    fn run(&mut self, dag: &mut Dag) -> Result<usize, OptimizerError>;
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
    /// Optimizal materialization plan
    run_omp_pass: bool,
    /// Logical rewriting
    run_lr_pass: bool,
}

impl<C, E> Optimizer<C, E>
where
    C: Connector + Send,
    E: Executor<C>,
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
        }
    }

    pub fn run(&mut self, dag: &mut Dag) -> Result<usize, OptimizerError> {
        if self.run_cse_pass {
            let mut pass: CSEPass<C, E> = CSEPass::new();
            let res = pass.run(dag)?;
        } else {
            debug!("skipping CSE pass");
        }
        if self.run_omp_pass {
            return Err(OptimizerError::NotImplemented("OMP".to_string()));
        } else {
            debug!("skipping OMP pass");
        }
        if self.run_lr_pass {
            return Err(OptimizerError::NotImplemented("LR".to_string()));
        } else {
            debug!("skipping LR pass");
        }
        Ok(0)
    }
}

#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    run_cse_pass: bool,
    run_omp_pass: bool,
    run_lr_pass: bool,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        OptimizerConfig {
            run_cse_pass: true,
            run_omp_pass: false,
            run_lr_pass: false,
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
}
