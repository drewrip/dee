use std::marker::PhantomData;

use log::debug;

use crate::{
    connectors::Connector,
    dag::Dag,
    executor::Executor,
    opt::{OptimizerError, OptimizerPass},
};

#[derive(Debug, Clone)]
pub struct CSEPass<C, E>
where
    C: Connector + Send,
    E: Executor<C>,
{
    a: PhantomData<C>,
    b: PhantomData<E>,
}

impl<C, E> CSEPass<C, E>
where
    C: Connector + Send,
    E: Executor<C>,
{
    pub fn new() -> Self {
        Self {
            a: PhantomData,
            b: PhantomData,
        }
    }
}

impl<C, E> OptimizerPass<C, E> for CSEPass<C, E>
where
    C: Connector + Send,
    E: Executor<C>,
{
    fn run(&mut self, dag: &mut Dag) -> Result<usize, OptimizerError> {
        debug!("Running CSEPass");
        Ok(0)
    }
}
