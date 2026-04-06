use async_trait::async_trait;
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use log::debug;
use std::marker::PhantomData;

use crate::{
    connectors::Connector,
    executor::Executor,
    opt::{Dag, OptimizerError, OptimizerPass},
};

#[derive(Debug, Clone)]
pub struct OMPPass<C, E>
where
    C: Connector + Send + 'static,
    E: Executor<C> + Send,
{
    a: PhantomData<C>,
    b: PhantomData<E>,
}

impl<C, E> OMPPass<C, E>
where
    C: Connector + Send + 'static,
    E: Executor<C> + Send,
{
    pub fn new() -> Self {
        Self {
            a: PhantomData,
            b: PhantomData,
        }
    }
}

#[async_trait]
impl<C, E> OptimizerPass<C, E> for OMPPass<C, E>
where
    C: Connector + Send + 'static,
    E: Executor<C> + Send,
{
    async fn run(&mut self, dag: &mut Dag) -> Result<usize, OptimizerError> {
        debug!("Running OMPPass");

        Ok(0)
    }
}
