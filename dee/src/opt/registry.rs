//! Name to optimization.
//!
//! The server addresses optimizations by name -- in a URL, in a
//! `dag_optimizations` row, in a benchmark config -- and needs to turn one
//! into a running optimization without a match on every call site. This is
//! the single place that mapping lives; adding an optimization means adding
//! it here and nowhere else.

use std::sync::Arc;

use crate::{
    connectors::Connector,
    executor::Executor,
    opt::{
        Optimization, OptimizerConfig,
        hmp::HMPPass,
        omp::OMPPass,
        parallelism::ParallelismTuning,
        pushdown::PushdownPass,
        step::{OptimizationType, StepPhase},
    },
};

/// Everything dee can optimize a DAG with, with the facts the server needs
/// before it builds one: whether stepping it costs runs, and when it steps.
pub struct OptimizationInfo {
    pub name: &'static str,
    pub optimization_type: OptimizationType,
    pub default_step_phase: StepPhase,
    pub doc: &'static str,
}

pub const OPTIMIZATIONS: &[OptimizationInfo] = &[
    OptimizationInfo {
        name: "hmp",
        optimization_type: OptimizationType::Continuous,
        default_step_phase: StepPhase::Both,
        doc: "Heuristic materialization plan. Ranks views by the operator CPU \
              time they account for, then searches that ranking for views \
              worth materializing, one candidate per DAG run.",
    },
    OptimizationInfo {
        name: "omp",
        optimization_type: OptimizationType::Continuous,
        default_step_phase: StepPhase::Both,
        doc: "Optimal materialization plan. Enumerates every materialization \
              of the most central nodes and measures each, one plan per DAG \
              run.",
    },
    OptimizationInfo {
        name: "parallelism",
        optimization_type: OptimizationType::Continuous,
        default_step_phase: StepPhase::Both,
        doc: "Tunes how many nodes the DAG runs at once. Ladders over \
              node-concurrency caps, one per DAG run, accepting a rung only \
              when every sample of it beats every sample of the incumbent.",
    },
    OptimizationInfo {
        name: "pushdown",
        optimization_type: OptimizationType::Once,
        default_step_phase: StepPhase::Before,
        doc: "Pushes the filters and projections its consumers apply into each \
              materialized node's own query. A pure rewrite: it measures \
              nothing and runs the DAG zero times.",
    },
];

pub fn info(name: &str) -> Option<&'static OptimizationInfo> {
    OPTIMIZATIONS.iter().find(|o| o.name == name)
}

pub fn names() -> Vec<&'static str> {
    OPTIMIZATIONS.iter().map(|o| o.name).collect()
}

/// Build the named optimization from `config`, or `None` if there is no such
/// optimization.
pub fn build<C, E>(
    name: &str,
    conn: Arc<C>,
    engine: Arc<E>,
    config: &OptimizerConfig,
) -> Option<Box<dyn Optimization<C, E>>>
where
    C: Connector + Send + Sync + 'static,
    E: Executor<C> + Send + Sync + 'static,
{
    match name {
        "hmp" => Some(Box::new(HMPPass::from_config(conn, engine, config))),
        "omp" => Some(Box::new(OMPPass::from_config(conn, engine, config))),
        "parallelism" => Some(Box::new(ParallelismTuning::from_config(config))),
        "pushdown" => Some(Box::new(PushdownPass::new(conn, engine))),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_every_configurable_pass_is_in_the_registry() {
        // `OptimizerConfig::enabled_passes` names passes the registry must be
        // able to build; a name in one and not the other is a config that
        // silently optimizes nothing.
        let all = OptimizerConfig::default().with_all_enabled();
        for name in all.enabled_passes() {
            assert!(info(name).is_some(), "'{name}' has no registry entry");
        }
    }

    #[test]
    fn test_the_kinds_are_what_the_server_schedules_on() {
        // The distinction the whole interface turns on: HMP and OMP earn their
        // decisions from measurements and so step around runs; pushdown does
        // not and so steps once.
        assert_eq!(
            info("hmp").unwrap().optimization_type,
            OptimizationType::Continuous
        );
        assert_eq!(
            info("omp").unwrap().optimization_type,
            OptimizationType::Continuous
        );
        assert_eq!(
            info("parallelism").unwrap().optimization_type,
            OptimizationType::Continuous
        );
        assert_eq!(
            info("pushdown").unwrap().optimization_type,
            OptimizationType::Once
        );
    }

    #[test]
    fn test_an_unknown_name_is_not_silently_something_else() {
        assert!(info("hpm").is_none());
    }
}
