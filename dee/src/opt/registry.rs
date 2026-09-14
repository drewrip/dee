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
        nodefusion::NodeFusionPass,
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
    /// How this optimization behaves with nothing configured.
    ///
    /// The answer for the options listing, which describes a pass before any
    /// config exists. It is *not* the answer for a pass that is about to be
    /// registered or stepped -- see [`OptimizationInfo::behaviour`].
    pub optimization_type: OptimizationType,
    pub default_step_phase: StepPhase,
    /// How this optimization behaves under a particular config.
    ///
    /// A function rather than the two fields alone because for `nodefusion`
    /// they are not constants: it is a `Once` rewrite by default and a
    /// `Continuous` search under `nodefusion_adaptive_materialize_ctes`, and
    /// the two need opposite handling everywhere the server branches on them --
    /// a `Once` registration is not stepped around runs at all, and a search
    /// stepped only `Before` would never see the measurement it exists to read.
    ///
    /// Lives here rather than at the call sites so that the registry stays the
    /// single place a pass's shape is described, which is the property this
    /// module's documentation claims.
    pub behaviour: fn(&OptimizerConfig) -> (OptimizationType, StepPhase),
    pub doc: &'static str,
}

/// The behaviour of an optimization whose shape does not depend on its config,
/// which is all of them but one.
const fn fixed(
    kind: OptimizationType,
    phase: StepPhase,
) -> fn(&OptimizerConfig) -> (OptimizationType, StepPhase) {
    match (kind, phase) {
        (OptimizationType::Continuous, StepPhase::Both) => {
            |_| (OptimizationType::Continuous, StepPhase::Both)
        }
        _ => |_| (OptimizationType::Once, StepPhase::Before),
    }
}

/// NodeFusion is a rewrite or a search depending on one setting.
fn nodefusion_behaviour(config: &OptimizerConfig) -> (OptimizationType, StepPhase) {
    if config.nodefusion_adaptive_materialize_ctes {
        // Both sides, always. The search proposes a candidate fusion on one and
        // reads what it measured on the other, so there is no coherent
        // `Before`-only or `After`-only setting of it -- unlike a converged HMP,
        // which can usefully be dropped to `After` to keep observing without
        // touching what runs.
        (OptimizationType::Continuous, StepPhase::Both)
    } else {
        (OptimizationType::Once, StepPhase::Before)
    }
}

pub const OPTIMIZATIONS: &[OptimizationInfo] = &[
    OptimizationInfo {
        name: "hmp",
        optimization_type: OptimizationType::Continuous,
        default_step_phase: StepPhase::Both,
        behaviour: fixed(OptimizationType::Continuous, StepPhase::Both),
        doc: "Heuristic materialization plan. Ranks views by the operator CPU \
              time they account for, then searches that ranking for views \
              worth materializing, one candidate per DAG run.",
    },
    OptimizationInfo {
        name: "omp",
        optimization_type: OptimizationType::Continuous,
        default_step_phase: StepPhase::Both,
        behaviour: fixed(OptimizationType::Continuous, StepPhase::Both),
        doc: "Optimal materialization plan. Enumerates every materialization \
              of the most central nodes and measures each, one plan per DAG \
              run.",
    },
    OptimizationInfo {
        name: "parallelism",
        optimization_type: OptimizationType::Continuous,
        default_step_phase: StepPhase::Both,
        behaviour: fixed(OptimizationType::Continuous, StepPhase::Both),
        doc: "Tunes how many nodes the DAG runs at once. Ladders over \
              node-concurrency caps, one per DAG run, accepting a rung only \
              when every sample of it beats every sample of the incumbent.",
    },
    OptimizationInfo {
        name: "pushdown",
        optimization_type: OptimizationType::Once,
        default_step_phase: StepPhase::Before,
        behaviour: fixed(OptimizationType::Once, StepPhase::Before),
        doc: "Pushes the filters and projections its consumers apply into each \
              materialized node's own query. A pure rewrite: it measures \
              nothing and runs the DAG zero times.",
    },
    OptimizationInfo {
        name: "nodefusion",
        // What the pass is *by default*. Unlike every other entry here this is
        // not a constant fact about the optimization: under
        // `nodefusion_adaptive_materialize_ctes` the pass is Continuous and
        // spends DAG runs, and `Optimization::optimization_type` on the built
        // pass is the authority. This table is consulted before a config
        // exists -- it describes the pass a caller gets with nothing set -- so
        // it names the rewrite, and anything scheduling on the answer must ask
        // the built pass instead.
        optimization_type: OptimizationType::Once,
        default_step_phase: StepPhase::Before,
        behaviour: nodefusion_behaviour,
        doc: "Fuses the DAG into one node. Every View a Table reads becomes a \
              CTE, every Table becomes a branch of one UNION ALL discriminated \
              by a `kind` column, and each Table node is rewritten to project \
              its own rows back out. A pure rewrite by default: it measures \
              nothing and runs the DAG zero times. Under \
              `nodefusion_adaptive_materialize_ctes` it instead searches for \
              which CTEs to mark MATERIALIZED, which costs one DAG run per \
              candidate.",
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
        "nodefusion" => Some(Box::new(NodeFusionPass::from_config(config))),
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
    fn test_nodefusion_is_a_rewrite_by_default_and_a_search_when_asked() {
        // The one optimization whose shape is a setting. Both halves matter and
        // both are easy to get wrong in a way nothing else catches:
        //
        //  * filed as `Once`, an adaptive search is never stepped around runs
        //    at all, so it silently does nothing;
        //  * filed as `Before`-only, it proposes a baseline and never reads the
        //    measurement it exists to read.
        let info = info("nodefusion").unwrap();
        let rewrite = OptimizerConfig::default().with_nodefusion_pass();
        assert_eq!(
            (info.behaviour)(&rewrite),
            (OptimizationType::Once, StepPhase::Before)
        );
        let search = rewrite.with_nodefusion_adaptive_materialize_ctes(true);
        assert_eq!(
            (info.behaviour)(&search),
            (OptimizationType::Continuous, StepPhase::Both)
        );
        // The static fields still describe the default, which is what the
        // options listing shows for a pass with nothing set.
        assert_eq!(info.optimization_type, OptimizationType::Once);
    }

    #[test]
    fn test_every_other_optimizations_shape_is_a_constant() {
        // `behaviour` must agree with the static fields wherever the shape does
        // not depend on config, or the two descriptions drift and which one a
        // caller happened to read decides how the pass gets driven.
        let config = OptimizerConfig::default().with_all_enabled();
        for o in OPTIMIZATIONS.iter().filter(|o| o.name != "nodefusion") {
            assert_eq!(
                (o.behaviour)(&config),
                (o.optimization_type, o.default_step_phase),
                "'{}' describes its shape two different ways",
                o.name
            );
        }
    }

    #[test]
    fn test_an_unknown_name_is_not_silently_something_else() {
        assert!(info("hpm").is_none());
    }
}
