//! What an [`Optimization`](crate::opt::Optimization) is handed, and what it
//! hands back.
//!
//! The central change these types encode: an optimization no longer runs the
//! DAG. The server does that -- it already schedules and drives every run --
//! and an optimization gets a turn on either side of each execution. A
//! continuous pass therefore expresses its search as "given what the last run
//! measured, what should the next one try", rather than as a loop that
//! executes candidates itself.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::{
    connectors::Connector,
    dag::Dag,
    executor::{ExecStats, Executor},
    opt::{report::PassOutcome, store::OptStore},
};

/// When an optimization does its work.
///
/// This is the distinction the server needs in order to know when to call
/// `step`, and the one the old single-`run` interface could not express.
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(rename_all = "lowercase")]
pub enum OptimizationType {
    /// Steps around every execution of the DAG and improves it over the DAG's
    /// lifetime, responding to what each run measured. OMP and HMP: both
    /// decide by measurement, and the measurements are the runs the DAG was
    /// going to perform anyway.
    Continuous,
    /// Steps exactly once, when explicitly invoked. Pushdown: a pure
    /// DAG-to-DAG rewrite that measures nothing and has nothing to revise.
    Once,
}

impl OptimizationType {
    pub fn as_str(&self) -> &'static str {
        match self {
            OptimizationType::Continuous => "continuous",
            OptimizationType::Once => "once",
        }
    }
}

impl std::str::FromStr for OptimizationType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "continuous" => Ok(OptimizationType::Continuous),
            "once" => Ok(OptimizationType::Once),
            other => Err(format!(
                "unknown optimization type '{other}'; expected continuous or once"
            )),
        }
    }
}

/// Which side of a DAG execution `step` is called on.
///
/// The optimization author picks the default -- a pass that only reads
/// measurements wants `After`, one that only rewrites wants `Before`, and a
/// search that does both wants `Both`. It is a setting rather than a constant
/// because it is also the dial for how intrusive a continuous optimization is:
/// dropping a converged pass to `After` leaves it observing without touching
/// what runs.
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(rename_all = "lowercase")]
pub enum StepPhase {
    Before,
    After,
    Both,
}

impl StepPhase {
    pub fn as_str(&self) -> &'static str {
        match self {
            StepPhase::Before => "before",
            StepPhase::After => "after",
            StepPhase::Both => "both",
        }
    }

    /// Whether a step configured as `self` should be called on `side`.
    ///
    /// `side` is always `Before` or `After` -- `Both` describes a setting, not
    /// a moment -- so asking `Both.includes(Both)` is a caller bug and answers
    /// false rather than inventing a meaning for it.
    pub fn includes(&self, side: StepPhase) -> bool {
        match side {
            StepPhase::Before => matches!(self, StepPhase::Before | StepPhase::Both),
            StepPhase::After => matches!(self, StepPhase::After | StepPhase::Both),
            StepPhase::Both => false,
        }
    }
}

impl std::str::FromStr for StepPhase {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "before" => Ok(StepPhase::Before),
            "after" => Ok(StepPhase::After),
            "both" => Ok(StepPhase::Both),
            other => Err(format!(
                "unknown step phase '{other}'; expected before, after or both"
            )),
        }
    }
}

/// What `register` and `deregister` are given: the store, scoped to this
/// optimization's namespace, and the DAG being registered on.
pub struct RegisterContext<'a> {
    pub store: &'a dyn OptStore,
    pub dag_id: &'a str,
    pub dag_name: &'a str,
}

/// The run a `step` is attached to.
///
/// `None` on a `Once` optimization's step, which is invoked on its own rather
/// than around an execution.
#[derive(Clone, Debug)]
pub struct RunContext {
    pub run_id: String,
    pub run_group_id: String,
    /// [`run_phase::WARMUP`] or [`run_phase::MEASURE`].
    ///
    /// The driver reports it rather than filtering, because whether a warmup
    /// is a usable measurement is the optimization's judgement: a search
    /// counting DAG runs against a budget may well want to spend one, while a
    /// search comparing candidate runtimes must not.
    pub run_phase: String,
    pub rep_index: i32,
    /// What the execution cost. Populated on an `After` step; `None` on a
    /// `Before` step, where the run has not happened yet.
    pub stats: Option<ExecStats>,
}

/// The phases a run can be in.
///
/// Named here rather than spelled inline at each comparison: an optimization
/// that tested for the wrong string would quietly ignore every run it was
/// supposed to learn from, and look exactly like one that had nothing to say.
pub mod run_phase {
    /// A run whose timing is deliberately discarded -- it exists to absorb
    /// cold-cache cost so the runs after it measure the DAG.
    pub const WARMUP: &str = "warmup";
    /// A run whose timing counts.
    pub const MEASURE: &str = "measure";
}

impl RunContext {
    /// Whether this run's timing is a usable measurement.
    pub fn is_measured(&self) -> bool {
        self.run_phase == run_phase::MEASURE
    }
}

/// Everything a `step` gets.
pub struct StepContext<'a, C, E>
where
    C: Connector + Send + Sync + 'static,
    E: Executor<C> + Send + Sync,
{
    pub store: &'a dyn OptStore,
    pub conn: Arc<C>,
    pub engine: Arc<E>,
    /// The DAG about to run (`Before`) or that just ran (`After`).
    ///
    /// A `Before` step may rewrite this in place, and the rewrite is what that
    /// one execution runs. It is *not* persisted: a search that minted a
    /// version per candidate would bury the DAG's real history under dozens of
    /// rejected experiments. Persisting is what [`StepOutcome::Promote`] asks
    /// for, once.
    pub dag: &'a mut Dag,
    pub dag_id: &'a str,
    pub dag_name: &'a str,
    pub dag_version: i32,
    /// `Before` or `After` -- never `Both`.
    pub side: StepPhase,
    pub run: Option<RunContext>,
}

impl<'a, C, E> StepContext<'a, C, E>
where
    C: Connector + Send + Sync + 'static,
    E: Executor<C> + Send + Sync,
{
    /// The stats of the run that just finished, if this is an `After` step on
    /// a run that produced any.
    pub fn stats(&self) -> Option<&ExecStats> {
        self.run.as_ref().and_then(|r| r.stats.as_ref())
    }

    /// Measured wall time of the run that just finished, in milliseconds.
    pub fn measured_ms(&self) -> Option<i64> {
        self.stats().map(|s| s.duration.num_milliseconds())
    }
}

/// What a `step` did.
#[derive(Clone, Debug)]
pub enum StepOutcome {
    /// Nothing to do on this run. The common case for a continuous pass that
    /// has converged, or that is waiting for a trial already in flight to be
    /// measured.
    Idle,
    /// The DAG was rewritten for this execution only, as a candidate under
    /// test. The server runs it and reports back on the `After` step.
    Trial {
        /// Human-readable identity of the candidate, for logs and reports.
        label: String,
        /// Cancel the run once it has taken this long, because a candidate
        /// already slower than the best plan needs no exact runtime to be
        /// rejected.
        ///
        /// Honoured by both drivers. Under the server's driver the run is the
        /// DAG's real work, so cancelling it is only half the answer: the run
        /// is then finished under `fallback`, rebuilding what the trial never
        /// got to. A budget with no `fallback` there means the candidate is
        /// measured to completion, because a pipeline that did not run is not
        /// an outcome a search gets to choose.
        budget_ms: Option<i64>,
        /// The DAG a cancelled trial is finished under: this search's
        /// incumbent, the best it has measured so far.
        ///
        /// `None` from a pass with no incumbent yet -- nothing has been
        /// measured, so there is nothing better to fall back to.
        fallback: Option<Box<Dag>>,
        record: Box<PassOutcome>,
    },
    /// A `Once` optimization finished its rewrite. The DAG in the context is
    /// the result and should be persisted.
    Rewrote { record: Box<PassOutcome> },
    /// The search converged. The DAG in the context is the winner and should
    /// be persisted as a new version.
    Promote { record: Box<PassOutcome> },
    /// Converged and finished. The optimization stays registered -- so its
    /// state and history remain readable -- but will not be stepped again
    /// until it is deregistered and registered afresh.
    Done { record: Box<PassOutcome> },
}

impl StepOutcome {
    /// Whether this outcome asks the server to store the context's DAG as a
    /// new version.
    pub fn persists(&self) -> bool {
        matches!(
            self,
            StepOutcome::Rewrote { .. } | StepOutcome::Promote { .. }
        )
    }

    /// Whether the optimization has finished and should not be stepped again.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            StepOutcome::Rewrote { .. } | StepOutcome::Promote { .. } | StepOutcome::Done { .. }
        )
    }

    /// The report this step produced, if it produced one.
    pub fn record(&self) -> Option<&PassOutcome> {
        match self {
            StepOutcome::Idle => None,
            StepOutcome::Trial { record, .. }
            | StepOutcome::Rewrote { record }
            | StepOutcome::Promote { record }
            | StepOutcome::Done { record } => Some(record),
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            StepOutcome::Idle => "idle",
            StepOutcome::Trial { .. } => "trial",
            StepOutcome::Rewrote { .. } => "rewrote",
            StepOutcome::Promote { .. } => "promote",
            StepOutcome::Done { .. } => "done",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_only_a_measure_run_counts_as_a_measurement() {
        // A warmup's timing is deliberately discarded, and a search that
        // compared candidates against one would be comparing cold-cache cost.
        let context = |phase: &str| RunContext {
            run_id: "r".into(),
            run_group_id: "g".into(),
            run_phase: phase.into(),
            rep_index: 0,
            stats: None,
        };
        assert!(context(run_phase::MEASURE).is_measured());
        assert!(!context(run_phase::WARMUP).is_measured());
        // The string the server's driver actually records. Getting this wrong
        // makes a continuous optimization silently ignore every run.
        assert_eq!(run_phase::MEASURE, "measure");
        assert_eq!(run_phase::WARMUP, "warmup");
    }

    #[test]
    fn test_step_phase_selects_the_sides_it_names() {
        assert!(StepPhase::Before.includes(StepPhase::Before));
        assert!(!StepPhase::Before.includes(StepPhase::After));
        assert!(!StepPhase::After.includes(StepPhase::Before));
        assert!(StepPhase::After.includes(StepPhase::After));
        assert!(StepPhase::Both.includes(StepPhase::Before));
        assert!(StepPhase::Both.includes(StepPhase::After));
    }

    #[test]
    fn test_both_is_a_setting_not_a_moment() {
        // The driver only ever asks about a real side. Answering true here
        // would let a caller step a pass twice for one execution.
        assert!(!StepPhase::Both.includes(StepPhase::Both));
        assert!(!StepPhase::Before.includes(StepPhase::Both));
    }

    #[test]
    fn test_phase_and_type_round_trip_through_their_wire_names() {
        // These names cross the API and land in the metadata database, so the
        // string form and the enum must not drift apart.
        for phase in [StepPhase::Before, StepPhase::After, StepPhase::Both] {
            assert_eq!(phase.as_str().parse::<StepPhase>().unwrap(), phase);
            assert_eq!(
                serde_json::to_value(phase).unwrap(),
                serde_json::json!(phase.as_str())
            );
        }
        for kind in [OptimizationType::Continuous, OptimizationType::Once] {
            assert_eq!(kind.as_str().parse::<OptimizationType>().unwrap(), kind);
            assert_eq!(
                serde_json::to_value(kind).unwrap(),
                serde_json::json!(kind.as_str())
            );
        }
    }

    #[test]
    fn test_only_a_finished_rewrite_asks_to_be_stored() {
        // A trial must never mint a version; that is the whole reason the
        // outcome distinguishes them.
        let record = || Box::new(PassOutcome::empty());
        assert!(!StepOutcome::Idle.persists());
        assert!(
            !StepOutcome::Trial {
                label: "c1".into(),
                budget_ms: None,
                fallback: None,
                record: record(),
            }
            .persists()
        );
        assert!(StepOutcome::Promote { record: record() }.persists());
        assert!(StepOutcome::Rewrote { record: record() }.persists());
        // Done is terminal but has nothing new to store -- Promote already did.
        assert!(!StepOutcome::Done { record: record() }.persists());
        assert!(StepOutcome::Done { record: record() }.is_terminal());
    }
}
