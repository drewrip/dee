//! `ParallelismTuning` -- how many nodes the DAG should run at once.
//!
//! The knob is [`Dag::max_parallelism`]: a cap on how many node queries the
//! executor has in flight simultaneously. dee's default is no cap, which
//! assumes that running more of the DAG at once is never worse. For an engine
//! that already parallelizes *inside* a query that assumption is wrong, and
//! measurably so -- one report node can saturate the machine on its own, and
//! three of them running concurrently make each roughly 5x slower rather than
//! the set 3x faster. On a warehouse where a single query cannot fill the box
//! (Postgres with a stock `max_parallel_workers_per_gather`) the decision
//! inverts and node-level concurrency is the only thing that can use the
//! cores. Neither answer is safe to hardcode, so this measures.
//!
//! **Why it runs before the materialization passes.** A materialization
//! speedup measured against an untuned parallelism level is crediting HMP or
//! OMP with a win that belongs here. The ordering in
//! [`OptimizerConfig::enabled_passes`] puts this first for that reason.
//!
//! **Acceptance is a rank test, not a threshold.** The tempting rule -- accept
//! when one measurement beats the incumbent by more than the noise floor --
//! assumes runtimes are unimodal, and they are not. A bimodal arm that reaches
//! a fast path a fifth of the time will produce a single measurement far
//! outside any noise floor while being no faster in the median. So a rung is
//! screened against the incumbent's best sample, then re-measured, and accepted
//! only if *every* sample it produced beats *every* sample the incumbent has.
//! That assumes nothing about the shape of the distribution: a bimodal arm
//! fails as soon as it draws from its slow mode. It costs one extra run per
//! promising rung, and the screen is what stops it costing one per rung.

use std::{marker::PhantomData, sync::Arc};

use async_trait::async_trait;
use log::debug;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{
    connectors::Connector,
    executor::Executor,
    opt::{
        Dag, Optimization, OptimizerConfig, OptimizerError,
        explain::{render_bar_row, render_card_grid, render_ranked_table},
        report::{IterationStat, PassDetail, PassOutcome, ParallelismDetail, RungResult},
        step::{OptimizationType, RegisterContext, StepContext, StepOutcome, StepPhase},
        store::{OptStore, Registration},
    },
};

const STATE_TABLE: &str = "opt_parallelism_state";
const TRIALS_TABLE: &str = "opt_parallelism_trials";

/// Fraction by which a trial may overrun the incumbent before it is worth
/// abandoning. A candidate already slower than the best known setting needs no
/// exact runtime to be rejected, so there is nothing to learn from letting it
/// finish -- only a censored observation ("at least this bad"), which is all
/// the acceptance test consumes.
const BUDGET_EPS: f64 = 0.25;

/// Where the ladder is, as persisted between steps.
///
/// The search advances one rung per DAG execution and the executions are the
/// DAG's own, so every decision it has made has to survive between them --
/// including across a server restart, which rebuilds the pass from nothing but
/// this row.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ParallelismState {
    /// `"baseline"`, `"searching"`, or `"converged"`.
    phase: String,
    /// Rungs still to try, in ladder order.
    pending: Vec<usize>,
    /// The setting the DAG arrived with, captured on the first baseline run.
    ///
    /// Kept explicitly rather than read back off the DAG at the end, because
    /// an `After` step is handed the *trial* the `Before` step installed, not
    /// the committed definition. Comparing the winner against that would call
    /// a search that changed nothing a promotion, and mint a version for it.
    baseline: Option<usize>,
    /// The setting currently believed best. `None` is unlimited, which is
    /// what an untuned DAG runs at.
    incumbent: Option<usize>,
    /// Every measurement of the incumbent, not just its best.
    ///
    /// The acceptance test compares sample *sets*, so the incumbent's spread
    /// is part of the decision rather than something summarized away. Keeping
    /// a scalar here is exactly the single-measurement rule the rank test
    /// exists to avoid.
    incumbent_samples: Vec<f64>,
    /// Samples collected so far for the rung under test: the screening run,
    /// then each confirmation.
    trial_samples: Vec<f64>,
    /// The rung being measured, and how far through its runs it is.
    in_flight: Option<InFlight>,
    /// Baseline repetitions still owed before the ladder starts.
    seed_remaining: usize,
    runs_used: usize,
    iterations: Vec<IterationStat>,
    /// One entry per rung the ladder resolved, for the report and `explain`.
    results: Vec<RungResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct InFlight {
    rung: usize,
    /// `"screen"` for the first run at a rung, `"confirm"` for the rest.
    stage: String,
}

impl ParallelismState {
    fn new() -> Self {
        Self {
            phase: "baseline".to_string(),
            pending: Vec::new(),
            baseline: None,
            incumbent: None,
            incumbent_samples: Vec::new(),
            trial_samples: Vec::new(),
            in_flight: None,
            seed_remaining: 0,
            runs_used: 0,
            iterations: Vec::new(),
            results: Vec::new(),
        }
    }

    /// The incumbent's best sample -- what a candidate has to beat to be worth
    /// confirming. `None` before anything has been measured.
    fn incumbent_best(&self) -> Option<f64> {
        self.incumbent_samples
            .iter()
            .copied()
            .reduce(f64::min)
    }

    /// The incumbent's worst sample: the runtime the DAG is currently known to
    /// be capable of, and so the honest headline number.
    fn incumbent_worst(&self) -> Option<f64> {
        self.incumbent_samples
            .iter()
            .copied()
            .reduce(f64::max)
    }
}

/// Human-readable name for a setting. `None` is the DAG's untuned state.
fn describe(rung: Option<usize>) -> String {
    match rung {
        Some(n) => format!("parallelism={n}"),
        None => "parallelism=unlimited".to_string(),
    }
}

#[derive(Debug, Clone)]
pub struct ParallelismTuning<C, E>
where
    C: Connector + Send + 'static + Sync,
    E: Executor<C> + Send + Sync,
{
    /// The rungs to try, before pruning against the DAG in front of them.
    ladder: Vec<usize>,
    /// Repetitions used to establish the baseline's sample set.
    seed_repeats: usize,
    /// Re-measurements a rung must survive after it passes the screen.
    confirm_runs: usize,
    /// Capture each iteration's CPU/memory/disk timeseries into its
    /// `IterationStat`.
    profile_iterations: bool,
    /// `Both` by author's default: a rung is installed before a run and judged
    /// after it.
    step_phase: StepPhase,
    /// Retained from the last `step` for `explain`.
    explain_data: Option<ParallelismDetail>,
    _conn: PhantomData<Arc<C>>,
    _engine: PhantomData<Arc<E>>,
}

impl<C, E> ParallelismTuning<C, E>
where
    C: Connector + Send + 'static + Sync,
    E: Executor<C> + Send + Sync,
{
    pub fn new(
        ladder: Vec<usize>,
        seed_repeats: usize,
        confirm_runs: usize,
        profile_iterations: bool,
    ) -> Self {
        Self {
            ladder,
            seed_repeats: seed_repeats.max(1),
            confirm_runs: confirm_runs.max(1),
            profile_iterations,
            step_phase: StepPhase::Both,
            explain_data: None,
            _conn: PhantomData,
            _engine: PhantomData,
        }
    }

    pub fn from_config(config: &OptimizerConfig) -> Self {
        Self::new(
            config.parallelism_ladder.clone(),
            config.parallelism_seed_repeats,
            config.parallelism_confirm_runs,
            config.profile_iterations,
        )
    }

    /// The ladder with rungs that cannot mean anything on this DAG removed.
    ///
    /// Two prunings, both exact rather than heuristic. A rung at or above the
    /// node count can never bind -- the DAG has fewer nodes than the cap
    /// allows in flight -- so it is the same execution as no cap at all, and
    /// several such rungs are the same execution as each other. And a rung
    /// equal to what the DAG already carries is the baseline, which has been
    /// measured already. Trying either costs a full DAG run to re-measure
    /// something known.
    fn rungs_for(&self, dag: &Dag) -> Vec<usize> {
        let nodes = dag.nodes.num_nodes().max(1);
        // What a setting actually does on a DAG this size. `None` is a cap of
        // `nodes`, since nothing more than that can ever be runnable.
        let effective = |rung: Option<usize>| rung.map(|n| n.clamp(1, nodes)).unwrap_or(nodes);

        let baseline = effective(dag.max_parallelism);
        let mut seen = vec![baseline];
        let mut rungs = Vec::new();
        for rung in &self.ladder {
            let eff = effective(Some(*rung));
            if seen.contains(&eff) {
                continue;
            }
            seen.push(eff);
            rungs.push(eff);
        }
        rungs
    }

    async fn load_state(
        &self,
        store: &dyn OptStore,
        dag_id: &str,
    ) -> Result<Option<ParallelismState>, OptimizerError> {
        let rows = match store
            .query(
                &format!("SELECT state FROM {STATE_TABLE} WHERE dag_id = ?"),
                &[json!(dag_id)],
            )
            .await
        {
            Ok(rows) => rows,
            // Not registered, or deregistered while a run was in flight.
            Err(e) if crate::opt::store::is_missing_table(&e) => return Ok(None),
            Err(e) => return Err(e.into()),
        };
        let Some(row) = rows.first() else {
            return Ok(None);
        };
        let raw = row.get("state").and_then(|v| v.as_str()).unwrap_or("");
        serde_json::from_str(raw)
            .map(Some)
            .map_err(|e| OptimizerError::Store(crate::opt::OptStoreError::Decode(e.to_string())))
    }

    async fn save_state(
        &self,
        store: &dyn OptStore,
        dag_id: &str,
        state: &ParallelismState,
    ) -> Result<(), OptimizerError> {
        let encoded = serde_json::to_string(state)
            .map_err(|e| OptimizerError::Store(crate::opt::OptStoreError::Decode(e.to_string())))?;
        store
            .execute(
                &format!("DELETE FROM {STATE_TABLE} WHERE dag_id = ?"),
                &[json!(dag_id)],
            )
            .await?;
        store
            .execute(
                &format!(
                    "INSERT INTO {STATE_TABLE} (dag_id, state, updated_at) VALUES (?, ?, now())"
                ),
                &[json!(dag_id), json!(encoded)],
            )
            .await?;
        Ok(())
    }

    async fn record_trial(
        &self,
        store: &dyn OptStore,
        dag_id: &str,
        run_id: &str,
        rung: Option<usize>,
        stage: &str,
        cost_ms: f64,
    ) -> Result<(), OptimizerError> {
        store
            .execute(
                &format!(
                    "INSERT INTO {TRIALS_TABLE} \
                     (dag_id, run_id, parallelism, stage, cost_ms, recorded_at) \
                     VALUES (?, ?, ?, ?, ?, now())"
                ),
                &[
                    json!(dag_id),
                    json!(run_id),
                    // NULL is unlimited, so the stored trace distinguishes
                    // "no cap" from any particular cap rather than encoding
                    // the former as a number it is not.
                    match rung {
                        Some(n) => json!(n),
                        None => serde_json::Value::Null,
                    },
                    json!(stage),
                    json!(cost_ms),
                ],
            )
            .await?;
        Ok(())
    }

    /// The runtime past which a trial can be abandoned.
    ///
    /// A candidate can never cost more than `1 + BUDGET_EPS` times the best
    /// known setting, however catastrophic it turns out to be. The cap tightens
    /// on its own as the DAG gets faster.
    ///
    /// Honoured only where the run exists solely to measure the candidate.
    /// Under the server the run is the DAG's real work and is measured to
    /// completion.
    fn budget(&self, state: &ParallelismState) -> Option<i64> {
        state
            .incumbent_worst()
            .map(|worst| (worst * (1.0 + BUDGET_EPS)).round() as i64)
    }

    // -----------------------------------------------------------------------
    // Before: install the rung this run should measure
    // -----------------------------------------------------------------------

    async fn step_before(
        &mut self,
        ctx: &mut StepContext<'_, C, E>,
    ) -> Result<StepOutcome, OptimizerError> {
        let Some(mut state) = self.load_state(ctx.store, ctx.dag_id).await? else {
            return Ok(StepOutcome::Idle);
        };

        match state.phase.as_str() {
            // The baseline is the DAG's own setting, so nothing is installed.
            "baseline" | "converged" => Ok(StepOutcome::Idle),
            "searching" => {
                // A rung proposed but never reported on -- a failed or
                // cancelled run. Re-install it rather than moving on, or the
                // ladder would skip a rung it never actually measured.
                if let Some(in_flight) = state.in_flight.clone() {
                    ctx.dag.max_parallelism = Some(in_flight.rung);
                    return Ok(StepOutcome::Trial {
                        label: describe(Some(in_flight.rung)),
                        budget_ms: self.budget(&state),
                        record: Box::new(self.outcome_from(&state)),
                    });
                }

                let Some(rung) = state.pending.first().copied() else {
                    return self.converge(ctx, state).await;
                };
                ctx.dag.max_parallelism = Some(rung);
                state.in_flight = Some(InFlight {
                    rung,
                    stage: "screen".to_string(),
                });
                state.trial_samples.clear();
                self.save_state(ctx.store, ctx.dag_id, &state).await?;
                Ok(StepOutcome::Trial {
                    label: describe(Some(rung)),
                    budget_ms: self.budget(&state),
                    record: Box::new(self.outcome_from(&state)),
                })
            }
            other => {
                debug!("ParallelismTuning: unrecognized state '{other}'; leaving the DAG alone");
                Ok(StepOutcome::Idle)
            }
        }
    }

    // -----------------------------------------------------------------------
    // After: judge what the run measured
    // -----------------------------------------------------------------------

    async fn step_after(
        &mut self,
        ctx: &mut StepContext<'_, C, E>,
    ) -> Result<StepOutcome, OptimizerError> {
        let Some(mut state) = self.load_state(ctx.store, ctx.dag_id).await? else {
            return Ok(StepOutcome::Idle);
        };
        if state.phase == "converged" {
            return Ok(StepOutcome::Idle);
        }
        let Some(run) = ctx.run.clone() else {
            return Ok(StepOutcome::Idle);
        };
        if !run.is_measured() {
            return Ok(StepOutcome::Idle);
        }

        // No stats on a measured run means the execution did not produce a
        // usable time -- it was cancelled at the budget, or failed. Either way
        // it is a censored observation, and a censored observation is enough
        // to reject: what it tells us is "at least as slow as the cap", and
        // the cap is already worse than the incumbent.
        let Some(stats) = run.stats.as_ref() else {
            return self.reject_censored(ctx, state).await;
        };
        let cost = stats.duration.num_milliseconds() as f64;
        let samples = if self.profile_iterations {
            stats.system_samples.clone()
        } else {
            Vec::new()
        };

        if state.phase == "baseline" {
            return self
                .observe_baseline(ctx, state, &run.run_id, cost, samples)
                .await;
        }

        let Some(in_flight) = state.in_flight.take() else {
            // A run this search did not propose. It measured the committed
            // DAG, not a rung, so there is nothing to attribute.
            return Ok(StepOutcome::Idle);
        };

        state.runs_used += 1;
        state.trial_samples.push(cost);
        state.iterations.push(IterationStat {
            iteration: state.iterations.len() + 1,
            runtime_ms: cost as i64,
            outcome: Some(in_flight.stage.clone()),
            system_samples: samples,
            ..Default::default()
        });
        self.record_trial(
            ctx.store,
            ctx.dag_id,
            &run.run_id,
            Some(in_flight.rung),
            &in_flight.stage,
            cost,
        )
        .await?;

        // Screening. A rung that cannot beat even the incumbent's luckiest
        // draw is rejected for the price of one run; only what survives this
        // is worth a confirmation. Without it, confirmation would double the
        // cost of the whole ladder rather than of its promising rungs.
        let best = state.incumbent_best().unwrap_or(f64::MAX);
        if in_flight.stage == "screen" && cost >= best {
            debug!(
                "ParallelismTuning: {} screened out at {cost:.2}ms (incumbent best {best:.2}ms)",
                describe(Some(in_flight.rung))
            );
            return self
                .resolve_rung(ctx, state, in_flight.rung, "rejected (screen)")
                .await;
        }

        // Confirmation. `trial_samples` holds the screen plus every
        // confirmation so far.
        if state.trial_samples.len() <= self.confirm_runs {
            state.in_flight = Some(InFlight {
                rung: in_flight.rung,
                stage: "confirm".to_string(),
            });
            self.save_state(ctx.store, ctx.dag_id, &state).await?;
            return Ok(StepOutcome::Idle);
        }

        // The rank test. Every sample this rung produced must beat every
        // sample the incumbent has -- an ordering claim, which assumes nothing
        // about the shape of either distribution. A bimodal rung that reached
        // its fast path during the screen fails here as soon as it draws from
        // its slow mode.
        let trial_worst = state
            .trial_samples
            .iter()
            .copied()
            .reduce(f64::max)
            .unwrap_or(f64::MAX);
        let accepted = trial_worst < best;

        if accepted {
            debug!(
                "ParallelismTuning: {} accepted; incumbent {:.2}ms -> {trial_worst:.2}ms",
                describe(Some(in_flight.rung)),
                state.incumbent_worst().unwrap_or(0.0),
            );
            state.incumbent = Some(in_flight.rung);
            // The pessimistic end of the rung's own samples, so the incumbent
            // is a time observed on every run of it rather than its best draw.
            state.incumbent_samples = state.trial_samples.clone();
        }
        let verdict = if accepted { "accepted" } else { "rejected (rank test)" };
        self.resolve_rung(ctx, state, in_flight.rung, verdict)
            .await
    }

    /// Record the first `seed_repeats` measurements as the incumbent's sample
    /// set, then open the ladder.
    async fn observe_baseline(
        &mut self,
        ctx: &mut StepContext<'_, C, E>,
        mut state: ParallelismState,
        run_id: &str,
        cost: f64,
        samples: Vec<crate::profile::SystemUsageSample>,
    ) -> Result<StepOutcome, OptimizerError> {
        if state.incumbent_samples.is_empty() {
            state.baseline = ctx.dag.max_parallelism;
            state.incumbent = ctx.dag.max_parallelism;
            state.seed_remaining = self.seed_repeats;
            state.pending = self.rungs_for(ctx.dag);
        }
        state.incumbent_samples.push(cost);
        state.seed_remaining = state.seed_remaining.saturating_sub(1);
        state.runs_used += 1;
        state.iterations.push(IterationStat {
            iteration: state.iterations.len() + 1,
            runtime_ms: cost as i64,
            outcome: Some("baseline".to_string()),
            system_samples: samples,
            ..Default::default()
        });
        self.record_trial(
            ctx.store,
            ctx.dag_id,
            run_id,
            state.incumbent,
            "baseline",
            cost,
        )
        .await?;

        if state.seed_remaining > 0 {
            self.save_state(ctx.store, ctx.dag_id, &state).await?;
            self.explain_data = Some(self.detail_from(&state));
            return Ok(StepOutcome::Idle);
        }

        state.results.push(RungResult {
            parallelism: state.baseline,
            samples: state.incumbent_samples.clone(),
            verdict: "baseline".to_string(),
        });

        // Nothing left to compare against: every rung on the ladder does the
        // same thing to this DAG as the setting it already has.
        if state.pending.is_empty() {
            debug!(
                "ParallelismTuning: no rung differs from {} on a {}-node DAG; nothing to search",
                describe(state.incumbent),
                ctx.dag.nodes.num_nodes()
            );
            return self.converge(ctx, state).await;
        }

        state.phase = "searching".to_string();
        debug!(
            "ParallelismTuning: baseline {} over {} run(s); ladder {:?}",
            describe(state.incumbent),
            state.incumbent_samples.len(),
            state.pending,
        );
        self.save_state(ctx.store, ctx.dag_id, &state).await?;
        self.explain_data = Some(self.detail_from(&state));
        Ok(StepOutcome::Idle)
    }

    /// A trial that produced no usable time. Rejected without a runtime, since
    /// "at least as slow as the budget" is all a censored run says and all the
    /// test needs.
    async fn reject_censored(
        &mut self,
        ctx: &mut StepContext<'_, C, E>,
        mut state: ParallelismState,
    ) -> Result<StepOutcome, OptimizerError> {
        let Some(in_flight) = state.in_flight.take() else {
            return Ok(StepOutcome::Idle);
        };
        state.runs_used += 1;
        debug!(
            "ParallelismTuning: {} produced no usable measurement; rejecting",
            describe(Some(in_flight.rung))
        );
        self.resolve_rung(ctx, state, in_flight.rung, "rejected (censored)")
            .await
    }

    /// Close out a rung: file its verdict, clear the trial, and either move to
    /// the next rung or converge.
    async fn resolve_rung(
        &mut self,
        ctx: &mut StepContext<'_, C, E>,
        mut state: ParallelismState,
        rung: usize,
        verdict: &str,
    ) -> Result<StepOutcome, OptimizerError> {
        state.results.push(RungResult {
            parallelism: Some(rung),
            samples: state.trial_samples.clone(),
            verdict: verdict.to_string(),
        });
        state.pending.retain(|r| *r != rung);
        state.trial_samples.clear();
        state.in_flight = None;

        if state.pending.is_empty() {
            return self.converge(ctx, state).await;
        }
        self.save_state(ctx.store, ctx.dag_id, &state).await?;
        self.explain_data = Some(self.detail_from(&state));
        Ok(StepOutcome::Idle)
    }

    /// The ladder is exhausted. Install the winner and hand it to the caller
    /// to store, or report that nothing beat the setting the DAG arrived with.
    async fn converge(
        &mut self,
        ctx: &mut StepContext<'_, C, E>,
        mut state: ParallelismState,
    ) -> Result<StepOutcome, OptimizerError> {
        state.phase = "converged".to_string();
        state.in_flight = None;
        // Against the setting the DAG arrived with, not against whatever the
        // last trial left in `ctx.dag`.
        let changed = state.incumbent != state.baseline;
        ctx.dag.max_parallelism = state.incumbent;

        self.save_state(ctx.store, ctx.dag_id, &state).await?;
        self.explain_data = Some(self.detail_from(&state));

        debug!(
            "ParallelismTuning converged on {} over {} run(s)",
            describe(state.incumbent),
            state.runs_used
        );

        let record = Box::new(self.outcome_from(&state));
        if changed {
            Ok(StepOutcome::Promote { record })
        } else {
            Ok(StepOutcome::Done { record })
        }
    }

    // -----------------------------------------------------------------------
    // Reporting
    // -----------------------------------------------------------------------

    fn detail_from(&self, state: &ParallelismState) -> ParallelismDetail {
        // The baseline is the first rung filed, and it is filed as soon as its
        // seed repetitions are in. Before that there is nothing to compare
        // against and both ends read as the samples collected so far.
        let baseline_ms = state
            .results
            .first()
            .and_then(|r| r.samples.iter().copied().reduce(f64::max))
            .or_else(|| state.incumbent_worst())
            .unwrap_or(0.0);
        let best = state.incumbent_worst().unwrap_or(baseline_ms);
        ParallelismDetail {
            baseline_parallelism: state.baseline,
            chosen_parallelism: state.incumbent,
            baseline_runtime_ms: baseline_ms,
            best_runtime_ms: best,
            opt_change: if baseline_ms > 0.0 {
                (best - baseline_ms) / baseline_ms
            } else {
                0.0
            },
            seed_repeats: self.seed_repeats,
            confirm_runs: self.confirm_runs,
            ladder: self.ladder.clone(),
            rungs: state.results.clone(),
        }
    }

    fn outcome_from(&self, state: &ParallelismState) -> PassOutcome {
        let detail = self.detail_from(state);
        // Rungs the ladder resolved, baseline excluded -- and, with what is
        // still pending, the size the working set started at.
        let considered = state.results.len().saturating_sub(1) as u32;
        let working_set_size = considered + state.pending.len() as u32;
        PassOutcome {
            dag_runs_used: state.runs_used as u32,
            // At most one thing is ever changed: the setting itself.
            changes_applied: u32::from(state.baseline != state.incumbent),
            candidates_considered: considered,
            working_set_size,
            iterations: state.iterations.clone(),
            detail: PassDetail::Parallelism(detail),
        }
    }

    fn explain_html(&self) -> String {
        let Some(data) = &self.explain_data else {
            return r#"<div class="panel"><p class="subtle">ParallelismTuning did not run.</p></div>"#
                .to_string();
        };

        let cards = render_card_grid(&[
            ("Baseline", describe(data.baseline_parallelism)),
            ("Chosen", describe(data.chosen_parallelism)),
            (
                "Baseline runtime",
                format!("{:.2} ms", data.baseline_runtime_ms),
            ),
            ("Best runtime", format!("{:.2} ms", data.best_runtime_ms)),
            ("Change", format!("{:+.1}%", data.opt_change * 100.0)),
        ]);

        let rung_rows: Vec<Vec<String>> = data
            .rungs
            .iter()
            .map(|r| {
                let samples = if r.samples.is_empty() {
                    "-".to_string()
                } else {
                    r.samples
                        .iter()
                        .map(|s| format!("{s:.0}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                vec![
                    describe(r.parallelism),
                    r.samples.len().to_string(),
                    samples,
                    r.verdict.clone(),
                ]
            })
            .collect();
        let rung_table = render_ranked_table(
            &["Setting", "Runs", "Samples (ms)", "Verdict"],
            &rung_rows,
        );

        // One bar per rung, at its worst sample -- the number the rank test
        // actually compared, not the flattering one.
        let worst = |r: &RungResult| r.samples.iter().copied().reduce(f64::max);
        let max_cost = data
            .rungs
            .iter()
            .filter_map(worst)
            .fold(data.baseline_runtime_ms, f64::max)
            .max(1.0);
        let bars: String = data
            .rungs
            .iter()
            .map(|r| {
                let value = worst(r);
                let label = if r.parallelism == data.chosen_parallelism && r.verdict != "baseline" {
                    format!("{} — chosen", describe(r.parallelism))
                } else {
                    describe(r.parallelism)
                };
                render_bar_row(
                    &label,
                    &match value {
                        Some(v) => format!("{v:.2} ms ({})", r.verdict),
                        None => r.verdict.clone(),
                    },
                    value.unwrap_or(0.0) / max_cost * 100.0,
                )
            })
            .collect();

        format!(
            r##"<div class="section-stack">
        {cards}
        <div class="panel">
          <h2>Why the ladder looks like this</h2>
          <div class="subtle">The configured ladder is {ladder:?}. A rung at or above the DAG's node count can never bind — the cap allows more nodes in flight than the DAG has — so it is the same execution as no cap, and rungs that collapse onto each other or onto the DAG's own setting are dropped rather than re-measured.</div>
          {rung_table}
        </div>
        <div class="panel">
          <h2>How each setting was judged</h2>
          <div class="subtle">The baseline is measured {seed} time(s). A rung is then screened against the incumbent's <em>best</em> sample, and only if it beats that is it re-measured {confirm} more time(s). It is accepted only if its <em>worst</em> sample still beats the incumbent's best — an ordering test, so a setting that is merely sometimes fast fails it. Bars show each setting's worst sample, which is the number compared.</div>
          <div class="plan-tree">{bars}</div>
        </div>
      </div>"##,
            ladder = data.ladder,
            seed = data.seed_repeats,
            confirm = data.confirm_runs,
        )
    }
}

#[async_trait]
impl<C, E> Optimization<C, E> for ParallelismTuning<C, E>
where
    C: Connector + Send + 'static + Sync,
    E: Executor<C> + Send + Sync,
{
    fn name(&self) -> &'static str {
        "parallelism"
    }

    /// Like HMP and OMP, this decides by measurement, and the measurements are
    /// runs the DAG was going to perform anyway.
    fn optimization_type(&self) -> OptimizationType {
        OptimizationType::Continuous
    }

    fn step_phase(&self) -> StepPhase {
        self.step_phase
    }

    fn set_step_phase(&mut self, phase: StepPhase) {
        self.step_phase = phase;
    }

    async fn register(
        &self,
        ctx: &RegisterContext<'_>,
    ) -> Result<Option<Registration>, OptimizerError> {
        ctx.store
            .execute(
                &format!(
                    "CREATE TABLE IF NOT EXISTS {STATE_TABLE} (
                         dag_id     VARCHAR PRIMARY KEY,
                         state      VARCHAR NOT NULL,
                         updated_at TIMESTAMPTZ NOT NULL
                     )"
                ),
                &[],
            )
            .await?;
        ctx.store
            .execute(
                &format!(
                    "CREATE TABLE IF NOT EXISTS {TRIALS_TABLE} (
                         dag_id      VARCHAR NOT NULL,
                         run_id      VARCHAR,
                         parallelism INTEGER,
                         stage       VARCHAR,
                         cost_ms     DOUBLE,
                         recorded_at TIMESTAMPTZ NOT NULL
                     )"
                ),
                &[],
            )
            .await?;

        if self.load_state(ctx.store, ctx.dag_id).await?.is_none() {
            self.save_state(ctx.store, ctx.dag_id, &ParallelismState::new())
                .await?;
        }
        Ok(Some(Registration::new([STATE_TABLE, TRIALS_TABLE])))
    }

    async fn deregister(
        &self,
        ctx: &RegisterContext<'_>,
    ) -> Result<Option<Registration>, OptimizerError> {
        for table in [STATE_TABLE, TRIALS_TABLE] {
            ctx.store
                .execute(
                    &format!("DELETE FROM {table} WHERE dag_id = ?"),
                    &[json!(ctx.dag_id)],
                )
                .await?;
        }
        let remaining = ctx
            .store
            .query(&format!("SELECT count(*) AS n FROM {STATE_TABLE}"), &[])
            .await?;
        let empty = remaining
            .first()
            .and_then(|r| r.get("n"))
            .and_then(|v| v.as_i64())
            .map(|n| n == 0)
            .unwrap_or(false);
        if empty {
            for table in [STATE_TABLE, TRIALS_TABLE] {
                ctx.store
                    .execute(&format!("DROP TABLE IF EXISTS {table}"), &[])
                    .await?;
            }
        }
        Ok(Some(Registration::new([STATE_TABLE, TRIALS_TABLE])))
    }

    async fn step(
        &mut self,
        ctx: &mut StepContext<'_, C, E>,
    ) -> Result<StepOutcome, OptimizerError> {
        match ctx.side {
            StepPhase::Before => self.step_before(ctx).await,
            StepPhase::After => self.step_after(ctx).await,
            StepPhase::Both => Ok(StepOutcome::Idle),
        }
    }

    fn explain(&self) -> Option<(String, String)> {
        Some(("ParallelismTuning".to_string(), self.explain_html()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectors::duckdb::DuckDBConnection;
    use crate::dag::TransformNode;
    use crate::executor::SimpleEngine;
    use crate::graph::Graph;
    use std::collections::{HashMap, HashSet};

    type Pass = ParallelismTuning<DuckDBConnection, SimpleEngine<DuckDBConnection>>;

    fn pass(ladder: Vec<usize>) -> Pass {
        ParallelismTuning::new(ladder, 2, 1, false)
    }

    fn dag_with(nodes: usize, max_parallelism: Option<usize>) -> Dag {
        let mut map = HashMap::new();
        for i in 0..nodes {
            let id = format!("n{i}");
            map.insert(
                id.clone(),
                TransformNode {
                    id,
                    query_text: "SELECT 1".to_string(),
                    materialize: crate::dag::MaterializeMode::View,
                    depends_on: HashSet::new(),
                    schema: None,
                },
            );
        }
        Dag {
            db: "duckdb".to_string(),
            nodes: Graph::new(map),
            sources: Vec::new(),
            max_parallelism,
        }
    }

    #[test]
    fn test_rungs_at_or_above_the_node_count_collapse_onto_each_other() {
        // On a 3-node DAG, caps of 4 and 8 are both "no cap": nothing more
        // than 3 nodes can ever be runnable. Measuring them separately buys
        // three identical executions.
        let rungs = pass(vec![1, 2, 4, 8]).rungs_for(&dag_with(3, None));
        assert_eq!(rungs, vec![1, 2]);
    }

    #[test]
    fn test_the_dags_own_setting_is_never_re_measured() {
        // It is the baseline, which has already cost seed_repeats runs.
        let rungs = pass(vec![1, 2, 4, 8]).rungs_for(&dag_with(20, Some(2)));
        assert_eq!(rungs, vec![1, 4, 8]);
    }

    #[test]
    fn test_an_untuned_dag_treats_unlimited_as_the_node_count() {
        // `None` means every runnable node starts at once, so on a 4-node DAG
        // the rung 4 is the baseline and must not be tried again.
        let rungs = pass(vec![1, 2, 4, 8]).rungs_for(&dag_with(4, None));
        assert_eq!(rungs, vec![1, 2]);
    }

    #[test]
    fn test_a_single_node_dag_has_nothing_to_search() {
        // Nothing can run concurrently with anything, so every rung is the
        // baseline and the ladder is empty rather than four wasted runs.
        assert!(pass(vec![1, 2, 4, 8]).rungs_for(&dag_with(1, None)).is_empty());
    }

    #[test]
    fn test_a_rung_of_zero_is_read_as_one_rather_than_a_stall() {
        // A cap of zero would mean "start nothing", which is not a schedule.
        let rungs = pass(vec![0, 2]).rungs_for(&dag_with(10, None));
        assert_eq!(rungs, vec![1, 2]);
    }

    #[test]
    fn test_the_incumbent_is_compared_at_both_ends_of_its_samples() {
        // The screen uses the incumbent's best sample and the budget uses its
        // worst. Collapsing these to one number is what the rank test exists
        // to avoid.
        let mut state = ParallelismState::new();
        state.incumbent_samples = vec![120.0, 100.0, 110.0];
        assert_eq!(state.incumbent_best(), Some(100.0));
        assert_eq!(state.incumbent_worst(), Some(120.0));
    }

    #[test]
    fn test_the_budget_bounds_a_trials_overrun_by_eps() {
        let mut state = ParallelismState::new();
        state.incumbent_samples = vec![100.0, 200.0];
        // Against the worst sample, so a run at the incumbent's own slow end
        // is not cancelled for being ordinary.
        assert_eq!(pass(vec![1]).budget(&state), Some(250));
    }

    #[test]
    fn test_nothing_measured_yet_means_no_budget() {
        // A cap derived from no measurement would be an arbitrary number.
        assert_eq!(pass(vec![1]).budget(&ParallelismState::new()), None);
    }

    #[test]
    fn test_unlimited_and_a_cap_are_described_differently() {
        // These land in run logs and in the report; a `None` rendered as a
        // number would claim a setting the DAG does not have.
        assert_eq!(describe(None), "parallelism=unlimited");
        assert_eq!(describe(Some(2)), "parallelism=2");
    }

    // -----------------------------------------------------------------------
    // Driving the state machine
    //
    // The rank test is the whole point of the pass, and it only exists across
    // a sequence of steps: install a rung, judge it, install the next. These
    // drive that sequence against a real store, feeding runtimes the harness
    // chooses, so what is asserted is the decision rather than the arithmetic
    // behind it.
    // -----------------------------------------------------------------------

    use crate::connectors::Connector;
    use crate::connectors::duckdb::DuckDBConfig;
    use crate::executor::{ExecStats, NodeStats};
    use crate::opt::store::MemoryStore;
    use crate::opt::step::{RunContext, run_phase};
    use chrono::{TimeDelta, Utc};

    /// A step harness: a registered pass, a store, and a DAG to step against.
    struct Harness {
        pass: Pass,
        store: MemoryStore,
        conn: Arc<DuckDBConnection>,
        engine: Arc<SimpleEngine<DuckDBConnection>>,
        dag: Dag,
        run_seq: usize,
    }

    impl Harness {
        async fn new(ladder: Vec<usize>, seed_repeats: usize, confirm_runs: usize) -> Self {
            let conn = DuckDBConnection::new(DuckDBConfig::new_from_path(":memory:".to_string()))
                .await
                .expect("in-memory duckdb");
            let engine = Arc::new(SimpleEngine::new(Arc::clone(&conn)).expect("engine"));
            let store = MemoryStore::open("parallelism").expect("store");
            let pass = ParallelismTuning::new(ladder, seed_repeats, confirm_runs, false);

            let harness = Self {
                pass,
                store,
                conn,
                engine,
                // 20 nodes, so no rung on a [1,2,4,8] ladder is pruned for
                // exceeding the node count.
                dag: dag_with(20, None),
                run_seq: 0,
            };
            harness
                .pass
                .register(&RegisterContext {
                    store: &harness.store,
                    dag_id: "d1",
                    dag_name: "pipeline",
                })
                .await
                .expect("register");
            harness
        }

        /// One `Before` step. Returns the outcome and the setting it installed.
        async fn before(&mut self) -> (StepOutcome, Option<usize>) {
            let mut dag = self.dag.clone();
            let mut ctx = StepContext {
                store: &self.store,
                conn: Arc::clone(&self.conn),
                engine: Arc::clone(&self.engine),
                dag: &mut dag,
                dag_id: "d1",
                dag_name: "pipeline",
                dag_version: 1,
                side: StepPhase::Before,
                run: None,
            };
            let outcome = self.pass.step(&mut ctx).await.expect("before step");
            let installed = dag.max_parallelism;
            // A promotion is the search handing back the DAG to store, which
            // is the only point the committed definition changes.
            if outcome.persists() {
                self.dag = dag;
            }
            (outcome, installed)
        }

        /// One `After` step reporting `ms`. `None` is a run that produced no
        /// usable time.
        async fn after(&mut self, installed: Option<usize>, ms: Option<i64>) -> StepOutcome {
            self.run_seq += 1;
            let mut dag = self.dag.clone();
            dag.max_parallelism = installed;
            let mut ctx = StepContext {
                store: &self.store,
                conn: Arc::clone(&self.conn),
                engine: Arc::clone(&self.engine),
                dag: &mut dag,
                dag_id: "d1",
                dag_name: "pipeline",
                dag_version: 1,
                side: StepPhase::After,
                run: Some(RunContext {
                    run_id: format!("r{}", self.run_seq),
                    run_group_id: "g".into(),
                    run_phase: run_phase::MEASURE.to_string(),
                    rep_index: self.run_seq as i32,
                    stats: ms.map(stats_of),
                }),
            };
            let outcome = self.pass.step(&mut ctx).await.expect("after step");
            if outcome.persists() {
                self.dag = dag;
            }
            outcome
        }

        /// A full run: install whatever the pass asks for, report `ms`.
        async fn run(&mut self, ms: i64) -> StepOutcome {
            let (before, installed) = self.before().await;
            if before.is_terminal() {
                return before;
            }
            self.after(installed, Some(ms)).await
        }
    }

    fn stats_of(ms: i64) -> ExecStats {
        let start = Utc::now();
        let finish = start + TimeDelta::milliseconds(ms);
        ExecStats {
            start,
            finish,
            duration: TimeDelta::milliseconds(ms),
            node_stats: std::collections::HashMap::<String, NodeStats>::new(),
            system_samples: Vec::new(),
        }
    }

    #[tokio::test]
    async fn test_the_baseline_is_measured_before_any_rung_is_installed() {
        // Seed repetitions measure the DAG as it stands. Installing a cap
        // during them would make the incumbent's sample set describe a setting
        // the DAG does not have.
        let mut h = Harness::new(vec![1, 2], 2, 1).await;
        for _ in 0..2 {
            let (outcome, installed) = h.before().await;
            assert!(matches!(outcome, StepOutcome::Idle), "baseline installs nothing");
            assert_eq!(installed, None);
            h.after(installed, Some(100)).await;
        }
        // Only now does the ladder open.
        let (outcome, installed) = h.before().await;
        assert!(matches!(outcome, StepOutcome::Trial { .. }));
        assert_eq!(installed, Some(1));
    }

    #[tokio::test]
    async fn test_a_rung_that_loses_the_screen_costs_exactly_one_run() {
        // The screen is what keeps confirmation from doubling the whole
        // ladder: a rung that cannot beat the incumbent's best sample is
        // rejected without a second measurement.
        let mut h = Harness::new(vec![1, 2], 2, 1).await;
        h.run(100).await;
        h.run(100).await;

        let (_, installed) = h.before().await;
        assert_eq!(installed, Some(1));
        h.after(installed, Some(150)).await;

        // The next Before must move to rung 2, not re-measure rung 1.
        let (_, installed) = h.before().await;
        assert_eq!(installed, Some(2));
    }

    #[tokio::test]
    async fn test_a_rung_that_wins_once_is_re_measured_before_it_is_believed() {
        let mut h = Harness::new(vec![1], 2, 1).await;
        h.run(100).await;
        h.run(100).await;

        let (_, installed) = h.before().await;
        assert_eq!(installed, Some(1));
        h.after(installed, Some(50)).await;

        // Same rung again -- this is the confirmation, not the next rung.
        let (outcome, installed) = h.before().await;
        assert!(matches!(outcome, StepOutcome::Trial { .. }));
        assert_eq!(installed, Some(1));
    }

    #[tokio::test]
    async fn test_a_bimodal_rung_is_rejected_by_the_rank_test() {
        // The failure the rank test exists for: a setting that reaches a fast
        // path some of the time. Its screening run beats the incumbent by
        // half, which any noise-floor rule would accept, and its confirmation
        // draws from the slow mode. Every sample must beat every incumbent
        // sample, so it is rejected and the DAG keeps its own setting.
        let mut h = Harness::new(vec![1], 2, 1).await;
        h.run(100).await;
        h.run(110).await;

        let (_, installed) = h.before().await;
        h.after(installed, Some(50)).await; // fast mode: passes the screen
        let (_, installed) = h.before().await;
        let outcome = h.after(installed, Some(105)).await; // slow mode

        // 105 does not beat the incumbent's best of 100, so the rung loses.
        assert!(matches!(outcome, StepOutcome::Done { .. }), "nothing to promote");
        let PassDetail::Parallelism(detail) = outcome.record().unwrap().detail.clone() else {
            panic!("expected parallelism detail");
        };
        assert_eq!(detail.chosen_parallelism, None);
        assert_eq!(detail.rungs.last().unwrap().verdict, "rejected (rank test)");
    }

    #[tokio::test]
    async fn test_a_rung_that_wins_every_sample_is_promoted() {
        let mut h = Harness::new(vec![1], 2, 1).await;
        h.run(100).await;
        h.run(110).await;

        let (_, installed) = h.before().await;
        h.after(installed, Some(50)).await;
        let (_, installed) = h.before().await;
        let outcome = h.after(installed, Some(60)).await;

        assert!(matches!(outcome, StepOutcome::Promote { .. }));
        let PassDetail::Parallelism(detail) = outcome.record().unwrap().detail.clone() else {
            panic!("expected parallelism detail");
        };
        assert_eq!(detail.chosen_parallelism, Some(1));
        // The incumbent becomes the pessimistic end of the winner's own
        // samples, not its luckiest draw.
        assert_eq!(detail.best_runtime_ms, 60.0);
        assert_eq!(detail.baseline_runtime_ms, 110.0);
        // And the promoted DAG is the one the caller stores.
        assert_eq!(h.dag.max_parallelism, Some(1));
    }

    #[tokio::test]
    async fn test_the_incumbent_only_ever_improves() {
        // The one guarantee that holds by construction. Rung 1 wins, then
        // rung 2 is measured against *it* rather than against the baseline,
        // and losing to it must not move the incumbent back.
        let mut h = Harness::new(vec![1, 2], 2, 1).await;
        h.run(100).await;
        h.run(100).await;

        let (_, installed) = h.before().await;
        assert_eq!(installed, Some(1));
        h.after(installed, Some(40)).await;
        let (_, installed) = h.before().await;
        h.after(installed, Some(45)).await; // rung 1 accepted at 45

        let (_, installed) = h.before().await;
        assert_eq!(installed, Some(2));
        // Beats the original baseline of 100, but not the incumbent's 40.
        let outcome = h.after(installed, Some(80)).await;

        let PassDetail::Parallelism(detail) = outcome.record().unwrap().detail.clone() else {
            panic!("expected parallelism detail");
        };
        assert_eq!(detail.chosen_parallelism, Some(1));
        assert_eq!(detail.best_runtime_ms, 45.0);
    }

    #[tokio::test]
    async fn test_a_trial_with_no_usable_time_is_rejected_rather_than_retried() {
        // A cancelled or failed run says "at least as slow as the budget",
        // which is enough to reject. Treating it as no observation would leave
        // the rung in flight and re-install it forever.
        let mut h = Harness::new(vec![1], 2, 1).await;
        h.run(100).await;
        h.run(100).await;

        let (_, installed) = h.before().await;
        assert_eq!(installed, Some(1));
        let outcome = h.after(installed, None).await;

        assert!(matches!(outcome, StepOutcome::Done { .. }));
        let PassDetail::Parallelism(detail) = outcome.record().unwrap().detail.clone() else {
            panic!("expected parallelism detail");
        };
        assert_eq!(detail.rungs.last().unwrap().verdict, "rejected (censored)");
    }

    #[tokio::test]
    async fn test_a_trial_that_never_reported_is_re_installed_not_skipped() {
        // A run that died between the two steps leaves the rung in flight. The
        // next Before has to try it again; moving on would mark a rung
        // resolved that was never measured.
        let mut h = Harness::new(vec![1, 2], 2, 1).await;
        h.run(100).await;
        h.run(100).await;

        let (_, installed) = h.before().await;
        assert_eq!(installed, Some(1));
        // No After step at all -- the run never came back.
        let (outcome, installed) = h.before().await;
        assert!(matches!(outcome, StepOutcome::Trial { .. }));
        assert_eq!(installed, Some(1), "the unmeasured rung is retried");
    }

    #[tokio::test]
    async fn test_a_warmup_teaches_the_search_nothing() {
        // Warmup timings are cold-cache cost. Counting one as a baseline
        // sample would set the incumbent from a run that measures the
        // warehouse rather than the DAG.
        let mut h = Harness::new(vec![1], 1, 1).await;
        let (_, installed) = h.before().await;
        let mut dag = h.dag.clone();
        dag.max_parallelism = installed;
        let mut ctx = StepContext {
            store: &h.store,
            conn: Arc::clone(&h.conn),
            engine: Arc::clone(&h.engine),
            dag: &mut dag,
            dag_id: "d1",
            dag_name: "pipeline",
            dag_version: 1,
            side: StepPhase::After,
            run: Some(RunContext {
                run_id: "w1".into(),
                run_group_id: "g".into(),
                run_phase: run_phase::WARMUP.to_string(),
                rep_index: 0,
                stats: Some(stats_of(9999)),
            }),
        };
        h.pass.step(&mut ctx).await.expect("after step");

        // Still in the baseline phase, still owed its seed run.
        let (outcome, installed) = h.before().await;
        assert!(matches!(outcome, StepOutcome::Idle));
        assert_eq!(installed, None);
    }

    #[tokio::test]
    async fn test_a_dag_with_no_rung_to_try_converges_without_spending_a_trial() {
        // A single-node DAG: every rung is the baseline. The search must
        // finish on the seed runs rather than measure four identical
        // executions.
        let mut h = Harness::new(vec![1, 2, 4, 8], 1, 1).await;
        h.dag = dag_with(1, None);
        let outcome = h.run(100).await;
        assert!(matches!(outcome, StepOutcome::Done { .. }));

        let PassDetail::Parallelism(detail) = outcome.record().unwrap().detail.clone() else {
            panic!("expected parallelism detail");
        };
        assert_eq!(detail.chosen_parallelism, None);
        assert_eq!(detail.rungs.len(), 1, "only the baseline was measured");
    }
}
