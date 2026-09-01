use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use itertools::{Itertools, repeat_n};
use log::debug;
use std::{marker::PhantomData, sync::Arc};

use crate::{
    connectors::Connector,
    dag::MaterializeMode,
    executor::Executor,
    opt::{
        Dag, Optimization, OptimizerConfig, OptimizerError,
        common::{landing_pad_name, make_temp},
        explain::{render_bar_row, render_card_grid, render_ranked_table},
        pushdown::PushdownPass,
        report::{CandidateScore, IterationStat, OmpDetail, PassDetail, PassOutcome},
        step::{OptimizationType, RegisterContext, StepContext, StepOutcome, StepPhase},
        store::{OptStore, Registration},
    },
};
use serde_json::json;

/// Where OMP is in its enumeration, as persisted between steps.
///
/// OMP measures every materialization of its candidate nodes, and measuring
/// means running the DAG. Under the server those runs are the DAG's own, so
/// the enumeration has to be resumable: a cursor into the plan list, plus what
/// has been learned so far. This is that.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct OmpState {
    /// `"baseline"`, `"searching"`, or `"converged"`.
    phase: String,
    baseline_cost: f64,
    best_cost: f64,
    /// The winning plan as one mode per entry of `top_candidates`.
    best_modes: Vec<String>,
    /// Every qualifying node with its centrality score, ranked.
    candidates: Vec<(String, usize)>,
    /// The nodes actually enumerated over, after `top_n`.
    top_candidates: Vec<String>,
    /// The DAG's own modes for `top_candidates`, so the plan equal to what is
    /// already committed is not paid for a second time.
    baseline_modes: Vec<String>,
    /// Position in the 2^N plan enumeration.
    plan_cursor: usize,
    runs_used: usize,
    iterations: Vec<IterationStat>,
    /// `(label, outcome, cost_ms)` per plan considered, for `explain`.
    attempts: Vec<(String, String, Option<f64>)>,
    in_flight: Option<OmpInFlight>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OmpInFlight {
    index: usize,
    modes: Vec<String>,
}

impl OmpState {
    fn new() -> Self {
        Self {
            phase: "baseline".to_string(),
            baseline_cost: 0.0,
            best_cost: f64::MAX,
            best_modes: Vec::new(),
            candidates: Vec::new(),
            top_candidates: Vec::new(),
            baseline_modes: Vec::new(),
            plan_cursor: 0,
            runs_used: 0,
            iterations: Vec::new(),
            attempts: Vec::new(),
            in_flight: None,
        }
    }
}

/// One materialization plan attempt, used by `explain`.
#[derive(Debug, Clone)]
struct OMPAttempt {
    label: String,
    outcome: String,
    cost_ms: Option<f64>,
    is_best: bool,
}

/// Everything `Explain::explain` needs to describe what the last `run()`
/// did and why, retained from otherwise-local data computed during `run()`.
#[derive(Debug, Clone)]
struct OMPExplainData {
    baseline_cost: f32,
    best_cost: f32,
    centrality: OMPCentrality,
    candidates: Vec<(String, usize)>,
    top_candidates: Vec<String>,
    best_plan: Vec<String>,
    attempts: Vec<OMPAttempt>,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OMPCentrality {
    #[default]
    OutDegree,
    Paths,
}

#[derive(Debug, Clone)]
pub struct OMPPass<C, E>
where
    C: Connector + Send + 'static + Sync,
    E: Executor<C> + Send + Sync,
{
    conn: Arc<C>,
    engine: Arc<E>,
    top_n: Option<usize>,
    centrality: OMPCentrality,
    early_termination: bool,
    use_pushdown: bool,
    /// Capture each iteration's CPU/memory/disk timeseries into its
    /// `IterationStat`.
    profile_iterations: bool,
    /// Which side of an execution to step on. `Both` by author's default: a
    /// plan is installed before a run and scored after it.
    step_phase: StepPhase,
    /// Data collected during the last `step()`, used by `explain`.
    explain_data: Option<OMPExplainData>,
    _phantom: PhantomData<C>,
}

impl<C, E> OMPPass<C, E>
where
    C: Connector + Send + 'static + Sync,
    E: Executor<C> + Send + Sync,
{
    pub fn new(
        conn: Arc<C>,
        engine: Arc<E>,
        top_n: Option<usize>,
        centrality: OMPCentrality,
        early_termination: bool,
        use_pushdown: bool,
        profile_iterations: bool,
    ) -> Self {
        Self {
            conn,
            engine,
            top_n,
            centrality,
            early_termination,
            use_pushdown,
            profile_iterations,
            step_phase: StepPhase::Both,
            explain_data: None,
            _phantom: PhantomData,
        }
    }
}

// ---------------------------------------------------------------------------
// The step interface
//
// OMP evaluates every materialization of its most central nodes by running
// each one. That enumeration used to happen inside a single call, buying its
// own runs; now it advances one plan per execution of the DAG, and the
// executions are the DAG's own. The enumeration is unchanged -- same
// candidates, same 2^N plans, same winner -- it is just resumable.
// ---------------------------------------------------------------------------

const STATE_TABLE: &str = "opt_omp_state";
const TRIALS_TABLE: &str = "opt_omp_trials";

impl<C, E> OMPPass<C, E>
where
    C: Connector + Send + 'static + Sync,
    E: Executor<C> + Send + Sync,
{
    pub fn from_config(conn: Arc<C>, engine: Arc<E>, config: &OptimizerConfig) -> Self {
        Self::new(
            conn,
            engine,
            config.omp_top,
            config.omp_centrality,
            config.omp_early_termination,
            config.omp_use_pushdown,
            config.profile_iterations,
        )
    }

    async fn load_state(
        &self,
        store: &dyn OptStore,
        dag_id: &str,
    ) -> Result<Option<OmpState>, OptimizerError> {
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
        state: &OmpState,
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

    /// Every qualifying node, ranked by the configured centrality metric.
    ///
    /// Only nodes with more than one downstream consumer AND more than one
    /// downstream path reaching a materialized node benefit from being
    /// materialized -- otherwise there is nothing to deduplicate. In `Paths`
    /// mode these two checks are still the filter; paths-to-sinks is used a
    /// second time only to break ties among the survivors.
    fn rank_candidates(&self, dag: &Dag) -> Vec<(String, usize)> {
        let mut candidates: Vec<(String, usize)> = dag
            .nodes
            .nodes()
            .map(|n| (n.id.clone(), dag.nodes.out_degree(&n.id)))
            .filter(|(_, out_degree)| *out_degree > 1)
            .filter(|(id, _)| dag.nodes.paths_to_sinks(id) > 1)
            .map(|(id, out_degree)| {
                let rank = match self.centrality {
                    OMPCentrality::OutDegree => out_degree,
                    OMPCentrality::Paths => dag.nodes.paths_to_sinks(&id),
                };
                (id, rank)
            })
            .collect();
        candidates.sort_by_key(|(_, rank)| *rank);
        candidates
    }

    /// The 2^N plans over `n` candidates, as mode-name lists.
    ///
    /// Regenerated from `n` on each step rather than persisted: the
    /// enumeration is a pure function of the candidate count, and a cursor
    /// into it is far smaller and less likely to go stale than the list.
    fn plans(n: usize) -> Vec<Vec<String>> {
        repeat_n(
            [MaterializeMode::View, MaterializeMode::TempTable].into_iter(),
            n,
        )
        .multi_cartesian_product()
        .map(|plan| plan.iter().map(|m| m.as_str().to_string()).collect())
        .collect()
    }

    fn describe_plan(top_candidates: &[String], modes: &[String]) -> String {
        top_candidates
            .iter()
            .zip(modes.iter())
            .map(|(id, mode)| format!("{id}={mode}"))
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Apply `modes` to `dag`, then optionally push predicates into it.
    async fn build_plan(
        &self,
        dag: &mut Dag,
        top_candidates: &[String],
        modes: &[String],
    ) -> Result<(), OptimizerError> {
        for (pos, mode) in modes.iter().enumerate() {
            match MaterializeMode::from(mode.clone()) {
                MaterializeMode::TempTable => {
                    make_temp(dag, &top_candidates[pos])?;
                }
                other => {
                    dag.nodes
                        .get_mut(top_candidates[pos].clone())
                        .ok_or_else(|| OptimizerError::Exec("missing node".to_string()))?
                        .materialize = other;
                }
            }
        }
        if self.use_pushdown {
            let mut pushdown = PushdownPass::new(self.conn.clone(), self.engine.clone());
            if let Err(e) = pushdown.rewrite(dag).await {
                debug!("OMPPass: pushdown failed for a plan, continuing without it: {e}");
            }
        }
        Ok(())
    }

    /// Drop `lp_*` landing pads left behind by an interrupted search.
    ///
    /// OMP inserts them while evaluating and removes them when it is done, but
    /// a process that died mid-search leaves stale relations that the next
    /// attempt would collide with. Missing ones are silently ignored.
    async fn sweep_landing_pads(&self, dag: &Dag) {
        let pads: Vec<String> = dag
            .nodes
            .nodes()
            .map(|n| landing_pad_name(&n.id))
            .collect();
        for lp in pads {
            self.conn
                .drop_relation(MaterializeMode::Table, lp.clone())
                .await
                .ok();
            self.conn.drop_relation(MaterializeMode::View, lp).await.ok();
        }
    }

    fn outcome_from(&self, state: &OmpState) -> PassOutcome {
        let best_cost = if state.best_cost == f64::MAX {
            state.baseline_cost
        } else {
            state.best_cost
        };
        let change = if state.baseline_cost > 0.0 {
            (best_cost - state.baseline_cost) / state.baseline_cost
        } else {
            0.0
        };
        let best_plan: Vec<String> = state
            .top_candidates
            .iter()
            .zip(state.best_modes.iter())
            .filter(|(_, mode)| mode.as_str() == MaterializeMode::TempTable.as_str())
            .map(|(id, _)| id.clone())
            .collect();

        PassOutcome {
            dag_runs_used: state.runs_used as u32,
            changes_applied: best_plan.len() as u32,
            candidates_considered: state.iterations.len().saturating_sub(1) as u32,
            working_set_size: state.top_candidates.len() as u32,
            iterations: state.iterations.clone(),
            detail: PassDetail::Omp(OmpDetail {
                baseline_value: state.baseline_cost,
                best_value: best_cost,
                opt_change: change,
                best_plan,
                centrality: format!("{:?}", self.centrality),
                candidates_ranked: state
                    .candidates
                    .iter()
                    .rev()
                    .map(|(node_id, rank)| CandidateScore {
                        node_id: node_id.clone(),
                        score: *rank as f64,
                    })
                    .collect(),
                early_termination: self.early_termination,
                use_pushdown: self.use_pushdown,
            }),
        }
    }

    fn remember_explain(&mut self, state: &OmpState) {
        let best_cost = if state.best_cost == f64::MAX {
            state.baseline_cost
        } else {
            state.best_cost
        };
        let mut attempts: Vec<OMPAttempt> = state
            .attempts
            .iter()
            .map(|(label, outcome, cost_ms)| OMPAttempt {
                label: label.clone(),
                outcome: outcome.clone(),
                cost_ms: *cost_ms,
                is_best: false,
            })
            .collect();
        for attempt in attempts.iter_mut() {
            if let Some(cost) = attempt.cost_ms
                && (cost - best_cost).abs() < 1e-6
            {
                attempt.is_best = true;
            }
        }
        let best_plan: Vec<String> = state
            .top_candidates
            .iter()
            .zip(state.best_modes.iter())
            .filter(|(_, mode)| mode.as_str() == MaterializeMode::TempTable.as_str())
            .map(|(id, _)| id.clone())
            .collect();

        self.explain_data = Some(OMPExplainData {
            baseline_cost: state.baseline_cost as f32,
            best_cost: best_cost as f32,
            centrality: self.centrality,
            candidates: state.candidates.clone(),
            top_candidates: state.top_candidates.clone(),
            best_plan,
            attempts,
        });
    }

    async fn step_before(
        &mut self,
        ctx: &mut StepContext<'_, C, E>,
    ) -> Result<StepOutcome, OptimizerError> {
        let Some(mut state) = self.load_state(ctx.store, ctx.dag_id).await? else {
            return Ok(StepOutcome::Idle);
        };

        match state.phase.as_str() {
            // The baseline is the DAG's own plan, so nothing is installed
            // before it. Sweeping first, though, is what stops a previous
            // search's landing pads from being measured as part of it.
            "baseline" => {
                self.sweep_landing_pads(ctx.dag).await;
                Ok(StepOutcome::Idle)
            }
            "converged" => Ok(StepOutcome::Idle),
            "searching" => {
                if let Some(in_flight) = state.in_flight.clone() {
                    // Proposed but never reported on -- a failed or cancelled
                    // run. Re-install it rather than scoring a run that never
                    // finished.
                    self.build_plan(ctx.dag, &state.top_candidates, &in_flight.modes)
                        .await?;
                    return Ok(StepOutcome::Trial {
                        label: Self::describe_plan(&state.top_candidates, &in_flight.modes),
                        budget_ms: self.budget(&state),
                        record: Box::new(self.outcome_from(&state)),
                    });
                }

                let plans = Self::plans(state.top_candidates.len());
                loop {
                    if state.plan_cursor >= plans.len() {
                        return self.promote(ctx, state).await;
                    }
                    let index = state.plan_cursor;
                    let modes = plans[index].clone();
                    state.plan_cursor += 1;

                    if modes == state.baseline_modes {
                        debug!("OMPPass: plan {} is the baseline, skipping", index + 1);
                        state.attempts.push((
                            format!(
                                "Plan {}: {}",
                                index + 1,
                                Self::describe_plan(&state.top_candidates, &modes)
                            ),
                            "skipped (same as baseline)".to_string(),
                            None,
                        ));
                        continue;
                    }

                    self.build_plan(ctx.dag, &state.top_candidates, &modes)
                        .await?;
                    state.in_flight = Some(OmpInFlight {
                        index,
                        modes: modes.clone(),
                    });
                    self.save_state(ctx.store, ctx.dag_id, &state).await?;
                    return Ok(StepOutcome::Trial {
                        label: Self::describe_plan(&state.top_candidates, &modes),
                        budget_ms: self.budget(&state),
                        record: Box::new(self.outcome_from(&state)),
                    });
                }
            }
            other => {
                debug!("OMPPass: unrecognized state '{other}'; leaving the DAG alone");
                Ok(StepOutcome::Idle)
            }
        }
    }

    /// The runtime past which a candidate can be abandoned: the best cost so
    /// far, since anything slower cannot win and its exact runtime is of no
    /// interest. `None` when early termination is off or nothing has been
    /// measured yet.
    fn budget(&self, state: &OmpState) -> Option<i64> {
        if !self.early_termination || state.best_cost == f64::MAX {
            return None;
        }
        Some(state.best_cost.round() as i64)
    }

    async fn promote(
        &mut self,
        ctx: &mut StepContext<'_, C, E>,
        mut state: OmpState,
    ) -> Result<StepOutcome, OptimizerError> {
        state.phase = "converged".to_string();
        state.in_flight = None;

        let empty = state.best_modes.is_empty() || state.best_modes == state.baseline_modes;
        if !empty {
            let top = state.top_candidates.clone();
            let modes = state.best_modes.clone();
            self.build_plan(ctx.dag, &top, &modes).await?;
        }

        // Whatever the last trial materialized -- landing pads included --
        // would otherwise persist and collide with the next run.
        self.sweep_landing_pads(ctx.dag).await;
        self.save_state(ctx.store, ctx.dag_id, &state).await?;
        self.remember_explain(&state);

        debug!(
            "OMPPass converged: {:.2}ms -> {:.2}ms over {} run(s)",
            state.baseline_cost, state.best_cost, state.runs_used
        );

        let record = Box::new(self.outcome_from(&state));
        if empty {
            Ok(StepOutcome::Done { record })
        } else {
            Ok(StepOutcome::Promote { record })
        }
    }

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
        let Some(stats) = run.stats.as_ref() else {
            return Ok(StepOutcome::Idle);
        };
        if !run.is_measured() {
            return Ok(StepOutcome::Idle);
        }

        let cost = stats.duration.num_milliseconds() as f64;

        if state.phase == "baseline" {
            state.baseline_cost = cost;
            state.best_cost = cost;
            state.runs_used = 1;
            state.candidates = self.rank_candidates(ctx.dag);
            state.top_candidates = {
                let iter = state.candidates.iter().rev();
                match self.top_n {
                    Some(n) => iter.take(n).map(|(id, _)| id.clone()).collect(),
                    None => iter.map(|(id, _)| id.clone()).collect(),
                }
            };
            state.baseline_modes = state
                .top_candidates
                .iter()
                .filter_map(|id| ctx.dag.nodes.get(id.clone()))
                .map(|n| n.materialize.as_str().to_string())
                .collect();
            state.best_modes = state.baseline_modes.clone();
            state.iterations.push(IterationStat {
                iteration: 1,
                runtime_ms: cost as i64,
                outcome: Some("baseline".to_string()),
                system_samples: if self.profile_iterations {
                    stats.system_samples.clone()
                } else {
                    Vec::new()
                },
                ..Default::default()
            });
            state.phase = "searching".to_string();

            debug!(
                "OMPPass baseline {cost:.2}ms; {} candidate node(s): {:?}; {} plan(s) to evaluate",
                state.top_candidates.len(),
                state.top_candidates,
                Self::plans(state.top_candidates.len()).len().saturating_sub(1),
            );
            self.record_trial(ctx.store, ctx.dag_id, &run.run_id, 0, "baseline", cost)
                .await?;
            self.save_state(ctx.store, ctx.dag_id, &state).await?;
            self.remember_explain(&state);
            return Ok(StepOutcome::Idle);
        }

        let Some(in_flight) = state.in_flight.take() else {
            // A run this search did not propose. It measured the committed
            // DAG, not a plan, so there is nothing to attribute.
            return Ok(StepOutcome::Idle);
        };

        state.runs_used += 1;
        let label = format!(
            "Plan {}: {}",
            in_flight.index + 1,
            Self::describe_plan(&state.top_candidates, &in_flight.modes)
        );
        state.iterations.push(IterationStat {
            iteration: state.iterations.len() + 1,
            runtime_ms: cost as i64,
            outcome: Some("ok".to_string()),
            system_samples: if self.profile_iterations {
                stats.system_samples.clone()
            } else {
                Vec::new()
            },
            ..Default::default()
        });
        state
            .attempts
            .push((label.clone(), format!("{cost:.2} ms"), Some(cost)));

        if cost < state.best_cost {
            debug!(
                "OMPPass: plan {} is new best: {cost:.2}ms (was {:.2}ms)",
                in_flight.index + 1,
                state.best_cost
            );
            state.best_cost = cost;
            state.best_modes = in_flight.modes.clone();
        }

        self.record_trial(
            ctx.store,
            ctx.dag_id,
            &run.run_id,
            in_flight.index,
            &in_flight.modes.join(","),
            cost,
        )
        .await?;
        self.save_state(ctx.store, ctx.dag_id, &state).await?;
        self.remember_explain(&state);
        Ok(StepOutcome::Idle)
    }

    async fn record_trial(
        &self,
        store: &dyn OptStore,
        dag_id: &str,
        run_id: &str,
        plan_index: usize,
        plan: &str,
        cost_ms: f64,
    ) -> Result<(), OptimizerError> {
        store
            .execute(
                &format!(
                    "INSERT INTO {TRIALS_TABLE} \
                     (dag_id, run_id, plan_index, plan, cost_ms, recorded_at) \
                     VALUES (?, ?, ?, ?, ?, now())"
                ),
                &[
                    json!(dag_id),
                    json!(run_id),
                    json!(plan_index),
                    json!(plan),
                    json!(cost_ms),
                ],
            )
            .await?;
        Ok(())
    }
}

#[async_trait]
impl<C, E> Optimization<C, E> for OMPPass<C, E>
where
    C: Connector + Send + 'static + Sync,
    E: Executor<C> + Send + Sync,
{
    fn name(&self) -> &'static str {
        "omp"
    }

    /// Like HMP, OMP decides by measurement -- it just measures every plan
    /// rather than searching heuristically.
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
                         plan_index  INTEGER NOT NULL,
                         plan        VARCHAR,
                         cost_ms     DOUBLE,
                         recorded_at TIMESTAMPTZ NOT NULL
                     )"
                ),
                &[],
            )
            .await?;

        if self.load_state(ctx.store, ctx.dag_id).await?.is_none() {
            self.save_state(ctx.store, ctx.dag_id, &OmpState::new())
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
        Some(("OMPPass".to_string(), self.explain_html()))
    }
}

impl<C, E> OMPPass<C, E>
where
    C: Connector + Send + 'static + Sync,
    E: Executor<C> + Send + Sync,
{
    fn explain_html(&self) -> String {
        let Some(data) = &self.explain_data else {
            return r#"<div class="panel"><p class="subtle">OMPPass did not run.</p></div>"#
                .to_string();
        };

        let change_pct = if data.baseline_cost > 0.0 {
            (data.best_cost - data.baseline_cost) / data.baseline_cost * 100.0
        } else {
            0.0
        };

        let centrality_label = match data.centrality {
            OMPCentrality::OutDegree => "out-degree",
            OMPCentrality::Paths => "paths-to-sinks",
        };

        let cards = render_card_grid(&[
            ("Baseline cost", format!("{:.2} ms", data.baseline_cost)),
            ("Best cost", format!("{:.2} ms", data.best_cost)),
            ("Change", format!("{change_pct:+.1}%")),
            ("Plans evaluated", data.attempts.len().to_string()),
            ("Materializations chosen", data.best_plan.len().to_string()),
        ]);

        let candidate_rows: Vec<Vec<String>> = data
            .candidates
            .iter()
            .rev()
            .map(|(id, rank)| {
                vec![
                    id.clone(),
                    rank.to_string(),
                    if data.top_candidates.contains(id) {
                        "yes".to_string()
                    } else {
                        "no".to_string()
                    },
                ]
            })
            .collect();
        let candidate_table = render_ranked_table(
            &["Node", centrality_label, "In working set"],
            &candidate_rows,
        );

        let max_cost = data
            .attempts
            .iter()
            .filter_map(|a| a.cost_ms)
            .fold(data.baseline_cost as f64, f64::max)
            .max(1.0);
        let attempt_bars: String = data
            .attempts
            .iter()
            .map(|a| {
                let label = if a.is_best {
                    format!("{} — chosen", a.label)
                } else {
                    a.label.clone()
                };
                render_bar_row(
                    &label,
                    &a.outcome,
                    a.cost_ms.unwrap_or(0.0) / max_cost * 100.0,
                )
            })
            .collect();

        format!(
            r##"<div class="section-stack">
        {cards}
        <div class="panel">
          <h2>Why these nodes were considered</h2>
          <div class="subtle">Only nodes with more than one downstream consumer (out-degree &gt; 1) and more than one downstream path to a TABLE/TEMP_TABLE node can benefit from materialization. Candidates are ranked by {centrality_label}; the working set is the top {top_n} of them.</div>
          {candidate_table}
        </div>
        <div class="panel">
          <h2>Plans evaluated</h2>
          <div class="subtle">Every View/TempTable assignment for the working set was tried (2^N combinations, baseline excluded). Early termination cancels a trial once it exceeds the current best runtime.</div>
          <div class="plan-tree">{attempt_bars}</div>
        </div>
      </div>"##,
            top_n = data.top_candidates.len(),
        )
    }
}
