use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use itertools::{Itertools, repeat_n};
use log::debug;
use std::{marker::PhantomData, sync::Arc};

use crate::{
    connectors::Connector,
    dag::MaterializeMode,
    executor::{Executor, ExecutorError, ProfilingConfig, SimpleEngine},
    opt::{
        Dag, Explain, OptimizerError, OptimizerPass,
        common::make_temp,
        explain::{render_bar_row, render_card_grid, render_ranked_table},
        pushdown::PushdownPass,
        report::{CandidateScore, IterationStat, OmpDetail, PassDetail, PassOutcome},
    },
};

/// One materialization plan attempt, used by `Explain::explain`.
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
    /// Data collected during the last `run()`, used by `Explain::explain`.
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
            explain_data: None,
            _phantom: PhantomData,
        }
    }
}

#[async_trait]
impl<C, E> OptimizerPass<C, E> for OMPPass<C, E>
where
    C: Connector + Send + 'static + Sync,
    E: Executor<C> + Send + Sync,
{
    async fn run(&mut self, dag: &mut Dag) -> Result<PassOutcome, OptimizerError> {
        debug!("Running OMPPass with centrality: {:?}", self.centrality);

        // Measure with a dedicated engine so profiling can be enabled for
        // this pass's runs independent of whatever engine `self.engine`
        // (used only for the pushdown pass below) was built with.
        let mut engine = SimpleEngine::new(self.conn.clone())
            .map_err(|e| OptimizerError::Exec(e.to_string()))?;
        if self.profile_iterations {
            engine = engine.with_profiling(ProfilingConfig::default());
        }

        // Run the baseline using the DAG's current materialization configuration.
        engine.cleanup(dag).await.unwrap();

        // Pre-flight: drop any lp_* landing-pad artifacts left over from a
        // previous process invocation.  OMP adds lp_N nodes during evaluation
        // and cleans them up at the end, but an interrupted prior run may have
        // left stale TABLE entries in the DB.  Try dropping lp_0..lp_99 as
        // both TABLE and VIEW.  Missing entries are silently ignored.
        if let Some(prefix) = dag.nodes.nodes().next().map(|n| {
            let id = &n.id;
            id.rfind("\".\"")
                .map(|pos| format!("{}\".", &id[..pos]))
                .unwrap_or_default()
        }) {
            for i in 0..100 {
                let lp = if prefix.is_empty() {
                    format!("lp_{i}")
                } else {
                    format!("{prefix}\"lp_{i}\"")
                };
                self.conn
                    .drop_relation(MaterializeMode::Table, lp.clone())
                    .await
                    .ok();
                self.conn
                    .drop_relation(MaterializeMode::View, lp)
                    .await
                    .ok();
            }
        }

        let baseline_stats = engine
            .run(dag)
            .await
            .map_err(|e| OptimizerError::Exec(format!("baseline run failed: {e}")))?;
        let baseline_cost = baseline_stats.duration.num_milliseconds() as f32;

        // Only nodes with more than one downstream consumer (out-degree > 1)
        // AND more than one downstream path reaching a TABLE/TEMP_TABLE node
        // benefit from materialization — otherwise there's nothing to
        // deduplicate. Rank the survivors by the chosen centrality metric;
        // in Paths mode, these two checks are still the filter — paths-to-sinks
        // is only used a second time to break ties among qualifying nodes.
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

        let top_candidates: Vec<String> = {
            let iter = candidates.iter().rev();
            match self.top_n {
                Some(n) => iter.take(n).map(|(id, _)| id.clone()).collect(),
                None => iter.map(|(id, _)| id.clone()).collect(),
            }
        };

        debug!("OMPPass: {} candidate node(s): {:?}", top_candidates.len(), top_candidates);

        // Record the baseline materialization modes for the selected candidates
        // so we can skip re-running that combination.
        let baseline_modes: Vec<MaterializeMode> = top_candidates
            .iter()
            .map(|id| dag.nodes.get(id.clone()).unwrap().materialize.clone())
            .collect();

        let mut iterations: Vec<IterationStat> = vec![IterationStat {
            iteration: 1,
            runtime_ms: baseline_cost as i64,
            system_samples: if self.profile_iterations {
                baseline_stats.system_samples.clone()
            } else {
                Vec::new()
            },
            outcome: Some("baseline".to_string()),
            ..Default::default()
        }];

        let mut best_cost = baseline_cost;
        let mut best_dag = dag.clone();

        // Enumerate all 2^N combinations of View / TempTable for the candidates.
        let plans: Vec<Vec<MaterializeMode>> = repeat_n(
            [MaterializeMode::View, MaterializeMode::TempTable].into_iter(),
            top_candidates.len(),
        )
        .multi_cartesian_product()
        .collect();

        debug!("OMPPass: {} plan(s) to evaluate (baseline excluded)", plans.len().saturating_sub(1));

        // Track the DAG that ran most recently so cleanup covers any landing-pad
        // nodes (lp_*) that make_temp adds to work_dag but that are absent from
        // the original dag.  Starts as dag.clone() because the baseline run used
        // the original dag.
        let mut last_run_dag = dag.clone();
        let mut attempts: Vec<OMPAttempt> = Vec::new();
        let describe_plan = |plan: &[MaterializeMode]| -> String {
            top_candidates
                .iter()
                .zip(plan.iter())
                .map(|(id, mode)| format!("{id}={}", mode.as_str()))
                .collect::<Vec<_>>()
                .join(", ")
        };

        for (i, plan) in plans.iter().enumerate() {
            // The baseline combination was already measured above — skip it.
            if *plan == baseline_modes {
                debug!("OMPPass: plan {} is the baseline, skipping", i + 1);
                attempts.push(OMPAttempt {
                    label: format!("Plan {}: {}", i + 1, describe_plan(plan)),
                    outcome: "skipped (same as baseline)".to_string(),
                    cost_ms: None,
                    is_best: false,
                });
                continue;
            }

            // Clean up whatever the previous trial materialized, including any
            // lp_* nodes that make_temp inserted into last_run_dag.
            engine.cleanup(&last_run_dag).await.unwrap();

            // Build the candidate DAG for this combination.
            let mut work_dag = dag.clone();
            let mut lp_counter = 0;
            for (pos, mode) in plan.iter().enumerate() {
                match mode {
                    MaterializeMode::TempTable => {
                        make_temp(&mut work_dag, &top_candidates[pos], &mut lp_counter)?;
                    }
                    other => {
                        work_dag
                            .nodes
                            .get_mut(top_candidates[pos].clone())
                            .ok_or_else(|| OptimizerError::Exec("missing node".to_string()))?
                            .materialize = *other;
                    }
                }
            }

            if self.use_pushdown {
                let mut pushdown_pass = PushdownPass::new(self.conn.clone(), self.engine.clone());
                if let Err(e) = pushdown_pass.run(&mut work_dag).await {
                    debug!("OMPPass: pushdown failed for plan {}, continuing without it: {e}", i + 1);
                }
            }

            let (current_cost, current_samples) = if self.early_termination {
                let cancel_tx = engine.cancel_sender();
                cancel_tx.send(false).ok();
                let budget_ms = best_cost as u64;
                let cancel_tx_timer = Arc::clone(&cancel_tx);
                let timer = tokio::spawn(async move {
                    tokio::time::sleep(std::time::Duration::from_millis(budget_ms)).await;
                    cancel_tx_timer.send(true).ok();
                });
                let result = engine.run(&work_dag).await;
                timer.abort();
                match result {
                    Ok(r) => (r.duration.num_milliseconds() as f32, r.system_samples),
                    Err(ExecutorError::Cancelled) => {
                        debug!("OMPPass: plan {} cancelled after {}ms budget", i + 1, budget_ms);
                        // The trial was killed at budget_ms, so its true
                        // runtime is unknown beyond "at least budget_ms".
                        // Record that lower bound so it still shows up in
                        // the iteration series instead of vanishing.
                        iterations.push(IterationStat {
                            iteration: iterations.len() + 1,
                            runtime_ms: budget_ms as i64,
                            system_samples: Vec::new(),
                            outcome: Some("cancelled".to_string()),
                            ..Default::default()
                        });
                        attempts.push(OMPAttempt {
                            label: format!("Plan {}: {}", i + 1, describe_plan(plan)),
                            outcome: format!("cancelled (exceeded {budget_ms}ms budget)"),
                            cost_ms: Some(budget_ms as f64),
                            is_best: false,
                        });
                        last_run_dag = work_dag;
                        continue;
                    }
                    Err(e) => {
                        return Err(OptimizerError::Exec(format!("plan {} run failed: {e}", i + 1)));
                    }
                }
            } else {
                let r = engine
                    .run(&work_dag)
                    .await
                    .map_err(|e| OptimizerError::Exec(format!("plan {} run failed: {e}", i + 1)))?;
                (r.duration.num_milliseconds() as f32, r.system_samples)
            };

            iterations.push(IterationStat {
                iteration: iterations.len() + 1,
                runtime_ms: current_cost as i64,
                system_samples: current_samples,
                outcome: Some("ok".to_string()),
                ..Default::default()
            });
            attempts.push(OMPAttempt {
                label: format!("Plan {}: {}", i + 1, describe_plan(plan)),
                outcome: format!("{current_cost:.2} ms"),
                cost_ms: Some(current_cost as f64),
                is_best: false,
            });

            if current_cost < best_cost {
                debug!(
                    "OMPPass: plan {} is new best: {:.2}ms (was {:.2}ms)",
                    i + 1,
                    current_cost,
                    best_cost
                );
                best_cost = current_cost;
                best_dag = work_dag.clone();
            }

            last_run_dag = work_dag;
        }

        // Drop whatever the last trial materialized (including any lp_* landing
        // pads).  Without this, those tables would persist in the DB across
        // process invocations and cause "already exists" errors on the next run.
        engine.cleanup(&last_run_dag).await.unwrap();

        let change = (best_cost - baseline_cost) / baseline_cost;
        debug!(
            "OMPPass: {:.2}ms -> {:.2}ms ({:.2}%)",
            baseline_cost,
            best_cost,
            change * 100.0,
        );

        let best_plan: Vec<String> = best_dag
            .nodes
            .nodes()
            .filter(|n| matches!(n.materialize, MaterializeMode::TempTable))
            .map(|n| n.id.clone())
            .collect();

        let outcome = PassOutcome {
            // The baseline plus every trial that actually ran.
            dag_runs_used: iterations.len() as u32,
            changes_applied: best_plan.len() as u32,
            candidates_considered: iterations.len().saturating_sub(1) as u32,
            working_set_size: top_candidates.len() as u32,
            iterations: iterations.clone(),
            detail: PassDetail::Omp(OmpDetail {
                baseline_value: baseline_cost as f64,
                best_value: best_cost as f64,
                opt_change: change as f64,
                best_plan: best_plan.clone(),
                centrality: format!("{:?}", self.centrality),
                candidates_ranked: candidates
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
        };

        for attempt in attempts.iter_mut() {
            if let Some(cost) = attempt.cost_ms
                && (cost - best_cost as f64).abs() < 1e-6
            {
                attempt.is_best = true;
            }
        }

        self.explain_data = Some(OMPExplainData {
            baseline_cost,
            best_cost,
            centrality: self.centrality,
            candidates,
            top_candidates,
            best_plan,
            attempts,
        });

        *dag = best_dag;
        Ok(outcome)
    }
}

impl<C, E> Explain for OMPPass<C, E>
where
    C: Connector + Send + 'static + Sync,
    E: Executor<C> + Send + Sync,
{
    fn explain_label(&self) -> String {
        "OMPPass".to_string()
    }

    fn explain(&self) -> String {
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
