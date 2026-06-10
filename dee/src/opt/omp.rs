use async_trait::async_trait;
use itertools::{Itertools, repeat_n};
use log::debug;
use std::{collections::HashMap, marker::PhantomData, sync::Arc};

use crate::{
    connectors::Connector,
    dag::MaterializeMode,
    executor::{Executor, ExecutorError},
    opt::{Dag, OptimizerError, OptimizerPass, common::make_temp, pushdown::PushdownPass},
};

#[derive(Debug, Clone, Copy, Default)]
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
    ) -> Self {
        Self {
            conn,
            engine,
            top_n,
            centrality,
            early_termination,
            use_pushdown,
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
    async fn run(&mut self, dag: &mut Dag) -> Result<HashMap<String, String>, OptimizerError> {
        debug!("Running OMPPass with centrality: {:?}", self.centrality);
        let mut stats = HashMap::new();

        // Run the baseline using the DAG's current materialization configuration.
        self.engine.cleanup(dag).await.unwrap();

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

        let baseline_cost = self
            .engine
            .run(dag)
            .await
            .map(|r| r.duration.num_milliseconds() as f32)
            .map_err(|e| OptimizerError::Exec(format!("baseline run failed: {e}")))?;

        // Rank nodes by the chosen centrality metric. Only nodes with more than
        // one downstream consumer benefit from materialization (out-degree > 1).
        let mut candidates: Vec<(String, usize)> = dag
            .nodes
            .nodes()
            .map(|n| {
                let rank = match self.centrality {
                    OMPCentrality::OutDegree => dag.nodes.out_degree(&n.id),
                    OMPCentrality::Paths => dag.nodes.paths_to_sinks(&n.id),
                };
                (n.id.clone(), rank)
            })
            .filter(|(_, d)| *d > 1)
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

        for (i, plan) in plans.iter().enumerate() {
            // The baseline combination was already measured above — skip it.
            if *plan == baseline_modes {
                debug!("OMPPass: plan {} is the baseline, skipping", i + 1);
                stats.insert(format!("attempt_{}", i + 1), "baseline(skipped)".to_string());
                continue;
            }

            // Clean up whatever the previous trial materialized, including any
            // lp_* nodes that make_temp inserted into last_run_dag.
            self.engine.cleanup(&last_run_dag).await.unwrap();

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

            let current_cost = if self.early_termination {
                let cancel_tx = self.engine.cancel_sender();
                cancel_tx.send(false).ok();
                let budget_ms = best_cost as u64;
                let cancel_tx_timer = Arc::clone(&cancel_tx);
                let timer = tokio::spawn(async move {
                    tokio::time::sleep(std::time::Duration::from_millis(budget_ms)).await;
                    cancel_tx_timer.send(true).ok();
                });
                let result = self.engine.run(&work_dag).await;
                timer.abort();
                match result {
                    Ok(r) => r.duration.num_milliseconds() as f32,
                    Err(ExecutorError::Cancelled) => {
                        debug!("OMPPass: plan {} cancelled after {}ms budget", i + 1, budget_ms);
                        stats.insert(
                            format!("attempt_{}", i + 1),
                            format!("cancelled({})", budget_ms),
                        );
                        last_run_dag = work_dag;
                        continue;
                    }
                    Err(e) => {
                        return Err(OptimizerError::Exec(format!("plan {} run failed: {e}", i + 1)));
                    }
                }
            } else {
                self.engine
                    .run(&work_dag)
                    .await
                    .map(|r| r.duration.num_milliseconds() as f32)
                    .map_err(|e| OptimizerError::Exec(format!("plan {} run failed: {e}", i + 1)))?
            };

            stats.insert(format!("attempt_{}", i + 1), current_cost.to_string());

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
        self.engine.cleanup(&last_run_dag).await.unwrap();

        let change = (best_cost - baseline_cost) / baseline_cost;
        debug!(
            "OMPPass: {:.2}ms -> {:.2}ms ({:.2}%)",
            baseline_cost,
            best_cost,
            change * 100.0,
        );

        stats.insert("baseline_value".into(), baseline_cost.to_string());
        stats.insert("best_value".into(), best_cost.to_string());
        stats.insert("opt_change".into(), change.to_string());

        let best_plan: Vec<String> = best_dag
            .nodes
            .nodes()
            .filter(|n| matches!(n.materialize, MaterializeMode::TempTable))
            .map(|n| n.id.clone())
            .collect();
        stats.insert("best_plan".into(), format!("{:?}", best_plan));

        *dag = best_dag;
        Ok(stats)
    }
}
