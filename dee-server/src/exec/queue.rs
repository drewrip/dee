//! The run queue: work the server has accepted but not yet started.
//!
//! `dee trigger` refuses when a DAG is busy, which is the right answer for an
//! operator and the wrong one for a benchmark. Measuring how a DAG adapts over
//! its own history means running it N times in succession, and the interesting
//! part is what happens *between* the runs -- an optimization landing, the
//! warehouse growing -- which is exactly what one group of N repetitions
//! cannot show, because a group pins one version and one driver.
//!
//! So a queue entry is an ordinary run group that was parked instead of
//! driven. This dispatcher is the only thing that starts one. It holds no
//! in-memory queue: the order is `(created_at, run_group_id)` in the store,
//! and the transition out of `pending` is a compare-and-set, so the queue
//! survives being read and written by request handlers at the same time.
//!
//! Serialization for a single DAG falls out of the existing one-job-per-DAG
//! rule rather than being enforced here: an entry whose DAG is busy is simply
//! passed over and reconsidered on the next wake-up.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::watch;

use crate::error::ServerError;
use crate::exec::driver;
use crate::state::AppState;
use crate::store::repo::{dags, runs};

/// How many entries one pass looks at. A pass is cheap, and anything past the
/// concurrency limit could not start on this pass anyway.
const SCAN_LIMIT: usize = 256;

pub struct Dispatcher {
    state: AppState,
}

impl Dispatcher {
    pub fn new(state: AppState) -> Self {
        Dispatcher { state }
    }

    /// Dispatch until shutdown.
    ///
    /// Woken by [`AppState::wake_queue`] -- which the driver calls as a group
    /// finishes -- so the next entry starts immediately rather than at the top
    /// of the next tick. The tick is the safety net: it catches a wake-up that
    /// raced an entry being enqueued, and it is what starts a queue whose DAG
    /// was freed by something that does not signal, such as a run reaped by
    /// the orphan sweep.
    pub async fn run_loop(self: Arc<Self>, tick: Duration, mut shutdown: watch::Receiver<bool>) {
        loop {
            tokio::select! {
                _ = self.state.queue_wake.notified() => {}
                _ = tokio::time::sleep(tick) => {}
                _ = shutdown.changed() => {}
            }
            if *shutdown.borrow() {
                log::info!("queue dispatcher stopping");
                break;
            }
            match self.dispatch_ready().await {
                Ok(started) if !started.is_empty() => {
                    log::info!("started {} queued run group(s)", started.len());
                }
                Ok(_) => {}
                Err(e) => log::error!("could not dispatch from the run queue: {e}"),
            }
        }
    }

    /// Start every entry whose turn has come. Returns the groups it started.
    pub async fn dispatch_ready(&self) -> Result<Vec<String>, ServerError> {
        let claimed = self.claim_ready().await?;
        // Spawned after the scan rather than inside it, and that ordering is
        // the whole of the per-DAG serialization. A driver runs concurrently
        // with the rest of the pass that started it, so a group that fails
        // fast used to be able to release its DAG at one of the loop's later
        // await points -- letting the next entry for the *same* DAG claim it
        // and start in the same pass. Nothing was ever run twice at once, so
        // it was safe; it just made "one entry per DAG at a time" true by
        // timing rather than by construction. Holding every claim across the
        // scan makes it true by construction, and costs no latency: a driver
        // calls `wake_queue` as it finishes, so the next entry starts on that
        // wake-up rather than waiting for a tick.
        for group_id in &claimed {
            tokio::spawn(driver::drive_group(self.state.clone(), group_id.clone()));
        }
        Ok(claimed)
    }

    /// Decide which entries start now and take ownership of them, without
    /// starting them.
    ///
    /// Separate from [`dispatch_ready`](Self::dispatch_ready) so the decision
    /// can be observed on its own. A caller that could only watch a finished
    /// pass could not tell "this entry was passed over because its DAG was
    /// busy" from "the entry ahead of it finished and this one took its turn",
    /// which are the same observation and different behaviours.
    async fn claim_ready(&self) -> Result<Vec<String>, ServerError> {
        let capacity = self.state.config.max_concurrent_runs.max(1);
        let mut started = Vec::new();

        for entry in runs::next_pending(&self.state.store, SCAN_LIMIT).await? {
            if self.state.runs.active_count().await >= capacity {
                break;
            }
            if runs::dispatch_blocker(&self.state.store, entry.dag_id.clone())
                .await?
                .is_some()
            {
                continue;
            }

            self.resolve_version(&entry).await?;

            // Compare-and-set first: whoever wins this owns the entry. A
            // concurrent `DELETE /v1/queue/{id}` either got there first, in
            // which case this is a no-op and the entry is already cancelled,
            // or it arrives after and finds nothing pending to cancel.
            if !runs::mark_dispatched(&self.state.store, entry.run_group_id.clone()).await? {
                continue;
            }

            if let Some(blocking) = self
                .state
                .runs
                .claim(&entry.dag_id, &entry.run_group_id)
                .await
            {
                // A manual trigger squeezed in between the blocker check and
                // here. The entry keeps its place; it starts on a later pass.
                log::debug!(
                    "queue entry {} yielded to {blocking}; still queued",
                    entry.run_group_id
                );
                runs::requeue(&self.state.store, entry.run_group_id.clone()).await?;
                continue;
            }

            log::info!(
                "dispatching queued run group {} for '{}'",
                entry.run_group_id,
                entry.dag_name
            );
            started.push(entry.run_group_id);
        }

        Ok(started)
    }

    /// Point an unpinned entry at whatever version is current now.
    ///
    /// An entry that named a version keeps it. One that did not was only ever
    /// asking for "the current version", and its turn -- not its submission --
    /// is when that question is finally being asked. This is what makes a
    /// queue of N runs show a DAG adapting: optimize halfway through and the
    /// remaining entries run the rewrite.
    async fn resolve_version(&self, entry: &runs::RunGroupRow) -> Result<(), ServerError> {
        if entry.pin_version {
            return Ok(());
        }
        let Some(dag) = dags::get_by_id(&self.state.store, entry.dag_id.clone()).await? else {
            return Ok(());
        };
        if dag.current_version == entry.dag_version {
            return Ok(());
        }
        runs::set_group_version(
            &self.state.store,
            entry.run_group_id.clone(),
            dag.current_version,
        )
        .await?;
        runs::log_event(
            &self.state.store,
            None,
            Some(entry.run_group_id.clone()),
            Some(entry.dag_id.clone()),
            "info",
            format!(
                "queued run moved from v{} to v{}, the current version at dispatch",
                entry.dag_version, dag.current_version
            ),
        )
        .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::config::ServerConfig;
    use crate::store::Store;
    use crate::store::repo::dags;
    use dee::file::DagFile;

    fn trivial_dag(query: &str) -> DagFile {
        // Nothing executes in these tests: no connection is registered, so a
        // dispatched group fails in its driver. What is under test is which
        // entry the queue picks and what it points at when it does.
        serde_json::from_str(&format!(
            r#"{{"nodes":[{{"id":"a","query_text":"{query}","depends_on":[],
                            "materialize":"view"}}],"sources":[]}}"#
        ))
        .unwrap()
    }

    async fn fixture() -> (Dispatcher, AppState, String) {
        let store = Store::open_temporary().unwrap();
        let state = AppState::new(store, ServerConfig::default(), "test-instance".into());
        let submitted = dags::submit(
            &state.store,
            dags::SubmitRequest {
                target: Some("wh".into()),
                ..dags::SubmitRequest::new("sales".into(), trivial_dag("select 1"), dags::Origin::Submitted)
            },
        )
        .await
        .unwrap();
        let dag_id = submitted.dag_id;
        (Dispatcher::new(state.clone()), state, dag_id)
    }

    async fn enqueue(state: &AppState, dag_id: &str, pin_version: bool) -> String {
        runs::create_group(
            &state.store,
            runs::RunRequest {
                dag_id: dag_id.to_string(),
                dag_version: 1,
                target: "wh".into(),
                trigger: "queue".into(),
                scheduled_for: None,
                warmups: 0,
                repetitions: 1,
                cleanup_before: true,
                collect_plans: false,
                sample_interval_ms: None,
                queued: true,
                pin_version,
            },
            "test-instance".into(),
        )
        .await
        .unwrap()
        .run_group_id
    }

    #[tokio::test]
    async fn test_one_dags_queue_starts_one_entry_at_a_time() {
        let (dispatcher, state, dag_id) = fixture().await;
        let first = enqueue(&state, &dag_id, true).await;
        let second = enqueue(&state, &dag_id, true).await;
        let third = enqueue(&state, &dag_id, true).await;

        // This is the whole contract of `dee queue add -n 3`: three entries,
        // one running, in submission order.
        //
        // Asserted against `claim_ready` rather than `dispatch_ready` on
        // purpose. Nothing runs in these tests -- no connection is registered,
        // so a driver fails immediately -- and a driver runs concurrently with
        // the pass that spawned it, so watching `dispatch_ready` means racing
        // the first entry's failure against the rest of the scan. Claiming is
        // the decision; spawning is what the decision authorizes.
        let started = dispatcher.claim_ready().await.unwrap();
        assert_eq!(started, vec![first.clone()], "the front of the queue, and only it");

        let waiting: Vec<String> = runs::next_pending(&state.store, 10)
            .await
            .unwrap()
            .into_iter()
            .map(|g| g.run_group_id)
            .collect();
        assert_eq!(waiting, vec![second.clone(), third]);

        // And the claim is what holds them back: while the first entry owns
        // the DAG, another pass finds nothing it may start.
        assert!(
            dispatcher.claim_ready().await.unwrap().is_empty(),
            "a second pass started an entry for a DAG that is already running one"
        );

        // Once the first entry is genuinely done, the next one's turn comes.
        // Both halves of "done" are needed: the in-memory claim *and* the
        // group's terminal status, since a dispatched group is a blocker in the
        // store whether or not this process still holds it -- which is what
        // stops a restarted server from starting a second run for a DAG whose
        // first one it has forgotten about.
        state.runs.finish(&first).await;
        assert!(
            dispatcher.claim_ready().await.unwrap().is_empty(),
            "releasing the in-memory claim alone unblocked the queue"
        );
        runs::finalize_group(&state.store, first.clone(), None).await.unwrap();
        assert_eq!(dispatcher.claim_ready().await.unwrap(), vec![second]);
    }

    #[tokio::test]
    async fn test_a_busy_dag_holds_its_whole_queue() {
        let (dispatcher, state, dag_id) = fixture().await;
        let first = enqueue(&state, &dag_id, true).await;

        // Something else -- a manual trigger, an optimization -- owns the DAG.
        state.runs.claim(&dag_id, "someone-else").await;
        runs::create_group(
            &state.store,
            runs::RunRequest {
                dag_id: dag_id.clone(),
                dag_version: 1,
                target: "wh".into(),
                trigger: "manual".into(),
                scheduled_for: None,
                warmups: 0,
                repetitions: 1,
                cleanup_before: true,
                collect_plans: false,
                sample_interval_ms: None,
                queued: false,
                pin_version: true,
            },
            "test-instance".into(),
        )
        .await
        .unwrap();

        assert!(dispatcher.dispatch_ready().await.unwrap().is_empty());
        let waiting = runs::next_pending(&state.store, 10).await.unwrap();
        assert_eq!(waiting.len(), 1);
        assert_eq!(waiting[0].run_group_id, first, "still first in line");
    }

    #[tokio::test]
    async fn test_entries_for_different_dags_do_not_wait_on_each_other() {
        let (dispatcher, state, dag_id) = fixture().await;
        let other = dags::submit(
            &state.store,
            dags::SubmitRequest {
                target: Some("wh".into()),
                ..dags::SubmitRequest::new("churn".into(), trivial_dag("select 2"), dags::Origin::Submitted)
            },
        )
        .await
        .unwrap()
        .dag_id;

        let a = enqueue(&state, &dag_id, true).await;
        let b = enqueue(&state, &other, true).await;

        let mut started = dispatcher.dispatch_ready().await.unwrap();
        started.sort();
        let mut expected = vec![a, b];
        expected.sort();
        assert_eq!(started, expected);
    }

    #[tokio::test]
    async fn test_an_unpinned_entry_runs_the_version_current_when_its_turn_comes() {
        let (dispatcher, state, dag_id) = fixture().await;
        let floating = enqueue(&state, &dag_id, false).await;
        let pinned = enqueue(&state, &dag_id, true).await;

        // Something rewrote the DAG while both entries were waiting -- an
        // optimization is the case this exists for.
        let v2 = dags::submit(
            &state.store,
            dags::SubmitRequest {
                target: Some("wh".into()),
                ..dags::SubmitRequest::new("sales".into(), trivial_dag("select 3"), dags::Origin::Submitted)
            },
        )
        .await
        .unwrap();
        assert_eq!(v2.version, 2);

        for id in [&floating, &pinned] {
            let entry = runs::get_group(&state.store, id.clone()).await.unwrap().unwrap();
            dispatcher.resolve_version(&entry).await.unwrap();
        }

        let moved = runs::get_group(&state.store, floating).await.unwrap().unwrap();
        assert_eq!(moved.dag_version, 2, "an unpinned entry follows the DAG");
        // The runs carry the version too; the benchmark reads it from there.
        let series = runs::runs_in_group(&state.store, moved.run_group_id).await.unwrap();
        assert!(series.iter().all(|r| r.dag_version == 2));

        let held = runs::get_group(&state.store, pinned).await.unwrap().unwrap();
        assert_eq!(held.dag_version, 1, "an explicit version is never moved");
    }

    #[tokio::test]
    async fn test_dispatching_never_exceeds_the_concurrency_limit() {
        let store = Store::open_temporary().unwrap();
        let config = ServerConfig {
            max_concurrent_runs: 1,
            ..ServerConfig::default()
        };
        let state = AppState::new(store, config, "test-instance".into());

        let mut dag_ids = Vec::new();
        for name in ["a", "b", "c"] {
            let submitted = dags::submit(
                &state.store,
                dags::SubmitRequest {
                    target: Some("wh".into()),
                    ..dags::SubmitRequest::new(
                        name.into(),
                        trivial_dag(&format!("select '{name}'")),
                        dags::Origin::Submitted,
                    )
                },
            )
            .await
            .unwrap();
            dag_ids.push(submitted.dag_id);
        }
        for dag_id in &dag_ids {
            enqueue(&state, dag_id, true).await;
        }

        let dispatcher = Dispatcher::new(state.clone());
        // Three free DAGs, one slot: the other two stay queued rather than
        // opening three concurrent workloads against the warehouses.
        assert_eq!(dispatcher.dispatch_ready().await.unwrap().len(), 1);
        assert_eq!(runs::next_pending(&state.store, 10).await.unwrap().len(), 2);
    }
}
