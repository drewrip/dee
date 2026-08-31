//! The scheduler.
//!
//! One loop, ticking on an interval, asking "what is due?" and starting it.
//! All the logic lives in [`Scheduler::tick_once`], which takes `now` as an
//! argument -- that is what makes a week of schedule behaviour testable in
//! microseconds instead of requiring a week.
//!
//! **No catchup.** The next firing is always computed from *now*, never from
//! the window that was missed. If the server is down from 02:00 to 09:00 on an
//! hourly schedule, it fires once for the current window on return and jumps
//! to 10:00 -- it does not replay the seven windows it slept through. The gap
//! is recorded as a `missed_window` skip so it is visible rather than silent.

use std::sync::Arc;

use chrono::{DateTime, Utc};

use crate::error::ServerError;
use crate::exec::driver;
use crate::sched::clock::Clock;
use crate::sched::cron::CronSpec;
use crate::state::AppState;
use crate::store::repo::{dags, runs, schedules};

/// What a tick did, for logging and for tests.
#[derive(Debug, Default)]
pub struct TickOutcome {
    pub fired: Vec<Fired>,
    pub skipped: Vec<Skipped>,
}

#[derive(Debug)]
pub struct Fired {
    pub dag_name: String,
    pub run_group_id: String,
    pub scheduled_for: DateTime<Utc>,
}

#[derive(Debug)]
pub struct Skipped {
    pub dag_name: String,
    pub reason: &'static str,
    pub scheduled_for: DateTime<Utc>,
    pub windows_skipped: i32,
    pub blocking_job_id: Option<String>,
}

pub struct Scheduler {
    state: AppState,
    clock: Arc<dyn Clock>,
}

impl Scheduler {
    pub fn new(state: AppState, clock: Arc<dyn Clock>) -> Self {
        Scheduler { state, clock }
    }

    /// Recompute every enabled schedule's next firing from `now`.
    ///
    /// Run at startup. Any schedule whose next firing is in the past belongs to
    /// a window that elapsed while this server was not running; it is recorded
    /// as missed and moved forward, never replayed.
    pub async fn realign_after_downtime(&self) -> Result<usize, ServerError> {
        let now = self.clock.now();
        let mut realigned = 0;

        for schedule in schedules::enabled(&self.state.store).await? {
            let spec = match self.spec_for(&schedule) {
                Some(spec) => spec,
                None => continue,
            };
            let Some(stale) = schedule.next_fire_at else {
                schedules::advance(
                    &self.state.store,
                    schedule.dag_id.clone(),
                    spec.next_after(now),
                    None,
                )
                .await?;
                continue;
            };
            if stale > now {
                continue;
            }

            let missed = spec.count_between(stale - chrono::Duration::seconds(1), now, 10_000);
            if missed > 0 {
                schedules::record_skip(
                    &self.state.store,
                    schedule.dag_id.clone(),
                    stale,
                    schedules::reason::MISSED_WINDOW,
                    None,
                    missed as i32,
                    Some("the server was not running for these windows".into()),
                )
                .await?;
                log::warn!(
                    "'{}' missed {missed} scheduled window(s) while the server was down",
                    schedule.dag_name
                );
            }
            schedules::advance(
                &self.state.store,
                schedule.dag_id.clone(),
                spec.next_after(now),
                None,
            )
            .await?;
            realigned += 1;
        }
        Ok(realigned)
    }

    /// Fire every schedule due at `now`.
    pub async fn tick_once(&self, now: DateTime<Utc>) -> Result<TickOutcome, ServerError> {
        let mut outcome = TickOutcome::default();

        for schedule in schedules::enabled(&self.state.store).await? {
            let Some(due_at) = schedule.next_fire_at else {
                continue;
            };
            if due_at > now {
                continue;
            }
            let Some(spec) = self.spec_for(&schedule) else {
                continue;
            };

            // Always computed from `now`. This single choice is the whole
            // catchup-free contract.
            let advanced = spec.next_after(now);

            // More than one window between the due one and now means the
            // scheduler itself fell behind -- a long tick, or a paused server.
            let extra = spec.count_between(due_at, now, 10_000);
            if extra > 0 {
                schedules::record_skip(
                    &self.state.store,
                    schedule.dag_id.clone(),
                    due_at,
                    schedules::reason::MISSED_WINDOW,
                    None,
                    extra as i32,
                    Some("more than one window elapsed before this tick".into()),
                )
                .await?;
                outcome.skipped.push(Skipped {
                    dag_name: schedule.dag_name.clone(),
                    reason: schedules::reason::MISSED_WINDOW,
                    scheduled_for: due_at,
                    windows_skipped: extra as i32,
                    blocking_job_id: None,
                });
            }

            match self.fire(&schedule, due_at).await {
                Ok(Some(run_group_id)) => {
                    outcome.fired.push(Fired {
                        dag_name: schedule.dag_name.clone(),
                        run_group_id,
                        scheduled_for: due_at,
                    });
                    schedules::advance(
                        &self.state.store,
                        schedule.dag_id.clone(),
                        advanced,
                        Some(now),
                    )
                    .await?;
                }
                Ok(None) => {
                    // `fire` recorded why. The schedule still advances: a DAG
                    // that is busy now should be tried at its next window, not
                    // retried in a tight loop.
                    schedules::advance(
                        &self.state.store,
                        schedule.dag_id.clone(),
                        advanced,
                        Some(now),
                    )
                    .await?;
                    if let Some(last) = self.last_skip(&schedule).await {
                        outcome.skipped.push(last);
                    }
                }
                Err(e) => {
                    log::error!("could not fire '{}': {e}", schedule.dag_name);
                    schedules::record_skip(
                        &self.state.store,
                        schedule.dag_id.clone(),
                        due_at,
                        schedules::reason::ERROR,
                        None,
                        1,
                        Some(e.to_string()),
                    )
                    .await?;
                    outcome.skipped.push(Skipped {
                        dag_name: schedule.dag_name.clone(),
                        reason: schedules::reason::ERROR,
                        scheduled_for: due_at,
                        windows_skipped: 1,
                        blocking_job_id: None,
                    });
                    schedules::advance(
                        &self.state.store,
                        schedule.dag_id.clone(),
                        advanced,
                        Some(now),
                    )
                    .await?;
                }
            }
        }

        Ok(outcome)
    }

    /// Start one scheduled run. `Ok(None)` means the window was skipped.
    async fn fire(
        &self,
        schedule: &schedules::ScheduleRow,
        scheduled_for: DateTime<Utc>,
    ) -> Result<Option<String>, ServerError> {
        let dag = dags::get(&self.state.store, schedule.dag_name.clone())
            .await?
            .ok_or_else(|| ServerError::NotFound("dag", schedule.dag_name.clone()))?;

        let Some(target) = schedule.target.clone().or_else(|| dag.default_target.clone()) else {
            schedules::record_skip(
                &self.state.store,
                schedule.dag_id.clone(),
                scheduled_for,
                schedules::reason::NO_TARGET,
                None,
                1,
                Some("neither the schedule nor the dag names a connection".into()),
            )
            .await?;
            return Ok(None);
        };

        // Ask the store as well as the in-memory claim: the store is what a
        // human sees, and it also covers an optimization started before this
        // server took over.
        if let Some(blocking) = runs::active_job(&self.state.store, dag.dag_id.clone()).await? {
            schedules::record_skip(
                &self.state.store,
                schedule.dag_id.clone(),
                scheduled_for,
                schedules::reason::OVERLAP,
                Some(blocking.clone()),
                1,
                Some("the previous job for this dag was still running".into()),
            )
            .await?;
            log::info!(
                "skipped '{}' at {scheduled_for}: {blocking} is still running",
                schedule.dag_name
            );
            return Ok(None);
        }

        let request = runs::RunRequest {
            dag_id: dag.dag_id.clone(),
            dag_version: dag.current_version,
            target,
            trigger: "schedule".into(),
            scheduled_for: Some(scheduled_for),
            warmups: 0,
            repetitions: 1,
            cleanup_before: true,
            collect_plans: false,
            sample_interval_ms: None,
            // A scheduled fire is driven immediately or skipped; it never
            // waits in the queue, because a window that has passed is not
            // worth running late.
            queued: false,
            pin_version: true,
        };
        let created =
            runs::create_group(&self.state.store, request, self.state.instance_id.clone()).await?;

        // Claim after creating the group so the claim and the row agree. Losing
        // here means a manual trigger won the race in between.
        if let Some(blocking) = self
            .state
            .runs
            .claim(&dag.dag_id, &created.run_group_id)
            .await
        {
            runs::finalize_group(
                &self.state.store,
                created.run_group_id,
                Some("another job started first".into()),
            )
            .await?;
            schedules::record_skip(
                &self.state.store,
                schedule.dag_id.clone(),
                scheduled_for,
                schedules::reason::OVERLAP,
                Some(blocking),
                1,
                Some("another job for this dag started first".into()),
            )
            .await?;
            return Ok(None);
        }

        tokio::spawn(driver::drive_group(
            self.state.clone(),
            created.run_group_id.clone(),
        ));
        Ok(Some(created.run_group_id))
    }

    fn spec_for(&self, schedule: &schedules::ScheduleRow) -> Option<CronSpec> {
        match CronSpec::parse(&schedule.cron, &schedule.timezone) {
            Ok(spec) => Some(spec),
            Err(e) => {
                // Expressions are validated when set, so this means the row was
                // edited outside the API, or a timezone database changed.
                log::error!(
                    "'{}' has an unusable schedule and will not fire: {e}",
                    schedule.dag_name
                );
                None
            }
        }
    }

    async fn last_skip(&self, schedule: &schedules::ScheduleRow) -> Option<Skipped> {
        let rows = schedules::skips(&self.state.store, Some(schedule.dag_name.clone()), 1)
            .await
            .ok()?;
        rows.into_iter().next().map(|row| Skipped {
            dag_name: row.dag_name,
            reason: match row.reason.as_str() {
                schedules::reason::OVERLAP => schedules::reason::OVERLAP,
                schedules::reason::NO_TARGET => schedules::reason::NO_TARGET,
                schedules::reason::MISSED_WINDOW => schedules::reason::MISSED_WINDOW,
                _ => schedules::reason::ERROR,
            },
            scheduled_for: row.scheduled_for,
            windows_skipped: row.windows_skipped,
            blocking_job_id: row.blocking_run_id,
        })
    }

    /// Tick until `shutdown` fires.
    pub async fn run_loop(
        self: Arc<Self>,
        interval: std::time::Duration,
        mut shutdown: tokio::sync::watch::Receiver<bool>,
    ) {
        let mut ticker = tokio::time::interval(interval);
        // A tick delayed by a long store write should not then fire a burst to
        // catch up; the schedules themselves already carry that information.
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

        loop {
            tokio::select! {
                _ = ticker.tick() => {
                    let now = self.clock.now();
                    match self.tick_once(now).await {
                        Ok(outcome) => {
                            for fired in &outcome.fired {
                                log::info!(
                                    "scheduled '{}' for {} as {}",
                                    fired.dag_name, fired.scheduled_for, fired.run_group_id
                                );
                            }
                        }
                        Err(e) => log::error!("scheduler tick failed: {e}"),
                    }
                }
                _ = shutdown.changed() => {
                    if *shutdown.borrow() {
                        log::info!("scheduler stopping");
                        return;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ServerConfig;
    use crate::sched::clock::MockClock;
    use crate::store::Store;
    use dee::file::DagFile;

    fn utc(iso: &str) -> DateTime<Utc> {
        iso.parse().unwrap()
    }

    fn trivial_dag() -> DagFile {
        // No target is registered in these tests, so nothing actually executes;
        // what is under test is which windows fire and which are skipped.
        serde_json::from_str(
            r#"{"nodes":[{"id":"a","query_text":"select 1","depends_on":[],
                          "materialize":"view"}],"sources":[]}"#,
        )
        .unwrap()
    }

    async fn fixture(clock: Arc<MockClock>) -> (Scheduler, AppState, String) {
        let store = Store::open_temporary().unwrap();
        let state = AppState::new(store, ServerConfig::default(), "test-instance".into());

        let submitted = dags::submit(
            &state.store,
            dags::SubmitRequest {
                target: Some("wh".into()),
                ..dags::SubmitRequest::new("sales".into(), trivial_dag(), dags::Origin::Submitted)
            },
        )
        .await
        .unwrap();

        let scheduler = Scheduler::new(state.clone(), clock);
        (scheduler, state, submitted.dag_id)
    }

    async fn set_schedule(
        state: &AppState,
        dag_id: &str,
        cron: &str,
        next_fire_at: Option<DateTime<Utc>>,
    ) {
        schedules::upsert(
            &state.store,
            dag_id.to_string(),
            cron.to_string(),
            "UTC".into(),
            true,
            None,
            next_fire_at,
        )
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_nothing_fires_before_a_schedule_is_due() {
        let clock = Arc::new(MockClock::at("2026-01-01T00:00:00Z"));
        let (scheduler, state, dag_id) = fixture(clock.clone()).await;
        set_schedule(&state, &dag_id, "0 * * * *", Some(utc("2026-01-01T01:00:00Z"))).await;

        let outcome = scheduler.tick_once(clock.now()).await.unwrap();
        assert!(outcome.fired.is_empty());
        assert!(outcome.skipped.is_empty());
    }

    #[tokio::test]
    async fn test_a_due_window_fires_once_and_advances() {
        let clock = Arc::new(MockClock::at("2026-01-01T01:00:00Z"));
        let (scheduler, state, dag_id) = fixture(clock.clone()).await;
        set_schedule(&state, &dag_id, "0 * * * *", Some(utc("2026-01-01T01:00:00Z"))).await;

        let outcome = scheduler.tick_once(clock.now()).await.unwrap();
        assert_eq!(outcome.fired.len(), 1);
        assert_eq!(outcome.fired[0].scheduled_for, utc("2026-01-01T01:00:00Z"));

        // Ticking again at the same instant must not re-fire the same window.
        let again = scheduler.tick_once(clock.now()).await.unwrap();
        assert!(again.fired.is_empty());

        let schedule = schedules::get(&state.store, dag_id).await.unwrap().unwrap();
        assert_eq!(schedule.next_fire_at, Some(utc("2026-01-01T02:00:00Z")));
    }

    #[tokio::test]
    async fn test_downtime_fires_once_and_records_the_gap_rather_than_catching_up() {
        // The catchup-free contract: down 02:00-09:00 on an hourly schedule
        // means one run on return, not seven.
        let clock = Arc::new(MockClock::at("2026-01-01T09:00:00Z"));
        let (scheduler, state, dag_id) = fixture(clock.clone()).await;
        set_schedule(&state, &dag_id, "0 * * * *", Some(utc("2026-01-01T02:00:00Z"))).await;

        let outcome = scheduler.tick_once(clock.now()).await.unwrap();

        assert_eq!(outcome.fired.len(), 1, "exactly one run, not a backlog");
        let missed: Vec<_> = outcome
            .skipped
            .iter()
            .filter(|s| s.reason == schedules::reason::MISSED_WINDOW)
            .collect();
        assert_eq!(missed.len(), 1);
        assert_eq!(missed[0].windows_skipped, 7, "03:00 through 09:00");

        // And it jumps forward from now, not from the window it missed.
        let schedule = schedules::get(&state.store, dag_id).await.unwrap().unwrap();
        assert_eq!(schedule.next_fire_at, Some(utc("2026-01-01T10:00:00Z")));
    }

    #[tokio::test]
    async fn test_a_busy_dag_skips_its_window_with_the_blocking_job_named() {
        let clock = Arc::new(MockClock::at("2026-01-01T01:00:00Z"));
        let (scheduler, state, dag_id) = fixture(clock.clone()).await;
        set_schedule(&state, &dag_id, "0 * * * *", Some(utc("2026-01-01T01:00:00Z"))).await;

        // Something is already running for this DAG.
        let blocking = runs::create_group(
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

        let outcome = scheduler.tick_once(clock.now()).await.unwrap();

        assert!(outcome.fired.is_empty(), "a busy dag must not start a second run");
        assert_eq!(outcome.skipped.len(), 1);
        assert_eq!(outcome.skipped[0].reason, schedules::reason::OVERLAP);
        assert_eq!(
            outcome.skipped[0].blocking_job_id.as_deref(),
            Some(blocking.run_group_id.as_str()),
            "the skip must name what blocked it"
        );

        // The schedule still moves on: a busy DAG is retried at its next
        // window, not spun on.
        let schedule = schedules::get(&state.store, dag_id).await.unwrap().unwrap();
        assert_eq!(schedule.next_fire_at, Some(utc("2026-01-01T02:00:00Z")));
    }

    #[tokio::test]
    async fn test_an_optimization_blocks_a_scheduled_window() {
        // An optimization runs the DAG against the same warehouse, so it has
        // to block exactly as a run does.
        let clock = Arc::new(MockClock::at("2026-01-01T01:00:00Z"));
        let (scheduler, state, dag_id) = fixture(clock.clone()).await;
        set_schedule(&state, &dag_id, "0 * * * *", Some(utc("2026-01-01T01:00:00Z"))).await;

        state
            .store
            .write({
                let dag_id = dag_id.clone();
                move |c| {
                    c.execute(
                        "INSERT INTO optimizations (optimization_id, dag_id, source_version,
                                                    target, status, config, instance_id)
                         VALUES ('opt-1', ?, 1, 'wh', 'running', '{}', 'i')",
                        duckdb::params![dag_id],
                    )?;
                    Ok(())
                }
            })
            .await
            .unwrap();

        let outcome = scheduler.tick_once(clock.now()).await.unwrap();
        assert!(outcome.fired.is_empty());
        assert_eq!(outcome.skipped[0].blocking_job_id.as_deref(), Some("opt-1"));
    }

    #[tokio::test]
    async fn test_a_paused_schedule_never_fires() {
        let clock = Arc::new(MockClock::at("2026-01-01T05:00:00Z"));
        let (scheduler, state, dag_id) = fixture(clock.clone()).await;
        set_schedule(&state, &dag_id, "0 * * * *", Some(utc("2026-01-01T01:00:00Z"))).await;
        schedules::set_enabled(&state.store, dag_id, false, None)
            .await
            .unwrap();

        let outcome = scheduler.tick_once(clock.now()).await.unwrap();
        assert!(outcome.fired.is_empty());
        assert!(outcome.skipped.is_empty());
    }

    #[tokio::test]
    async fn test_a_dag_with_no_target_skips_rather_than_failing_the_tick() {
        let clock = Arc::new(MockClock::at("2026-01-01T01:00:00Z"));
        let (scheduler, state, dag_id) = fixture(clock.clone()).await;
        state
            .store
            .write(|c| {
                c.execute("UPDATE dags SET default_target = NULL", [])?;
                Ok(())
            })
            .await
            .unwrap();
        set_schedule(&state, &dag_id, "0 * * * *", Some(utc("2026-01-01T01:00:00Z"))).await;

        let outcome = scheduler.tick_once(clock.now()).await.unwrap();
        assert!(outcome.fired.is_empty());
        assert_eq!(outcome.skipped[0].reason, schedules::reason::NO_TARGET);
    }

    #[tokio::test]
    async fn test_realign_after_downtime_moves_stale_schedules_forward() {
        // What a restart does: never replay, always record the gap.
        let clock = Arc::new(MockClock::at("2026-01-02T00:00:00Z"));
        let (scheduler, state, dag_id) = fixture(clock.clone()).await;
        set_schedule(&state, &dag_id, "0 * * * *", Some(utc("2026-01-01T00:00:00Z"))).await;

        assert_eq!(scheduler.realign_after_downtime().await.unwrap(), 1);

        let schedule = schedules::get(&state.store, dag_id).await.unwrap().unwrap();
        assert_eq!(schedule.next_fire_at, Some(utc("2026-01-02T01:00:00Z")));

        let skips = schedules::skips(&state.store, Some("sales".into()), 10)
            .await
            .unwrap();
        assert_eq!(skips.len(), 1);
        assert_eq!(skips[0].reason, schedules::reason::MISSED_WINDOW);
        // 24 hourly windows across the day, plus the one falling exactly at
        // the restart instant -- realign declines to replay that too, since a
        // window due the moment we came back is still history.
        assert_eq!(skips[0].windows_skipped, 25);

        // And after realigning, an immediate tick fires nothing.
        assert!(scheduler.tick_once(clock.now()).await.unwrap().fired.is_empty());
    }

    #[tokio::test]
    async fn test_realign_leaves_a_future_schedule_alone() {
        let clock = Arc::new(MockClock::at("2026-01-01T00:00:00Z"));
        let (scheduler, state, dag_id) = fixture(clock.clone()).await;
        set_schedule(&state, &dag_id, "0 * * * *", Some(utc("2026-01-01T05:00:00Z"))).await;

        assert_eq!(scheduler.realign_after_downtime().await.unwrap(), 0);
        let schedule = schedules::get(&state.store, dag_id).await.unwrap().unwrap();
        assert_eq!(schedule.next_fire_at, Some(utc("2026-01-01T05:00:00Z")));
        assert!(schedules::skips(&state.store, None, 10).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_an_unparseable_schedule_is_skipped_without_stopping_the_tick() {
        // Expressions are validated when set, so this means the row was edited
        // outside the API. One bad schedule must not stop every other one.
        let clock = Arc::new(MockClock::at("2026-01-01T01:00:00Z"));
        let (scheduler, state, dag_id) = fixture(clock.clone()).await;
        set_schedule(&state, &dag_id, "0 * * * *", Some(utc("2026-01-01T01:00:00Z"))).await;
        state
            .store
            .write(|c| {
                c.execute("UPDATE schedules SET cron = 'nonsense'", [])?;
                Ok(())
            })
            .await
            .unwrap();

        let outcome = scheduler.tick_once(clock.now()).await.unwrap();
        assert!(outcome.fired.is_empty());
        let _ = dag_id;
    }

    #[tokio::test]
    async fn test_several_windows_over_time_fire_once_each() {
        // Walk a clock through a day and confirm one run per window.
        let clock = Arc::new(MockClock::at("2026-01-01T00:00:00Z"));
        let (scheduler, state, dag_id) = fixture(clock.clone()).await;
        set_schedule(&state, &dag_id, "0 * * * *", Some(utc("2026-01-01T01:00:00Z"))).await;

        let mut fired = 0;
        for _ in 0..6 {
            clock.advance(chrono::Duration::hours(1));
            let outcome = scheduler.tick_once(clock.now()).await.unwrap();
            fired += outcome.fired.len();
            // Each fired run has to finish before the next window, or the
            // overlap policy would (correctly) skip it.
            for f in &outcome.fired {
                runs::finalize_group(&state.store, f.run_group_id.clone(), None)
                    .await
                    .unwrap();
                state.runs.finish(&f.run_group_id).await;
            }
        }
        assert_eq!(fired, 6, "one run per hourly window");
    }
}
