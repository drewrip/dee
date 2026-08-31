//! Tracks the work this server currently owns.
//!
//! Two jobs. It enforces one active job per DAG -- the skip policy the
//! scheduler relies on and manual triggers respect -- and it holds the cancel
//! handles, so an HTTP request can stop a run some other task is driving.
//!
//! The in-memory map is authoritative because this server is the only executor
//! of its own work, and the startup orphan sweep makes that true again across
//! restarts.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::{Mutex, Semaphore};

use crate::exec::driver::CancelHandle;

struct Active {
    dag_id: String,
    cancel: Option<CancelHandle>,
}

pub struct RunManager {
    active: Mutex<HashMap<String, Active>>,
    /// Caps DAG runs executing at once across all DAGs, so a burst of due
    /// schedules cannot open unbounded work against the warehouses.
    pub semaphore: Arc<Semaphore>,
}

impl RunManager {
    pub fn new(max_concurrent_runs: usize) -> Self {
        RunManager {
            active: Mutex::new(HashMap::new()),
            semaphore: Arc::new(Semaphore::new(max_concurrent_runs.max(1))),
        }
    }

    /// Claim `dag_id` for `job_id`.
    ///
    /// Returns the id of the job already holding it, if any. Claiming happens
    /// under the same lock as the check, so two triggers arriving together
    /// cannot both win.
    pub async fn claim(&self, dag_id: &str, job_id: &str) -> Option<String> {
        let mut active = self.active.lock().await;
        if let Some((existing, _)) = active.iter().find(|(_, a)| a.dag_id == dag_id) {
            return Some(existing.clone());
        }
        active.insert(
            job_id.to_string(),
            Active {
                dag_id: dag_id.to_string(),
                cancel: None,
            },
        );
        None
    }

    pub async fn register_cancel(&self, job_id: &str, cancel: CancelHandle) {
        if let Some(entry) = self.active.lock().await.get_mut(job_id) {
            entry.cancel = Some(cancel);
        }
    }

    pub async fn finish(&self, job_id: &str) {
        self.active.lock().await.remove(job_id);
    }

    /// Ask a job to stop. Returns false if it is not running here.
    ///
    /// Cancellation is only observed between node dispatches, so this requests
    /// a stop rather than performing one.
    pub async fn cancel(&self, job_id: &str) -> bool {
        let active = self.active.lock().await;
        match active.get(job_id).and_then(|a| a.cancel.as_ref()) {
            Some(cancel) => {
                let _ = cancel.send(true);
                true
            }
            // Claimed but not yet driving: there is nothing to signal, and the
            // driver checks the flag before its first repetition anyway.
            None => active.contains_key(job_id),
        }
    }

    /// Signal every active job, for shutdown.
    pub async fn cancel_all(&self) -> usize {
        let active = self.active.lock().await;
        let mut signalled = 0;
        for entry in active.values() {
            if let Some(cancel) = &entry.cancel {
                let _ = cancel.send(true);
                signalled += 1;
            }
        }
        signalled
    }

    pub async fn active_count(&self) -> usize {
        self.active.lock().await.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_one_job_per_dag() {
        let manager = RunManager::new(4);
        assert!(manager.claim("dag-a", "job-1").await.is_none());

        // The skip policy: a second job for the same DAG is told what blocks it.
        assert_eq!(manager.claim("dag-a", "job-2").await.as_deref(), Some("job-1"));
        // A different DAG is unaffected.
        assert!(manager.claim("dag-b", "job-3").await.is_none());
    }

    #[tokio::test]
    async fn test_finishing_releases_the_dag() {
        let manager = RunManager::new(4);
        manager.claim("dag-a", "job-1").await;
        manager.finish("job-1").await;

        assert!(manager.claim("dag-a", "job-2").await.is_none());
        assert_eq!(manager.active_count().await, 1);
    }

    #[tokio::test]
    async fn test_concurrent_claims_produce_exactly_one_winner() {
        let manager = Arc::new(RunManager::new(4));
        let mut handles = Vec::new();
        for i in 0..16 {
            let manager = manager.clone();
            handles.push(tokio::spawn(async move {
                manager.claim("dag-a", &format!("job-{i}")).await.is_none()
            }));
        }
        let mut winners = 0;
        for h in handles {
            if h.await.unwrap() {
                winners += 1;
            }
        }
        assert_eq!(winners, 1, "the claim check and insert must be atomic");
    }

    #[tokio::test]
    async fn test_cancel_reports_whether_the_job_is_known() {
        let manager = RunManager::new(4);
        manager.claim("dag-a", "job-1").await;

        // Claimed but not yet driving: still a job this server owns.
        assert!(manager.cancel("job-1").await);
        assert!(!manager.cancel("job-unknown").await);

        let (tx, rx) = tokio::sync::watch::channel(false);
        manager.register_cancel("job-1", Arc::new(tx)).await;
        assert!(manager.cancel("job-1").await);
        assert!(*rx.borrow(), "the engine's cancel flag was not raised");
    }

    #[tokio::test]
    async fn test_cancel_all_signals_every_driving_job() {
        let manager = RunManager::new(4);
        let mut receivers = Vec::new();
        for (i, dag) in ["a", "b", "c"].iter().enumerate() {
            let job = format!("job-{i}");
            manager.claim(dag, &job).await;
            let (tx, rx) = tokio::sync::watch::channel(false);
            manager.register_cancel(&job, Arc::new(tx)).await;
            receivers.push(rx);
        }

        assert_eq!(manager.cancel_all().await, 3);
        assert!(receivers.iter().all(|r| *r.borrow()));
    }
}
