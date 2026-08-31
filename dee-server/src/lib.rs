//! The dee server: a daemon that owns a DAG registry, a schedule, and the
//! history of everything it has run.

pub mod api;
pub mod config;
pub mod error;
pub mod exec;
pub mod hash;
pub mod sched;
pub mod state;
pub mod store;

use std::future::Future;

use std::sync::Arc;

use crate::config::ServerConfig;
use crate::error::ServerError;
use crate::state::{AppState, VERSION};
use crate::sched::clock::SystemClock;
use crate::sched::scheduler::Scheduler;
use crate::store::Store;

/// Open the store, take ownership of any work a previous server left behind,
/// and serve until `shutdown` resolves.
///
/// The order matters: the orphan sweep must complete before anything can be
/// scheduled, or a stale `running` row would make the overlap check refuse to
/// start the DAG it belongs to.
pub async fn serve<S>(config: ServerConfig, shutdown: S) -> Result<(), ServerError>
where
    S: Future<Output = ()> + Send + 'static,
{
    let store = Store::open(&config.metadata_db, config.store_pool_size)?;

    let instance_id = store::new_id();
    let swept = store::sweep_orphans(&store).await?;
    if swept.runs > 0 || swept.run_groups > 0 || swept.optimizations > 0 {
        log::warn!(
            "recovered from an unclean shutdown: marked {} run(s), {} group(s) and {} \
             optimization(s) as orphaned",
            swept.runs,
            swept.run_groups,
            swept.optimizations
        );
    }

    // Bind before building state so `config.bind` is the address actually in
    // use. With `--bind 127.0.0.1:0` the requested port is 0, which is not
    // something a client can be told to connect to.
    let listener = tokio::net::TcpListener::bind(config.bind)
        .await
        .map_err(|e| ServerError::Internal(format!("binding {}: {e}", config.bind)))?;
    let local_addr = listener
        .local_addr()
        .map_err(|e| ServerError::Internal(e.to_string()))?;

    let mut config = config;
    config.bind = local_addr;

    store::register_instance(
        &store,
        instance_id.clone(),
        local_addr.to_string(),
        VERSION.to_string(),
    )
    .await?;

    let config_tick_interval = config.tick_interval;
    let state = AppState::new(store.clone(), config, instance_id.clone());
    let state_connectors = state.connectors.clone();
    let state_runs = state.runs.clone();
    let app = api::router(state.clone());

    // Realign before the scheduler starts. Windows that elapsed while this
    // server was not running are recorded as missed and skipped, never
    // replayed -- so a server that was down overnight comes back to one run,
    // not eight hours of backlog.
    let scheduler = Arc::new(Scheduler::new(state.clone(), Arc::new(SystemClock)));
    match scheduler.realign_after_downtime().await {
        Ok(n) if n > 0 => log::info!("realigned {n} schedule(s) after downtime"),
        Ok(_) => {}
        Err(e) => log::error!("could not realign schedules: {e}"),
    }

    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    let scheduler_task = tokio::spawn(
        Arc::clone(&scheduler).run_loop(config_tick_interval, shutdown_rx),
    );

    // Printed on stdout, not through the logger: a supervising process (the
    // benchmark harness) binds to port 0 and reads the chosen port from here.
    println!("dee-server listening on http://{local_addr}");

    // A cached connector holds its database open. Without this sweep a server
    // that has run many DAGs pins every warehouse it ever touched -- for a
    // benchmark sweep, one file per cell -- until it runs out of descriptors.
    let janitor = {
        let connectors = state_connectors.clone();
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(std::time::Duration::from_secs(60));
            ticker.tick().await;
            loop {
                ticker.tick().await;
                let dropped = connectors.sweep_idle().await;
                if dropped > 0 {
                    log::info!("released {dropped} idle connector pool(s)");
                }
            }
        })
    };

    let result = axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()));

    // Stop scheduling first, so nothing new starts while we are draining.
    let _ = shutdown_tx.send(true);
    let _ = scheduler_task.await;
    janitor.abort();

    // Ask in-flight runs to stop and give them a moment. Cancellation is only
    // observed between node dispatches, so anything still inside a long node
    // will outlive this -- the next boot's orphan sweep is what makes those
    // rows terminal.
    let signalled = state_runs.cancel_all().await;
    if signalled > 0 {
        log::info!("asked {signalled} in-flight job(s) to stop");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while state_runs.active_count().await > 0 && std::time::Instant::now() < deadline {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
        let stragglers = state_runs.active_count().await;
        if stragglers > 0 {
            log::warn!(
                "{stragglers} job(s) did not stop in time; they will be marked orphaned on the \
                 next start"
            );
        }
    }

    store::mark_instance_stopped(&store, instance_id).await?;
    result
}
