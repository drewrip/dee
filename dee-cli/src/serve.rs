use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use dee_server::config::{DEFAULT_PORT, ServerConfig, default_data_dir};

use crate::ServeCommand;

pub async fn serve(cmd: ServeCommand) -> Result<(), Box<dyn std::error::Error>> {
    let bind: SocketAddr = match &cmd.bind {
        Some(addr) => addr.parse().map_err(|e| format!("invalid --bind '{addr}': {e}"))?,
        None => SocketAddr::from(([127, 0, 0, 1], DEFAULT_PORT)),
    };
    let metadata_db: PathBuf = cmd
        .metadata_db
        .map(PathBuf::from)
        .unwrap_or_else(|| default_data_dir().join("dee.duckdb"));

    let mut config = ServerConfig {
        bind,
        metadata_db,
        ..ServerConfig::default()
    };
    if let Some(ms) = cmd.tick_interval_ms {
        config.tick_interval = Duration::from_millis(ms);
    }
    if let Some(n) = cmd.max_concurrent_runs {
        config.max_concurrent_runs = n;
    }

    dee_server::serve(config, shutdown_signal()).await?;
    Ok(())
}

/// Resolves on SIGINT or SIGTERM. Both matter: a terminal user sends the
/// first, a supervisor or container runtime the second, and a run left
/// mid-flight by an unhandled signal has to be recovered by the next boot's
/// orphan sweep instead of being finished cleanly here.
async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut sig) => {
                sig.recv().await;
            }
            Err(e) => {
                log::warn!("cannot listen for SIGTERM: {e}");
                std::future::pending::<()>().await;
            }
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => log::info!("received interrupt, shutting down"),
        _ = terminate => log::info!("received SIGTERM, shutting down"),
    }
}
