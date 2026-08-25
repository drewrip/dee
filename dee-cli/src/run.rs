use chrono::Utc;
use dee::{
    connections::Connection,
    connectors::{Connector, duckdb::DuckDBConnection, postgres::PostgresConnection},
    dag::Dag,
    executor::{Executor, ProfilingConfig, SimpleEngine},
    file::DagFile,
    profile::{
        DagRunProfile, ProfileReport, build_dag_run_profile, render_profile_html,
        render_profile_summary,
    },
};
use log::info;

use std::collections::HashMap;
use std::fs;

use crate::RunCommand;

pub async fn run(run_cmd: RunCommand) -> Result<(), Box<dyn std::error::Error>> {
    let connections_files: HashMap<String, Connection> =
        serde_json::from_str(&fs::read_to_string(&run_cmd.connections)?)?;
    let target_connection = connections_files.get(&run_cmd.target).ok_or_else(|| {
        format!(
            "connection '{}' not found in '{}'",
            run_cmd.target, run_cmd.connections
        )
    })?;
    // --report-json is a machine-readable consumer of the profile, so it
    // implies profiling just like the human-facing flags do.
    let profiling_enabled = run_cmd.profile
        || run_cmd.profile_dump.is_some()
        || run_cmd.profile_viz.is_some()
        || run_cmd.report_json.is_some();

    let runs = match &target_connection {
        Connection::DuckDB(config) => {
            let conn = DuckDBConnection::new(config.clone()).await?;
            run_with_connector(conn, &run_cmd, profiling_enabled).await?
        }
        Connection::Postgres(config) => {
            let conn = PostgresConnection::new(config.clone()).await?;
            run_with_connector(conn, &run_cmd, profiling_enabled).await?
        }
    };

    if profiling_enabled {
        let report = ProfileReport {
            generated_at: Utc::now(),
            runs,
        };

        println!("{}", render_profile_summary(&report));

        if let Some(path) = &run_cmd.report_json {
            fs::write(path, serde_json::to_string_pretty(&report)?)?;
        }

        if let Some(path) = &run_cmd.profile_dump {
            let payload = serde_json::to_string_pretty(&report)?;
            fs::write(path, payload)?;
        }

        if let Some(path) = &run_cmd.profile_viz {
            fs::write(path, render_profile_html(&report)?)?;
        }
    }

    Ok(())
}

async fn run_with_connector<C>(
    conn: std::sync::Arc<C>,
    run_cmd: &RunCommand,
    profiling_enabled: bool,
) -> Result<Vec<DagRunProfile>, Box<dyn std::error::Error>>
where
    C: Connector + Send + Sync + 'static,
{
    let mut engine = SimpleEngine::new(conn)?;
    if let Some(dump_plans) = &run_cmd.dump_plans {
        engine = engine.with_plans_dir(dump_plans.clone());
    }
    if profiling_enabled {
        let mut profiling_config = ProfilingConfig {
            collect_plans: true,
            ..ProfilingConfig::default()
        };
        if let Some(ms) = run_cmd.profile_interval_ms {
            profiling_config.sample_interval = std::time::Duration::from_millis(ms);
        }
        engine = engine.with_profiling(profiling_config);
    }

    let repeat = run_cmd.repeat.max(1);
    let warmups = run_cmd.warmups;
    let total = warmups + repeat;

    let mut runs = Vec::new();
    for dag_file_path in &run_cmd.dag_files {
        info!("Starting DAG: {}", dag_file_path);
        let dag_file: DagFile = serde_json::from_str(&fs::read_to_string(dag_file_path)?)?;
        let dag = Dag::try_from(dag_file)?;

        // Every repetition happens inside this one process against this one
        // connection pool, so the timings in ExecStats measure the DAG and
        // not the CLI's startup.
        for i in 0..total {
            let is_warmup = i < warmups;
            let phase = if is_warmup { "warmup" } else { "measure" };

            // Each repetition starts from a clean slate, otherwise an earlier
            // repetition's materialized tables would still be present.
            engine.cleanup(&dag).await?;
            let exec_stats = engine.run(&dag).await?;

            info!(
                "Finished DAG: {} [{} {}/{}] in {}ms",
                dag_file_path,
                phase,
                if is_warmup { i + 1 } else { i - warmups + 1 },
                if is_warmup { warmups } else { repeat },
                exec_stats.duration.num_milliseconds()
            );

            if profiling_enabled {
                let mut profile = build_dag_run_profile(dag_file_path, &dag, &exec_stats);
                profile.phase = phase.to_string();
                profile.rep_index = if is_warmup { i } else { i - warmups };
                runs.push(profile);
            }
        }
    }

    Ok(runs)
}
