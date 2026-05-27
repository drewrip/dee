use dee::{
    connections::Connection,
    connectors::{Connector, duckdb::DuckDBConnection, postgres::PostgresConnection},
    dag::Dag,
    executor::{Executor, SimpleEngine},
    file::DagFile,
    opt::{Optimizer, OptimizerConfig},
};
use log::info;
use serde::Serialize;

use std::{collections::HashMap, error::Error};
use std::{fs, sync::Arc};

use crate::OptCommand;

pub async fn opt(opt_cmd: OptCommand) -> Result<(), Box<dyn Error>> {
    info!("Optimizing DAG: {}", opt_cmd.dag_file);

    let connections_files: HashMap<String, Connection> =
        serde_json::from_str(&fs::read_to_string(&opt_cmd.connections)?)?;
    let target_connection = connections_files.get(&opt_cmd.target).ok_or_else(|| {
        format!(
            "connection '{}' not found in '{}'",
            opt_cmd.target, opt_cmd.connections
        )
    })?;
    let dag_file: DagFile = serde_json::from_str(&fs::read_to_string(opt_cmd.dag_file)?)?;
    let mut dag = Dag::try_from(dag_file)?;

    let mut config = OptimizerConfig::new()
        .with_omp_top(opt_cmd.omp_top)
        .with_omp_early_termination(!opt_cmd.omp_exhaust)
        .with_hmp_no_plan_dups(opt_cmd.hmp_no_plan_dups);

    if let Some(enabled_passes) = opt_cmd.enable {
        config = config.with_all_disabled();
        for pass_name in enabled_passes {
            config.set_pass(&pass_name, true);
        }
    } else if let Some(disabled_passes) = opt_cmd.disable {
        config = config.with_all_enabled();
        for pass_name in disabled_passes {
            config.set_pass(&pass_name, false);
        }
    }

    let centrality = match opt_cmd.omp_node_centrality {
        crate::CliOMPCentrality::Outdegree => dee::opt::omp::OMPCentrality::OutDegree,
        crate::CliOMPCentrality::Paths => dee::opt::omp::OMPCentrality::Paths,
    };
    config = config.with_omp_centrality(centrality);

    let opt_stats = match &target_connection {
        Connection::DuckDB(config_conn) => {
            let conn = DuckDBConnection::new(config_conn.clone()).await?;
            let engine = SimpleEngine::new(conn.clone())?;
            engine.cleanup(&dag).await?;
            let mut optimizer =
                Optimizer::new_with_config(conn, Arc::new(engine), config).stats_on_passes(true);
            optimizer.run(&mut dag).await?
        }
        Connection::Postgres(config_conn) => {
            let conn = PostgresConnection::new(config_conn.clone()).await?;
            let engine = SimpleEngine::new(conn.clone())?;
            engine.cleanup(&dag).await?;
            let mut optimizer =
                Optimizer::new_with_config(conn, Arc::new(engine), config).stats_on_passes(true);
            optimizer.run(&mut dag).await?
        }
    };
    if opt_cmd.stats {
        let mut buf = Vec::new();
        let formatter = serde_json::ser::PrettyFormatter::with_indent(b"    ");
        let mut ser = serde_json::Serializer::with_formatter(&mut buf, formatter);
        opt_stats.serialize(&mut ser)?;
        let stats_str = String::from_utf8(buf)?;
        println!("{}", stats_str);
    }
    let new_dag_file: DagFile = DagFile::from(dag);
    let mut buf = Vec::new();
    let formatter = serde_json::ser::PrettyFormatter::with_indent(b"    ");
    let mut ser = serde_json::Serializer::with_formatter(&mut buf, formatter);
    new_dag_file.serialize(&mut ser)?;
    let out_str = String::from_utf8(buf)?;
    if let Some(output) = opt_cmd.output {
        fs::write(output, out_str)?;
    } else {
        println!("{}", out_str);
    }
    Ok(())
}
