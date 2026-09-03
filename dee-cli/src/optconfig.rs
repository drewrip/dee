//! The optimizer's flags, shared by every command that can carry a config.
//!
//! `dee dag submit`, `dee dag optimizer` and `dee optimize` all describe the
//! same `OptimizerConfig`, so they take the same arguments from one place.
//!
//! Every option is tri-state on purpose. A flag that was not passed emits
//! nothing, and the server fills it in from the DAG's stored configuration --
//! that is what makes `dee optimize pipeline` run under the settings the DAG
//! was submitted with. If these were ordinary clap defaults, every invocation
//! would send a complete config and the stored one could never take effect.
//!
//! Boolean flags therefore use `--flag` for true and `--flag=false` for false,
//! rather than presence alone, so a stored `true` can be turned off for one
//! run without rewriting the DAG's configuration.

use std::fs;
use std::path::PathBuf;

use clap::Args;
use serde_json::{Map, Value, json};

#[derive(clap::ValueEnum, Clone, Debug)]
pub enum CliOMPCentrality {
    Outdegree,
    Paths,
}

#[derive(clap::ValueEnum, Clone, Debug)]
pub enum CliHMPStrategy {
    Breadth,
    Greedy,
}

#[derive(Args, Clone, Debug)]
pub struct OptimizerArgs {
    /// Passes to run, starting from everything off. Comma separated:
    /// parallelism, hmp, omp, pushdown.
    #[arg(long, value_delimiter = ',', conflicts_with = "disable")]
    pub enable: Option<Vec<String>>,
    /// Passes to skip, starting from everything on.
    #[arg(long, value_delimiter = ',', conflicts_with = "enable")]
    pub disable: Option<Vec<String>>,

    /// Read the config from a JSON file. Individual flags override it.
    #[arg(long, value_name = "FILE")]
    pub optimizer_config: Option<PathBuf>,

    /// OMP: consider only the top N nodes by centrality.
    #[arg(long)]
    pub omp_top: Option<usize>,
    /// OMP: how to rank nodes when picking candidates.
    #[arg(long)]
    pub omp_node_centrality: Option<CliOMPCentrality>,
    /// OMP: evaluate every candidate instead of stopping early.
    #[arg(long, require_equals = true, num_args = 0..=1, default_missing_value = "true")]
    pub omp_exhaust: Option<bool>,
    /// OMP: do not run pushdown before each candidate evaluation.
    #[arg(long, require_equals = true, num_args = 0..=1, default_missing_value = "true")]
    pub omp_no_pushdown: Option<bool>,

    /// HMP: rank views by the duplicate work they cause downstream.
    #[arg(long, require_equals = true, num_args = 0..=1, default_missing_value = "true")]
    pub hmp_downstream_cost: Option<bool>,
    /// HMP: DAG runs to spend searching for candidates.
    #[arg(long)]
    pub hmp_max_runs: Option<usize>,
    /// HMP: fraction of operator CPU time used to build the working set.
    #[arg(long)]
    pub hmp_top_cpu_time: Option<f64>,
    /// HMP: divide a view's CPU time by its estimated cardinality when ranking.
    #[arg(long, require_equals = true, num_args = 0..=1, default_missing_value = "true")]
    pub hmp_normalize_with_cardinality: Option<bool>,
    /// HMP: how to search the node ranking.
    #[arg(long)]
    pub hmp_strategy: Option<CliHMPStrategy>,
    /// HMP: hypotheses the greedy strategy's beam search keeps alive.
    #[arg(long)]
    pub hmp_beam_width: Option<usize>,
    /// HMP: do not run pushdown before each candidate evaluation.
    #[arg(long, require_equals = true, num_args = 0..=1, default_missing_value = "true")]
    pub hmp_no_pushdown: Option<bool>,
    /// HMP: log the operator ranking table after the baseline run.
    #[arg(long, require_equals = true, num_args = 0..=1, default_missing_value = "true")]
    pub hmp_show_operators: Option<bool>,
    /// HMP: log the node ranking table after the baseline run.
    #[arg(long, require_equals = true, num_args = 0..=1, default_missing_value = "true")]
    pub hmp_show_nodes: Option<bool>,

    /// ParallelismTuning: node-concurrency caps to measure. Comma separated.
    #[arg(long, value_delimiter = ',')]
    pub parallelism_ladder: Option<Vec<usize>>,
    /// ParallelismTuning: runs spent measuring the DAG's current setting.
    #[arg(long)]
    pub parallelism_seed_repeats: Option<usize>,
    /// ParallelismTuning: re-measurements a rung must survive to be accepted.
    #[arg(long)]
    pub parallelism_confirm_runs: Option<usize>,

    /// Capture a resource timeseries for every candidate run.
    #[arg(long, require_equals = true, num_args = 0..=1, default_missing_value = "true")]
    pub profile_iterations: Option<bool>,
}

impl OptimizerArgs {
    /// The config these flags describe, or `None` if none were given.
    ///
    /// `None` is the signal the server reads as "use what the DAG already
    /// has", so it must stay distinguishable from an empty object.
    pub fn to_json(&self) -> Result<Option<Value>, Box<dyn std::error::Error>> {
        let mut config: Map<String, Value> = match &self.optimizer_config {
            Some(path) => serde_json::from_str::<Value>(&fs::read_to_string(path)?)?
                .as_object()
                .cloned()
                .ok_or("the optimizer config file must contain a JSON object")?,
            None => Map::new(),
        };

        // `--enable` starts from nothing on, so the resulting pass set is
        // exactly what was asked for and does not shift when dee's defaults
        // change. Neither flag leaves pass selection to the DAG.
        match (&self.enable, &self.disable) {
            (Some(passes), _) => {
                for (key, name) in PASSES {
                    config.insert(key.into(), json!(passes.iter().any(|p| p == name)));
                }
            }
            (None, Some(passes)) => {
                for (key, name) in PASSES {
                    config.insert(key.into(), json!(!passes.iter().any(|p| p == name)));
                }
            }
            (None, None) => {}
        }

        let mut set = |key: &str, value: Option<Value>| {
            if let Some(value) = value {
                config.insert(key.into(), value);
            }
        };

        set("omp_top", self.omp_top.map(|v| json!(v)));
        set(
            "omp_centrality",
            self.omp_node_centrality.as_ref().map(|c| match c {
                CliOMPCentrality::Outdegree => json!("outdegree"),
                CliOMPCentrality::Paths => json!("paths"),
            }),
        );
        // These two name the behaviour they turn off, so the stored value is
        // their negation.
        set("omp_early_termination", self.omp_exhaust.map(|v| json!(!v)));
        set("omp_use_pushdown", self.omp_no_pushdown.map(|v| json!(!v)));
        set("hmp_use_pushdown", self.hmp_no_pushdown.map(|v| json!(!v)));

        set("hmp_downstream_cost", self.hmp_downstream_cost.map(|v| json!(v)));
        set("hmp_max_runs", self.hmp_max_runs.map(|v| json!(v)));
        set("hmp_top_cpu_time", self.hmp_top_cpu_time.map(|v| json!(v)));
        set(
            "hmp_normalize_with_cardinality",
            self.hmp_normalize_with_cardinality.map(|v| json!(v)),
        );
        set(
            "hmp_strategy",
            self.hmp_strategy.as_ref().map(|s| match s {
                CliHMPStrategy::Breadth => json!("breadth"),
                CliHMPStrategy::Greedy => json!("greedy"),
            }),
        );
        set("hmp_beam_width", self.hmp_beam_width.map(|v| json!(v)));
        set("parallelism_ladder", self.parallelism_ladder.as_ref().map(|v| json!(v)));
        set(
            "parallelism_seed_repeats",
            self.parallelism_seed_repeats.map(|v| json!(v)),
        );
        set(
            "parallelism_confirm_runs",
            self.parallelism_confirm_runs.map(|v| json!(v)),
        );
        set("profile_iterations", self.profile_iterations.map(|v| json!(v)));

        // The server refuses a path here, so only the log-the-table form is
        // reachable remotely: the flag chooses between `""` and off.
        set(
            "hmp_show_operators",
            self.hmp_show_operators.map(|v| if v { json!("") } else { Value::Null }),
        );
        set(
            "hmp_show_nodes",
            self.hmp_show_nodes.map(|v| if v { json!("") } else { Value::Null }),
        );

        Ok(if config.is_empty() {
            None
        } else {
            Some(Value::Object(config))
        })
    }
}

const PASSES: [(&str, &str); 4] = [
    ("run_parallelism_pass", "parallelism"),
    ("run_hmp_pass", "hmp"),
    ("run_omp_pass", "omp"),
    ("run_pushdown_pass", "pushdown"),
];

/// Print a resolved config, showing only what bears on the passes that will run.
///
/// A full `OptimizerConfig` is twenty-odd fields, most of which belong to a pass
/// that is switched off. Printing all of them before every optimization buries
/// the two or three settings that actually determine the result, so the `omp_`
/// and `hmp_` prefixes -- which the field names already follow -- are used to
/// drop the irrelevant ones.
pub fn print_config(config: &Value) {
    let enabled: Vec<&str> = PASSES
        .iter()
        .filter(|(key, _)| config[*key].as_bool().unwrap_or(false))
        .map(|(_, name)| *name)
        .collect();
    println!(
        "  passes: {}",
        if enabled.is_empty() {
            "none".to_string()
        } else {
            enabled.join(", ")
        }
    );

    let mut keys: Vec<&String> = config
        .as_object()
        .map(|o| o.keys().collect())
        .unwrap_or_default();
    keys.sort();
    for key in keys {
        let relevant = match key.split_once('_') {
            Some((prefix, _)) if prefix == "omp" || prefix == "hmp" || prefix == "parallelism" => {
                enabled.contains(&prefix)
            }
            // `run_*_pass` is already reported as the pass list, and `explain`
            // is a property of the request rather than of the optimizer.
            _ => !PASSES.iter().any(|(pass, _)| pass == key) && key != "explain",
        };
        if relevant {
            println!("  {key}: {}", config[key]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Harness {
        #[command(flatten)]
        optimizer: OptimizerArgs,
    }

    fn config_from(args: &[&str]) -> Value {
        let mut argv = vec!["dee"];
        argv.extend_from_slice(args);
        Harness::parse_from(argv)
            .optimizer
            .to_json()
            .expect("to_json")
            .expect("some config")
    }

    #[test]
    fn test_the_ladder_reaches_the_server_as_a_list_of_numbers() {
        // Comma-separated on the command line, a JSON array on the wire.
        // `OptimizerConfig::parallelism_ladder` is `Vec<usize>`, and a string
        // here would be rejected by `deny_unknown_fields`'s stricter cousin,
        // type checking, at the far end.
        let config = config_from(&["--parallelism-ladder", "1,2,4"]);
        assert_eq!(config["parallelism_ladder"], json!([1, 2, 4]));
    }

    #[test]
    fn test_enable_names_the_new_pass_and_turns_the_others_off() {
        // `--enable` starts from everything off, so a pass missing from
        // `PASSES` would be left at whatever the DAG had stored -- silently
        // running something that was not asked for.
        let config = config_from(&["--enable", "parallelism"]);
        assert_eq!(config["run_parallelism_pass"], json!(true));
        assert_eq!(config["run_hmp_pass"], json!(false));
        assert_eq!(config["run_omp_pass"], json!(false));
        assert_eq!(config["run_pushdown_pass"], json!(false));
    }

    #[test]
    fn test_disable_leaves_the_new_pass_on() {
        let config = config_from(&["--disable", "hmp"]);
        assert_eq!(config["run_parallelism_pass"], json!(true));
        assert_eq!(config["run_hmp_pass"], json!(false));
    }

    #[test]
    fn test_no_flags_still_defers_to_the_dags_own_settings() {
        // The tri-state property the whole module rests on.
        assert!(
            Harness::parse_from(["dee"]).optimizer.to_json().unwrap().is_none(),
            "an empty invocation sent a config"
        );
    }
}
