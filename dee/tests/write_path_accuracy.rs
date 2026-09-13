//! How accurately does the split write model price a DuckDB materialization?
//!
//! Not a test -- an experiment, run explicitly:
//!
//! ```text
//! WPA_DB=/path/to/warehouse.duckdb WPA_MODELS=/path/to/models.json \
//!   cargo test --release -p dee --test write_path_accuracy -- --ignored --nocapture
//! ```
//!
//! `WPA_MODELS` is a JSON array of `{"name": ..., "sql": ...}` in dependency
//! order. For each one the experiment does two things the production path does:
//!
//!  1. **Asks** the engine which sink it would use, through
//!     [`Connector::write_path_for`] -- no execution, which is the situation
//!     HMP is actually in when it prices a View.
//!  2. **Measures** it, by materializing the model with profiling on and
//!     reading the sink's own name and timing back off the executed plan.
//!
//! Then it scores three things: how often the prediction was right, and how
//! well the fitted constants price the write both pooled into one and split by
//! sink. Leave-one-out, so every number is a prediction of a table the constant
//! was not fitted on.

use std::sync::Arc;

use dee::connectors::Connector;
use dee::connectors::duckdb::{DuckDBConfig, DuckDBConnection};
use dee::dag::MaterializeMode;
use dee::opt::learned::LearnedCostModel;
use dee::plan::{PlanNode, observed_write_path};
use serde_json::Value;

/// A predicate picking which observations a constant may be fitted on.
type Filter<'a> = Box<dyn Fn(&Observed, &Observed) -> bool + 'a>;
/// A predicate picking which observations a summary line covers.
type Member<'a> = Box<dyn Fn(&Observed) -> bool + 'a>;

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

/// One materialization: what we predicted, what happened, and what it cost.
struct Observed {
    project: String,
    name: String,
    predicted: String,
    actual: String,
    seconds: f64,
    rows: f64,
    plans: Vec<PlanNode>,
}

/// How many rows the operator below the write emitted.
fn payload_rows(node: &PlanNode) -> Option<f64> {
    if dee::plan::is_write_operator(&node.operator) {
        return node
            .children
            .first()
            .and_then(|c| c.cardinality)
            .map(|c| c as f64);
    }
    node.children.iter().find_map(payload_rows)
}

/// The write operator's own exclusive time, wherever it sits.
fn write_seconds(node: &PlanNode) -> Option<f64> {
    if dee::plan::is_write_operator(&node.operator) {
        return node.exclusive_time_s;
    }
    node.children.iter().find_map(write_seconds)
}

async fn observe(conn: &Arc<DuckDBConnection>, name: &str, sql: &str) -> Option<Observed> {
    // (1) Ask, without executing: the engine names the sink when it is handed
    // the write to plan rather than the SELECT under it. No fallback -- an
    // engine that will not say leaves the write unpriced.
    let predicted = conn.write_path_for(sql).await.ok()??;

    // (2) Measure. Drop first so a rerun is not measuring a no-op.
    let _ = conn
        .drop_relation(MaterializeMode::Table, name.to_string())
        .await;
    let (_, profile) = conn
        .new_relation_and_explain(MaterializeMode::Table, name.to_string(), sql.to_string())
        .await
        .ok()?;
    let plans = conn.parse_plan(&profile?)?;
    let actual = observed_write_path(&plans);
    let seconds = plans.iter().find_map(write_seconds)?;
    // Rows written = the cardinality of the operator feeding the sink. The
    // sink's own is a one-row count of what it did.
    let rows = plans.iter().find_map(payload_rows)?;

    Some(Observed {
        name: name.to_string(),
        predicted,
        actual,
        seconds,
        rows,
        plans,
        project: String::new(),
    })
}

/// Fit the write constants from every observation the filter admits, then
/// price `target` with them. The filter is what makes this honest: nothing is
/// ever scored against a constant its own sample went into.
fn predict(all: &[Observed], target: &Observed, split: bool, keep: &dyn Fn(&Observed) -> bool) -> Option<f64> {
    let mut model = LearnedCostModel::new();
    for o in all.iter() {
        if !keep(o) {
            continue;
        }
        if split {
            model.observe_write(&format!("{}.{}", o.project, o.name), &o.plans, o.rows);
        } else {
            // The old behaviour: every sink fitted into one constant. Forced by
            // renaming both sinks to the same thing before observing.
            let pooled: Vec<PlanNode> = o.plans.iter().cloned().map(pool_sinks).collect();
            model.observe_write(&format!("{}.{}", o.project, o.name), &pooled, o.rows);
        }
    }
    if split {
        model.write_cost(&target.predicted, &target.plans, target.rows)
    } else {
        let pooled: Vec<PlanNode> = target.plans.iter().cloned().map(pool_sinks).collect();
        model.write_cost("INSERT", &pooled, target.rows)
    }
}

fn pool_sinks(mut node: PlanNode) -> PlanNode {
    if dee::plan::is_write_operator(&node.operator) {
        node.operator = "INSERT".to_string();
    }
    node.children = node.children.into_iter().map(pool_sinks).collect();
    node
}

#[tokio::test]
#[ignore = "needs a built DuckDB warehouse"]
async fn how_accurate_are_the_split_write_constants() {
    // "proj=db.duckdb:models.json,proj2=..."; more than one project makes the
    // leave-one-project-out score below meaningful.
    let sets = env_or("WPA_SETS", "");
    assert!(!sets.is_empty(), "set WPA_SETS");

    let mut all = Vec::new();
    for set in sets.split(',') {
        let (proj, rest) = set.split_once('=').expect("proj=db:models");
        let (db, models_path) = rest.rsplit_once(':').expect("db:models");
        let conn = DuckDBConnection::new(DuckDBConfig::new_from_path(db.to_string()))
            .await
            .expect("open the warehouse");
        let models: Vec<Value> =
            serde_json::from_str(&std::fs::read_to_string(models_path).expect("models file"))
                .expect("models json");
        for m in &models {
            let (name, sql) = (m["name"].as_str().unwrap(), m["sql"].as_str().unwrap());
            match observe(&conn, name, sql).await {
                Some(mut o) => {
                    o.project = proj.to_string();
                    all.push(o);
                }
                None => println!("  (skipped {proj}.{name})"),
            }
        }
    }
    assert!(all.len() >= 4, "not enough materializations to score");

    // ---- 1. Was the sink predicted correctly? -------------------------------
    println!("\n=== SINK PREDICTION (from EXPLAIN alone, nothing executed) ===");
    println!("{:<26} {:<22} {:<22}", "model", "asked-the-engine", "actual");
    let mut right = 0;
    for o in &all {
        let ok = o.predicted == o.actual;
        right += ok as usize;
        println!(
            "{:<26} {:<22} {:<22} {}",
            o.name,
            o.predicted,
            o.actual,
            if ok { "ok" } else { "MISS" }
        );
    }
    println!(
        "\n  {right}/{} correct ({:.1}%)",
        all.len(),
        100.0 * right as f64 / all.len() as f64
    );

    // ---- 1b. Does each path's constant transfer between DAGs? ---------------
    // The split is only worth having if a constant fitted on one project
    // prices another. Fitted per project per sink, byte-weighted, so a
    // disagreement here is the constant failing to transfer rather than one
    // table being odd.
    println!("\n=== FITTED CONSTANT PER PROJECT PER SINK (s/byte, median over relations) ===");
    let projects: Vec<String> = {
        let mut p: Vec<String> = all.iter().map(|o| o.project.clone()).collect();
        p.dedup();
        p
    };
    println!("{:<10} {:>26} {:>26}", "project", "BATCH_CREATE_TABLE_AS", "CREATE_TABLE_AS");
    for proj in &projects {
        let mut cells = Vec::new();
        for sink in ["BATCH_CREATE_TABLE_AS", "CREATE_TABLE_AS"] {
            let mut model = LearnedCostModel::new();
            let mut n = 0;
            for o in all
                .iter()
                .filter(|o| &o.project == proj && o.actual == sink && o.seconds >= 0.010)
            {
                model.observe_write(&format!("{}.{}", o.project, o.name), &o.plans, o.rows);
                n += 1;
            }
            cells.push(match model.write_seconds_per_byte(sink) {
                Some(k) => format!("{k:.3e} (n={n})"),
                None => "-".to_string(),
            });
        }
        println!("{:<10} {:>26} {:>26}", proj, cells[0], cells[1]);
    }

    // ---- 1c. Raw samples, so the shape of the cost can be fitted outside ----
    // Bytes as the model itself counts them: fit a one-sample model and invert
    // its constant, which uses exactly the payload width `write_cost` would.
    println!("\n=== SAMPLES  project,name,sink,rows,bytes,seconds ===");
    for o in all.iter() {
        let mut one = LearnedCostModel::new();
        one.observe_write(&o.name, &o.plans, o.rows);
        let bytes = match one.write_seconds_per_byte(&o.actual) {
            Some(k) if k > 0.0 => o.seconds / k,
            _ => continue,
        };
        println!(
            "SAMPLE,{},{},{},{:.0},{:.0},{:.6}",
            o.project, o.name, o.actual, o.rows, bytes, o.seconds
        );
    }

    // ---- 2. How well do the constants price the write? ----------------------
    // Two regimes. Within a DAG, leaving one table out -- which is the worst
    // case, because splitting halves the samples behind each constant. And
    // across DAGs, fitting on every other project and predicting this one --
    // which is what the engine-keyed store actually does, since the constants
    // are shared by every DAG on the backend.
    let regimes: [(&str, Filter<'_>); 2] = [
        (
            "leave-one-table-out",
            Box::new(|o: &Observed, t: &Observed| !(o.project == t.project && o.name == t.name)),
        ),
        (
            "leave-one-project-out",
            Box::new(|o: &Observed, t: &Observed| o.project != t.project),
        ),
    ];

    for (regime, keep) in regimes.iter() {
        println!("\n=== WRITE COST -- {regime} ===");
        println!(
            "{:<30} {:>10} {:>20} {:>20}",
            "model", "measured", "pooled", "split-by-sink"
        );
        let (mut pooled_err, mut split_err) = (Vec::new(), Vec::new());
        let (mut pooled_tot, mut split_tot, mut meas_tot) = (0.0, 0.0, 0.0);
        for o in all.iter() {
            let f = |x: &Observed| keep(x, o);
            let p = predict(&all, o, false, &f);
            let s = predict(&all, o, true, &f);
            let fmt = |v: Option<f64>| match v {
                Some(v) => format!("{:.1}ms ({:.2}x)", v * 1000.0, v / o.seconds),
                None => "unpriced".to_string(),
            };
            println!(
                "{:<30} {:>9.1}ms {:>20} {:>20}",
                format!("{}.{}", o.project, o.name),
                o.seconds * 1000.0,
                fmt(p),
                fmt(s)
            );
            // A write under 10ms is below the noise of the thing being priced
            // and cannot move a makespan; scoring it drowns the signal.
            if o.seconds < 0.010 {
                continue;
            }
            meas_tot += o.seconds;
            if let Some(v) = p {
                pooled_err.push((v / o.seconds).ln().abs());
                pooled_tot += v;
            }
            if let Some(v) = s {
                split_err.push((v / o.seconds).ln().abs());
                split_tot += v;
            }
        }
        let summarize = |label: &str, mut e: Vec<f64>, tot: f64| {
            if e.is_empty() {
                println!("  {label:<18} no predictions");
                return;
            }
            e.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!(
                "  {label:<18} n={:<3} median {:.2}x  p90 {:.2}x  worst {:.2}x   \
                 summed {:.2}s vs {:.2}s measured ({:.2}x)",
                e.len(),
                e[e.len() / 2].exp(),
                e[((0.9 * e.len() as f64) as usize).min(e.len() - 1)].exp(),
                e.last().unwrap().exp(),
                tot,
                meas_tot,
                tot / meas_tot,
            );
        };
        println!("\n  (writes of 10ms or more only -- anything less cannot move a makespan)");
        summarize("pooled:", pooled_err, pooled_tot);
        summarize("split by sink:", split_err, split_tot);

        // The split was supposed to fix a systematic bias between the two
        // sinks, so score them apart: a median over both together hides
        // whichever one it fixed.
        // The buffered sink is the one with the dead zone: under ~614,400 rows
        // DuckDB never persists inside the operator, so a constant fitted
        // across both regimes is fitted on two different things. Scored apart
        // to show whether that is what is left.
        let groups: Vec<(String, Member<'_>)> = vec![
            (
                "BATCH_CREATE_TABLE_AS".into(),
                Box::new(|o: &Observed| o.actual == "BATCH_CREATE_TABLE_AS"),
            ),
            (
                "CREATE_TABLE_AS >deadzone".into(),
                Box::new(|o: &Observed| o.actual == "CREATE_TABLE_AS" && o.rows > 614_400.0),
            ),
            (
                "CREATE_TABLE_AS <deadzone".into(),
                Box::new(|o: &Observed| o.actual == "CREATE_TABLE_AS" && o.rows <= 614_400.0),
            ),
        ];
        for (sink, member) in groups.iter() {
            let (mut pe, mut se) = (Vec::new(), Vec::new());
            let (mut pt, mut st, mut mt) = (0.0, 0.0, 0.0);
            for o in all.iter().filter(|o| member(o) && o.seconds >= 0.010) {
                let f = |x: &Observed| keep(x, o);
                mt += o.seconds;
                if let Some(v) = predict(&all, o, false, &f) {
                    pe.push((v / o.seconds).ln().abs());
                    pt += v;
                }
                if let Some(v) = predict(&all, o, true, &f) {
                    se.push((v / o.seconds).ln().abs());
                    st += v;
                }
            }
            let med = |mut e: Vec<f64>| {
                e.sort_by(|a, b| a.partial_cmp(b).unwrap());
                if e.is_empty() { f64::NAN } else { e[e.len() / 2].exp() }
            };
            println!(
                "    {sink:<26} n={:<3} pooled median {:.2}x (sum {:.2}x)   \
                 split median {:.2}x (sum {:.2}x)",
                se.len(),
                med(pe),
                pt / mt,
                med(se),
                st / mt,
            );
        }
    }
}
