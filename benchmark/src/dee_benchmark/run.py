import os
import subprocess
import time
import shutil
import yaml
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from .plot import (
    plot_data,
    plot_deep_dive,
    plot_hmp_iterations,
    plot_omp_iterations,
    plot_pushdown_comparison,
    plot_resource_usage,
)


def run_cmd(cmd, cwd=None, env=None, capture=True):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, env=env, capture_output=capture, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        if not capture:
            print(f"Exit code: {result.returncode}")
        result.check_returncode()
    return result.stdout


def generate_connections_json(src_project_dir, dest_project_dir, requested_db_type, max_mem=None, threads=None):
    profiles_path = src_project_dir / "profiles.yml"
    if not profiles_path.exists():
        return None, None

    with open(profiles_path, "r") as f:
        profiles_yml = yaml.safe_load(f)

    # dbt profiles.yml can have multiple profiles, but usually it's one for the project
    target_profile_name = None
    for name in profiles_yml:
        if name != "config":
            target_profile_name = name
            break

    if not target_profile_name:
        return None, None

    profile_cfg = profiles_yml[target_profile_name]
    outputs = profile_cfg.get("outputs", {})

    dee_connections = {}
    final_target = None

    for output_name, output_cfg in outputs.items():
        db_type = output_cfg.get("type")
        if db_type != requested_db_type:
            continue

        dee_cfg = {"type": db_type}

        if db_type == "duckdb":
            target_key = "dev"
            path = output_cfg.get("path")
            if path:
                p = Path(path)
                # Handle relative paths - they are relative to dbt project dir
                if not p.is_absolute():
                    src_db_path = src_project_dir / p
                else:
                    src_db_path = p

                dest_db_path = dest_project_dir / src_db_path.name
                if src_db_path.exists():
                    print(f"Linking database from {src_db_path} to {dest_db_path}...")
                    dest_db_path.symlink_to(src_db_path.resolve())
                else:
                    print(
                        f"Warning: Source database file {src_db_path} does not exist."
                    )

                dee_cfg["database"] = str(dest_db_path.absolute())
            dee_cfg["num_connections"] = 16
            if max_mem:
                dee_cfg["max_memory"] = max_mem
            if threads:
                dee_cfg["threads"] = threads
        elif db_type == "postgres":
            target_key = "postgres"
            dee_cfg["host"] = output_cfg.get("host")
            dee_cfg["port"] = output_cfg.get("port")
            dee_cfg["user"] = output_cfg.get("user")
            dee_cfg["password"] = output_cfg.get("password")
            dee_cfg["database"] = output_cfg.get("dbname")
            dee_cfg["num_connections"] = output_cfg.get("threads", 4)
        else:
            continue

        dee_connections[target_key] = dee_cfg
        final_target = target_key
        # We only need one output of the requested type
        break

    if not dee_connections:
        return None, None

    connections_json_path = dest_project_dir / "connections.json"
    with open(connections_json_path, "w") as f:
        json.dump(dee_connections, f, indent=4)

    return str(connections_json_path), final_target


def setup_project(project_name, dag_bench_root, tmp_bench_dir, dee_cli_path, db_type, max_mem=None, threads=None):
    """Copy a project into tmp_bench_dir, dbt-compile it, convert its manifest
    into a dee DAG, and generate a connections.json for it.

    Returns a dict with `dest_project_path`, `dag_json_path`, `connections_json`,
    and `target` on success, or `None` if the project couldn't be set up
    (already printed a warning/error explaining why).
    """
    src_project_path = Path(dag_bench_root) / "projects" / project_name
    dest_project_path = tmp_bench_dir / project_name

    if not src_project_path.exists():
        print(f"Error: Project {project_name} not found at {src_project_path}")
        return None

    shutil.copytree(src_project_path, dest_project_path)

    # 1. dbt compile
    dbt_target = "dev" if db_type == "duckdb" else "postgres"
    run_cmd(["dbt", "compile", "--target", dbt_target], cwd=dest_project_path)

    manifest_path = dest_project_path / "target" / "manifest.json"
    dag_json_path = dest_project_path / "dag.json"

    # 2. convert
    run_cmd(
        [
            dee_cli_path,
            "convert",
            "--format",
            "dbt",
            "-o",
            str(dag_json_path),
            str(manifest_path),
        ]
    )

    connections_json, target = generate_connections_json(
        src_project_path, dest_project_path, db_type, max_mem=max_mem, threads=threads
    )
    if not connections_json:
        print(
            f"Warning: Could not generate connections.json for {project_name} with type {db_type}"
        )
        return None

    return {
        "dest_project_path": dest_project_path,
        "dag_json_path": dag_json_path,
        "connections_json": connections_json,
        "target": target,
    }


def build_opt_cmd(
    dee_cli_path,
    connections_json,
    target,
    output_path,
    dag_json_path,
    stats=True,
    omp_top=None,
    omp_node_centrality=None,
    omp_exhaust=False,
    omp_no_pushdown=False,
    enable=None,
    disable=None,
    hmp_downstream_cost=False,
    hmp_max_runs=None,
    hmp_top_cpu_time=None,
    hmp_show_operators=None,
    hmp_show_nodes=None,
    hmp_normalize_with_cardinality=False,
    hmp_strategy=None,
    hmp_no_pushdown=False,
    explain_path=None,
    profile_iterations=False,
):
    """Build a `dee-cli opt` command line from the optimizer knobs shared by
    both the standard single-DAG benchmark and the pushdown A/B comparison."""
    opt_cmd = [
        dee_cli_path,
        "opt",
        "--connections",
        connections_json,
        "--target",
        target,
        "-o",
        str(output_path),
        str(dag_json_path),
    ]
    if stats:
        opt_cmd.insert(2, "--stats")
    if omp_top:
        opt_cmd.extend(["--omp-top", str(omp_top)])
    if omp_node_centrality:
        opt_cmd.extend(["--omp-node-centrality", omp_node_centrality])
    if omp_exhaust:
        opt_cmd.append("--omp-exhaust")
    if omp_no_pushdown:
        opt_cmd.append("--omp-no-pushdown")
    if enable:
        opt_cmd.extend(["--enable", enable])
    if disable:
        opt_cmd.extend(["--disable", disable])
    if hmp_downstream_cost:
        opt_cmd.append("--hmp-downstream-cost")
    if hmp_max_runs:
        opt_cmd.extend(["--hmp-max-runs", str(hmp_max_runs)])
    if hmp_top_cpu_time is not None:
        opt_cmd.extend(["--hmp-top-cpu-time", str(hmp_top_cpu_time)])
    if hmp_show_operators is not None:
        if hmp_show_operators:
            opt_cmd.extend(["--hmp-show-operators", hmp_show_operators])
        else:
            opt_cmd.append("--hmp-show-operators")
    if hmp_show_nodes is not None:
        if hmp_show_nodes:
            opt_cmd.extend(["--hmp-show-nodes", hmp_show_nodes])
        else:
            opt_cmd.append("--hmp-show-nodes")
    if hmp_normalize_with_cardinality:
        opt_cmd.append("--hmp-normalize-with-cardinality")
    if hmp_strategy:
        opt_cmd.extend(["--hmp-strategy", hmp_strategy])
    if hmp_no_pushdown:
        opt_cmd.append("--hmp-no-pushdown")
    if explain_path:
        opt_cmd.append(f"--explain={explain_path}")
    if profile_iterations:
        opt_cmd.append("--profile-iterations")
    return opt_cmd


def run_multiple_times(
    dee_cli_path, connections_json, target, dag_path, iterations,
    profile=False, profile_dir=None, profile_interval_ms=None,
):
    """Run a DAG `iterations` times (plus a ~10% warmup fraction that isn't
    timed), returning the wall-clock seconds for each timed run.

    When `profile` is set, every timed iteration is also run with
    `--profile`/`--profile-dump`, and the second return value is a list
    (one entry per iteration) of that run's `system_samples` (CPU/memory/
    disk timeseries); otherwise the second return value is `None`."""
    warmup_iters = int(iterations * 0.1)
    if warmup_iters > 0:
        print(f"  Running {warmup_iters} warmup iterations...")
        for _ in range(warmup_iters):
            run_cmd(
                [
                    dee_cli_path,
                    "run",
                    "--connections",
                    connections_json,
                    "--target",
                    target,
                    str(dag_path),
                ]
            )

    times = []
    resource_samples = [] if profile else None
    for i in range(iterations):
        print(f"  Iteration {i + 1}/{iterations}...")
        cmd = [
            dee_cli_path,
            "run",
            "--connections",
            connections_json,
            "--target",
            target,
        ]
        profile_path = None
        if profile:
            profile_path = Path(profile_dir) / f"profile_{i}.json"
            cmd += ["--profile", "--profile-dump", str(profile_path)]
            if profile_interval_ms is not None:
                cmd += ["--profile-interval-ms", str(profile_interval_ms)]
        cmd.append(str(dag_path))

        start = time.time()
        run_cmd(cmd)
        times.append(time.time() - start)

        if profile:
            with open(profile_path, "r") as f:
                report = json.load(f)
            runs = report.get("runs", [])
            resource_samples.append(runs[0]["system_samples"] if runs else [])

    return times, resource_samples


def benchmark(
    config_file,
    dag_bench_root,
    dee_cli_path,
    db_type,
    deep_dive=False,
    n=5,
    max_mem=None,
    threads=None,
    omp_top=None,
    omp_node_centrality=None,
    omp_exhaust=False,
    omp_no_pushdown=False,
    enable=None,
    disable=None,
    hmp_downstream_cost=False,
    hmp_max_runs=None,
    hmp_top_cpu_time=None,
    hmp_show_operators=None,
    hmp_show_nodes=None,
    hmp_normalize_with_cardinality=False,
    hmp_strategy=None,
    hmp_no_pushdown=False,
    explain_dir=None,
    profile=False,
    profile_interval_ms=None,
):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    projects_to_run = config.get("projects", [])
    results = []

    tmp_bench_dir = Path("tmp_projects")
    if tmp_bench_dir.exists():
        shutil.rmtree(tmp_bench_dir)
    tmp_bench_dir.mkdir(parents=True)

    for project_name in projects_to_run:
        print(f"\n--- Benchmarking Project: {project_name} ---")

        setup = setup_project(
            project_name, dag_bench_root, tmp_bench_dir, dee_cli_path, db_type,
            max_mem=max_mem, threads=threads,
        )
        if not setup:
            continue

        dest_project_path = setup["dest_project_path"]
        dag_json_path = setup["dag_json_path"]
        connections_json = setup["connections_json"]
        target = setup["target"]
        opt_dag_json_path = dest_project_path / "dag_opt.json"

        # 3. optimize
        print(f"Optimizing DAG for {project_name}...")
        explain_path = None
        if explain_dir:
            explain_dir_path = Path(explain_dir)
            explain_dir_path.mkdir(parents=True, exist_ok=True)
            explain_path = explain_dir_path / f"{project_name}.html"

        opt_cmd = build_opt_cmd(
            dee_cli_path,
            connections_json,
            target,
            opt_dag_json_path,
            dag_json_path,
            omp_top=omp_top,
            omp_node_centrality=omp_node_centrality,
            omp_exhaust=omp_exhaust,
            omp_no_pushdown=omp_no_pushdown,
            enable=enable,
            disable=disable,
            hmp_downstream_cost=hmp_downstream_cost,
            hmp_max_runs=hmp_max_runs,
            hmp_top_cpu_time=hmp_top_cpu_time,
            hmp_show_operators=hmp_show_operators,
            hmp_show_nodes=hmp_show_nodes,
            hmp_normalize_with_cardinality=hmp_normalize_with_cardinality,
            hmp_strategy=hmp_strategy,
            hmp_no_pushdown=hmp_no_pushdown,
            explain_path=explain_path,
            profile_iterations=profile,
        )

        opt_stats_json = run_cmd(opt_cmd)
        opt_stats = json.loads(opt_stats_json)

        num_iters = n if deep_dive else 1
        print(
            f"Running {num_iters} iteration(s) for original and optimized versions..."
        )

        original_times, original_resource_samples = run_multiple_times(
            dee_cli_path, connections_json, target, dag_json_path, num_iters,
            profile=profile, profile_dir=dest_project_path, profile_interval_ms=profile_interval_ms,
        )
        optimized_times, optimized_resource_samples = run_multiple_times(
            dee_cli_path, connections_json, target, opt_dag_json_path, num_iters,
            profile=profile, profile_dir=dest_project_path, profile_interval_ms=profile_interval_ms,
        )

        original_time = sum(original_times) / num_iters
        optimized_time = sum(optimized_times) / num_iters

        result = {
            "project": project_name,
            "original_time": original_time,
            "optimized_time": optimized_time,
            "speedup": original_time / optimized_time if optimized_time > 0 else 0,
            "opt_stats": opt_stats,
        }

        if deep_dive:
            result["original_distribution"] = original_times
            result["optimized_distribution"] = optimized_times

        if profile:
            result["original_resource_samples"] = original_resource_samples
            result["optimized_resource_samples"] = optimized_resource_samples

        results.append(result)

    return results


def benchmark_pushdown_comparison(
    config_file,
    dag_bench_root,
    dee_cli_path,
    db_type,
    n=5,
    max_mem=None,
    threads=None,
    hmp_downstream_cost=False,
    hmp_max_runs=None,
    hmp_top_cpu_time=None,
    hmp_normalize_with_cardinality=False,
    hmp_strategy=None,
    regression_tolerance=0.02,
    profile=False,
    profile_interval_ms=None,
):
    """A/B benchmark: for every project in `config_file`, run HMP alone and
    HMP+pushdown, then execute *both* resulting DAGs `n` times each (plus a
    warmup fraction) to get a stable read on whether pushdown actually helps.

    Both optimizer runs share the exact same HMP settings — the only
    difference between them is whether the pushdown pass also runs — so any
    runtime difference is attributable to pushdown itself, not to HMP making
    a different materialization choice.
    """
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    projects_to_run = config.get("projects", [])
    results = []

    tmp_bench_dir = Path("tmp_projects_pushdown_compare")
    if tmp_bench_dir.exists():
        shutil.rmtree(tmp_bench_dir)
    tmp_bench_dir.mkdir(parents=True)

    for project_name in projects_to_run:
        print(f"\n--- Comparing pushdown for Project: {project_name} ---")

        setup = setup_project(
            project_name, dag_bench_root, tmp_bench_dir, dee_cli_path, db_type,
            max_mem=max_mem, threads=threads,
        )
        if not setup:
            continue

        dest_project_path = setup["dest_project_path"]
        dag_json_path = setup["dag_json_path"]
        connections_json = setup["connections_json"]
        target = setup["target"]

        hmp_only_dag_path = dest_project_path / "dag_hmp.json"
        hmp_pushdown_dag_path = dest_project_path / "dag_hmp_pushdown.json"

        hmp_kwargs = dict(
            hmp_downstream_cost=hmp_downstream_cost,
            hmp_max_runs=hmp_max_runs,
            hmp_top_cpu_time=hmp_top_cpu_time,
            hmp_normalize_with_cardinality=hmp_normalize_with_cardinality,
            hmp_strategy=hmp_strategy,
        )

        print(f"Optimizing {project_name} with HMP only (no pushdown)...")
        hmp_only_stats = json.loads(run_cmd(build_opt_cmd(
            dee_cli_path, connections_json, target, hmp_only_dag_path, dag_json_path,
            enable="hmp", profile_iterations=profile, **hmp_kwargs,
        )))

        print(f"Optimizing {project_name} with HMP + pushdown...")
        hmp_pushdown_stats = json.loads(run_cmd(build_opt_cmd(
            dee_cli_path, connections_json, target, hmp_pushdown_dag_path, dag_json_path,
            enable="hmp,pushdown", profile_iterations=profile, **hmp_kwargs,
        )))

        print(f"Running {n} iteration(s) for HMP-only and HMP+pushdown DAGs...")
        hmp_only_times, hmp_only_resource_samples = run_multiple_times(
            dee_cli_path, connections_json, target, hmp_only_dag_path, n,
            profile=profile, profile_dir=dest_project_path, profile_interval_ms=profile_interval_ms,
        )
        hmp_pushdown_times, hmp_pushdown_resource_samples = run_multiple_times(
            dee_cli_path, connections_json, target, hmp_pushdown_dag_path, n,
            profile=profile, profile_dir=dest_project_path, profile_interval_ms=profile_interval_ms,
        )

        hmp_only_arr = np.array(hmp_only_times)
        hmp_pushdown_arr = np.array(hmp_pushdown_times)

        hmp_only_mean = hmp_only_arr.mean()
        hmp_pushdown_mean = hmp_pushdown_arr.mean()
        speedup = hmp_only_mean / hmp_pushdown_mean if hmp_pushdown_mean > 0 else 0

        # A speedup below (1 - tolerance) means pushdown made this project's
        # DAG measurably *slower* than HMP alone — the one outcome pushdown
        # should never produce, so flag it plainly rather than let it hide
        # in an average.
        is_regression = speedup < (1 - regression_tolerance)

        result = {
            "project": project_name,
            "hmp_only_time": hmp_only_mean,
            "hmp_pushdown_time": hmp_pushdown_mean,
            "hmp_only_median": float(np.median(hmp_only_arr)),
            "hmp_pushdown_median": float(np.median(hmp_pushdown_arr)),
            "hmp_only_std": float(hmp_only_arr.std()),
            "hmp_pushdown_std": float(hmp_pushdown_arr.std()),
            "speedup": speedup,
            "is_regression": bool(is_regression),
            "hmp_only_distribution": hmp_only_times,
            "hmp_pushdown_distribution": hmp_pushdown_times,
            "hmp_only_stats": hmp_only_stats,
            "hmp_pushdown_stats": hmp_pushdown_stats,
        }

        if profile:
            result["hmp_only_resource_samples"] = hmp_only_resource_samples
            result["hmp_pushdown_resource_samples"] = hmp_pushdown_resource_samples

        results.append(result)

    return results


def visualize(results):
    if not results:
        print("No results to visualize.")
        return

    # Print summary table
    df = pd.DataFrame(results)
    print("\nBenchmark Results:")
    cols = ["project", "original_time", "optimized_time", "speedup"]
    print(df[cols].to_string())

    if any("original_distribution" in r for r in results):
        print("\nDeep Dive Statistics:")
        for r in results:
            if "original_distribution" in r:
                print(f"Project {r['project']}:")
                for label, dist in [
                    ("Original", r["original_distribution"]),
                    ("Optimized", r["optimized_distribution"]),
                ]:
                    arr = np.array(dist)
                    print(
                        f"  {label}: median={np.median(arr):.4f}s, min={arr.min():.4f}s, max={arr.max():.4f}s, std={arr.std():.4f}s"
                    )

    plot_path = "results.png"
    plot_data(results, plot_path)

    if any(r.get("original_resource_samples") for r in results):
        plot_resource_usage(results, "resource_usage.png", "original_resource_samples", "optimized_resource_samples")


def visualize_pushdown_comparison(results):
    if not results:
        print("No results to visualize.")
        return

    df = pd.DataFrame(results)
    print("\nPushdown A/B Comparison Results:")
    cols = ["project", "hmp_only_time", "hmp_pushdown_time", "speedup", "is_regression"]
    print(df[cols].to_string())

    print("\nPer-project distributions:")
    for r in results:
        print(f"Project {r['project']}:")
        for label, key in [
            ("HMP only", "hmp_only_distribution"),
            ("HMP + pushdown", "hmp_pushdown_distribution"),
        ]:
            arr = np.array(r[key])
            print(
                f"  {label}: mean={arr.mean():.4f}s, median={np.median(arr):.4f}s, "
                f"min={arr.min():.4f}s, max={arr.max():.4f}s, std={arr.std():.4f}s"
            )

    regressions = [r["project"] for r in results if r["is_regression"]]
    if regressions:
        print(
            f"\nWARNING: pushdown made these projects SLOWER than HMP alone: {regressions}"
        )
    else:
        print("\nPushdown never made any project slower than HMP alone.")

    plot_pushdown_comparison(results, "pushdown_comparison.png")

    if any(r.get("hmp_only_resource_samples") for r in results):
        plot_resource_usage(
            results, "resource_usage.png",
            "hmp_only_resource_samples", "hmp_pushdown_resource_samples",
            variant_a_label="HMP only", variant_b_label="HMP + pushdown",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to yaml config file")
    parser.add_argument(
        "--db-type",
        choices=["duckdb", "postgres"],
        default="duckdb",
        help="Database type to benchmark (duckdb or postgres)",
    )
    parser.add_argument(
        "--deep-dive",
        action="store_true",
        help="Run optimized and original versions multiple times to compare distributions",
    )
    parser.add_argument(
        "--compare-pushdown",
        action="store_true",
        help=(
            "Run a separate A/B benchmark comparing HMP alone vs HMP+pushdown for every "
            "project in --config, each run --n times for a stable comparison. Ignores "
            "--enable/--disable/--omp-*/--deep-dive."
        ),
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of iterations per version when --deep-dive or --compare-pushdown is enabled",
    )
    parser.add_argument(
        "--regression-tolerance",
        type=float,
        default=0.02,
        help=(
            "Fractional tolerance (default 0.02 = 2%%) below which HMP+pushdown being "
            "slower than HMP alone is treated as noise rather than a real regression. "
            "Only used with --compare-pushdown."
        ),
    )
    parser.add_argument(
        "--max-mem",
        help="Maximum memory for DuckDB connections (e.g., '10GB', '512MB'). Only available for duckdb.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        help="Maximum number of threads for DuckDB connections. Only available for duckdb.",
    )
    parser.add_argument(
        "--omp-top",
        type=int,
        help="Number of top views to consider for materialization in OMPPass",
    )
    parser.add_argument(
        "--omp-node-centrality",
        choices=["outdegree", "paths"],
        help="Node centrality metric for OMPPass (outdegree or paths)",
    )
    parser.add_argument(
        "--omp-exhaust",
        action="store_true",
        help="Disable early termination in OMPPass and evaluate all candidate plans",
    )
    parser.add_argument(
        "--omp-no-pushdown",
        action="store_true",
        help="Disable running the pushdown pass on each OMP candidate DAG before benchmarking it (enabled by default)",
    )
    parser.add_argument(
        "--enable",
        help="Comma-separated list of optimization passes to enable",
    )
    parser.add_argument(
        "--disable",
        help="Comma-separated list of optimization passes to disable",
    )
    parser.add_argument(
        "--hmp-downstream-cost",
        action="store_true",
        help=(
            "Rank HMP VIEW candidates by the total cost of duplicate computation "
            "they introduce downstream, instead of an estimated cost to run the "
            "VIEW itself"
        ),
    )
    parser.add_argument(
        "--hmp-max-runs",
        type=int,
        help="Max number of DAG runs HMPPass uses to search for materialization candidates",
    )
    parser.add_argument(
        "--hmp-top-cpu-time",
        type=float,
        help="Fraction (0, 1.0] of total operator CPU time used to build HMPPass's working set",
    )
    parser.add_argument(
        "--hmp-show-operators",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help=(
            "Log a table of HMP operator rankings after the baseline run. "
            "Optionally pass a path to also write the table there."
        ),
    )
    parser.add_argument(
        "--hmp-show-nodes",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help=(
            "Log a table of HMP node rankings after the baseline run. "
            "Optionally pass a path to also write the table there."
        ),
    )
    parser.add_argument(
        "--hmp-normalize-with-cardinality",
        action="store_true",
        help=(
            "Rank HMP VIEW candidates by total CPU time divided by the View's "
            "estimated cardinality (from its EXPLAIN plan), instead of raw total CPU time"
        ),
    )
    parser.add_argument(
        "--hmp-strategy",
        choices=["breadth", "greedy"],
        help="Search strategy HMP uses to select which VIEWs to materialize",
    )
    parser.add_argument(
        "--hmp-no-pushdown",
        action="store_true",
        help="Disable running the pushdown pass on each HMP candidate DAG before benchmarking it (enabled by default)",
    )
    parser.add_argument(
        "--explain",
        metavar="DIR",
        help="Directory to write per-project optimizer explain HTML reports to",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Capture CPU/memory/disk timeseries for every timed iteration via dee-cli --profile",
    )
    parser.add_argument(
        "--profile-interval-ms",
        type=int,
        default=None,
        help="Sampling interval in ms for --profile (defaults to dee-cli's built-in 250ms)",
    )
    args = parser.parse_args()

    if args.max_mem and args.db_type != "duckdb":
        print("Error: --max-mem is only supported for duckdb backend.")
        exit(1)

    if args.threads and args.db_type != "duckdb":
        print("Error: --threads is only supported for duckdb backend.")
        exit(1)

    dag_bench = os.environ.get("DAG_BENCH")
    if not dag_bench:
        print("Error: DAG_BENCH environment variable not set")
        exit(1)

    dee_root = os.environ.get("DEE_PATH", os.getcwd())
    dee_cli = os.path.abspath(os.path.join(dee_root, "target/release/dee-cli"))
    if not os.path.exists(dee_cli):
        print(
            f"Error: dee-cli not found at {dee_cli}. Please build the project or set DEE_PATH."
        )
        exit(1)

    if args.compare_pushdown:
        results = benchmark_pushdown_comparison(
            args.config,
            dag_bench,
            dee_cli,
            args.db_type,
            n=max(args.n, 2),
            max_mem=args.max_mem,
            threads=args.threads,
            hmp_downstream_cost=args.hmp_downstream_cost,
            hmp_max_runs=args.hmp_max_runs,
            hmp_top_cpu_time=args.hmp_top_cpu_time,
            hmp_normalize_with_cardinality=args.hmp_normalize_with_cardinality,
            hmp_strategy=args.hmp_strategy,
            regression_tolerance=args.regression_tolerance,
            profile=args.profile,
            profile_interval_ms=args.profile_interval_ms,
        )
        visualize_pushdown_comparison(results)

        results_path = Path("pushdown_comparison_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {results_path.absolute()}")
        return

    results = benchmark(
        args.config,
        dag_bench,
        dee_cli,
        args.db_type,
        deep_dive=args.deep_dive,
        n=args.n,
        max_mem=args.max_mem,
        threads=args.threads,
        omp_top=args.omp_top,
        omp_node_centrality=args.omp_node_centrality,
        omp_exhaust=args.omp_exhaust,
        omp_no_pushdown=args.omp_no_pushdown,
        enable=args.enable,
        disable=args.disable,
        hmp_downstream_cost=args.hmp_downstream_cost,
        hmp_max_runs=args.hmp_max_runs,
        hmp_top_cpu_time=args.hmp_top_cpu_time,
        hmp_show_operators=args.hmp_show_operators,
        hmp_show_nodes=args.hmp_show_nodes,
        hmp_normalize_with_cardinality=args.hmp_normalize_with_cardinality,
        hmp_strategy=args.hmp_strategy,
        hmp_no_pushdown=args.hmp_no_pushdown,
        explain_dir=args.explain,
        profile=args.profile,
        profile_interval_ms=args.profile_interval_ms,
    )
    visualize(results)

    # Save results to JSON for record
    results_path = Path("results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_path.absolute()}")

    if args.deep_dive:
        plot_deep_dive(results, "deep-dive.png")

    if any(r.get("opt_stats", {}).get("HMPPass") for r in results):
        plot_hmp_iterations(results, "hmp_iterations.png")

    if any(r.get("opt_stats", {}).get("OMPPass") for r in results):
        plot_omp_iterations(results, "omp_iterations.png")


if __name__ == "__main__":
    main()
