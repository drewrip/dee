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
from .plot import plot_data, plot_deep_dive


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
                    print(f"Copying database from {src_db_path} to {dest_db_path}...")
                    shutil.copy2(src_db_path, dest_db_path)
                else:
                    print(
                        f"Warning: Source database file {src_db_path} does not exist."
                    )

                dee_cfg["database"] = str(dest_db_path.absolute())
            dee_cfg["num_connections"] = output_cfg.get("threads", 1)
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
    omp_cost=None,
    omp_node_centrality=None,
    enable=None,
    disable=None,
    hmp_no_plan_dups=False,
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
        src_project_path = Path(dag_bench_root) / "projects" / project_name
        dest_project_path = tmp_bench_dir / project_name

        if not src_project_path.exists():
            print(f"Error: Project {project_name} not found at {src_project_path}")
            continue

        shutil.copytree(src_project_path, dest_project_path)

        # 1. dbt compile
        dbt_target = "dev" if db_type == "duckdb" else "postgres"
        run_cmd(["dbt", "compile", "--target", dbt_target], cwd=dest_project_path)

        manifest_path = dest_project_path / "target" / "manifest.json"

        dag_json_path = dest_project_path / "dag.json"
        opt_dag_json_path = dest_project_path / "dag_opt.json"

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
            continue

        # 3. optimize
        print(f"Optimizing DAG for {project_name}...")
        opt_cmd = [
            dee_cli_path,
            "opt",
            "--stats",
            "--connections",
            connections_json,
            "--target",
            target,
            "-o",
            str(opt_dag_json_path),
            str(dag_json_path),
        ]
        if omp_top:
            opt_cmd.extend(["--omp-top", str(omp_top)])
        if omp_cost:
            opt_cmd.extend(["--omp-cost", omp_cost])
        if omp_node_centrality:
            opt_cmd.extend(["--omp-node-centrality", omp_node_centrality])
        if enable:
            opt_cmd.extend(["--enable", enable])
        if disable:
            opt_cmd.extend(["--disable", disable])
        if hmp_no_plan_dups:
            opt_cmd.append("--hmp-no-plan-dups")

        opt_stats_json = run_cmd(opt_cmd)
        opt_stats = json.loads(opt_stats_json)

        def run_multiple_times(dag_path, iterations):
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
            for i in range(iterations):
                print(f"  Iteration {i + 1}/{iterations}...")
                start = time.time()
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
                times.append(time.time() - start)
            return times

        num_iters = n if deep_dive else 1
        print(
            f"Running {num_iters} iteration(s) for original and optimized versions..."
        )

        original_times = run_multiple_times(dag_json_path, num_iters)
        optimized_times = run_multiple_times(opt_dag_json_path, num_iters)

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
        "--n",
        type=int,
        default=5,
        help="Number of iterations per version when --deep-dive is enabled",
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
        "--omp-cost",
        choices=["actual", "estimate"],
        help="Cost metric for OMPPass (actual or estimate)",
    )
    parser.add_argument(
        "--omp-node-centrality",
        choices=["outdegree", "paths"],
        help="Node centrality metric for OMPPass (outdegree or paths)",
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
        "--hmp-no-plan-dups",
        action="store_true",
        help="Disable duplicate operator counting within a single plan in HMPPass",
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
    dee_cli = os.path.abspath(os.path.join(dee_root, "target/debug/dee-cli"))
    if not os.path.exists(dee_cli):
        print(
            f"Error: dee-cli not found at {dee_cli}. Please build the project or set DEE_PATH."
        )
        exit(1)

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
        omp_cost=args.omp_cost,
        omp_node_centrality=args.omp_node_centrality,
        enable=args.enable,
        disable=args.disable,
        hmp_no_plan_dups=args.hmp_no_plan_dups,
    )
    visualize(results)

    # Save results to JSON for record
    results_path = Path("results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_path.absolute()}")

    if args.deep_dive:
        plot_deep_dive(results, "deep-dive.png")


if __name__ == "__main__":
    main()
