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


def generate_profiles_json(src_project_dir, dest_project_dir, requested_db_type, max_mem=None):
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

    dee_profiles = {}
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

        dee_profiles[target_key] = dee_cfg
        final_target = target_key
        # We only need one output of the requested type
        break

    if not dee_profiles:
        return None, None

    profiles_json_path = dest_project_dir / "profiles.json"
    with open(profiles_json_path, "w") as f:
        json.dump(dee_profiles, f, indent=4)

    return str(profiles_json_path), final_target


def benchmark(
    config_file,
    dag_bench_root,
    dee_cli_path,
    db_type,
    deep_dive=False,
    n=5,
    max_mem=None,
    omp_top=None,
    omp_cost=None,
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

        profiles_json, target = generate_profiles_json(
            src_project_path, dest_project_path, db_type, max_mem=max_mem
        )
        if not profiles_json:
            print(
                f"Warning: Could not generate profiles.json for {project_name} with type {db_type}"
            )
            continue

        # 3. optimize
        print(f"Optimizing DAG for {project_name}...")
        opt_cmd = [
            dee_cli_path,
            "opt",
            "--stats",
            "--profiles",
            profiles_json,
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

        opt_stats_json = run_cmd(opt_cmd)
        opt_stats = json.loads(opt_stats_json)

        if deep_dive:
            print(
                f"Deep dive: Running {n} iterations for original and optimized versions..."
            )

            def run_multiple_times(dag_path, iterations):
                warmup_iters = int(iterations * 0.1)
                if warmup_iters > 0:
                    print(f"  Running {warmup_iters} warmup iterations...")
                    for _ in range(warmup_iters):
                        run_cmd(
                            [
                                dee_cli_path,
                                "run",
                                "--profiles",
                                profiles_json,
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
                            "--profiles",
                            profiles_json,
                            "--target",
                            target,
                            str(dag_path),
                        ]
                    )
                    times.append(time.time() - start)
                return times

            original_times = run_multiple_times(dag_json_path, n)
            optimized_times = run_multiple_times(opt_dag_json_path, n)

            original_time = sum(original_times) / n
            optimized_time = sum(optimized_times) / n

            results.append(
                {
                    "project": project_name,
                    "original_time": original_time,
                    "optimized_time": optimized_time,
                    "speedup": original_time / optimized_time
                    if optimized_time > 0
                    else 0,
                    "original_distribution": original_times,
                    "optimized_distribution": optimized_times,
                    "opt_stats": opt_stats,
                }
            )
        else:
            # Extract values from OMPPass stats
            omp_stats = opt_stats.get("OMPPass", {})
            # New format uses baseline_value/best_value
            # Old format used baseline_runtime/best_runtime
            original_val = float(omp_stats.get("baseline_value") or omp_stats.get("baseline_runtime", 0))
            optimized_val = float(omp_stats.get("best_value") or omp_stats.get("best_runtime", 0))

            # If the metric is Runtime, these are milliseconds.
            # For Cost, we'll still treat them similarly for the sake of the speedup ratio,
            # but note that the absolute values in original_time/optimized_time might not be seconds.
            original_time = original_val / 1000.0
            optimized_time = optimized_val / 1000.0

            results.append(
                {
                    "project": project_name,
                    "original_time": original_time,
                    "optimized_time": optimized_time,
                    "speedup": original_time / optimized_time
                    if optimized_time > 0
                    else 0,
                    "opt_stats": opt_stats,
                }
            )

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
        "--omp-top",
        type=int,
        help="Number of top views to consider for materialization in OMPPass",
    )
    parser.add_argument(
        "--omp-cost",
        choices=["actual", "estimate"],
        help="Cost metric for OMPPass (actual or estimate)",
    )
    args = parser.parse_args()

    if args.max_mem and args.db_type != "duckdb":
        print("Error: --max-mem is only supported for duckdb backend.")
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
        omp_top=args.omp_top,
        omp_cost=args.omp_cost,
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
