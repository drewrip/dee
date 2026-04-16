import os
import subprocess
import time
import shutil
import yaml
import json
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def run_cmd(cmd, cwd=None, env=None, capture=True):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, env=env, capture_output=capture, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        if not capture:
            print(f"Exit code: {result.returncode}")
        result.check_returncode()
    return result.stdout


def get_db_file_from_profiles(project_dir):
    profiles_path = project_dir / "profiles.yml"
    if not profiles_path.exists():
        # Check ~/.dbt/profiles.yml as fallback if needed,
        # but the prompt says local profiles.yml
        return None

    with open(profiles_path, "r") as f:
        profiles = yaml.safe_load(f)

    # Heuristic to find the db path in duckdb profile
    for profile_name, profile_cfg in profiles.items():
        if profile_name == "config":
            continue
        outputs = profile_cfg.get("outputs", {})
        target = profile_cfg.get("target", "dev")
        if target in outputs:
            path = outputs[target].get("path")
            if path:
                p = Path(path)
                if not p.is_absolute():
                    return str(project_dir / p)
                return path
    return None


def benchmark(config_file, dag_bench_root, dee_cli_path):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    projects_to_run = config.get("projects", [])
    results = []

    tmp_bench_dir = Path("benchmark/tmp_projects")
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
        run_cmd(["dbt", "compile"], cwd=dest_project_path)

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

        db_file_src = get_db_file_from_profiles(src_project_path)
        if not db_file_src:
            print(f"Warning: Could not find db_file for {project_name}")
            continue

        db_file_src_path = Path(db_file_src)
        db_file_local_path = dest_project_path / db_file_src_path.name

        if db_file_src_path.exists():
            print(
                f"Copying database from {db_file_src_path} to {db_file_local_path}..."
            )
            shutil.copy2(db_file_src_path, db_file_local_path)
        else:
            print(f"Warning: Source database file {db_file_src_path} does not exist.")
            # We'll continue, maybe it's created during run, but usually it should exist for dbt benchmarks

        db_file = str(db_file_local_path)

        # 3. optimize (this now includes baseline and optimized runs)
        print(f"Optimizing DAG for {project_name}...")
        opt_stats_json = run_cmd(
            [
                dee_cli_path,
                "opt",
                "--stats",
                "--db-file",
                db_file,
                "-o",
                str(opt_dag_json_path),
                str(dag_json_path),
            ]
        )
        opt_stats = json.loads(opt_stats_json)

        # Extract times from OMPPass stats (values are in milliseconds as strings)
        omp_stats = opt_stats.get("OMPPass", {})
        original_time_ms = float(omp_stats.get("baseline_runtime", 0))
        optimized_time_ms = float(omp_stats.get("best_runtime", 0))

        original_time = original_time_ms / 1000.0
        optimized_time = optimized_time_ms / 1000.0

        results.append(
            {
                "project": project_name,
                "original_time": original_time,
                "optimized_time": optimized_time,
                "speedup": original_time / optimized_time if optimized_time > 0 else 0,
                "opt_stats": opt_stats,
            }
        )

    return results


def visualize(results):
    if not results:
        print("No results to visualize.")
        return

    plot_data = []
    project_names = []

    for res in results:
        project_name = res.get("project", "Unknown")
        opt_stats = res.get("opt_stats", {})
        omp_stats = opt_stats.get("OMPPass", {})

        baseline = float(omp_stats.get("baseline_runtime", 0))
        if baseline <= 0:
            continue

        attempts = []
        for key, value in omp_stats.items():
            if key.startswith("attempt_"):
                attempt_runtime = float(value)
                # Calculate percent reduction: (baseline - attempt) / baseline * 100
                reduction = (baseline - attempt_runtime) / baseline * 100
                attempts.append(reduction)

        if attempts:
            plot_data.append(attempts)
            project_names.append(project_name)

    if not plot_data:
        print("No optimization attempt data found to visualize.")
        return

    # Print summary table
    df = pd.DataFrame(results)
    print("\nBenchmark Results:")
    print(df[["project", "original_time", "optimized_time", "speedup"]].to_string())

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.boxplot(plot_data, labels=project_names)

    # Overlay raw points without jitter
    for i, attempts in enumerate(plot_data):
        # x-position is 1-indexed for boxplot, set to center axis
        x_pos = i + 1
        x = [x_pos] * len(attempts)
        ax.scatter(x, attempts, alpha=0.6, color="red", s=25)

        # Annotate max value
        max_val = max(attempts)
        if max_val < 0:
            label = "0%*"
            # Position for negative values should be slightly above the whisker or 0
            # but since boxplot whiskers go below 0, we'll use 0 as a baseline if max is negative
            ann_pos = max(0, max_val) 
        else:
            label = f"{max_val:.1f}%"
            ann_pos = max_val

        # Add a bit of vertical offset (approx 2% of the y-axis range)
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        offset = y_range * 0.02
        ax.text(x_pos, ann_pos + offset, label, ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel("Reduction in Runtime (%)")
    ax.set_title("Distribution of Performance Improvements across Optimization Attempts")
    plt.xticks(rotation=45)
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    plot_path = "benchmark/results.png"
    plt.savefig(plot_path)
    print(f"\nVisualization saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to yaml config file")
    args = parser.parse_args()

    dag_bench = os.environ.get("DAG_BENCH")
    if not dag_bench:
        print("Error: DAG_BENCH environment variable not set")
        exit(1)

    dee_cli = os.path.abspath("target/debug/dee-cli")
    if not os.path.exists(dee_cli):
        print(f"Error: dee-cli not found at {dee_cli}. Please build the project first.")
        exit(1)

    results = benchmark(args.config, dag_bench, dee_cli)
    visualize(results)

    # Save results to JSON for record
    with open("benchmark/results.json", "w") as f:
        json.dump(results, f, indent=4)
