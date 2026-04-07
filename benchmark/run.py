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

        # 3. run original
        print(f"Running original DAG for {project_name}...")
        start = time.perf_counter()
        run_cmd([dee_cli_path, "run", "--db-file", db_file, str(dag_json_path)])
        end = time.perf_counter()
        original_time = end - start

        # 4. optimize
        print(f"Optimizing DAG for {project_name}...")
        opt_output = run_cmd(
            [dee_cli_path, "opt", "--db-file", db_file, str(dag_json_path)]
        )
        with open(opt_dag_json_path, "w") as f:
            f.write(opt_output)

        # 5. run optimized
        print(f"Running optimized DAG for {project_name}...")
        start = time.perf_counter()
        run_cmd([dee_cli_path, "run", "--db-file", db_file, str(opt_dag_json_path)])
        end = time.perf_counter()
        optimized_time = end - start

        results.append(
            {
                "project": project_name,
                "original_time": original_time,
                "optimized_time": optimized_time,
                "speedup": original_time / optimized_time if optimized_time > 0 else 0,
            }
        )

    return results


def visualize(results):
    if not results:
        print("No results to visualize.")
        return

    df = pd.DataFrame(results)
    print("\nBenchmark Results:")
    print(df.to_string())

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    df.plot(x="project", y=["original_time", "optimized_time"], kind="bar", ax=ax)
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Performance Comparison: Original vs Optimized")
    plt.xticks(rotation=45)
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
