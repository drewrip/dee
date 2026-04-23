import os
import subprocess
import time
import shutil
import yaml
import json
import argparse
from pathlib import Path
import pandas as pd
from .plot import plot_data


def run_cmd(cmd, cwd=None, env=None, capture=True):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, env=env, capture_output=capture, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        if not capture:
            print(f"Exit code: {result.returncode}")
        result.check_returncode()
    return result.stdout


def generate_profiles_json(src_project_dir, dest_project_dir, requested_db_type):
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


def benchmark(config_file, dag_bench_root, dee_cli_path, db_type):
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
            src_project_path, dest_project_path, db_type
        )
        if not profiles_json:
            print(
                f"Warning: Could not generate profiles.json for {project_name} with type {db_type}"
            )
            continue

        # 3. optimize (this now includes baseline and optimized runs)
        print(f"Optimizing DAG for {project_name}...")
        opt_stats_json = run_cmd(
            [
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

    # Print summary table
    df = pd.DataFrame(results)
    print("\nBenchmark Results:")
    print(df[["project", "original_time", "optimized_time", "speedup"]].to_string())

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
    args = parser.parse_args()

    dag_bench = os.environ.get("DAG_BENCH")
    if not dag_bench:
        print("Error: DAG_BENCH environment variable not set")
        exit(1)

    dee_root = os.environ.get("DEE_PATH", os.getcwd())
    dee_cli = os.path.abspath(os.path.join(dee_root, "target/debug/dee-cli"))
    if not os.path.exists(dee_cli):
        print(f"Error: dee-cli not found at {dee_cli}. Please build the project or set DEE_PATH.")
        exit(1)

    results = benchmark(args.config, dag_bench, dee_cli, args.db_type)
    visualize(results)

    # Save results to JSON for record
    results_path = Path("results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_path.absolute()}")


if __name__ == "__main__":
    main()
