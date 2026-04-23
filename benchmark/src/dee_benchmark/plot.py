import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_data(results, output_path):
    if not results:
        print("No results to plot.")
        return

    plot_data_points = []
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
            plot_data_points.append(attempts)
            project_names.append(project_name)

    if not plot_data_points:
        print("No optimization attempt data found to plot.")
        return

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.boxplot(plot_data_points, labels=project_names)

    # Overlay raw points without jitter
    for i, attempts in enumerate(plot_data_points):
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

    plt.savefig(output_path)
    print(f"\nVisualization saved to {output_path}")


def plot_results(results_path, output_path):
    if not Path(results_path).exists():
        print(f"Error: {results_path} not found.")
        return

    with open(results_path, "r") as f:
        results = json.load(f)

    plot_data(results, output_path)


def main():
    results_json = "results.json"
    output_png = "results_plot.png"
    plot_results(results_json, output_png)


if __name__ == "__main__":
    main()
