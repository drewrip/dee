import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
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

        baseline = float(omp_stats.get("baseline_value") or omp_stats.get("baseline_runtime", 0))
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


def plot_deep_dive(results, output_path):
    if not results:
        print("No results to plot.")
        return

    deep_dive_results = [r for r in results if "original_distribution" in r]
    if not deep_dive_results:
        print("No deep dive results found to plot.")
        return

    projects = [r["project"] for r in deep_dive_results]
    
    all_normalized_dists = []
    positions = []
    
    for i, r in enumerate(deep_dive_results):
        orig_dist = np.array(r["original_distribution"])
        opt_dist = np.array(r["optimized_distribution"])
        
        # Normalize by the median of the original distribution
        baseline = np.median(orig_dist)
        if baseline == 0:
            normalized_opt = opt_dist
        else:
            normalized_opt = opt_dist / baseline
        
        all_normalized_dists.append(normalized_opt)
        positions.append(i)

    fig, ax = plt.subplots(figsize=(12, 7))
    bp = ax.boxplot(all_normalized_dists, positions=positions, widths=0.5, 
                    patch_artist=True, showfliers=True)

    for patch in bp['boxes']:
        patch.set_facecolor('steelblue')

    # Annotate medians above each bar
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    offset = y_range * 0.02

    for i, dist in enumerate(all_normalized_dists):
        median_val = np.median(dist)
        # Find the top for annotation (max value or whisker top)
        max_val = np.max(dist)
        
        ax.text(positions[i], max_val + offset, f"{median_val:.3f}", 
                ha='center', va='bottom', fontweight='bold', color='steelblue')

    # Add a horizontal line at y=1.0 to represent the original median baseline
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Original Median')

    ax.set_ylabel('Relative Runtime (vs Original Median)')
    ax.set_title('Deep Dive Performance Comparison (Optimized vs Original Median)')
    ax.set_xticks(range(len(projects)))
    ax.set_xticklabels(projects, rotation=45, ha='right')
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"\nDeep dive visualization saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize benchmark results.")
    parser.add_argument(
        "--results", 
        default="results.json", 
        help="Path to the results JSON file (default: results.json)"
    )
    parser.add_argument(
        "--output", 
        default="results_plot.png", 
        help="Path to save the output plot (default: results_plot.png)"
    )
    parser.add_argument(
        "--deep-dive", 
        action="store_true", 
        help="Generate a deep-dive plot instead of the standard reduction plot"
    )
    
    args = parser.parse_args()

    if not Path(args.results).exists():
        print(f"Error: {args.results} not found.")
        return

    with open(args.results, "r") as f:
        results = json.load(f)

    if args.deep_dive:
        plot_deep_dive(results, args.output)
    else:
        plot_data(results, args.output)


if __name__ == "__main__":
    main()
