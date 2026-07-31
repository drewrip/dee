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
                # Skip attempts with no measurable runtime (cancelled or baseline).
                if value.startswith("cancelled(") or value.startswith("baseline("):
                    continue
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
    ax.boxplot(plot_data_points, tick_labels=project_names)

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


def plot_pushdown_comparison(results, output_path):
    """Bar chart comparing HMP-only vs HMP+pushdown mean runtime per project,
    from `benchmark_pushdown_comparison`'s results (each a dict with
    `hmp_only_distribution` / `hmp_pushdown_distribution` lists of per-run
    wall-clock seconds)."""
    if not results:
        print("No results to plot.")
        return

    projects = [r["project"] for r in results]
    hmp_only_means = [np.mean(r["hmp_only_distribution"]) for r in results]
    hmp_pushdown_means = [np.mean(r["hmp_pushdown_distribution"]) for r in results]
    hmp_only_stds = [np.std(r["hmp_only_distribution"]) for r in results]
    hmp_pushdown_stds = [np.std(r["hmp_pushdown_distribution"]) for r in results]

    x = np.arange(len(projects))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))
    bars_only = ax.bar(
        x - width / 2, hmp_only_means, width, yerr=hmp_only_stds,
        label="HMP only", color="darkorange", capsize=4,
    )
    bars_pushdown = ax.bar(
        x + width / 2, hmp_pushdown_means, width, yerr=hmp_pushdown_stds,
        label="HMP + pushdown", color="steelblue", capsize=4,
    )

    for i, r in enumerate(results):
        speedup = r["speedup"]
        color = "red" if r["is_regression"] else "green"
        top = max(hmp_only_means[i] + hmp_only_stds[i], hmp_pushdown_means[i] + hmp_pushdown_stds[i])
        ax.text(
            x[i], top, f"{speedup:.2f}x", ha="center", va="bottom",
            fontweight="bold", color=color,
        )

    ax.set_ylabel("Mean Runtime (s)")
    ax.set_title("HMP Only vs HMP + Pushdown Runtime by Project")
    ax.set_xticks(x)
    ax.set_xticklabels(projects, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    plt.savefig(output_path)
    print(f"\nPushdown comparison visualization saved to {output_path}")


def _plot_iteration_resource_usage(ax, project_name, pass_name, iterations):
    """Peak CPU%/memory per iteration, from each `IterationStat`'s
    `system_samples` (only present when the optimizer ran with
    `--profile-iterations`). Plotted on twin y-axes against iteration
    number, next to the runtime-vs-iteration line so the two are directly
    comparable."""
    iters = [it["iteration"] for it in iterations if it.get("system_samples")]
    peak_cpu = [
        max((s.get("cpu_percent") or 0.0) for s in it["system_samples"])
        for it in iterations
        if it.get("system_samples")
    ]
    peak_mem = [
        max((s.get("memory_bytes") or 0) for s in it["system_samples"])
        for it in iterations
        if it.get("system_samples")
    ]

    if not iters:
        ax.text(0.5, 0.5, "No resource samples", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    ax.plot(iters, peak_cpu, marker="o", color="steelblue", linewidth=2, markersize=6, label="Peak CPU (%)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Peak CPU (%)", color="steelblue")
    ax.tick_params(axis="y", labelcolor="steelblue")
    ax.set_xticks(iters)
    ax.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax.twinx()
    peak_mem_mb = [m / (1024 * 1024) for m in peak_mem]
    ax2.plot(iters, peak_mem_mb, marker="s", color="darkorange", linewidth=2, markersize=6, label="Peak Memory (MB)")
    ax2.set_ylabel("Peak Memory (MB)", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax.set_title(f"{project_name}: {pass_name} Resource Usage over Iterations")


def _plot_pass_iterations(results, output_path, pass_name):
    if not results:
        print("No results to plot.")
        return

    pass_results = []
    for r in results:
        pass_stats = r.get("opt_stats", {}).get(pass_name)
        if not pass_stats or "iterations" not in pass_stats:
            continue
        iterations = json.loads(pass_stats["iterations"])
        if not iterations:
            continue
        pass_results.append((r.get("project", "Unknown"), iterations))

    if not pass_results:
        print(f"No {pass_name} iteration data found to plot.")
        return

    # `system_samples` per iteration is only present when the DAG was
    # optimized with `--profile-iterations` (dee-benchmark passes this
    # automatically whenever --profile is set). When present, add a second
    # column per project plotting how CPU/memory usage moved across
    # iterations, side by side with the runtime line.
    has_profile = any(
        it.get("system_samples") for _, iterations in pass_results for it in iterations
    )
    n = len(pass_results)
    n_cols = 2 if has_profile else 1
    fig, axes = plt.subplots(
        n, n_cols, figsize=(10 * n_cols, 4.5 * n), squeeze=False
    )

    for idx, (project_name, iterations) in enumerate(pass_results):
        ax = axes[idx][0]
        iters = [it["iteration"] for it in iterations]
        runtimes_ms = [it["runtime_ms"] for it in iterations]
        baseline = runtimes_ms[0]

        # Distance of each iteration's runtime to the baseline (first run).
        # Negative distances mean that iteration was faster than baseline.
        distances = [rt - baseline for rt in runtimes_ms]
        total_distance = sum(distances)
        net_change_pct = (total_distance / baseline * 100) if baseline else 0.0

        # Only iterations after the baseline are actual optimization attempts;
        # the baseline's distance to itself (0) isn't a candidate "best improvement".
        trial_distances = distances[1:]
        if total_distance < 0:
            payoff_label = "Done"
        elif not trial_distances or min(trial_distances) >= 0:
            payoff_label = "Never"
        else:
            best_improvement = min(trial_distances)
            payoff_iters = total_distance / abs(best_improvement)
            payoff_label = f"{payoff_iters:.1f}"

        ax.plot(iters, runtimes_ms, marker="o", color="steelblue", linewidth=2, markersize=6)
        ax.axhline(y=baseline, color="red", linestyle="--", alpha=0.6, label="Baseline (iteration 1)")

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Runtime (ms)")
        ax.set_title(f"{project_name}: {pass_name} Runtime over Iterations")
        ax.set_xticks(iters)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right")

        annotation = (
            f"Total net change: {net_change_pct:+.1f}%\n"
            f"Expected payoff iterations: {payoff_label}"
        )
        ax.text(
            0.02,
            0.02,
            annotation,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="steelblue"),
        )

        if has_profile:
            _plot_iteration_resource_usage(axes[idx][1], project_name, pass_name, iterations)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"\n{pass_name} iteration visualization saved to {output_path}")


def plot_hmp_iterations(results, output_path):
    _plot_pass_iterations(results, output_path, "HMPPass")


def plot_omp_iterations(results, output_path):
    _plot_pass_iterations(results, output_path, "OMPPass")


def plot_resource_usage(
    results, output_path,
    variant_a_key="original_resource_samples", variant_b_key="optimized_resource_samples",
    variant_a_label="Original", variant_b_label="Optimized",
):
    """Per-project stacked subplots (one row per project, one column per
    metric) plotting each variant's CPU/memory/disk timeseries, sampled by
    `dee-cli run --profile` (see `run_multiple_times`). Every timed
    iteration's timeseries is drawn as a translucent line so run-to-run
    variance is visible; metrics with no data across all results (e.g.
    Postgres, which doesn't report disk usage) are skipped entirely."""
    projects_with_samples = [
        r for r in results if r.get(variant_a_key) or r.get(variant_b_key)
    ]
    if not projects_with_samples:
        print("No resource usage samples found to plot (re-run with --profile).")
        return

    candidate_metrics = [
        ("cpu_percent", "CPU (%)"),
        ("memory_bytes", "Memory (bytes)"),
        ("disk_bytes", "Disk size (bytes)"),
        ("read_bytes", "Disk read (bytes)"),
        ("written_bytes", "Disk written (bytes)"),
    ]

    def metric_has_data(field):
        for r in projects_with_samples:
            for key in (variant_a_key, variant_b_key):
                for iteration in r.get(key) or []:
                    if any(s.get(field) is not None for s in iteration):
                        return True
        return False

    active_metrics = [(field, label) for field, label in candidate_metrics if metric_has_data(field)]
    if not active_metrics:
        print("No resource usage samples found to plot (re-run with --profile).")
        return

    n_rows = len(projects_with_samples)
    n_cols = len(active_metrics)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    variants = [(variant_a_key, variant_a_label, "darkorange"), (variant_b_key, variant_b_label, "steelblue")]

    for row, r in enumerate(projects_with_samples):
        for col, (field, label) in enumerate(active_metrics):
            ax = axes[row][col]
            for key, variant_label, color in variants:
                for i, iteration in enumerate(r.get(key) or []):
                    xs = [s["elapsed_ms"] / 1000.0 for s in iteration if s.get(field) is not None]
                    ys = [s[field] for s in iteration if s.get(field) is not None]
                    if not xs:
                        continue
                    # Only label the first iteration per variant so the legend
                    # doesn't grow with the number of iterations.
                    ax.plot(
                        xs, ys, color=color, alpha=0.5, linewidth=1.5,
                        label=variant_label if i == 0 else None,
                    )
            ax.set_title(f"{r['project']} — {label}")
            ax.set_xlabel("Elapsed (s)")
            ax.set_ylabel(label)
            ax.grid(True, linestyle="--", alpha=0.5)
            if ax.get_legend_handles_labels()[0]:
                ax.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"\nResource usage visualization saved to {output_path}")


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
    parser.add_argument(
        "--hmp-iterations",
        action="store_true",
        help="Generate the HMPPass runtime-over-iterations plot instead of the standard reduction plot",
    )
    parser.add_argument(
        "--omp-iterations",
        action="store_true",
        help="Generate the OMPPass runtime-over-iterations plot instead of the standard reduction plot",
    )
    parser.add_argument(
        "--resource-usage",
        action="store_true",
        help="Generate the CPU/memory/disk-over-time plot instead of the standard reduction plot",
    )

    args = parser.parse_args()

    if not Path(args.results).exists():
        print(f"Error: {args.results} not found.")
        return

    with open(args.results, "r") as f:
        results = json.load(f)

    if args.deep_dive:
        plot_deep_dive(results, args.output)
    elif args.hmp_iterations:
        plot_hmp_iterations(results, args.output)
    elif args.omp_iterations:
        plot_omp_iterations(results, args.output)
    elif args.resource_usage:
        plot_resource_usage(results, args.output)
    else:
        plot_data(results, args.output)


if __name__ == "__main__":
    main()
