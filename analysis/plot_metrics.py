import matplotlib.pyplot as plt
import numpy as np

from utils.metrics import analyze_simulation_metrics


def plot_metrics(
        sheep_pos_log, dog_pos_log, sheep_vel_log, dog_vel_log=None,
        save_path="metrics_analysis.png", algorithm_name="Algorithm"
):
    """Generate comprehensive metrics plots."""

    metrics = analyze_simulation_metrics(sheep_pos_log, sheep_vel_log)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(
        f"Flock Collective Behavior metrics — {algorithm_name}",
        fontsize=16, fontweight="bold"
    )

    time = metrics["time"]

    # Cohesion
    ax = axes[0]
    ax.plot(time, metrics["cohesion"], linewidth=2, color="blue", label="Cohesion")
    ax.axhline(
        np.mean(metrics["cohesion"]),
        color="blue", linestyle="--", alpha=0.5,
        label=f"Mean: {np.mean(metrics['cohesion']):.3f}m"
    )
    ax.fill_between(
        time, metrics["cohesion"], alpha=0.3, color="blue"
    )
    ax.set_ylabel("Cohesion (m)", fontsize=11)
    ax.set_title("Group Cohesion: Mean Distance to Barycenter", fontsize=12)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    # Polarization
    ax = axes[1]
    ax.plot(
        time, metrics["polarization"], linewidth=2, color="green",
        label="Polarization"
    )
    ax.axhline(
        np.mean(metrics["polarization"]),
        color="green", linestyle="--", alpha=0.5,
        label=f"Mean: {np.mean(metrics['polarization']):.3f}"
    )
    ax.fill_between(
        time, metrics["polarization"], alpha=0.3, color="green"
    )
    ax.set_ylabel("Polarization", fontsize=11)
    ax.set_title(
        "Group Polarization: Velocity Alignment (0=random, 1=aligned)",
        fontsize=12
    )
    ax.set_ylim([0, 1])
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    # Elongation
    ax = axes[2]
    ax.plot(
        time, metrics["elongation"], linewidth=2, color="red",
        label="Elongation"
    )
    ax.axhline(
        np.mean(metrics["elongation"]),
        color="red", linestyle="--", alpha=0.5,
        label=f"Mean: {np.mean(metrics['elongation']):.3f}"
    )
    ax.fill_between(
        time, metrics["elongation"], alpha=0.3, color="red"
    )
    ax.set_ylabel("Elongation Ratio", fontsize=11)
    ax.set_xlabel("Time (frames)", fontsize=11)
    ax.set_title("Group Elongation: Length/Width Ratio", fontsize=12)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"metrics plot saved to {save_path}")
    plt.show()

    # Print summary statistics
    print_metrics_summary(metrics, algorithm_name)

    return metrics


def print_metrics_summary(metrics, algorithm_name="Algorithm"):
    """Print summary statistics for metrics."""
    print("\n" + "=" * 70)
    print(f"COLLECTIVE BEHAVIOR METRICS SUMMARY — {algorithm_name}".center(70))
    print("=" * 70)
    print(f"\nCOHESION (mean distance to group center):")
    print(f"  Mean:     {np.mean(metrics['cohesion']):.4f} m")
    print(f"  Std Dev:  {np.std(metrics['cohesion']):.4f} m")
    print(f"  Min:      {np.min(metrics['cohesion']):.4f} m")
    print(f"  Max:      {np.max(metrics['cohesion']):.4f} m")

    print(f"\nPOLARIZATION (velocity alignment):")
    print(f"  Mean:     {np.mean(metrics['polarization']):.4f}")
    print(f"  Std Dev:  {np.std(metrics['polarization']):.4f}")
    print(f"  Min:      {np.min(metrics['polarization']):.4f}")
    print(f"  Max:      {np.max(metrics['polarization']):.4f}")

    print(f"\nELONGATION (length/width ratio):")
    print(f"  Mean:     {np.mean(metrics['elongation']):.4f}")
    print(f"  Std Dev:  {np.std(metrics['elongation']):.4f}")
    print(f"  Min:      {np.min(metrics['elongation']):.4f}")
    print(f"  Max:      {np.max(metrics['elongation']):.4f}")
    print("=" * 70 + "\n")


def plot_metrics_comparison(
        metrics_list, algorithm_names, save_path="metrics_comparison.png"
):
    """Generate comparison plots for multiple algorithms."""

    if len(metrics_list) != len(algorithm_names):
        raise ValueError("metrics_list and algorithm_names must have same length")

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 11))
    fig.suptitle(
        "Flock metrics",
        fontsize=16, fontweight="bold"
    )

    # Cohesion comparison
    ax = axes[0]
    for i, (metrics, name) in enumerate(zip(metrics_list, algorithm_names)):
        time = metrics["time"]
        color = colors[i % len(colors)]
        ax.plot(
            time, metrics["cohesion"], linewidth=2, label=name, color=color,
            alpha=0.8
        )
    ax.set_ylabel("Cohesion (m)", fontsize=11)
    ax.set_title("Group Cohesion: Mean Distance to Barycenter", fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)

    # Polarization comparison
    ax = axes[1]
    for i, (metrics, name) in enumerate(zip(metrics_list, algorithm_names)):
        time = metrics["time"]
        color = colors[i % len(colors)]
        ax.plot(
            time, metrics["polarization"], linewidth=2, label=name, color=color,
            alpha=0.8
        )
    ax.set_ylabel("Polarization", fontsize=11)
    ax.set_title(
        "Group Polarization: Velocity Alignment (0=random, 1=aligned)",
        fontsize=12
    )
    ax.set_ylim([0, 1])
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)

    # Elongation comparison
    ax = axes[2]
    for i, (metrics, name) in enumerate(zip(metrics_list, algorithm_names)):
        time = metrics["time"]
        color = colors[i % len(colors)]
        ax.plot(
            time, metrics["elongation"], linewidth=2, label=name, color=color,
            alpha=0.8
        )
    ax.set_ylabel("Elongation Ratio", fontsize=11)
    ax.set_xlabel("Time (frames)", fontsize=11)
    ax.set_title("Group Elongation: Length/Width Ratio", fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Comparison plot saved to {save_path}\n")
    plt.show()

    # Print comparison table
    print_comparison_summary(metrics_list, algorithm_names)


def print_comparison_summary(metrics_list, algorithm_names):
    """Print side-by-side comparison of metrics."""
    print("\n" + "=" * 100)
    print("ALGORITHM COMPARISON: COLLECTIVE BEHAVIOR METRICS".center(100))
    print("=" * 100)

    # Cohesion
    print("\nCOHESION (mean distance to group center) [m]:")
    print(f"{'Algorithm':<25} {'Mean':<15} {'Std Dev':<15} {'Min':<15} {'Max':<15}")
    print("-" * 100)
    for metrics, name in zip(metrics_list, algorithm_names):
        print(
            f"{name:<25} {np.mean(metrics['cohesion']):<15.4f} "
            f"{np.std(metrics['cohesion']):<15.4f} "
            f"{np.min(metrics['cohesion']):<15.4f} "
            f"{np.max(metrics['cohesion']):<15.4f}"
        )

    # Polarization
    print("\nPOLARIZATION (velocity alignment) [0-1]:")
    print(f"{'Algorithm':<25} {'Mean':<15} {'Std Dev':<15} {'Min':<15} {'Max':<15}")
    print("-" * 100)
    for metrics, name in zip(metrics_list, algorithm_names):
        print(
            f"{name:<25} {np.mean(metrics['polarization']):<15.4f} "
            f"{np.std(metrics['polarization']):<15.4f} "
            f"{np.min(metrics['polarization']):<15.4f} "
            f"{np.max(metrics['polarization']):<15.4f}"
        )

    # Elongation
    print("\nELONGATION (length/width ratio):")
    print(f"{'Algorithm':<25} {'Mean':<15} {'Std Dev':<15} {'Min':<15} {'Max':<15}")
    print("-" * 100)
    for metrics, name in zip(metrics_list, algorithm_names):
        print(
            f"{name:<25} {np.mean(metrics['elongation']):<15.4f} "
            f"{np.std(metrics['elongation']):<15.4f} "
            f"{np.min(metrics['elongation']):<15.4f} "
            f"{np.max(metrics['elongation']):<15.4f}"
        )
    print("=" * 100 + "\n")


if __name__ == "__main__":

    USE_ORIGINAL = True
    USE_STROMBOM = True

    # =========================================================================
    # ALGORITHM 1: Strombom Model
    # =========================================================================

    if USE_STROMBOM:
        print("\n" + "=" * 70)
        print("RUNNING STROMBOM ALGORITHM".center(70))
        print("=" * 70 + "\n")

        from movement_algorithms.Stormbom_movement_functions import (
            simulate_model_strombom_main
        )

        strombom_params = dict(
            num_sheep=30,
            box_length=100.0,
            ra_dist=2.0,  # sheep–sheep repulsion distance
            rs_range=25.0,  # dog detection range
            n_neighbors=4,
            d_step=1.0,
            ds=2.0,  # dog is faster than sheep
            h_weight=0.5,
            c_weight=1,
            ra_weight=2.0,
            rs_weight=1.5,
            e_noise=0.03,
            p_move=0.05,
            collecting_offset=0.0,
            driving_offset=10.0,
            goal=(0.0, 0.0),
            num_iterations=2000,
        )

        print("Simulation parameters:")
        for key, val in strombom_params.items():
            print(f"  {key:<25}: {val}")

        print("\nRunning simulation...")
        (sheep_pos_log, dog_pos_log, sheep_vel_log, dog_vel_log,
         dog_speeds_log, _, _, _) = simulate_model_strombom_main(**strombom_params)

        print("Analyzing metrics...")
        metrics_strombom = plot_metrics(
            sheep_pos_log, dog_pos_log, sheep_vel_log, dog_vel_log,
            save_path="metrics_strombom.png",
            algorithm_name="Strombom Model"
        )

    # =========================================================================
    # ALGORITHM 2: Original Model
    # =========================================================================

    if USE_ORIGINAL:
        print("\n" + "=" * 70)
        print("RUNNING ORIGINAL ALGORITHM".center(70))
        print("=" * 70 + "\n")

        from movement_algorithms.original_movement_functions import simulate_model

        original_params = dict(
            num_sheep=30,
            box_length=50,
            sheep_repulsion_radius=1.0,
            dog_repulsion_radius=5.0,
            num_neighbors_for_attraction=10,
            num_random_attraction_neighbors=5,
            num_alignment_neighbors=5,
            sheep_speed=0.1,
            dog_speed=0.25,
            persistence_weight=0.5,
            sheep_repulsion_weight=1.5,
            dog_repulsion_weight=2.0,
            noise_weight=0.05,
            attraction_weight=2.0,
            alignment_weight=0.1,
            non_cohesive_distance=3.0,
            driving_offset=2.0,
            collecting_offset=1.5,
            num_iterations=2000,
        )

        print("Simulation parameters:")
        for key, val in original_params.items():
            print(f"  {key:<35}: {val}")

        print("\nRunning simulation...")
        (sheep_pos_log, dog_pos_log, sheep_vel_log, dog_vel_log,
         dog_speeds_log, _, _, _) = simulate_model(**original_params)

        print("Analyzing metrics...")
        metrics_original = plot_metrics(
            sheep_pos_log, dog_pos_log, sheep_vel_log, dog_vel_log,
            save_path="metrics_original.png",
            algorithm_name="Original Model"
        )

    # =========================================================================
    # COMPARISON (only if both algorithms are enabled)
    # =========================================================================

    if USE_STROMBOM and USE_ORIGINAL:
        print("\n" + "=" * 70)
        print("GENERATING COMPARISON PLOTS".center(70))
        print("=" * 70 + "\n")

        plot_metrics_comparison(
            [metrics_strombom, metrics_original],
            ["Strombom Model", "Original Model"],
            save_path="metrics_comparison.png"
        )
