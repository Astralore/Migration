"""
Visualization utilities for microservice migration experiments.
Includes training curves, cost breakdown, and performance comparison charts.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Use a clean style for publication-ready figures
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})


def plot_training_curves(results, save_path="outputs/training_curves.png", title_prefix=""):
    """
    Plot Loss / Reward / Epsilon training curves and save to disk.

    Parameters
    ----------
    results : dict
        Must contain 'loss_history', 'reward_history', 'epsilon_history'.
    save_path : str
        Output image path.
    title_prefix : str
        Prefix for subplot titles (e.g. "DQN", "Hybrid").
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)

    # Loss
    ax = axes[0]
    losses = results.get('loss_history', [])
    if losses:
        ax.plot(losses, alpha=0.3, color='steelblue', linewidth=0.5)
        window = min(50, max(1, len(losses) // 10))
        if len(losses) >= window:
            smoothed = np.convolve(losses, np.ones(window) / window, mode='valid')
            ax.plot(range(window - 1, len(losses)), smoothed,
                    color='steelblue', linewidth=2, label=f'MA-{window}')
        ax.set_ylabel('MSE Loss')
        ax.set_title(f'{title_prefix} Training Loss'.strip())
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Reward
    ax = axes[1]
    rewards = results.get('reward_history', [])
    if rewards:
        ax.plot(rewards, alpha=0.3, color='coral', linewidth=0.5)
        window = min(50, max(1, len(rewards) // 10))
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')
            ax.plot(range(window - 1, len(rewards)), smoothed,
                    color='coral', linewidth=2, label=f'MA-{window}')
        ax.set_ylabel('DAG-level Reward')
        ax.set_title(f'{title_prefix} Per-Decision Reward'.strip())
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Epsilon
    ax = axes[2]
    epsilons = results.get('epsilon_history', [])
    if epsilons:
        ax.plot(epsilons, color='seagreen', linewidth=1.5)
        ax.set_ylabel('Epsilon')
        ax.set_xlabel('Decision Step')
        ax.set_title(f'{title_prefix} Exploration Rate Decay'.strip())
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Training curves saved to: {save_path}")


def plot_cost_breakdown(results_dict, save_path="outputs/cost_breakdown.png"):
    """
    Plot stacked bar chart showing cost breakdown (access latency, communication,
    migration) for each algorithm.

    Parameters
    ----------
    results_dict : dict
        Keys are algorithm names, values contain 'total_access_latency',
        'total_communication_cost', 'total_migration_cost'.
    save_path : str
        Output image path.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    algorithms = list(results_dict.keys())
    access_latency = [results_dict[alg].get('total_access_latency', 0) for alg in algorithms]
    communication = [results_dict[alg].get('total_communication_cost', 0) for alg in algorithms]
    migration = [results_dict[alg].get('total_migration_cost', 0) for alg in algorithms]

    x = np.arange(len(algorithms))
    width = 0.6

    fig, ax = plt.subplots(figsize=(10, 7))

    # Stacked bars
    bars1 = ax.bar(x, access_latency, width, label='Access Latency', color='#3498db', edgecolor='white')
    bars2 = ax.bar(x, communication, width, bottom=access_latency, label='Communication Cost', color='#e74c3c', edgecolor='white')
    bottom_for_migration = [a + c for a, c in zip(access_latency, communication)]
    bars3 = ax.bar(x, migration, width, bottom=bottom_for_migration, label='Migration Cost', color='#2ecc71', edgecolor='white')

    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Total Cost (Cumulative)')
    ax.set_title('Cost Breakdown by Algorithm (Proactive Mode)')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on each segment
    def add_labels(bars, values, bottom_values=None):
        for i, (bar, val) in enumerate(zip(bars, values)):
            if val > 0:
                height = bar.get_height()
                base = bottom_values[i] if bottom_values else 0
                ax.annotate(f'{val:.1f}',
                            xy=(bar.get_x() + bar.get_width() / 2, base + height / 2),
                            ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    add_labels(bars1, access_latency)
    add_labels(bars2, communication, access_latency)
    add_labels(bars3, migration, bottom_for_migration)

    # Add total on top
    for i, alg in enumerate(algorithms):
        total = access_latency[i] + communication[i] + migration[i]
        ax.annotate(f'Total: {total:.1f}',
                    xy=(i, total + max(access_latency) * 0.02),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Cost breakdown chart saved to: {save_path}")


def plot_performance_metrics(proactive_results, reactive_results, save_path="outputs/violation_comparison.png"):
    """
    Plot bar chart comparing Real Violations (V) between Proactive and Reactive
    modes for all algorithms.

    Parameters
    ----------
    proactive_results : dict
        Results from Proactive mode.
    reactive_results : dict
        Results from Reactive mode.
    save_path : str
        Output image path.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    algorithms = list(proactive_results.keys())
    pro_violations = [proactive_results[alg].get('total_violations', 0) for alg in algorithms]
    rea_violations = [reactive_results[alg].get('total_violations', 0) for alg in algorithms]

    x = np.arange(len(algorithms))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars_reactive = ax.bar(x - width / 2, rea_violations, width, label='Reactive Mode', color='#e74c3c', edgecolor='black', linewidth=0.8)
    bars_proactive = ax.bar(x + width / 2, pro_violations, width, label='Proactive Mode', color='#2ecc71', edgecolor='black', linewidth=0.8)

    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Real SLA Violations (V)')
    ax.set_title('SLA Violation Comparison: Proactive vs Reactive')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    def add_value_labels(bars, values):
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    add_value_labels(bars_reactive, rea_violations)
    add_value_labels(bars_proactive, pro_violations)

    # Add reduction percentage annotation
    for i, alg in enumerate(algorithms):
        rea_v = rea_violations[i]
        pro_v = pro_violations[i]
        if rea_v > 0:
            reduction = (rea_v - pro_v) / rea_v * 100
            y_pos = max(rea_v, pro_v) + max(rea_violations) * 0.05
            ax.annotate(f'{reduction:+.1f}%',
                        xy=(i, y_pos),
                        ha='center', va='bottom', fontsize=9, color='#27ae60' if reduction > 0 else '#c0392b')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Violation comparison chart saved to: {save_path}")
