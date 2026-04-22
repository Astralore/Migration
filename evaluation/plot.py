"""Unified training curve plotting for DQN-based microservice migration algorithms."""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
