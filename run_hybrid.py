#!/usr/bin/env python
"""Entry point: Hybrid RL-Refined SA Microservice Migration."""

import time
import numpy as np
import pandas as pd

from core.data_loader import load_data, DEFAULT_TAXI_PATH, DEFAULT_SERVER_PATH
from algorithms.hybrid_sa_dqn import run_hybrid_microservice_fair
from evaluation.metrics import compute_score
from evaluation.plot import plot_training_curves

CHUNK_SIZE = 10000


def main():
    print("=" * 60)
    print("  Hybrid RL-Refined SA Microservice Migration — Full Run")
    print("=" * 60)

    df = load_data(DEFAULT_TAXI_PATH, chunk_size=CHUNK_SIZE)
    servers_df = pd.read_csv(DEFAULT_SERVER_PATH)
    print(f"  Edge servers loaded: {len(servers_df)}")

    print(f"\n{'=' * 60}")
    print("  Running Hybrid Microservice Migration ...")
    print(f"{'=' * 60}")
    t_start = time.time()
    results = run_hybrid_microservice_fair(df, servers_df)
    elapsed = time.time() - t_start

    M = results['total_migrations']
    V = results['total_violations']
    score = compute_score(M, V)

    print(f"\n{'=' * 60}")
    print(f"  RESULTS — Hybrid RL-Refined SA Microservice Migration")
    print(f"{'=' * 60}")
    print(f"  Migrations (M)         : {M}")
    print(f"  Violations (V)         : {V}")
    print(f"  Score (M + 0.5*V)      : {score:.1f}")
    print(f"  Total Reward           : {results['total_reward']:.2f}")
    print(f"  Hybrid Decisions       : {results['decision_count']}")
    n_losses = len(results['loss_history'])
    print(f"  Training Steps (loss)  : {n_losses}")
    if n_losses > 0:
        print(f"  Avg Loss (last 100)    : {np.mean(results['loss_history'][-100:]):.4f}")
    if results['reward_history']:
        print(f"  Avg Reward (last 100)  : {np.mean(results['reward_history'][-100:]):.4f}")
    print(f"  Elapsed Time           : {elapsed:.1f}s")
    print(f"{'=' * 60}")

    if n_losses > 0:
        plot_training_curves(results, save_path="outputs/hybrid_microservice_training.png",
                             title_prefix="Hybrid")


if __name__ == "__main__":
    main()
