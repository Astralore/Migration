#!/usr/bin/env python
"""Entry point: SA Microservice DAG Migration."""

import time
import numpy as np
import pandas as pd

from core.data_loader import (
    load_data,
    DEFAULT_TAXI_PATH,
    DEFAULT_SERVER_PATH,
    DEFAULT_PROCESSED_TAXI_PATH,
)
from algorithms.sa import run_sa_microservice_fair
from evaluation.metrics import compute_score

# 与 run_comparison 一致：优先 `data/processed/*.csv`，缺失则从原始 CSV 走同参清洗
_ACTIVE_USERS_LIMIT = 100
_MIN_VEHICLE_POINTS = 100


def main():
    print("=" * 60)
    print("  SA Microservice DAG Migration — Full Run")
    print("=" * 60)

    df = load_data(
        DEFAULT_TAXI_PATH,
        sample_fraction=1.0,
        active_users_limit=_ACTIVE_USERS_LIMIT,
        min_vehicle_points=_MIN_VEHICLE_POINTS,
        processed_csv=DEFAULT_PROCESSED_TAXI_PATH,
    )
    servers_df = pd.read_csv(DEFAULT_SERVER_PATH)
    print(f"  Edge servers loaded: {len(servers_df)}")

    print(f"\n{'=' * 60}")
    print("  Running SA Microservice Migration ...")
    print(f"{'=' * 60}")
    t_start = time.time()
    results = run_sa_microservice_fair(df, servers_df)
    elapsed = time.time() - t_start

    M = results['total_migrations']
    V = results['total_violations']
    score = compute_score(M, V)

    print(f"\n{'=' * 60}")
    print(f"  RESULTS — SA Microservice DAG Migration")
    print(f"{'=' * 60}")
    print(f"  Migrations (M)         : {M}")
    print(f"  Violations (V)         : {V}")
    print(f"  Score (M + 0.5*V)      : {score:.1f}")
    print(f"  Total Reward           : {results['total_reward']:.2f}")
    print(f"  SA Decisions           : {results['decision_count']}")
    if results['reward_history']:
        print(f"  Avg Reward (last 100)  : {np.mean(results['reward_history'][-100:]):.4f}")
    print(f"  Elapsed Time           : {elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
