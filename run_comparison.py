#!/usr/bin/env python
"""Entry point: run all three microservice migration algorithms and compare."""

import time
import pandas as pd

from core.data_loader import load_data, DEFAULT_TAXI_PATH, DEFAULT_SERVER_PATH
from prediction.simple_predictor import SimpleTrajectoryPredictor
from algorithms.dqn import run_dqn_microservice_fair
from algorithms.sa import run_sa_microservice_fair
from algorithms.hybrid_sa_dqn import run_hybrid_microservice_fair
from evaluation.metrics import print_ranking
from evaluation.plot import plot_training_curves

CHUNK_SIZE = 10000
PROACTIVE = True
FORECAST_HORIZON = 5


def main():
    print("=" * 60)
    print("  Microservice Migration — Full Comparison")
    print(f"  Proactive mode: {PROACTIVE}")
    print("=" * 60)

    df = load_data(DEFAULT_TAXI_PATH, chunk_size=CHUNK_SIZE)
    servers_df = pd.read_csv(DEFAULT_SERVER_PATH)
    print(f"  Edge servers loaded: {len(servers_df)}")

    predictor = None
    if PROACTIVE:
        predictor = SimpleTrajectoryPredictor(forecast_horizon=FORECAST_HORIZON)
        predictor.fit(df)
        print(f"  Predictor fitted: {len(predictor.velocity_factors)} taxis with velocity data")

    all_results = {}

    # SA
    print(f"\n{'=' * 60}")
    print("  [1/3] Running SA ...")
    print(f"{'=' * 60}")
    t0 = time.time()
    all_results["SA"] = run_sa_microservice_fair(
        df, servers_df, predictor=predictor, proactive=PROACTIVE,
    )
    print(f"  SA done in {time.time() - t0:.1f}s")

    # DQN
    print(f"\n{'=' * 60}")
    print("  [2/3] Running DQN ...")
    print(f"{'=' * 60}")
    t0 = time.time()
    all_results["DQN"] = run_dqn_microservice_fair(
        df, servers_df, predictor=predictor, proactive=PROACTIVE,
    )
    print(f"  DQN done in {time.time() - t0:.1f}s")

    # Hybrid
    print(f"\n{'=' * 60}")
    print("  [3/3] Running Hybrid SA-DQN ...")
    print(f"{'=' * 60}")
    t0 = time.time()
    all_results["Hybrid SA-DQN"] = run_hybrid_microservice_fair(
        df, servers_df, predictor=predictor, proactive=PROACTIVE,
    )
    print(f"  Hybrid done in {time.time() - t0:.1f}s")

    # Ranking
    print_ranking(all_results)

    # Training curves for DQN-based methods
    for name, key in [("DQN", "DQN"), ("Hybrid", "Hybrid SA-DQN")]:
        res = all_results[key]
        if res.get('loss_history'):
            plot_training_curves(
                res,
                save_path=f"outputs/{name.lower()}_microservice_training.png",
                title_prefix=name,
            )


if __name__ == "__main__":
    main()
