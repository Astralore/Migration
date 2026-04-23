#!/usr/bin/env python
"""
Entry point: run all three microservice migration algorithms and compare.
Supports both Proactive and Reactive modes for full experimental analysis.
"""

import time
import pandas as pd

from core.data_loader import load_data, DEFAULT_TAXI_PATH, DEFAULT_SERVER_PATH
from prediction.simple_predictor import SimpleTrajectoryPredictor
from algorithms.dqn import run_dqn_microservice_fair
from algorithms.sa import run_sa_microservice_fair
from algorithms.hybrid_sa_dqn import run_hybrid_microservice_fair
from algorithms.hybrid_sac import run_hybrid_sac_microservice
from evaluation.metrics import print_ranking, print_proactive_analysis
from evaluation.plot import plot_training_curves, plot_cost_breakdown, plot_performance_metrics

CHUNK_SIZE = 10000
PROACTIVE = True
FORECAST_HORIZON = 15  # Extended horizon for better proactive detection


def run_all_algorithms(df, servers_df, predictor, proactive, label=""):
    """Run SA, DQN, Hybrid and return results dict."""
    results = {}
    mode_str = "Proactive" if proactive else "Reactive"

    # SA
    print(f"\n{'=' * 60}")
    print(f"  [{label}] Running SA ({mode_str}) ...")
    print(f"{'=' * 60}")
    t0 = time.time()
    results["SA"] = run_sa_microservice_fair(
        df, servers_df, predictor=predictor, proactive=proactive,
    )
    print(f"  SA done in {time.time() - t0:.1f}s")

    # DQN
    print(f"\n{'=' * 60}")
    print(f"  [{label}] Running DQN ({mode_str}) ...")
    print(f"{'=' * 60}")
    t0 = time.time()
    results["DQN"] = run_dqn_microservice_fair(
        df, servers_df, predictor=predictor, proactive=proactive,
    )
    print(f"  DQN done in {time.time() - t0:.1f}s")

    # Hybrid SAC (新的 Trigger-Conditioned GAT + Discrete SAC)
    print(f"\n{'=' * 60}")
    print(f"  [{label}] Running Hybrid SAC ({mode_str}) ...")
    print(f"{'=' * 60}")
    t0 = time.time()
    results["Hybrid SAC"] = run_hybrid_sac_microservice(
        df, servers_df, predictor=predictor, proactive=proactive,
    )
    print(f"  Hybrid SAC done in {time.time() - t0:.1f}s")

    return results


def main():
    print("=" * 80)
    print("  Microservice Migration — Full Comparison (Asymmetric Cost Model)")
    print(f"  Proactive mode: {PROACTIVE}")
    print("=" * 80)

    df = load_data(DEFAULT_TAXI_PATH, chunk_size=CHUNK_SIZE)
    servers_df = pd.read_csv(DEFAULT_SERVER_PATH)
    print(f"  Edge servers loaded: {len(servers_df)}")

    # Fit predictor (needed for both modes if we want proactive features)
    predictor = SimpleTrajectoryPredictor(forecast_horizon=FORECAST_HORIZON)
    predictor.fit(df)
    print(f"  Predictor fitted: {len(predictor.velocity_factors)} taxis with velocity data")

    # Run Proactive mode
    print("\n" + "#" * 80)
    print("  PHASE 1: Proactive Mode (Predictive Migration)")
    print("#" * 80)
    proactive_results = run_all_algorithms(df, servers_df, predictor, proactive=True, label="Proactive")

    print("\n" + "#" * 80)
    print("  Proactive Mode Results")
    print("#" * 80)
    print_ranking(proactive_results)

    # Training curves for DQN-based methods (Proactive)
    for name, key in [("DQN", "DQN"), ("Hybrid_SAC", "Hybrid SAC")]:
        res = proactive_results[key]
        if res.get('loss_history'):
            plot_training_curves(
                res,
                save_path=f"outputs/{name.lower()}_proactive_training.png",
                title_prefix=f"{name} (Proactive)",
            )

    # Optionally run Reactive mode for comparison
    reactive_results = None
    if PROACTIVE:
        print("\n" + "#" * 80)
        print("  PHASE 2: Reactive Mode (Baseline Comparison)")
        print("#" * 80)
        reactive_results = run_all_algorithms(df, servers_df, predictor, proactive=False, label="Reactive")

        print("\n" + "#" * 80)
        print("  Reactive Mode Results")
        print("#" * 80)
        print_ranking(reactive_results)

        # Training curves for DQN-based methods (Reactive)
        for name, key in [("DQN", "DQN"), ("Hybrid_SAC", "Hybrid SAC")]:
            res = reactive_results[key]
            if res.get('loss_history'):
                plot_training_curves(
                    res,
                    save_path=f"outputs/{name.lower()}_reactive_training.png",
                    title_prefix=f"{name} (Reactive)",
                )

    # Print comparative analysis
    print_proactive_analysis(proactive_results, reactive_results)

    # Final summary for paper
    print("\n" + "=" * 80)
    print("  PAPER SUMMARY")
    print("=" * 80)
    if reactive_results:
        for name in ["SA", "DQN", "Hybrid SAC"]:
            pro = proactive_results[name]
            rea = reactive_results[name]
            pro_v, rea_v = pro['total_violations'], rea['total_violations']
            pro_m, rea_m = pro['total_migrations'], rea['total_migrations']
            pro_d = pro.get('proactive_decisions', 0)

            if rea_v > 0:
                v_reduction = (rea_v - pro_v) / rea_v * 100
            else:
                v_reduction = 0

            print(f"\n  {name}:")
            print(f"    - Proactive decisions: {pro_d}")
            print(f"    - Real Violations: {rea_v} -> {pro_v} ({v_reduction:+.1f}%)")
            print(f"    - Migrations: {rea_m} -> {pro_m}")
    print("\n" + "=" * 80)

    # Generate visualization charts
    print("\n" + "#" * 80)
    print("  GENERATING VISUALIZATIONS")
    print("#" * 80)

    # Cost breakdown chart (Proactive mode)
    plot_cost_breakdown(proactive_results, save_path="outputs/cost_breakdown.png")

    # Violation comparison chart (Proactive vs Reactive)
    if reactive_results:
        plot_performance_metrics(proactive_results, reactive_results, save_path="outputs/violation_comparison.png")

    print("\n  All visualizations generated successfully!")


if __name__ == "__main__":
    main()
