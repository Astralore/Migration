    # Run all methods
    mig_reactive, viol_reactive = run_reactive_fair()
    mig_heuristic, viol_heuristic = run_heuristic_fair()
    mig_sa, viol_sa = run_sa_fair()
    mig_ql, viol_ql = run_ql_fair()
    mig_hybrid, viol_hybrid = run_hybrid_sa_ql_fair()
    mig_dqn, viol_dqn = run_dqn_fair()
    mig_ilp, viol_ilp = run_ilp_fair()
    
    # Results
    print("\n" + "="*120)
    print("✅ FAIR COMPARISON RESULTS - ALL METHODS")
    print("="*120)
    print(f"Method                 | Migrations | Violations | Score (M+0.5V)")
    print("-" * 75)
    print(f"Reactive Baseline      | {mig_reactive:10d} | {viol_reactive:10d} | {mig_reactive + 0.5*viol_reactive:12.1f}")
    print(f"Heuristic              | {mig_heuristic:10d} | {viol_heuristic:10d} | {mig_heuristic + 0.5*viol_heuristic:12.1f}")
    print(f"Simulated Annealing    | {mig_sa:10d} | {viol_sa:10d} | {mig_sa + 0.5*viol_sa:12.1f}")
    print(f"Q-Learning             | {mig_ql:10d} | {viol_ql:10d} | {mig_ql + 0.5*viol_ql:12.1f}")
    print(f"Hybrid SA-Q-Learning   | {mig_hybrid:10d} | {viol_hybrid:10d} | {mig_hybrid + 0.5*viol_hybrid:12.1f}")
    print(f"Deep Q-Network         | {mig_dqn:10d} | {viol_dqn:10d} | {mig_dqn + 0.5*viol_dqn:12.1f}")
    print(f"ILP (Integer LP)       | {mig_ilp:10d} | {viol_ilp:10d} | {mig_ilp + 0.5*viol_ilp:12.1f}")
    
    # Ranking
    methods = [
        ("Reactive Baseline", mig_reactive, viol_reactive),
        ("Heuristic", mig_heuristic, viol_heuristic),
        ("Simulated Annealing", mig_sa, viol_sa),
        ("Q-Learning", mig_ql, viol_ql),
        ("Hybrid SA-Q-Learning", mig_hybrid, viol_hybrid),
        ("Deep Q-Network", mig_dqn, viol_dqn),
        ("ILP (Integer LP)", mig_ilp, viol_ilp)
    ]
    
    rankings = []
    for name, mig, viol in methods:
        score = mig + 0.5 * viol
        rankings.append((name, score, mig, viol))
    
    rankings.sort(key=lambda x: x[1])
    
    print(f"\n🏆 RANKING (Lower Score = Better):")
    for i, (name, score, mig, viol) in enumerate(rankings, 1):
        print(f"   {i}. {name:22} - Score: {score:6.1f} (Mig: {mig:3d}, Viol: {viol:3d})")
    
    best_method = rankings[0]
    print(f"\n🥇 WINNER: {best_method[0]} (Score: {best_method[1]:.1f})")
    
    return rankings

# Example of how to call it:
"""
# Load your data
df = load_data('combined_taxi_with_health.csv', sample_fraction=0.2, chunk_size=10000)
servers_df = pd.read_csv('edge_server_locations.csv')
predictor = SimpleTrajectoryPredictor(forecast_horizon=10)
predictor.fit(df)

# Run fair comparison
results = main_fair_comparison_all_methods(df, servers_df, predictor)
"""

# If you want to run it directly:
if __name__ == "__main__":
    df = load_data('combined_taxi_with_health.csv', sample_fraction=0.9, chunk_size=1000000) #You can change the sample fraction, Forecast_horizon, chunk_size ...etc
    servers_df = pd.read_csv('edge_server_locations.csv')
    predictor = SimpleTrajectoryPredictor(forecast_horizon=10)
    predictor.fit(df)

# Run fair comparison
    results = main_fair_comparison_all_methods(df, servers_df, predictor)
    # You need to provide your actual data here
    print("Please call:")
    print("results = main_fair_comparison_all_methods(df, servers_df, predictor)")
    print("Where df, servers_df, predictor are your actual data")