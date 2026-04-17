#all comparsion included
def main_fair_comparison_all_methods(df, servers_df, predictor):
    """
    Run fair comparison of all methods with your data
    """
    print("Running Fair Comparison of All Methods...")
    print("="*60)
    
    # IDENTICAL data processing for all methods
    timestamps = sorted(df['date_time'].unique())
    df_grouped = df.groupby('date_time')
    
    # Use same subset for fair comparison
    sample_interval = max(1, len(timestamps) // 2000) #You can change timesteps 
    sample_timestamps = timestamps[::sample_interval][:2000]  # change timestamps for all methods
    
    print(f"Processing {len(sample_timestamps)} timestamps (same for all methods)")
    print(f"Processing taxi count: {df[df['date_time'].isin(sample_timestamps)]['taxi_id'].nunique()}")
    
    # Method 1: Reactive Baseline
    def run_reactive_fair():
        print("\n1. Running Reactive Baseline (Fair Comparison)...")
        total_migrations = 0
        total_violations = 0
        taxi_assignments = {}
        
        pbar = tqdm(total=len(sample_timestamps), desc="Reactive")
        for timestamp in sample_timestamps:
            current_rows = df_grouped.get_group(timestamp)
            for _, row in current_rows.iterrows():
                taxi_id = row['taxi_id']
                current_lat, current_lon = row['latitude'], row['longitude']
                
                if taxi_id not in taxi_assignments:
                    nearest_server = find_k_nearest_servers(current_lat, current_lon, servers_df, k=1)[0]
                    taxi_assignments[taxi_id] = {
                        'server_id': nearest_server[0],
                        'timestamp': timestamp
                    }
                    continue
                
                current_server_id = taxi_assignments[taxi_id]['server_id']
                server_info = servers_df[servers_df['edge_server_id'] == current_server_id].iloc[0]
                server_lat, server_lon = server_info['latitude'], server_info['longitude']
                
                current_dist = haversine_distance(current_lat, current_lon, server_lat, server_lon)
                
                if current_dist > 15.0:
                    total_violations += 1
                    nearest_server = find_k_nearest_servers(current_lat, current_lon, servers_df, k=1)[0]
                    new_server_id = nearest_server[0]
                    
                    if new_server_id != current_server_id:
                        total_migrations += 1
                        taxi_assignments[taxi_id] = {
                            'server_id': new_server_id,
                            'timestamp': timestamp
                        }
            pbar.update(1)
        pbar.close()
        return total_migrations, total_violations
    