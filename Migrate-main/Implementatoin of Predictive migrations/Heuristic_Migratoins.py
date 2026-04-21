 # Method 2: Heuristic
    def run_heuristic_fair():
        print("\n2. Running Heuristic (Fair Comparison)...")
        total_migrations = 0
        total_violations = 0
        taxi_assignments = {}
        
        pbar = tqdm(total=len(sample_timestamps), desc="Heuristic")
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
                    
                    candidates = find_k_nearest_servers(current_lat, current_lon, servers_df, k=3)
                    future_positions = predictor.predict_future(current_lon, current_lat, taxi_id, 10)
                    
                    best_server_id = current_server_id
                    best_future_violations = float('inf')
                    
                    for candidate_id, _, cand_lat, cand_lon in candidates:
                        future_violations = 0
                        for fut_lon, fut_lat in future_positions[:8]:  # Limit for performance
                            dist = haversine_distance(fut_lat, fut_lon, cand_lat, cand_lon)
                            if dist > 15.0:
                                future_violations += 1
                        
                        if future_violations < best_future_violations:
                            best_future_violations = future_violations
                            best_server_id = candidate_id
                    
                    if best_server_id != current_server_id:
                        total_migrations += 1
                        taxi_assignments[taxi_id] = {
                            'server_id': best_server_id,
                            'timestamp': timestamp
                        }
            pbar.update(1)
        pbar.close()
        return total_migrations, total_violations
    