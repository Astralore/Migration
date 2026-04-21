   # Method 3: Simulated Annealing
    def run_sa_fair():
        print("\n3. Running Simulated Annealing (Fair Comparison)...")
        total_migrations = 0
        total_violations = 0
        taxi_assignments = {}
        
        pbar = tqdm(total=len(sample_timestamps), desc="Simulated Annealing")
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
                    
                    # SA implementation
                    temperature = 50.0
                    cooling_rate = 0.99
                    current_future_violations = sum(
                        1 for fut_lon, fut_lat in future_positions[:8]
                        if haversine_distance(fut_lat, fut_lon, server_lat, server_lon) > 15.0
                    )
                    
                    best_server_id = current_server_id
                    best_future_violations = current_future_violations
                    
                    for _ in range(15):  # You can change (current is: 15 iterations)
                        if candidates:
                            candidate_id, _, cand_lat, cand_lon = random.choice(candidates)
                            future_violations = sum(
                                1 for fut_lon, fut_lat in future_positions[:8]
                                if haversine_distance(fut_lat, fut_lon, cand_lat, cand_lon) > 15.0
                            )
                            
                            delta = future_violations - best_future_violations
                            if delta < 0 or random.random() < math.exp(-delta / temperature):
                                if future_violations < best_future_violations:
                                    best_future_violations = future_violations
                                    best_server_id = candidate_id
                        
                        temperature *= cooling_rate
                        if temperature < 1e-3:
                            break
                    
                    if best_server_id != current_server_id:
                        total_migrations += 1
                        taxi_assignments[taxi_id] = {
                            'server_id': best_server_id,
                            'timestamp': timestamp
                        }
            pbar.update(1)
        pbar.close()
        return total_migrations, total_violations