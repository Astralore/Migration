    # Method 4: Q-Learning
    def run_ql_fair():
        print("\n4. Running Q-Learning (Fair Comparison)...")
        total_migrations = 0
        total_violations = 0
        taxi_assignments = {}
        
        # Simple Q-table
        q_table = defaultdict(lambda: np.zeros(4))
        learning_rate = 0.1
        discount_factor = 0.9
        epsilon = 0.2
        
        pbar = tqdm(total=len(sample_timestamps), desc="Q-Learning")
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
                    
                    # State representation
                    candidates = find_k_nearest_servers(current_lat, current_lon, servers_df, k=3)
                    
                    state = (
                        round(current_lat / 90.0 * 10),
                        round(current_lon / 180.0 * 10),
                        min(int(current_dist / 5), 5)
                    )
                    
                    # Q-Learning decision
                    if random.random() < epsilon:
                        action = random.randint(0, 3)
                    else:
                        action = np.argmax(q_table[state])
                    
                    # Execute action
                    if action == 0:
                        chosen_server_id = current_server_id
                    elif action == 1 and candidates:
                        chosen_server_id = candidates[0][0]
                    elif action == 2 and len(candidates) > 1:
                        chosen_server_id = candidates[1][0]
                    elif action == 3 and len(candidates) > 2:
                        chosen_server_id = candidates[2][0]
                    else:
                        chosen_server_id = candidates[0][0] if candidates else current_server_id
                    
                    # Migration
                    if chosen_server_id != current_server_id:
                        total_migrations += 1
                        taxi_assignments[taxi_id] = {
                            'server_id': chosen_server_id,
                            'timestamp': timestamp
                        }
                    
                    # Reward and learning
                    new_server_info = servers_df[servers_df['edge_server_id'] == chosen_server_id].iloc[0]
                    new_dist = haversine_distance(current_lat, current_lon, 
                                                new_server_info['latitude'], new_server_info['longitude'])
                    
                    reward = 0
                    if new_dist <= 15.0:
                        reward += 6
                    else:
                        reward -= 8
                    
                    if chosen_server_id != current_server_id:
                        reward -= 1
                    
                    # Q-learning update
                    next_state = (
                        round(current_lat / 90.0 * 10),
                        round(current_lon / 180.0 * 10),
                        min(int(new_dist / 5), 5)
                    )
                    
                    q_table[state][action] += learning_rate * (
                        reward + discount_factor * np.max(q_table[next_state]) - q_table[state][action]
                    )
            pbar.update(1)
        pbar.close()
        return total_migrations, total_violations
    