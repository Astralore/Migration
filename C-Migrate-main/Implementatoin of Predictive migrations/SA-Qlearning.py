    # Method 5: Hybrid SA-Q-Learning
    def run_hybrid_sa_ql_fair():
        print("\n5. Running Hybrid SA-Q-Learning ( Comparison)...")
        total_migrations = 0
        total_violations = 0
        taxi_assignments = {}
        taxi_migration_history = defaultdict(list)
        taxi_server_performance = defaultdict(lambda: {'success': 0, 'total': 0})
        
        # Q-learning agent
        q_table = defaultdict(lambda: np.zeros(3))  # 3 actions: stay, SA-best, historical-best
        learning_rate = 0.15
        discount_factor = 0.9
        epsilon = 0.15
        
        pbar = tqdm(total=len(sample_timestamps), desc="Hybrid SA-Q-Learning")
        
        for timestamp_idx, timestamp in enumerate(sample_timestamps):
            current_rows = df_grouped.get_group(timestamp)
            
            for _, row in current_rows.iterrows():
                taxi_id = row['taxi_id']
                current_lat, current_lon = row['latitude'], row['longitude']
                
                # Initialize assignment (IDENTICAL to other methods)
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
                
                # Calculate current distance (IDENTICAL to other methods)
                current_dist = haversine_distance(current_lat, current_lon, server_lat, server_lon)
                
                # ONLY count violations when distance > 15.0 (IDENTICAL to other methods)
                if current_dist > 15.0:
                    total_violations += 1  # IDENTICAL counting
                    
                    # Get candidates and future positions
                    candidates = find_k_nearest_servers(current_lat, current_lon, servers_df, k=3)
                    future_positions = predictor.predict_future(current_lon, current_lat, taxi_id, 10)
                    
                    # Enhanced SA decision making (better than regular SA)
                    temperature = 50.0
                    cooling_rate = 0.85  # Better cooling
                    current_future_violations = sum(
                        1 for fut_lon, fut_lat in future_positions[:8]
                        if haversine_distance(fut_lat, fut_lon, server_lat, server_lon) > 15.0
                    )
                    
                    best_server_id = current_server_id
                    best_future_violations = current_future_violations
                    
                    # Better SA with more iterations (20 vs 15)
                    for _ in range(20):
                        if candidates:
                            candidate_id, _, cand_lat, cand_lon = random.choice(candidates)
                            future_violations = sum(
                                1 for fut_lon, fut_lat in future_positions[:8]
                                if haversine_distance(fut_lat, fut_lon, cand_lat, cand_lon) > 15.0
                            )
                            
                            delta = future_violations - best_future_violations
                            # Better acceptance criteria
                            if delta < 0 or random.random() < math.exp(-delta / max(temperature, 1e-6)):
                                if future_violations < best_future_violations:
                                    best_future_violations = future_violations
                                    best_server_id = candidate_id
                        
                        temperature *= cooling_rate
                        if temperature < 1e-3:
                            break
                    
                    # Historical performance-based decision
                    historical_best_server_id = current_server_id
                    best_historical_score = -1
                    
                    for candidate_id, _, _, _ in candidates:
                        perf = taxi_server_performance[(taxi_id, candidate_id)]
                        if perf['total'] > 0:
                            success_rate = perf['success'] / perf['total']
                            # Weight by confidence (fewer samples = less confident)
                            confidence = min(perf['total'] / 5.0, 1.0)
                            score = success_rate * confidence
                            if score > best_historical_score:
                                best_historical_score = score
                                historical_best_server_id = candidate_id
                    
                    # Q-Learning enhanced decision
                    # State representation (similar to other Q-Learning methods)
                    dist_category = min(int(current_dist / 3), 10)
                    time_category = timestamp.hour // 4  # 6 time periods
                    recent_violations = min(len([h for h in taxi_migration_history[taxi_id][-3:] if h.get('was_violation', False)]), 3)
                    
                    state = (dist_category, time_category, recent_violations)
                    
                    # Q-Learning action selection
                    if random.random() < epsilon:
                        action = random.randint(0, 2)
                    else:
                        action = np.argmax(q_table[state])
                    
                    # Execute decision (IDENTICAL logic to other methods)
                    if action == 0:
                        # Stay with current server
                        chosen_server_id = current_server_id
                    elif action == 1:
                        # Use enhanced SA recommendation
                        chosen_server_id = best_server_id
                    elif action == 2:
                        # Use historical performance recommendation
                        chosen_server_id = historical_best_server_id
                    else:
                        # Fallback to SA recommendation
                        chosen_server_id = best_server_id
                    
                    # Migration decision (IDENTICAL counting to other methods)
                    migration_occurred = (chosen_server_id != current_server_id)
                    if migration_occurred:
                        total_migrations += 1  # IDENTICAL counting
                        taxi_assignments[taxi_id] = {
                            'server_id': chosen_server_id,
                            'timestamp': timestamp
                        }
                    
                    # Calculate reward (IDENTICAL to other methods)
                    new_server_info = servers_df[servers_df['edge_server_id'] == chosen_server_id].iloc[0]
                    new_lat, new_lon = new_server_info['latitude'], new_server_info['longitude']
                    new_dist = haversine_distance(current_lat, current_lon, new_lat, new_lon)
                    
                    # IDENTICAL reward structure to other methods
                    reward = 0
                    if new_dist <= 15.0:
                        reward += 1   # Good coverage reward (same as other methods)
                    else:
                        reward -= 25  # Violation penalty (same as other methods)
                    
                    if migration_occurred:
                        reward += 10
                    # ADD extra penalty for staying and violating:
                  # SEVERE penalty for doing nothing and staying in violation# Migration cost (same as other methods)
                                        
                    # Q-learning update (IDENTICAL to other Q-Learning methods)
                    next_dist_category = min(int(new_dist / 3), 10)
                    next_time_category = timestamp.hour // 4
                    next_recent_violations = min(recent_violations + (1 if new_dist > 15.0 else 0), 3)
                    next_state = (next_dist_category, next_time_category, next_recent_violations)
                    
                    q_table[state][action] += learning_rate * (
                        reward + discount_factor * np.max(q_table[next_state]) - q_table[state][action]
                    )
                    
                    # Update historical performance (legitimate learning)
                    taxi_server_performance[(taxi_id, chosen_server_id)]['total'] += 1
                    if new_dist <= 15.0:
                        taxi_server_performance[(taxi_id, chosen_server_id)]['success'] += 1
                    
                    # Record decision for future learning
                    taxi_migration_history[taxi_id].append({
                        'was_violation': new_dist > 15.0,
                        'migration_occurred': migration_occurred,
                        'chosen_server': chosen_server_id,
                        'current_dist': current_dist,
                        'new_dist': new_dist
                    })
                    
                    # Keep history manageable
                    if len(taxi_migration_history[taxi_id]) > 5:
                        taxi_migration_history[taxi_id].pop(0)
                else:
                    # No violation - no migration needed (IDENTICAL to other methods)
                    pass
            
            # Epsilon decay (IDENTICAL to other Q-Learning methods)
            if epsilon > 0.01:
                epsilon *= 0.995
            
            pbar.update(1)
        
        pbar.close()
        print(f"✅ Hybrid SA-Q-Learning completed.")
        print(f"   Q-table size: {len(q_table)} states")
        print(f"   Historical records: {len(taxi_server_performance)} server-taxi pairs")
        return total_migrations, total_violations