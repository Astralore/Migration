    # Method 7: ILP (Integer Linear Programming)
    def run_ilp_fair():
        print("\n7. Running ILP (Fair Comparison)...")
        total_migrations = 0
        total_violations = 0
        taxi_assignments = {}
        
        try:
            # Check if PuLP is available
            import pulp
            
            pbar = tqdm(total=len(sample_timestamps), desc="ILP")
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
                        
                        # Simple ILP-like optimization using candidates
                        candidates = find_k_nearest_servers(current_lat, current_lon, servers_df, k=3)
                        future_positions = predictor.predict_future(current_lon, current_lat, taxi_id, 10)
                        
                        # Create small ILP problem for this decision
                        try:
                            # Decision variables: which server to choose
                            prob = pulp.LpProblem(f"Taxi_{taxi_id}_Decision", pulp.LpMinimize)
                            
                            # Binary variables for server selection
                            server_vars = {}
                            for j, (candidate_id, _, _, _) in enumerate(candidates):
                                server_vars[candidate_id] = pulp.LpVariable(f"server_{candidate_id}", cat='Binary')
                            
                            # Constraint: exactly one server must be selected
                            prob += pulp.lpSum([server_vars[cid] for cid, _, _, _ in candidates]) == 1
                            
                            # Objective: minimize weighted sum of violations and migration cost
                            violation_weight = 10.0
                            migration_weight = 1.0
                            
                            objective_terms = []
                            for candidate_id, _, cand_lat, cand_lon in candidates:
                                # Calculate future violations for this candidate
                                future_violations = 0
                                for fut_lon, fut_lat in future_positions[:5]:  # Limit for performance
                                    dist = haversine_distance(fut_lat, fut_lon, cand_lat, cand_lon)
                                    if dist > 15.0:
                                        future_violations += 1
                                
                                # Migration cost (0 if staying, 1 if migrating)
                                migration_cost = 0 if candidate_id == current_server_id else 1
                                
                                # Add to objective
                                objective_terms.append(violation_weight * future_violations * server_vars[candidate_id])
                                objective_terms.append(migration_weight * migration_cost * server_vars[candidate_id])
                            
                            prob += pulp.lpSum(objective_terms)
                            
                            # Solve (with time limit for performance)
                            prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=1))
                            
                            # Extract solution
                            chosen_server_id = current_server_id  # Default
                            if pulp.LpStatus[prob.status] == 'Optimal':
                                for candidate_id, _, _, _ in candidates:
                                    if server_vars[candidate_id].varValue and server_vars[candidate_id].varValue > 0.5:
                                        chosen_server_id = candidate_id
                                        break
                            else:
                                # Fallback to heuristic if ILP fails
                                best_violations = float('inf')
                                for candidate_id, _, cand_lat, cand_lon in candidates:
                                    violations = sum(1 for fut_lon, fut_lat in future_positions[:5]
                                                   if haversine_distance(fut_lat, fut_lon, cand_lat, cand_lon) > 15.0)
                                    if violations < best_violations:
                                        best_violations = violations
                                        chosen_server_id = candidate_id
                        except:
                            # Simple fallback if ILP fails
                            chosen_server_id = candidates[0][0] if candidates else current_server_id
                        
                        # Migration
                        if chosen_server_id != current_server_id:
                            total_migrations += 1
                            taxi_assignments[taxi_id] = {
                                'server_id': chosen_server_id,
                                'timestamp': timestamp
                            }
                pbar.update(1)
            pbar.close()
        except ImportError:
            print("PuLP not available - using simple heuristic instead")
            # Fallback to simple heuristic if ILP not available
            pbar = tqdm(total=len(sample_timestamps), desc="ILP (Fallback)")
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
                        
                        # Simple heuristic fallback
                        candidates = find_k_nearest_servers(current_lat, current_lon, servers_df, k=3)
                        future_positions = predictor.predict_future(current_lon, current_lat, taxi_id, 10)
                        
                        best_server_id = current_server_id
                        best_violations = float('inf')
                        
                        for candidate_id, _, cand_lat, cand_lon in candidates:
                            violations = sum(1 for fut_lon, fut_lat in future_positions[:5]
                                           if haversine_distance(fut_lat, fut_lon, cand_lat, cand_lon) > 15.0)
                            if violations < best_violations:
                                best_violations = violations
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
    