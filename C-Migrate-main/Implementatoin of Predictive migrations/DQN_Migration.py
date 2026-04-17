    # Method 6: DQN
    def run_dqn_fair():
        print("\n6. Running Deep Q-Network (Fair Comparison)...")
        total_migrations = 0
        total_violations = 0
        taxi_assignments = {}
        
        # Simple DQN
        class SimpleDQN(nn.Module):
            def __init__(self, input_size=10, hidden_size=64, action_size=4):
                super(SimpleDQN, self).__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, action_size)
                )
            
            def forward(self, x):
                return self.network(x)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        q_network = SimpleDQN().to(device)
        target_network = SimpleDQN().to(device)
        target_network.load_state_dict(q_network.state_dict())
        optimizer = optim.Adam(q_network.parameters(), lr=0.001)
        
        memory = deque(maxlen=2000)
        epsilon = 0.3
        epsilon_decay = 0.995
        epsilon_min = 0.01
        
        pbar = tqdm(total=len(sample_timestamps), desc="Deep Q-Network")
        step_count = 0
        
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
                    
                    # State
                    candidates = find_k_nearest_servers(current_lat, current_lon, servers_df, k=3)
                    
                    state = np.array([
                        current_lat / 90.0,
                        current_lon / 180.0,
                        min(current_dist / 50.0, 1.0),
                        timestamp.hour / 24.0,
                        timestamp.weekday() / 7.0,
                        len(candidates) / 10.0,
                        0.0, 0.0, 0.0,  # Placeholder features
                        1.0 if current_dist > 15.0 else 0.0
                    ], dtype=np.float32)
                    
                    # DQN decision
                    if random.random() < epsilon:
                        action = random.randint(0, 3)
                    else:
                        with torch.no_grad():
                            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                            q_values = q_network(state_tensor)
                            action = q_values.argmax().item()
                    
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
                    
                    # Reward
                    new_server_info = servers_df[servers_df['edge_server_id'] == chosen_server_id].iloc[0]
                    new_dist = haversine_distance(current_lat, current_lon, 
                                                new_server_info['latitude'], new_server_info['longitude'])
                    
                    reward = 0
                    if new_dist <= 15.0:
                        reward += 8
                    else:
                        reward -= 12
                    
                    if chosen_server_id != current_server_id:
                        reward -= 2
                    
                    # Store experience
                    memory.append((state, action, reward, state, False))  # Simplified
                    
                    # Training
                    if len(memory) >= 32:
                        batch = random.sample(memory, 32)
                        states = torch.FloatTensor([e[0] for e in batch]).to(device)
                        actions = torch.LongTensor([e[1] for e in batch]).to(device)
                        rewards = torch.FloatTensor([e[2] for e in batch]).to(device)
                        next_states = torch.FloatTensor([e[3] for e in batch]).to(device)
                        dones = torch.BoolTensor([e[4] for e in batch]).to(device)
                        
                        current_q = q_network(states).gather(1, actions.unsqueeze(1))
                        with torch.no_grad():
                            next_q = target_network(next_states).max(1)[0].unsqueeze(1)
                        target_q = rewards.unsqueeze(1) + (0.95 * next_q * ~dones.unsqueeze(1))
                        
                        loss = F.mse_loss(current_q, target_q)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        if epsilon > epsilon_min:
                            epsilon *= epsilon_decay
                    
                    step_count += 1
                    if step_count % 50 == 0:
                        target_network.load_state_dict(q_network.state_dict())
            pbar.update(1)
        pbar.close()
        return total_migrations, total_violations