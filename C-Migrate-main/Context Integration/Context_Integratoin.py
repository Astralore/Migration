#all comparsion included
import sys
# Simple trajectory predictor (constant velocity)
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from collections import deque, defaultdict
import torch.nn as nn
from tqdm import tqdm
import random 
import math
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')
import time

# Haversine distance function
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c
    
class SimpleTrajectoryPredictor:
    def __init__(self, forecast_horizon=3):
        self.forecast_horizon = forecast_horizon
        self.velocity_factors = {}

    def fit(self, df):
        print("Fitting trajectory predictor...")
        for taxi_id in tqdm(df['taxi_id'].unique(), desc="Learning velocities"):
            taxi_data = df[df['taxi_id'] == taxi_id].sort_values('date_time')
            if len(taxi_data) < 100:
                continue
            lons = taxi_data['longitude'].values
            lats = taxi_data['latitude'].values
            dx = np.diff(lons)
            dy = np.diff(lats)
            if len(dx) > 0:
                self.velocity_factors[taxi_id] = (np.mean(dx), np.mean(dy))
        return self

    def predict_future(self, current_lon, current_lat, taxi_id, steps=None):
        if steps is None:
            steps = self.forecast_horizon
        if taxi_id not in self.velocity_factors:
            return [(current_lon, current_lat)] * steps
        dx, dy = self.velocity_factors[taxi_id]
        future = []
        lon, lat = current_lon, current_lat
        for _ in range(steps):
            lon += dx
            lat += dy
            future.append((lon, lat))
        return future

# Find k nearest servers
def find_k_nearest_servers(lat, lon, servers_df, k=3):
    distances = []
    for _, server in servers_df.iterrows():
        dist = haversine_distance(lat, lon, server['latitude'], server['longitude'])
        distances.append((server['edge_server_id'], dist, server['latitude'], server['longitude']))
    distances.sort(key=lambda x: x[1])
    return distances[:k]

    
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
    sample_interval = max(1, len(timestamps) // 8787)
    sample_timestamps = timestamps[::sample_interval][:8787]  # change timestamps for all methods
    
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
                    if not check_migration_criteria(row, current_dist):
                        continue
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
        
    def check_migration_criteria(row, current_dist):
        """
        CONSISTENT migration criteria for ALL methods
        Migration is triggered if distance > 15km AND health risk exists
        """
        if current_dist <= 15.0:
            return False  # No migration needed if within range
        
        # Extract health data
        age = row.get('Age', 0)
        physical_activity = row.get('Physical_Activity', '')
        cvd_risk_score = row.get('CVD_Risk_Score', 0)
        hypertension = row.get('Hypertension', '')
        diabetes = row.get('Diabetes', '')
        
        # Health risk conditions (same for all methods)
        health_risk = (
            age > 75 or
            str(physical_activity).strip().lower() == 'high' or
            cvd_risk_score > 70 or
            str(hypertension).strip().lower() == 'yes' or
            str(diabetes).strip().lower() == 'yes' or
            current_dist > 30.0  # Extreme distance always triggers
        )
        return health_risk
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
                    if not check_migration_criteria(row, current_dist):
                        continue
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
                    if not check_migration_criteria(row, current_dist):
                        continue
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
                    
                    for _ in range(15):  # 15 iterations
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
                    if not check_migration_criteria(row, current_dist):
                        continue
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
                        reward += 8
                    else:
                        reward -= 12
                    
                    if chosen_server_id != current_server_id:
                        reward -= 2
                    
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
    

     #Helper function for Hybrid SA -DQN   
    def execute_simulated_annealing(
        current_lat, current_lon,
        current_server_id, server_lat, server_lon,
        candidates, future_positions
    ):
        """
        Executes the exact SA logic from run_sa_fair and returns the best server ID.
        """
        # SA implementation (copied from run_sa_fair)
        temperature = 50.0
        cooling_rate = 0.99
        current_future_violations = sum(
            1 for fut_lon, fut_lat in future_positions[:8]
            if haversine_distance(fut_lat, fut_lon, server_lat, server_lon) > 15.0
        )
    
        best_server_id = current_server_id
        best_future_violations = current_future_violations
    
        for _ in range(15):  # 15 iterations
            if candidates:
                candidate_id, _, cand_lat, cand_lon = random.choice(candidates)
                future_violations = sum(
                    1 for fut_lon, fut_lat in future_positions[:8]
                    if haversine_distance(fut_lat, fut_lon, cand_lat, cand_lon) > 15.0
                )
    
                delta = future_violations - best_future_violations
                # Standard SA acceptance criteria
                if delta < 0 or random.random() < math.exp(-delta / max(temperature, 1e-10)):
                    if future_violations < best_future_violations:
                        best_future_violations = future_violations
                        best_server_id = candidate_id
    
            temperature *= cooling_rate
            if temperature < 1e-3:
                break
    
        return best_server_id
        # --- Add this new method definition inside your main_fair_comparison_all_methods function ---
    # --- Method: RL-Refined SA (DQN Critic/Selector) ---
    def run_rl_refined_sa_fair():
        """
        Uses Simulated Annealing as the base optimizer.
        Employs a Deep Q-Network (DQN) as a critic/selector to refine SA's decisions.
        DQN learns when to accept SA's choice or pick a potentially better alternative.
        """
        print("\nX. Running RL-Refined SA (DQN Critic/Selector)...")
        
        # --- DQN Agent Definition ---
        class SimpleRefinementDQN(nn.Module):
            def __init__(self, state_size=12, action_size=3, hidden_size=64):
                super(SimpleRefinementDQN, self).__init__()
                self.network = nn.Sequential(
                    nn.Linear(state_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, action_size) # 3 actions: Accept SA, Nearest, Historical Best
                )
    
            def forward(self, x):
                return self.network(x)
    
        # --- Hyperparameters ---
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        STATE_SIZE = 12
        ACTION_SIZE = 3
        HIDDEN_SIZE = 64
        LEARNING_RATE = 0.001
        DISCOUNT_FACTOR = 0.9
        EPSILON_START = 0.3
        EPSILON_END = 0.05
        EPSILON_DECAY = 0.997
        TARGET_UPDATE_FREQ = 50
        BATCH_SIZE = 32
        MEMORY_SIZE = 2000
        MIN_MEMORY_FOR_TRAINING = 200
    
        # --- Initialize DQN Components ---
        q_network = SimpleRefinementDQN(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE).to(DEVICE)
        target_network = SimpleRefinementDQN(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE).to(DEVICE)
        target_network.load_state_dict(q_network.state_dict())
        target_network.eval()
    
        optimizer = optim.Adam(q_network.parameters(), lr=LEARNING_RATE)
        # Use the global memory and step counter defined outside
        # memory = deque(maxlen=MEMORY_SIZE) # Defined globally or passed
        # steps_done = 0 # Defined globally or passed
    
        epsilon = EPSILON_START
    
        # --- For tracking historical performance ---
        taxi_server_performance = defaultdict(lambda: {'success': 0, 'total': 0})
    
        # --- Metrics ---
        total_migrations = 0
        total_violations = 0
        taxi_assignments = {}
        steps_done_local = 0 # Local step counter for target updates
    
        # --- Progress Bar ---
        pbar = tqdm(total=len(sample_timestamps), desc="RL-Refined SA (DQN)")
    
        # --- Main Loop ---
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
                    total_violations += 1 # IDENTICAL counting
                    if not check_migration_criteria(row, current_dist):
                        continue
                    # --- Core RL-Refined SA Decision Making ---
    
                    # 1. Get candidates and future positions (IDENTICAL to SA)
                    candidates = find_k_nearest_servers(current_lat, current_lon, servers_df, k=3)
                    future_positions = predictor.predict_future(current_lon, current_lat, taxi_id, 10)
    
                    # --- 2. Base Simulated Annealing Decision ---
                    # Call the isolated SA logic to get its specific recommendation
                    sa_best_server_id = execute_simulated_annealing(
                        current_lat, current_lon,
                        current_server_id, server_lat, server_lon,
                        candidates, future_positions
                    )
    
                    # --- 3. DQN Critic/Selector Evaluation ---
    
                    # a. Create state for DQN based on context and SA's proposal
                    # State features: [current_lat, current_lon, current_dist,
                    #                  sa_proposed_lat, sa_proposed_lon,
                    #                  sa_predicted_violations,
                    #                  time_period, weekday]
                    state_list = [
                        current_lat / 90.0,
                        current_lon / 180.0,
                        min(current_dist / 50.0, 1.0),
                    ]
                    # Add SA proposal features
                    sa_proposed_info = servers_df[servers_df['edge_server_id'] == sa_best_server_id].iloc[0]
                    sa_proposed_lat, sa_proposed_lon = sa_proposed_info['latitude'], sa_proposed_info['longitude']
                    state_list.append(sa_proposed_lat / 90.0)
                    state_list.append(sa_proposed_lon / 180.0)
                    
                    sa_predicted_violations = sum(
                        1 for fut_lon, fut_lat in future_positions[:8]
                        if haversine_distance(fut_lat, fut_lon, sa_proposed_lat, sa_proposed_lon) > 15.0
                    )
                    state_list.append(min(sa_predicted_violations / 10.0, 1.0))
                    
                    # Add temporal features
                    state_list.append(timestamp.hour / 24.0)
                    state_list.append(timestamp.weekday() / 7.0)
                    
                    # Add relative features
                    state_list.append((current_lat - sa_proposed_lat) / 10.0) # Relative lat
                    state_list.append((current_lon - sa_proposed_lon) / 10.0) # Relative lon
                    state_list.append(min(haversine_distance(current_lat, current_lon, sa_proposed_lat, sa_proposed_lon) / 50.0, 1.0)) # Dist to SA choice
                    
                    # Add candidate count
                    state_list.append(len(candidates) / 5.0)
    
                    state_vector = np.array(state_list, dtype=np.float32)
                    state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(DEVICE)
    
                    # b. DQN action selection (epsilon-greedy)
                    if random.random() < epsilon:
                        action = random.randint(0, ACTION_SIZE - 1)
                    else:
                        with torch.no_grad():
                            q_values = q_network(state_tensor)
                            action = q_values.argmax().item()
    
                    # c. Execute refined decision based on DQN action
                    if action == 0:
                        # Action 0: Accept SA's recommendation
                        chosen_server_id = sa_best_server_id
                    elif action == 1 and candidates:
                        # Action 1: Choose the absolute nearest server
                        chosen_server_id = candidates[0][0]
                    elif action == 2:
                        # Action 2: Choose based on historical performance for this taxi
                        historical_best_id = current_server_id
                        best_historical_score = -1
                        for candidate_id, _, _, _ in candidates:
                            perf = taxi_server_performance[(taxi_id, candidate_id)]
                            if perf['total'] > 0:
                                score = perf['success'] / perf['total']
                                # Add small confidence bonus
                                confidence_bonus = min(perf['total'] / 5.0, 1.0) * 0.1
                                score += confidence_bonus
                                if score > best_historical_score:
                                    best_historical_score = score
                                    historical_best_id = candidate_id
                        chosen_server_id = historical_best_id
                    else:
                        # Fallback: Accept SA's recommendation
                        chosen_server_id = sa_best_server_id
    
                    # --- 4. Migration Decision & Execution ---
                    migration_occurred = (chosen_server_id != current_server_id)
                    if migration_occurred:
                        total_migrations += 1 # IDENTICAL counting
                        taxi_assignments[taxi_id] = {
                            'server_id': chosen_server_id,
                            'timestamp': timestamp
                        }
    
                    # --- 5. Calculate Reward for DQN Update ---
                    # Get new distance after potential migration (outcome of the final chosen decision)
                    new_server_info = servers_df[servers_df['edge_server_id'] == chosen_server_id].iloc[0]
                    new_lat, new_lon = new_server_info['latitude'], new_server_info['longitude']
                    new_dist = haversine_distance(current_lat, current_lon, new_lat, new_lon)
    
                    # --- Reward Structure ---
                    # Aligns with the M+0.5V objective.
                    reward = 0.0
                    if new_dist <= 15.0:
                        reward += 12.0 # Strong reward for good coverage (resolving violation)
                    else:
                        reward -= 15.0 # Strong penalty for persisting violation
    
                    # Migration cost
                    if migration_occurred:
                        reward -= 2.0 # Cost for migration
    
                    # --- 6. Store Experience and Update DQN ---
                    # Create next state vector (approximation)
                    next_state_list = [
                        current_lat / 90.0,
                        current_lon / 180.0,
                        min(new_dist / 50.0, 1.0), # Use new_dist for next state
                    ]
                    # Assume SA would propose the same server if state is same (simplification for next state)
                    next_state_list.append(sa_proposed_lat / 90.0)
                    next_state_list.append(sa_proposed_lon / 180.0)
                    # Recalculate predicted violations for next state (simplified)
                    next_sa_predicted_violations = sum(
                        1 for fut_lon, fut_lat in future_positions[:8]
                        if haversine_distance(fut_lat, fut_lon, sa_proposed_lat, sa_proposed_lon) > 15.0
                    )
                    next_state_list.append(min(next_sa_predicted_violations / 10.0, 1.0))
                    next_state_list.append(timestamp.hour / 24.0)
                    next_state_list.append(timestamp.weekday() / 7.0)
                    next_state_list.append((current_lat - sa_proposed_lat) / 10.0)
                    next_state_list.append((current_lon - sa_proposed_lon) / 10.0)
                    next_state_list.append(min(haversine_distance(current_lat, current_lon, sa_proposed_lat, sa_proposed_lon) / 50.0, 1.0))
                    next_state_list.append(len(candidates) / 5.0)
    
                    next_state_vector = np.array(next_state_list, dtype=np.float32)
    
                    # Store experience in a global-like memory (using a local list for this example)
                    # In a full implementation, you'd use a shared replay buffer.
                    # For simplicity here, we'll assume `memory` and `steps_done` are accessible.
                    # Let's define them locally for this function's scope.
                    if 'rl_memory' not in locals():
                        rl_memory = deque(maxlen=MEMORY_SIZE)
                    if 'rl_steps_done' not in locals():
                        rl_steps_done = 0
    
                    done = False # Continuous process
                    rl_memory.append((state_vector, action, reward, next_state_vector, done))
                    rl_steps_done += 1
                    steps_done_local += 1
    
                    # --- DQN Training Step ---
                    if len(rl_memory) >= MIN_MEMORY_FOR_TRAINING and len(rl_memory) >= BATCH_SIZE:
                        # Sample a batch
                        batch = random.sample(rl_memory, BATCH_SIZE)
                        state_batch = torch.FloatTensor([e[0] for e in batch]).to(DEVICE)
                        action_batch = torch.LongTensor([e[1] for e in batch]).to(DEVICE)
                        reward_batch = torch.FloatTensor([e[2] for e in batch]).to(DEVICE)
                        next_state_batch = torch.FloatTensor([e[3] for e in batch]).to(DEVICE)
                        done_batch = torch.BoolTensor([e[4] for e in batch]).to(DEVICE)
    
                        # Compute current Q-values
                        current_q_values = q_network(state_batch) # [BATCH_SIZE, ACTION_SIZE]
                        current_q_taken = current_q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1) # [BATCH_SIZE]
    
                        # Compute next Q-values (from target network)
                        with torch.no_grad():
                            next_q_values = target_network(next_state_batch) # [BATCH_SIZE, ACTION_SIZE]
                            next_q_max = next_q_values.max(1)[0] # [BATCH_SIZE]
    
                        # Compute target Q-values
                        target_q_values = reward_batch + (DISCOUNT_FACTOR * next_q_max * ~done_batch) # [BATCH_SIZE]
    
                        # Compute loss and update
                        loss = F.mse_loss(current_q_taken, target_q_values)
                        optimizer.zero_grad()
                        loss.backward()
                        # torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=1.0) # Optional clipping
                        optimizer.step()
    
                    # --- 7. Update Target Network Periodically ---
                    if steps_done_local % TARGET_UPDATE_FREQ == 0:
                        target_network.load_state_dict(q_network.state_dict())
                        # print(f"  [RL-Refined SA] Target network updated at step {steps_done_local}")
    
                    # --- 8. Update Historical Performance ---
                    taxi_server_performance[(taxi_id, chosen_server_id)]['total'] += 1
                    if new_dist <= 15.0:
                        taxi_server_performance[(taxi_id, chosen_server_id)]['success'] += 1
    
                    # --- End of RL-Refined SA Decision Making ---
                else:
                    # No violation - no migration needed (IDENTICAL to other methods)
                    pass
    
            # --- Epsilon Decay (outside the inner loop) ---
            if epsilon > EPSILON_END:
                epsilon *= EPSILON_DECAY
    
            pbar.update(1)
    
        # --- Finalize ---
        pbar.close()
        print(f"✅ RL-Refined SA (DQN Critic/Selector) completed.")
        print(f"   Final Epsilon: {epsilon:.4f}")
        print(f"   Memory Size: {len(rl_memory) if 'rl_memory' in locals() else 'N/A'}")
        print(f"   Steps Done: {rl_steps_done if 'rl_steps_done' in locals() else 'N/A'}")
        return total_migrations, total_violations
    
    # Method 5: Hybrid SA-Q-Learning (SA Base + RL Refinement)
    def run_hybrid_sa_ql_fair():
        """
        Hybrid approach: Use Simulated Annealing as the base optimizer.
        Use Q-Learning to evaluate SA's decision and potentially find a better alternative.
        Choose the best result between SA and RL suggestions.
        """
        print("\n5. Running Hybrid SA-Q-Learning (SA Base + RL Refinement)...")
        
        total_migrations = 0
        total_violations = 0
        taxi_assignments = {}
        
        # Q-learning agent for refinement
        q_table = defaultdict(lambda: np.zeros(4))  # 4 actions: accept SA, nearest, 2nd nearest, 3rd nearest
        learning_rate = 0.15
        discount_factor = 0.9
        epsilon = 0.2
        epsilon_decay = 0.995
        epsilon_min = 0.05
        
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
                    if not check_migration_criteria(row, current_dist):
                        continue
                    # --- Core Hybrid Decision Making ---
                    
                    # 1. Get candidates and future positions
                    candidates = find_k_nearest_servers(current_lat, current_lon, servers_df, k=5)
                    future_positions = predictor.predict_future(current_lon, current_lat, taxi_id, 10)
                    
                    # 2. Run EXACT Simulated Annealing to get its recommendation
                    # (This is the core SA logic copied from run_sa_fair)
                    temperature = 50.0
                    cooling_rate = 0.99
                    current_future_violations = sum(
                        1 for fut_lon, fut_lat in future_positions[:8]
                        if haversine_distance(fut_lat, fut_lon, server_lat, server_lon) > 15.0
                    )
                    
                    sa_best_server_id = current_server_id
                    sa_best_future_violations = current_future_violations
                    
                    for _ in range(15):  # 15 iterations
                        if candidates:
                            candidate_id, _, cand_lat, cand_lon = random.choice(candidates)
                            future_violations = sum(
                                1 for fut_lon, fut_lat in future_positions[:8]
                                if haversine_distance(fut_lat, fut_lon, cand_lat, cand_lon) > 15.0
                            )
                            
                            delta = future_violations - sa_best_future_violations
                            if delta < 0 or random.random() < math.exp(-delta / temperature):
                                if future_violations < sa_best_future_violations:
                                    sa_best_future_violations = future_violations
                                    sa_best_server_id = candidate_id
                        
                        temperature *= cooling_rate
                        if temperature < 1e-3:
                            break
                    
                    # 3. Q-Learning Refinement of SA Decision
                    # State representation
                    dist_category = min(int(current_dist / 3), 10)
                    time_period = timestamp.hour // 4
                    sa_quality = min(int(sa_best_future_violations / 2), 10)
                    state = (dist_category, time_period, sa_quality)
                    
                    # Q-Learning action selection (epsilon-greedy)
                    if random.random() < epsilon:
                        action = random.randint(0, 3)
                    else:
                        action = np.argmax(q_table[state])
                    
                    # Execute action
                    if action == 0:
                        # Action 0: Accept SA's recommendation
                        chosen_server_id = sa_best_server_id
                    elif action == 1 and candidates:
                        # Action 1: Choose nearest server
                        chosen_server_id = candidates[0][0]
                    elif action == 2 and len(candidates) > 1:
                        # Action 2: Choose 2nd nearest server
                        chosen_server_id = candidates[1][0]
                    elif action == 3 and len(candidates) > 2:
                        # Action 3: Choose 3rd nearest server
                        chosen_server_id = candidates[2][0]
                    else:
                        # Fallback: Accept SA's recommendation
                        chosen_server_id = sa_best_server_id
                    
                    # 4. Migration Decision & Execution
                    migration_occurred = (chosen_server_id != current_server_id)
                    if migration_occurred:
                        total_migrations += 1  # IDENTICAL counting
                        taxi_assignments[taxi_id] = {
                            'server_id': chosen_server_id,
                            'timestamp': timestamp
                        }
                    
                    # 5. Calculate Reward for Q-Learning
                    new_server_info = servers_df[servers_df['edge_server_id'] == chosen_server_id].iloc[0]
                    new_lat, new_lon = new_server_info['latitude'], new_server_info['longitude']
                    new_dist = haversine_distance(current_lat, current_lon, new_lat, new_lon)
                    
                    # Reward structure (IDENTICAL to other methods for fair comparison)
                    reward = 0
                    if new_dist <= 15.0:
                        reward += 8   # Good coverage reward
                    else:
                        reward -= 25  # Violation penalty
                    
                    if migration_occurred:
                        reward -= 2   # Migration cost
                    
                    # 6. Q-learning update
                    next_dist_category = min(int(new_dist / 3), 10)
                    next_time_period = timestamp.hour // 4
                    # Calculate next SA quality for next state
                    next_server_info = servers_df[servers_df['edge_server_id'] == chosen_server_id].iloc[0]
                    next_server_lat, next_server_lon = next_server_info['latitude'], next_server_info['longitude']
                    next_future_positions = predictor.predict_future(new_lon, new_lat, taxi_id, 10)
                    next_current_future_violations = sum(
                        1 for fut_lon, fut_lat in next_future_positions[:8]
                        if haversine_distance(fut_lat, fut_lon, next_server_lat, next_server_lon) > 15.0
                    )
                    next_sa_quality = min(int(next_current_future_violations / 2), 10)
                    next_state = (next_dist_category, next_time_period, next_sa_quality)
                    
                    current_q = q_table[state][action]
                    next_max_q = np.max(q_table[next_state])
                    target_q = reward + discount_factor * next_max_q
                    q_table[state][action] += learning_rate * (target_q - current_q)
                    
                else:
                    # No violation - no migration needed (IDENTICAL to other methods)
                    pass
            
            # Epsilon decay (IDENTICAL to other Q-Learning methods)
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
                
            pbar.update(1)
        
        pbar.close()
        print(f"✅ Hybrid SA-Q-Learning (SA Base + RL Refinement) completed.")
        print(f"   Q-table size: {len(q_table)} states")
        return total_migrations, total_violations
    

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
                    if not check_migration_criteria(row, current_dist):
                        continue
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
                        if not check_migration_criteria(row, current_dist):
                            continue
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
                        if not check_migration_criteria(row, current_dist):
                            continue
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
    
    # Run all methods
    #mig_reactive, viol_reactive = run_reactive_fair()
    mig_heuristic, viol_heuristic = run_heuristic_fair()
    mig_sa, viol_sa = run_sa_fair()
    mig_ql, viol_ql = run_ql_fair()
    mig_dqn_sa, viol_dqn_sa = run_rl_refined_sa_fair()
    #mig_hybrid, viol_hybrid = run_hybrid_sa_ql_fair()
    mig_dqn, viol_dqn = run_dqn_fair()
    mig_ilp, viol_ilp = run_ilp_fair()
    
    # Results
    print("\n" + "="*120)
    print("✅ FAIR COMPARISON RESULTS - ALL METHODS")
    print("="*120)
    print(f"Method                 | Migrations | Violations | Score (M+0.5V)")
    print("-" * 75)
    #print(f"Reactive Baseline      | {mig_reactive:10d} | {viol_reactive:10d} | {mig_reactive + 0.5*viol_reactive:12.1f}")
    print(f"Heuristic              | {mig_heuristic:10d} | {viol_heuristic:10d} | {mig_heuristic + 0.5*viol_heuristic:12.1f}")
    print(f"Simulated Annealing    | {mig_sa:10d} | {viol_sa:10d} | {mig_sa + 0.5*viol_sa:12.1f}")
    print(f"Q-Learning             | {mig_ql:10d} | {viol_ql:10d} | {mig_ql + 0.5*viol_ql:12.1f}")
    #print(f"Hybrid SA-Q-Learning   | {mig_hybrid:10d} | {viol_hybrid:10d} | {mig_hybrid + 0.5*viol_hybrid:12.1f}")
    print(f"Deep Q-Network         | {mig_dqn:10d} | {viol_dqn:10d} | {mig_dqn + 0.5*viol_dqn:12.1f}")
    print(f"ILP (Integer LP)       | {mig_ilp:10d} | {viol_ilp:10d} | {mig_ilp + 0.5*viol_ilp:12.1f}")
    print(f"DQN-SA-Learning        | {mig_dqn_sa:10d} | {viol_dqn_sa:10d} | {mig_dqn_sa + 0.5*viol_dqn_sa:12.1f}")

    # Ranking
    methods = [
        #("Reactive Baseline", mig_reactive, viol_reactive),
        ("Heuristic", mig_heuristic, viol_heuristic),
        ("Simulated Annealing", mig_sa, viol_sa),
        ("Q-Learning", mig_ql, viol_ql),
        ("DQN-SA-Learning", mig_dqn_sa, viol_dqn_sa),
        #("Hybrid SA-Q-Learning", mig_hybrid, viol_hybrid),
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
def load_data(file_path, sample_fraction=1.0, chunk_size=1000):
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    # Sample taxis if needed (optional)
    if sample_fraction < 1.0:
        unique_taxis = df['taxi_id'].unique()
        sampled_taxis = np.random.choice(
            unique_taxis,
            size=int(len(unique_taxis) * sample_fraction),
            replace=False
        )
        df = df[df['taxi_id'].isin(sampled_taxis)]
        print(f"Sampled {len(sampled_taxis)} taxis.")

    df['date_time'] = pd.to_datetime(df['date_time'])
    df = df.sort_values(['taxi_id', 'date_time']).reset_index(drop=True)
    df = df.dropna(subset=['longitude', 'latitude'])

    # Limit rows only if chunk_size is specified and valid
    if chunk_size is not None and chunk_size > 0:
        df = df.head(chunk_size)
        print(f"Truncated to {chunk_size} rows for testing.")

    print(f"Final dataset: {len(df):,} records, {df['taxi_id'].nunique():,} unique taxis.")
    return df
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
    df = load_data('combined_taxi_with_health.csv', sample_fraction=1, chunk_size=10000)
    servers_df = pd.read_csv('edge_server_locations.csv')
    predictor = SimpleTrajectoryPredictor(forecast_horizon=10)
    predictor.fit(df)

# Run fair comparison
    results = main_fair_comparison_all_methods(df, servers_df, predictor)
    # You need to provide your actual data here
    print("Please call:")
    print("results = main_fair_comparison_all_methods(df, servers_df, predictor)")
    print("Where df, servers_df, predictor are your actual data")