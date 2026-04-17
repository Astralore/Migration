import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Haversine distance function
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

# Simple spatial transformer model for trajectory prediction
class SimpleSpatialTransformer:
    def __init__(self, lookback=5, forecast=1):
        self.lookback = lookback
        self.forecast = forecast
        self.velocity_factors = {}
        self.accuracy_metrics = {}
        
    def fit(self, X, y=None):
        # Calculate velocity factors from training data
        print("Training model...")
        
        for taxi_id in tqdm(X['taxi_id'].unique(), desc="Calculating velocity factors"):
            taxi_data = X[X['taxi_id'] == taxi_id].sort_values('date_time')
            
            if len(taxi_data) > 1:
                # Calculate average velocity components
                lons = taxi_data['longitude'].values
                lats = taxi_data['latitude'].values
                
                # Calculate velocity components
                dx = np.diff(lons)
                dy = np.diff(lats)
                
                # Store average velocity components for this taxi
                if len(dx) > 0 and len(dy) > 0:
                    self.velocity_factors[taxi_id] = (np.mean(dx), np.mean(dy))
        
        return self
    
    def predict(self, X):
        predictions = []
        
        for idx, row in X.iterrows():
            taxi_id = row['taxi_id']
            
            if taxi_id in self.velocity_factors:
                dx, dy = self.velocity_factors[taxi_id]
                
                # Simple prediction: current position + average velocity
                pred_lon = row['longitude'] + dx * self.forecast
                pred_lat = row['latitude'] + dy * self.forecast
            else:
                # If no history, just return the current position
                pred_lon = row['longitude']
                pred_lat = row['latitude']
            
            predictions.append((pred_lon, pred_lat))
        
        return predictions
    
    def evaluate(self, X, y_true_lon, y_true_lat):
        """Evaluate model performance"""
        y_pred = self.predict(X)
        pred_lons = [p[0] for p in y_pred]
        pred_lats = [p[1] for p in y_pred]
        
        # Calculate errors
        rmse_lon = np.sqrt(mean_squared_error(y_true_lon, pred_lons))
        rmse_lat = np.sqrt(mean_squared_error(y_true_lat, pred_lats))
        mae_lon = mean_absolute_error(y_true_lon, pred_lons)
        mae_lat = mean_absolute_error(y_true_lat, pred_lats)
        
        # Calculate Haversine distance error
        distances = []
        for i in range(len(y_true_lon)):
            dist = haversine_distance(y_true_lat[i], y_true_lon[i], pred_lats[i], pred_lons[i])
            distances.append(dist)
        
        mean_distance_error = np.mean(distances)
        median_distance_error = np.median(distances)
        std_distance_error = np.std(distances)
        
        # Calculate accuracy within different thresholds
        accuracy_1km = np.mean(np.array(distances) <= 1.0) * 100
        accuracy_2km = np.mean(np.array(distances) <= 2.0) * 100
        accuracy_5km = np.mean(np.array(distances) <= 5.0) * 100
        
        self.accuracy_metrics = {
            'rmse_lon': rmse_lon,
            'rmse_lat': rmse_lat,
            'mae_lon': mae_lon,
            'mae_lat': mae_lat,
            'mean_distance_error_km': mean_distance_error,
            'median_distance_error_km': median_distance_error,
            'std_distance_error_km': std_distance_error,
            'accuracy_1km': accuracy_1km,
            'accuracy_2km': accuracy_2km,
            'accuracy_5km': accuracy_5km,
            'distances': distances
        }
        
        return self.accuracy_metrics
    
    def save_model(self, filepath):
        """Save the trained model to a file"""
        model_data = {
            'velocity_factors': self.velocity_factors,
            'lookback': self.lookback,
            'forecast': self.forecast,
            'accuracy_metrics': self.accuracy_metrics
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model from a file"""
        model_data = joblib.load(filepath)
        self.velocity_factors = model_data['velocity_factors']
        self.lookback = model_data['lookback']
        self.forecast = model_data['forecast']
        self.accuracy_metrics = model_data.get('accuracy_metrics', {})
        print(f"Model loaded from {filepath}")
        return self

# Load and preprocess data
def load_data(file_path, sample_fraction=0.1, chunk_size=1000):
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    # Sample data for faster processing
    if sample_fraction < 10.0:
        unique_taxis = df['taxi_id'].unique()
        sampled_taxis = np.random.choice(unique_taxis, 
                                        size=int(len(unique_taxis) * sample_fraction), 
                                        replace=False)
        df = df[df['taxi_id'].isin(sampled_taxis)]
    
    # Convert date_time to datetime
    df['date_time'] = pd.to_datetime(df['date_time'])
    
    # Sort by taxi_id and date_time
    df = df.sort_values(['taxi_id', 'date_time'])
    
    # Create shifted columns for next position (target)
    df['next_longitude'] = df.groupby('taxi_id')['longitude'].shift(-1)
    df['next_latitude'] = df.groupby('taxi_id')['latitude'].shift(-1)
    
    # Remove rows with no next position
    df = df.dropna(subset=['next_longitude', 'next_latitude'])
    
    # Return only a chunk of data for testing
    if chunk_size > 0:
        df = df.head(chunk_size)
    
    return df

# Function to find k nearest servers
def find_k_nearest_servers(lat, lon, servers_df, k=3):
    distances = []
    for _, server in servers_df.iterrows():
        dist = haversine_distance(lat, lon, server['latitude'], server['longitude'])
        distances.append((server['edge_server_id'], dist))
    
    # Sort by distance and return top k
    distances.sort(key=lambda x: x[1])
    return [server_id for server_id, _ in distances[:k]]

# Define your DQN network
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# Replay buffer for experience replay
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    def __len__(self):
        return len(self.buffer)

# Simulated Annealing function
def simulated_annealing(initial_servers, current_server, lat, lon, candidates, servers_df, temp=100, cooling_rate=0.95, max_iter=50):
    def compute_cost(assignment):
        total_dist = 0
        for s_id in assignment:
            s_info = servers_df[servers_df['edge_server_id'] == s_id].iloc[0]
            dist = haversine_distance(lat, lon, s_info['latitude'], s_info['longitude'])
            total_dist += dist
        return total_dist
    current_assignment = [current_server]
    current_cost = compute_cost(current_assignment)
    best_assignment = current_assignment
    best_cost = current_cost
    temp_current = temp
    for _ in range(max_iter):
        neighbor = current_assignment.copy()
        new_server = random.choice(candidates)
        neighbor[0] = new_server
        neighbor_cost = compute_cost(neighbor)
        delta = neighbor_cost - current_cost
        if delta < 0 or random.random() < np.exp(-delta / temp_current):
            current_assignment = neighbor
            current_cost = neighbor_cost
            if current_cost < best_cost:
                best_assignment = current_assignment
                best_cost = current_cost
        temp_current *= cooling_rate
        if temp_current < 1e-3:
            break
    return best_assignment

# Function to find the nearest server
def find_nearest_server(lat, lon, servers_df):
    min_dist = float('inf')
    server_id = None
    for _, s in servers_df.iterrows():
        dist = haversine_distance(lat, lon, s['latitude'], s['longitude'])
        if dist < min_dist:
            min_dist = dist
            server_id = s['edge_server_id']
    return server_id, min_dist

# Main simulation function
def run_simulation(trajectory_model, df, servers, use_prediction=True):
    total_migrations = 0
    total_violations = 0
    taxi_assignments = {}
    epsilon = 1.0
    
    # Prepare device and models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQN(4, 4).to(device)
    target_net = DQN(4, 4).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    replay_buffer = ReplayBuffer(10000)
    
    # Timestamps from data
    timestamps = sorted(df['date_time'].unique())
    
    # Create progress bar
    total_steps = len(timestamps)
    pbar = tqdm(total=total_steps, desc="Processing timestamps")

    for epoch in range(1):  # single epoch
        for t_idx, timestamp in enumerate(timestamps):
            df_time = df[df['date_time'] == timestamp]
            for _, row in df_time.iterrows():
                lat, lon, taxi_id = row['latitude'], row['longitude'], row['taxi_id']
                if taxi_id not in taxi_assignments:
                    # assign initially to nearest server
                    server_id, _ = find_nearest_server(lat, lon, servers)
                    taxi_assignments[taxi_id] = server_id
                    continue

                current_server_id = taxi_assignments[taxi_id]
                current_server_info = servers[servers['edge_server_id'] == current_server_id].iloc[0]
                server_lat, server_lon = current_server_info['latitude'], current_server_info['longitude']

                # Candidate servers
                candidates = find_k_nearest_servers(lat, lon, servers, k=3)
                if current_server_id not in candidates:
                    candidates.append(current_server_id)

                # Prediction step
                if use_prediction:
                    pred = trajectory_model.predict(pd.DataFrame([row]))
                    pred_lon, pred_lat = pred[0]
                else:
                    pred_lon, pred_lat = lon, lat

                # Compute distances from predicted position to candidate servers
                distances = []
                for s_id in candidates:
                    s_info = servers[servers['edge_server_id'] == s_id].iloc[0]
                    dist = haversine_distance(pred_lat, pred_lon, s_info['latitude'], s_info['longitude'])
                    distances.append(dist)

                # Current server index in candidates
                try:
                    current_server_idx = candidates.index(current_server_id)
                except:
                    current_server_idx = 0

                min_dist = min(distances)

                # Build state
                state_np = np.array([
                    lat,
                    lon,
                    current_server_idx / max(1, len(candidates)),
                    min_dist / 100
                ], dtype=np.float32)
                state = torch.FloatTensor(state_np).unsqueeze(0).to(device)

                # Epsilon-greedy action
                if random.random() < epsilon:
                    action = random.randint(0, 3)
                else:
                    with torch.no_grad():
                        q_vals = policy_net(state)
                        action = q_vals.argmax().item()

                # Map action to server
                if action == 0:
                    chosen_server = current_server_id
                else:
                    chosen_server = candidates[action - 1]

                # Use SA to refine migration
                initial_assignment = [current_server_id]
                improved_assignment = simulated_annealing(
                    initial_servers=initial_assignment,
                    current_server=current_server_id,
                    lat=lat,
                    lon=lon,
                    candidates=candidates,
                    servers_df=servers,
                    temp=100,
                    cooling_rate=0.95,
                    max_iter=50
                )
                chosen_server = improved_assignment[0]

                # Compute distance from predicted position to chosen server
                chosen_server_info = servers[servers['edge_server_id'] == chosen_server].iloc[0]
                dist_chosen = haversine_distance(pred_lat, pred_lon, chosen_server_info['latitude'], chosen_server_info['longitude'])

                migration_occurred = False
                if chosen_server != current_server_id:
                    # Migrate
                    taxi_assignments[taxi_id] = chosen_server
                    total_migrations += 1
                    migration_occurred = True

                # Calculate violation based on current position
                dist_current_server = haversine_distance(lat, lon, current_server_info['latitude'], current_server_info['longitude'])
                violation = False
                if dist_current_server > 15:
                    total_violations += 1
                    violation = True

                # Next position prediction
                if use_prediction:
                    pred_next = trajectory_model.predict(pd.DataFrame([row]))
                    pred_lon_next, pred_lat_next = pred_next[0]
                else:
                    pred_lon_next, pred_lat_next = lon, lat

                # Next min distance
                distances_next = []
                for s_id in candidates:
                    s_info = servers[servers['edge_server_id'] == s_id].iloc[0]
                    dist_next = haversine_distance(pred_lat_next, pred_lon_next, s_info['latitude'], s_info['longitude'])
                    distances_next.append(dist_next)
                min_dist_next = min(distances_next)

                # Next server index
                try:
                    next_server_idx = candidates.index(chosen_server)
                except:
                    next_server_idx = 0

                # Next state
                next_state_np = np.array([
                    lat,
                    lon,
                    next_server_idx / max(1, len(candidates)),
                    min_dist_next / 100
                ], dtype=np.float32)
                next_state = torch.FloatTensor(next_state_np).unsqueeze(0).to(device)

                # Store experience
                reward = -1 if not migration_occurred else 0
                if violation:
                    reward -= 2
                replay_buffer.push(
                    state.cpu().numpy().squeeze(),
                    action,
                    reward,
                    next_state.cpu().numpy().squeeze(),
                    1 if violation else 0
                )

                # Learning step
                if len(replay_buffer) >= 64:
                    batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = replay_buffer.sample(64)
                    batch_states = torch.FloatTensor(batch_states).to(device)
                    batch_next_states = torch.FloatTensor(batch_next_states).to(device)
                    batch_actions = torch.LongTensor(batch_actions).unsqueeze(1).to(device)
                    batch_rewards = torch.FloatTensor(batch_rewards).unsqueeze(1).to(device)
                    batch_dones = torch.FloatTensor(batch_dones).unsqueeze(1).to(device)

                    q_vals = policy_net(batch_states).gather(1, batch_actions)
                    with torch.no_grad():
                        next_q = target_net(batch_next_states).max(1)[0].unsqueeze(1)
                    target_q = batch_rewards + 0.99 * next_q * (1 - batch_dones)

                    loss = nn.MSELoss()(q_vals, target_q)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # Update progress bar after processing each timestamp
            pbar.update(1)
            pbar.set_postfix({
                'Migrations': total_migrations,
                'Violations': total_violations,
                'Epsilon': f"{epsilon:.3f}"
            })

        epsilon = max(0.05, epsilon * 0.995)
        if (epoch + 1) % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())

    pbar.close()
    print(f"Use prediction={use_prediction} | Migrations={total_migrations} | Violations={total_violations}")
    return total_migrations, total_violations

# Main function
def main():
    # Parameters
    data_file = 'combined_taxi_with_health.csv'
    model_file = 'taxi_trajectory_model.pkl'
    sample_fraction = 0.1  # Use 10% of data for faster processing
    chunk_size = 1000  # Process only the first 1000 rows for testing
    
    # Load data
    df = load_data(data_file, sample_fraction, chunk_size)
    
    print(f"Data loaded: {len(df):,} records")
    print(f"Unique taxis: {df['taxi_id'].nunique():,}")
    
    # Split data into train and test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Create and train model
    start_time = time.time()
    model = SimpleSpatialTransformer(lookback=5, forecast=3)
    model.fit(train_df)
    training_time = time.time() - start_time
    
    # Evaluate model
    print("\nEvaluating model...")
    test_actual_lons = test_df['next_longitude'].values
    test_actual_lats = test_df['next_latitude'].values
    
    accuracy_metrics = model.evaluate(test_df, test_actual_lons, test_actual_lats)
    
    # Print results
    print("\n" + "="*60)
    print("MODEL PERFORMANCE METRICS")
    print("="*60)
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Longitude RMSE: {accuracy_metrics['rmse_lon']:.6f} degrees")
    print(f"Latitude RMSE: {accuracy_metrics['rmse_lat']:.6f} degrees")
    print(f"Longitude MAE: {accuracy_metrics['mae_lon']:.6f} degrees")
    print(f"Latitude MAE: {accuracy_metrics['mae_lat']:.6f} degrees")
    print(f"Mean Haversine distance error: {accuracy_metrics['mean_distance_error_km']:.2f} km")
    print(f"Median Haversine distance error: {accuracy_metrics['median_distance_error_km']:.2f} km")
    print(f"Std Dev of distance error: {accuracy_metrics['std_distance_error_km']:.2f} km")
    print(f"Accuracy within 1 km: {accuracy_metrics['accuracy_1km']:.2f}%")
    print(f"Accuracy within 2 km: {accuracy_metrics['accuracy_2km']:.2f}%")
    print(f"Accuracy within 5 km: {accuracy_metrics['accuracy_5km']:.2f}%")
    
    # Plot error distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(accuracy_metrics['distances'], bins=50, alpha=0.7, color='blue')
    plt.axvline(accuracy_metrics['mean_distance_error_km'], color='red', linestyle='dashed', 
                linewidth=2, label=f'Mean: {accuracy_metrics["mean_distance_error_km"]:.2f} km')
    plt.axvline(accuracy_metrics['median_distance_error_km'], color='green', linestyle='dashed', 
                linewidth=2, label=f'Median: {accuracy_metrics["median_distance_error_km"]:.2f} km')
    plt.xlabel('Haversine Distance Error (km)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot accuracy at different thresholds
    plt.subplot(1, 2, 2)
    thresholds = [1, 2, 5]
    accuracies = [accuracy_metrics['accuracy_1km'], 
                 accuracy_metrics['accuracy_2km'], 
                 accuracy_metrics['accuracy_5km']]
    plt.bar(range(len(thresholds)), accuracies, color='orange', alpha=0.7)
    plt.xticks(range(len(thresholds)), [f'{t} km' for t in thresholds])
    plt.xlabel('Distance Threshold')
    plt.ylabel('Accuracy (%)')
    plt.title('Prediction Accuracy at Different Thresholds')
    for i, v in enumerate(accuracies):
        plt.text(i, v + 1, f'{v:.1f}%', ha='center')
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_performance.png', dpi=150)
    plt.show()
    
    # Save the trained model
    model.save_model(model_file)
    
    # Load server locations
    servers = pd.read_csv('edge_server_locations.csv')  # Ensure this CSV is correct
    
    # Run simulations with and without prediction
    print("\nRunning simulations...")
    migrations_with_pred, violations_with_pred = run_simulation(model, test_df, servers, use_prediction=True)
    migrations_without_pred, violations_without_pred = run_simulation(model, test_df, servers, use_prediction=False)
    
    print("\n=== Summary ===")
    print(f"Migrations with prediction: {migrations_with_pred}")
    print(f"Violations with prediction: {violations_with_pred}")
    print(f"Migrations without prediction: {migrations_without_pred}")
    print(f"Violations without prediction: {violations_without_pred}")
    
    # Calculate improvement
    if migrations_without_pred > 0:
        migration_reduction = ((migrations_without_pred - migrations_with_pred) / migrations_without_pred) * 100
    else:
        migration_reduction = 0
        
    if violations_without_pred > 0:
        violation_reduction = ((violations_without_pred - violations_with_pred) / violations_without_pred) * 100
    else:
        violation_reduction = 0
    
    print(f"\nImprovement from using prediction:")
    print(f"Migration reduction: {migration_reduction:.2f}%")
    print(f"Violation reduction: {violation_reduction:.2f}%")

if __name__ == "__main__":
    main()