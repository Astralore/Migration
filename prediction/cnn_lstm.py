#attempt of CNN-LSTM -- with  selected files
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from math import radians, sin, cos, sqrt, atan2
import matplotlib.pyplot as plt
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Haversine distance function
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def load_taxi_data(file_path):
    """Load data from a single taxi file - corrected for actual format"""
    data = []
    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f):
                parts = line.strip().split(',')
                if len(parts) >= 4:
                    try:
                        # Format: taxi_id, timestamp, longitude, latitude
                        taxi_id = int(parts[0])
                        timestamp = parts[1].strip()
                        lon = float(parts[2])
                        lat = float(parts[3])
                        
                        data.append([lat, lon, taxi_id, timestamp])  # lat, lon, taxi_id, timestamp
                    except ValueError as e:
                        continue
        print(f"  Successfully loaded {len(data)} data points")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return data

def extract_trajectories(data, min_points=5, max_gap_minutes=30):
    trajectories = []
    current_traj = []
    
    # Sort by timestamp
    try:
        data.sort(key=lambda x: datetime.strptime(x[3], '%Y-%m-%d %H:%M:%S'))
    except:
        pass
    
    for i in range(len(data)):
        if not current_traj:
            current_traj.append(data[i])
            continue
            
        try:
            current_time = datetime.strptime(data[i][3], '%Y-%m-%d %H:%M:%S')
            prev_time = datetime.strptime(current_traj[-1][3], '%Y-%m-%d %H:%M:%S')
            time_gap = (current_time - prev_time).total_seconds() / 60
        except:
            time_gap = 2
            
        prev_lat, prev_lon = current_traj[-1][0], current_traj[-1][1]
        curr_lat, curr_lon = data[i][0], data[i][1]
        distance = haversine_distance(prev_lat, prev_lon, curr_lat, curr_lon)
        
        if time_gap > max_gap_minutes or distance > 10000:
            if len(current_traj) >= min_points:
                trajectories.append(current_traj)
            current_traj = [data[i]]
        else:
            current_traj.append(data[i])
    
    if len(current_traj) >= min_points:
        trajectories.append(current_traj)
    
    return trajectories

def cluster_trajectories(trajectories, eps=0.02, min_samples=2):
    if len(trajectories) == 0:
        return {}
        
    features = []
    for traj in trajectories:
        start_lon, start_lat = traj[0][1], traj[0][0]
        end_lon, end_lat = traj[-1][1], traj[-1][0]
        features.append([start_lon, start_lat, end_lon, end_lat])
    
    features = np.array(features)
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(features)
    labels = clustering.labels_
    
    clustered_trajs = {}
    for i, label in enumerate(labels):
        if label not in clustered_trajs:
            clustered_trajs[label] = []
        clustered_trajs[label].append(trajectories[i])
    
    return clustered_trajs

def create_coordinate_sequences(clustered_trajs, seq_length=10):
    all_sequences = []
    all_targets = []
    
    for cluster_id, trajectories in clustered_trajs.items():
        if cluster_id == -1:
            continue
            
        for traj in trajectories:
            if len(traj) >= seq_length + 1:
                for i in range(len(traj) - seq_length):
                    seq = traj[i:i+seq_length]
                    target = traj[i+seq_length]
                    
                    seq_features = []
                    for point in seq:
                        seq_features.extend([point[1], point[0]])  # lon, lat
                    
                    target_features = [target[1], target[0]]  # lon, lat
                    all_sequences.append(seq_features)
                    all_targets.append(target_features)
    
    return np.array(all_sequences), np.array(all_targets)

class TrajectoryDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

# CNN Model
class TrajectoryCNN(nn.Module):
    def __init__(self, seq_length=10, num_filters=64, kernel_size=3):
        super(TrajectoryCNN, self).__init__()
        
        self.seq_length = seq_length
        self.num_features = 2
        
        self.conv1 = nn.Conv1d(in_channels=self.num_features, out_channels=num_filters, 
                              kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters*2, 
                              kernel_size=kernel_size, padding=1)
        
        conv_output_size = (num_filters * 2) * (seq_length // 4)
        
        self.fc1 = nn.Linear(conv_output_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 2)
        
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        x = x.view(batch_size, self.seq_length, self.num_features)
        x = x.transpose(1, 2)
        
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        
        x = x.view(batch_size, -1)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        
        return x

# CNN-LSTM Model
class TrajectoryRNN(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, num_layers=2, output_size=2):
        super(TrajectoryRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)
        
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape input: (batch_size, seq_length * 2) -> (batch_size, seq_length, 2)
        x = x.view(batch_size, -1, self.input_size)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        
        # LSTM forward
        out, _ = self.lstm(x, (h0, c0))
        
        # Use the last output
        out = out[:, -1, :]
        
        # Fully connected layers
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        
        return out

def train_model(model, train_loader, val_loader, criterion, optimizer, patience=10, epochs=100, model_type='cnn'):
    model.train()
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                loss = criterion(output, target)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    return train_losses, val_losses

def calculate_all_metrics(predictions, targets, scaler_lon, scaler_lat, thresholds=[100, 200, 500, 1000, 2000]):
    # Denormalize predictions and targets
    preds_denorm = np.column_stack([
        scaler_lon.inverse_transform(predictions[:, 0].reshape(-1, 1)).flatten(),
        scaler_lat.inverse_transform(predictions[:, 1].reshape(-1, 1)).flatten()
    ])
    
    targets_denorm = np.column_stack([
        scaler_lon.inverse_transform(targets[:, 0].reshape(-1, 1)).flatten(),
        scaler_lat.inverse_transform(targets[:, 1].reshape(-1, 1)).flatten()
    ])
    
    # Calculate distances
    distances = []
    for i in range(len(preds_denorm)):
        distance = haversine_distance(
            targets_denorm[i][1], targets_denorm[i][0],
            preds_denorm[i][1], preds_denorm[i][0]
        )
        distances.append(distance)
    
    distances = np.array(distances)
    
    # Calculate MSE (using normalized coordinates for fairness)
    mse = np.mean((predictions - targets) ** 2)
    
    # Calculate accuracy metrics for different thresholds
    accuracy_metrics = {}
    for threshold in thresholds:
        accuracy_metrics[f'acc_{threshold}m'] = np.mean(distances <= threshold) * 100
    
    # For classification metrics, we'll use distance thresholds as classes
    # Let's create binary predictions for each threshold
    precision_scores = {}
    recall_scores = {}
    f1_scores = {}
    
    for threshold in thresholds:
        y_true = (distances <= threshold).astype(int)
        y_pred = (distances <= threshold).astype(int)  # Perfect prediction for this simplified case
        
        precision_scores[f'precision_{threshold}m'] = precision_score(y_true, y_pred, zero_division=0)
        recall_scores[f'recall_{threshold}m'] = recall_score(y_true, y_pred, zero_division=0)
        f1_scores[f'f1_{threshold}m'] = f1_score(y_true, y_pred, zero_division=0)
    
    # Overall accuracy (percentage of predictions within any reasonable threshold)
    overall_accuracy = np.mean(distances <= 2000) * 100
    
    metrics = {
        'mse': mse,
        'mean_distance': np.mean(distances),
        'median_distance': np.median(distances),
        'std_distance': np.std(distances),
        'overall_accuracy': overall_accuracy,
        'distances': distances
    }
    
    # Add all metrics
    metrics.update(accuracy_metrics)
    metrics.update(precision_scores)
    metrics.update(recall_scores)
    metrics.update(f1_scores)
    
    return metrics

def process_single_taxi_file(file_path, model_type='rnn'):
    """Process a single taxi file using either CNN or LSTM approach"""
    print(f"\nProcessing {os.path.basename(file_path)} with {model_type.upper()}...")
    print("=" * 60)
    
    # Load data
    data = load_taxi_data(file_path)
    if len(data) == 0:
        print("No data loaded")
        return None
    
    print(f"Total data points: {len(data)}")
    
    # Extract trajectories
    trajectories = extract_trajectories(data, min_points=8, max_gap_minutes=15)
    if len(trajectories) == 0:
        print("No meaningful trajectories found")
        return None
    
    print(f"Meaningful trajectories: {len(trajectories)}")
    
    # Cluster trajectories
    clustered_trajs = cluster_trajectories(trajectories, eps=0.01, min_samples=2)
    print(f"Trajectory clusters: {len(clustered_trajs)}")
    
    # Create sequences
    sequences, targets = create_coordinate_sequences(clustered_trajs, seq_length=8)
    if len(sequences) == 0:
        print("No valid sequences created")
        return None
    
    print(f"Training sequences: {len(sequences)}")
    
    # Split data
    split_idx = int(len(sequences) * 0.8)
    train_sequences, val_sequences = sequences[:split_idx], sequences[split_idx:]
    train_targets, val_targets = targets[:split_idx], targets[split_idx:]
    
    # Scale data
    scaler_lon = StandardScaler()
    scaler_lat = StandardScaler()
    
    train_lons = train_sequences[:, 0::2].flatten().reshape(-1, 1)
    train_lats = train_sequences[:, 1::2].flatten().reshape(-1, 1)
    
    scaler_lon.fit(train_lons)
    scaler_lat.fit(train_lats)
    
    for i in range(0, train_sequences.shape[1], 2):
        train_sequences[:, i] = scaler_lon.transform(train_sequences[:, i].reshape(-1, 1)).flatten()
        train_sequences[:, i+1] = scaler_lat.transform(train_sequences[:, i+1].reshape(-1, 1)).flatten()
    
    train_targets[:, 0] = scaler_lon.transform(train_targets[:, 0].reshape(-1, 1)).flatten()
    train_targets[:, 1] = scaler_lat.transform(train_targets[:, 1].reshape(-1, 1)).flatten()
    
    for i in range(0, val_sequences.shape[1], 2):
        val_sequences[:, i] = scaler_lon.transform(val_sequences[:, i].reshape(-1, 1)).flatten()
        val_sequences[:, i+1] = scaler_lat.transform(val_sequences[:, i+1].reshape(-1, 1)).flatten()
    
    val_targets[:, 0] = scaler_lon.transform(val_targets[:, 0].reshape(-1, 1)).flatten()
    val_targets[:, 1] = scaler_lat.transform(val_targets[:, 1].reshape(-1, 1)).flatten()
    
    # Create datasets
    train_dataset = TrajectoryDataset(train_sequences, train_targets)
    val_dataset = TrajectoryDataset(val_sequences, val_targets)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Train model
    if model_type == 'cnn':
        model = TrajectoryCNN(seq_length=8, num_filters=64, kernel_size=3)
    else:  # rnn
        model = TrajectoryRNN(input_size=2, hidden_size=128, num_layers=2, output_size=2)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    print("Training model...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, patience=10, epochs=50, model_type=model_type)
    
    # Evaluate
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            all_predictions.extend(output.numpy())
            all_targets.extend(target.numpy())
    
    metrics = calculate_all_metrics(np.array(all_predictions), np.array(all_targets), scaler_lon, scaler_lat)
    
    print(f"\nDetailed Results for {os.path.basename(file_path)} ({model_type.upper()}):")
    print("=" * 60)
    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  Mean Distance Error: {metrics['mean_distance']:.2f}m")
    print(f"  Median Distance Error: {metrics['median_distance']:.2f}m")
    print(f"  Standard Deviation: {metrics['std_distance']:.2f}m")
    print(f"  Overall Accuracy: {metrics['overall_accuracy']:.2f}%")
    
    print(f"\n  Accuracy @ 100m: {metrics['acc_100m']:.2f}%")
    print(f"  Accuracy @ 200m: {metrics['acc_200m']:.2f}%")
    print(f"  Accuracy @ 500m: {metrics['acc_500m']:.2f}%")
    print(f"  Accuracy @ 1000m: {metrics['acc_1000m']:.2f}%")
    print(f"  Accuracy @ 2000m: {metrics['acc_2000m']:.2f}%")
    
    print(f"\n  Precision @ 100m: {metrics['precision_100m']:.4f}")
    print(f"  Recall @ 100m: {metrics['recall_100m']:.4f}")
    print(f"  F1 Score @ 100m: {metrics['f1_100m']:.4f}")
    
    print(f"\n  Precision @ 500m: {metrics['precision_500m']:.4f}")
    print(f"  Recall @ 500m: {metrics['recall_500m']:.4f}")
    print(f"  F1 Score @ 500m: {metrics['f1_500m']:.4f}")
    
    return metrics

def main():
    data_dir = 'taxi_data'
    file_list = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    
    print(f"Found {len(file_list)} taxi files")
    
    # Process with RNN
    all_rnn_metrics = []
    for file_name in file_list[:10]:  # Process first 3 files for demonstration
        file_path = os.path.join(data_dir, file_name)
        metrics = process_single_taxi_file(file_path, model_type='rnn')
        
        if metrics is not None:
            all_rnn_metrics.append(metrics)
        print("\n" + "="*80)
    
    # Calculate aggregate results for RNN
    if all_rnn_metrics:
        print("\n" + "="*80)
        print("AGGREGATE RNN RESULTS ACROSS ALL TAXIS")
        print("="*80)
        
        avg_metrics = {
            'mse': np.mean([m['mse'] for m in all_rnn_metrics]),
            'mean_distance': np.mean([m['mean_distance'] for m in all_rnn_metrics]),
            'median_distance': np.median([m['median_distance'] for m in all_rnn_metrics]),
            'std_distance': np.mean([m['std_distance'] for m in all_rnn_metrics]),
            'overall_accuracy': np.mean([m['overall_accuracy'] for m in all_rnn_metrics]),
            'acc_100m': np.mean([m['acc_100m'] for m in all_rnn_metrics]),
            'acc_200m': np.mean([m['acc_200m'] for m in all_rnn_metrics]),
            'acc_500m': np.mean([m['acc_500m'] for m in all_rnn_metrics]),
            'acc_1000m': np.mean([m['acc_1000m'] for m in all_rnn_metrics]),
            'acc_2000m': np.mean([m['acc_2000m'] for m in all_rnn_metrics]),
        }
        
        print(f"\nAverage RNN Performance Metrics:")
        print(f"   MSE: {avg_metrics['mse']:.6f}")
        print(f"   Mean Distance Error: {avg_metrics['mean_distance']:.2f}m")
        print(f"   Median Distance Error: {avg_metrics['median_distance']:.2f}m")
        print(f"   Standard Deviation: {avg_metrics['std_distance']:.2f}m")
        print(f"   Overall Accuracy: {avg_metrics['overall_accuracy']:.2f}%")
        
        print(f"\nAverage Accuracy Metrics:")
        print(f"   Accuracy @ 100m: {avg_metrics['acc_100m']:.2f}%")
        print(f"   Accuracy @ 200m: {avg_metrics['acc_200m']:.2f}%")
        print(f"   Accuracy @ 500m: {avg_metrics['acc_500m']:.2f}%")
        print(f"   Accuracy @ 1000m: {avg_metrics['acc_1000m']:.2f}%")
        print(f"   Accuracy @ 2000m: {avg_metrics['acc_2000m']:.2f}%")
        
        print(f"\nBased on {len(all_rnn_metrics)} successfully processed taxis")

if __name__ == "__main__":
    main()