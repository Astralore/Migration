# LSTM-based Trajectory Prediction 
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
                        print(f"  Warning: Could not parse line {line_num}: {line.strip()} - {e}")
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
        print("  Warning: Could not sort by timestamp, using file order")
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
            time_gap = 2  # Assume 2 minute gap if timestamp parsing fails
            
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
        start_lon, start_lat = traj[0][1], traj[0][0]  # lon, lat
        end_lon, end_lat = traj[-1][1], traj[-1][0]    # lon, lat
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

# LSTM Model (Replacing CNN)
class TrajectoryLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, num_layers=2, output_size=2, dropout=0.3):
        super(TrajectoryLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape input: (batch_size, seq_length * 2) -> (batch_size, seq_length, 2)
        x = x.view(batch_size, -1, self.input_size)
        
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        
        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Use the last output from the LSTM
        out = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        
        return out

def train_model_with_convergence(model, train_loader, val_loader, criterion, optimizer, patience=10, epochs=100):
    """Train model with detailed convergence tracking"""
    model.train()
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    convergence_data = []
    
    print(f"\nTraining LSTM for {epochs} epochs with patience {patience}")
    print("Epoch | Train Loss | Val Loss  | Improvement | Status")
    print("------|------------|-----------|-------------|--------")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping for LSTM stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                loss = criterion(output, target)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Check for improvement
        improvement = best_val_loss - avg_val_loss
        status = "✓" if improvement > 0 else "✗"
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_lstm_model.pth')
        else:
            patience_counter += 1
        
        # Store convergence data
        convergence_data.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'improvement': improvement,
            'best_val_loss': best_val_loss
        })
        
        # Print progress
        if epoch % 5 == 0 or epoch == epochs - 1 or patience_counter >= patience:
            print(f"{epoch:4d}  | {avg_train_loss:8.6f}  | {avg_val_loss:8.6f}  | {improvement:+.6f}  | {status}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break
    
    # Load best model
    model.load_state_dict(torch.load('best_lstm_model.pth'))
    
    return train_losses, val_losses, convergence_data

def analyze_convergence(convergence_data):
    """Analyze and report convergence patterns"""
    if not convergence_data:
        return {}
    
    final_epoch = convergence_data[-1]['epoch']
    initial_train_loss = convergence_data[0]['train_loss']
    final_train_loss = convergence_data[-1]['train_loss']
    initial_val_loss = convergence_data[0]['val_loss']
    final_val_loss = convergence_data[-1]['val_loss']
    
    train_reduction = ((initial_train_loss - final_train_loss) / initial_train_loss) * 100
    val_reduction = ((initial_val_loss - final_val_loss) / initial_val_loss) * 100
    
    # Find when validation loss stopped improving significantly
    best_val_epoch = 0
    best_val_loss = float('inf')
    for i, data in enumerate(convergence_data):
        if data['val_loss'] < best_val_loss:
            best_val_loss = data['val_loss']
            best_val_epoch = data['epoch']
    
    convergence_metrics = {
        'final_epoch': final_epoch,
        'initial_train_loss': initial_train_loss,
        'final_train_loss': final_train_loss,
        'train_reduction_pct': train_reduction,
        'initial_val_loss': initial_val_loss,
        'final_val_loss': final_val_loss,
        'val_reduction_pct': val_reduction,
        'best_val_epoch': best_val_epoch,
        'best_val_loss': best_val_loss,
        'convergence_epoch': best_val_epoch,
        'overfitting_gap': final_train_loss - final_val_loss,
    }
    
    return convergence_metrics

def calculate_comprehensive_metrics(predictions, targets, scaler_lon, scaler_lat, threshold=100):
    """Calculate comprehensive metrics including MSE, MAE, Precision, Recall, F1"""
    
    # Calculate MSE and MAE on normalized coordinates
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(mse)
    
    # Denormalize for distance calculations
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
    
    # Distance-based accuracy metrics
    accuracy_metrics = {
        'mean_distance': np.mean(distances),
        'median_distance': np.median(distances),
        'std_distance': np.std(distances),
        'acc_100m': np.mean(distances <= 100) * 100,
        'acc_200m': np.mean(distances <= 200) * 100,
        'acc_500m': np.mean(distances <= 500) * 100,
        'acc_1000m': np.mean(distances <= 1000) * 100,
        'acc_2000m': np.mean(distances <= 2000) * 100,
    }
    
    # Classification metrics (using threshold for binary classification)
    y_true = (distances <= threshold).astype(int)
    y_pred = (distances <= threshold).astype(int)
    
    # Handle case where all predictions are the same class
    if len(np.unique(y_true)) > 1:
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
    else:
        precision = recall = f1 = accuracy = 0.0
    
    classification_metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy,
    }
    
    # Regression metrics
    regression_metrics = {
        'mse': mse,
        'mae': mae,
        'rmse': rmse
    }
    
    # Combine all metrics
    return {**accuracy_metrics, **classification_metrics, **regression_metrics}

def process_single_taxi_file(file_path):
    """Process a single taxi file using the LSTM approach"""
    print(f"\nProcessing {os.path.basename(file_path)} with LSTM...")
    print("=" * 60)
    
    # Load data
    data = load_taxi_data(file_path)
    if len(data) == 0:
        print("No data loaded")
        return None, None
    
    print(f"Total data points: {len(data)}")
    
    # Extract trajectories
    trajectories = extract_trajectories(data, min_points=8, max_gap_minutes=15)
    if len(trajectories) == 0:
        print("No meaningful trajectories found")
        return None, None
    
    print(f"Meaningful trajectories: {len(trajectories)}")
    
    # Cluster trajectories
    clustered_trajs = cluster_trajectories(trajectories, eps=0.01, min_samples=2)
    print(f"Trajectory clusters: {len(clustered_trajs)}")
    
    # Create sequences
    sequences, targets = create_coordinate_sequences(clustered_trajs, seq_length=8)
    if len(sequences) == 0:
        print("No valid sequences created")
        return None, None
    
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
    
    # Train LSTM model
    model = TrajectoryLSTM(input_size=2, hidden_size=128, num_layers=2, output_size=2, dropout=0.3)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Train with convergence tracking
    train_losses, val_losses, convergence_data = train_model_with_convergence(
        model, train_loader, val_loader, criterion, optimizer, patience=10, epochs=50
    )
    
    # Analyze convergence
    convergence_metrics = analyze_convergence(convergence_data)
    
    # Evaluate
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            all_predictions.extend(output.numpy())
            all_targets.extend(target.numpy())
    
    # Calculate comprehensive metrics
    metrics = calculate_comprehensive_metrics(np.array(all_predictions), np.array(all_targets), scaler_lon, scaler_lat)
    
    print(f"\nComprehensive Results for {os.path.basename(file_path)}:")
    print("=" * 60)
    
    print(f"\nConvergence Analysis:")
    print(f"  Final Epoch: {convergence_metrics['final_epoch']}")
    print(f"  Best Validation Loss: {convergence_metrics['best_val_loss']:.6f} (epoch {convergence_metrics['best_val_epoch']})")
    print(f"  Training Loss Reduction: {convergence_metrics['train_reduction_pct']:.1f}%")
    print(f"  Validation Loss Reduction: {convergence_metrics['val_reduction_pct']:.1f}%")
    print(f"  Overfitting Gap: {convergence_metrics['overfitting_gap']:.6f}")
    
    print(f"\nRegression Metrics:")
    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  MAE: {metrics['mae']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  Mean Distance Error: {metrics['mean_distance']:.2f}m")
    print(f"  Median Distance Error: {metrics['median_distance']:.2f}m")
    print(f"  Standard Deviation: {metrics['std_distance']:.2f}m")
    
    print(f"\nAccuracy Metrics:")
    print(f"  Accuracy @ 100m: {metrics['acc_100m']:.2f}%")
    print(f"  Accuracy @ 200m: {metrics['acc_200m']:.2f}%")
    print(f"  Accuracy @ 500m: {metrics['acc_500m']:.2f}%")
    print(f"  Accuracy @ 1000m: {metrics['acc_1000m']:.2f}%")
    print(f"  Accuracy @ 2000m: {metrics['acc_2000m']:.2f}%")
    
    print(f"\nClassification Metrics (100m threshold):")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    
    return metrics, convergence_metrics

def main():
    data_dir = 'taxi_data'  # Directory containing taxi files
    file_list = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    
    print(f"Found {len(file_list)} taxi files")
    
    all_metrics = []
    all_convergence = []
    
    for file_name in file_list:
        file_path = os.path.join(data_dir, file_name)
        metrics, convergence = process_single_taxi_file(file_path)
        
        if metrics is not None and convergence is not None:
            all_metrics.append(metrics)
            all_convergence.append(convergence)
        print("\n" + "="*80)
    
    # Calculate aggregate results
    if all_metrics and all_convergence:
        print("\n" + "="*80)
        print("AGGREGATE RESULTS ACROSS ALL TAXIS (LSTM)")
        print("="*80)
        
        # Performance metrics
        avg_metrics = {
            'mse': np.mean([m['mse'] for m in all_metrics]),
            'mae': np.mean([m['mae'] for m in all_metrics]),
            'rmse': np.mean([m['rmse'] for m in all_metrics]),
            'mean_distance': np.mean([m['mean_distance'] for m in all_metrics]),
            'median_distance': np.median([m['median_distance'] for m in all_metrics]),
            'std_distance': np.mean([m['std_distance'] for m in all_metrics]),
            'acc_100m': np.mean([m['acc_100m'] for m in all_metrics]),
            'acc_200m': np.mean([m['acc_200m'] for m in all_metrics]),
            'acc_500m': np.mean([m['acc_500m'] for m in all_metrics]),
            'acc_1000m': np.mean([m['acc_1000m'] for m in all_metrics]),
            'acc_2000m': np.mean([m['acc_2000m'] for m in all_metrics]),
            'precision': np.mean([m['precision'] for m in all_metrics]),
            'recall': np.mean([m['recall'] for m in all_metrics]),
            'f1_score': np.mean([m['f1_score'] for m in all_metrics]),
            'accuracy': np.mean([m['accuracy'] for m in all_metrics]),
        }
        
        # Convergence metrics
        avg_convergence = {
            'avg_final_epoch': np.mean([c['final_epoch'] for c in all_convergence]),
            'avg_best_val_epoch': np.mean([c['best_val_epoch'] for c in all_convergence]),
            'avg_train_reduction': np.mean([c['train_reduction_pct'] for c in all_convergence]),
            'avg_val_reduction': np.mean([c['val_reduction_pct'] for c in all_convergence]),
            'avg_overfitting_gap': np.mean([c['overfitting_gap'] for c in all_convergence]),
        }
        
        print(f"\nAverage Convergence Metrics:")
        print(f"  Average Final Epoch: {avg_convergence['avg_final_epoch']:.1f}")
        print(f"  Average Best Validation Epoch: {avg_convergence['avg_best_val_epoch']:.1f}")
        print(f"  Average Training Loss Reduction: {avg_convergence['avg_train_reduction']:.1f}%")
        print(f"  Average Validation Loss Reduction: {avg_convergence['avg_val_reduction']:.1f}%")
        print(f"  Average Overfitting Gap: {avg_convergence['avg_overfitting_gap']:.6f}")
        
        print(f"\nAverage Performance Metrics:")
        print(f"Regression Metrics:")
        print(f"  MSE: {avg_metrics['mse']:.6f}")
        print(f"  MAE: {avg_metrics['mae']:.6f}")
        print(f"  RMSE: {avg_metrics['rmse']:.6f}")
        print(f"  Mean Distance Error: {avg_metrics['mean_distance']:.2f}m")
        print(f"  Median Distance Error: {avg_metrics['median_distance']:.2f}m")
        print(f"  Standard Deviation: {avg_metrics['std_distance']:.2f}m")
        
        print(f"\nAccuracy Metrics:")
        print(f"  Accuracy @ 100m: {avg_metrics['acc_100m']:.2f}%")
        print(f"  Accuracy @ 200m: {avg_metrics['acc_200m']:.2f}%")
        print(f"  Accuracy @ 500m: {avg_metrics['acc_500m']:.2f}%")
        print(f"  Accuracy @ 1000m: {avg_metrics['acc_1000m']:.2f}%")
        print(f"  Accuracy @ 2000m: {avg_metrics['acc_2000m']:.2f}%")
        
        print(f"\nClassification Metrics (100m threshold):")
        print(f"  Precision: {avg_metrics['precision']:.4f}")
        print(f"  Recall: {avg_metrics['recall']:.4f}")
        print(f"  F1 Score: {avg_metrics['f1_score']:.4f}")
        print(f"  Accuracy: {avg_metrics['accuracy']:.4f}")
        
        print(f"\nBased on {len(all_metrics)} successfully processed taxis")

if __name__ == "__main__":
    main()