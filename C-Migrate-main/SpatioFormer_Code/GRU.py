# PROPER GRU-based Trajectory Prediction with Debugging
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

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def load_taxi_data(file_path):
    """Load data from a single taxi file"""
    data = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 4:
                    try:
                        taxi_id = int(parts[0])
                        timestamp = parts[1].strip()
                        lon = float(parts[2])
                        lat = float(parts[3])
                        data.append([lat, lon, taxi_id, timestamp])
                    except ValueError:
                        continue
        print(f"  Loaded {len(data)} data points")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return data

def create_sequences_directly(data, seq_length=10):
    """Create sequences directly without clustering for better comparison"""
    sequences = []
    targets = []
    
    # Use a sliding window approach
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        target = data[i+seq_length]
        
        seq_features = []
        for point in seq:
            seq_features.extend([point[1], point[0]])  # lon, lat
        
        target_features = [target[1], target[0]]  # lon, lat
        sequences.append(seq_features)
        targets.append(target_features)
    
    return np.array(sequences), np.array(targets)

class TrajectoryDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

# PROPER GRU Model with different architecture
class TrajectoryGRU(nn.Module):
    def __init__(self, input_size=2, hidden_size=256, num_layers=3, output_size=2, dropout=0.4):
        super(TrajectoryGRU, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        
        # GRU layers with different architecture
        self.gru1 = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.gru2 = nn.GRU(hidden_size, hidden_size//2, 1, batch_first=True)
        
        # Different fully connected architecture
        self.fc1 = nn.Linear(hidden_size//2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, output_size)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape and process through GRU
        x = x.view(batch_size, -1, self.input_size)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        
        # GRU forward pass
        out, _ = self.gru1(x, h0)
        out, _ = self.gru2(out)
        
        # Use the last output
        out = out[:, -1, :]
        
        # Fully connected layers with different architecture
        out = self.relu(self.bn1(self.fc1(out)))
        out = self.dropout(out)
        out = self.relu(self.bn2(self.fc2(out)))
        out = self.dropout(out)
        out = self.relu(self.fc3(out))
        out = self.dropout(out)
        out = self.fc4(out)
        
        return out

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=100):
    """Proper training function with debugging"""
    model.train()
    best_val_loss = float('inf')
    
    print("Training GRU model...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                val_loss += criterion(output, target).item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_gru_model.pth')
        
        if epoch % 20 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch}: Train Loss: {avg_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
    
    model.load_state_dict(torch.load('best_gru_model.pth'))
    return best_val_loss

def calculate_metrics(predictions, targets, scaler_lon, scaler_lat):
    """Calculate comprehensive metrics"""
    
    # Denormalize predictions
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
        dist = haversine_distance(
            targets_denorm[i][1], targets_denorm[i][0],
            preds_denorm[i][1], preds_denorm[i][0]
        )
        distances.append(dist)
    
    distances = np.array(distances)
    
    # Calculate all accuracy metrics
    metrics = {
        'mse': np.mean((predictions - targets) ** 2),
        'mae': np.mean(np.abs(predictions - targets)),
        'mean_distance': np.mean(distances),
        'median_distance': np.median(distances),
        'std_distance': np.std(distances),
        'acc_100m': np.mean(distances <= 100) * 100,
        'acc_200m': np.mean(distances <= 200) * 100,
        'acc_500m': np.mean(distances <= 500) * 100,
        'acc_1000m': np.mean(distances <= 1000) * 100,
        'acc_2000m': np.mean(distances <= 2000) * 100,
    }
    
    return metrics

def process_taxi_file(file_path):
    """Process a single taxi file with proper GRU"""
    print(f"\nProcessing {os.path.basename(file_path)}")
    print("=" * 50)
    
    # Load data
    data = load_taxi_data(file_path)
    if len(data) < 100:
        print("Insufficient data")
        return None
    
    # Create sequences directly
    sequences, targets = create_sequences_directly(data, seq_length=10)
    if len(sequences) < 50:
        print("Not enough sequences")
        return None
    
    # Split data (80/20)
    split_idx = int(len(sequences) * 0.8)
    train_sequences, val_sequences = sequences[:split_idx], sequences[split_idx:]
    train_targets, val_targets = targets[:split_idx], targets[split_idx:]
    
    # Scale data
    scaler_lon = StandardScaler()
    scaler_lat = StandardScaler()
    
    # Fit on training data only
    train_lons = train_sequences[:, 0::2].flatten().reshape(-1, 1)
    train_lats = train_sequences[:, 1::2].flatten().reshape(-1, 1)
    
    scaler_lon.fit(train_lons)
    scaler_lat.fit(train_lats)
    
    # Scale training data
    for i in range(0, train_sequences.shape[1], 2):
        train_sequences[:, i] = scaler_lon.transform(train_sequences[:, i].reshape(-1, 1)).flatten()
        train_sequences[:, i+1] = scaler_lat.transform(train_sequences[:, i+1].reshape(-1, 1)).flatten()
    
    train_targets[:, 0] = scaler_lon.transform(train_targets[:, 0].reshape(-1, 1)).flatten()
    train_targets[:, 1] = scaler_lat.transform(train_targets[:, 1].reshape(-1, 1)).flatten()
    
    # Scale validation data
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
    
    # Initialize GRU model with different architecture
    model = TrajectoryGRU(input_size=2, hidden_size=256, num_layers=3, output_size=2, dropout=0.4)
    
    # Count parameters to verify model is different
    total_params = sum(p.numel() for p in model.parameters())
    print(f"GRU Model Parameters: {total_params}")
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Train model
    best_val_loss = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=100)
    
    # Evaluate
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            all_predictions.extend(output.numpy())
            all_targets.extend(target.numpy())
    
    metrics = calculate_metrics(np.array(all_predictions), np.array(all_targets), scaler_lon, scaler_lat)
    
    print(f"\nGRU Results:")
    print(f"  Best Validation Loss: {best_val_loss:.6f}")
    print(f"  Mean Distance: {metrics['mean_distance']:.2f}m")
    print(f"  Accuracy @ 100m: {metrics['acc_100m']:.2f}%")
    print(f"  Accuracy @ 200m: {metrics['acc_200m']:.2f}%")
    print(f"  Accuracy @ 500m: {metrics['acc_500m']:.2f}%")
    print(f"  Accuracy @ 1000m: {metrics['acc_1000m']:.2f}%")
    print(f"  Accuracy @ 2000m: {metrics['acc_2000m']:.2f}%")
    
    return metrics

def main():
    data_dir = 'taxi_data'
    file_list = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    
    print(f"Found {len(file_list)} taxi files")
    
    results = []
    
    for file_name in file_list[:10]:  # Process first 2 files
        file_path = os.path.join(data_dir, file_name)
        metrics = process_taxi_file(file_path)
        
        if metrics:
            results.append(metrics)
        print("\n" + "="*50)
    
    if results:
        print("\nFINAL AGGREGATED RESULTS (GRU):")
        print("=" * 50)
        print(f"  Mean Distance: {np.mean([r['mean_distance'] for r in results]):.2f}m")
        print(f"  Accuracy @ 100m: {np.mean([r['acc_100m'] for r in results]):.2f}%")
        print(f"  Accuracy @ 200m: {np.mean([r['acc_200m'] for r in results]):.2f}%")
        print(f"  Accuracy @ 500m: {np.mean([r['acc_500m'] for r in results]):.2f}%")
        print(f"  Accuracy @ 1000m: {np.mean([r['acc_1000m'] for r in results]):.2f}%")
        print(f"  Accuracy @ 2000m: {np.mean([r['acc_2000m'] for r in results]):.2f}%")

if __name__ == "__main__":
    main()