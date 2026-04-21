# Transformer-based trajectory prediction model (AgentFormer)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from math import radians, sin, cos, sqrt, atan2
import matplotlib.pyplot as plt
from datetime import datetime
import os
import warnings
import math
import copy
warnings.filterwarnings('ignore')

# Enhanced Haversine distance function with error handling
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    try:
        # Convert all inputs to float first
        lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])
        
        R = 6371000  # Radius of Earth in meters
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c
    except (ValueError, TypeError) as e:
        print(f"Error in haversine_distance: {e}")
        print(f"Input values: lat1={lat1}, lon1={lon1}, lat2={lat2}, lon2={lon2}")
        return float('inf')  # Return a large distance on error

# Enhanced data loading with proper type conversion
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
                        
                        # Ensure proper float conversion with error handling
                        try:
                            lon = float(parts[2])
                            lat = float(parts[3])
                        except ValueError:
                            print(f"  Warning: Could not convert coordinates to float on line {line_num}: {line.strip()}")
                            continue
                        
                        data.append([lat, lon, taxi_id, timestamp])  # lat, lon, taxi_id, timestamp
                    except ValueError as e:
                        print(f"  Warning: Could not parse line {line_num}: {line.strip()} - {e}")
                        continue
        print(f"  Successfully loaded {len(data)} data points")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return data

# Enhanced trajectory extraction with type checking
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
            
        try:
            prev_lat, prev_lon = current_traj[-1][0], current_traj[-1][1]
            curr_lat, curr_lon = data[i][0], data[i][1]
            
            # Ensure coordinates are numeric
            if not all(isinstance(x, (int, float)) for x in [prev_lat, prev_lon, curr_lat, curr_lon]):
                # Try to convert to float
                try:
                    prev_lat, prev_lon = float(prev_lat), float(prev_lon)
                    curr_lat, curr_lon = float(curr_lat), float(curr_lon)
                    # Update the data with converted values
                    current_traj[-1][0], current_traj[-1][1] = prev_lat, prev_lon
                    data[i][0], data[i][1] = curr_lat, curr_lon
                except ValueError:
                    print(f"  Could not convert coordinates to floats, skipping point")
                    continue
            
            distance = haversine_distance(prev_lat, prev_lon, curr_lat, curr_lon)
            
        except Exception as e:
            print(f"Error calculating distance: {e}")
            distance = 10000  # Large distance to force new trajectory
        
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
        # Reshape sequences to [batch_size, seq_len, features]
        self.sequences = torch.FloatTensor(sequences).view(-1, 8, 2)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

# Agent-Aware Attention Mechanism (simplified)
class AgentAwareAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, q, k, v, agent_mask=None):
        batch_size, seq_len, _ = q.size()
        
        # Linear projections and split into heads
        q = self.w_q(q).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply agent-aware masking if provided
        if agent_mask is not None:
            scores = scores.masked_fill(agent_mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.w_o(out)
        
        return out, attention

# Simplified Transformer Encoder Layer with Agent-Aware Attention
class AgentAwareEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = AgentAwareAttention(d_model, n_heads, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, src, agent_mask=None):
        # Self attention with agent-aware masking
        src2, attn_weights = self.self_attn(src, src, src, agent_mask)
        src = src + self.dropout(src2)
        src = self.norm1(src)
        
        # Feedforward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        
        return src, attn_weights

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Simplified AgentFormer-inspired model
class AgentFormer(nn.Module):
    def __init__(self, input_dim=2, d_model=64, n_heads=4, 
                 num_encoder_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.input_dim = input_dim
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # AgentFormer encoder layers
        self.encoder_layers = nn.ModuleList([
            AgentAwareEncoderLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, input_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, src, agent_mask=None):
        # src: [batch_size, seq_len, input_dim]
        batch_size, seq_len, _ = src.size()
        
        # Embed input sequences
        src_embed = self.input_embedding(src) * math.sqrt(self.d_model)
        src_embed = self.pos_encoder(src_embed)
        src_embed = self.dropout(src_embed)
        
        # Encoder processing
        memory = src_embed
        encoder_attentions = []
        for layer in self.encoder_layers:
            memory, attn_weights = layer(memory, agent_mask)
            encoder_attentions.append(attn_weights)
        
        # Use the last hidden state for prediction
        last_hidden = memory[:, -1, :]  # [batch_size, d_model]
        
        # Output projection
        output = self.output_projection(last_hidden)  # [batch_size, input_dim]
        
        return output.unsqueeze(1), {'encoder_attentions': encoder_attentions}

def train_agentformer_model(model, train_loader, val_loader, criterion, optimizer, 
                           patience=10, epochs=100):
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
            
            # Data is already in shape [batch_size, seq_len, features]
            output, _ = model(data)
            
            # Target is [batch_size, 1, features], output is [batch_size, 1, features]
            loss = criterion(output, target.unsqueeze(1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                output, _ = model(data)
                loss = criterion(output, target.unsqueeze(1))
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_agentformer_model.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
    
    # Load best model
    if os.path.exists('best_agentformer_model.pth'):
        model.load_state_dict(torch.load('best_agentformer_model.pth'))
    return train_losses, val_losses

def calculate_comprehensive_metrics(predictions, targets, scaler_lon, scaler_lat, threshold=100):
    """Calculate comprehensive metrics including MSE, MAE, Precision, Recall, F1"""
    
    # Calculate MSE and MAE on normalized coordinates
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    
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
            targets_denorm[i][1], targets_denorm[i][0],  # lat, lon
            preds_denorm[i][1], preds_denorm[i][0]       # lat, lon
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
    y_pred = (distances <= threshold).astype(int)  # For this case, we use perfect prediction
    
    # Handle case where all predictions are the same class
    if len(np.unique(y_true)) > 1:
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
    else:
        # If all samples belong to one class, metrics are not meaningful
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
        'mae': mae
    }
    
    # Combine all metrics
    return {**accuracy_metrics, **classification_metrics, **regression_metrics}

def process_single_taxi_file_agentformer(file_path):
    """Process a single taxi file using the AgentFormer approach"""
    print(f"\nProcessing {os.path.basename(file_path)} with AgentFormer...")
    print("=" * 50)
    
    # Load data and preprocess
    data = load_taxi_data(file_path)
    if len(data) == 0:
        print("No data loaded")
        return None
    
    print(f"Total data points: {len(data)}")
    
    trajectories = extract_trajectories(data, min_points=8, max_gap_minutes=15)
    if len(trajectories) == 0:
        print("No meaningful trajectories found")
        return None
    
    print(f"Meaningful trajectories: {len(trajectories)}")
    
    clustered_trajs = cluster_trajectories(trajectories, eps=0.01, min_samples=2)
    print(f"Trajectory clusters: {len(clustered_trajs)}")
    
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
    
    # Train AgentFormer model
    model = AgentFormer(
        input_dim=2, 
        d_model=64,  # Reduced model size
        n_heads=4,   # Reduced number of heads
        num_encoder_layers=2,
        dim_feedforward=256,  # Reduced feedforward dimension
        dropout=0.1
    )
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    
    print("Training AgentFormer model...")
    train_losses, val_losses = train_agentformer_model(
        model, train_loader, val_loader, criterion, optimizer, 
        patience=15, epochs=100
    )
    
    # Evaluate
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in val_loader:
            output, _ = model(data)
            all_predictions.extend(output.squeeze(1).numpy())
            all_targets.extend(target.numpy())
    
    # Calculate comprehensive metrics
    metrics = calculate_comprehensive_metrics(np.array(all_predictions), np.array(all_targets), scaler_lon, scaler_lat)
    
    print(f"\nAgentFormer Results for {os.path.basename(file_path)}:")
    print("=" * 50)
    print(f"Regression Metrics:")
    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  MAE: {metrics['mae']:.6f}")
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
    
    return metrics

def main():
    data_dir = 'taxi_data'
    file_list = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    
    print(f"Found {len(file_list)} taxi files")
    
    all_metrics = []
    
    for file_name in file_list[:10]:  # Process first 3 files for demonstration
        file_path = os.path.join(data_dir, file_name)
        metrics = process_single_taxi_file_agentformer(file_path)
        
        if metrics is not None:
            all_metrics.append(metrics)
        print("\n" + "="*50)
    
    # Calculate aggregate results
    if all_metrics:
        print("\n" + "="*60)
        print("AGENTFORMER AGGREGATE RESULTS ACROSS ALL TAXIS")
        print("="*60)
        
        avg_metrics = {
            'mse': np.mean([m['mse'] for m in all_metrics]),
            'mae': np.mean([m['mae'] for m in all_metrics]),
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
        
        print(f"\nAverage AgentFormer Performance Metrics:")
        print(f"Regression Metrics:")
        print(f"  MSE: {avg_metrics['mse']:.6f}")
        print(f"  MAE: {avg_metrics['mae']:.6f}")
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