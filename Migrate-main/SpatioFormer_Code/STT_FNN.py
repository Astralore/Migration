# STT-FNN Trajectory Prediction
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from math import radians, sin, cos, sqrt, atan2
import matplotlib.pyplot as plt
from datetime import datetime
import os
import warnings
import time
import math
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

def create_sequences(data, seq_length=20, pred_length=1):
    """Create sequences for STT-FNNinput"""
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_length - pred_length + 1):
        seq = data[i:i+seq_length]
        target = data[i+seq_length:i+seq_length+pred_length]
        
        seq_features = []
        for point in seq:
            seq_features.extend([point[1], point[0]])  # lon, lat
        
        target_features = []
        for point in target:
            target_features.extend([point[1], point[0]])  # lon, lat
        
        sequences.append(seq_features)
        targets.append(target_features)
    
    return np.array(sequences), np.array(targets)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe_buffer', self.pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x

class TransformerTrajectoryPredictor(nn.Module):
    def __init__(self, input_dim=2, d_model=128, nhead=8, num_layers=4, 
                 dim_feedforward=512, dropout=0.1, seq_length=20):
        super(TransformerTrajectoryPredictor, self).__init__()
        
        self.d_model = d_model
        self.seq_length = seq_length
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, seq_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Predict (lon, lat)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        # src shape: [batch_size, seq_len, input_dim]
        batch_size, seq_len, _ = src.shape
        
        # Input projection
        src_proj = self.input_projection(src) * math.sqrt(self.d_model)
        
        # Add positional encoding
        src_encoded = self.pos_encoder(src_proj)
        src_encoded = self.layer_norm(src_encoded)
        
        # Generate mask if not provided
        if src_mask is None:
            src_mask = self.generate_square_subsequent_mask(seq_len).to(src.device)
        
        # Transformer encoder
        memory = self.transformer_encoder(src_encoded, mask=src_mask)
        
        # Use only the last output for prediction (many-to-one)
        output = memory[:, -1, :]  # Take the last time step
        output = self.output_projection(output)
        
        return output

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class TrajectoryDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

def train_transformer(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=100, patience=10):
    """Train transformer model with early stopping"""
    model.train()
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    print("Training Transformer model...")
    print("Epoch | Train Loss | Val Loss  | LR       | Time (s)")
    print("------|------------|-----------|----------|---------")
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        total_train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Reshape data for transformer: (batch_size, seq_len, 2)
            data_reshaped = data.view(data.size(0), -1, 2)
            
            output = model(data_reshaped)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_train_loss += loss.item()
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data_reshaped = data.view(data.size(0), -1, 2)
                output = model(data_reshaped)
                loss = criterion(output, target)
                total_val_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_transformer_model.pth')
        else:
            patience_counter += 1
        
        epoch_time = time.time() - start_time
        
        if epoch % 5 == 0 or epoch == epochs - 1 or patience_counter >= patience:
            print(f"{epoch:4d}  | {avg_train_loss:8.6f}  | {avg_val_loss:8.6f}  | {current_lr:.2e} | {epoch_time:7.2f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    model.load_state_dict(torch.load('best_transformer_model.pth'))
    return train_losses, val_losses, best_val_loss

def calculate_comprehensive_metrics(predictions, targets, scaler_lon, scaler_lat):
    """Calculate comprehensive metrics for trajectory prediction"""
    
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
            targets_denorm[i][1], targets_denorm[i][0],  # lat, lon
            preds_denorm[i][1], preds_denorm[i][0]       # lat, lon
        )
        distances.append(dist)
    
    distances = np.array(distances)
    
    # Calculate all metrics
    metrics = {
        'mse': np.mean((predictions - targets) ** 2),
        'mae': np.mean(np.abs(predictions - targets)),
        'rmse': np.sqrt(np.mean((predictions - targets) ** 2)),
        'mean_distance': np.mean(distances),
        'median_distance': np.median(distances),
        'std_distance': np.std(distances),
        'acc_100m': np.mean(distances <= 100) * 100,
        'acc_200m': np.mean(distances <= 200) * 100,
        'acc_500m': np.mean(distances <= 500) * 100,
        'acc_1000m': np.mean(distances <= 1000) * 100,
        'acc_2000m': np.mean(distances <= 2000) * 100,
    }
    
    return metrics, distances

def process_taxi_file_with_transformer(file_path):
    """Process a single taxi file with STT-FNN"""
    print(f"\nProcessing {os.path.basename(file_path)} with Transformer")
    print("=" * 60)
    
    # Load data
    data = load_taxi_data(file_path)
    if len(data) < 150:
        print("Insufficient data")
        return None
    
    # Create sequences with longer context for transformer
    sequences, targets = create_sequences(data, seq_length=20, pred_length=1)
    if len(sequences) < 50:
        print("Not enough sequences")
        return None
    
    print(f"Created {len(sequences)} sequences with context length 20")
    
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
    
    # Initialize Transformer model
    model = TransformerTrajectoryPredictor(
        input_dim=2,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        seq_length=20
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Transformer Parameters: {total_params:,}")
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Train model
    train_losses, val_losses, best_val_loss = train_transformer(
        model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=100, patience=15
    )
    
    # Evaluate
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data_reshaped = data.view(data.size(0), -1, 2)
            output = model(data_reshaped)
            all_predictions.extend(output.numpy())
            all_targets.extend(target.numpy())
    
    metrics, distances = calculate_comprehensive_metrics(
        np.array(all_predictions), np.array(all_targets), scaler_lon, scaler_lat
    )
    
    print(f"\nTransformer Results:")
    print("=" * 40)
    print(f"  Best Validation Loss: {best_val_loss:.6f}")
    print(f"  Mean Distance Error: {metrics['mean_distance']:.2f}m")
    print(f"  Median Distance Error: {metrics['median_distance']:.2f}m")
    print(f"  Standard Deviation: {metrics['std_distance']:.2f}m")
    
    print(f"\nAccuracy Metrics:")
    print(f"  Accuracy @ 100m: {metrics['acc_100m']:.2f}%")
    print(f"  Accuracy @ 200m: {metrics['acc_200m']:.2f}%")
    print(f"  Accuracy @ 500m: {metrics['acc_500m']:.2f}%")
    print(f"  Accuracy @ 1000m: {metrics['acc_1000m']:.2f}%")
    print(f"  Accuracy @ 2000m: {metrics['acc_2000m']:.2f}%")
    
    print(f"\nRegression Metrics:")
    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  MAE: {metrics['mae']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    
    return metrics

def main():
    data_dir = 'taxi_data'
    file_list = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    
    print(f"Found {len(file_list)} taxi files")
    print("Using Transformer architecture for trajectory prediction")
    print("=" * 70)
    
    all_metrics = []
    
    for file_name in file_list[:10]:  # Process first 2 files
        file_path = os.path.join(data_dir, file_name)
        metrics = process_taxi_file_with_transformer(file_path)
        
        if metrics:
            all_metrics.append(metrics)
        print("\n" + "="*70)
    
    if all_metrics:
        print("\nFINAL AGGREGATED RESULTS (STT-FNN):")
        print("=" * 70)
        
        print(f"\nDistance Metrics:")
        print(f"  Mean Distance Error: {np.mean([m['mean_distance'] for m in all_metrics]):.2f}m")
        print(f"  Median Distance Error: {np.median([m['median_distance'] for m in all_metrics]):.2f}m")
        print(f"  Std Distance Error: {np.mean([m['std_distance'] for m in all_metrics]):.2f}m")
        
        print(f"\nAccuracy Metrics:")
        print(f"  Accuracy @ 100m: {np.mean([m['acc_100m'] for m in all_metrics]):.2f}%")
        print(f"  Accuracy @ 200m: {np.mean([m['acc_200m'] for m in all_metrics]):.2f}%")
        print(f"  Accuracy @ 500m: {np.mean([m['acc_500m'] for m in all_metrics]):.2f}%")
        print(f"  Accuracy @ 1000m: {np.mean([m['acc_1000m'] for m in all_metrics]):.2f}%")
        print(f"  Accuracy @ 2000m: {np.mean([m['acc_2000m'] for m in all_metrics]):.2f}%")
        
        print(f"\nRegression Metrics:")
        print(f"  MSE: {np.mean([m['mse'] for m in all_metrics]):.6f}")
        print(f"  MAE: {np.mean([m['mae'] for m in all_metrics]):.6f}")
        print(f"  RMSE: {np.mean([m['rmse'] for m in all_metrics]):.6f}")
        
        print(f"\nBased on {len(all_metrics)} successfully processed taxis")

if __name__ == "__main__":
    main()