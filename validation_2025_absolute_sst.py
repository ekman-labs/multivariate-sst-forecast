"""
validation_2025_absolute_sst.py - Absolute SST Forecasting (Non-Anomaly)

TARGET:
Predicting ACTUAL Sea Surface Temperature (e.g., 29°C, 30°C), 
NOT the anomaly.

Changes from previous version:
1. Input/Target column changed from 'sst_anomaly' to 'sst_actual'.
2. Removed climatology subtraction in test data loading.
3. Visualization labels updated.

Author: Feby - Data Science Portfolio
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import warnings
import os

warnings.filterwarnings('ignore')

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# ============================================================================
# CONFIGURATION
# ============================================================================
SST_INDO_FILE = "data/processed/sst_indo_clean.csv"
NINO34_FILE = "data/raw/nina34.anom.data.txt"
SST_2025_NC = "data_sst/sst.day.mean.2025.nc"

os.makedirs("output/models", exist_ok=True)
os.makedirs("output/figures", exist_ok=True)
CHECKPOINT_PATH = "output/models/best_model_2025_absolute.pt"

# Region bounds
LAT_MIN, LAT_MAX = -11, 6
LON_MIN, LON_MAX = 95, 141

# Model parameters
LOOKBACK = 48
INPUT_SIZE = 4        # [SST_Actual, Nino, Sin, Cos]
HIDDEN_SIZE = 128 
NUM_LAYERS = 2
OUTPUT_SIZE = 2       # [SST_Actual, Nino]
DROPOUT = 0.1

# Training parameters
EPOCHS = 150
BATCH_SIZE = 16
LEARNING_RATE = 0.001
PATIENCE = 25

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# HELPER CLASSES & FUNCTIONS
# ============================================================================

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'   EarlyStopping: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'   Val loss improved ({self.val_loss_min:.6f} -> {val_loss:.6f}). Saving...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def add_time_features(df):
    """Menambahkan fitur waktu siklis."""
    df = df.copy()
    df['month'] = df.index.month
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # PERUBAHAN 1: Select 'sst_actual' instead of 'sst_anomaly'
    return df[['sst_actual', 'nino34', 'sin_month', 'cos_month']]


# ============================================================================
# DATA LOADING
# ============================================================================

def load_training_data(sst_file, nino_file):
    """Load training data (Absolute SST)."""
    sst_df = pd.read_csv(sst_file)
    sst_df['date'] = pd.to_datetime(sst_df['date'])
    sst_df = sst_df.set_index('date')
    
    records = []
    with open(nino_file, 'r') as f:
        lines = f.readlines()
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 13: continue
        try:
            year = int(parts[0])
            if year < 1900 or year > 2100: continue
        except ValueError: continue
        for month_idx, val in enumerate(parts[1:13]):
            try:
                v = float(val)
                if v < -90: continue
                records.append({'date': pd.Timestamp(year=year, month=month_idx+1, day=1), 'nino34': v})
            except ValueError: continue
    nino_df = pd.DataFrame(records).set_index('date').sort_index()
    
    # PERUBAHAN 2: Join dan ambil 'sst_actual'
    merged = sst_df.join(nino_df, how='inner')[['sst_actual', 'nino34']].dropna()
    merged_final = add_time_features(merged)
    
    print(f"✓ Training data: {merged.index[0]:%Y-%m} to {merged.index[-1]:%Y-%m} ({len(merged)} records)")
    return merged_final


def load_2025_sst_only(nc_file):
    """Load 2025 SST Ground Truth (Absolute)."""
    ds = xr.open_dataset(nc_file)
    ds_indo = ds.sel(lat=slice(LAT_MIN, LAT_MAX), lon=slice(LON_MIN, LON_MAX))
    ds_monthly = ds_indo.resample(time='MS').mean(dim='time')
    sst_mean = ds_monthly['sst'].mean(dim=['lat', 'lon'])
    
    df_2025 = pd.DataFrame({
        'date': pd.to_datetime(sst_mean['time'].values),
        'sst_actual': sst_mean.values
    }).set_index('date')
    ds.close()
    
    # PERUBAHAN 3: Tidak ada pengurangan climatology. Langsung return actual.
    print(f"✓ Test data (2025): {df_2025.index[0]:%Y-%m} to {df_2025.index[-1]:%Y-%m} ({len(df_2025)} records)")
    print(f"  Range Suhu: {df_2025['sst_actual'].min():.2f}°C - {df_2025['sst_actual'].max():.2f}°C")
    
    return df_2025[['sst_actual']]


# ============================================================================
# MODEL (SAMA SEPERTI SEBELUMNYA)
# ============================================================================

class LSTMForecaster(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, output_size=2, dropout=0.1):
        super(LSTMForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size,
                            num_layers=num_layers, 
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
                            
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


def create_sequences(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, :])
        y.append(data[i, :2]) 
    return np.array(X), np.array(y)


def train_model(model, train_loader, val_loader, epochs, lr):
    criterion = nn.HuberLoss(delta=1.0) 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True, path=CHECKPOINT_PATH)
    
    train_losses, val_losses = [], []
    
    print("\n" + "=" * 50)
    print(f"TRAINING (Absolute SST Mode)")
    print("=" * 50)
    
    for epoch in range(epochs):
        model.train()
        batch_loss = []
        for X_b, y_b in train_loader:
            pred = model(X_b)
            loss = criterion(pred, y_b)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_loss.append(loss.item())
        train_losses.append(np.mean(batch_loss))
        
        model.eval()
        batch_val_loss = []
        with torch.no_grad():
            for X_v, y_v in val_loader:
                pred = model(X_v)
                batch_val_loss.append(criterion(pred, y_v).item())
        val_losses.append(np.mean(batch_val_loss))
        
        scheduler.step(val_losses[-1])
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{epochs}] Train: {train_losses[-1]:.6f} | Val: {val_losses[-1]:.6f}")
        
        early_stopping(val_losses[-1], model)
        if early_stopping.early_stop:
            print(f"\n⚡ Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    print("✓ Loaded best model")
    
    return train_losses, val_losses


# ============================================================================
# RECURSIVE FORECASTING
# ============================================================================

def recursive_forecast(model, initial_sequence, scaler, n_months=12):
    model.eval()
    current_seq = initial_sequence.copy()
    next_month_idx = 1 
    
    predictions_scaled = []
    
    print("\n  Recursive Forecasting (Absolute SST):")
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    for month in range(n_months):
        X = torch.FloatTensor(current_seq).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            pred_core = model(X).cpu().numpy().flatten()
        
        predictions_scaled.append(pred_core)
        
        sin_feat = np.sin(2 * np.pi * next_month_idx / 12)
        cos_feat = np.cos(2 * np.pi * next_month_idx / 12)
        
        new_row = np.array([pred_core[0], pred_core[1], sin_feat, cos_feat])
        current_seq = np.vstack([current_seq[1:], new_row])
        
        if month < len(months):
            # Tampilkan nilai suhu asli (bukan anomali)
            # Nilai ini masih dalam bentuk scaled (-1 s/d 1), akan di-inverse di bawah
            pass 
        
        next_month_idx += 1
    
    predictions_scaled = np.array(predictions_scaled)
    predictions_full = scaler.inverse_transform(predictions_scaled)
    
    return predictions_full[:, 0], predictions_full[:, 1]


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(actual, predicted, pred_nino, dates, rmse, train_losses, val_losses):
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), dpi=100)
    
    months = [d.strftime('%b %Y') for d in dates]
    x = range(len(dates))
    
    # Plot 1: SST Comparison (Absolute)
    ax1 = axes[0]
    ax1.plot(x, actual, 'b-o', linewidth=2.5, markersize=8, label='Actual 2025')
    ax1.plot(x, predicted, 'r--s', linewidth=2.5, markersize=8, label='Forecast (Recursive)')
    # PERUBAHAN 4: Label sumbu Y
    ax1.set_ylabel('Temperature (°C)', fontsize=12) 
    ax1.set_title(f'Absolute SST Forecast 2025 (Indonesia)\nRMSE = {rmse:.4f}°C', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(months, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Nino Forecast
    ax2 = axes[1]
    ax2.plot(x, pred_nino, 'purple', linestyle='--', marker='^', label='Predicted Niño 3.4')
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax2.fill_between(x, 0.5, 3.0, color='red', alpha=0.1, label='El Niño Zone')
    ax2.fill_between(x, -0.5, -3.0, color='blue', alpha=0.1, label='La Niña Zone')
    ax2.set_xticks(x)
    ax2.set_xticklabels(months, rotation=45, ha='right')
    ax2.set_ylabel('Niño 3.4 Index')
    ax2.set_title('Internal Niño 3.4 Forecast', fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Loss
    ax3 = axes[2]
    ax3.plot(train_losses, 'g-', label='Training Loss')
    ax3.plot(val_losses, 'orange', linestyle='--', label='Validation Loss')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Huber Loss')
    ax3.set_title('Training Progress', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/figures/validation_2025_absolute.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: output/figures/validation_2025_absolute.png")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("ABSOLUTE SST FORECASTING: Year 2025")
    print("Target: Actual Sea Surface Temperature (Not Anomaly)")
    print("=" * 70)
    
    # 1. Load Data
    print("\n[Step 1/6] Loading training data (Absolute SST)...")
    train_df = load_training_data(SST_INDO_FILE, NINO34_FILE)
    
    print("\n[Step 2/6] Loading 2025 SST data (Absolute Ground Truth)...")
    # PERUBAHAN 5: Panggil fungsi load baru tanpa parameter train_df
    test_2025 = load_2025_sst_only(SST_2025_NC)
    
    # 2. Normalize
    print("\n[Step 3/6] Normalizing...")
    # Karena suhu asli berkisar 26-31 derajat, scaling sangat penting
    scaler = MinMaxScaler(feature_range=(0, 1)) # Range 0-1 lebih aman untuk nilai absolut positif
    
    data_values = train_df.values
    data_core = data_values[:, :2]  # SST_Actual, Nino
    data_time = data_values[:, 2:]  # Sin, Cos
    
    data_core_scaled = scaler.fit_transform(data_core)
    train_scaled = np.hstack([data_core_scaled, data_time])
    
    # 3. Create sequences
    print("\n[Step 4/6] Creating sequences...")
    X_full, y_full = create_sequences(train_scaled, LOOKBACK)
    
    train_size = int(len(X_full) * 0.9)
    X_train, y_train = X_full[:train_size], y_full[:train_size]
    X_val, y_val = X_full[train_size:], y_full[train_size:]
    
    print(f"  Training samples: {len(X_train)} | Validation: {len(X_val)}")
    
    X_train_t = torch.FloatTensor(X_train).to(DEVICE)
    y_train_t = torch.FloatTensor(y_train).to(DEVICE)
    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    y_val_t = torch.FloatTensor(y_val).to(DEVICE)
    
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False)
    
    # 4. Train
    print("\n[Step 5/6] Training...")
    model = LSTMForecaster(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT).to(DEVICE)
    train_losses, val_losses = train_model(model, train_loader, val_loader, EPOCHS, LEARNING_RATE)
    
    # 5. Forecast
    print("\n[Step 6/6] Forecasting 2025...")
    initial_seq = train_scaled[-LOOKBACK:] 
    n_months_to_predict = len(test_2025)
    
    pred_sst, pred_nino = recursive_forecast(model, initial_seq, scaler, n_months=n_months_to_predict)
    
    # Metrics
    actual_sst = test_2025['sst_actual'].values[:n_months_to_predict]
    pred_sst = pred_sst[:len(actual_sst)]
    pred_nino = pred_nino[:len(actual_sst)]
    
    rmse = np.sqrt(np.mean((pred_sst - actual_sst) ** 2))
    mae = np.mean(np.abs(pred_sst - actual_sst))
    corr = np.corrcoef(actual_sst, pred_sst)[0, 1] if len(actual_sst) > 1 else 0
    
    print("\n" + "=" * 50)
    print("2025 ABSOLUTE FORECASTING METRICS")
    print("=" * 50)
    print(f"RMSE:        {rmse:.4f} °C")
    print(f"MAE:         {mae:.4f} °C")
    print(f"Correlation: {corr:.4f}")
    
    # Monthly breakdown
    print("\n Monthly Results:")
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for i in range(len(pred_sst)):
        err = pred_sst[i] - actual_sst[i]
        print(f"  {months[i]}: Pred={pred_sst[i]:.2f}°C | Actual={actual_sst[i]:.2f}°C | Diff={err:+.2f}°C")
    
    plot_results(actual_sst, pred_sst, pred_nino, test_2025.index[:len(pred_sst)], rmse, train_losses, val_losses)
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    
    return model, scaler

if __name__ == "__main__":
    model, scaler = main()