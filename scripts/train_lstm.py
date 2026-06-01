import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, 
    f1_score, 
    precision_score, 
    recall_score, 
    accuracy_score
)
from src.features import build_features
from src.models import HybridLSTM, ConsumptionDataset

def prepare_data():
    # 1. Load data
    data_path = os.path.join("data", "synthetic_consumption.parquet")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Synthetic dataset not found at {data_path}")
        
    df_raw = pl.read_parquet(data_path)
    
    # 2. Add forecast_mensual and daily forecast
    df = df_raw.with_columns(
        forecast_mensual=(pl.col("consumo_promedio_diario").cast(pl.Float64) * 20.0)
    )
    df = df.with_columns(
        daily_forecast=(pl.col("forecast_mensual") / 20.0)
    )
    
    # 3. Define target variables for day t+1
    is_peak = pl.col("consumo_real") > (3.0 * pl.col("daily_forecast"))
    df = df.with_columns(
        target_is_peak=is_peak.shift(-1).over(["planta", "sku"]).cast(pl.Int8)
    )
    
    # Drop rows where target is NaN (the last day of each plant/sku)
    df = df.filter(pl.col("target_is_peak").is_not_null())
    
    # 4. Split raw data to compute training SKU statistics
    df_train_raw = df.filter(pl.col("fecha") <= pl.date(2025, 10, 31))
    _, sku_stats = build_features(df_train_raw)
    
    # Compute features for the entire dataset using training SKU statistics
    df_feat, _ = build_features(df, sku_stats=sku_stats)
    
    # Convert to pandas for grouping and sequence creation
    df_pd = df_feat.to_pandas()
    
    # One-hot encode plant deterministically
    for p in ["Planta_A", "Planta_B", "Planta_C"]:
        df_pd[f"planta_{p}"] = (df_pd["planta"] == p).astype(float)
        
    # Define the 8 static/context columns
    context_cols = [
        "sku_historical_peak_rate", 
        "sku_historical_cv", 
        "daily_forecast", 
        "planta_Planta_A", 
        "planta_Planta_B", 
        "planta_Planta_C", 
        "day_of_month", 
        "days_to_end_of_month"
    ]
    
    # 5. Extract sequences (15 days), contexts (8 features), and targets
    sequences = []
    contexts = []
    targets = []
    dates = []
    
    # Group by plant and SKU to ensure chronological time-series per series
    grouped = df_pd.groupby(["planta", "sku"])
    for _, group in grouped:
        consumption_series = group["consumo_real"].values
        context_matrix = group[context_cols].values
        target_series = group["target_is_peak"].values
        date_series = pd.to_datetime(group["fecha"]).values
        
        # Slide over the series with 15 days window
        for idx in range(14, len(group)):
            seq = consumption_series[idx - 14 : idx + 1]
            context = context_matrix[idx]
            target = target_series[idx]
            dt = date_series[idx]
            
            sequences.append(seq)
            contexts.append(context)
            targets.append(target)
            dates.append(dt)
            
    seqs_arr = np.array(sequences)
    ctxs_arr = np.array(contexts)
    targets_arr = np.array(targets)
    dates_arr = np.array(dates)
    
    # 6. Chronological split of the generated windows
    train_mask = dates_arr <= pd.Timestamp("2025-10-31")
    test_mask = dates_arr >= pd.Timestamp("2025-11-01")
    
    X_train_seq = seqs_arr[train_mask]
    X_train_ctx = ctxs_arr[train_mask]
    y_train = targets_arr[train_mask]
    
    X_test_seq = seqs_arr[test_mask]
    X_test_ctx = ctxs_arr[test_mask]
    y_test = targets_arr[test_mask]
    
    return X_train_seq, X_train_ctx, y_train, X_test_seq, X_test_ctx, y_test

def train_lstm_model():
    # 1. Prepare data
    print("Preparing sequences and contexts for LSTM training...")
    X_train_seq, X_train_ctx, y_train, X_test_seq, X_test_ctx, y_test = prepare_data()
    
    # 2. Create datasets and dataloaders
    train_dataset = ConsumptionDataset(X_train_seq, X_train_ctx, y_train)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    # 3. Instantiate model, loss, and optimizer
    model = HybridLSTM(sequence_length=15, context_size=8, hidden_size=32, lstm_layers=1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 4. Training loop
    epochs = 15
    print(f"Training Hybrid LSTM model for {epochs} epochs...")
    model.train()
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for seqs, ctxs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(seqs, ctxs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(targets)
        epoch_loss /= len(train_dataset)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.4f}")
            
    # 5. Evaluate on Test Set
    model.eval()
    with torch.no_grad():
        test_seq_tensor = torch.tensor(X_test_seq, dtype=torch.float32).unsqueeze(-1)
        test_ctx_tensor = torch.tensor(X_test_ctx, dtype=torch.float32)
        
        logits = model(test_seq_tensor, test_ctx_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        preds = (probs >= 0.5).astype(int)
        
    # Compute metrics
    auc = roc_auc_score(y_test, probs)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    
    print("\n--- HYBRID LSTM EVALUATION (TEST SET) ---")
    print(f"ROC AUC:  {auc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision:{prec:.4f}")
    print(f"Recall:   {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    
    # Save the trained model weights
    os.makedirs("plots", exist_ok=True)
    model_save_path = os.path.join("plots", "lstm_model.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to: {model_save_path}")

if __name__ == "__main__":
    train_lstm_model()
