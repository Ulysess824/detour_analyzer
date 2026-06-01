import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import polars as pl
import plotly.graph_objects as go
from lightgbm import LGBMClassifier, LGBMRegressor
from src.features import build_features

def execute_pipeline_and_visualize():
    # 1. Load data
    data_path = os.path.join("data", "synthetic_consumption.parquet")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Synthetic dataset not found at {data_path}")
        
    df_raw = pl.read_parquet(data_path)
    
    # 2. Add forecast_mensual (average * 20) and daily forecast
    df = df_raw.with_columns(
        forecast_mensual=(pl.col("consumo_promedio_diario").cast(pl.Float64) * 20.0)
    )
    df = df.with_columns(
        daily_forecast=(pl.col("forecast_mensual") / 20.0)
    )
    
    # 3. Define target variables for day t+1
    is_peak = pl.col("consumo_real") > (3.0 * pl.col("daily_forecast"))
    df = df.with_columns(
        target_is_peak=is_peak.shift(-1).over(["planta", "sku"]).cast(pl.Int8),
        target_consumo_real=pl.col("consumo_real").shift(-1).over(["planta", "sku"])
    )
    df = df.filter(pl.col("target_is_peak").is_not_null())
    
    # 4. Chronological train/test split (Jan-Oct train, Nov-Dec test)
    df_train_raw = df.filter(pl.col("fecha") <= pl.date(2025, 10, 31))
    
    # Compute features on training set to get SKU statistics
    _, sku_stats = build_features(df_train_raw)
    
    # Compute features on whole dataset using training SKU statistics
    df_feat, _ = build_features(df, sku_stats=sku_stats)
    
    df_train_feat = df_feat.filter(pl.col("fecha") <= pl.date(2025, 10, 31))
    df_test_feat = df_feat.filter(pl.col("fecha") >= pl.date(2025, 11, 1))
    
    # Define features
    feature_cols = [
        "lag_1", "lag_2", "lag_3", "lag_5", "lag_10",
        "rolling_mean_3", "rolling_std_3", "rolling_max_3",
        "rolling_mean_5", "rolling_std_5", "rolling_max_5",
        "rolling_mean_10", "rolling_std_10", "rolling_max_10",
        "day_of_month", "days_to_end_of_month",
        "daily_forecast",
        "cum_actual_consumption_month_lag1", "cum_forecast_month_lag1", "cum_deviation_month_lag1",
        "sku_historical_peak_rate", "sku_historical_cv"
    ]
    
    # 5. Training
    X_train_c = df_train_feat[feature_cols].to_pandas()
    y_train_c = df_train_feat["target_is_peak"].to_pandas()
    
    df_train_r = df_train_feat.filter(pl.col("target_is_peak") == 1)
    X_train_r = df_train_r[feature_cols].to_pandas()
    y_train_r = df_train_r["target_consumo_real"].to_pandas()
    
    print("Training models...")
    clf = LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
    clf.fit(X_train_c, y_train_c)
    
    reg = LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
    reg.fit(X_train_r, y_train_r)
    
    # 6. Evaluation on Test Set
    X_test = df_test_feat[feature_cols].to_pandas()
    y_test_c = df_test_feat["target_is_peak"].to_pandas()
    y_test_r = df_test_feat["target_consumo_real"].to_pandas()
    
    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    y_pred_class = (y_pred_prob >= 0.5).astype(int)
    y_pred_reg = reg.predict(X_test)
    
    # Combined inference
    y_pred_final = np.where(y_pred_class == 1, y_pred_reg, X_test["daily_forecast"].values)
    
    # 7. Create visualization dataframe for Test set (Nov-Dec 2025)
    # Group and pick a sample plant/SKU
    df_vis = df_test_feat.to_pandas()
    df_vis["y_test_real"] = y_test_r
    df_vis["y_pred_final"] = y_pred_final
    df_vis["predicted_class"] = y_pred_class
    df_vis["actual_class"] = y_test_c
    
    # Filter for Planta_A and SKU_01
    sample_vis = df_vis[(df_vis["planta"] == "Planta_A") & (df_vis["sku"] == "SKU_01")].copy()
    sample_vis = sample_vis.sort_values("fecha")
    
    # Classify prediction cases (TP, FP, FN, TN)
    conditions = [
        (sample_vis["actual_class"] == 1) & (sample_vis["predicted_class"] == 1),
        (sample_vis["actual_class"] == 0) & (sample_vis["predicted_class"] == 1),
        (sample_vis["actual_class"] == 1) & (sample_vis["predicted_class"] == 0)
    ]
    choices = ["True Positive", "False Positive", "False Negative"]
    sample_vis["peak_status"] = np.select(conditions, choices, default="Normal")
    
    # 8. Plotly Figure Creation
    print("Generating Plotly interactive visualization...")
    fig = go.Figure()
    
    # Actual Consumption (Line)
    fig.add_trace(go.Scatter(
        x=sample_vis["fecha"],
        y=sample_vis["y_test_real"],
        mode="lines+markers",
        name="Consumo Real (Actual)",
        line=dict(color="#1f77b4", width=2),
        marker=dict(size=4)
    ))
    
    # Predicted Consumption (Line)
    fig.add_trace(go.Scatter(
        x=sample_vis["fecha"],
        y=sample_vis["y_pred_final"],
        mode="lines+markers",
        name="Consumo Predicho (Pipeline)",
        line=dict(color="#ff7f0e", width=2, dash="dash"),
        marker=dict(size=4)
    ))
    
    # Expected Daily Forecast (Baseline)
    fig.add_trace(go.Scatter(
        x=sample_vis["fecha"],
        y=sample_vis["daily_forecast"],
        mode="lines",
        name="Forecast Diario Esperado",
        line=dict(color="#2ca02c", width=1.5, dash="dot")
    ))
    
    # Peak Threshold (3x expected forecast)
    fig.add_trace(go.Scatter(
        x=sample_vis["fecha"],
        y=3.0 * sample_vis["daily_forecast"],
        mode="lines",
        name="Umbral de Pico (3x Forecast)",
        line=dict(color="#d62728", width=1.5, dash="dashdot")
    ))
    
    # Highlights: True Positives (Green markers)
    tp = sample_vis[sample_vis["peak_status"] == "True Positive"]
    fig.add_trace(go.Scatter(
        x=tp["fecha"],
        y=tp["y_test_real"],
        mode="markers",
        name="Pico Correcto (True Positive)",
        marker=dict(color="green", size=10, symbol="triangle-up", line=dict(width=1.5, color="black"))
    ))
    
    # Highlights: False Positives (Red markers)
    fp = sample_vis[sample_vis["peak_status"] == "False Positive"]
    fig.add_trace(go.Scatter(
        x=fp["fecha"],
        y=fp["y_pred_final"],
        mode="markers",
        name="Falsa Alarma (False Positive)",
        marker=dict(color="red", size=10, symbol="triangle-down", line=dict(width=1.5, color="black"))
    ))
    
    # Highlights: False Negatives (Purple markers)
    fn = sample_vis[sample_vis["peak_status"] == "False Negative"]
    fig.add_trace(go.Scatter(
        x=fn["fecha"],
        y=fn["y_test_real"],
        mode="markers",
        name="Pico No Detectado (False Negative)",
        marker=dict(color="purple", size=10, symbol="x", line=dict(width=1.5, color="black"))
    ))
    
    # Layout configuration
    fig.update_layout(
        title="Prediccion de Picos de Consumo - Planta A / SKU 01 (Nov-Dic 2025)",
        xaxis_title="Fecha",
        yaxis_title="Consumo (TO)",
        legend_title="Leyenda",
        hovermode="x unified",
        template="plotly_white",
        width=1_100,
        height=600
    )
    
    # Save to plots/ directory
    os.makedirs("plots", exist_ok=True)
    html_output_path = os.path.join("plots", "interactive_prediction.html")
    fig.write_html(html_output_path)
    print(f"Interactive plot successfully saved to: {html_output_path}")

if __name__ == "__main__":
    execute_pipeline_and_visualize()
