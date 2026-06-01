import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import shap
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import (
    roc_auc_score, 
    f1_score, 
    precision_score, 
    recall_score, 
    accuracy_score, 
    mean_absolute_error, 
    mean_squared_error
)
from src.features import build_features

def run_pipeline():
    # 1. Load data
    data_path = os.path.join("data", "synthetic_consumption.parquet")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Synthetic dataset not found at {data_path}")
        
    df_raw = pl.read_parquet(data_path)
    
    # 2. Add forecast_mensual (based on average * 20) and daily forecast
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
    
    # Drop rows where target is NaN (the last day of each plant/sku)
    df = df.filter(pl.col("target_is_peak").is_not_null())
    
    # 4. Split chronologically into Train (Jan-Oct 2025) and Test (Nov-Dec 2025)
    # Train threshold: 2025-10-31
    # Test starts: 2025-11-01
    df_train_raw = df.filter(pl.col("fecha") <= pl.date(2025, 10, 31))
    
    # 5. Compute features avoiding lookahead leakage
    # First, build features on training set to get the SKU statistics
    _, sku_stats = build_features(df_train_raw)
    
    # Now build features on the entire dataset using the training SKU statistics
    df_feat, _ = build_features(df, sku_stats=sku_stats)
    
    # Split the featurized dataset back into train and test subsets
    df_train_feat = df_feat.filter(pl.col("fecha") <= pl.date(2025, 10, 31))
    df_test_feat = df_feat.filter(pl.col("fecha") >= pl.date(2025, 11, 1))
    
    # Define features to use
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
    
    # 6. Prepare training and test sets
    X_train_c = df_train_feat[feature_cols].to_pandas()
    y_train_c = df_train_feat["target_is_peak"].to_pandas()
    
    X_test = df_test_feat[feature_cols].to_pandas()
    y_test_c = df_test_feat["target_is_peak"].to_pandas()
    y_test_r = df_test_feat["target_consumo_real"].to_pandas()
    
    # 7. Train Stage 1: Classifier (Peak detection)
    print("Training Stage 1 Classifier...")
    clf = LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
    clf.fit(X_train_c, y_train_c)
    
    # 8. Train Stage 2: Regressor (Consumption magnitude)
    # Train only on rows where there actually was a peak
    print("Training Stage 2 Regressor...")
    df_train_r = df_train_feat.filter(pl.col("target_is_peak") == 1)
    X_train_r = df_train_r[feature_cols].to_pandas()
    y_train_r = df_train_r["target_consumo_real"].to_pandas()
    
    reg = LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
    reg.fit(X_train_r, y_train_r)
    
    # 9. Evaluate Stage 1 (Classifier) on Test Set
    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    y_pred_class = (y_pred_prob >= 0.5).astype(int)
    
    auc = roc_auc_score(y_test_c, y_pred_prob)
    acc = accuracy_score(y_test_c, y_pred_class)
    prec = precision_score(y_test_c, y_pred_class, zero_division=0)
    rec = recall_score(y_test_c, y_pred_class, zero_division=0)
    f1 = f1_score(y_test_c, y_pred_class, zero_division=0)
    
    print("\n--- STAGE 1 CLASSIFIER EVALUATION (TEST SET) ---")
    print(f"ROC AUC:  {auc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision:{prec:.4f}")
    print(f"Recall:   {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    
    # 10. Evaluate Two-Stage Pipeline (Inference)
    # If peak predicted (class 1) -> use Stage 2 Regressor
    # If no peak predicted (class 0) -> use daily forecast (baseline)
    y_pred_reg = reg.predict(X_test)
    y_pred_final = np.where(y_pred_class == 1, y_pred_reg, X_test["daily_forecast"].values)
    
    mae = mean_absolute_error(y_test_r, y_pred_final)
    rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_final))
    wape = np.sum(np.abs(y_test_r - y_pred_final)) / np.sum(y_test_r)
    
    print("\n--- TWO-STAGE PIPELINE EVALUATION (TEST SET) ---")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"WAPE: {wape:.4f}")
    
    # 11. Interpretability with SHAP
    print("\nGenerating SHAP explanations...")
    os.makedirs("plots", exist_ok=True)
    
    # SHAP for Classifier
    explainer_c = shap.TreeExplainer(clf)
    shap_values_c = explainer_c(X_test)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_c, X_test, show=False)
    plt.title("SHAP Summary Plot - Stage 1 Classifier", fontsize=14)
    plt.tight_layout()
    plt.savefig("plots/shap_classifier_summary.png", dpi=150)
    plt.close()
    
    # SHAP for Regressor
    explainer_r = shap.TreeExplainer(reg)
    shap_values_r = explainer_r(X_train_r)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_r, X_train_r, show=False)
    plt.title("SHAP Summary Plot - Stage 2 Regressor", fontsize=14)
    plt.tight_layout()
    plt.savefig("plots/shap_regressor_summary.png", dpi=150)
    plt.close()
    
    print("SHAP plots saved successfully in plots/ directory.")

if __name__ == "__main__":
    run_pipeline()
