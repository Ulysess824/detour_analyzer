import sys
import os

# 1. Agregar la raiz del proyecto al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from src.utils.data_transformation import transform_data
from src.features import build_features

def run_regression_pipeline():
    # 2. Ruta del dataset real e inicializacion
    data_path = os.path.join("data", "historico_consumo.parquet")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No se encontro el dataset historico en {data_path}")
        
    print("Cargando y transformando datos...")
    df_raw = pl.read_parquet(data_path)
    
    # 3. Transformar los datos reales y renombrar la columna para build_features
    umbral_factor = 2.0
    df_transformed = transform_data(df_raw, umbral_pico=umbral_factor)
    df = df_transformed.rename({"consumo_anterior_to": "consumo_real"})
    
    # Calcular la variable continua de consumo para mañana (target)
    df = df.with_columns(
        target_consumo_real=pl.col("consumo_real").shift(-1).over(["planta", "sku"])
    )
    df = df.filter(pl.col("target_consumo_real").is_not_null())
    
    # 4. Dividir cronologicamente
    fecha_corte = pl.date(2025, 10, 31)
    df_train_raw = df.filter(pl.col("fecha") <= fecha_corte)
    
    # Construir variables con estadisticas de train
    print("Construyendo variables...")
    _, sku_stats = build_features(df_train_raw)
    df_feat, _ = build_features(df, sku_stats=sku_stats)
    
    # Dividir de nuevo las variables calculadas
    df_train_feat = df_feat.filter(pl.col("fecha") <= fecha_corte)
    df_test_feat = df_feat.filter(pl.col("fecha") > fecha_corte)
    
    # Columnas de entrada para el modelo
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
    
    # Agregar las variables de un dia de la semana y mes binarios de manera dinamica
    features_calendario = [c for c in df_feat.columns if c.startswith("day_of_week_") or c.startswith("month_")]
    feature_cols.extend(features_calendario)
    
    # 5. Preparar matrices para el regresor
    X_train = df_train_feat[feature_cols].to_pandas()
    y_train = df_train_feat["target_consumo_real"].to_pandas()
    
    X_test = df_test_feat[feature_cols].to_pandas()
    y_test_real = df_test_feat["target_consumo_real"].to_pandas()
    y_test_class = df_test_feat["target_is_peak"].to_pandas()
    
    # 6. Entrenar regresor de LightGBM
    print("Entrenando regresor LGBMRegressor...")
    reg = LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
    reg.fit(X_train, y_train)
    
    # 7. Inferencia del consumo
    prediccion_consumo = reg.predict(X_test)
    
    # Clasificar como pico si la prediccion de consumo supera el umbral (umbral * daily_forecast)
    daily_forecast_vals = X_test["daily_forecast"].values
    umbral_limite = umbral_factor * daily_forecast_vals
    
    prediccion_pico = (prediccion_consumo > umbral_limite).astype(int)
    
    # Calcular metricas de regresion
    mae = mean_absolute_error(y_test_real, prediccion_consumo)
    rmse = np.sqrt(mean_squared_error(y_test_real, prediccion_consumo))
    
    # Calcular metricas de clasificacion derivadas del regresor
    precision = precision_score(y_test_class, prediccion_pico, zero_division=0)
    recall = recall_score(y_test_class, prediccion_pico, zero_division=0)
    f1 = f1_score(y_test_class, prediccion_pico, zero_division=0)
    
    print("\n=== Evaluacion del Regresor (Consumo) ===")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    print("\n=== Metricas de Clasificacion Derivadas ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # 8. Graficar el resultado para una serie especifica del test set
    df_test_vis = df_test_feat.to_pandas()
    df_test_vis["pred_consumo"] = prediccion_consumo
    df_test_vis["umbral_pico"] = umbral_limite
    
    # Seleccionar una combinacion planta/sku con suficientes datos
    df_test_vis["planta_sku"] = df_test_vis["planta"] + "_" + df_test_vis["sku"]
    series_counts = df_test_vis["planta_sku"].value_counts()
    best_series = series_counts.index[0]
    
    print(f"\nGraficando prediccion de la serie mas representativa: {best_series}")
    df_vis = df_test_vis[df_test_vis["planta_sku"] == best_series].sort_values("fecha")
    
    # Crear la figura
    plt.figure(figsize=(12, 6))
    plt.plot(df_vis["fecha"], df_vis["target_consumo_real"], label="Consumo Real", color="#1f77b4", linewidth=2)
    plt.plot(df_vis["fecha"], df_vis["pred_consumo"], label="Consumo Predicho", color="#ff7f0e", linestyle="--", linewidth=2)
    plt.plot(df_vis["fecha"], df_vis["umbral_pico"], label="Umbral de Pico (2x Forecast)", color="#d62728", linestyle=":", linewidth=1.5)
    
    plt.title(f"Prediccion de Consumo Diario y Umbral de Pico - Serie: {best_series}", fontsize=14)
    plt.xlabel("Fecha", fontsize=12)
    plt.ylabel("Consumo (TO)", fontsize=12)
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    # Guardar grafico
    os.makedirs("plots", exist_ok=True)
    plot_path = os.path.join("plots", "regression_test_plot.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Grafico guardado con éxito en: {plot_path}")

if __name__ == "__main__":
    run_regression_pipeline()
