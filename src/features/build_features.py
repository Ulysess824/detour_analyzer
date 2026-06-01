import polars as pl

def build_features(df: pl.DataFrame, sku_stats: pl.DataFrame = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    r"""
    Función que construye variables para el dataset de consumo diario que hace lo siguiente:

    *   Calcula 5 variables de retardo (lags) de consumo.
    *   Calcula 9 variables de ventanas móviles (medias, desviaciones estándar, máximos).
    *   Calcula 21 variables de calendario (día, días restantes de mes, 5 días de la semana y 12 meses binarios).
    *   Calcula 4 variables de desvío acumulado del forecast mensual.
    *   Calcula 2 variables de perfil histórico del SKU (tasa de picos y coeficiente de variación).
    *

    """
    # 1. Ensure DataFrame is sorted chronologically within each group
    # This is critical for shifts and rolling windows
    df = df.sort(["planta", "sku", "fecha"])
    
    # 2. Daily forecast and peak helper
    # Expected daily forecast is monthly forecast divided by 20 working days
    daily_forecast = pl.col("forecast_mensual") / 20.0
    is_peak = pl.col("consumo_real") > (3.0 * daily_forecast)
    
    # 3. Base shifted column to avoid lookahead bias (data leakage)
    df = df.with_columns(
        consumo_real_lag1=pl.col("consumo_real").shift(1).over(["planta", "sku"])
    )
    
    # 4. Generate lags (5 features)
    # Shifts of the consumption: lag 1, 2, 3, 5, and 10 days
    lag_exprs = [
        pl.col("consumo_real").shift(lag).over(["planta", "sku"]).alias(f"lag_{lag}")
        for lag in [1, 2, 3, 5, 10]
    ]
    
    # 5. Generate rolling windows (9 features)
    # Mean, standard deviation, and maximum over windows of 3, 5, and 10 days
    # Computed on consumo_real_lag1 to prevent lookahead leakage
    rolling_exprs = []
    for w in [3, 5, 10]:
        rolling_exprs.extend([
            pl.col("consumo_real_lag1").rolling_mean(window_size=w).over(["planta", "sku"]).alias(f"rolling_mean_{w}"),
            pl.col("consumo_real_lag1").rolling_std(window_size=w).over(["planta", "sku"]).alias(f"rolling_std_{w}"),
            pl.col("consumo_real_lag1").rolling_max(window_size=w).over(["planta", "sku"]).alias(f"rolling_max_{w}"),
        ])
        
    # 6. Calendar features (21 features total: day_of_month, days_to_end_of_month, 7 weekdays, 12 months)
    calendar_exprs = [
        pl.col("fecha").dt.day().alias("day_of_month"),
        (pl.col("fecha").dt.month_end() - pl.col("fecha")).dt.total_days().alias("days_to_end_of_month")
    ]
    # One-hot encoding of working days of the week (1=Monday, 5=Friday)
    calendar_exprs.extend([
        pl.when(pl.col("fecha").dt.weekday() == d).then(1).otherwise(0).alias(f"day_of_week_{d}")
        for d in range(1, 6)
    ])
    # One-hot encoding of months (1=January, 12=December)
    calendar_exprs.extend([
        pl.when(pl.col("fecha").dt.month() == m).then(1).otherwise(0).alias(f"month_{m}")
        for m in range(1, 13)
    ])
    
    # 7. Deviation features from cumulative monthly forecast (4 features)
    # We calculate daily_forecast first as a temporary column to compute cum sums
    df = df.with_columns(
        daily_forecast=daily_forecast,
        year_month=(pl.col("fecha").dt.year() * 100 + pl.col("fecha").dt.month())
    )
    
    # Cumulative monthly sums up to today (including today)
    cum_exprs = [
        pl.col("consumo_real").cum_sum().over(["planta", "sku", "year_month"]).alias("_cum_actual_month"),
        pl.col("daily_forecast").cum_sum().over(["planta", "sku", "year_month"]).alias("_cum_forecast_month"),
    ]
    df = df.with_columns(cum_exprs)
    
    # Shift cumulative values by 1 to get yesterday's totals (preventing leakage for prediction of today)
    # and compute deviation: cum_actual - cum_forecast
    df = df.with_columns(
        cum_actual_consumption_month_lag1=(pl.col("_cum_actual_month") - pl.col("consumo_real")).alias("cum_actual_consumption_month_lag1"),
        cum_forecast_month_lag1=(pl.col("_cum_forecast_month") - pl.col("daily_forecast")).alias("cum_forecast_month_lag1")
    )
    
    df = df.with_columns(
        cum_deviation_month_lag1=(pl.col("cum_actual_consumption_month_lag1") - pl.col("cum_forecast_month_lag1")).alias("cum_deviation_month_lag1")
    )
    
    # Clean up temporary cumulative columns
    df = df.drop(["_cum_actual_month", "_cum_forecast_month"])
    
    # 8. Compute or apply historical SKU profile (2 features)
    # - Historical peak rate (peaks / total records)
    # - Coefficient of variation (std / mean)
    if sku_stats is None:
        # Create peak boolean column for calculation
        df_with_peaks = df.with_columns(is_peak=is_peak)
        
        # Calculate historical stats grouped by SKU
        sku_stats = (
            df_with_peaks.group_by("sku")
            .agg([
                pl.col("is_peak").mean().alias("sku_historical_peak_rate"),
                (pl.col("consumo_real").std() / pl.col("consumo_real").mean())
                .fill_nan(0.0)
                .fill_null(0.0)
                .alias("sku_historical_cv")
            ])
        )
    
    # Apply all computed expressions to df
    df = df.with_columns(lag_exprs + rolling_exprs + calendar_exprs)
    
    # Join the SKU historical stats
    df = df.join(sku_stats, on="sku", how="left")
    
    # Clean up temporary helper columns
    df = df.drop(["consumo_real_lag1", "year_month"])
    
    return df, sku_stats
