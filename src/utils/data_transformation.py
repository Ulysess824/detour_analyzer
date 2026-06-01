from typing import List
import polars as pl


def transform_data(
    data: pl.DataFrame,
    columnas_para_sku: List[str] = ["Grade", "Sbgr", "Gram", "Width"],
    columna_fecha: str = "fecha_fichero",
    columna_consumo: str = "Stock on hand\n(TO)",
    columna_planta: str = "Customer",
    columna_forecast: str = "Total Fcst M\n(TO)",
    umbral_pico: float = 2.0,
) -> pl.DataFrame:
    r"""
    Transforma el DataFrame crudo del sistema fuente al formato analitico completo.

    *   Convierte fecha, construye SKU, calcula consumo diario por diferencia de stock.
    *   Calcula target_is_peak y target_consumo_real para el dia siguiente (t+1).
    *   Agrega consumo_real y daily_forecast listos para build_features.
    *   Codifica la columna planta con one-hot encoding para usarla como feature.
    *
    """

    data_transformed = (
        data.with_columns(
            pl.col(columna_fecha).str.to_date("%d.%m.%Y").alias("fecha"),
            pl.concat_str(
                [pl.col(c).cast(pl.String) for c in columnas_para_sku],
                separator="_",
            ).alias("sku"),
        )
        .sort([columna_planta, "sku", "fecha"])
        .with_columns(
            pl.col(columna_consumo)
            .diff(n=1)
            .over([columna_planta, "sku"])
            .mul(-1)
            .alias("consumo_anterior_to")
        )
        .rename(
            {
                columna_planta: "planta",
                columna_consumo: "stock_planta",
                columna_forecast: "forecast_mensual",
            }
        )
        .filter(pl.col("Strategy") == "VMI")
        .with_columns(
            # Si el consumo_anterior_to es negativo (aumento de stock por abastecimiento),
            # se reemplaza por None para luego imputarlo con forward_fill y backward_fill.
            # Esto evita registrar las entradas de stock como consumo y estabiliza la serie.
            # Aplicamos .abs() al final para eliminar el signo de los ceros negativos (-0.0).
            consumo_real=pl.when(pl.col("consumo_anterior_to") >= 0)
            .then(pl.col("consumo_anterior_to"))
            .otherwise(None)
            .forward_fill()
            .backward_fill()
            .over(["planta", "sku"])
            .fill_null(0.0)
            .abs(),
            daily_forecast=pl.col("forecast_mensual") / 20.0,
        )
        .with_columns(
            target_is_peak=(
                pl.col("consumo_real")
                > (umbral_pico * pl.col("daily_forecast"))
            )
            .shift(-1)
            .over(["planta", "sku"])
            .cast(pl.Int8),
            target_consumo_real=pl.col("consumo_real")
            .shift(-1)
            .over(["planta", "sku"]),
        )
        .filter(pl.col("target_is_peak").is_not_null())
    )

    # One-hot encoding de planta para uso directo como feature en el modelo
    plantas = data_transformed["planta"].unique().sort().to_list()
    planta_exprs = [
        pl.when(pl.col("planta") == p)
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias(f"planta_{p}")
        for p in plantas
    ]
    data_transformed = data_transformed.with_columns(planta_exprs)

    base_cols = [
        "planta", "sku", "fecha",
        "consumo_real", "consumo_anterior_to", "stock_planta",
        "forecast_mensual", "daily_forecast",
        "target_is_peak", "target_consumo_real",
    ]
    planta_cols = [f"planta_{p}" for p in plantas]

    return data_transformed.select(base_cols + planta_cols)
