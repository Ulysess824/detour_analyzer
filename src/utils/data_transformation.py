from typing import List
import polars as pl

def transform_data(
    data: pl.DataFrame,
    columnas_para_sku: List[str] = ["Grade", "Sbgr", "Gram", "Width"],
    columna_fecha: str = "fecha_fichero",
    columna_consumo: str = "Hist. Month\nto date\n(TO)",
    columna_planta: str = "Customer",
    columna_forecast: str = "Total Fcst M\n(TO)",
    columna_stock_actual: str = "Stock on hand\n(TO)",
    umbral_pico: float = 2.0,
) -> pl.DataFrame:
    r"""
    Transforma el DataFrame crudo del sistema fuente al formato analitico completo.

    *   Convierte fecha, construye SKU y calcula el consumo diario a partir de la columna
        Month-to-date (MTD), controlando el reinicio del acumulado al cambiar de mes.
    *   Desplaza la fecha un dia (el valor MTD del dia D refleja el consumo del dia D-1).
    *   Imputa el primer dia laboral del mes con la media movil de los ultimos consumos
        positivos y reemplaza los consumos negativos por el valor anterior.
    *   Calcula daily_forecast, target_is_peak y target_consumo_real para el dia siguiente (t+1).
    """

    # 1. Parsear fecha, construir SKU y ordenar cronologicamente
    df_t = data.with_columns(
        pl.col(columna_fecha).str.to_date("%d.%m.%Y").alias("fecha"),
        pl.concat_str(
            [pl.col(c).cast(pl.String) for c in columnas_para_sku],
            separator="_",
        ).alias("sku"),
    ).sort([columna_planta, "sku", "fecha"])

    if columna_stock_actual in data.columns:
        df_t = df_t.rename({columna_stock_actual: "stock_actual"})
    else:
        df_t = df_t.with_columns(pl.lit(None).alias("stock_actual"))


    # 2. Controlar el reinicio de mes de la columna MTD antes de aplicar diff
    df_t = df_t.with_columns(
        es_inicio_mes=(
            pl.col("fecha").dt.month()
            != pl.col("fecha").shift(1).over([columna_planta, "sku"]).dt.month()
        ).fill_null(True)
    ).with_columns(
        consumo_real_to=pl.when(pl.col("es_inicio_mes"))
        .then(pl.col(columna_consumo))
        .otherwise(pl.col(columna_consumo).diff(1).over([columna_planta, "sku"]))
    )

    # 3. Desplazar la fecha (el consumo es del dia anterior)
    df_t = df_t.with_columns(
        pl.col("fecha").shift(1).over([columna_planta, "sku"]).alias("fecha")
    ).filter(pl.col("fecha").is_not_null())

    # 4. Calcular si es el primer dia laboral del mes sobre la fecha ajustada
    df_t = df_t.with_columns(
        es_primer_dia_laboral=pl.col("fecha")
        == pl.col("fecha")
        .filter(pl.col("fecha").dt.weekday() <= 5)
        .min()
        .over(pl.col("fecha").dt.truncate("1mo"))
    )

    # 5. Aplicar rolling_mean en el primer dia laboral ignorando ceros y negativos
    df_t = df_t.with_columns(
        consumo_real_to=pl.when(pl.col("es_primer_dia_laboral"))
        .then(
            pl.when(pl.col("consumo_real_to") > 0)
            .then(pl.col("consumo_real_to"))
            .otherwise(None)
            .shift(1)
            .rolling_mean(window_size=5, min_periods=1)
            .over([columna_planta, "sku"])
            .fill_null(0.0)
        )
        .otherwise(pl.col("consumo_real_to"))
    )

    # 6. Cuando el consumo_real_to es negativo se toma el consumo anterior
    df_t = df_t.with_columns(
        consumo_real_to=pl.when(pl.col("consumo_real_to") < 0)
        .then(pl.col("consumo_real_to").shift(1).over([columna_planta, "sku"]))
        .otherwise(pl.col("consumo_real_to"))
    )

    # 7. Renombrar planta y forecast, derivar consumo_real limpio y daily_forecast
    df_t = (
        df_t.rename(
            {
                columna_planta: "planta",
                columna_forecast: "forecast_mensual",
            }
        )
        .with_columns(
            consumo_real=pl.col("consumo_real_to").fill_null(0.0).abs(),
            daily_forecast=pl.col("forecast_mensual") / 20.0,
        )
        # 8. Targets para el dia siguiente (t+1)
        .with_columns(
            target_is_peak=(
                pl.col("consumo_real") > (umbral_pico * pl.col("daily_forecast"))
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

    # 9. Seleccion final de columnas y renombrado de columnas originales extra
    base_cols = [
        "planta", "sku", "fecha", "stock_actual",
        "consumo_real", "consumo_real_to",
        "forecast_mensual", "daily_forecast",
        "target_is_peak", "target_consumo_real",
    ]

    # Identificar columnas de la data original que no están mapeadas en la salida
    extra_original_cols = [
        col for col in data.columns
        if col not in [columna_planta, columna_stock_actual, columna_forecast, columna_fecha]
        and col not in base_cols
    ]

    # Mapeo propuesto de nombres limpios
    column_mapping = {
        "Grade": "grade",
        "Sbgr": "subgroup",
        "Gram": "grammage",
        "Width": "width",
        "Product": "product_code",
        "Strategy": "strategy",
        "Safety DS\n(Days)": "safety_days",
        "Safety Stock\n(TO)": "safety_stock_to",
        "Hist. Month\nto date\n(TO)": "hist_month_to_date_to",
        "Fcst Month\nto date\n(TO": "fcst_month_to_date_to",
        "Dev\nHist-Fcst\nM to date\n(TO)": "dev_hist_fcst_month_to_date_to",
        "Deviation\nHist-Fcst\nM to date\n(%)": "dev_hist_fcst_month_to_date_pct",
        "Stock\nvs SStk\n(TO)": "stock_vs_safety_stock_to",
        "Stock\nvs SStk\n(%)": "stock_vs_safety_stock_pct",
        "STOCK EN PITEA": "stock_en_pitea",
        "PEDIDO NOV": "pedido_nov",
    }

    # Función interna para limpiar nombres de columnas inesperadas dinámicamente
    def _clean_name(name: str) -> str:
        import re
        cleaned = re.sub(r"[\n()\[\]%\\/]", " ", name)
        cleaned = re.sub(r"[\s-]+", "_", cleaned).strip("_").lower()
        return cleaned

    # Construir diccionario de renombres
    rename_dict = {}
    for col in extra_original_cols:
        if col in column_mapping:
            rename_dict[col] = column_mapping[col]
        else:
            rename_dict[col] = _clean_name(col)

    df_t = df_t.rename(rename_dict)

    # Añadir las nuevas columnas renombradas a la selección final
    final_cols = base_cols + [rename_dict[col] for col in extra_original_cols]

    return df_t.select(final_cols)

