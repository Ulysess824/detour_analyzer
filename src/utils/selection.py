import random
from typing import Optional
import polars as pl

def random_sku(
    df: pl.DataFrame,
    seed: int | None = None,
    sku_col: str = "sku",
    fecha_col: str = "fecha",
) -> pl.DataFrame:
    r"""
    Función que selecciona un SKU al azar del año más reciente que hace lo siguiente:

    *   Fija la semilla del generador aleatorio de Python si se especifica.
    *   Determina el año más reciente presente en la columna de fecha.
    *   Obtiene los SKUs únicos que tienen registros en ese año más reciente.
    *   Elige uno de esos SKUs de forma aleatoria.
    *   Retorna el DataFrame con todo el historial (todas las fechas) del SKU elegido.
    *

    """
    # 1. Configurar la semilla si se proporciona
    if seed is not None:
        random.seed(seed)

    # 2. Validar que existan filas en el dataset
    if df.height == 0:
        return df.filter(pl.lit(False))

    # 3. Determinar el año más reciente presente en la data
    anio_mas_reciente = df.select(pl.col(fecha_col).dt.year().max()).item()

    # 4. Obtener los SKUs únicos con registros en ese año
    skus_recientes = (
        df.filter(pl.col(fecha_col).dt.year() == anio_mas_reciente)
        .select(sku_col)
        .unique()
        .sort(sku_col)
    )

    # 5. Validar que existan SKUs en el año más reciente
    if skus_recientes.height == 0:
        return df.filter(pl.lit(False))

    # 6. Elegir un SKU al azar del conjunto real
    indice_aleatorio = random.randint(0, skus_recientes.height - 1)
    sku_seleccionado = skus_recientes[indice_aleatorio, sku_col]

    # 7. Retornar todo el historial del SKU elegido
    return df.filter(pl.col(sku_col) == sku_seleccionado)


def select_random_group(df: pl.DataFrame, seed: Optional[int] = None) -> pl.DataFrame:
    r"""
    Función que selecciona un grupo de planta y SKU al azar que hace lo siguiente:

    *   Fija la semilla del generador aleatorio de Python si se especifica.
    *   Obtiene todas las combinaciones reales y ordenadas de plantas y SKUs existentes en el dataset.
    *   Extrae una combinación (planta y SKU) de forma aleatoria del conjunto real.
    *   Filtra y retorna el DataFrame conteniendo únicamente las filas correspondientes al grupo elegido.
    *

    """
    # 1. Configurar la semilla si se proporciona
    if seed is not None:
        random.seed(seed)
        
    # 2. Obtener las combinaciones reales y únicas de planta y SKU del dataset
    df_combinaciones = df.select(["planta", "sku"]).unique().sort(["planta", "sku"])
    
    # 3. Validar que existan combinaciones en el dataset
    if len(df_combinaciones) == 0:
        return df.filter(pl.lit(False))
        
    # 4. Elegir un índice de fila al azar de las combinaciones reales
    indice_aleatorio = random.randint(0, len(df_combinaciones) - 1)
    
    # 5. Obtener los valores reales del grupo elegido
    planta_seleccionada = df_combinaciones[indice_aleatorio, "planta"]
    sku_seleccionado = df_combinaciones[indice_aleatorio, "sku"]
    
    # 6. Filtrar el DataFrame original para conservar solo el grupo real seleccionado
    return df.filter(
        (pl.col("planta") == planta_seleccionada) & 
        (pl.col("sku") == sku_seleccionado)
    )

