import random
from typing import Optional
import polars as pl

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

