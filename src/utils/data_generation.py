import os
import numpy as np
import pandas as pd

def generate_data():
    r"""
    Función que genera un dataset de consumo sintético que hace lo siguiente:

    *   Define dimensiones de 3 plantas y 50 SKUs.
    *   Simula consumos diarios para todo el año 2025 (365 días) con medias mensuales variables.
    *   Genera consumos con un 10% de probabilidad de ser un "pico" (> 3 veces el promedio diario esperado).
    *   Calcula el desfase de consumo del día anterior dentro de cada grupo de planta y SKU.
    *   Exporta el DataFrame resultante ordenado y en formato Parquet en `data/synthetic_consumption.parquet`.
    *

    """
    # Set seed for reproducibility
    np.random.seed(42)

    # Define dimensions
    plants = ["Planta_A", "Planta_B", "Planta_C"]
    skus = [f"SKU_{i:02d}" for i in range(1, 51)]
    
    # Date range for the year 2025 (365 days)
    dates = pd.date_range(start="2025-01-01", end="2025-12-31", freq="D")
    
    records = []
    
    # Generate data per combination of plant and sku
    for plant in plants:
        for sku in skus:
            # Generate average daily consumption for each of the 12 months
            # Discrete values between 2 and 8 (inclusive) to allow 3x peaks within [1.0, 30.0]
            monthly_averages = {
                month: int(np.random.randint(2, 9))
                for month in range(1, 13)
            }
            
            # Generate daily records
            for date in dates:
                month = date.month
                avg_consumption = monthly_averages[month]
                
                # 10% probability of peak consumption (> 3 * average)
                is_peak_day = np.random.rand() < 0.10
                
                if is_peak_day:
                    # Peak: continuous value between 3 * average + 1.0 and 30.0
                    low_bound = min(3.0 * avg_consumption + 1.0, 29.0)
                    real_consumption = float(np.random.uniform(low_bound, 30.0))
                else:
                    # Normal day: average + normal noise, capped below 3 * average
                    noise = np.random.normal(loc=0.0, scale=1.0)
                    up_bound = 3.0 * avg_consumption - 0.1
                    real_consumption = float(np.clip(avg_consumption + noise, 1.0, up_bound))
                
                records.append({
                    "fecha": date,
                    "planta": plant,
                    "sku": sku,
                    "consumo_promedio_diario": avg_consumption,
                    "consumo_real": real_consumption
                })
                
    # Create DataFrame
    df = pd.DataFrame(records)
    df["consumo_promedio_diario"] = df["consumo_promedio_diario"].astype(int)
    
    # Sort values to ensure chronological order for shifting
    df = df.sort_values(by=["planta", "sku", "fecha"]).reset_index(drop=True)
    
    # Compute the consumption of the previous day (shifted by 1 within each plant/sku group)
    df["consumo_real_dia_anterior"] = df.groupby(["planta", "sku"])["consumo_real"].shift(1)
    
    # Rearrange columns as requested by the user, and keep current day's real consumption for reference
    df = df[[
        "fecha", 
        "planta", 
        "sku", 
        "consumo_promedio_diario", 
        "consumo_real_dia_anterior", 
        "consumo_real"
    ]]
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Save to Parquet format
    output_path = os.path.join("data", "synthetic_consumption.parquet")
    df.to_parquet(output_path, index=False)
    
    print(f"Dataset successfully generated with {len(df):_} rows.")
    print(f"Saved to: {output_path}")
