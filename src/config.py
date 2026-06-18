import os

# Obtener la ruta absoluta de la raíz del proyecto
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))

# Directorios del proyecto
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, "notebooks")
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")

# Rutas específicas de los archivos de datos
HISTORICO_CONSUMO = os.path.join(DATA_DIR, "historico_consumo.parquet")
SYNTHETIC_CONSUMPTION = os.path.join(DATA_DIR, "synthetic_consumption.parquet")

# Crear directorios si no existen
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Diccionario para acceso dinámico o mapeado
DATA_PATHS = {
    "historico": HISTORICO_CONSUMO,
    "synthetic": SYNTHETIC_CONSUMPTION,
}
