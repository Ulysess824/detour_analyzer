# Sub-plan 3: entorno reproducible

Rama `feature/reproducible-env`. Archivos: `requirements/base.txt`, `evaluation.txt`, `app.txt`, `dev.txt`, `requirements.txt` y `.github/workflows/tests.yml`.
No se cambió código del proyecto.

## Requirements

Se fijan solo las dependencias directas, con `==`, a las versiones con las que se hicieron todos los resultados (así valen en Windows y en Linux).

| Archivo | Contenido |
|---|---|
| `requirements/base.txt` | numpy 2.4.6, pandas 3.0.6, scikit-learn 1.9.1, lightgbm 4.7.0, xgboost 3.2.0, openpyxl 3.1.5 |
| `requirements/evaluation.txt` | base, scipy 1.17.1, optuna 5.0.0 |
| `requirements/app.txt` | base, streamlit 1.64.0, plotly 7.1.0 |
| `requirements/dev.txt` | evaluation, pytest 9.1.1, matplotlib 3.11.2, ipykernel 7.4.0 |
| `requirements.txt` | dev y app (para que `pip install -r requirements.txt` siga funcionando) |

Se quitó `joblib` del archivo: ningún archivo del proyecto lo importa (viene con scikit-learn). `matplotlib` pasó a dev porque solo lo usa el notebook.
Python: 3.11 o 3.12 (pandas 3 y numpy 2.4 exigen 3.11 o más).

## Comprobaciones hechas (Linux, Python 3.11, entornos nuevos)

| Prueba | Resultado |
|---|---|
| Entorno solo con `base.txt` (1 min 24 s de instalación) y `predict_next_month.py` | 2.247 series, total 53.219,6 TO; el CSV es **idéntico byte a byte** a `results/forecast_2026-10.csv` (`c4ffe614…`); su manifiesto registra las versiones del entorno |
| Entorno con `dev.txt` y `pytest` | 131 pruebas pasan |
| Entorno con `app.txt` encima | Los módulos del dashboard se importan y el servidor responde (HTTP 200, sin errores en el log) |
| Todo lo que importa `src/engine/` | Está en el entorno de `base.txt` |
| Hilos | Con 1, 2 y 4 hilos el pronóstico tiene el mismo hash |

## CI (`.github/workflows/tests.yml`)

En cada subida y en cada pull request: instala `requirements/dev.txt` y corre `pytest`, en Linux y Windows con Python 3.11 y 3.12 (4 trabajos). Las pruebas
usan solo datos sintéticos; los datos del repositorio no se leen ni se muestran.

## Pendiente de confirmar fuera de este entorno
- Que el CI pase en los 4 trabajos (en particular Windows y Python 3.12, que no pude probar aquí).
- Que el pronóstico de octubre dé el mismo hash en tu PC con Windows. Si no coincide, comparar valor por valor: se esperarían diferencias de decimales.
