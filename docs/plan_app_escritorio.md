# Plan: app de escritorio que reemplaza el dashboard

Objetivo: un programa de Windows (`detour.exe`) que usa todo el backend y lo hace manejable por una persona sin conocimientos avanzados. Lo que
vale es la predicción del modelo; la del planificador es orientativa.

## Decisiones del usuario

| Tema | Decisión |
|---|---|
| Tecnología | pywebview (ventana nativa con WebView2) + FastAPI local en 127.0.0.1 + ECharts incluido en la app (sin internet) + PyInstaller (carpeta) |
| Estilo | Tarjetas blancas, barra lateral izquierda, colores índigo, naranja y verde azulado, logo de la empresa (`presentacion/logo_smurfit_westrock.png`) |
| Reafinación | Cada 6 meses; la app muestra la última fecha y la próxima por vencer |
| Datos | Sección para cargar los exportes SAP desde la app |
| Extras | Inicio, Datos, Ayuda, Descargar (PNG y Excel) e Historial de pronósticos |
| Contraseña del ajuste | "Marlen" |

## Secciones

1. **Inicio**: estado (último mes con datos, último pronóstico, próximo ajuste, alertas).
2. **Consumo real**: filtros por planta y por nombre de SKU, ambos escribibles con sugerencias de la coincidencia más cercana.
3. **Predicción**: genera el pronóstico del mes siguiente con barra de avance.
4. **Seguimiento**: al cerrar el mes compara el pronóstico guardado con el real (acierto por SKU, planta y total, alertas).
5. **Ajuste de hiperparámetros**: bloqueado con contraseña; muestra la última fecha y la próxima (6 meses).
6. **Datos**: carga de exportes SAP con la validación de entrada.
7. **Ayuda**, **Descargar** (PNG y Excel) e **Historial** de pronósticos.

## Fases

Cada fase termina con pruebas, capturas y confirmación del usuario antes de fusionar a `main`.

| Fase | Contenido |
|---|---|
| 0 | Quitar el dashboard Streamlit y mover la lógica de los scripts a servicios en `src/` (hecha) |
| 1 | Servicios y API (consumo real, búsqueda aproximada, estado); capa `src/app/` en `tests/test_layers.py` |
| 2 | Estructura visual, Inicio y Consumo real |
| 3 | Predicción y Seguimiento |
| 4 | Ajuste con contraseña, `results/ajustes.json` y `results/tuning_produccion.json` (separado del de investigación, protegido por `check_tuning_window`) |
| 5 | Datos, Ayuda, Descargar e Historial |
| 6 | Lanzador pywebview, spec de PyInstaller, compilación en CI de Windows y guía de instalación |

## Fase 0 (hecha)

- Se eliminaron `app/` y `.streamlit/`; `requirements/app.txt` queda en `-r base.txt` hasta la fase 1.
- Servicios nuevos, que usan los scripts y la app: `src/engine/prediction_run.py` (`run_prediction`), `src/evaluation/tuning_run.py` (`run_tuning`) y
  `src/evaluation/tracking_run.py` (`track_saved_forecast`, `track_backtest`, `save_tracking`). `predict_next_month.py`, `tune_trees.py` y
  `track_forecast.py` son envolturas finas.
- Equivalencia comprobada: el pronóstico de octubre, la tabla del seguimiento y el JSON de una afinación corta salen igual que antes.
- `tests/test_services.py` (7 pruebas): 184 pruebas en total, cobertura de `src/` por encima de 90%.

## Contraseña

Se guarda solo el hash (scrypt), no el texto, en un archivo de configuración. Una contraseña en una app local evita usos accidentales; no es
seguridad real: quien tenga los archivos de la app podría saltársela.

## Límites

- El `.exe` no se puede compilar ni probar en este entorno (Linux): lo compila el CI de Windows.
- La interfaz se verifica con capturas de Chromium sin ventana.
- La reafinación completa tarda unos 20 minutos (lgbm 3 min, xgb 9 min, rf 8 min); la app debe mostrar el avance.
