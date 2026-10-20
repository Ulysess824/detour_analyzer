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
| Contraseña del ajuste | La definió el usuario; en el repositorio solo está su hash (`config/seguridad.json`), provisoria |

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
| 1 (hecha) | Servicios y API (consumo real, búsqueda aproximada, estado); capa `src/app/` en `tests/test_layers.py` |
| 2 (hecha) | Estructura visual, Inicio y Consumo real |
| 3 (hecha) | Predicción y Seguimiento |
| 4 (hecha) | Ajuste con contraseña, `results/ajustes.json` y `results/tuning_produccion.json` (separado del de investigación, protegido por `check_tuning_window`) |
| 5 (hecha, en revisión) | Datos, Ayuda, Descargar e Historial |
| 6 | Lanzador pywebview, spec de PyInstaller, compilación en CI de Windows y guía de instalación |

## Fase 0 (hecha)

- Se eliminaron `app/` y `.streamlit/`; `requirements/app.txt` queda en `-r base.txt` hasta la fase 1.
- Servicios nuevos, que usan los scripts y la app: `src/engine/prediction_run.py` (`run_prediction`), `src/evaluation/tuning_run.py` (`run_tuning`) y
  `src/evaluation/tracking_run.py` (`track_saved_forecast`, `track_backtest`, `save_tracking`). `predict_next_month.py`, `tune_trees.py` y
  `track_forecast.py` son envolturas finas.
- Equivalencia comprobada: el pronóstico de octubre, la tabla del seguimiento y el JSON de una afinación corta salen igual que antes.
- `tests/test_services.py` (7 pruebas): 184 pruebas en total, cobertura de `src/` por encima de 90%.

## Fase 1 (hecha)

Capa `src/app/` (la prueba de capas ya la admite) y API local de solo lectura:

| Endpoint | Qué devuelve |
|---|---|
| `GET /api/salud` | `{"ok": true}` |
| `GET /api/estado` | Datos (filas, series, último día, último mes completo), mes del próximo pronóstico, último pronóstico, último seguimiento, seguimiento pendiente y ajuste (último, próximo, estado) |
| `GET /api/plantas`, `GET /api/skus?planta=` | Opciones para los filtros |
| `GET /api/sugerencias?campo=planta\|sku&q=&planta=` | Nombres más cercanos a lo que se escribe |
| `GET /api/consumo?planta=&sku=&nivel=mensual\|diario&desde=&hasta=` | Serie, resumen y desglose (por planta, o por SKU con los 8 mayores y "Otros") |

- Lo escrito se acepta si coincide con un nombre sin importar mayúsculas, tildes ni signos; si no, la API responde 404 con las sugerencias más cercanas.
- Sin datos responde 503 con el mensaje "carga los exportes en Datos"; el estado igual responde.
- Un SKU sin planta suma ese nombre en todas las plantas; un SKU que apareció tarde empieza en su primer mes (antes de eso, vacío).
- El último mes incompleto se marca como parcial y queda fuera del resumen (misma regla que el pronóstico).
- `results/ajustes.json` guarda la fecha del último ajuste (2026-10-05, la del commit de `results/tuning_trees.json`) y el intervalo de 6 meses; la fase 4 lo escribe al reafinar.
- 25 pruebas nuevas (`tests/test_app_api.py`, `tests/test_app_search.py`); 6 mutaciones inyectadas, las 6 detectadas.

## Fase 2 (hecha)

Pantallas servidas por la propia API (`src/app/static/`: `index.html`, `app.css`, `app.js`, `logo.png`, `vendor/echarts.min.js` 5.5.1 con su licencia Apache-2.0; sin internet ni compilación).

- **Estructura:** barra lateral blanca con el logo, 9 secciones (las que aún no existen salen atenuadas y muestran "disponible en una próxima versión"), encabezado con la fecha del último dato, tarjetas blancas, índigo como color principal, naranja y verde azulado de apoyo.
- **Inicio:** tarjetas de Datos, Próximo pronóstico, Último seguimiento (con las alertas en palabras) y Ajuste (próxima fecha y estado), avisos si el seguimiento está pendiente o el ajuste vence, y el gráfico del consumo total.
- **Consumo real:** planta y SKU escribibles con sugerencias (flechas y Enter), "Limpiar filtros", cuatro indicadores, gráfico mensual (total o desglose por planta / por SKU) o diario con rango de fechas, y vista de tabla. Si lo escrito no existe, ofrece los nombres cercanos como botones.
- **Gráficos:** paleta de 8 colores validada con el validador del skill de visualización (sin fallos; naranja, verde azulado y azul claro quedan bajo 3:1 de contraste, por eso cada gráfico tiene leyenda, tooltip y vista de tabla). Máximo 7 series y "Otros" en gris. Líneas de 2 px, barras de hasta 24 px con extremo redondeado, mes parcial sombreado.
- Solo tema claro (el de la imagen de referencia).
- Verificación: capturas con Chromium sin ventana contra los datos reales (`results/` y `data/`), 2 pruebas nuevas (los archivos que carga la página existen y el JavaScript no tiene errores de sintaxis; `node --check`).

## Fase 3 (hecha)

**Predicción.** Genera el pronóstico del mes siguiente al último mes completo con el mismo servicio que el script (`run_prediction`), en un hilo aparte con barra de avance (unos 25 s con los datos reales). Si ya existe, pide confirmación antes de reemplazarlo; si el último mes de los datos está incompleto, el botón queda bloqueado con el motivo. Muestra el pronóstico de la selección (planta y SKU escribibles), contra el último mes real, el rango entre LightGBM, XGBoost y Random Forest (el desacuerdo entre modelos, no un intervalo estadístico), el gráfico con los últimos 24 meses y la tabla por planta (al tocar una planta se ven sus SKU) o por SKU.

**Seguimiento.** Cuando un mes pronosticado ya cerró en los datos, avisa y el botón "Medir acierto" lo compara con el real (`track_saved_forecast`) y lo guarda en `results/seguimiento.csv` y, si existe `data/detour.db`, en la base. Muestra el acierto por mes del modelo principal (SKU, planta y total; rombos para los meses reales medidos y puntos para la prueba histórica), el detalle de un mes (cinco modelos, plantas de peor a mejor acierto, alertas en palabras, series sin pronóstico) y las 15 series donde más falló en toneladas (error = pronóstico − real).

| Endpoint | Qué hace |
|---|---|
| `GET /api/pronosticos` | Pronósticos guardados |
| `GET /api/pronostico?mes=&planta=&sku=&limite=` | Detalle del pronóstico de una selección |
| `POST /api/pronostico/generar` `{sobrescribir}` | Empieza la tarea; 409 si ya existe y no se permite reemplazar, o si hay otra tarea en marcha |
| `GET /api/trabajos/activo`, `GET /api/trabajos/{id}` | Estado y avance de la tarea (la pantalla la retoma si se sale y se vuelve) |
| `GET /api/seguimiento` | Historial de acierto, umbrales y meses por medir |
| `GET /api/seguimiento/detalle?mes=&origen=` | Detalle de un mes (producción o prueba histórica) |
| `POST /api/seguimiento/medir` `{mes}` | Mide un mes cerrado |

- Una prueba de extremo a extremo con datos sintéticos encontró un error real: el pronóstico se guardaba en la carpeta desde donde se corría y no en la de la app; ahora la ruta sale de `AppPaths`.
- `.coverage` dejó de estar en git (estaba por error desde el sub-plan 4).
- 13 pruebas nuevas en `tests/test_app_forecast.py` (la tarea en segundo plano, el detalle, la medición, el orden de los meses, el mes parcial); 7 mutaciones inyectadas, las 7 detectadas después de reforzar 3 pruebas. 224 pruebas, cobertura de `src/` 97%.
- Capturas con Chromium sin ventana contra datos reales: se generó el pronóstico de septiembre desde los datos de agosto, se cerró el mes y se midió (66,2 / 88,4 / 99,2, igual que la prueba histórica).

## Fase 4 (hecha)

**Contraseña.** `src/app/security.py`: solo se guarda el hash (scrypt, o PBKDF2 si esa instalación de Python no tiene scrypt) con su sal en `config/seguridad.json`; el texto no está en ningún archivo del repositorio. Al desbloquear se crea una sesión en memoria (30 minutos sin usarla, se pierde al cerrar la app); cinco claves malas seguidas bloquean la pantalla un minuto. Para cambiarla: `python scripts/set_app_password.py`. Es provisoria y evita usos accidentales; no es seguridad real: quien tenga los archivos de la app puede saltársela, y el hash de una clave corta se puede adivinar si alguien ve el archivo.

**Ajuste de hiperparámetros.** La pantalla muestra siempre (sin clave) la fecha del último ajuste, la del próximo (cada 6 meses) y el estado. Con la clave, "Empezar el ajuste" corre el mismo `run_tuning` del script en segundo plano con avance por prueba, sobre los datos hasta el último mes completo y con los últimos 12 meses como prueba (no se usan para elegir). Dos tipos: Completo (100, 100 y 40 pruebas, unos 25 minutos) y Rápido (15, 15 y 8, unos 11 minutos, para probar).

- El resultado queda como **candidato** (`results/tuning_candidato.json`): no cambia nada hasta que se adopte. La pantalla muestra, por modelo, el acierto en SKU, planta y total con los parámetros de antes y los nuevos, y avisa si no mejora de forma clara.
- **Adoptar** guarda el candidato como `results/tuning_produccion.json`, mueve el anterior a `results/ajustes_anteriores/`, actualiza `results/ajustes.json` (fecha y historial) y desde ahí los pronósticos de la app usan esos parámetros; el manifiesto de cada pronóstico registra el archivo y su hash. **Descartar** lo borra y no mueve la fecha.
- Los parámetros de investigación (`results/tuning_trees.json`) no se tocan: siguen protegidos por `check_tuning_window` en los scripts de backtest.
- Si el último mes de los datos está incompleto, o hay otra tarea en marcha, no se puede empezar.

| Endpoint | Qué hace |
|---|---|
| `GET /api/ajuste` | Estado, historial, candidato, tipos de ajuste y si hay clave (sin clave) |
| `POST /api/ajuste/desbloquear` `{clave}` | Devuelve el token de la sesión; 401 con intentos restantes, 429 si está bloqueado |
| `POST /api/ajuste/bloquear`, `/reafinar` `{tipo}`, `/adoptar`, `/descartar` | Con el encabezado `X-Ajuste-Token` |

- 15 pruebas nuevas (`tests/test_app_ajuste.py`): hash y sal, sesiones y vencimiento, bloqueo y reinicio de intentos, el flujo completo (ajustar, adoptar, respaldo, el pronóstico usa el archivo de producción), descartar, mes incompleto y el avance por prueba. Una de ellas encontró un error de un segundo en el aviso de bloqueo.
- **Tiempos medidos con datos reales** (ajuste rápido): con la comprobación del script (h=1 y h=3, 3 semillas) tardó 33 minutos, de los cuales la búsqueda eran 3; casi todo el tiempo era la comprobación en los 12 meses de prueba. En la app esa comprobación es solo h=1 y una semilla (las semillas difieren en unos 0,05 puntos), y el ajuste rápido pasó a 10,7 minutos; el completo se estima en 25 (la búsqueda de 17 minutos más la comprobación). Los tiempos de CLAUDE.md anteriores eran solo de la búsqueda.
- Se agregó `on_trial` a `run_study` para el avance por prueba (el script de línea de comandos sigue igual).

## Fase 5 (hecha)

**Datos.** Se eligen (o sueltan) los exportes de SAP en Excel. La app los sube a `data/staging/`, los **revisa sin escribir nada** y solo guarda si se acepta:
1. Los lee con el mismo lector y la misma regla `drop_zeros` con que se armó `data/consumos_long.csv` y los suma a la tabla actual.
2. Un día que ya estaba debe traer el mismo valor (se deja una sola vez). Si trae otro, la carga se detiene y el usuario elige: no cargar, o usar los valores del exporte nuevo.
3. Corre las mismas validaciones de entrada del sub-plan 1 sobre el resultado y las muestra en español, con ejemplos de filas. Los errores bloquean; los avisos no. El mes parcial es un aviso (se puede cargar, pero no se pronostica hasta que esté completo).
4. Al aplicar, copia la tabla anterior a `data/respaldo/` (los 10 últimos), escribe la nueva y recarga. "Deshacer la última carga" devuelve el respaldo más reciente. Si los datos cambiaron después de la revisión, se rehúsa.
- Comprobado con datos reales: con la tabla hasta agosto y el exporte de octubre 2025 a septiembre 2026, agrega 10.522 filas (septiembre), deja 116.977 días repetidos una vez y el resultado es **idéntico byte a byte** a `data/consumos_long.csv` (hash 161f8ff6d81d). La lectura usa `float_precision="round_trip"` para no alterar ningún dígito de los valores que no se tocan.
- Si se usa la base SQLite (`data/detour.db`), hay que reconstruirla después (`scripts/build_database.py --rebuild`); la app lee el CSV.

**Descargar.** Excel del pronóstico de un mes (resumen, por planta, cada SKU con los tres modelos, su rango y el último mes real), del consumo real con los filtros (total por mes y una fila por serie, o días) y del seguimiento (historial, modelos, plantas y mayores errores). Cada gráfico tiene un botón "Imagen (PNG)" (2144 × 800 px comprobado). Cada pronóstico se compara con el mes anterior a él, aunque ya haya meses posteriores cargados.

**Historial.** Cada pronóstico guardado con su fecha, el último día de los datos, el archivo de parámetros, el total y, medido, su acierto; "Ver" lo abre en Predicción y "Excel" lo descarga.

**Ayuda.** Los tres pasos del mes, glosario, preguntas frecuentes, dónde están los archivos y la versión (`0.1.0-beta`).

| Endpoint | Qué hace |
|---|---|
| `GET /api/datos` | Datos actuales y respaldos |
| `POST /api/datos/subir` (varios .xlsx) | Guarda los archivos en una carpeta temporal |
| `POST /api/datos/revisar` `{id, politica}` | Revisión en segundo plano (`detener` o `usar_nuevo`) |
| `POST /api/datos/aplicar`, `/cancelar` `{id}`, `/deshacer` | Aplicar, cancelar o deshacer |
| `GET /api/historial`, `/api/acerca` | Historial y versión |
| `GET /api/descargas/pronostico`, `/consumo`, `/seguimiento` | Archivos Excel |

- 41 pruebas nuevas (datos, descargas, historial y la comparación con el mes anterior); 7 errores inyectados en la carga, los 7 detectados (uno solo lo detectó una prueba nueva con valores de 17 dígitos).
- El CI de Windows encontró un error real que Linux no ve: `read_export` dejaba abierto el archivo Excel y en Windows no se podía borrar la carpeta temporal tras cargar; ahora cierra el libro (y una prueba lo vigila).
- La descarga usa `fetch` y un enlace temporal; falta comprobar cómo se comporta dentro de la ventana de Windows (fase 6).

## Contraseña

Se guarda solo el hash (scrypt), no el texto, en un archivo de configuración. Una contraseña en una app local evita usos accidentales; no es
seguridad real: quien tenga los archivos de la app podría saltársela.

## Límites

- El `.exe` no se puede compilar ni probar en este entorno (Linux): lo compila el CI de Windows.
- La interfaz se verifica con capturas de Chromium sin ventana.
- La reafinación completa tarda unos 20 minutos (lgbm 3 min, xgb 9 min, rf 8 min); la app debe mostrar el avance.
