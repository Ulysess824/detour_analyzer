# Plan: método de Croston y modelos estocásticos asociados (pronóstico semanal)

Rama `experiment/croston`. Solo archivos nuevos; no se modifica ningún modelo ni resultado existente (`src/engine/`, `src/evaluation/` y los resultados mensuales no cambian). Estado: plan, sin código.

Decisiones del usuario: la metodología es la del artículo de la sección siguiente; el ejemplo es SVIG `TSL/01/1/100gsm/2300mm`; el test es todo 2026; el pronóstico es **semanal**.

## Alcance: qué no se toca

Es un experimento. La app de escritorio y todo lo que ya funciona quedan intactos:

| No se toca | Por qué |
|---|---|
| `src/app/` (servicios, API, pantallas, empaquetado), `packaging/`, `config/` | El experimento no se conecta a la app; no hay endpoint, pantalla ni botón de Croston |
| `src/engine/`, `src/evaluation/` | Se importan para leer datos (`load_consumption`) y, si hace falta, métricas; no se editan |
| Scripts, resultados y tests existentes, `data/`, `requirements/` | Solo se agregan archivos nuevos. Sin dependencias nuevas |
| `scripts/predict_next_month.py` y el pronóstico de producción | Croston no entra al ensemble ni a la predicción mensual |

Solo se agregan archivos nuevos en `src/experiments/`, `scripts/`, `tests/`, `results/croston_weekly_*.csv` y `docs/`; las únicas ediciones a archivos existentes son documentación (`bibliografia/README.md`, `src/experiments/bibliography_utils.py` para registrar la referencia, y `CLAUDE.md` / README al terminar). Comprobación al cerrar cada fase: `git diff --stat origin/main` no debe listar nada bajo `src/app/`, `src/engine/` ni `src/evaluation/`, y `tests/test_layers.py` debe seguir pasando (la capa `experiments` no importa `app`).

## Fases

| Fase | Contenido | Salida | Criterio para pasar |
|---|---|---|---|
| 0 | Plan, referencia y rama (hecho) | este documento, `bibliografia/hyndman_croston.pdf` | revisión del usuario |
| 1 | `weekly_utils.py`: agregación semanal desde `data/consumos_long.csv`, descarte de semanas parciales, serie del ejemplo | tabla semanal de SVIG `TSL/01/100gsm/2300mm/1200-1400` | 64 semanas completas, 6 en cero, 427,2 TO (= 429,2 TO diarios menos 2,0 TO de la semana parcial del 2026-09-28) |
| 2 | `croston_utils.py`: `Z`, `P`, `F_cr`, `F_sb0`, `F_sb1`, Croston modificado y log modificado, estimación de `α` y `σ` | pronósticos de una semana | pruebas calculadas a mano y sin fugas |
| 3 | Intervalos 95%: fórmula cerrada y simulación con semilla fija | límites por semana | los límites contienen al pronóstico; sin negativos tras truncar |
| 4 | Rolling origin semanal 2026, referencias (`naive`, `mean4`, `mean8`, `ses`) y `ml_mean` mensual repartido | `results/croston_weekly_*.csv` | sin datos posteriores al origen; reparto mensual suma el mes |
| 5 | Métricas, cobertura, criterios de decisión, tabla a mano del ejemplo | informe en este documento | los tres criterios se evalúan tal como se fijaron arriba |
| 6 | Documentación: `CLAUDE.md`, README, resultados | commit en la rama | cobertura de `src/` ≥ 90%, `git diff` sin `src/app/` |

Cada fase termina con un commit en `experiment/croston`. No se hace pull request ni se mezcla con `main` sin pedido explícito del usuario.

## Metodología: Shenstone y Hyndman (2005)

*Stochastic models underlying Croston's method for intermittent demand forecasting*, Journal of Forecasting 24, 389-402. Archivo: `bibliografia/hyndman_croston.pdf` (14 páginas, subido por el usuario; clave `shenstone2005`). Las ecuaciones siguientes están leídas del PDF.

### Lo que el artículo da

Notación: `Y_t` demanda del periodo `t`; `Y*_j` tamaño de la j-ésima demanda positiva; `Q_j` periodos entre `Y*_{j-1}` e `Y*_j`; `Z_j` y `P_j` pronósticos de tamaño e intervalo.

1. **Croston (ec. 1-3).** SES separado de tamaño e intervalo, actualizado solo en los periodos con demanda, con el mismo `alpha`:
   `Z_j = (1-α) Z_{j-1} + α Y*_j`, `P_j = (1-α) P_{j-1} + α Q_j`; pronóstico de cualquier horizonte `F_cr = Z_l / P_l` (`l` = última demanda).
2. **Dos correcciones del pronóstico puntual** (aproximaciones de `m1 = Φ[(1-P)/σ] Z`): `F_sb0 = (1-α/2) Z_l/P_l` (Syntetos y Boylan 2000) y `F_sb1 = Z_l / (P_l · c^(P_l-1))` (Syntetos y Boylan 2001, tal como está impresa; el artículo usa `c = 100` y `α = 0,5`). La fórmula de `F_sb1` se verifica contra la fuente antes de usarla.
3. **Cuatro modelos estocásticos** (el artículo prueba que cualquier modelo detrás de Croston es no estacionario y definido sobre los reales, negativos incluidos):

   | Modelo | Tamaños | Intervalos | Pronóstico a h pasos |
   |---|---|---|---|
   | Croston (5) | ARIMA(0,1,1) | ARIMA(0,1,1) | solo a un paso (Teorema 3): `m1 = ρ Z`, `ρ = Φ[(1-P)/σ]` |
   | log-Croston (6) | log ARIMA(0,1,1) | log ARIMA(0,1,1) | solo a un paso: `m1 = ψ exp{Z + σ²/2}`, `ψ = Φ(-P/σ)` |
   | Croston modificado (7, Snyder 2002) | ARIMA(0,1,1) | geométrica i.i.d. con media `p` | `m_h = Z/p` (ec. 11) |
   | log-Croston modificado (9, Snyder 2002) | log ARIMA(0,1,1) | geométrica i.i.d. con media `p` | `m_h = exp{Z + σ²/(2p) · [p + α²(h-1)]} / p` (ec. 16) |

   Para los dos modificados también hay varianza del pronóstico (ec. 12 y 17) y media y varianza de la demanda sobre `h` periodos (ec. 13-14 y 18-19).
4. **Intervalos de predicción.** Se simulan trayectorias del modelo (el artículo usa 10.000), con los intervalos entre demandas redondeados al siguiente entero positivo, y se toman percentiles (2,5% y 97,5% para el 95%). Para los modelos modificados hay fórmula cerrada (Teorema 4, ec. 22 y 23). Los límites negativos se truncan en 0, lo que cambia la cobertura.
5. **Recomendación del artículo:** el log-Croston modificado es el candidato más atractivo (definido en los positivos y con media y varianza tratables). Todos los modelos son no estacionarios: la varianza crece con el horizonte y los intervalos se ensanchan.

### Lo que el artículo no da

- No compara exactitud: solo muestra una serie mensual de repuestos (Saturn, 1998-2002) y observa que los pronósticos puntuales de los modelos son muy parecidos y los intervalos muy distintos.
- No dice cómo se estiman `α` y `σ`, ni protocolo de evaluación. Se fijan aquí (siguiente sección).
- No trata semanas ni demanda continua en toneladas; las series del artículo son enteros por mes.

## Qué se implementa

Módulo nuevo `src/experiments/croston_utils.py` (solo numpy/pandas y lo que ya está instalado):

1. **Preparación de la serie:** tamaños `Y*_j` (semanas con consumo positivo) e intervalos `Q_j` en semanas.
2. **Estimación, solo con datos hasta el origen:** `α` y `σ_e` de cada ARIMA(0,1,1) (tamaños, o log de tamaños) por mínimos cuadrados de los errores de un paso, `σ²` = media de los errores al cuadrado; `α` restringido a (0, 2) como en el artículo. Para los modelos modificados, `p` = media de los intervalos observados. Estados iniciales `Z_0`, `P_0` = primer valor observado. Con menos de 3 demandas positivas el modelo no se ajusta y la serie queda fuera (se informa).
3. **Pronóstico puntual** de la semana siguiente: `F_cr`, `F_sb0`, `F_sb1`, Croston modificado (ec. 11), log-Croston modificado (ec. 16), y los dos modelos de un paso (Teorema 3, ec. 20 y 21).
4. **Intervalos 95%:** fórmula cerrada (ec. 22 y 23) para los modificados; simulación de 10.000 trayectorias con semilla fija para Croston y log-Croston.
5. **Demanda sobre 4 semanas** (≈ un mes): `M_h` de los modelos modificados (ec. 13 y 18), para compararla con el mes real y con el `ml_mean` mensual.

Script `scripts/croston_weekly.py` (CLI delgada: `--planta`, `--sku`, `--test-year`, `--horizon`, `--draws`, `--seed`). Resultados en `results/croston_weekly_*.csv`. Pruebas en `tests/test_croston.py`.

## Versión semanal

- Todo el código semanal vive en `src/experiments/`, con su propia agregación a partir de `data/consumos_long.csv` (diario). No se toca `Panel`, `make_frame` ni los modelos de árboles: el `ml_mean` es mensual y aquí no se reentrena.
- Semana = lunes a domingo. Las semanas parciales (primera y última del dato) se descartan. Una semana sin fila de consumo vale 0, solo dentro del rango en que la serie existe. El consumo en domingo cuenta, como en el resto del proyecto.
- Origen = última semana completa conocida; `h=1` es la semana siguiente. `h=4` como segundo horizonte (los pronósticos de Croston son planos en el horizonte).
- Rolling origin semanal: cada semana de 2026 se pronostica con datos hasta la semana anterior, ventana expansiva. Nada posterior al origen entra en el pronóstico ni en la estimación de `α` y `σ`.

## Evaluación

Modelos comparados, todos con la misma información: los cinco pronósticos de Croston de arriba; referencias `naive` (semana anterior), `mean4`, `mean8` y `ses` (suavizado exponencial simple de la serie semanal); y el `ml_mean` mensual repartido por semana (el pronóstico del mes de `results/member_forecasts.csv`, h=1, repartido por días hábiles; una semana que cruza dos meses toma la parte de cada mes; solo en los meses presentes en la caché; es una referencia al modelo ya adoptado, no un modelo semanal).

Métricas:
- WAPE y acierto (1 − WAPE) = 1 − Σ|y−p| / Σy, MAE en TO y `bias_pct` (> 0 = pronóstico bajo), por semana.
- Acierto del acumulado mensual: suma de las semanas pronosticadas del mes contra el real del mes (y `M_4` de los modelos modificados).
- **Intervalos:** cobertura empírica del 95% (fracción de semanas con el real dentro) y ancho medio, por modelo. Aquí el artículo sí aporta algo que el proyecto no tenía: intervalos.

Criterios fijados antes de ver resultados:
1. Algún modelo de Croston supera al `naive` y a `mean4` en acierto semanal en 2026.
2. Su cobertura 95% queda entre 85% y 100% (con 38 semanas, un cubrimiento menor indica intervalos demasiado estrechos).
3. Si ninguno cumple, Croston queda como referencia y no se adopta, ni siquiera para esta serie.

Una sola serie es un ejemplo ilustrativo, no una conclusión; el siguiente paso, si es prometedor, es el grupo intermitente completo con la misma evaluación.

## Ejemplo

Planta **SVIG**, SKU **`TSL/01/1/100gsm/2300mm`** (formato del planificador: grade / sub-grade / gsm / ancho). Test: **todo 2026**.

- **Nombre en el consumo:** `TSL/01/100gsm/2300mm/1200-1400` (grade / sub-grade / gsm / ancho / ancho del core). Es la única serie de SVIG con 100 gsm y 2300 mm. El `1` extra del nombre del planificador no aparece en el consumo; hay que confirmar con el usuario que es solo el formato.
- **Historia:** de 2025-07-09 a 2026-09-29, 134 filas diarias (51 en 2025 y 83 en 2026), 429,2 TO en total (156,1 TO en 2025 y 273,2 TO en 2026). Sin consumo en domingo.
- **Semanas completas** desde la semana del 2025-07-07: 64, de ellas 6 en cero (3 en 2025 y 3 en 2026), 427,2 TO (los 2,0 TO restantes están en la semana parcial del 2026-09-28). Media semanal 6,0 TO en 2025 y 7,0 TO en 2026 (desviación 5,3 y 5,0; máximo 21,4 TO). Mensual: de 6,0 a 46,8 TO.
- **Test 2026:** 38 semanas completas con lunes en 2026 (270,4 TO, 3 en cero); la semana que empieza el 2026-09-28 es parcial y se descarta. Entrenamiento inicial: las 26 semanas con lunes en 2025 (la del 2025-12-29 incluye días de enero y cuenta como 2025).
- **Con tres semanas en cero de 38, el intervalo medio entre demandas `p` es cercano a 1.** En el modelo modificado, `m_h = Z/p` queda muy cerca de `Z` (SES del tamaño), y Croston casi no se distingue de un SES. La serie es irregular (tamaño muy variable) más que intermitente. Es un resultado esperable, no un fallo: este ejemplo prueba el método en el caso en que menos ayuda; los intervalos y el log-Croston modificado (tamaños positivos) sí pueden aportar.
- Se muestra a mano, semana a semana: serie, `Z`, `P`, pronósticos, intervalo 95%, real y error.

Limitación: solo 26 semanas de historia antes del test, sin un año completo para estacionalidad, y la serie nació en 2025-07. Los resultados describen este SKU y no se extrapolan.

## Contexto de la intermitencia en el proyecto (datos mensuales)

Clasificación de Syntetos-Boylan por serie (ADI y CV², cortes 1,32 y 0,49), con toda la historia mensual de las 1.925 series con 12 meses o más: 793 suaves, 107 erráticas, 769 intermitentes y 255 irregulares; 1.302 tienen algún mes en cero. La medición incluye los meses de test y solo dimensiona la extensión futura del experimento a más series.

## Entregables

| Pieza | Archivo |
|---|---|
| Croston, SBA, modelos modificados, intervalos | `src/experiments/croston_utils.py` |
| Agregación semanal, rolling origin semanal, reparto del `ml_mean` mensual por semana | `src/experiments/weekly_utils.py` |
| CLI delgada | `scripts/croston_weekly.py` |
| Resultados | `results/croston_weekly_*.csv` |
| Pruebas | `tests/test_croston.py` |

Pruebas previstas: sin fugas (cambiar el objetivo o datos posteriores al origen no cambia el pronóstico), los estados `Z` y `P` solo se actualizan con demanda, serie constante da la constante, `F_sb0 = (1 − α/2) F_cr`, un caso calculado a mano de `Z`, `P` y `m_h`, intervalos que contienen al pronóstico y no son negativos tras truncar, semilla fija da el mismo intervalo simulado, semana parcial descartada, reparto mensual a semanas suma el mes, y `tests/test_layers.py` (experiments solo importa engine y evaluation). Cobertura de `src/` ≥ 90%. Al terminar se actualiza `CLAUDE.md` y el README.

## Riesgos

- 38 semanas de test en una sola serie: cualquier diferencia pequeña no es concluyente.
- Los modelos suponen no estacionariedad y errores normales (en logaritmos para los log): las semanas en cero y los picos (hasta 21 TO con media de 7) pueden romper el supuesto.
- La estimación de `α` y `σ` con 26 a 64 semanas es inestable; se informa `α` por origen y se compara con el `ses` simple.
- El artículo no valida exactitud: lo que se mida aquí no tiene respaldo en él, solo la forma de los modelos.
- No elegir `α` ni el modelo con las semanas de test.
