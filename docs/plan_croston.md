# Plan: método de Croston y modelos estocásticos asociados (pronóstico semanal)

Rama `experiment/croston`. Solo archivos nuevos; no se modifica ningún modelo ni resultado existente (`src/engine/`, `src/evaluation/` y los resultados mensuales no cambian). Estado: versión semanal hecha (fases 0 a 6) versión diaria hecha (secciones "Versión diaria" y "Resultados diarios") prueba con 16 series de SVIG hecha (sección "Varias series de SVIG") y calendario de lunes a viernes, perfil por día y probabilidad de consumo (sección final).

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

## Resultados (SVIG `TSL/01/100gsm/2300mm/1200-1400`, test 2026)

Datos: `results/croston_weekly_*.csv`; tabla semana a semana de h=1 con real, todos los pronósticos y los límites: `results/croston_weekly_ejemplo.csv`. Estados al final de la serie: `alpha` estimado 0,2 a 0,3, `P` entre 1,01 y 1,08, intervalo medio `p` 1,1, `sigma` de los tamaños 4,8 a 5,2 TO (la media semanal es 7 TO).

**Acierto semanal, h=1** (38 semanas, real 270,4 TO; sesgo > 0 = pronóstico bajo):

| Modelo | Acierto | Sesgo |
|---|---|---|
| `mean8` | 49,6% | +10,0% |
| `ses` | 49,1% | +11,4% |
| `croston` con alpha 0,1 | 48,3% | +14,0% |
| Croston modificado | 47,6% | +14,3% |
| `sb0` | 46,6% | +20,7% |
| log-Croston modificado | 45,7% | +2,0% |
| `croston` | 45,7% | +9,3% |
| `mean4` | 41,0% | +7,6% |
| `sb1` | 37,3% | +25,7% |
| log-Croston (modelo, simulado) | 34,0% | +50,4% |
| Croston (modelo, simulado) | 31,9% | +56,2% |
| `naive` | 26,0% | +2,7% |

A h=4 el orden es parecido (`mean8` 50,8%, log-Croston modificado 50,7%, `croston` 50,3%, `naive` 28,2%).

**Mes completo** (9 meses de 2026; el modelo semanal se evalúa con el pronóstico de la última semana completa antes del mes por los días hábiles; `ml_mean` conoce hasta 6 días más): `ses` 72,2%, `ml_mean` 72,2%, `mean8` 72,0%, `croston` con alpha 0,1 71,4%, log-Croston modificado 71,2%, Croston modificado 68,2%, `croston` 67,5%, `sb0` 63,0%, `naive` 41,6%.

**Intervalos al 95%, h=1:**

| Modelo | Cobertura | Ancho medio (TO) | Semanas con límite inferior en 0 |
|---|---|---|---|
| Croston (simulado) | 92,1% | 14,5 | 38 de 38 |
| log-Croston (simulado) | 94,7% | 19,6 | 38 de 38 |
| Croston modificado (fórmula) | 94,7% | 19,0 | 38 de 38 |
| log-Croston modificado (fórmula) | 100% | 25,3 | 38 de 38 |

### Criterios fijados antes de ver resultados

1. **Algún modelo de Croston supera a `naive` y `mean4`:** se cumple. Lo hacen `croston`, `sb0`, Croston modificado y log-Croston modificado (la vara es `mean4`, 41,0%).
2. **Su cobertura está entre 85% y 100%:** se cumple para el Croston modificado y el log-Croston modificado.
3. **Si no se cumplen, queda como referencia:** como se cumplen 1 y 2, la regla escrita no lo deja solo como referencia.

### Lectura honesta

- **La vara era baja.** Los criterios se compararon contra `naive` y `mean4`, que son las referencias más débiles. `mean8` y `ses` superan a todos los modelos de Croston con alpha estimado y a su vez no superan al `ml_mean` mensual. Los criterios no se cambian después de ver esto; por eso se informa aquí.
- **Con tres semanas en cero de 38, Croston es casi un suavizado simple.** El intervalo medio es 1,1 y `Z/P` queda cerca del nivel de los tamaños, igual que `ses`.
- **El Teorema 3 subpronostica por construcción.** Como `P >= 1`, la probabilidad `rho = Phi((1-P)/sigma_Q)` no pasa de 1/2 y el pronóstico a un paso de los modelos de Croston y log-Croston es la mitad del nivel (+50% de sesgo). El artículo lo plantea como aproximación de `m_1`; no se corrigió.
- **`sb1` subpronostica (+25,7%)** con la fórmula impresa en el artículo; sigue pendiente verificarla contra Syntetos y Boylan (2001).
- **Los intervalos cubren pero no informan.** Todos incluyen el cero y miden de 2 a 3,5 veces el consumo semanal medio; que la cobertura quede entre 92% y 100% es fácil con intervalos tan anchos. El modelo modificado sí acierta la frecuencia de ceros: dice que hay cerca de 9% de semanas en cero (`1 - 1/p`) y hubo 3 de 38 (8%).
- **La serie crece en 2026** (6,0 TO en febrero, 46,5 / 35,8 / 41,9 TO en julio, agosto y septiembre; semanas de 0 a 19,3 TO). Todos los suavizados se retrasan y el sesgo es positivo (pronóstico bajo). Este experimento no usa estacionalidad ni tendencia.
- **Selección sobre el test:** de siete modelos candidatos se mira cuál supera la vara en las mismas 38 semanas; hay sesgo de selección y una sola serie.

### Qué sigue (no se hace sin pedido)

- Repetir con las series intermitentes de verdad (grupo ADI ≥ 1,32 del plan) para ver si Croston aporta donde sí hay muchos ceros.
- Verificar `F_sb1` y las condiciones de `alpha` fuera de la grilla (0, 1].
- Decidir si interesan los intervalos (única parte que el proyecto no tenía) con otro método menos ancho.

## Versión diaria (decisión del usuario, después de la semanal)

El usuario pidió pasar a pronóstico **a nivel diario**. Todo sigue en archivos nuevos de `src/experiments/`, `scripts/` y `tests/`; el backend y la app no se tocan, y la versión semanal queda como estaba (sus resultados no cambian).

- **Serie** (`daily_utils.daily_series`): un valor por día hábil (lunes a sábado), desde el primer consumo del SKU hasta el último día hábil del dato. Un domingo con consumo se suma al lunes siguiente (conserva los totales); un domingo que sea el último día del dato se descarta. Los festivos son días en cero.
- **Modelos:** los mismos de la versión semanal con el periodo en días (`Q_j` en días hábiles). Referencias: `naive`, `mean6` y `mean24` (una y cuatro semanas de días hábiles) y `ses`. Rolling origin diario: cada día de 2026 se pronostica al cierre del día anterior (h=1) con datos hasta ese día.
- **Agregados:** semana = 6 × el pronóstico diario del sábado anterior; mes = días hábiles del mes × el pronóstico diario del último día hábil antes del mes (el `ml_mean` tiene la misma información: datos hasta el fin del mes anterior). La versión plana ignora que el log-Croston modificado depende de `h` (la diferencia es del orden del 2%).
- **Métrica:** con más de la mitad de los días en cero, el pronóstico que minimiza el error absoluto es la mediana (0) y el que minimiza el cuadrático es la media `Z/p`. El WAPE diario premia pronosticar cerca de 0 y castiga a Croston aunque estime bien la tasa; por eso las métricas **primarias** son las agregadas (semana y mes) y el WAPE y el RMSE diarios son secundarios.

### Criterios nuevos (fijados antes de ver los resultados diarios)

Los criterios de la versión semanal se escribieron sabiendo que Croston no ganaba a `mean8` ni a `ses`, y por eso la vara era baja. Estos suben la vara. El mismo modelo de Croston (alpha estimado) debe:

1. superar a `naive`, `mean6`, `mean24` y `ses` en acierto semanal agregado;
2. igualar o superar al `ml_mean` en acierto mensual;
3. tener cobertura diaria del 95% entre 85% y 100%.

Solo se adopta si el mismo modelo cumple los tres; si no, queda como referencia. Los modelos candidatos son los siete con alpha estimado (`croston`, `sb0`, `sb1`, `modified`, `modified_log`, `croston_model`, `log_croston_model`).

## Resultados diarios (SVIG `TSL/01/100gsm/2300mm/1200-1400`, test 2026)

Datos: `results/croston_daily_*.csv`; script `scripts/croston_daily.py`. Serie: 385 días hábiles (2025-07-09 a 2026-09-30), 134 con consumo (35%); test: 234 días hábiles de 2026. En 2026 el día medio es 1,17 TO, el día con consumo 3,29 TO (máximo 10,7 TO) y 64,5% de los días están en cero.

**Acierto semanal agregado** (38 semanas; sesgo > 0 = pronóstico bajo):

| Modelo | Acierto | Sesgo |
|---|---|---|
| log-Croston (modelo, simulado) | 49,9% | +17,8% |
| `croston` | 48,5% | +12,9% |
| `sb0` | 48,1% | +15,1% |
| `croston` con alpha 0,1 | 47,2% | +7,7% |
| Croston modificado | 47,1% | +16,4% |
| log-Croston modificado | 47,0% | +10,1% |
| `ses` | 44,4% | +9,8% |
| `mean24` | 41,0% | +7,6% |
| Croston (modelo, simulado) | 37,6% | +44,4% |
| `mean6` | 26,0% | +2,7% |
| `naive` | -14,7% | +84,0% |
| `sb1` | 0,0% | +99,95% |

**Acierto mensual** (9 meses de 2026, mismo origen que el `ml_mean`): log-Croston modificado 74,8%, `croston` 74,4%, `sb0` 73,7%, Croston modificado 73,2%, **`ml_mean` 72,2%**, `croston` con alpha 0,1 70,6%, `ses` 64,6%, `mean24` 63,9%, log-Croston (modelo) 53,3%, Croston (modelo) 50,7%, `mean6` 41,1%, `naive` -56,0%.

**Diario** (234 días; secundario): ningún modelo pasa de acierto 0 (WAPE entre 100% y 164%). RMSE: `croston` 1,956, log-Croston modificado 1,957, `sb0` 1,958, Croston modificado 1,960, `ses` 1,977, `mean24` 1,990, `mean6` 2,098, `naive` 2,922: en error cuadrático los modelos de Croston son los mejores, como se esperaba.

**Intervalos diarios al 95%:** Croston (simulado) 94,9% de cobertura y ancho medio 5,3 TO; log-Croston (simulado) 97,4% y 6,2 TO; Croston modificado 95,3% y 5,6 TO; log-Croston modificado 98,3% y 7,1 TO. El límite inferior es 0 en los 234 días en los cuatro.

### Criterios

1. **Semanal:** la vara es `ses` (44,4%). La superan `croston`, `sb0`, Croston modificado, log-Croston modificado y el log-Croston simulado.
2. **Mensual contra `ml_mean` (72,2%):** lo igualan o superan `croston`, `sb0`, Croston modificado y log-Croston modificado.
3. **Cobertura diaria entre 85% y 100%:** la cumplen los cuatro modelos con intervalo.

**Cumplen los tres: Croston modificado y log-Croston modificado.** Con la regla escrita, para esta serie se adoptan.

### Lectura honesta

- **Es una sola serie.** 38 semanas y 9 meses; la ventaja sobre el `ml_mean` es de 1,0 a 2,6 puntos, dentro del ruido de 9 meses. Se eligió entre siete modelos mirando las mismas semanas y meses del test (sesgo de selección). No es una prueba de que Croston supere al `ml_mean`.
- **A nivel diario Croston sí está en su terreno** (intervalo medio de unos 3 días, 65% de días en cero) y mejora sobre su propia versión semanal: acierto mensual 74,4% contra 67,5% de la corrida semanal y semanal 48,5% contra 45,7%. No se probó por qué; una hipótesis es que el día resuelve mejor la estructura de ceros y tamaños.
- **Contexto de la corrida semanal:** su `mean8` (49,6% semanal) queda por encima de los Croston diarios salvo el log-Croston simulado (49,9%); esa referencia no estaba en el conjunto diario (`mean24` es de cuatro semanas) y no se agregó después de ver los resultados.
- **`sb1` queda inservible a nivel diario.** Con la fórmula tal como está impresa, `Z/(P·100^(P-1))` y `P` cercano a 3, el factor es de 10.000 y el pronóstico es casi 0 (acierto 0%). Sigue pendiente verificar la fórmula contra Syntetos y Boylan (2001).
- **Los modelos de un paso del Teorema 3 subpronostican** en 39% a 45% (siguen siendo una aproximación del artículo, no corregida); por eso no ganan aquí.
- **El WAPE diario no sirve para decidir**, por la razón de la sección anterior; el RMSE diario favorece a Croston.
- **Los intervalos cubren pero son anchos** (de 5 a 7 TO para un día de 1,2 TO de media), y el límite inferior es siempre 0. Dicen "este día puede ser 0 o hasta unos 6 TO", no más.

### Qué sigue (no se hace sin pedido)

- Probar otras series intermitentes con la misma evaluación diaria, y un mes de más (`2026-10`) cuando exista, antes de pensar en adopción.
- Verificar `F_sb1` y la restricción de alpha a (0, 1].
- Decidir cómo se usaría un pronóstico diario en el proyecto (hoy todo es mensual): no se conecta a la app ni a `predict_next_month.py` sin una decisión aparte.

## Varias series de SVIG (decisión del usuario)

El usuario dio 16 SKUs de SVIG (con su cliente) para repetir la prueba diaria: `docs/croston_svig_skus.txt`. Script `scripts/croston_daily_batch.py`, módulo `src/experiments/croston_batch.py`, resultados `results/croston_batch_*.csv`. Archivos nuevos; el backend y la app no se tocan.

- **Emparejamiento** con el consumo por grade, gsm y ancho (sin sub-grade ni ancho del core). 15 calzan solos. `HP2/01/1/155gsm/1990mm` calza con dos series: la principal (91 días con consumo, 143,5 TO en 2026) y otra que termina en `/Subst.` (9 días en 2025, sin consumo en 2026); se usa la principal (`ignore_substitutes`). `KS/01/257/165gsm/2200mm` calza con `KS/257/165gsm/2200mm/1200-1450`, coherente con el pendiente de KS.
- **Series:** de 7% a 41% de los días hábiles con consumo; 8 empiezan entre 2024-10 y 2025-07 y llegan a 2026 con entre 19 y 68 días con consumo. Son la muestra que dio el usuario, no una muestra al azar de todo el catálogo.
- **Protocolo:** el mismo de la versión diaria (rolling origin diario en 2026, semana y mes agregados, `ml_mean` con la misma información), por serie, juntando los resultados.
- **Agregación:** acierto agrupado `1 - sum|y-p| / sum y` sobre todas las series y periodos (lo dominan las series grandes); acierto por serie; número de series donde un modelo supera a otro, con prueba de signos exacta; bootstrap que remuestrea series completas (2.000 remuestreos, intervalo de 90%) para la diferencia agrupada contra `ml_mean` y `ses` (informativo, con 16 series es ancho).

### Criterios fijados antes de correr

El mismo modelo de Croston (alpha estimado) debe: (1) superar a `naive`, `mean6`, `mean24` y `ses` en acierto semanal agrupado; (2) igualar o superar al `ml_mean` en acierto mensual agrupado **y** ganarle en al menos la mitad de las series; (3) tener cobertura diaria agrupada del 95% entre 85% y 100%.

### Resultados (16 series, test 2026)

**Acierto semanal agrupado** (608 series-semanas; sesgo > 0 = pronóstico bajo):

| Modelo | Acierto | Sesgo |
|---|---|---|
| `sb0` | 40,3% | +13,3% |
| `croston` | 39,9% | +9,7% |
| log-Croston (modelo) | 39,6% | +15,6% |
| `croston` con alpha 0,1 | 39,6% | +6,9% |
| Croston modificado | 39,1% | +21,4% |
| log-Croston modificado | 39,0% | +17,3% |
| `ses` | 37,2% | +10,4% |
| `mean24` | 36,0% | +6,0% |
| `mean6` | 21,1% | +2,6% |
| `naive` | -4,2% | +93,2% |

**Acierto mensual agrupado** (144 series-meses, `ml_mean` con la misma información):

| Modelo | Acierto | Sesgo |
|---|---|---|
| `croston` | 65,5% | +11,3% |
| `sb0` | 65,3% | +14,8% |
| **`ml_mean`** | **64,6%** | +11,8% |
| `croston` con alpha 0,1 | 64,1% | +8,2% |
| log-Croston modificado | 62,6% | +17,1% |
| Croston modificado | 62,0% | +21,2% |
| `ses` | 58,2% | +9,7% |
| `mean24` | 56,1% | +7,0% |
| log-Croston (modelo) | 55,6% | +25,2% |
| Croston (modelo) | 50,8% | +37,5% |

**Contra `ml_mean` y `ses`, mes** (diferencia de acierto agrupado y su intervalo de 90% por bootstrap de series; series ganadas de 16; p de la prueba de signos):

| Modelo | Contra | Diferencia | Intervalo 90% | Series ganadas | p |
|---|---|---|---|---|---|
| `croston` | `ml_mean` | +0,9 | -0,6 a +2,4 | 11 | 0,21 |
| `croston` | `ses` | +7,3 | +4,5 a +10,8 | 15 | 0,001 |
| `sb0` | `ml_mean` | +0,7 | -0,6 a +2,1 | 10 | 0,45 |
| `sb0` | `ses` | +7,1 | +3,9 a +11,1 | 14 | 0,004 |
| Croston modificado | `ml_mean` | -2,6 | -6,2 a +0,7 | 9 | 0,80 |
| log-Croston modificado | `ml_mean` | -2,0 | -5,7 a +1,5 | 10 | 0,45 |
| Croston (modelo) | `ml_mean` | -13,9 | -16,3 a -10,5 | 3 | 0,02 |

**Diario agrupado** (3.744 series-días; secundario): WAPE entre 100% y 164% en todos, RMSE de `croston` 1,420, `sb0` 1,420, log-Croston modificado 1,425, Croston modificado 1,426, `ses` 1,438, `mean24` 1,437, `naive` 2,050. **Intervalos diarios al 95%:** cobertura 95,8% (Croston simulado), 95,8% (log-Croston simulado), 96,4% (modificado) y 97,1% (log modificado), con ancho medio de 3,7 a 4,3 TO y límite inferior en 0 en todos los días.

### Criterios

1. **Semanal:** la vara es `ses` (37,2%). La superan `sb0`, `croston`, Croston modificado, log-Croston modificado y log-Croston (modelo).
2. **Mensual contra `ml_mean`:** lo cumplen `croston` (65,5%, gana en 11 de 16 series) y `sb0` (65,3%, 10 de 16). Los modificados quedan por debajo del `ml_mean`.
3. **Cobertura diaria:** la cumplen los cuatro modelos que tienen intervalo.

**No hay un modelo que cumpla los tres: no se adopta.** Los dos que cumplen el 2 (`croston` y `sb0`) no tienen intervalo; los que lo tienen pierden contra `ml_mean`. Es un defecto de cómo escribí los criterios: el 3 solo puede cumplirlo un modelo con intervalo, y los dos que ganan en acierto no tienen. No se reescribe después de ver el resultado; en una versión futura conviene separar el criterio de exactitud (1 y 2) del de los intervalos.

### Lectura honesta

- **Croston supera con claridad a los suavizados simples** (`ses` y `mean24`, 15 de 16 series, intervalo de 90% de +4,5 a +10,8 puntos). Eso no ocurría de forma tan clara en la serie única.
- **Frente al `ml_mean` es un empate:** +0,9 puntos con un intervalo de -0,6 a +2,4 y 11 de 16 series (p = 0,21). Con 16 series y 9 meses no se distingue de cero.
- **Exploración posterior (cortes elegidos después de ver los resultados, sin valor de prueba):** en las 6 series con menos de 20% de días con consumo (550,9 TO, 54 series-meses) el Croston modificado tiene 56,0%, el log modificado 55,5%, `croston` 53,2% y el `ml_mean` 51,7%. En las 10 con 20% o más (1.890,8 TO) `croston` tiene 69,1%, `sb0` 68,5% y el `ml_mean` 68,4%. La ventaja de Croston sobre el `ml_mean` es algo mayor en las más intermitentes, pero son 6 series y el corte no estaba fijado de antemano.
- **Casos concretos:** en `K/01/160gsm/2200mm` el `ml_mean` gana a Croston (75,9% contra 72,5%) y en `K/01/110gsm/2000mm` Croston gana por mucho (33,5% contra 1,6%, con 33 días de consumo en toda la historia).
- **Los modelos del Teorema 3 subpronostican** (+37,5% de sesgo agrupado el Croston simulado) y pierden contra `ml_mean` en 13 de 16 series; `sb1` con la fórmula impresa da casi 0 en todas las series (acierto 0,2%). Ninguno de los dos es candidato.
- **Sesgo de selección y alcance:** se miran siete modelos sobre los mismos 9 meses; las series son las que dio el usuario y están en una sola planta; ocho tienen poca historia antes de 2026. No se puede extrapolar a otras plantas.

### Qué sigue (no se hace sin pedido)

- Otras plantas y las 769 series intermitentes del maestro, con los criterios 1 y 2 (exactitud) separados del de intervalos.
- Octubre de 2026 como prueba fuera de muestra cuando exista.
- Verificar `F_sb1` contra Syntetos y Boylan (2001).

## Calendario de lunes a viernes, perfil por día y probabilidad de consumo

Al ver el gráfico diario, el usuario observó que el modelo siempre pronostica consumo aunque muchos días no lo tienen. Un análisis descriptivo de los 16 SKUs (toda la historia, no una prueba) mostró: la probabilidad de consumo es de 30% a 33% de lunes a jueves, 23% el viernes y 0,3% el sábado; no sube con los días sin consumo (cerca de 26% hasta 5 días sin consumo y 17% a 20% después de 6); cuando hay consumo la mediana es 1,65 TO y la media 2,16 TO. Se implementaron dos cambios, sin tocar la app, el motor ni los resultados anteriores (las corridas con los argumentos por defecto reproducen exactamente `results/croston_batch_*.csv`).

### Métodos

- **Calendario** (`daily_utils.daily_series(weekdays=5)`, `scripts/croston_daily_batch.py --weekdays 5`): lunes a viernes; el consumo de sábado y domingo se suma al lunes siguiente. Semana = 5 días y mes = sus días de lunes a viernes. Las medias móviles pasan a 5 y 20 días (`mean5`, `mean20`).
- **Perfil por día** (`weekday_profile.py`, `--profile-shrink 20`): factor `w_k = 1 + (r_k - 1) n_k / (n_k + 20)`, con `r_k` el consumo medio de ese día de la semana sobre el consumo medio de todos los días y `n_k` los días de ese tipo en el historial hasta el origen; se reescala a media 1. El pronóstico del día es `w_k` por el pronóstico plano. La contracción 20 se fijó sin mirar el test. El total de una semana no cambia; el del mes cambia según cuántos viernes tenga. Cada modelo tiene una copia `<modelo>_dow`.
- **Probabilidad y tamaño** (`croston_probability.py`, `scripts/croston_probability.py`): el pronóstico plano de Croston es `Z/P = (1/P) x Z`, probabilidad diaria `1/P` por tamaño `Z`. Se comparan la probabilidad de Croston (`croston`), la frecuencia histórica (`freq`) y sus versiones con perfil por día de la semana (`_dow`, perfil calculado sobre el indicador "hubo consumo") con el Brier, la habilidad frente a la frecuencia y el log loss; los tamaños (`Z`, media, mediana y media de los últimos tres) se evalúan solo en los días con consumo.

### Criterios fijados antes de correr

1. El perfil por día baja el RMSE diario agrupado de Croston.
2. La probabilidad por día de Croston supera a la frecuencia histórica (habilidad de Brier mayor que 0).
3. El calendario de lunes a viernes no baja el acierto semanal ni mensual de Croston más de 1 punto frente a lunes a sábado.
4. El tamaño `Z` tiene menor WAPE que la mediana histórica en los días con consumo.

### Resultados (16 SKUs de SVIG, test 2026: 3.120 series-día, 31,9% con consumo)

| Criterio | Resultado | Cumple |
|---|---|---|
| 1. Perfil baja el RMSE diario | RMSE de Croston 1,520 sin perfil y 1,521 con perfil; el error absoluto medio baja de 0,971 a 0,969; con los otros modelos el RMSE también sube un poco | No |
| 2. Probabilidad supera a la frecuencia | habilidad +1,0% con perfil y +1,7% sin perfil; con perfil es positiva en 7 de 16 SKUs (máximo +13,9% en `TSL/01/150gsm/2450mm`, mínimo -5,8% en `K/01/125gsm/2200mm`) | Sí, por poco |
| 3. Calendario no daña | semanal 39,9 a 39,9; mensual 65,5 a 65,6 | Sí |
| 4. `Z` mejor que la mediana | WAPE 51,6% contra 52,2%; la media simple tiene 50,7%, y la media de los últimos tres 59,8% | Sí, por poco |

- **Acierto diario agrupado de Croston:** -34,2% con lunes a sábado y -24,1% con lunes a viernes (-23,8% con perfil). La mejora viene de dejar de pronosticar consumo el sábado; no es una comparación sobre los mismos días. Todos los modelos siguen con WAPE diario mayor que 100%.
- **Mensual agrupado con lunes a viernes:** `croston` 65,6%, `croston_dow` 65,6%, `sb0` 65,3%, `ml_mean` 64,6%, log-Croston modificado 62,7%, Croston modificado 62,2%, `ses` 61,3%, `mean20` 56,9%. Croston frente al `ml_mean`: +1,0 puntos (bootstrap de series de -0,6 a +2,5; gana en 11 de 16, p = 0,21). Frente a `ses`: +4,3 (de +1,7 a +7,8; 13 de 16, p = 0,021). La ventaja sobre `ses` bajó de +7,3 a +4,3 puntos porque el calendario de lunes a viernes también lo mejora a él (58,2% a 61,3%).
- **Calibración de la probabilidad de Croston por día:** la probabilidad media pronosticada es 31,0% y la real 31,9% (la frecuencia histórica pronostica 27,2%, porque el consumo de 2026 sube). Los tramos bajos subestiman (24,7% pronosticado contra 29,2% real en el tramo de 0,2 a 0,3) y el más alto sobreestima (66,1% contra 55,6% entre 0,6 y 1,0).

### Lectura honesta

- **El perfil por día no sirve en este test.** El viernes bajo se ve en toda la historia, pero al usarlo como pronóstico fuera de muestra no baja el RMSE ni mejora la probabilidad más que sin él.
- **La probabilidad de Croston solo supera a la frecuencia histórica por 1 a 2 puntos de habilidad.** Coincide con lo medido antes: los días sin consumo no anticipan cuándo vendrá el siguiente. Saber qué día cae el consumo con solo el historial no es posible aquí; para eso harían falta datos de pedidos o del cliente.
- **Lo que sí mejora es el calendario.** Quitar el sábado evita pronosticar consumo cuando casi seguro no habrá, y no cambia el acierto semanal ni mensual.
- **El tamaño:** `Z` supera a la mediana, pero la media histórica es algo mejor; las tres diferencias son de 1 punto.
- **Alcance:** 16 SKUs de una planta y 9 meses; se evaluaron varias variantes sobre el mismo test. La contracción 20 y la grilla de `alpha` no se ajustaron.

### Qué sigue (no se hace sin pedido)

- Repartir el pronóstico mensual del `ml_mean` por día con el perfil de día de la semana y compararlo con Croston diario.
- Probar si datos de pedidos, del planificador o del cliente dan una señal de qué día habrá consumo.
- Las 769 series intermitentes del maestro, con exactitud e intervalos como criterios separados.
