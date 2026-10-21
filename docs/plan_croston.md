# Plan: método de Croston para series intermitentes

Rama `experiment/croston`. Solo archivos nuevos; no se modifica ningún modelo ni resultado existente. Estado: plan, sin código. Decisiones del usuario: ejemplo SVIG `TSL/01/1/100gsm/2300mm`, test todo 2026 y pronóstico **semanal** (sección "Versión semanal"); las secciones mensuales de abajo quedan como contexto del proyecto y su evaluación mensual pasa a ser secundaria.

## Idea

Hay series con muchos meses en cero. Los modelos de árboles y el `naive` las tratan como una serie continua. Croston separa dos cosas: **cuánto** se consume cuando hay consumo (tamaño) y **cada cuántos meses** hay consumo (intervalo). Pronostica ambas con suavizado exponencial simple y las divide.

Pregunta del experimento: ¿mejora Croston (o su variante SBA) el `ml_mean` y el `naive` en las series intermitentes, con las mismas reglas de evaluación del proyecto?

## Referencia

Shenstone y Hyndman (2005), *Stochastic models underlying Croston's method for intermittent demand forecasting*, Journal of Forecasting 24(6). Archivo esperado: `bibliografia/hyndman_croston.pdf`, clave `shenstone2005` en `scripts/download_bibliografia.py`.

- El entorno bloqueó la descarga (403 de robjhyndman.com). Hay que permitir ese dominio en la política de red y correr `python scripts/download_bibliografia.py --only shenstone2005`, o dejar el PDF a mano en `bibliografia/`.
- Antes de citar cifras o conclusiones del artículo hay que leer el original. Aquí solo se usa lo conocido del método.
- Resultado conocido del artículo y de la literatura a verificar al leerlo: los modelos que justifican a Croston tienen propiedades poco naturales, y el pronóstico de Croston puede estar sesgado (de ahí la variante SBA, Syntetos y Boylan).

## Método

Para una serie con historia hasta el origen `o`, solo con meses conocidos:

1. Meses con consumo positivo: tamaños `z_i` e intervalos `p_i` (meses entre consumos positivos).
2. Suavizado exponencial simple de `z` y de `p` con parámetro `alpha`, actualizado solo cuando hay consumo.
3. Pronóstico de `o+1`:
   - Croston clásico: `z_hat / p_hat`.
   - SBA: `(1 - alpha/2) * z_hat / p_hat`.
4. `alpha` fijo (0,1 y 0,2, como valores de referencia) y, como variante, elegido por serie en una ventana pasada; nunca con el mes objetivo.
5. Pronóstico plano en el horizonte; h=1 primero, h=2 después si h=1 funciona.

Unidad: mes por (planta, SKU), igual que el resto del proyecto. Un mes con cero significa cero consumo (los negativos ya están recortados a 0). Un mes anterior a la primera aparición de la serie (NaN en el panel) no cuenta.

## Qué series entran

Clasificación de Syntetos-Boylan-Croston por serie, con la historia hasta el origen:

- `ADI` = meses observados / meses con consumo.
- `CV²` = (desviación / media)² de los tamaños positivos.
- Cortes habituales: ADI 1,32 y CV² 0,49 → suave, errática, intermitente, irregular (lumpy).

En el panel completo (2.247 series, 33 meses; 1.925 con 12 meses o más de historia):

| Medida | Series (≥ 12 meses) |
|---|---|
| Con algún mes en cero | 1.302 |
| Cero en más del 25% de los meses | 1.008 |
| Cero en más del 50% de los meses | 856 |
| Suave / errática / intermitente / irregular | 793 / 107 / 769 / 255 |

La medición usa toda la historia (incluye meses de test), solo para dimensionar el experimento. En la evaluación la clasificación se recalcula en cada origen con datos hasta ese origen.

Grupos a comparar: intermitentes + irregulares (ADI ≥ 1,32) como grupo principal; las demás como control (Croston no debería ganar ahí).

## Evaluación

- Rolling origin h=1 sobre los 12 meses de prueba (2025-10..2026-09). Solo divisiones cronológicas.
- Mismo universo y misma tabla que `member_forecasts.csv` (se une por `series, o, t`), de modo que los números son comparables con el `ml_mean`.
- Modelos: `croston`, `sba`, `naive`, `ml_mean`; y una combinación `ml_mean` + `sba` (promedio simple) para ver si aporta algo al ensemble.
- Métricas: WAPE y acierto (1 − WAPE) a nivel SKU, planta y total (planta y total por suma de los SKU); `bias_pct` > 0 = pronóstico bajo. Se informa por grupo (intermitente / control) y por mes.
- Los ceros reales pesan en el WAPE solo por lo que predicen de más; además se informa el porcentaje de meses con consumo en el que el modelo pronostica 0 o casi 0.
- Criterios fijados antes de ver resultados (para no elegir después):
  1. Croston/SBA supera al `naive` en el grupo intermitente.
  2. Supera o iguala al `ml_mean` en el grupo intermitente a nivel SKU.
  3. No empeora planta ni total al sustituir o combinar solo en ese grupo.
  Si falla el 2 y el 3, queda como referencia y no se adopta.
- Prueba de significancia: la misma `model_confidence_set` del proyecto, con la advertencia ya anotada (12 meses es poco; solo diferencias grandes son confiables).

## Versión semanal (decisión del usuario)

El usuario quiere pronosticar por **semanas**, no por meses, sin modificar el backend existente (`src/engine/`, `src/evaluation/` y los resultados mensuales no cambian). Consecuencias:

- Todo el código semanal vive en `src/experiments/` (módulo propio, con su propia agregación semanal a partir de `data/consumos_long.csv`, que es diario). No se toca `Panel`, `make_frame` ni los modelos de árboles: el `ml_mean` es mensual y aquí no se reentrena.
- Semana = lunes a domingo. Las semanas parciales (primera y última del dato) se descartan. Una semana sin fila de consumo vale 0, solo dentro del rango en que la serie existe. El consumo en domingo cuenta (como en el resto del proyecto).
- El origen es la última semana completa conocida; h=1 es la semana siguiente. Si Croston funciona, h=4 como segundo horizonte.
- Rolling origin semanal: cada semana de 2026 se pronostica con datos hasta la semana anterior (ventana expansiva). Nada posterior al origen entra en el pronóstico ni en `alpha`.
- Comparación, todos con la misma información:
  - `croston`, `sba` (alpha 0,1 y 0,2);
  - `naive` (semana anterior), `mean4` y `mean8` (media de las últimas 4 y 8 semanas), `ses` (suavizado exponencial simple de la serie semanal);
  - `ml_mean` mensual repartido por semana (referencia al modelo ya adoptado): el pronóstico del mes de `results/member_forecasts.csv` (h=1) se reparte entre las semanas del mes por días hábiles; una semana que cruza dos meses toma la parte de cada mes. Solo se calcula en los meses con pronóstico en la caché; es una referencia, no un modelo semanal.
  - La estacionalidad anual no es evaluable: la serie de ejemplo empieza el 2025-07-09.
- Métricas semanales: WAPE y acierto (1 − WAPE) = 1 − Σ|y−p| / Σy, `bias_pct` > 0 = pronóstico bajo; además MAE en TO y acierto del acumulado mensual (suma de las semanas pronosticadas del mes contra el real del mes), para comparar con el nivel mensual del proyecto.
- Con una sola serie la prueba es un ejemplo ilustrativo, no una conclusión. Si el ejemplo es prometedor, el siguiente paso es el grupo intermitente completo con la misma evaluación semanal.

## Ejemplo

Planta **SVIG**, SKU **`TSL/01/1/100gsm/2300mm`** (formato del planificador: grade / sub-grade / gsm / ancho). Test: **todo 2026**.

- **Nombre en el consumo:** `TSL/01/100gsm/2300mm/1200-1400` (grade / sub-grade / gsm / ancho / ancho del core). Es la única serie de SVIG con 100 gsm y 2300 mm. El `1` extra del nombre del planificador no aparece en el consumo; hay que confirmarlo con el usuario si no es solo el formato.
- **Historia disponible:** de 2025-07-09 a 2026-09-29, 134 filas diarias (51 en 2025, 83 en 2026), 429,2 TO en total (156,1 TO en 2025, 273,2 TO en 2026). Sin consumo en domingo.
- **Semanas completas** desde 2025-07 (lunes a domingo): 65, de ellas 6 en cero (3 en 2025, 3 en 2026). La media semanal es 6,0 TO en 2025 y 7,0 TO en 2026 (desviación 5,3 y 5,0; máximo 21,4 TO). Mensual: de 6,0 a 46,8 TO.
- **Test 2026:** 38 semanas completas (la semana que empieza el 2026-09-28 es parcial y se descarta). Entrenamiento inicial: las 26 semanas de 2025 con consumo. Tres semanas en cero de 38 son pocos ceros: la serie es irregular (tamaño muy variable) más que intermitente. Esto es relevante para el experimento: Croston aporta poco si casi no hay ceros, y SBA/SES pueden quedar iguales.
- **Se muestra a mano**, semana a semana: serie, `z_hat`, `p_hat`, pronóstico de Croston y SBA, real y error, junto a `naive`, `mean4` y `ses`.

Limitación de esta serie: solo 26 semanas de historia antes del test, sin un año completo para estacionalidad, y la serie nació en 2025-07 (SKU nuevo). Los resultados describen este SKU y no se extrapolan.

## Entregables (cuando se implemente)

| Pieza | Archivo |
|---|---|
| Croston, SBA, clasificación ADI/CV² | `src/experiments/croston_utils.py` |
| Agregación semanal, rolling origin semanal, reparto del `ml_mean` mensual por semana | `src/experiments/weekly_utils.py` |
| CLI delgada (`--planta`, `--sku`, `--test-year`, `--alpha`, `--horizon`) | `scripts/croston_weekly.py` |
| Resultados | `results/croston_weekly_*.csv` |
| Pruebas | `tests/test_croston.py` |

Pruebas previstas: sin fugas (cambiar el mes objetivo o posteriores no cambia el pronóstico), serie constante da la constante, serie con un solo consumo, serie toda en cero da 0, intervalo p_hat correcto en un ejemplo calculado a mano, SBA = (1 − alpha/2) × Croston, la capa `src/experiments` solo importa engine y evaluation (`tests/test_layers.py`). Cobertura de `src/` ≥ 90%.

Al terminar se actualiza `CLAUDE.md` (tabla de scripts y resultados) y el README.

## Riesgos

- 12 meses de prueba: diferencias pequeñas no son concluyentes.
- Las series del proyecto son mensuales y mezclan SKUs de volumen muy distinto; el WAPE lo domina el volumen grande, donde Croston no debería aportar. Por eso el informe por grupo.
- Croston ignora estacionalidad y tendencia; el `ml_mean` las usa. Si Croston gana, será por estabilizar los ceros, no por modelar mejor la demanda.
- Elegir `alpha` con los mismos meses de prueba sería sesgo de selección: se fija `alpha` o se elige solo con meses anteriores al origen.
