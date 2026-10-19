# Plan y resultados: mes completado con la media como entrada del ml_mean

Rama `experiment/nowcast-ml-mean`. Solo archivos nuevos; no se modificó ningún modelo ni resultado existente.

## Idea

El mes objetivo `t` se pronostica a mediados del mes anterior (`t-1`), cuando solo se conocen sus primeros 15 días (la misma situación del planificador).
El mes `t-1` se completa con la media y entra como dato al `ml_mean` a un mes, que pronostica `t`.

| Mes que se pronostica | Mes que se completa (día 15) | Último mes completo real |
|---|---|---|
| Junio | Mayo | Abril |
| Julio | Junio | Mayo |
| Agosto | Julio | Junio |
| Septiembre | Agosto | Julio |

## Método (A)

1. **Mes completado**, por serie: consumo real de los primeros 15 días + media diaria de los días hábiles vistos (sin domingos) × días hábiles que faltan.
   Días con consumo: los vistos, escalados al mes.
2. **Variables:** las 25 del `ml_mean` con origen `t-1`, con el mes completado y la historia real hasta `t-2`.
3. **Modelo:** `ml_mean` a un mes (LightGBM, XGBoost y Random Forest con los hiperparámetros ajustados), entrenado solo con meses reales completos (objetivos hasta `t-2`).
4. **Salida:** pronóstico de `t` por serie; planta y total son sumas.

Código: `src/experiments/nowcast_input_utils.py` (reutiliza `with_pseudo_month`, `frame_at_origin`, `ml_forecasts`) y `scripts/nowcast_input_ml_mean.py`.
Pruebas: `tests/test_nowcast_input.py` (el resto real del mes en curso y todo lo posterior no cambian el pronóstico).

```bash
python scripts/nowcast_input_ml_mean.py data/consumos_long.csv --months 2026-06 2026-07 2026-08 2026-09
python scripts/nowcast_input_ml_mean.py data/consumos_long.csv --last-months 12
```

## Comparación (mismas series: vivas en `t-2`)

| Método | Información |
|---|---|
| A. Mes completado con la media | Mes `t-2` real y 15 días de `t-1` |
| B. Directa | Solo hasta `t-2` |
| C. Directa + 15 días | Hasta `t-2` y variables del mes parcial (`mtd`) |
| D. Referencia | El mes `t-1` real completo (cota superior) |
| F. Iterada | Hasta `t-2`; el mes `t-1` lo completa un primer pronóstico del modelo |
| E. Naive | Mes `t-2` repetido |

## Resultados

Junio a septiembre de 2026 (8.626 series y mes), acierto (1 − WAPE) acumulado:

| Método | SKU | Planta | Total | SKU, junio / julio / agosto / septiembre |
|---|---|---|---|---|
| A. Mes completado con la media | 65,4% | 89,1% | 97,7% | 67,0 / 66,0 / 63,6 / 64,5 |
| B. Directa | 63,0% | 89,6% | 96,5% | 66,2 / 62,6 / 59,4 / 63,0 |
| C. Directa + 15 días | **65,8%** | **90,0%** | **98,3%** | 67,4 / 66,6 / 63,7 / 64,8 |
| D. Referencia | 68,1% | 90,8% | 98,9% | 69,4 / 68,8 / 67,1 / 66,6 |
| F. Iterada | 61,7% | 88,4% | 95,2% | 63,7 / 61,8 / 58,6 / 62,1 |
| E. Naive | 57,5% | 83,3% | 90,7% | 60,2 / 59,8 / 50,8 / 57,9 |

Contraste con los 12 meses de prueba (octubre de 2025 a septiembre de 2026): A 68,1% / 89,7% / 95,2%; B 66,1%; C 68,7% / 90,9% / 96,0%; D 71,0% / 91,8% / 97,5%; F 65,4%; E 62,1%.

- **A mejora a la directa** (+2,0 puntos por SKU en 12 meses; gana en 10 de 12 meses y en los 4 de junio a septiembre) y a la iterada (+2,7).
- **A no mejora a "directa + 15 días"** (−0,6 puntos en 12 meses; C gana en 11 de 12 meses): la media completa el mes con la misma información que las variables
  parciales (`mtd_proj` ya proyecta el total del mes), solo que por otra vía.
- **D** indica lo que quedaría por ganar si se conociera el mes anterior completo: unos 2,7 puntos por SKU.
- Los resultados de junio a septiembre por serie están en `results/nowcast_input_ml_mean_2026-06_09.csv`.

### Contra el planificador y el modelo interno (SCAN, 156 SKU y mes, junio a septiembre)

El planificador pronostica a mitad de mes, así que el modelo comparable en información es A.

| | Acierto | Sesgo | Más cerca del real (de 156) |
|---|---|---|---|
| Modelo a mitad de mes (A) | 84,0% | +0,8% | 77 |
| Planificador | 83,2% | +3,2% | 44 |
| Modelo interno | 76,7% | +7,3% | 35 |
| Modelo con el mes anterior completo (D) | 85,3% | −0,4% | (en el deck original: 69, 50 y 37) |

Por mes (modelo A / planificador / modelo interno): junio 81,4 / 81,5 / 71,8; julio 83,9 / 82,3 / 79,8; agosto 86,2 / 88,3 / 78,2; septiembre 84,6 / 80,6 / 76,8.
El deck "Planificador contra modelo" tiene tres diapositivas nuevas con estos números (`mitad-mes`, `mitad-meses`, `mitad-cercania`).

## Límites
- Cuatro meses de objetivo en la lectura principal; el contraste de 12 meses va en la misma dirección. Sin prueba estadística pareada.
- El modelo se entrena con meses reales completos y recibe uno estimado; no se probó entrenar con meses completados de la misma forma.
- La comparación con el planificador es de una sola planta (SCAN) y 156 SKU y mes.
