# Prueba grande del nowcasting con la media, enero a septiembre de 2026

Rama `experiment/nowcast-media-2026`. Código: `src/experiments/nowcast_backtest_utils.py`, `scripts/nowcast_media_2026.py`, `tests/test_nowcast_backtest.py`.
No se modificó ningún archivo existente de código ni de pruebas.

## Pregunta

Con los primeros 15 días de cada mes y el resto relleno con la media diaria, ¿se acierta el consumo del mes? Por planta y por SKU, en los 9 meses de 2026.

## Diseño

| Salida | Pronóstico | Real contra el que se mide |
|---|---|---|
| Resto del mes | media de los días vistos (sin domingos) × días no domingo que faltan | consumo de los días 16 a fin de mes |
| Mes completo | 15 días reales + resto estimado | consumo del mes |

- Modelos del nowcast: media (la estrategia a probar), mediana, Monte Carlo historia + mes y Monte Carlo solo mes (200 escenarios).
- Referencias disponibles antes de que empiece el mes: `ml_mean` a un mes (`results/member_forecasts.csv`) y el mes anterior completo. Se comparan solo en el mes completo, porque predicen todo el mes.
- Series: las vivas el mes anterior (cubren entre 98,1% y 99,9% del volumen real de cada mes).
- Acierto = 1 − WAPE en tres niveles (SKU con todas las series juntas, planta y total); sesgo = pronóstico / real − 1. El acumulado junta los 9 meses; su total es 1 − Σ|F − R| / Σ R sobre los totales mensuales.
- Solo se leen días anteriores al corte; una prueba lo verifica.
- Tablas (`results/`): `nowcast_2026_global.csv`, `nowcast_2026_planta.csv`, `nowcast_2026_planta_sku.csv` (una fila por planta, SKU y mes, 18.984 filas) y `nowcast_2026_planta_sku_resumen.csv` (cada planta y SKU acumulado en los meses).

## Criterios fijados antes de ver los resultados

1. La media gana o empata (dentro de medio punto) con la mediana y los dos Monte Carlo en SKU, planta y total, en el resto, acumulado.
2. En el mes completo, la media iguala o supera a `ml_mean` a un mes en planta y en total, en al menos 6 de los 9 meses.
3. El sesgo del total del mes completo queda dentro de ±5% en más de la mitad de los meses.
4. Se listan las plantas donde el nowcast es peor que `ml_mean` a un mes.

## Resultados (18.984 SKU-mes, 9 meses)

Mes completo, acumulado:

| Modelo | SKU | Planta | Total | Sesgo del total |
|---|---|---|---|---|
| Monte Carlo historia + mes | 79,8% | 92,7% | 95,6% | −1,4% |
| **Media** | 79,5% | 92,2% | 95,9% | −2,1% |
| Monte Carlo solo mes | 79,4% | 92,1% | 95,7% | −1,9% |
| `ml_mean` a un mes | 70,4% | 91,7% | 96,7% | +0,8% |
| Mediana | 69,8% | 75,5% | 75,6% | −24,4% |
| Mes anterior | 67,2% | 88,6% | 93,0% | +0,6% |

Resto del mes, acumulado:

| Modelo | SKU | Planta | Total | Sesgo del total |
|---|---|---|---|---|
| Monte Carlo historia + mes | 61,1% | 85,9% | 91,5% | −2,8% |
| **Media** | 60,5% | 84,9% | 92,1% | −4,1% |
| Monte Carlo solo mes | 60,3% | 84,8% | 91,8% | −3,7% |
| Mediana | 41,9% | 52,7% | 53,0% | −47,0% |

Criterios:

| Criterio | Resultado | Detalle |
|---|---|---|
| 1. Media gana o empata con la mediana y los Monte Carlo | No cumple | Monte Carlo historia + mes la supera por 0,6 puntos en SKU y 1,0 en planta; la media gana en el total (92,1% contra 91,5%) |
| 2. Media ≥ `ml_mean` a un mes en al menos 6 de 9 meses | No cumple | planta 5 de 9, total 5 de 9 |
| 3. Sesgo del total dentro de ±5% | Cumple | 7 de 9 meses |
| 4. Plantas peores que `ml_mean` (mes completo, acumulado) | Informativo | PCEL, SALC, SALI, SAOI, SCAN, SCRS, SCVA, SQUA, STTA |

Otras lecturas:
- En SKU-mes, la media queda más cerca del real que `ml_mean` a un mes en 14.562 de 18.984 casos (sin empates), y `ml_mean` en 4.422.
- Los dos meses con peor sesgo son enero (−11,6%) y abril (−11,7%): en ambos, el ritmo diario de los primeros 15 días fue muy inferior al de la segunda mitad (enero 1.659 contra 2.102 TO por día con consumo, abril 1.831 contra 2.501). La causa no se verificó; pueden ser feriados de la primera quincena.
- Los meses en que `ml_mean` supera a la media en planta y en total incluyen enero, abril, junio y (en planta) agosto.
- Por planta (acierto del total, media contra `ml_mean`): ganan la media MMRO (91,0 contra 80,7), SALM (91,3 contra 88,1), SPAL (93,9 contra 90,1) y SVIG (95,2 contra 93,2); pierde claramente SAOI (85,5 contra 93,4), STTA (82,3 contra 87,7) y PCEL (84,3 contra 88,2). STTA es la peor a nivel SKU (28,6%).
- Día de corte (media, acumulado):

| Corte | Resto: SKU / planta / total | Mes completo: SKU / planta / total |
|---|---|---|
| Día 10 | 56,7 / 79,7 / 85,3 | 70,4 / 86,1 / 90,0 |
| Día 15 | 60,5 / 84,9 / 92,1 | 79,5 / 92,2 / 95,9 |
| Día 20 | 59,1 / 83,7 / 90,7 | 85,9 / 94,4 / 96,8 |

El mes completo mejora con el corte sobre todo porque hay más días reales conocidos; en el resto, pasar del día 15 al 20 no mejora.

## Límites

- El nowcast usa 15 días reales del mes objetivo y `ml_mean` a un mes no: la comparación del mes completo no es de igual información. El resto del mes no se pudo comparar con `ml_mean` sin suponer un reparto de su pronóstico en el mes (se descartó la prorrata).
- Nueve meses y un solo año: los meses ganados son una referencia, no una prueba estadística.
- Los domingos no cuentan como día de consumo; los feriados no se tratan.
