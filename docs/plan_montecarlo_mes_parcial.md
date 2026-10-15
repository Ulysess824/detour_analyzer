# Plan: completar el resto del mes por simulación de Monte Carlo

Rama `experiment/montecarlo-mes-parcial`. Solo archivos nuevos: no se modificó ningún módulo, script ni resultado existente.

## Idea

A mitad de mes (por ejemplo a mediados de octubre) se conoce el consumo de los primeros días. Se simula el consumo de los días que faltan,
se arma un mes completo (primeros días reales + días simulados) y ese mes alimenta el modelo a un mes para pronosticar el siguiente
(noviembre). Se repite K veces y se promedian los pronósticos (imputación múltiple de Rubin).

## Qué respalda la bibliografía y qué no (`bibliografia/README.md`)

| Pieza | Respaldo |
|---|---|
| Usar los datos incompletos del período en curso | Giannone, Reichlin y Small (2008): modelo de factores y filtro de Kalman. **No usa Monte Carlo**: rellena con el valor esperado, dentro del modelo |
| Varios sorteos y combinar los resultados | Rubin (1987): un solo valor subestima la incertidumbre |
| Simular consumo intermitente por día | Willemain et al. (2004); criticado por Gardner (2005) y Syntetos, Babai y Gardner (2015) |
| Usar el mes simulado **como si fuera dato real** | No encontré un trabajo que lo respalde. Es la estrategia *iterada* (ya medida: 65,4% contra 66,2% de la directa) |

## Fases

| Fase | Qué se hace | Estado |
|---|---|---|
| 0 | Rama y archivos nuevos (`montecarlo_utils.py`, `montecarlo_resto_mes.py`, `test_montecarlo.py`) | Hecha |
| 1 | Calendario: probabilidad de que cada planta consuma cada día de la semana, estimada con los últimos 365 días. Los domingos y feriados salen de los datos | Hecha |
| 2 | Simulador del resto del mes y su evaluación contra lo que pasó | Hecha (resultados abajo) |
| 3 | Mes completado = primeros días reales + días simulados, K sorteos por serie | Pendiente |
| 4 | Variables del mes completado con el modelo a un mes ya existente, un pronóstico por sorteo, promedio entre sorteos | Pendiente |
| 5 | Backtest de 12 meses contra directa (66,2%), directa + mes parcial (68,7%), iterada (65,4%) y naive (62,1%), con ablaciones: valor esperado en lugar de sorteos, K = 1 y K = 100, corte al día 10, 15 y 20 | Pendiente |
| 6 | Intervalos de pronóstico y su calibración | Pendiente |

Criterio para adoptarlo: mejorar a "directa + mes parcial" al menos 1 punto por SKU en 9 de los 12 meses, con intervalo de confianza que no incluya 0.
Si no, se documenta como resultado negativo.

## Simulador (fase 2)

Con solo lo conocido al corte, para cada día restante:

1. **La planta trabaja:** sorteo Bernoulli con la frecuencia con que esa planta consumió ese día de la semana en los últimos 365 días
   (compartido por todos los SKU de la planta ese día).
2. **El SKU consume:** Bernoulli con la proporción de días de trabajo con consumo (últimos 365 días), actualizada con los días vistos del mes
   (peso del prior: 10 días).
3. **Tamaño:** valor diario sorteado de los días con consumo del propio SKU en la misma ventana.

Calendario: el consumo es casi todo de lunes a viernes (97,5% de las toneladas; sábado 2,4%; domingo 0,1%). No se fijó la regla a mano:
el perfil sale de los datos.

## Resultados de la fase 2

Septiembre de 2026, corte al día 15, 200 sorteos, 2.231 series. Real de los primeros 15 días: 26.068 TO; real del resto: 24.416 TO;
simulado del resto: 27.684 TO (banda 10-90%: 26.653 a 28.733).

| Método | Acierto SKU (resto) | Planta | Total | Sesgo | Acierto SKU (mes completo) |
|---|---|---|---|---|---|
| Monte Carlo (media) | 49,4% | 79,1% | 86,6% | +13,4% | 75,5% |
| Prorrata de los días hábiles | 49,8% | 80,7% | 93,2% | +6,8% | 75,7% |
| Tasa diaria de 6 meses | 44,5% | 80,1% | 85,6% | +14,4% | 73,1% |

Últimos 12 meses (acierto SKU del resto del mes): Monte Carlo 60,0%, prorrata 59,8%, tasa de 6 meses 56,1%. Sesgo del total entre −14,2% y +25,4%
según el mes (promedio +0,2%).

Calibración de la banda 10-90%: a nivel SKU cubre el 84,1% de los casos (nominal 80%); **a nivel total solo el 25%** (en septiembre, 0%).

## Lectura y siguientes pasos

- A nivel SKU el Monte Carlo es **equivalente a la prorrata** y mejor que la tasa histórica. No aporta ganancia en la media.
- Los intervalos del total son demasiado estrechos: los sorteos de los SKU son independientes y falta un efecto común de nivel del mes
  (en septiembre el resto del mes real fue 6% menor que la prorrata y 12% menor que el simulado). Antes de la fase 3 conviene añadir un multiplicador de nivel mensual
  sorteado de la distribución histórica de (resto del mes / primeros días), y escalar los tamaños con la intensidad de la primera mitad.
- Septiembre es un caso malo (49,4%): SALI cayó de nivel y tuvo solo 17 días con consumo.
- Fases 3 a 5 solo valen la pena si, con esos ajustes, el simulador supera a la prorrata a nivel planta y total.

## Nowcasting month benchmark (cuatro modelos)

`src/utils/now_casting_current_month.py` y `scripts/now_casting_month_benchmark.py`. Modelos: media, mediana, Monte Carlo con historia y mes en curso, Monte Carlo
solo con el mes en curso. Todos usan solo lo conocido al corte (día 15); prueba automática en `tests/test_now_casting_current_month.py`.

Acierto (1 − WAPE) del resto del mes, promedio de 12 meses: SKU 60,0% (Monte Carlo con historia), 59,8% (media), 59,3% (Monte Carlo solo mes), 41,1% (mediana);
planta 85,0 / 84,6 / 84,2 / 53,4%; total 90,9 / 92,2 / 91,3 / 53,6%. Las tres primeras opciones quedan dentro de un punto: no se distinguen.
Septiembre de 2026: gana el Monte Carlo solo con el mes (SKU 50,1%, planta 81,2%) y la media a nivel total (93,2%); SALI (cayó de nivel) tiene acierto negativo en todos.
