# Plan: estrategias a mitad de mes y sección "Ajustes" en la app

Rama `prueba_nowcasting` (desde `main`, commit `13c870a`). Este documento es solo el plan: no cambia código.

## 1. Objetivo

El planificador entrega su forecast a mitad de mes. En ese momento el último mes completo es `t-2` y el mes `t-1` está en curso (unos 15 días conocidos). El pronóstico debe usar **todos los datos disponibles**, incluido el mes en curso, y la estrategia debe poder cambiarse desde una sección nueva de la app llamada **Ajustes**.

Hoy la app pronostica solo el mes siguiente al último mes completo (h=1) y bloquea el botón si el último mes está incompleto. Ni la directa a dos meses, ni la versión con mes parcial, ni el nowcasting de la media están conectados a la app (README, "Pendiente").

## 2. Cuál sirve mejor a mitad de mes

Objetivo: pronosticar el mes `t` desde el último mes completo `t-2`, con datos del mes en curso. Prueba 2025-10..2026-09, acierto SKU / planta / total (`compare_horizon_strategies.py`, 24.407 SKU-mes):

| Estrategia | SKU | Planta | Total | Lectura |
|---|---|---|---|---|
| Directa + primeros 15 días del mes en curso (`parcial`) | **68,7** | **90,9** | 96,0 | La mejor en SKU; gana a la directa en los 12 meses |
| Mes en curso completado con la media, como entrada del `ml_mean` (`nowcast_media`) | 68,1 | n. d. | n. d. | 0,6 puntos bajo `parcial` en SKU; en junio a septiembre: 65,4 SKU / 89,1 planta / **97,7 total** contra 65,8 de `parcial` |
| Directa a dos meses (`directa`) | 66,2 | 90,0 | 94,1 | No usa el mes en curso |
| Iterada (`iterada`) | 65,4 | 89,8 | 94,4 | Peor o igual que la directa en los 12 meses; no adoptada |
| Naive | 62,1 | 85,6 | 90,6 | Referencia |
| Un mes antes (`h1`) | 71,0 | 91,8 | 97,5 | Solo posible con el mes anterior completo |

**Recomendación por defecto a mitad de mes: `parcial` (directa + primeros 15 días).** Es la que más acierta por SKU y usa el mes en curso sin estimar nada. `nowcast_media` queda muy cerca y es la mejor en el total de junio a septiembre; es la alternativa si se prefiere un mes completado que se pueda mostrar.

Para estimar **el resto del mes en curso** (lo que falta de `t-1`), la decisión es la **media** (SKU 79,5 / planta 92,2 / total 95,9 en el mes completo, enero a septiembre de 2026). Mediana y Monte Carlo solo mes están descartados; Monte Carlo historia + mes empata y queda como investigación futura (solo meses atípicos).

Salvedades que la app debe mostrar: solo se validó el corte en el día 15; los parámetros ajustados son los de h=1; en SCAN (SKU grandes) `parcial` no mejora a la directa (84,1% contra 84,3%); la media falla en meses con cambio de ritmo entre quincenas (enero y abril de 2026, sesgo del total de unos -11,6%).

## 3. Estrategias del selector

Los identificadores son los que ya usa `consumos_predichos.estrategia` en SQLite.

| Id | Nombre en la app | Qué hace | Disponible cuando |
|---|---|---|---|
| `automatica` | Automática (recomendada) | Elige según el estado de los datos, ver la sección 4 | Siempre |
| `h1` | Mes siguiente (un mes) | Entrena y pronostica un mes adelante, como hoy | Último mes completo |
| `directa` | Dos meses, directa | Pronostica `t` desde `t-2` | Siempre |
| `parcial` | Dos meses, directa + primeros días del mes en curso | Añade las variables del mes en curso hasta el corte | Mes en curso con al menos el corte en días vistos |
| `nowcast_media` | Dos meses, mes en curso completado con la media | Completa el mes en curso con la media y lo usa como mes observado | Igual que `parcial` |
| `iterada` | Dos meses, iterada | Pronostica `t-1` y lo usa para `t` | Siempre; marcada "no recomendada" |

Ajustes de la sección:

- Estrategia (lista de arriba; por defecto `automatica`).
- Día de corte del mes en curso: **automático** por defecto (el último día con consumo en los datos, sección 4); se puede fijar a mano. El corte manual no puede pasar del último registro (usaría días que no existen) y muestra un aviso si queda fuera del rango validado.
- Qué hacer cuando el corte queda **por encima** del rango validado (por ejemplo día 28): usar la estrategia elegida (`parcial` por defecto) o pasar a la directa con el último mes completo. Por debajo del rango, la app usa la directa y lo avisa.
- Método para estimar el resto del mes (solo informativo en la pantalla Predicción): media por defecto.

## 4. Modo automático y detección del corte

El planificador no entrega siempre el mismo día, así que el corte no se fija en 15: **se detecta del último registro de los datos**.

**Cómo se detecta** (se reutiliza `incomplete_last_month_message` de `data_checks`, que ya decide si el último mes está terminado):

1. `ultimo` = la fecha más reciente con consumo en el maestro (sumando todas las plantas).
2. El último mes está **completo** si todos los días entre `ultimo` y el fin de ese mes son domingos. Si no, está **en curso**.
3. En curso: `corte` = día del mes de `ultimo`, y `dias_vistos` = días hábiles (sin domingos) hasta `ultimo`.

El código de las variables ya admite cualquier corte: `add_partial_features(frame, S, day)` y `month_to_date` calculan el consumo acumulado de los primeros `day` días de **cada mes del histórico**, así que el modelo se entrena con el mismo corte que tiene el mes en curso, sin reentrenar nada distinto del flujo normal. Lo único que falta es saber desde qué día el mes en curso aporta algo (fase 1).

| Estado de los datos | Qué hace | Mes pronosticado |
|---|---|---|
| Último mes completo | `h1` | El siguiente |
| Mes en curso con `corte` dentro del rango validado, o por encima si Ajustes lo permite (por defecto sí) | `parcial` (o la estrategia elegida en Ajustes) con ese `corte` | El siguiente al mes en curso |
| Mes en curso con `corte` por debajo del rango validado | `directa` con el último mes completo, avisando que falta información | El siguiente al mes en curso |

**Riesgos de la detección** (la pantalla muestra siempre "datos hasta <fecha>, corte día N" para confirmarlo):

- Los ceros se descartan al cargar (`--drop-zeros`), así que si el exporte llegaba hasta un día sin consumo en ninguna planta, el último registro quedaría antes. Con 15 plantas es raro, pero un feriado entre semana al final del exporte lo produciría. Se resuelve mostrando la fecha detectada y permitiendo fijar el corte a mano en Ajustes.
- Un último registro en sábado cuenta como día visto; un domingo no suma días hábiles (las variables ya cuentan sin domingos).
- Si el exporte llega a mitad de semana, el corte es ese día, no el viernes anterior: todos los días hábiles visto cuentan.

El "usar todos los datos disponibles" se cumple: el mes en curso entra siempre que su `corte` esté en el rango donde aporta.

## 5. Diseño de código

Capas (`tests/test_layers.py`): `engine <- evaluation <- experiments <- app`. La producción (`run_prediction`, usado por el script y la app) está en `engine`, así que lo que use debe estar en `engine`.

- **Ya en engine:** `partial_utils` (`month_to_date`, `add_partial_features`), `make_frame`, `member_utils`.
- **Mover a engine** (y que `evaluation`/`experiments` lo importen, sin duplicar): de `evaluation/horizon_utils.py`, `mask_panel`, `with_pseudo_month`, `ml_forecasts`; de `experiments/now_casting_current_month.py`, `seen_days` y `forecast_media`; de `experiments/nowcast_input_utils.py`, `completed_month`.
- **Detección:** `detect_month_state(df)` en `data_checks` (devuelve estado, último registro, mes en curso, `corte` y días hábiles vistos), sobre `incomplete_last_month_message`; la usan `resolve_strategy`, `run_prediction` y la pantalla de Datos.
- **Nuevo `src/engine/strategies.py`:** `forecast_with_strategy(S, tuned, strategy, cutoff_day)` (el corte llega de la detección) que devuelve el mismo formato que hoy (`planta, sku, mes, lgbm, xgb, rf, ml_mean, naive`) más `mes_origen` y `estrategia`; y `resolve_strategy(estado_de_los_datos, ajustes)` para el modo automático.
- **`run_prediction`:** parámetros `strategy` y `cutoff_day`; reemplaza el bloqueo por mes incompleto por la resolución de la estrategia; el manifiesto guarda estrategia, corte, mes en curso y qué meses son estimados.
- **`scripts/predict_next_month.py`:** opciones `--strategy` y `--cutoff-day`.
- **App:** `src/app/settings_service.py` (lee y escribe `config/ajustes_app.json`, valida los valores), endpoints `GET/POST /api/ajustes`, pantalla `static/views_ajustes.js`, entrada "Ajustes" en el menú. `forecast_service.generate` pasa los ajustes a `run_prediction`.
- **Predicción:** muestra la estrategia usada, el mes pronosticado y, si hubo mes en curso, una tarjeta con el real hasta el corte más la estimación del resto con la media (marcada como estimación). Seguimiento y SQLite guardan la estrategia de cada corrida para medir cada una por separado.
- **Nombre:** ya existe "Ajuste de hiperparámetros" (con clave). "Ajustes" puede confundirse; propuesta: dejar los dos y rotular "Ajustes: estrategia de pronóstico" y "Ajuste de hiperparámetros: reentrenar".

## 6. Pruebas

- Sin fuga del futuro: con el corte en el día `d`, nada posterior al día `d` del mes en curso cambia el resultado (se altera el consumo posterior y el pronóstico no se mueve).
- Equivalencia: `forecast_with_strategy` reproduce las columnas `p_direct`, `p_partial`, `p_iterated` de `results/horizon_strategies.csv` y `nowcast_input_forecast` del experimento (diferencia máxima 0,001 TO).
- Regresión: con el último mes completo y `h1`, el pronóstico de octubre de 2026 (`results/forecast_2026-10.csv`) no cambia.
- Detección: mes completo, mes en curso con cortes 5, 12, 15, 20 y 28, último registro en sábado y en domingo, mes terminado en domingos, exporte que acaba en feriado; el `corte` y los días hábiles vistos son los esperados.
- Modo automático: los tres estados de la sección 4; el mes pronosticado y el aviso son los esperados.
- Corte variable: con cortes distintos de 15 no hay fuga (nada posterior al corte cambia el resultado) y el modelo se entrena con el mismo corte que el mes en curso.
- Ajustes: valores inválidos rechazados, archivo ausente o dañado usa los valores por defecto, la estrategia elegida llega a `run_prediction`.
- Capas: `tests/test_layers.py` pasa; cobertura de `src/` de al menos 90%; mutaciones en `resolve_strategy` y en el corte.
- Windows CI: tests y autoprueba del `.exe` con `--forecast` (la diferencia entre plataformas ronda 0,06%).

## 7. Fases

0. **Esta rama y este plan.**
1. **Validación previa, curva por día de corte: hecha** (sección 9). Falta la misma curva para `nowcast_media`, que se puede correr después porque la estrategia por defecto es `parcial`.
2. **Engine: hecha.** Piezas movidas a \`engine\` (\`month_in_progress\`, \`month_state\`, \`strategies\`), 23 pruebas nuevas, y las tres estrategias comparadas con la caché del backtest sobre los datos reales (diferencia máxima 0,00000 TO). Las pruebas encontraron un error que habría afectado la producción: \`build_panel\` termina el calendario en el último registro, así que el mes en curso aparecía con menos días de los que tiene; \`complete_calendar\` lo corrige.
3. **Script:** `--strategy`, `--cutoff-day`, manifiesto.
4. **App backend:** servicio de ajustes y endpoints, `forecast_service` con la estrategia.
5. **App pantallas:** Ajustes y los cambios en Predicción.
6. **Seguimiento:** estrategia por corrida.
7. **Documentación:** README (incluida la tabla "Versiones", que hoy dice que la v2 es el commit `2876c4f`), `CLAUDE.md`, `docs/plan_app_escritorio.md`, `docs/guia_instalacion.md`.
8. **Opcional:** Monte Carlo historia + mes como método del resto del mes; empata con la media, así que no entra salvo que se decida.

Criterio de cierre: CI verde en Linux y Windows, el pronóstico a h=1 sin cambios, y las cifras de la sección 2 reproducidas dentro de 0,05 puntos.

## 8. Decisiones

1. "Ajustes" no lleva clave; la estrategia queda registrada en cada corrida.
2. El modo automático prefiere `parcial` (directa + días del mes en curso).
3. El corte se detecta del último registro de los datos y también se puede fijar a mano en Ajustes (nunca después del último registro). Fuera del rango validado la pantalla avisa.
4. Si el corte detectado supera el rango validado por arriba, se usa `parcial` igual; Ajustes permite cambiarlo a la directa con el último mes completo.

El rango validado salió de la fase 1: desde el día 10 (sección 9).

## 9. Resultado de la fase 1: acierto de `parcial` según el día de corte

Misma prueba que `compare_horizon_strategies.py` (2025-10..2026-09, 12 meses, 24.407 SKU-mes, `ml_mean`, parámetros de h=1), solo cambia el día de corte de la estrategia `parcial`. El corte 15 reproduce las cifras publicadas. Tabla en `results/curva_corte_parcial.csv`.

| Estrategia | SKU | Planta | Total | Meses de 12 que gana a la directa (SKU) |
|---|---|---|---|---|
| Directa (sin mes en curso) | 66,1 | 90,0 | 94,1 | |
| Iterada | 65,4 | 89,8 | 94,4 | |
| Parcial, corte día 5 | 66,8 | 89,8 | 93,8 | 10 |
| Parcial, corte día 8 | 67,2 | 89,9 | 94,2 | 11 |
| Parcial, corte día 10 | 67,8 | 90,5 | 94,8 | 12 |
| Parcial, corte día 12 | 68,2 | 90,8 | 95,1 | 12 |
| Parcial, corte día 15 | 68,7 | 90,9 | 96,0 | 12 |
| Parcial, corte día 18 | 69,2 | 91,0 | 95,9 | 12 |
| Parcial, corte día 20 | 69,4 | 91,3 | 96,6 | 12 |
| Parcial, corte día 22 | 69,6 | 91,2 | 96,4 | 12 |
| Parcial, corte día 25 | 70,2 | 91,8 | 97,5 | 12 |
| Parcial, corte día 28 | 70,7 | 92,1 | 97,6 | 12 |
| Un mes antes (referencia) | 71,0 | 91,8 | 97,5 | |

Lectura:

- El acierto sube con el día de corte y no tiene saltos: cuantos más días del mes en curso se conocen, mejor. En el día 25 y el 28 ya está cerca de pronosticar con el mes anterior completo (70,2 y 70,7 contra 71,0).
- **Rango validado: desde el día 10.** Desde ahí `parcial` supera a la directa en SKU, planta y total y gana en los 12 meses. En el día 5 y el 8 mejora poco en SKU (10 y 11 meses de 12) y no mejora a la directa en planta ni en total (89,8 y 93,8 en el día 5; 89,9 y 94,2 en el día 8, contra 90,0 y 94,1).
- Regla para la app: corte de 10 o más, `parcial`; corte menor que 10, directa y aviso. Por encima del rango no hay límite: el día 28 es el mejor.
- Salvedad: es el universo de todos los SKU vivos. En los SKU grandes de SCAN los 15 días no mejoraban a la directa (84,1% contra 84,3%), y esta curva no lo cubre.
- Falta la curva de `nowcast_media` (mes completado con la media como entrada): se corre aparte con `nowcast_input_ml_mean.py --cutoff-day N`.
