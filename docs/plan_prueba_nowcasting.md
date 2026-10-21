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
- Día de corte del mes en curso: **automático** por defecto (el último día con consumo en los datos, sección 4); se puede fijar a mano.
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
| Mes en curso con `corte` dentro del rango validado | `parcial` (o la estrategia elegida en Ajustes) con ese `corte` | El siguiente al mes en curso |
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
1. **Validación previa, curva por día de corte:** correr `compare_horizon_strategies.py --cutoff-day N` (el script ya acepta el corte y guarda la caché por corte) y `nowcast_media_2026.py --cutoffs` con N = 5, 8, 10, 12, 15, 18, 20, 22, 25 y 28. Resultado: acierto de `parcial`, `nowcast_media` y `directa` en función del día. De ahí salen el **rango validado** (desde qué día `parcial` supera a `directa` en la mayoría de los 12 meses) y si algún corte alto se acerca a `h1`. Hasta tener esa curva, el único corte respaldado es el 15.
2. **Engine:** mover las piezas de la sección 5, crear `strategies.py`, tests de equivalencia y de no fuga.
3. **Script:** `--strategy`, `--cutoff-day`, manifiesto.
4. **App backend:** servicio de ajustes y endpoints, `forecast_service` con la estrategia.
5. **App pantallas:** Ajustes y los cambios en Predicción.
6. **Seguimiento:** estrategia por corrida.
7. **Documentación:** README (incluida la tabla "Versiones", que hoy dice que la v2 es el commit `2876c4f`), `CLAUDE.md`, `docs/plan_app_escritorio.md`, `docs/guia_instalacion.md`.
8. **Opcional:** Monte Carlo historia + mes como método del resto del mes; empata con la media, así que no entra salvo que se decida.

Criterio de cierre: CI verde en Linux y Windows, el pronóstico a h=1 sin cambios, y las cifras de la sección 2 reproducidas dentro de 0,05 puntos.

## 8. Decisiones

Tomadas:

1. "Ajustes" no lleva clave; la estrategia queda registrada en cada corrida.
2. El modo automático prefiere `parcial` (directa + días del mes en curso).
3. No se ofrece un corte manual sin validar; el corte se detecta del último registro de los datos y solo se usa dentro del rango validado (fase 1). Antes de la fase 1 solo está respaldado el corte 15.

Pendientes:

- Si se permite fijar el corte a mano en Ajustes (útil si el exporte termina en feriado) o solo ver el detectado.
- Qué hacer si el corte detectado supera el rango validado por arriba (por ejemplo día 28): usar `parcial` igual o esperar al mes completo.
