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
2. **Engine: hecha.** Piezas movidas a `engine` (`month_in_progress`, `month_state`, `strategies`), 23 pruebas nuevas, y las tres estrategias comparadas con la caché del backtest sobre los datos reales (diferencia máxima 0,00000 TO). Las pruebas encontraron un error que habría afectado la producción: `build_panel` termina el calendario en el último registro, así que el mes en curso aparecía con menos días de los que tiene; `complete_calendar` lo corrige.
3. **Script: hecha.** `run_prediction` y `predict_next_month.py` aceptan `--strategy` y `--cutoff-day`; el manifiesto guarda la sección `strategy`. Con el mes completo el pronóstico de octubre sale idéntico byte a byte. Un pronóstico que no es `h1` se guarda como `forecast_<mes>_<estrategia>.csv` para no reemplazar al de un mes de anticipación del mismo mes (la app todavía solo lista `forecast_AAAA-MM.csv`: fase 4).
4. **App backend: hecha.** `settings_service`, `GET/PUT /api/ajustes`, el pronóstico sigue los ajustes guardados (sin bloqueo con el mes en curso), los archivos `forecast_<mes>_<estrategia>.csv` se listan en Predicción, historial y estado, y la descarga de Excel acepta la estrategia. 12 pruebas nuevas y mutaciones capturadas. Ajustes no lleva clave.
5. **App pantallas: hecha.** Sección Ajustes (`views_ajustes.js`), "Ajuste de hiperparámetros" renombrada a "Hiperparámetros", Predicción con estrategia, tarjeta del mes en curso y referencia correcta, Inicio, Historial, Descargar y Ayuda con la estrategia. 5 pruebas de backend nuevas, mutaciones capturadas y capturas con Chromium sin errores de consola. La ventana real (WebView2) solo se prueba en Windows; la guía de instalación se corrige en la fase 7.
6. **Seguimiento: hecha.** Estrategia y corte en cada fila del historial y en la base (esquema 3), umbrales propios a dos meses, pruebas históricas de `directa`, `parcial` e `iterada`, medición y comparación por estrategia, pantallas de Seguimiento, Inicio, Historial y Descargar. 26 pruebas nuevas, mutaciones capturadas; con datos reales, el seguimiento de septiembre de 2026 en producción coincide con la prueba histórica de cada estrategia.
7. **Documentación: hecha.** README (tabla "Versiones" con la v3, sección "Estrategias a mitad de mes", app, trazabilidad, seguimiento, flujo y pendiente), `CLAUDE.md`, `docs/plan_app_escritorio.md` (fase 7) y la guía de instalación.
8. **Monte Carlo para el resto del mes: hecha (pedida por el usuario).** Ajuste `metodo_resto` (media por defecto) en la sección Ajustes, con el acierto de cada método sobre los días que faltan; la tarjeta "El mes en curso" usa el método elegido y muestra el total con el otro; la estrategia `nowcast_media` siempre usa la media. No cambia ningún pronóstico. 8 pruebas nuevas y mutaciones capturadas.

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

## 10. Plan de la fase 5: pantallas

Objetivo: que la estrategia se vea y se cambie desde la app. Hoy el backend ya funciona (fase 4) pero las pantallas siguen con la lógica de un solo mes: el botón de Predicción se bloquea con el mes en curso y todo habla del "mes siguiente". Archivos que cambian: `src/app/static/` (`index.html`, `app.js`, `views_forecast.js`, `views_extra.js`, nuevo `views_ajustes.js`, `app.css`), `status_service.py`, `forecast_service.py` y el texto de `data_service.py`.

### 10.1 Cambios de backend que las pantallas necesitan (5a)

| Cambio | Para qué |
|---|---|
| `status_service.app_status` devuelve `plan` (estrategia, corte, mes de origen, mes pronosticado, avisos, error) y `proximo_pronostico` pasa a ser el mes que pronostica ese plan | Predicción e Inicio dejan de calcular el mes siguiente por su cuenta |
| `forecast_detail` agrega `mes_en_curso`: mes, corte, días vistos y restantes, real hasta el corte, estimación del resto con la media y total estimado, para la planta y el SKU elegidos; solo cuando la estrategia usa el mes en curso (`parcial`, `nowcast_media`) | La tarjeta "Mes en curso" de Predicción. Se calcula con `complete_calendar`, `seen_days` y `forecast_media` del engine, sin duplicar lógica |
| `forecast_detail` indica si el mes de referencia es el anterior al pronóstico o el último mes completo | Rotular bien "Contra el último mes real" cuando el mes anterior sigue abierto |
| Texto del aviso `mes_parcial` de la carga de datos | Hoy dice que no se podrá pronosticar hasta que llegue el mes completo; ya no es cierto |

### 10.2 Sección nueva "Ajustes" (5b)

Entrada del menú: **Ajustes** (id `ajustes`), justo antes de **Hiperparámetros**; subtítulo "Cómo se calcula el pronóstico". La sección que hoy se llama "Ajuste de hiperparámetros" pasa a llamarse **Hiperparámetros** (menú, título y textos de la app, de la Ayuda y de la documentación) para que los dos nombres no se confundan; su id y su comportamiento no cambian. Archivo `views_ajustes.js`. Tarjetas, de arriba abajo:

1. **Estado de los datos y plan:** último registro, si el mes está completo o en curso, día de corte, días vistos y restantes; qué estrategia correría ahora, de qué mes parte y qué mes pronostica. Los avisos del plan (por ejemplo "hay solo 7 días, se usa la directa") y el error del corte manual se muestran en un recuadro de aviso.
2. **Estrategia:** una lista de opciones con su nombre, la explicación y el acierto (SKU, planta, total) en etiquetas; "Automática" lleva la etiqueta "Recomendada". Si el mes está completo, las opciones que necesitan el mes en curso muestran la nota "Con el mes completo se usa un mes de anticipación" (se pueden elegir igual: valen para la próxima vez).
3. **Día de corte:** Automático (muestra el día detectado) o Manual (número entre 1 y el día detectado). Aviso si queda bajo el día 10. Deshabilitado con el mes completo.
4. **Gráfico "Cuánto mejora según el día de corte":** acierto por SKU, planta y total de `parcial` del día 5 al 28, con líneas de referencia de la directa y del pronóstico a un mes, marca en el día 10 y en el corte actual. Botón de imagen PNG como en las otras pantallas.
5. **Guardar** y **Volver a automático**. El botón de guardar se activa solo cuando hay cambios; al guardar se vuelve a pedir `/api/ajustes` y se actualiza el plan. Errores 422 en un recuadro.

### 10.3 Predicción (5c)

- **Tarjeta de generación:** título "Pronóstico de <mes del plan>", con la estrategia, el corte y el mes de origen; los avisos del plan; enlace "Cambiar en Ajustes". El botón solo se bloquea si el plan tiene error o no hay datos (ya no por el mes en curso). La confirmación para reemplazar es por mes y estrategia.
- **Selector de pronóstico:** lista "mes · estrategia" (`disponibles_detalle`), con la estrategia en español y el corte.
- **Tarjeta "Mes en curso"** (solo `parcial` y `nowcast_media`): real hasta el día de corte, estimación del resto con la media (marcada como estimación) y total estimado del mes, para la planta y el SKU elegidos.
- **Tarjeta "Contra el último mes real":** rotula el mes de referencia y avisa que el mes anterior sigue en curso cuando es el caso.
- **Gráfico:** el mes en curso aparece sombreado como parcial (ya existe) y el pronóstico queda en el mes siguiente.

### 10.4 Inicio, Historial, Descargar y Ayuda (5d)

- **Inicio:** la tarjeta "Próximo pronóstico" muestra mes y estrategia; "listo" cuando el último pronóstico guardado coincide con el plan.
- **Historial:** columna "Estrategia"; "Ver" y "Excel" por mes y estrategia.
- **Descargar:** el selector de pronóstico incluye la estrategia.
- **Renombrar:** "Ajuste de hiperparámetros" pasa a "Hiperparámetros" en el menú (`app.js`), el título de la pantalla (`views_tuning.js`) y los textos que la nombran.
- **Ayuda:** corregir "Mes parcial" y "No me deja generar el pronóstico"; entradas nuevas para "Estrategia de pronóstico" y "Día de corte". La guía de instalación se corrige en la fase 7.

### 10.5 Verificación (5e)

- **Pruebas de backend** (≈12): `plan` en el estado, `mes_en_curso` igual a un cálculo manual (real hasta el corte, media del resto, total) para todo, una planta y un SKU, la referencia del último real, el texto de la carga de datos.
- **Mutaciones** en el cálculo de `mes_en_curso` y en el plan del estado.
- **Navegador sin ventana** (Chromium con Playwright, como en las fases anteriores): capturas de Ajustes con el mes completo, con el mes en curso, con un corte manual inválido y con la opción "nowcast"; Predicción con `h1`, `parcial` y `nowcast_media`; Inicio e Historial. Se revisa que no haya errores en la consola y que nada se desborde en anchos de 1100 y 1400 px. Las capturas se te envían.
- **Sintaxis de JavaScript** con `node --check` en todos los archivos.
- **Windows:** la ventana real (WebView2) solo se puede probar en tu equipo o en el CI de Windows.

### 10.6 Decisiones que tomo salvo que digas otra cosa

1. La sección nueva se llama "Ajustes" y va antes de "Hiperparámetros" (nombre nuevo de la sección que se llamaba "Ajuste de hiperparámetros"). Confirmado.
2. La tarjeta "Mes en curso" solo aparece con `parcial` y `nowcast_media`. Confirmado.
3. Las opciones que necesitan el mes en curso se pueden elegir aunque el mes esté completo (valen para la próxima vez). Confirmado.
4. El gráfico por día de corte usa los números ya medidos (constantes del módulo), no recalcula nada. Confirmado.

## 11. Plan de la fase 6: seguimiento por estrategia

Objetivo: poder medir contra el real cada pronóstico guardado según la estrategia con que se hizo, y comparar las estrategias entre sí con el consumo real. Hoy el seguimiento solo mide el pronóstico de un mes (`h1`): `to_measure`, `measure`, `top_errors` y el Excel suponen el archivo `forecast_<mes>.csv`, y `tracked_months` da por medido un mes en cuanto se mide cualquiera de sus pronósticos. Los pronósticos de mitad de mes (`forecast_<mes>_<estrategia>.csv`) ya se guardan y se listan, pero no se pueden medir.

### 11.1 Qué hay hoy y qué cambia

| Pieza | Hoy | Cambio |
|---|---|---|
| Filas del historial (`results/seguimiento.csv`) | `run_id, mes, nivel, planta, modelo, acierto, wape, bias_pct, n, real_TO, pronostico_TO, alerta, creado_en` | Dos columnas al final: `estrategia` y `corte` (día de corte, vacío si no aplica). Las filas viejas se leen como `h1` (todo lo anterior a la fase 3 lo fue) |
| Base de datos (`seguimiento`, `corridas`) | esquema 2 | Esquema 3: `estrategia` en las dos tablas y `corte` en `corridas`, con migración automática de una base en esquema 2 (`ALTER TABLE`, sin perder filas) |
| Estrategia de un pronóstico guardado | no se guarda | Sale del manifiesto (`strategy.used`, `strategy.cutoff_day`); si no hay manifiesto, del nombre del archivo; si tampoco, `h1` |
| Identidad de lo medido | el mes | el par (mes, estrategia); volver a generar un pronóstico ya medido no lo ofrece otra vez, como hoy |
| Umbrales de alerta | los mismos para todo (SKU 64, planta 85, total 90) | `h1` igual; las estrategias a dos meses con umbrales propios más bajos (propuesta en 11.5) |
| Prueba histórica | solo `backtest-h1` (12 meses) | Además `backtest-directa`, `backtest-parcial` (corte 15) y `backtest-iterada`, tomadas de `results/horizon_strategies.csv` (las mismas 12 pruebas de la fase 1); sirven de referencia para juzgar los meses reales. `nowcast_media` no tiene prueba guardada |

### 11.2 Subpasos

**6a. Datos y motor** (`engine`, `evaluation`)
- `TRACKING_COLUMNS` con `estrategia` y `corte`; `tracking_rows` los recibe; `read_tracking_csv` rellena lo que falte; `write_tracking_csv` conserva la clave (run_id, mes).
- `db_utils`: esquema 3 y migración; `upsert_run` y `replace_tracking` con las columnas nuevas.
- `tracking_run.strategy_of(forecast_path, manifest)` y `thresholds_for(strategy)`; `track_saved_forecast` guarda la estrategia y usa los umbrales que le tocan.
- `track_backtest` generalizado: `backtest_forecasts` para el caché de `h1` y uno nuevo para el caché de estrategias a dos meses (series vivas en el mes de origen; en `parcial` e `iterada` solo `ml_mean` y `naive`, porque el caché no guarda los tres modelos).
- `scripts/track_forecast.py`: muestra la estrategia al medir y `--backtest` carga las cuatro pruebas históricas.

**6b. Servicios de la app** (`tracking_service`, `history_service`, `status_service`, `export_service`, `forecast_service`)
- `to_measure` devuelve pares (mes, estrategia) de pronósticos guardados cuyo mes ya cerró y que no se midieron.
- `measure(mes, estrategia)` usa el archivo de esa estrategia; `top_errors` y `detail` aceptan la estrategia (en `detail`, una corrida por mes y estrategia: la última guardada).
- `history` agrega `estrategia` a cada punto y un bloque `comparacion`: por estrategia, meses medidos en producción, acierto medio por SKU, planta y total, y el de su prueba histórica.
- API: `POST /api/seguimiento/medir` con `estrategia` (por defecto `h1`), `GET /api/seguimiento/detalle?mes&estrategia`, Excel de seguimiento con la estrategia en el nombre y el resumen.
- Estado de Inicio: el aviso de "ya cerró" y la tarjeta de último seguimiento dicen la estrategia; el historial de pronósticos marca "medido" por (mes, estrategia).

**6c. Pantallas**
- **Seguimiento:** avisos por (mes, estrategia) con su botón "Medir acierto"; selector de estrategia sobre el gráfico (producción con rombos y prueba histórica de esa estrategia con círculos); tarjeta nueva "Comparación de estrategias" (tabla con medias en producción y en la prueba histórica); el detalle indica estrategia y corte y, cuando el mismo mes tiene varias estrategias medidas, una mini tabla "Mismo mes, otras estrategias".
- **Inicio e Historial:** el estado "Por medir" y "Medido" por estrategia.
- **Ayuda:** una entrada sobre cómo se mide cada estrategia y por qué no se comparan a ciegas.

### 11.3 Cuidados al comparar

- Un pronóstico de mitad de mes y uno de un mes anterior al mes pronosticado tienen información distinta (el segundo ve un mes más completo): la comparación dice qué es cada uno y no los ordena como si fueran iguales.
- Cada estrategia se evalúa sobre sus propias series (las vivas en su mes de origen); la tabla muestra cuántas series entran.
- Con pocos meses medidos las medias no son concluyentes: la tabla lo dice cuando hay menos de 3 meses.

### 11.4 Verificación (6d)

- **Equivalencia con lo medido:** el acierto por SKU de cada mes de `backtest-directa`, `backtest-parcial` e `backtest-iterada` coincide con la tabla por mes de la fase 1 (por ejemplo, octubre de 2025: directa 64,0%, iterada 62,9%, parcial 66,3%) y el de `backtest-h1` no cambia.
- **Migración:** una base en esquema 2 con datos pasa a 3 sin perder filas; un CSV viejo sin las columnas se lee como `h1`.
- **Pruebas** (≈20): estrategia desde el manifiesto, desde el nombre y por defecto; (mes, estrategia) en `to_measure` y `measure`; dos estrategias del mismo mes que se miden por separado y no se pisan; umbrales por estrategia; `comparacion`; endpoints y Excel; estado de Inicio; el CSV de producción no se contamina con pronósticos de pruebas (la guarda que ya existe).
- **Mutaciones** en la identidad (mes, estrategia), los umbrales y la lectura de filas viejas.
- **Capturas** de Seguimiento con un mes medido por dos estrategias, con avisos pendientes y con solo prueba histórica; sin errores de consola.

### 11.5 Decisiones que tomo salvo que digas otra cosa

1. **Umbrales a dos meses:** SKU 60, planta 83 y total 87 para `directa`, `parcial`, `nowcast_media` e `iterada` (la misma distancia que hoy hay entre `h1` y sus umbrales, aplicada a lo que acierta la directa); `h1` no cambia. Si quieres los mismos umbrales para todos, dímelo.
2. **Pruebas históricas por estrategia:** se cargan `directa`, `parcial` (corte 15) e `iterada`; `nowcast_media` queda sin prueba histórica hasta que se corra y se guarde.
3. **Gráfico de Seguimiento:** por defecto la estrategia del último pronóstico medido en producción; si no hay, `h1`.
4. **Lo ya medido:** las filas de producción anteriores se consideran `h1`.
5. **Estimación del mes en curso:** no se mide (decidido). El seguimiento mide solo los pronósticos guardados, cuando cierra el mes pronosticado.

Decisión sobre la fase 8: queda como opcional (decidido); no se construye salvo que se pida.

