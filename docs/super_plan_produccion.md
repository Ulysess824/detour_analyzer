# Super plan: llevar el modelo a uso en producción

Lo que vale es la predicción del modelo; la comparación con el planificador es orientativa. Los sub-planes se hacen uno por uno, cada uno en su
rama, con pruebas y documentación, y se fusionan a `main` con confirmación.

| # | Sub-plan | Entregable | Termina cuando | Estado |
|---|---|---|---|---|
| 1 | Validación de entrada | `src/engine/data_checks.py`, `scripts/validate_input.py`, tablas en `results/validacion_*.csv` | Los datos malos se detienen con un mensaje claro y el maestro real pasa sin errores | Hecho (ver `docs/plan_validacion_entrada.md`) |
| 2 | Trazabilidad de cada corrida | Cada pronóstico guarda fecha, huella de datos, huella de parámetros y commit; logs; comprobación de la salida (sin NaN ni negativos, total en un rango razonable) | Dado un pronóstico se puede decir con qué datos y parámetros se hizo | Pendiente |
| 3 | Entorno reproducible | `requirements` separados (producción, afinación, app, desarrollo), versiones fijas, `pyproject`, CI que corra las pruebas en cada subida | Un entorno nuevo reproduce el pronóstico de octubre con diferencia 0,0 TO | Pendiente |
| 4 | Cobertura de pruebas del ajuste | Pruebas para `member_utils` (46%), `tuning_utils` (48%) y `evaluation_utils` (51%) | Esos módulos pasan de 70% de cobertura, comprobado con bugs inyectados | Pendiente |
| 5 | Seguimiento contra el real | Al cerrar cada mes, el pronóstico se compara con el real y el acierto se guarda en la base | Existe una tabla mensual de acierto por planta y total y una alerta si cae por debajo de un umbral | Pendiente |

Condicionales:

- **A.** Cablear el horizonte de dos meses, o el nowcasting con la media, en producción. Solo si la predicción se necesita antes de que cierre el mes anterior.
- **B.** Orden interno: ciclo `db_utils` / `panel_utils`, lógica de 5 scripts a `src/`, imports sin usar.
- **C.** Confirmar con el dueño de los datos los cambios de nivel de SALI, SALM y SPAL (no es código).

## Auditoría que origina el plan (resumen)

- Modularidad buena: tres capas verificadas por prueba, sin función de más de 60 líneas ni módulo de más de 280.
- Pronóstico reproducible: octubre de 2026 en 24 s, 2.247 series, diferencia 0,0 TO contra el guardado.
- Antes del sub-plan 1, la carga fallaba en silencio ante duplicados, minúsculas, espacios, negativos, un mes faltante o valores absurdos.
- Sin CI, sin versiones fijas, un solo `requirements`, sin trazabilidad de corridas; cobertura baja en `member_utils` y `tuning_utils`.
