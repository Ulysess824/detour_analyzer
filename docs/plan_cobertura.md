# Sub-plan 4: pruebas para el ajuste del modelo

Rama `feature/test-coverage`. Solo se agregaron pruebas, `coverage` a `requirements/dev.txt` y un piso de cobertura al CI: no se tocó ningún archivo
de `src/` ni de las pruebas existentes.

## Cobertura antes y después

| Módulo | Antes | Después |
|---|---|---|
| `engine/member_utils.py` | 48% | 100% |
| `evaluation/tuning_utils.py` | 48% | 99% |
| `evaluation/evaluation_utils.py` | 51% | 100% |
| `evaluation/search_space_utils.py` | 0% | 100% |
| `evaluation/eda_utils.py` | 0% | 100% |
| `evaluation/internal_utils.py` | 0% | 100% |
| `src/` completo | 87% | 96% |

Quedan por debajo de 85% solo `engine/transform_utils.py` (66%: la lectura del Excel ancho de SAP y `write_csv`) y `engine/run_manifest.py` (79%: los
caminos en que git falla). Suite: 164 pruebas (131 previas y 33 nuevas), unos 41 s.

## Pruebas nuevas

| Archivo | Qué comprueba |
|---|---|
| `tests/test_member_utils.py` (5) | Una fila por serie viva y mes de prueba; `y` coincide con el real; pronósticos finitos y no negativos; **cambiar los meses posteriores al origen no cambia ningún pronóstico de ese origen**; mismas entradas, mismos pronósticos; los parámetros afinados cambian solo a su modelo |
| `tests/test_evaluation_utils.py` (7) | Los reales de planta y total coinciden con el panel; la suma por SKU es el pronóstico de la planta y del total; una planta sin filas suma 0; las tablas por nivel se ordenan por acierto (100%, 90% y 0% para un modelo perfecto, uno 10% alto y uno en cero); sesgo negativo si el pronóstico es alto; los empates comparten posición; los errores que se compensan mejoran los niveles superiores |
| `tests/test_tuning_results.py` (8) | `rolling_test` reentrena antes de cada mes con meses ya conocidos (en h=1 y h=3) y una vez por semilla; los resúmenes coinciden con un cálculo a mano; `compare_base_and_tuned` devuelve h=1, h=3, naive y las corridas; `to_jsonable` convierte tipos de numpy y quita las claves privadas; el resultado se escribe como JSON; `study_summary` y `print_comparison` |
| `tests/test_search_space.py` (9) | Cada valor muestreado queda dentro del rango documentado, para los tres modelos; la configuración muestreada entrena un modelo real; el trial base cae dentro del espacio |
| `tests/test_eda_and_internal.py` (4) | Las tablas descriptivas contra un cálculo a mano (y la de planta suma lo de la de serie); código SAP a SKU con "gana el primer archivo"; el formato largo del modelo interno y las filas sin código conocido |

## Errores inyectados (en copias del código; todos detectados)

| # | Error | Detectado |
|---|---|---|
| 1 | `rolling_test` entrena con el mes objetivo (fuga) | Sí |
| 2 | `collect_member_forecasts` entrena con todos los datos | Sí |
| 3 | El real de la planta es el de otro mes | Sí |
| 4 | Ranking invertido | Sí |
| 5 | Promedio de semillas mal calculado | Sí |
| 6 | Un límite del espacio de búsqueda cambiado | Sí |
| 7 | `to_jsonable` deja tipos de numpy | Sí |
| 8 | Signo del sesgo invertido | Sí |

## Piso de cobertura en el CI

`.github/workflows/tests.yml` corre `coverage run --source=src -m pytest` y después `coverage report --fail-under=90`: si la cobertura de `src/` baja de 90%,
el trabajo falla (hoy está en 96%). Son dos pasos separados para que un fallo de las pruebas nunca quede tapado por el reporte. `coverage` 7.16.2 se agregó a
`requirements/dev.txt`. Si un cambio válido es difícil de probar y frena el CI, se baja el piso o se escribe la prueba.
