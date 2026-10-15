# Bibliografía

Papers que respaldan el experimento de completar el resto del mes por simulación (ver `docs/plan_montecarlo_mes_parcial.md`).
Los PDF no se guardan en git (`bibliografia/.gitignore`): se descargan con

```bash
python scripts/download_bibliografia.py          # todos los disponibles
python scripts/download_bibliografia.py --list   # muestra clave, archivo y enlaces
```

El script solo usa copias de acceso abierto. Si el servidor está bloqueado por la política de red del entorno, lo informa y sigue.

| Clave | Referencia | Para qué se usa | Descarga |
|---|---|---|---|
| `giannone2008` | Giannone, Reichlin y Small (2008), *Nowcasting: the real-time informational content of macroeconomic data*, Journal of Monetary Economics 55(4), 665-676 | Usar datos incompletos del período en curso con un modelo de factores y el filtro de Kalman. No usa simulación de Monte Carlo | Script (versión de trabajo; la versión de la revista es de pago) |

## Referencias citadas en el plan y todavía sin descarga automática

Todas se encontraron por búsqueda (resúmenes); hay que leer el original antes de citarlas.

| Referencia | Para qué se usa |
|---|---|
| Rubin (1987), *Multiple Imputation for Nonresponse in Surveys*, Wiley | Rellenar con varios sorteos y combinar los resultados. Un solo valor subestima la incertidumbre. Es un libro: no hay copia abierta |
| Willemain, Smith, Smart y Schwarz (2004), *Forecasting intermittent demand in manufacturing: a comparative evaluation of Croston's method*, International Journal of Forecasting 20(3) | Simulación de demanda intermitente (Markov para los ceros y *bootstrap* para los tamaños) |
| Gardner y Koehler (2005), comentario a Willemain et al., International Journal of Forecasting 21(3) | Crítica al método anterior |
| Syntetos, Babai y Gardner (2015), *Forecasting intermittent inventory demands: simple parametric methods vs. bootstrapping*, Journal of Business Research 68(8) | Los métodos paramétricos simples rinden bien frente al *bootstrap* |
| Ghysels, Santa-Clara y Valkanov (2004), *The MIDAS touch: mixed data sampling regression models*, CIRANO 2004s-20 | Usar datos diarios directamente en un modelo mensual, sin rellenar |
| Bańbura, Giannone, Modugno y Reichlin (2013), *Now-casting and the real-time data flow*, Handbook of Economic Forecasting | Actualización del pronóstico cuando llega cada dato nuevo |
| Marcellino, Stock y Watson (2006), *A comparison of direct and iterated multistep AR methods for forecasting macroeconomic time series*, Journal of Econometrics 135 | Directa contra iterada (usar un pronóstico como dato observado) |
| Ben Taieb, Bontempi, Atiya y Sorjamaa (2012), *A review and comparison of strategies for multi-step ahead time series forecasting*, Expert Systems with Applications 39(8) | Lo mismo, con más estrategias |

Para añadir una referencia descargable, agrégala a `REFERENCES` en `src/utils/bibliography_utils.py`.
