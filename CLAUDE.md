# detour_analyzer: agent index

Read this first; open source files only for the function you need to change. README.md (Spanish) has the user-facing docs and result tables.

**Goal:** forecast monthly consumption of paper SKUs per planta from SAP exports, aggregated bottom-up to planta and total.
**Rules:** chronological splits and rolling origin only (never random). Code and comments in English. Scripts in `scripts/` are thin CLIs (`sys.path.insert` + argparse); logic goes in `src/utils/`, one responsibility per module. Run from the repo root.
**Metrics:** WAPE = sum|y-p| / sum y; accuracy = 1 - WAPE; bias_pct > 0 means under-forecast.
**Env:** `python` is not on PATH in the Bash tool (Windows Store alias); use PowerShell or locate the interpreter.
**Do not:** reintroduce neural nets (removed as too heavy, c2e3330); edit files when the user pastes a code chunk, answer with the corrected chunk in chat.

## Data
`data/consumos_long.csv` (master): `planta, sku, fecha, consumo`; 347,502 rows, 15 plantas, 2024-01-02..2026-09-30 (33 months, last complete month 2026-09). `sku` = material description. Negatives clipped to 0. Built from `consumos_2024_2025.xlsx` + `consumos_2025_10_2026_09.xlsx` (both long) with `--drop-zeros`; the two overlap in 2025-10..12 with identical values, `combine_exports` keeps them once and stops only if values differ. `consumos_2026.xlsx` (wide, to 2026-06) and `consumos_2024_ene.xlsx` are superseded and no longer in the repo's flow.
Planner forecasts: `data/planner_forecast_<YYYY-MM>.csv` (`planta, mes, sku_planner, product_sap, strategy, forecast_to` in TO), SCAN only, months 2026-06, 2026-07, 2026-09 (August missing). June has no strategy column in the source; it was filled from the July file by product. `sku_planner` is `grade/subgrade/gsm/width` without core width.

## Pipeline: script -> utils it uses -> output
| script | main utils | output |
|---|---|---|
| transform_consumos.py | transform_utils (`read_export`, `combine_exports`, `write_csv`) | data/consumos_long.csv |
| eda_consumos.py | eda_utils (`eda_by_planta_sku`, `eda_by_planta`) | data/eda_*.csv |
| compare_models.py (`--cutoff`) | row_model_utils, metrics_utils (`row_metrics`, `rank_by_mae`) | console; daily-row models: naive, Croston, hurdle, LGBM/XGB Tweedie, ARIMA |
| forecast_monthly.py (`--horizons`, `--daily`) | evaluation_utils (`evaluate`, `tercile_table`), tree_utils | console |
| tune_trees.py | tuning_utils, search_space_utils, tree_utils | results/tuning_trees.json |
| build_tuning_viewer.py | viewer_utils + src/utils/templates/tuning_viewer.html | results/tuning_viewer.html (not committed) |
| plot_pred_vs_real.py | plot_utils | PNG |
| compare_planner.py (`--month`) | planner_utils, ensemble_utils | results/planner_comparison_<month>.csv (planner vs models, one month) |
| run_ensembles.py (`--refit`) | member_utils, ensemble_utils, econometric_utils, mcs_utils, evaluation_utils, cache_utils | results/member_forecasts.csv (cache) + `.meta.json`, results/ensembles.json |

## Core objects
- `Panel` (panel_utils): arrays series x month: `monthly` (NaN before first appearance), `active_days`, `daily` (zero-filled), `planta_monthly`, `total_monthly`, `business_days` (no Sundays), `labels`, `first_month`, `planta_code`. Built by `build_panel(load_consumption(csv))`. Window helpers: `window_mean/std/rate`.
- `make_frame(S, horizon)` (feature_utils): one row per (origin `o`, series alive); target month `t = o + h`; columns `series, o, t, planta_code, y`, the 25 `SERIES_FEATURES`, and baseline forecasts `p_<name>` (`BASELINES`). All windows end at the origin: no leakage. `direct_forecasts` applies the simple models to an aggregated series.
- `fit_predict_trees(kind, params, train, test, seed)` (tree_utils): `kind` in lgbm / xgb / rf; LGBM/XGB use Tweedie with early stopping on the last 2 target months; `TREE_BASE` holds the pre-tuning params. `daily_lgbm_forecast` = daily-then-aggregate.
- `reconcile` (evaluation_utils): rescales SKU forecasts to the direct planta `mean6_x_seasonal`. `score_models_by_level` ranks models at sku/planta/total.
- Tuning: `make_folds` (3 walk-forward folds x 3 months), `run_study` (TPE, baseline enqueued as trial 0), `compare_base_and_tuned` (h=1 and h=3 on the test months), `to_jsonable`.
- Ensembles: `ENSEMBLE_GROUPS` = ml (lgbm, xgb, rf), econ (naive, mean3/6/12, perday6, seasonal_naive, mean6_x_planta, ses, damped_holt, arima), ml_econ. `METHODS` = mean, median, trimmed (20%), weighted (1/past MAE, only months <= t-h). 25 models total. `model_confidence_set(loss, period, block, n_boot)` = Hansen-Lunde-Nason, block bootstrap over months.

## Guards and tests
- `cache_utils`: `panel_fingerprint` (hash of series, months and values, independent of line endings), `expected_meta`/`check_meta` (the cache `results/member_forecasts.meta.json` must match data, horizon, test window and tuned params, else stop with "--refit"), `check_alignment` (the series index stored in a cache must reproduce the panel's actuals).
- The series index is the position in the alphabetical (planta, sku) list (`Panel.keys`); adding one SKU to the master shifts it, so a stale cache is rejected instead of silently misread.
- `tuning_utils.tuning_meta` is saved as `meta` inside each model of `results/tuning_trees.json` (windows, folds, data hash); `check_tuning_window` rejects tuned parameters whose tuning window reaches the test months (leak); `run_study` refuses to resume an Optuna `--storage` study fitted to other data. The viewer takes its windows and folds from `meta`.
- Tests: `python -m pytest -q` (synthetic data, about 5 s): no leakage in features, panel and daily sums, ensemble weights use only known months, reconcile totals, metric sign, tuning folds, cache and tuning guards, transform overlap rules, planner SKU matching, tree determinism, MCS. The suite was checked by mutation: 8 injected bugs were all caught.

## Dashboard (app/, Streamlit, read-only)
Run from the repo root: `python -m streamlit run app/dashboard.py` (use the Python 3.12 path above). Never retrains; reads `results/` and `data/`. UI text in Spanish, no emojis, retrofuturistic theme.
- `dashboard.py`: header, sidebar (level selector, fixed horizon h=1), best-model cards, four tabs.
- `theme.py`: palette, fonts, CSS, `plotly_layout`, `add_real_trace` and `add_model_trace`. Every chart that has an actual series must draw it with `add_real_trace` (thick amber line with glow, added after the models) so the real line always stands out. `loaders.py`: cached readers, `MODEL_GROUPS` (ML, classical econometrics, baselines, ensembles); ensembles are computed with `add_ensembles` on `member_forecasts.csv`.
- `planner.py`: tab PLANIFICADOR. Planner forecast vs real vs the sidebar models for the SCAN SKUs of one month, read from `results/planner_comparison_<month>.csv` (written by `scripts/compare_planner.py`, logic in `src/utils/planner_utils.py`). Selectors: month (or all months together) and strategy (VMI / NO VMI). One month: SKU chart with the real as amber markers (not a line, since SKUs are not a time axis) and the planner as magenta diamonds. All months: accuracy per month, planner as a thick magenta line.
- `real_vs_pred.py`: tab REAL VS PREDICHO. Planta and SKU are searchable selectboxes between the cards and the tabs ("Todas"/"Todos" = no filter, so no scope radio); the model pills live in the fixed sidebar. `ranking.py`: tab RANKING (table by rank from `ensembles.json`).
- Done: stages 1 to 3 (table only, no bar chart yet). Pending: stage 3 bar chart, stage 4 (CONJUNTO DE CONFIANZA, MCS), stage 5 (tuning tab), stage 6 (polish, missing-file messages, app/README).

## Latest results (test 2025-10..2026-09, h=1)
SKU ~70%, planta ~92%, total ~97%. Best: ml_mean (SKU 70.4%, total 97.5%), ml_median (planta 91.9%). MCS 90%: 4 of 25 stay at SKU level (LGBM, XGB and all classical models are discarded), 11 at planta, 17 at total. Optuna gain is small and not consistent (only Random Forest improves SKU accuracy, 68.6% to 69.9%). `ml_trimmed` == `ml_mean` (3 members).
Planner vs models (SCAN, 2026-06/07/09, 122 SKU-months, `compare_planner.py`): planner accuracy 81.3% (rank 24 of 26, bias -4.7%, over-forecast in 76 of 122), ml_weighted/ml_mean 85.3%, lgbm 85.1%. The planner beat ml_mean in 44 of 122 SKU-months. By month: 81.1% / 82.0% / 80.6% vs ml_mean 83.8% / 86.0% / 86.1%. VMI SKUs (96, 97% of the volume): planner 81.9% vs ml_mean 86.2%. NO VMI SKUs (26, tiny volume): planner 61.3% vs ml_mean 57.1%. Three months and one planta: a reference and not a proof. Unknown whether "Plant Forecast (TO)" is expected consumption or a production plan with buffer.

## Open items
- [ ] Inference script: train on full history, predict next month (none in scripts/).
- [ ] `data/consumos_long.csv` can show modified in `git status` with an empty `git diff` on Windows (line endings); check before committing.
- [ ] Planner comparison covers SCAN and 3 months (no August). New months: add `data/planner_forecast_<month>.csv` and run `compare_planner.py --month <month>`; the tab finds the files by name.
- [ ] Branches: `main` has the data and scripts, `feature/dashboard` has `app/`; not merged yet.
- [ ] Not tested yet: the dashboard (`app/`), the row-level models (`compare_models.py`) and the MCS size problem (see below).
- [ ] MCS with 12 test months and 25 models is oversized (simulation: it excludes about 6 of 25 truly equal models); only large loss gaps are reliable. Winner selection is done on the same test months (selection bias).
- [ ] No check that the last month of a new export is complete.

Update this file when a script, module or result changes.
