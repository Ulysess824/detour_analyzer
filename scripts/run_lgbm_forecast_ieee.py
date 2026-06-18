"""Tune LGBMRegressor on the real data, forecast a random 2026 plant/SKU and plot it (IEEE style).

Pipeline
--------
1. Transform the raw ERP export, restrict to the most active series and a recent window.
2. Build leakage-safe features with FeatureEngineBuilder.
3. Tune the LGBMRegressor wrapper (n_estimators / learning_rate / max_depth) with Optuna on a
   chronological inner validation split, then refit on the full train window.
4. Pick a random (planta, sku) present in the 2026 test window with select_random_group and
   forecast it.
5. Render a professional IEEE-style figure (serif fonts, double-column size, 300 dpi) showing
   the real vs predicted next-day consumption, the peak threshold, and the evaluation metrics
   embedded in the legend.
"""

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import optuna
import polars as pl
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.feature_engine_features import FeatureEngineBuilder
from src.models.lgbm_regressor import LGBMRegressor
from src.utils.data_transformation import transform_data
from src.utils.regression_metrics import RegressionMetrics
from src.utils.selection import select_random_group

DATA_PATH = ROOT / "data" / "historico_consumo.parquet"
MIN_DATE = pl.date(2024, 6, 1)
VAL_CUTOFF = pl.date(2026, 1, 31)   # inner: train <= this, validation in (this, TEST_CUTOFF]
TEST_CUTOFF = pl.date(2026, 3, 31)  # train <= this, test > this (Apr-May 2026)
N_SERIES = 400
N_TRIALS = 30
MIN_TEST_POINTS = 8                 # a plottable series needs enough test observations
SEED = None                         # None -> a different random series each run


def load_panel() -> pl.DataFrame:
    """Transform the raw export, keep the most active series and add calendar features."""
    df = transform_data(pl.read_parquet(DATA_PATH), umbral_pico=2.0)
    df = df.filter(pl.col("fecha") >= MIN_DATE)

    activity = (
        df.group_by(["planta", "sku"])
        .agg(pl.col("consumo_real").sum().alias("total"))
        .sort("total", descending=True)
        .head(N_SERIES)
        .select(["planta", "sku"])
    )
    df = df.join(activity, on=["planta", "sku"], how="inner")

    return df.with_columns(
        pl.col("fecha").dt.day().alias("day_of_month"),
        (pl.col("fecha").dt.month_end() - pl.col("fecha"))
        .dt.total_days()
        .alias("days_to_end_of_month"),
    ).sort(["planta", "sku", "fecha"])


def build_features(panel: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """Fit the feature builder on the train window and transform the full panel (no leakage)."""
    builder = FeatureEngineBuilder(
        value_cols="consumo_real",
        group_cols=("planta", "sku"),
        date_col="fecha",
        lags=(1, 2, 3, 5, 7),
        windows=(3, 5, 10),
        window_functions=("mean", "std", "max"),
        add_expanding=True,
        expanding_functions=("mean",),
    )
    builder.fit(panel.filter(pl.col("fecha") <= TEST_CUTOFF))
    df_feat, lag_cols = builder.transform(panel)
    feature_cols = lag_cols + ["daily_forecast", "day_of_month", "days_to_end_of_month"]
    return df_feat, feature_cols


def tune_lgbm(df_feat: pl.DataFrame, feature_cols: list[str]) -> dict:
    """Optuna search over the wrapper hyperparameters, scored by MAE on a chronological holdout."""
    inner_train = df_feat.filter(
        (pl.col("fecha") <= VAL_CUTOFF) & pl.col("target_consumo_real").is_not_null()
    )
    inner_val = df_feat.filter(
        (pl.col("fecha") > VAL_CUTOFF)
        & (pl.col("fecha") <= TEST_CUTOFF)
        & pl.col("target_consumo_real").is_not_null()
    )
    y_val = inner_val["target_consumo_real"].to_numpy()

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
        }
        model = LGBMRegressor(**params)
        model.fit(inner_train, x_predictor=feature_cols, y="target_consumo_real")
        preds = model.predict(inner_val)
        return float(np.mean(np.abs(y_val - preds)))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    print(f"Best inner-validation MAE: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study.best_params


def set_ieee_style() -> None:
    """Apply IEEE-paper matplotlib defaults: serif fonts, thin rules, tight high-dpi layout."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.3,
            "grid.linewidth": 0.4,
            "grid.alpha": 0.4,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def plot_forecast_ieee(
    series: pl.DataFrame,
    y_pred: np.ndarray,
    metrics: RegressionMetrics,
    plant: str,
    sku: str,
    save_path: Path,
) -> None:
    """IEEE-style figure: real vs predicted next-day consumption with the metrics in the legend."""
    set_ieee_style()
    fechas = series["fecha"].to_numpy()
    y_true = series["target_consumo_real"].to_numpy()
    threshold = (2.0 * series["daily_forecast"]).to_numpy()

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.plot(fechas, y_true, color="#1a1a1a", linewidth=1.4, label="Real (t+1)")
    ax.plot(
        fechas, y_pred, color="#c1121f", linewidth=1.4, linestyle="--",
        marker="o", markersize=2.5, label="LightGBM forecast",
    )
    ax.plot(
        fechas, threshold, color="#2a6f97", linewidth=1.0, linestyle=":",
        label="Peak threshold (2x daily forecast)",
    )

    ax.set_title(f"Next-day consumption forecast - {plant} / {sku}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Consumption (TO)")
    ax.grid(True, which="major")
    ax.margins(x=0.01)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

    # Primary legend: the plotted series.
    series_legend = ax.legend(loc="upper left", framealpha=0.9, edgecolor="0.3")
    ax.add_artist(series_legend)

    # Secondary legend: evaluation metrics as text-only entries (handlelength=0).
    metric_text = [
        f"MAE  = {metrics.mae:.3f}",
        f"RMSE = {metrics.rmse:.3f}",
        f"WAPE = {metrics.wape:.3f}",
        f"MAPE = {metrics.mape:.3f}",
        f"R$^2$  = {metrics.r2:.3f}",
        f"n    = {metrics.n_obs:_}",
    ]
    proxies = [Line2D([], [], linestyle="none") for _ in metric_text]
    ax.legend(
        proxies, metric_text, loc="upper right", title="Evaluation metrics",
        handlelength=0, handletextpad=0, framealpha=0.9, edgecolor="0.3",
        fontsize=7.5, alignment="left",
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Figure saved to: {save_path}")


def main() -> None:
    warnings.filterwarnings("ignore")
    print("Loading and transforming real data...")
    panel = load_panel()
    df_feat, feature_cols = build_features(panel)

    df_train = df_feat.filter(
        (pl.col("fecha") <= TEST_CUTOFF) & pl.col("target_consumo_real").is_not_null()
    )
    df_test = df_feat.filter(
        (pl.col("fecha") > TEST_CUTOFF) & pl.col("target_consumo_real").is_not_null()
    )
    print(f"Train rows: {df_train.height:_}  |  Test rows: {df_test.height:_}  "
          f"|  Features: {len(feature_cols):_}")

    print("\nTuning LGBMRegressor with Optuna...")
    best_params = tune_lgbm(df_feat, feature_cols)

    print("\nRefitting tuned model on the full train window...")
    model = LGBMRegressor(**best_params)
    model.fit(df_train, x_predictor=feature_cols, y="target_consumo_real")

    # Pick a random series that has enough test points in 2026 to plot.
    plottable = (
        df_test.group_by(["planta", "sku"]).len().filter(pl.col("len") >= MIN_TEST_POINTS)
    )
    test_pool = df_test.join(plottable.select(["planta", "sku"]), on=["planta", "sku"], how="inner")
    series = select_random_group(test_pool, seed=SEED).sort("fecha")
    plant = series["planta"][0]
    sku = series["sku"][0]
    print(f"\nSelected random 2026 series: {plant} / {sku} ({series.height:_} test points)")

    y_pred = model.predict(series)
    y_true = series["target_consumo_real"].to_numpy()
    y_baseline = series["consumo_real"].to_numpy()
    metrics = RegressionMetrics(model_name="LGBMRegressor", n_features=len(feature_cols)).compute(
        y_true, y_pred, y_baseline=y_baseline
    )
    print("\n" + metrics.summary())

    safe_sku = sku.replace("/", "-").replace(" ", "_")
    save_path = ROOT / "plots" / f"forecast_ieee_{plant}_{safe_sku}.png"
    plot_forecast_ieee(series, y_pred, metrics, plant, sku, save_path)


if __name__ == "__main__":
    main()
