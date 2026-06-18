from __future__ import annotations

from typing import Callable

import numpy as np
import polars as pl

from src.utils.regression_metrics import RegressionMetrics

from .base import FrameLike, LagSelectionStrategy, LagSpec, build_lag_features, is_pandas, to_polars


class LagFeatureValidator:
    r"""
    Compare lag-selection strategies head to head with chronological walk-forward CV.

    *   For each strategy: fit a global LagSpec on the provided (training) panel, build the
        evaluation features, and score a fresh model across expanding walk-forward folds.
    *   Ranks strategies by primary_metric and exposes the winning spec.
    *   model_factory must return a fresh sklearn-style estimator: fit(X, y) / predict(X)
        on numpy arrays. This keeps the validator model-agnostic and easy to test.
    *
    """

    def __init__(
        self,
        strategies: list[LagSelectionStrategy],
        model_factory: Callable[[], object],
        n_splits: int = 4,
        primary_metric: str = "wape",
        target_col: str = "target_consumo_real",
        peak_threshold: float = 2.0,
        date_col: str = "fecha",
        group_cols: tuple[str, str] = ("planta", "sku"),
    ) -> None:
        self.strategies = strategies
        self.model_factory = model_factory
        self.n_splits = n_splits
        self.primary_metric = primary_metric
        self.target_col = target_col
        self.peak_threshold = peak_threshold
        self.date_col = date_col
        self.group_cols = group_cols

        self.results_: pl.DataFrame | None = None
        self.best_spec_: LagSpec | None = None
        self.specs_: dict[str, LagSpec] = {}

    def _fold_boundaries(self, dates: list) -> list:
        # Expanding-window boundaries: split the timeline into n_splits + 1 segments
        n = len(dates)
        step = n // (self.n_splits + 1)
        return [dates[step * (i + 1)] for i in range(self.n_splits)]

    def _walk_forward(self, df_feat: pl.DataFrame, feature_cols: list[str]) -> dict:
        df_feat = df_feat.drop_nulls(subset=feature_cols + [self.target_col])
        dates = df_feat.select(self.date_col).unique().sort(self.date_col)[self.date_col].to_list()
        if len(dates) < (self.n_splits + 2):
            raise ValueError("Not enough distinct dates for the requested number of splits.")
        boundaries = self._fold_boundaries(dates)

        maes, wapes, recalls, lifts = [], [], [], []
        for i, bound in enumerate(boundaries):
            next_bound = boundaries[i + 1] if i + 1 < len(boundaries) else dates[-1]
            train = df_feat.filter(pl.col(self.date_col) <= bound)
            test = df_feat.filter(
                (pl.col(self.date_col) > bound) & (pl.col(self.date_col) <= next_bound)
            )
            if train.height == 0 or test.height == 0:
                continue

            x_train = train.select(feature_cols).to_numpy()
            y_train = train[self.target_col].to_numpy()
            x_test = test.select(feature_cols).to_numpy()
            y_test = test[self.target_col].to_numpy()

            model = self.model_factory()
            model.fit(x_train, y_train)
            y_pred = np.asarray(model.predict(x_test))

            baseline = test["lag_1"].to_numpy()
            m = RegressionMetrics(model_name="lag_eval", n_features=len(feature_cols))
            m.compute(y_test, y_pred, y_baseline=baseline)
            maes.append(m.mae)
            wapes.append(m.wape)
            if m.baseline_wape:
                lifts.append(m.wape_lift_pct)

            # Derived peak recall: pred > peak_threshold * daily_forecast
            thr = self.peak_threshold * test["daily_forecast"].to_numpy()
            pred_peak = y_pred > thr
            true_peak = y_test > thr
            tp = int(np.sum(pred_peak & true_peak))
            fn = int(np.sum(~pred_peak & true_peak))
            recalls.append(tp / (tp + fn) if (tp + fn) > 0 else np.nan)

        return {
            "mae": float(np.nanmean(maes)),
            "wape": float(np.nanmean(wapes)),
            "peak_recall": float(np.nanmean(recalls)),
            "lift_vs_lag1_pct": float(np.nanmean(lifts)) if lifts else np.nan,
        }

    def run(self, df_raw: FrameLike) -> FrameLike:
        r"""
        Compare every strategy with walk-forward CV. Accepts a polars or pandas
        DataFrame and returns the ranked report in the same flavour it received.
        """
        return_pandas = is_pandas(df_raw)
        df_raw = to_polars(df_raw)
        rows = []
        for strat in self.strategies:
            spec = strat.fit_panel(
                df_raw, group_cols=self.group_cols, date_col=self.date_col
            )
            self.specs_[strat.name] = spec
            df_feat, feature_cols = build_lag_features(
                df_raw, spec, date_col=self.date_col, group_cols=self.group_cols
            )
            scores = self._walk_forward(df_feat, feature_cols)
            rows.append({
                "method": strat.name,
                "n_lags": len(spec.lags),
                "lags": str(spec.lags),
                "windows": str(spec.windows),
                **scores,
            })

        results = pl.DataFrame(rows)
        # Lower is better for mae/wape; higher is better for recall/lift
        descending = self.primary_metric in ("peak_recall", "lift_vs_lag1_pct")
        results = results.sort(self.primary_metric, descending=descending)
        self.results_ = results
        self.best_spec_ = self.specs_[results[0, "method"]]
        if return_pandas:
            return results.to_pandas()
        return results
