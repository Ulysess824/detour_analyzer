from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RegressionMetrics:
    """
    Computes and reports regression metrics for consumption prediction models.

    Designed to be embedded in any regressor class. Supports walk-forward
    evaluation and optional comparison against a lag-1 persistence baseline.

    Usage
    -----
    metrics = RegressionMetrics()
    metrics.compute(y_true, y_pred, y_baseline=y_lag1)
    print(metrics.summary())
    df = metrics.to_dataframe()
    """

    model_name: str = "Model"
    n_features: int = 0

    # Computed metrics (populated by .compute())
    n_obs: int = field(default=0, init=False)
    mae: float = field(default=np.nan, init=False)
    rmse: float = field(default=np.nan, init=False)
    wape: float = field(default=np.nan, init=False)
    mape: float = field(default=np.nan, init=False)
    bias: float = field(default=np.nan, init=False)
    r2: float = field(default=np.nan, init=False)
    adj_r2: float = field(default=np.nan, init=False)

    # Baseline comparison (lag-1 persistence)
    baseline_mae: Optional[float] = field(default=None, init=False)
    baseline_rmse: Optional[float] = field(default=None, init=False)
    baseline_wape: Optional[float] = field(default=None, init=False)
    mae_lift_pct: Optional[float] = field(default=None, init=False)
    wape_lift_pct: Optional[float] = field(default=None, init=False)

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_baseline: Optional[np.ndarray] = None,
    ) -> "RegressionMetrics":
        """
        Compute all metrics from ground-truth and prediction arrays.

        Parameters
        ----------
        y_true : array-like, shape (n,)
        y_pred : array-like, shape (n,)
        y_baseline : array-like, shape (n,), optional
            Lag-1 persistence values for baseline comparison.
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true, y_pred = y_true[mask], y_pred[mask]

        self.n_obs = len(y_true)
        errors = y_true - y_pred

        self.mae = float(np.mean(np.abs(errors)))
        self.rmse = float(np.sqrt(np.mean(errors ** 2)))
        self.bias = float(np.mean(errors))

        sum_true = np.sum(np.abs(y_true))
        self.wape = float(np.sum(np.abs(errors)) / sum_true) if sum_true > 0 else np.nan

        nonzero = y_true != 0
        if np.any(nonzero):
            self.mape = float(np.mean(np.abs(errors[nonzero] / y_true[nonzero])))
        else:
            self.mape = np.nan

        mean_y = np.mean(y_true)
        ss_tot = np.sum((y_true - mean_y) ** 2)
        ss_res = np.sum(errors ** 2)
        self.r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

        n, p = self.n_obs, self.n_features
        if n > p + 1 and not np.isnan(self.r2):
            self.adj_r2 = float(1.0 - (1.0 - self.r2) * (n - 1.0) / (n - p - 1.0))
        else:
            self.adj_r2 = self.r2

        if y_baseline is not None:
            self._compute_baseline(y_true, y_baseline, mask)

        return self

    def _compute_baseline(
        self, y_true: np.ndarray, y_baseline: np.ndarray, mask: np.ndarray
    ) -> None:
        y_bl = np.asarray(y_baseline, dtype=float)[mask]
        bl_errors = y_true - y_bl

        self.baseline_mae = float(np.mean(np.abs(bl_errors)))
        self.baseline_rmse = float(np.sqrt(np.mean(bl_errors ** 2)))

        sum_true = np.sum(np.abs(y_true))
        self.baseline_wape = (
            float(np.sum(np.abs(bl_errors)) / sum_true) if sum_true > 0 else np.nan
        )

        if self.baseline_mae > 0:
            self.mae_lift_pct = float((self.baseline_mae - self.mae) / self.baseline_mae * 100.0)
        if self.baseline_wape and self.baseline_wape > 0:
            self.wape_lift_pct = float(
                (self.baseline_wape - self.wape) / self.baseline_wape * 100.0
            )

    def summary(self) -> str:
        border = "=" * 80
        sub = "-" * 80

        rows = [
            border,
            f"{'Regression Metrics Summary — ' + self.model_name:^80}",
            border,
            f"  No. Observations : {self.n_obs:_}",
            f"  No. Features     : {self.n_features:_}",
            sub,
            f"  {'Metric':<30} {'Value':>15}",
            sub,
            f"  {'MAE':<30} {self.mae:>15.4f}",
            f"  {'RMSE':<30} {self.rmse:>15.4f}",
            f"  {'WAPE':<30} {self.wape:>15.4f}",
            f"  {'MAPE':<30} {self.mape:>15.4f}",
            f"  {'Bias (mean residual)':<30} {self.bias:>15.4f}",
            f"  {'R-squared':<30} {self.r2:>15.4f}",
            f"  {'Adj. R-squared':<30} {self.adj_r2:>15.4f}",
        ]

        if self.baseline_mae is not None:
            mae_lift_str = f"{self.mae_lift_pct:>14.2f}%" if self.mae_lift_pct is not None else f"{'N/A':>15}"
            wape_lift_str = f"{self.wape_lift_pct:>14.2f}%" if self.wape_lift_pct is not None else f"{'N/A':>15}"
            baseline_wape_str = f"{self.baseline_wape:>15.4f}" if self.baseline_wape is not None and not np.isnan(self.baseline_wape) else f"{'N/A':>15}"
            rows += [
                sub,
                f"  {'Baseline (lag-1 persistence)':<30} {'Value':>15}",
                sub,
                f"  {'Baseline MAE':<30} {self.baseline_mae:>15.4f}",
                f"  {'Baseline RMSE':<30} {self.baseline_rmse:>15.4f}",
                f"  {'Baseline WAPE':<30} {baseline_wape_str}",
                f"  {'MAE Lift vs Baseline':<30} {mae_lift_str}",
                f"  {'WAPE Lift vs Baseline':<30} {wape_lift_str}",
            ]

        rows.append(border)
        return "\n".join(rows)

    def to_dict(self) -> dict:
        d = {
            "model": self.model_name,
            "n_obs": self.n_obs,
            "mae": self.mae,
            "rmse": self.rmse,
            "wape": self.wape,
            "mape": self.mape,
            "bias": self.bias,
            "r2": self.r2,
            "adj_r2": self.adj_r2,
        }
        if self.baseline_mae is not None:
            d.update(
                {
                    "baseline_mae": self.baseline_mae,
                    "baseline_rmse": self.baseline_rmse,
                    "baseline_wape": self.baseline_wape,
                    "mae_lift_pct": self.mae_lift_pct,
                    "wape_lift_pct": self.wape_lift_pct,
                }
            )
        return d

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.to_dict()])
