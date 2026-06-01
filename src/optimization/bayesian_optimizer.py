from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd
import polars as pl
import optuna

import plotly.graph_objects as go

optuna.logging.set_verbosity(optuna.logging.WARNING)

_TEMPLATE = "plotly_white"
_PALETTE = {
    "trial": "#aec7e8",
    "best": "#1f77b4",
    "param": "#ff7f0e",
}


class BayesianOptimizer:
    """
    Model-agnostic Bayesian hyperparameter optimizer using Optuna (TPE sampler).

    Compatible with any model that implements:
        model = ModelClass(**hyperparams)
        model.fit(df_train, **fit_kwargs)
        model.predict(df_test) -> np.ndarray

    Walk-forward expanding-window CV is used internally — never random splits.
    sku_stats leakage across inner folds is an accepted approximation for
    hyperparameter search (build_features should be called before this class).

    Usage
    -----
    def lgbm_space(trial):
        return {
            "n_estimators":  trial.suggest_int("n_estimators", 100, 1_000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth":     trial.suggest_int("max_depth", 3, 10),
        }

    opt = BayesianOptimizer(
        model_class=LGBMRegressor,
        param_space_fn=lgbm_space,
        fit_kwargs={"x_predictor": feature_cols, "y": "target_consumo_real"},
        metric="mae",
        n_trials=50,
        n_cv_folds=3,
        val_size=30,
    )
    study = opt.optimize(df_train)
    print(opt.best_params())
    best_model = opt.best_model(df_train)
    opt.plot_optimization_history().show()
    opt.plot_param_importance().show()
    """

    def __init__(
        self,
        model_class,
        param_space_fn: Callable,
        fit_kwargs: dict,
        metric: str = "mae",
        n_trials: int = 50,
        n_cv_folds: int = 1,
        val_size: int = 30,
        direction: str = "minimize",
        verbose: bool = False,
    ) -> None:
        if metric not in {"mae", "wape", "rmse"}:
            raise ValueError(f"metric must be 'mae', 'wape', or 'rmse'; got '{metric}'")
        if "y" not in fit_kwargs:
            raise ValueError("fit_kwargs must contain a 'y' key with the target column name")

        self._model_class = model_class
        self._param_space_fn = param_space_fn
        self._fit_kwargs = fit_kwargs
        self._metric = metric
        self._n_trials = n_trials
        self._n_cv_folds = n_cv_folds
        self._val_size = val_size
        self._direction = direction
        self._verbose = verbose
        self._y_col: str = fit_kwargs["y"]
        self._study: optuna.Study | None = None

        if verbose:
            optuna.logging.set_verbosity(optuna.logging.INFO)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(self, df_train: pl.DataFrame) -> optuna.Study:
        """
        Run Bayesian optimization over n_trials with walk-forward CV on df_train.

        Sorts df_train by 'fecha' internally — the input does not need to be sorted.
        Returns the completed optuna.Study for further inspection.
        """
        df_sorted = df_train.sort("fecha")
        n = len(df_sorted)
        min_required = self._n_cv_folds * self._val_size + self._val_size
        if n < min_required:
            raise ValueError(
                f"df_train has {n} rows but needs at least {min_required} "
                f"for {self._n_cv_folds} folds of size {self._val_size}."
            )

        self._study = optuna.create_study(
            direction=self._direction,
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        self._study.optimize(
            self._make_objective(df_sorted),
            n_trials=self._n_trials,
            show_progress_bar=self._verbose,
        )
        return self._study

    def best_params(self) -> dict:
        """Return hyperparameters of the best trial."""
        self._check_fitted()
        return self._study.best_params

    def best_value(self) -> float:
        """Return the best metric value achieved."""
        self._check_fitted()
        return self._study.best_value

    def best_model(self, df_train: pl.DataFrame):
        """
        Instantiate the best model and retrain on the full df_train.

        Returns a fitted model ready for predict().
        """
        self._check_fitted()
        model = self._model_class(**self._study.best_params)
        model.fit(df_train, **self._fit_kwargs)
        return model

    def results_dataframe(self) -> pd.DataFrame:
        """
        Return all completed trials as a DataFrame.

        Columns: trial_id, <metric>, <hyperparameter columns>.
        """
        self._check_fitted()
        rows = []
        for t in self._study.trials:
            if t.state == optuna.trial.TrialState.COMPLETE:
                row = {"trial_id": t.number, self._metric: t.value}
                row.update(t.params)
                rows.append(row)
        return (
            pd.DataFrame(rows)
            .sort_values("trial_id")
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_optimization_history(self) -> go.Figure:
        """
        Trial metric values (scatter) + best-so-far line.

        Shows how the Bayesian search converges toward the optimum.
        """
        self._check_fitted()
        completed = [
            t for t in self._study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        trial_ids = [t.number for t in completed]
        values = [t.value for t in completed]

        best_so_far: list[float] = []
        current_best = float("inf") if self._direction == "minimize" else float("-inf")
        for v in values:
            if self._direction == "minimize":
                current_best = min(current_best, v)
            else:
                current_best = max(current_best, v)
            best_so_far.append(current_best)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trial_ids,
            y=values,
            mode="markers",
            name="Trial value",
            marker=dict(color=_PALETTE["trial"], size=7, opacity=0.8),
        ))
        fig.add_trace(go.Scatter(
            x=trial_ids,
            y=best_so_far,
            mode="lines",
            name="Best so far",
            line=dict(color=_PALETTE["best"], width=2.5),
        ))
        fig.update_layout(
            title=(
                f"Bayesian Optimization History — "
                f"{self._model_class.__name__} ({self._metric.upper()})"
            ),
            xaxis_title="Trial",
            yaxis_title=self._metric.upper(),
            template=_TEMPLATE,
            hovermode="x unified",
        )
        return fig

    def plot_param_importance(self) -> go.Figure:
        """
        Horizontal bar chart of hyperparameter importance using fANOVA.

        Requires at least 4 completed trials for a meaningful estimate.
        """
        self._check_fitted()
        importances = optuna.importance.get_param_importances(self._study)
        params = list(importances.keys())[::-1]
        imp_values = [importances[p] for p in params]

        fig = go.Figure(go.Bar(
            x=imp_values,
            y=params,
            orientation="h",
            marker=dict(
                color=imp_values,
                colorscale="Blues",
                showscale=False,
            ),
            text=[f"{v:.3f}" for v in imp_values],
            textposition="outside",
        ))
        fig.update_layout(
            title=f"Hyperparameter Importance (fANOVA) — {self._model_class.__name__}",
            xaxis_title="Importance",
            template=_TEMPLATE,
            height=max(350, len(params) * 30 + 120),
        )
        return fig

    def plot_parallel_coordinates(self) -> go.Figure:
        """
        Parallel coordinates plot across all trials.

        Each line is one trial. Color encodes metric value.
        Useful for spotting which regions of the search space perform best.
        """
        self._check_fitted()
        df = self.results_dataframe()
        param_cols = [c for c in df.columns if c not in {"trial_id", self._metric}]

        dimensions = []
        for col in param_cols:
            col_data = df[col]
            if col_data.dtype == object:
                categories = col_data.unique().tolist()
                dimensions.append(dict(
                    label=col,
                    values=[categories.index(v) for v in col_data],
                    tickvals=list(range(len(categories))),
                    ticktext=categories,
                ))
            else:
                dimensions.append(dict(
                    label=col,
                    values=col_data.tolist(),
                    range=[col_data.min(), col_data.max()],
                ))

        dimensions.append(dict(
            label=self._metric.upper(),
            values=df[self._metric].tolist(),
            range=[df[self._metric].min(), df[self._metric].max()],
        ))

        colorscale = "RdYlGn_r" if self._direction == "minimize" else "RdYlGn"

        fig = go.Figure(go.Parcoords(
            line=dict(
                color=df[self._metric],
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(title=self._metric.upper()),
            ),
            dimensions=dimensions,
        ))
        fig.update_layout(
            title=f"Parallel Coordinates — {self._model_class.__name__}",
            template=_TEMPLATE,
            height=500,
        )
        return fig

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_objective(self, df_sorted: pl.DataFrame) -> Callable:
        def objective(trial: optuna.Trial) -> float:
            params = self._param_space_fn(trial)
            fold_metrics: list[float] = []
            for fold_idx in range(self._n_cv_folds):
                df_fold_train, df_fold_val = self._split_fold(df_sorted, fold_idx)
                model = self._model_class(**params)
                model.fit(df_fold_train, **self._fit_kwargs)
                y_pred = model.predict(df_fold_val)
                y_true = df_fold_val[self._y_col].to_numpy()
                fold_metrics.append(self._compute_metric(y_true, y_pred))
            return float(np.mean(fold_metrics))
        return objective

    def _split_fold(
        self, df_sorted: pl.DataFrame, fold_idx: int
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Expanding-window walk-forward split.

        fold_idx=0 is the oldest fold (smallest train, first val window).
        fold_idx=n_cv_folds-1 is the newest fold (largest train, last val window).

        Example with n_cv_folds=3, val_size=30, n=300:
          fold 0: train=rows[0:210], val=rows[210:240]
          fold 1: train=rows[0:240], val=rows[240:270]
          fold 2: train=rows[0:270], val=rows[270:300]
        """
        n = len(df_sorted)
        folds_ahead = self._n_cv_folds - fold_idx
        val_end = n - (folds_ahead - 1) * self._val_size
        val_start = val_end - self._val_size
        df_fold_val = df_sorted.slice(val_start, self._val_size)
        df_fold_train = df_sorted.slice(0, val_start)
        return df_fold_train, df_fold_val

    def _compute_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if self._metric == "mae":
            return float(np.mean(np.abs(y_true - y_pred)))
        if self._metric == "rmse":
            return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        if self._metric == "wape":
            denom = np.sum(np.abs(y_true))
            if denom == 0:
                return float("inf")
            return float(np.sum(np.abs(y_true - y_pred)) / denom)
        raise ValueError(f"Unknown metric: {self._metric}")

    def _check_fitted(self) -> None:
        if self._study is None:
            raise RuntimeError("Call optimize() before accessing results.")
