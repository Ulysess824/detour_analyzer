from __future__ import annotations

import math
import numpy as np
import pandas as pd
from typing import Optional, Sequence

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


_PALETTE = {
    "actual": "#1f77b4",
    "pred": "#ff7f0e",
    "threshold": "#d62728",
    "tp": "#2ca02c",
    "fp": "#d62728",
    "fn": "#9467bd",
    "tn": "#aec7e8",
    "residual": "#8c564b",
    "zero": "#7f7f7f",
}

_TEMPLATE = "plotly_white"


class ConsumptionPlotter:
    """
    Interactive Plotly visualisations for consumption prediction diagnostics.

    All methods return a go.Figure so callers can further customise or save.
    Call .show() on the returned figure or pass save_path to write HTML.

    Usage
    -----
    plotter = ConsumptionPlotter(model_name="LightGBM")
    fig = plotter.plot_forecast(df, date_col="fecha", actual_col="consumo",
                                pred_col="pred_consumo", threshold_col="umbral_pico")
    fig.show()
    """

    def __init__(self, model_name: str = "Model") -> None:
        self.model_name = model_name

    # ------------------------------------------------------------------
    # 1. Forecast vs Actual with Peak Classification overlay
    # ------------------------------------------------------------------
    def plot_forecast(
        self,
        df: pd.DataFrame,
        date_col: str,
        actual_col: str,
        pred_col: str,
        threshold_col: Optional[str] = None,
        plant: Optional[str] = None,
        sku: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """
        Line chart: actual consumption vs predicted, with optional peak threshold.
        Points are coloured by classification outcome (TP, FP, FN, TN).
        """
        data = df.copy()
        if plant and sku:
            data = data[(data["planta"] == plant) & (data["sku"] == sku)]

        data = data.sort_values(date_col)

        if threshold_col and threshold_col in data.columns:
            data["_is_actual_peak"] = data[actual_col] > data[threshold_col]
            data["_is_pred_peak"] = data[pred_col] > data[threshold_col]
            data["_outcome"] = data.apply(
                lambda r: (
                    "TP" if r["_is_actual_peak"] and r["_is_pred_peak"]
                    else "FP" if not r["_is_actual_peak"] and r["_is_pred_peak"]
                    else "FN" if r["_is_actual_peak"] and not r["_is_pred_peak"]
                    else "TN"
                ),
                axis=1,
            )

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=data[date_col], y=data[actual_col],
            name="Actual Consumption",
            line=dict(color=_PALETTE["actual"], width=2),
            mode="lines",
        ))

        fig.add_trace(go.Scatter(
            x=data[date_col], y=data[pred_col],
            name=f"Predicted ({self.model_name})",
            line=dict(color=_PALETTE["pred"], width=2, dash="dash"),
            mode="lines",
        ))

        if threshold_col and threshold_col in data.columns:
            fig.add_trace(go.Scatter(
                x=data[date_col], y=data[threshold_col],
                name="Peak Threshold (2x Forecast)",
                line=dict(color=_PALETTE["threshold"], width=1.5, dash="dot"),
                mode="lines",
            ))

            outcome_colors = {"TP": _PALETTE["tp"], "FP": _PALETTE["fp"],
                              "FN": _PALETTE["fn"], "TN": _PALETTE["tn"]}
            for outcome, color in outcome_colors.items():
                mask = data["_outcome"] == outcome
                if mask.any():
                    fig.add_trace(go.Scatter(
                        x=data.loc[mask, date_col],
                        y=data.loc[mask, actual_col],
                        name=outcome,
                        mode="markers",
                        marker=dict(color=color, size=8, symbol="circle"),
                    ))

        series_label = f"{plant} / {sku}" if plant and sku else "All series"
        fig.update_layout(
            title=dict(
                text=f"Actual vs Predicted Consumption — {series_label}",
                font=dict(size=16),
            ),
            xaxis_title="Date",
            yaxis_title="Consumption (TO)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template=_TEMPLATE,
            hovermode="x unified",
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    # ------------------------------------------------------------------
    # 2. Residuals Analysis Dashboard (3-panel)
    # ------------------------------------------------------------------
    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dates: Optional[Sequence] = None,
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """
        Three-panel residual diagnostics:
          1. Residuals over time (or index)
          2. Residual distribution (histogram + KDE)
          3. Residuals vs Fitted (heteroskedasticity check)
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        residuals = y_true - y_pred
        x_axis = dates if dates is not None else np.arange(len(residuals))

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=(
                "Residuals over Time",
                "Residual Distribution",
                "Residuals vs Fitted",
            ),
        )

        # Panel 1: residuals over time
        fig.add_trace(go.Scatter(
            x=list(x_axis), y=residuals,
            mode="lines",
            name="Residual",
            line=dict(color=_PALETTE["residual"], width=1),
        ), row=1, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color=_PALETTE["zero"], row=1, col=1)

        # Panel 2: histogram
        fig.add_trace(go.Histogram(
            x=residuals,
            nbinsx=40,
            name="Distribution",
            marker_color=_PALETTE["actual"],
            opacity=0.75,
        ), row=1, col=2)

        # Panel 3: residuals vs fitted
        fig.add_trace(go.Scatter(
            x=y_pred, y=residuals,
            mode="markers",
            name="Residuals vs Fitted",
            marker=dict(color=_PALETTE["pred"], size=5, opacity=0.6),
        ), row=1, col=3)
        fig.add_hline(y=0, line_dash="dash", line_color=_PALETTE["zero"], row=1, col=3)

        fig.update_layout(
            title=f"Residual Diagnostics — {self.model_name}",
            showlegend=False,
            template=_TEMPLATE,
            height=420,
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    # ------------------------------------------------------------------
    # 3. SHAP Feature Importance (bar chart)
    # ------------------------------------------------------------------
    def plot_shap_importance(
        self,
        shap_values: np.ndarray,
        feature_names: Sequence[str],
        top_n: int = 20,
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """
        Horizontal bar chart of mean absolute SHAP values.

        Parameters
        ----------
        shap_values : ndarray, shape (n_samples, n_features)
        feature_names : list of str
        top_n : int
            Number of top features to display.
        """
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        order = np.argsort(mean_abs_shap)[::-1][:top_n]
        importance = mean_abs_shap[order][::-1]
        features = np.array(feature_names)[order][::-1]

        fig = go.Figure(go.Bar(
            x=importance,
            y=features,
            orientation="h",
            marker=dict(
                color=importance,
                colorscale="Blues",
                showscale=False,
            ),
        ))

        fig.update_layout(
            title=f"SHAP Feature Importance (Mean |SHAP|) — {self.model_name}",
            xaxis_title="Mean |SHAP value|",
            yaxis_title="Feature",
            template=_TEMPLATE,
            height=max(400, top_n * 22),
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    # ------------------------------------------------------------------
    # 4. PR Curve
    # ------------------------------------------------------------------
    def plot_pr_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        pr_auc: Optional[float] = None,
        save_path: Optional[str] = None,
    ) -> go.Figure:
        from sklearn.metrics import precision_recall_curve

        y_true = np.asarray(y_true, dtype=int)
        y_proba = np.asarray(y_pred_proba, dtype=float)
        precision, recall, _ = precision_recall_curve(y_true, y_proba)

        label = f"PR curve (AUC = {pr_auc:.4f})" if pr_auc is not None else "PR curve"

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode="lines",
            name=label,
            line=dict(color=_PALETTE["pred"], width=2),
            fill="tozeroy",
            fillcolor="rgba(255,127,14,0.10)",
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[np.mean(y_true)] * 2,
            mode="lines",
            name="No-skill baseline",
            line=dict(color=_PALETTE["zero"], dash="dash"),
        ))

        fig.update_layout(
            title=f"Precision-Recall Curve — {self.model_name}",
            xaxis_title="Recall",
            yaxis_title="Precision",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1.05]),
            template=_TEMPLATE,
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    # ------------------------------------------------------------------
    # 5. ROC Curve
    # ------------------------------------------------------------------
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        roc_auc: Optional[float] = None,
        save_path: Optional[str] = None,
    ) -> go.Figure:
        from sklearn.metrics import roc_curve

        y_true = np.asarray(y_true, dtype=int)
        y_proba = np.asarray(y_pred_proba, dtype=float)
        fpr, tpr, _ = roc_curve(y_true, y_proba)

        label = f"ROC (AUC = {roc_auc:.4f})" if roc_auc is not None else "ROC"

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode="lines",
            name=label,
            line=dict(color=_PALETTE["actual"], width=2),
            fill="tozeroy",
            fillcolor="rgba(31,119,180,0.10)",
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            name="Random classifier",
            line=dict(color=_PALETTE["zero"], dash="dash"),
        ))

        fig.update_layout(
            title=f"ROC Curve — {self.model_name}",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1.05]),
            template=_TEMPLATE,
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    # ------------------------------------------------------------------
    # 6. Multi-model metrics comparison (bar chart)
    # ------------------------------------------------------------------
    def plot_model_comparison(
        self,
        metrics_list: list[dict],
        metric_cols: Sequence[str] = ("mae", "rmse", "wape"),
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """
        Grouped bar chart comparing a set of metrics across multiple models.

        Parameters
        ----------
        metrics_list : list of dicts
            Each dict must contain a 'model' key and the metric columns.
        metric_cols : tuple of str
            Metrics to compare (must exist in each dict).
        """
        df = pd.DataFrame(metrics_list)

        fig = go.Figure()
        colors = px.colors.qualitative.Plotly

        for i, metric in enumerate(metric_cols):
            if metric not in df.columns:
                continue
            fig.add_trace(go.Bar(
                name=metric.upper(),
                x=df["model"],
                y=df[metric],
                marker_color=colors[i % len(colors)],
                text=df[metric].round(4),
                textposition="outside",
            ))

        fig.update_layout(
            barmode="group",
            title="Model Comparison — Regression Metrics",
            xaxis_title="Model",
            yaxis_title="Metric Value",
            template=_TEMPLATE,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    # ------------------------------------------------------------------
    # 7. Calibration curve (reliability diagram)
    # ------------------------------------------------------------------
    def plot_calibration(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        n_bins: int = 10,
        save_path: Optional[str] = None,
    ) -> go.Figure:
        from sklearn.calibration import calibration_curve

        y_true = np.asarray(y_true, dtype=int)
        y_proba = np.asarray(y_pred_proba, dtype=float)
        fraction_of_positives, mean_predicted = calibration_curve(
            y_true, y_proba, n_bins=n_bins
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=mean_predicted, y=fraction_of_positives,
            mode="lines+markers",
            name=self.model_name,
            line=dict(color=_PALETTE["pred"], width=2),
            marker=dict(size=8),
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            name="Perfect calibration",
            line=dict(color=_PALETTE["zero"], dash="dash"),
        ))

        fig.update_layout(
            title=f"Calibration Curve (Reliability Diagram) — {self.model_name}",
            xaxis_title="Mean Predicted Probability",
            yaxis_title="Fraction of Positives",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            template=_TEMPLATE,
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    # ------------------------------------------------------------------
    # 8. Scatter actual vs predicted (bias & variance diagnosis)
    # ------------------------------------------------------------------
    def plot_scatter_actual_vs_pred(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dates: Optional[Sequence] = None,
        plant: Optional[str] = None,
        sku: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """
        Scatter plot of actual vs predicted consumption.

        Each point is one day. The diagonal Y=X is perfect prediction.
        Points above the diagonal = underprediction; below = overprediction.
        Color encodes absolute error magnitude.
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        abs_error = np.abs(y_true - y_pred)

        hover_text = (
            [f"Date: {d}<br>Actual: {a:.2f}<br>Pred: {p:.2f}<br>|Error|: {e:.2f}"
             for d, a, p, e in zip(dates, y_true, y_pred, abs_error)]
            if dates is not None
            else [f"Actual: {a:.2f}<br>Pred: {p:.2f}<br>|Error|: {e:.2f}"
                  for a, p, e in zip(y_true, y_pred, abs_error)]
        )

        axis_min = float(min(y_true.min(), y_pred.min())) * 0.95
        axis_max = float(max(y_true.max(), y_pred.max())) * 1.05
        r2 = float(1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=[axis_min, axis_max],
            y=[axis_min, axis_max],
            mode="lines",
            name="Perfect prediction (Y=X)",
            line=dict(color=_PALETTE["zero"], dash="dash", width=1.5),
            hoverinfo="skip",
        ))

        fig.add_trace(go.Scatter(
            x=y_true,
            y=y_pred,
            mode="markers",
            name=self.model_name,
            text=hover_text,
            hoverinfo="text",
            marker=dict(
                color=abs_error,
                colorscale="YlOrRd",
                size=7,
                opacity=0.75,
                colorbar=dict(title="|Error|", thickness=14),
                showscale=True,
            ),
        ))

        series_label = f"{plant} / {sku}" if plant and sku else "All series"
        fig.update_layout(
            title=f"Actual vs Predicted — {series_label} (R²={r2:.3f})",
            xaxis_title="Actual Consumption (TO)",
            yaxis_title="Predicted Consumption (TO)",
            xaxis=dict(range=[axis_min, axis_max]),
            yaxis=dict(range=[axis_min, axis_max]),
            template=_TEMPLATE,
            hovermode="closest",
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    # ------------------------------------------------------------------
    # 9. Absolute error over time
    # ------------------------------------------------------------------
    def plot_error_over_time(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dates: Sequence,
        baseline_mae: Optional[float] = None,
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """
        Absolute error per day over time with MAE reference line.

        Regions above MAE are highlighted to surface error clusters.
        Pass baseline_mae (lag-1 MAE) to compare model error against the baseline.
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        abs_error = np.abs(y_true - y_pred)
        model_mae = float(np.mean(abs_error))

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=list(dates),
            y=abs_error,
            mode="lines",
            name=f"|Error| ({self.model_name})",
            line=dict(color=_PALETTE["pred"], width=1.5),
            fill="tozeroy",
            fillcolor="rgba(255,127,14,0.12)",
        ))

        fig.add_hline(
            y=model_mae,
            line_dash="solid",
            line_color=_PALETTE["pred"],
            line_width=1.5,
            annotation_text=f"MAE model = {model_mae:.2f}",
            annotation_position="top right",
        )

        if baseline_mae is not None:
            fig.add_hline(
                y=baseline_mae,
                line_dash="dash",
                line_color=_PALETTE["zero"],
                line_width=1.5,
                annotation_text=f"MAE baseline (lag-1) = {baseline_mae:.2f}",
                annotation_position="bottom right",
            )

        fig.update_layout(
            title=f"Absolute Error over Time — {self.model_name}",
            xaxis_title="Date",
            yaxis_title="|Actual - Predicted| (TO)",
            template=_TEMPLATE,
            hovermode="x unified",
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    # ------------------------------------------------------------------
    # 10. Forecast grid — actual vs predicted per (planta, sku)
    # ------------------------------------------------------------------
    def plot_forecast_grid(
        self,
        df: pd.DataFrame,
        date_col: str,
        actual_col: str,
        pred_col: str,
        max_series: int = 12,
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """
        Subplot grid showing actual vs predicted for each (planta, sku).

        Series are sorted by descending MAE so the worst-performing appear first.
        Capped at max_series panels to keep the figure readable.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: planta, sku, date_col, actual_col, pred_col.
        max_series : int
            Maximum number of (planta, sku) panels to render.
        """
        series_mae = (
            df.groupby(["planta", "sku"])
            .apply(lambda g: np.mean(np.abs(g[actual_col] - g[pred_col])))
            .sort_values(ascending=False)
            .head(max_series)
        )
        series_list = list(series_mae.index)
        n = len(series_list)

        n_cols = min(3, n)
        n_rows = math.ceil(n / n_cols)

        subplot_titles = [f"{p} / {s}" for p, s in series_list]
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=subplot_titles,
            shared_xaxes=False,
        )

        for idx, (plant, sku) in enumerate(series_list):
            row = idx // n_cols + 1
            col = idx % n_cols + 1
            subset = df[(df["planta"] == plant) & (df["sku"] == sku)].sort_values(date_col)
            show_legend = idx == 0

            fig.add_trace(go.Scatter(
                x=subset[date_col],
                y=subset[actual_col],
                mode="lines",
                name="Actual",
                line=dict(color=_PALETTE["actual"], width=1.5),
                showlegend=show_legend,
            ), row=row, col=col)

            fig.add_trace(go.Scatter(
                x=subset[date_col],
                y=subset[pred_col],
                mode="lines",
                name=f"Predicted ({self.model_name})",
                line=dict(color=_PALETTE["pred"], width=1.5, dash="dash"),
                showlegend=show_legend,
            ), row=row, col=col)

        fig.update_layout(
            title=f"Actual vs Predicted by Series (sorted by MAE desc) — top {n}",
            template=_TEMPLATE,
            height=320 * n_rows,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        if save_path:
            fig.write_html(save_path)

        return fig
