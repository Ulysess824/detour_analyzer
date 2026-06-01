from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ClassificationMetrics:
    """
    Computes and reports classification metrics for peak-detection models.

    Designed for Stage 2 (derived classification) where peaks are rare events.
    Prioritises Recall and PR-AUC over accuracy-based metrics.

    Usage
    -----
    metrics = ClassificationMetrics(beta=2.0)
    metrics.compute(y_true, y_pred_proba, threshold=0.5)
    print(metrics.summary())
    df = metrics.to_dataframe()
    """

    model_name: str = "Model"
    beta: float = 2.0  # F-beta weight; beta > 1 favours recall

    # Populated by .compute()
    n_obs: int = field(default=0, init=False)
    n_positive: int = field(default=0, init=False)
    prevalence: float = field(default=np.nan, init=False)
    threshold: float = field(default=0.5, init=False)

    tp: int = field(default=0, init=False)
    fp: int = field(default=0, init=False)
    fn: int = field(default=0, init=False)
    tn: int = field(default=0, init=False)

    precision: float = field(default=np.nan, init=False)
    recall: float = field(default=np.nan, init=False)
    f1: float = field(default=np.nan, init=False)
    fbeta: float = field(default=np.nan, init=False)
    accuracy: float = field(default=np.nan, init=False)
    specificity: float = field(default=np.nan, init=False)

    roc_auc: float = field(default=np.nan, init=False)
    pr_auc: float = field(default=np.nan, init=False)
    ks_stat: float = field(default=np.nan, init=False)
    adversarial_auc: Optional[float] = field(default=None, init=False)

    def compute(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = 0.5,
        X_train: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None,
    ) -> "ClassificationMetrics":
        """
        Compute all classification metrics.

        Parameters
        ----------
        y_true : array-like, shape (n,)
            Binary ground-truth labels (0/1).
        y_pred_proba : array-like, shape (n,)
            Predicted probabilities for the positive class.
        threshold : float
            Decision boundary for converting probabilities to labels.
        X_train, X_test : array-like, optional
            When provided, adversarial AUC is computed to detect distribution shift.
        """
        from sklearn.metrics import (
            roc_auc_score,
            average_precision_score,
        )

        y_true = np.asarray(y_true, dtype=int)
        y_proba = np.asarray(y_pred_proba, dtype=float)
        # Handle nan and inf values to avoid errors in sklearn metrics
        y_proba = np.nan_to_num(y_proba, nan=0.0, posinf=9_999_999.0, neginf=-9_999_999.0)
        y_pred = (y_proba >= threshold).astype(int)

        self.threshold = threshold
        self.n_obs = len(y_true)
        self.n_positive = int(np.sum(y_true))
        self.prevalence = self.n_positive / self.n_obs if self.n_obs > 0 else np.nan

        self.tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        self.fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        self.fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        self.tn = int(np.sum((y_pred == 0) & (y_true == 0)))

        self.precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        self.recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
        self.specificity = self.tn / (self.tn + self.fp) if (self.tn + self.fp) > 0 else 0.0
        self.accuracy = (self.tp + self.tn) / self.n_obs if self.n_obs > 0 else np.nan

        b2 = self.beta ** 2
        denom = b2 * self.precision + self.recall
        self.fbeta = (1 + b2) * self.precision * self.recall / denom if denom > 0 else 0.0
        f1_denom = self.precision + self.recall
        self.f1 = 2 * self.precision * self.recall / f1_denom if f1_denom > 0 else 0.0

        if len(np.unique(y_true)) > 1:
            self.roc_auc = float(roc_auc_score(y_true, y_proba))
            self.pr_auc = float(average_precision_score(y_true, y_proba))
            self.ks_stat = self._compute_ks(y_true, y_proba)
        else:
            self.roc_auc = np.nan
            self.pr_auc = np.nan
            self.ks_stat = np.nan

        if X_train is not None and X_test is not None:
            self.adversarial_auc = self._compute_adversarial_auc(X_train, X_test)

        return self

    def _compute_ks(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        scores_pos = y_proba[y_true == 1]
        scores_neg = y_proba[y_true == 0]
        if len(scores_pos) == 0 or len(scores_neg) == 0:
            return np.nan
        thresholds = np.sort(np.unique(y_proba))
        tprs = np.array([np.mean(scores_pos >= t) for t in thresholds])
        fprs = np.array([np.mean(scores_neg >= t) for t in thresholds])
        return float(np.max(np.abs(tprs - fprs)))

    def _compute_adversarial_auc(
        self, X_train: np.ndarray, X_test: np.ndarray
    ) -> float:
        """
        Trains a LightGBM to distinguish train vs test samples.
        AUC near 0.5 means no distribution shift.
        """
        try:
            from lightgbm import LGBMClassifier
            from sklearn.model_selection import cross_val_score

            X_adv = np.vstack([X_train, X_test])
            y_adv = np.array([0] * len(X_train) + [1] * len(X_test))
            clf = LGBMClassifier(n_estimators=100, verbose=-1, n_jobs=-1)
            scores = cross_val_score(clf, X_adv, y_adv, cv=3, scoring="roc_auc")
            return float(np.mean(scores))
        except Exception:
            return np.nan

    def summary(self) -> str:
        border = "=" * 80
        sub = "-" * 80

        shift_line = ""
        if self.adversarial_auc is not None:
            direction = "OK (no shift)" if abs(self.adversarial_auc - 0.5) < 0.05 else "WARNING: shift detected"
            shift_line = f"  {'Adversarial AUC':<30} {self.adversarial_auc:>15.4f}  <- {direction}"

        rows = [
            border,
            f"{'Classification Metrics Summary — ' + self.model_name:^80}",
            border,
            f"  No. Observations : {self.n_obs:_}",
            f"  Positives (peaks): {self.n_positive:_}  ({self.prevalence:.2%} prevalence)",
            f"  Decision threshold: {self.threshold:.4f}   |   F-beta (beta={self.beta}): {self.fbeta:.4f}",
            sub,
            f"  {'Confusion Matrix':<30}",
            f"  {'TP':<10} {self.tp:>6}    {'FP':<10} {self.fp:>6}",
            f"  {'FN':<10} {self.fn:>6}    {'TN':<10} {self.tn:>6}",
            sub,
            f"  {'Metric':<30} {'Value':>15}",
            sub,
            f"  {'Precision':<30} {self.precision:>15.4f}",
            f"  {'Recall (Sensitivity)':<30} {self.recall:>15.4f}",
            f"  {'Specificity':<30} {self.specificity:>15.4f}",
            f"  {'F1-Score':<30} {self.f1:>15.4f}",
            f"  {'F-beta Score (beta={:.1f})':<30} {self.fbeta:>15.4f}".format(self.beta),
            f"  {'Accuracy':<30} {self.accuracy:>15.4f}",
            sub,
            f"  {'ROC-AUC':<30} {self.roc_auc:>15.4f}",
            f"  {'PR-AUC (avg precision)':<30} {self.pr_auc:>15.4f}",
            f"  {'KS Statistic':<30} {self.ks_stat:>15.4f}",
        ]

        if shift_line:
            rows.append(shift_line)

        rows.append(border)
        return "\n".join(rows)

    def to_dict(self) -> dict:
        d = {
            "model": self.model_name,
            "n_obs": self.n_obs,
            "n_positive": self.n_positive,
            "prevalence": self.prevalence,
            "threshold": self.threshold,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "precision": self.precision,
            "recall": self.recall,
            "specificity": self.specificity,
            "f1": self.f1,
            "fbeta": self.fbeta,
            "accuracy": self.accuracy,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "ks_stat": self.ks_stat,
            "adversarial_auc": self.adversarial_auc,
        }
        return d

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.to_dict()])

    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        metric: str = "fbeta",
    ) -> float:
        """
        Grid-search threshold in [0.05, 0.95] to maximise the chosen metric.

        Parameters
        ----------
        metric : str
            One of 'fbeta', 'f1', 'recall', 'precision'.

        Returns
        -------
        float
            Threshold that maximises the chosen metric.
        """
        best_thresh, best_val = 0.5, -1.0
        for t in np.arange(0.05, 0.96, 0.01):
            self.compute(y_true, y_pred_proba, threshold=float(t))
            val = getattr(self, metric, 0.0)
            if val > best_val:
                best_val = val
                best_thresh = float(t)
        self.compute(y_true, y_pred_proba, threshold=best_thresh)
        return best_thresh
