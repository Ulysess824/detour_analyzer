from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from feature_engine.timeseries.forecasting import (
    ExpandingWindowFeatures,
    LagFeatures,
    WindowFeatures,
)
from scipy import stats
# pyrefly: ignore [missing-import]
from tsfresh import extract_features
# pyrefly: ignore [missing-import]
from tsfresh.feature_extraction.settings import (
    ComprehensiveFCParameters,
    EfficientFCParameters,
    MinimalFCParameters,
)
from tsfresh.utilities.dataframe_functions import impute, roll_time_series

FrameLike = pl.DataFrame | pd.DataFrame

# Named tsfresh calculator batteries, selectable by string from the builder config.
_FC_PARAMETER_SETS: dict[str, type] = {
    "minimal": MinimalFCParameters,
    "efficient": EfficientFCParameters,
    "comprehensive": ComprehensiveFCParameters,
}

# Broad grid used by FeatureEngineBuilder(auto=True): generates a large battery automatically
# (like tsfresh's automatic extraction) so the user does not have to pick lags/windows/functions.
_AUTO_LAGS: tuple[int, ...] = tuple(range(1, 15))  # 1..14
_AUTO_WINDOWS: tuple[int, ...] = (2, 3, 5, 7, 10, 14, 21, 30)
_AUTO_WINDOW_FUNCTIONS: tuple[str, ...] = (
    "mean",
    "std",
    "min",
    "max",
    "sum",
    "median",
    "var",
    "skew",
    "kurt",
)
_AUTO_EXPANDING_FUNCTIONS: tuple[str, ...] = ("mean", "std", "min", "max", "sum")


def to_polars(df: FrameLike) -> pl.DataFrame:
    r"""Accept a polars or pandas DataFrame and return a polars DataFrame."""
    if isinstance(df, pl.DataFrame):
        return df
    if isinstance(df, pd.DataFrame):
        return pl.from_pandas(df)
    raise TypeError(f"Expected a polars or pandas DataFrame, got {type(df).__name__}.")


def is_pandas(df: FrameLike) -> bool:
    r"""True when the input is a pandas DataFrame (used to round-trip the return type)."""
    return isinstance(df, pd.DataFrame)


class FeatureEngineBuilder:
    r"""
    Additive lag/rolling/expanding feature builder backed by feature-engine.

    *   Wraps feature-engine LagFeatures, WindowFeatures and ExpandingWindowFeatures so a
        panel of (planta, sku, fecha, value) rows is returned UNCHANGED plus new columns
        named <value>_lag_<n>, <value>_window_<w>_<fn> and <value>_expanding_<fn>.
    *   feature-engine (<= 1.9.4) operates on a single contiguous series, so every transform
        runs PER group. Each series is sorted chronologically and processed in isolation, which
        prevents shift/rolling from bleeding across plant/SKU boundaries (no cross-series leakage).
    *   WindowFeatures and ExpandingWindowFeatures use periods=1 by default, so the rolling
        statistic for day t never includes day t itself: it is built from t-1 backwards, which is
        the forecasting-safe convention (mirrors build_features rolling-on-lag_1).
    *   Stateless across train/test: lag and rolling features are deterministic functions of the
        input rows, so fit only records the produced column names; transform reproduces an
        identical feature set on test data (no fitted statistics, no leakage).
    *   Accepts and returns either polars or pandas (round-trips the input flavour). Never
        shuffles and never random-splits: the chronological order is always preserved.
    *   auto=True turns on an automatic mode that, like tsfresh's automatic extraction, builds a
        large feature battery without the user choosing anything: it overrides lags/windows/
        functions with a broad grid (lags 1..14, windows 2..30, the mean/std/min/max/sum/median/
        var/skew/kurt statistics, plus expanding mean/std/min/max/sum). Explicit lags/windows/
        window_functions/expanding_functions arguments are ignored when auto=True.
    *

    Example
    -------
    >>> # Manual: the user picks exactly which features to build.
    >>> builder = FeatureEngineBuilder(
    ...     value_cols="consumo_real",
    ...     group_cols=("planta", "sku"),
    ...     date_col="fecha",
    ...     lags=(1, 2, 3, 5, 7),
    ...     windows=(3, 5, 10),
    ...     window_functions=("mean", "std", "max"),
    ... )
    >>> df_train_feat, feature_cols = builder.fit_transform(df_train)
    >>> df_test_feat, _ = builder.transform(df_test)
    >>>
    >>> # Automatic: a large battery is generated with no manual specification.
    >>> auto_builder = FeatureEngineBuilder(value_cols="consumo_real", auto=True)
    >>> df_train_feat, feature_cols = auto_builder.fit_transform(df_train)  # ~150 columns
    """

    def __init__(
        self,
        *,
        value_cols: str | list[str] = "consumo_real",
        group_cols: tuple[str, str] = ("planta", "sku"),
        date_col: str = "fecha",
        auto: bool = False,
        lags: tuple[int, ...] = (1, 2, 3, 5, 7),
        windows: tuple[int, ...] = (3, 5, 10),
        window_functions: tuple[str, ...] = ("mean", "std", "max"),
        add_expanding: bool = True,
        expanding_functions: tuple[str, ...] = ("mean",),
        missing_values: str = "ignore",
    ) -> None:
        self.value_cols: list[str] = (
            [value_cols] if isinstance(value_cols, str) else list(value_cols)
        )
        self.group_cols = group_cols
        self.date_col = date_col
        self.auto = auto

        if auto:
            # Automatic mode: ignore the per-knob arguments and explode a broad grid so the
            # builder generates a large feature set on its own (tsfresh-style auto extraction).
            lags = _AUTO_LAGS
            windows = _AUTO_WINDOWS
            window_functions = _AUTO_WINDOW_FUNCTIONS
            expanding_functions = _AUTO_EXPANDING_FUNCTIONS
            add_expanding = True

        self.lags = tuple(lags)
        self.windows = tuple(windows)
        self.window_functions = tuple(window_functions)
        self.add_expanding = add_expanding
        self.expanding_functions = tuple(expanding_functions)
        self.missing_values = missing_values

        self._transformers: list | None = None
        self.feature_cols_: list[str] = []

    def _build_transformers(self) -> list:
        r"""Instantiate the feature-engine transformers requested by the configuration."""
        transformers: list = []
        if self.lags:
            transformers.append(
                LagFeatures(
                    variables=self.value_cols,
                    periods=list(self.lags),
                    missing_values=self.missing_values,
                    sort_index=True,
                )
            )
        if self.windows:
            transformers.append(
                WindowFeatures(
                    variables=self.value_cols,
                    window=list(self.windows),
                    functions=list(self.window_functions),
                    missing_values=self.missing_values,
                    sort_index=True,
                )
            )
        if self.add_expanding:
            transformers.append(
                ExpandingWindowFeatures(
                    variables=self.value_cols,
                    functions=list(self.expanding_functions),
                    missing_values=self.missing_values,
                    sort_index=True,
                )
            )
        return transformers

    @staticmethod
    def _added_columns(before: pd.DataFrame, after: pd.DataFrame) -> list[str]:
        r"""Return the columns present in after but not in before, preserving order."""
        known = set(before.columns)
        return [c for c in after.columns if c not in known]

    def _to_sorted_pandas(self, df: FrameLike) -> tuple[bool, pd.DataFrame]:
        r"""Sort chronologically per group and hand back a pandas frame with a fresh RangeIndex."""
        return_pandas = is_pandas(df)
        df_pl = to_polars(df).sort([*self.group_cols, self.date_col])
        df_pd = df_pl.to_pandas().reset_index(drop=True)
        return return_pandas, df_pd

    def fit(self, df: FrameLike) -> "FeatureEngineBuilder":
        r"""
        Fit the transformers and record the names of the features they produce.

        No statistics are learned from the data: lag/rolling/expanding columns are deterministic,
        so fit only validates the requested variables and captures the output column names.
        """
        for col in self.value_cols:
            if col not in to_polars(df).columns:
                raise KeyError(f"value column '{col}' not found in the input DataFrame.")

        _, df_pd = self._to_sorted_pandas(df)
        self._transformers = self._build_transformers()

        feature_cols: list[str] = []
        # Names do not depend on values nor on group boundaries, so a single fit/transform pass
        # over the whole frame is enough to capture them.
        for transformer in self._transformers:
            transformer.fit(df_pd)
            transformed = transformer.transform(df_pd)
            feature_cols.extend(self._added_columns(df_pd, transformed))
        self.feature_cols_ = feature_cols
        return self

    def transform(self, df: FrameLike) -> tuple[FrameLike, list[str]]:
        r"""
        Append the lag/rolling/expanding columns to df, returning (df, feature_cols).

        Processing is per group so no statistic crosses a plant/SKU boundary. The original rows,
        order and columns are preserved; only the engineered columns are added.
        """
        if self._transformers is None:
            raise RuntimeError("Call fit (or fit_transform) before transform.")

        return_pandas, df_pd = self._to_sorted_pandas(df)

        group_feature_frames: list[pd.DataFrame] = []
        for _, group in df_pd.groupby(list(self.group_cols), sort=False):
            collected: list[pd.DataFrame] = []
            for transformer in self._transformers:
                transformed = transformer.transform(group)
                added = self._added_columns(group, transformed)
                collected.append(transformed[added])
            group_feature_frames.append(pd.concat(collected, axis=1))

        # Reassemble in the original (sorted) row order via the shared RangeIndex.
        features_pd = pd.concat(group_feature_frames).sort_index()
        out_pd = pd.concat([df_pd, features_pd[self.feature_cols_]], axis=1)

        if return_pandas:
            return out_pd, list(self.feature_cols_)
        return pl.from_pandas(out_pd), list(self.feature_cols_)

    def fit_transform(self, df: FrameLike) -> tuple[FrameLike, list[str]]:
        r"""Fit on df and return the transformed frame plus the produced feature columns."""
        return self.fit(df).transform(df)


class TSFreshFeatureBuilder:
    r"""
    Additive tsfresh feature builder that mirrors FeatureEngineBuilder's panel contract.

    *   Same input/output shape rule as FeatureEngineBuilder: a panel of
        (planta, sku, fecha, value) rows is returned UNCHANGED plus one extra column per
        tsfresh feature. The row count never changes; only feature columns are appended.
    *   Per (series, day t) tsfresh rolls a trailing window (max_timeshift / min_timeshift)
        that ends at t and never includes t+1, so the features describe the past only
        (forecasting-safe, no look-ahead). The window ending at day t is joined back onto the
        original row for that day, which is what keeps the output rectangular.
    *   The feature battery is chosen by name via fc_parameters: "minimal" (a handful of cheap
        summary statistics), "efficient" (the full battery minus the costly calculators) or
        "comprehensive". A raw tsfresh settings dict is also accepted.
    *   Columns are named <value_col>__<tsfresh_feature> (tsfresh's own convention), e.g.
        consumo_real__mean, consumo_real__standard_deviation,
        consumo_real__fft_coefficient__attr_"abs"__coeff_2.
    *   Stateless across train/test in the same sense as FeatureEngineBuilder: the produced
        column set is a deterministic function of fc_parameters and the value columns, so fit
        records the column names and transform reindexes to them (missing -> null, extras
        dropped) so the train and test feature sets are identical (no leakage, no re-selection).
    *   No FRESH hypothesis-test selection is applied here: every feature in the chosen battery
        is appended. Use src/features/build_features.build_features when you want the
        target-aware FRESH-selected subset instead of the full matrix.
    *   Accepts and returns either polars or pandas (round-trips the input flavour). Never
        shuffles and never random-splits: chronological order is always preserved.
    *

    Example
    -------
    >>> builder = TSFreshFeatureBuilder(
    ...     value_cols="consumo_real",
    ...     group_cols=("planta", "sku"),
    ...     date_col="fecha",
    ...     fc_parameters="minimal",      # or "efficient" / "comprehensive"
    ...     max_timeshift=10,
    ...     min_timeshift=3,
    ... )
    >>> df_train_feat, feature_cols = builder.fit_transform(df_train)
    >>> df_test_feat, _ = builder.transform(df_test)   # same columns as train
    """

    def __init__(
        self,
        *,
        value_cols: str | list[str] = "consumo_real",
        group_cols: tuple[str, str] = ("planta", "sku"),
        date_col: str = "fecha",
        fc_parameters: str | dict = "minimal",
        max_timeshift: int = 10,
        min_timeshift: int = 3,
        n_jobs: int = 0,
    ) -> None:
        self.value_cols: list[str] = (
            [value_cols] if isinstance(value_cols, str) else list(value_cols)
        )
        self.group_cols = group_cols
        self.date_col = date_col
        self.fc_parameters = fc_parameters
        self.max_timeshift = max_timeshift
        self.min_timeshift = min_timeshift
        self.n_jobs = n_jobs

        self.feature_cols_: list[str] = []

    def _resolve_fc_parameters(self) -> dict:
        r"""Turn the fc_parameters config (name or dict) into a tsfresh settings dict."""
        if isinstance(self.fc_parameters, dict):
            return self.fc_parameters
        try:
            return _FC_PARAMETER_SETS[self.fc_parameters]()
        except KeyError as exc:
            valid = ", ".join(sorted(_FC_PARAMETER_SETS))
            raise ValueError(
                f"Unknown fc_parameters '{self.fc_parameters}'. Use one of: {valid}, "
                f"or pass a tsfresh settings dict."
            ) from exc

    def _to_long(self, df_pl: pl.DataFrame) -> pd.DataFrame:
        r"""
        Reshape the panel into the tsfresh long format expected by roll_time_series.

        Fuses the group columns into a single series_id and stacks the value columns into a
        (series_id, date, kind, value) frame so tsfresh names features per original column.
        """
        base = (
            df_pl.with_columns(
                pl.concat_str(
                    [pl.col(c).cast(pl.String) for c in self.group_cols], separator="__"
                ).alias("series_id")
            )
            .select(["series_id", self.date_col, *self.value_cols])
            .sort(["series_id", self.date_col])
        )
        long = base.unpivot(
            index=["series_id", self.date_col],
            on=self.value_cols,
            variable_name="kind",
            value_name="value",
        )
        return long.to_pandas()

    def _compute_feature_frame(self, df_pl: pl.DataFrame) -> pl.DataFrame:
        r"""
        Roll every series, extract the tsfresh battery and return a joinable polars frame.

        Output columns: group_cols + date_col + one column per tsfresh feature, with one row
        per (series, window_end_date). This is what gets joined back onto the panel.
        """
        long_pd = self._to_long(df_pl)
        rolled = roll_time_series(
            long_pd,
            column_id="series_id",
            column_sort=self.date_col,
            column_kind="kind",
            max_timeshift=self.max_timeshift,
            min_timeshift=self.min_timeshift,
            rolling_direction=1,
            disable_progressbar=True,
            n_jobs=self.n_jobs,
        )
        # roll_time_series keeps the original id column and adds a tuple "id"
        # (series_id, window_end); drop the old one so it is not treated as data.
        rolled = rolled.drop(columns=["series_id"])

        x = extract_features(
            rolled,
            column_id="id",
            column_sort=self.date_col,
            column_kind="kind",
            column_value="value",
            default_fc_parameters=self._resolve_fc_parameters(),
            impute_function=impute,
            disable_progressbar=True,
            n_jobs=self.n_jobs,
        )
        # The tuple id encodes (series_id, window_end_date): expose it as a MultiIndex.
        x.index = pd.MultiIndex.from_tuples(x.index, names=["series_id", self.date_col])

        feat = pl.from_pandas(x.reset_index())
        parts = pl.col("series_id").str.split_exact("__", 1)
        return (
            feat.with_columns(parts.alias("_parts"))
            .with_columns(
                pl.col("_parts").struct.field("field_0").alias(self.group_cols[0]),
                pl.col("_parts").struct.field("field_1").alias(self.group_cols[1]),
            )
            .drop(["series_id", "_parts"])
            .with_columns(pl.col(self.date_col).cast(df_pl.schema[self.date_col]))
        )

    def _assemble(
        self, df_pl: pl.DataFrame, feat_pl: pl.DataFrame, feature_cols: list[str]
    ) -> pl.DataFrame:
        r"""Left-join the feature frame onto the panel, keeping rows/order and column layout."""
        # Guarantee every fitted feature column exists (null where the battery did not produce
        # it on this slice) and is selected in the recorded order, so train/test align exactly.
        for col in feature_cols:
            if col not in feat_pl.columns:
                feat_pl = feat_pl.with_columns(pl.lit(None).alias(col))
        feat_pl = feat_pl.select([*self.group_cols, self.date_col, *feature_cols])
        return df_pl.join(
            feat_pl, on=[*self.group_cols, self.date_col], how="left"
        ).sort([*self.group_cols, self.date_col])

    def fit(self, df: FrameLike) -> "TSFreshFeatureBuilder":
        r"""Run one extraction pass to record the produced feature column names."""
        for col in self.value_cols:
            if col not in to_polars(df).columns:
                raise KeyError(f"value column '{col}' not found in the input DataFrame.")

        df_pl = to_polars(df).sort([*self.group_cols, self.date_col])
        feat_pl = self._compute_feature_frame(df_pl)
        self.feature_cols_ = [
            c for c in feat_pl.columns if c not in (*self.group_cols, self.date_col)
        ]
        return self

    def transform(self, df: FrameLike) -> tuple[FrameLike, list[str]]:
        r"""
        Append the tsfresh feature columns to df, returning (df, feature_cols).

        The original rows, order and columns are preserved; only the engineered columns are
        added, reindexed to the feature set captured during fit.
        """
        if not self.feature_cols_:
            raise RuntimeError("Call fit (or fit_transform) before transform.")

        return_pandas = is_pandas(df)
        df_pl = to_polars(df).sort([*self.group_cols, self.date_col])
        feat_pl = self._compute_feature_frame(df_pl)
        out_pl = self._assemble(df_pl, feat_pl, self.feature_cols_)

        if return_pandas:
            return out_pl.to_pandas(), list(self.feature_cols_)
        return out_pl, list(self.feature_cols_)

    def fit_transform(self, df: FrameLike) -> tuple[FrameLike, list[str]]:
        r"""Fit on df and return the transformed frame plus the produced feature columns."""
        return_pandas = is_pandas(df)
        df_pl = to_polars(df).sort([*self.group_cols, self.date_col])
        feat_pl = self._compute_feature_frame(df_pl)
        self.feature_cols_ = [
            c for c in feat_pl.columns if c not in (*self.group_cols, self.date_col)
        ]
        out_pl = self._assemble(df_pl, feat_pl, self.feature_cols_)

        if return_pandas:
            return out_pl.to_pandas(), list(self.feature_cols_)
        return out_pl, list(self.feature_cols_)


class FreshFeatureSelector:
    r"""
    FRESH-style univariate feature selection, compatible with the builders' (df, feature_cols) output.

    Reimplements the methodology tsfresh uses in select_features (Christ et al., 2016/2018):

    *   Per-feature hypothesis test. For every candidate feature a univariate test of
        H0 "the feature is irrelevant for the target" is run, and a p-value is computed. The
        test is chosen from the dtypes of the (feature, target) pair, exactly like FRESH:
          - target real,   feature real   -> Kendall's tau rank correlation
          - target real,   feature binary -> Mann-Whitney U (target split by the two feature values)
          - target binary, feature real   -> Mann-Whitney U (feature split by the two classes)
          - target binary, feature binary -> Fisher exact test on the 2x2 table
        All are non-parametric (rank based): no normality or linearity assumed.
    *   Multiple-testing control. Because hundreds/thousands of features are tested at once, the
        raw p-values are corrected with Benjamini-Yekutieli, which bounds the False Discovery Rate
        at fdr_level (default 0.05) and is valid under arbitrary dependence between tests (tsfresh
        features are heavily correlated, so BY is the correct choice, not Benjamini-Hochberg).
    *   It is a univariate FILTER: each feature is judged in isolation against the target, so
        redundancy/interactions are not handled here (leave that to the downstream model).
    *   Needs the target, so fit only on TRAIN. transform then drops the rejected feature columns
        from any frame (train or test) using the stored selection, so no re-selection on test
        (no leakage). Rows, order and the non-feature columns are preserved.
    *   Accepts and returns either polars or pandas (round-trips the input flavour).
    *

    Example
    -------
    >>> builder = TSFreshFeatureBuilder(fc_parameters="efficient")
    >>> df_train_feat, feature_cols = builder.fit_transform(df_train)
    >>> df_test_feat, _ = builder.transform(df_test)
    >>>
    >>> selector = FreshFeatureSelector(fdr_level=0.05)
    >>> df_train_sel, kept = selector.fit_transform(
    ...     df_train_feat, feature_cols, y="target_consumo_real"
    ... )
    >>> df_test_sel, _ = selector.transform(df_test_feat)   # same kept columns, no re-selection
    >>> selector.relevance_table_      # p_value / p_value_adjusted / relevant per feature
    """

    def __init__(
        self,
        *,
        fdr_level: float = 0.05,
        ml_task: str = "auto",
        min_samples: int = 3,
    ) -> None:
        self.fdr_level = fdr_level
        self.ml_task = ml_task  # "auto" | "regression" | "classification"
        self.min_samples = min_samples

        self.feature_cols_: list[str] = []
        self.selected_features_: list[str] = []
        self.relevance_table_: pd.DataFrame | None = None

    def _infer_target_binary(self, y: pd.Series) -> bool:
        r"""Decide whether the target is treated as binary (classification) or real (regression)."""
        if self.ml_task == "classification":
            return True
        if self.ml_task == "regression":
            return False
        return y.dropna().nunique() <= 2

    def _single_p_value(
        self, x: pd.Series, y: pd.Series, target_binary: bool
    ) -> tuple[str, float]:
        r"""Run the FRESH test matching the (feature, target) dtypes and return (test_name, p)."""
        pair = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
        if pair.shape[0] < self.min_samples or pair["x"].nunique() < 2:
            # Constant or near-empty feature carries no information: treat as irrelevant.
            return "constant", 1.0

        feature_binary = pair["x"].nunique() <= 2

        if target_binary:
            classes = pair["y"].unique()
            if classes.size < 2:
                return "constant_target", 1.0
            if feature_binary:
                table = pd.crosstab(pair["x"], pair["y"]).to_numpy()
                if table.shape == (2, 2):
                    return "fisher", float(stats.fisher_exact(table)[1])
                return "chi2", float(stats.chi2_contingency(table)[1])
            g0 = pair.loc[pair["y"] == classes[0], "x"].to_numpy()
            g1 = pair.loc[pair["y"] == classes[1], "x"].to_numpy()
            if g0.size == 0 or g1.size == 0:
                return "mann_whitney", 1.0
            return "mann_whitney", float(
                stats.mannwhitneyu(g0, g1, alternative="two-sided").pvalue
            )

        # Real-valued target.
        if feature_binary:
            values = pair["x"].unique()
            g0 = pair.loc[pair["x"] == values[0], "y"].to_numpy()
            g1 = pair.loc[pair["x"] == values[1], "y"].to_numpy()
            if g0.size == 0 or g1.size == 0:
                return "mann_whitney", 1.0
            return "mann_whitney", float(
                stats.mannwhitneyu(g0, g1, alternative="two-sided").pvalue
            )
        _, p = stats.kendalltau(pair["x"].to_numpy(), pair["y"].to_numpy())
        return "kendall", 1.0 if np.isnan(p) else float(p)

    @staticmethod
    def _benjamini_yekutieli(
        pvals: np.ndarray, alpha: float
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Benjamini-Yekutieli FDR control: return (reject mask, adjusted p-values)."""
        m = pvals.size
        if m == 0:
            return np.array([], dtype=bool), np.array([])

        order = np.argsort(pvals, kind="mergesort")
        ranked = pvals[order]
        # c(m) = sum_{i=1}^m 1/i, the harmonic correction that makes BY valid under dependence.
        c_m = float(np.sum(1.0 / np.arange(1, m + 1)))
        ranks = np.arange(1, m + 1)

        # Step-up rejection: largest k with p_(k) <= (k / (m * c(m))) * alpha rejects ranks 1..k.
        thresholds = (ranks / (m * c_m)) * alpha
        below = ranked <= thresholds
        reject_sorted = np.zeros(m, dtype=bool)
        if below.any():
            k = int(np.max(np.nonzero(below)[0]))
            reject_sorted[: k + 1] = True

        # BY-adjusted p-values: enforce monotonicity with a cumulative min from the largest rank.
        adj_sorted = ranked * m * c_m / ranks
        adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
        adj_sorted = np.clip(adj_sorted, 0.0, 1.0)

        reject = np.empty(m, dtype=bool)
        p_adj = np.empty(m)
        reject[order] = reject_sorted
        p_adj[order] = adj_sorted
        return reject, p_adj

    def fit(
        self, df: FrameLike, feature_cols: list[str], y: str | np.ndarray | pd.Series
    ) -> "FreshFeatureSelector":
        r"""Score every feature, apply BY correction and record the selected feature set."""
        df_pd = to_polars(df).to_pandas()
        if isinstance(y, str):
            if y not in df_pd.columns:
                raise KeyError(f"target column '{y}' not found in the input DataFrame.")
            y_series = df_pd[y]
        else:
            y_series = pd.Series(np.asarray(y), index=df_pd.index)
        if len(y_series) != len(df_pd):
            raise ValueError("y length does not match the number of rows in df.")

        missing = [c for c in feature_cols if c not in df_pd.columns]
        if missing:
            raise KeyError(f"feature columns missing from df: {missing[:5]}")

        # FRESH needs a defined target: rows with a null target cannot vote on relevance.
        valid = y_series.notna()
        df_pd = df_pd.loc[valid]
        y_series = y_series.loc[valid]
        target_binary = self._infer_target_binary(y_series)

        self.feature_cols_ = list(feature_cols)
        records: list[dict] = []
        for col in self.feature_cols_:
            test_name, p_value = self._single_p_value(
                df_pd[col], y_series, target_binary
            )
            records.append({"feature": col, "test": test_name, "p_value": p_value})

        pvals = np.array([r["p_value"] for r in records], dtype=float)
        reject, p_adj = self._benjamini_yekutieli(pvals, self.fdr_level)

        table = pd.DataFrame(records)
        table["p_value_adjusted"] = p_adj
        table["relevant"] = reject
        table["target_kind"] = "binary" if target_binary else "real"
        self.relevance_table_ = table.sort_values("p_value").reset_index(drop=True)

        self.selected_features_ = [
            r["feature"] for r, keep in zip(records, reject) if keep
        ]
        return self

    def transform(self, df: FrameLike) -> tuple[FrameLike, list[str]]:
        r"""Drop the rejected feature columns from df, returning (df, selected_features)."""
        if self.relevance_table_ is None:
            raise RuntimeError("Call fit (or fit_transform) before transform.")

        return_pandas = is_pandas(df)
        df_pl = to_polars(df)
        rejected = [
            c
            for c in self.feature_cols_
            if c not in set(self.selected_features_) and c in df_pl.columns
        ]
        out_pl = df_pl.drop(rejected)

        if return_pandas:
            return out_pl.to_pandas(), list(self.selected_features_)
        return out_pl, list(self.selected_features_)

    def fit_transform(
        self, df: FrameLike, feature_cols: list[str], y: str | np.ndarray | pd.Series
    ) -> tuple[FrameLike, list[str]]:
        r"""Fit the selection on df and return the frame restricted to the selected features."""
        return self.fit(df, feature_cols, y).transform(df)
