from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import polars as pl
from tsfresh import extract_features, select_features
from tsfresh.feature_extraction.settings import (
    EfficientFCParameters,
    from_columns,
)
from tsfresh.utilities.dataframe_functions import impute, roll_time_series

FrameLike = pl.DataFrame | pd.DataFrame


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


def _to_long(
    df: pl.DataFrame, value_col: str, group_cols: tuple[str, str], date_col: str
) -> pd.DataFrame:
    r"""
    Reshape the panel into the tsfresh long format expected by roll_time_series.

    *   Fuses the two group columns into a single series_id (tsfresh needs one id column).
    *   Returns a pandas frame with columns: series_id, <date_col>, <value_col>.
    *
    """
    return (
        df.with_columns(
            pl.concat_str(
                [pl.col(c).cast(pl.String) for c in group_cols], separator="__"
            ).alias("series_id")
        )
        .select(["series_id", date_col, value_col])
        .sort(["series_id", date_col])
        .to_pandas()
    )


def _build_target(
    df: pl.DataFrame, target_col: str, group_cols: tuple[str, str], date_col: str
) -> pd.Series:
    r"""
    Build the forecasting target indexed by (series_id, date) to align with extract_features.

    *   The rolled window ending at day t is paired with target_col at row t, which already
        holds the consumption of t+1 (shifted upstream in transform_data) -> no leakage.
    *
    """
    return (
        df.with_columns(
            pl.concat_str(
                [pl.col(c).cast(pl.String) for c in group_cols], separator="__"
            ).alias("series_id")
        )
        .select(["series_id", date_col, target_col])
        .to_pandas()
        .set_index(["series_id", date_col])[target_col]
    )


def _extract(
    rolled: pd.DataFrame,
    date_col: str,
    default_fc_parameters: dict | None = None,
    kind_to_fc_parameters: dict | None = None,
) -> pd.DataFrame:
    r"""Run extract_features on rolled data and normalise the index to a MultiIndex."""
    kwargs: dict = dict(
        column_id="id",
        column_sort=date_col,
        impute_function=impute,
        disable_progressbar=True,
        n_jobs=0,  # 0 avoids the Windows multiprocessing __main__ guard requirement
    )
    if kind_to_fc_parameters is not None:
        kwargs["kind_to_fc_parameters"] = kind_to_fc_parameters
    else:
        kwargs["default_fc_parameters"] = default_fc_parameters or EfficientFCParameters()

    x = extract_features(rolled, **kwargs)
    # roll_time_series ids are tuples (series_id, window_end_date); expose them as a MultiIndex
    x.index = pd.MultiIndex.from_tuples(x.index, names=["series_id", date_col])
    return x


def _features_to_polars(
    features: pd.DataFrame, group_cols: tuple[str, str], date_col: str
) -> pl.DataFrame:
    r"""Convert the (series_id, date)-indexed feature matrix back to a joinable polars frame."""
    feat = pl.from_pandas(features.reset_index())
    parts = pl.col("series_id").str.split_exact("__", 1)
    return (
        feat.with_columns(parts.alias("_parts"))
        .with_columns(
            pl.col("_parts").struct.field("field_0").alias(group_cols[0]),
            pl.col("_parts").struct.field("field_1").alias(group_cols[1]),
        )
        .drop(["series_id", "_parts"])
    )


def build_features(
    df: FrameLike,
    sku_stats: dict | None = None,
    *,
    value_col: str = "consumo_real",
    target_col: str = "target_consumo_real",
    group_cols: tuple[str, str] = ("planta", "sku"),
    date_col: str = "fecha",
    max_timeshift: int = 20,
    min_timeshift: int = 5,
    default_fc_parameters: dict | None = None,
) -> tuple[FrameLike, dict]:
    r"""
    Build forecasting features with tsfresh on top of value_col, returning (df, sku_stats).

    *   Reshapes the panel and rolls a window (max_timeshift / min_timeshift) per series so
        every (series, day t) becomes a sub-series ending at t (tsfresh roll_time_series).
    *   On train (sku_stats is None): extracts the full tsfresh battery, then runs the FRESH
        hypothesis-test selection against target_col (consumption of t+1) and stores the
        chosen feature set as a reusable kind_to_fc_parameters spec inside sku_stats.
    *   On test (sku_stats provided): extracts ONLY the selected features via the stored spec
        and never re-runs selection, so the train/test feature set is identical (no leakage).
    *   Features are joined back onto the original rows by (group_cols, date_col). A lag_1
        persistence helper is kept for downstream baselines.
    *   Accepts and returns either polars or pandas (round-trips the input flavour).
    *
    """
    return_pandas = is_pandas(df)
    df_pl = to_polars(df).sort([*group_cols, date_col])

    if target_col not in df_pl.columns:
        df_pl = df_pl.with_columns(
            pl.col(value_col).shift(-1).over(list(group_cols)).alias(target_col)
        )

    long_df = _to_long(df_pl, value_col, group_cols, date_col)
    rolled = roll_time_series(
        long_df,
        column_id="series_id",
        column_sort=date_col,
        max_timeshift=max_timeshift,
        min_timeshift=min_timeshift,
        rolling_direction=1,
        disable_progressbar=True,
        n_jobs=0,
    )
    # roll_time_series keeps the original id column; drop it so extract_features does not
    # treat the string series_id as a value column (the new tuple "id" replaces it).
    rolled = rolled.drop(columns=["series_id"])

    if sku_stats is None:
        # TRAIN: extract everything, then keep only the FRESH-relevant features.
        x = _extract(rolled, date_col, default_fc_parameters=default_fc_parameters)

        y = _build_target(df_pl, target_col, group_cols, date_col).reindex(x.index)
        mask = y.notna()
        x_selected = select_features(x[mask], y[mask])
        if x_selected.shape[1] == 0:
            # FRESH found nothing significant: fall back to the full extracted battery so
            # downstream models are not handed an empty feature matrix.
            x_selected = x[mask]

        selected_columns = list(x_selected.columns)
        sku_stats = {
            "value_col": value_col,
            "selected_columns": selected_columns,
            "kind_to_fc_parameters": from_columns(x_selected),
            "max_timeshift": max_timeshift,
            "min_timeshift": min_timeshift,
        }
        features = x[selected_columns]
    else:
        # TEST: reproduce exactly the train-selected features (no selection here).
        x = _extract(
            rolled,
            date_col,
            kind_to_fc_parameters=sku_stats["kind_to_fc_parameters"],
        )
        for col in sku_stats["selected_columns"]:
            if col not in x.columns:
                x[col] = 0.0
        features = x[sku_stats["selected_columns"]]

    feat_pl = _features_to_polars(features, group_cols, date_col)
    # Align the date dtype with the input (Date for transform_data output, Datetime for
    # a pandas round-trip) so the join keys match.
    feat_pl = feat_pl.with_columns(pl.col(date_col).cast(df_pl.schema[date_col]))
    out = (
        df_pl.join(feat_pl, on=[*group_cols, date_col], how="left")
        .sort([*group_cols, date_col])
        .with_columns(
            lag_1=pl.col("consumo_real").shift(1).over(list(group_cols))
        )
    )

    if return_pandas:
        return out.to_pandas(), sku_stats
    return out, sku_stats


def _average_mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    r"""Compute average mutual information between two 1D arrays."""
    import numpy as np

    c_xy, bin_x, bin_y = np.histogram2d(x, y, bins=bins)
    c_x, _ = np.histogram(x, bins=bin_x)
    c_y, _ = np.histogram(y, bins=bin_y)

    n = len(x)
    p_xy = c_xy / n
    p_x = c_x / n
    p_y = c_y / n

    ami = 0.0
    for i in range(len(p_x)):
        for j in range(len(p_y)):
            if p_xy[i, j] > 0.0 and p_x[i] > 0.0 and p_y[j] > 0.0:
                ami += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))
    return float(ami)


def estimate_delay_ami(series: np.ndarray, max_tau: int = 30) -> int:
    r"""Estimate the delay tau using average mutual information first local minimum."""
    import numpy as np

    s = series[~np.isnan(series)]
    if s.size <= 2:
        return 1
    max_tau = min(max_tau, s.size // 3)
    if max_tau < 1:
        return 1

    ami_curve = []
    for k in range(1, max_tau + 1):
        ami = _average_mutual_information(s[:-k], s[k:])
        ami_curve.append(ami)

    for i in range(1, len(ami_curve) - 1):
        if ami_curve[i] < ami_curve[i - 1] and ami_curve[i] < ami_curve[i + 1]:
            return i + 1

    if len(ami_curve) > 0:
        return int(np.argmin(ami_curve) + 1)
    return 1


def estimate_dimension_fnn(
    series: np.ndarray,
    tau: int,
    max_m: int = 10,
    r_tol: float = 15.0,
    a_tol: float = 2.0,
) -> int:
    r"""Estimate embedding dimension m using false nearest neighbors method."""
    import numpy as np
    from scipy.spatial import KDTree

    s = series[~np.isnan(series)]
    n = s.size
    std_s = np.std(s)
    if std_s == 0.0:
        return 2

    fnn_percentages = []
    for m in range(1, max_m):
        start_idx = (m - 1) * tau
        if start_idx >= n - 1:
            break

        points_m = []
        for i in range(start_idx, n):
            vec = [s[i - j * tau] for j in range(m)]
            points_m.append(vec)

        points_m = np.array(points_m)
        num_points = len(points_m)
        if num_points < 3:
            break

        tree = KDTree(points_m)
        dists, indices = tree.query(points_m, k=2)
        r_d = dists[:, 1]
        nn_idx = indices[:, 1]

        false_neighbors = 0
        valid_points = 0

        for idx in range(num_points):
            orig_t = start_idx + idx
            if orig_t - m * tau < 0:
                continue

            nn_t = start_idx + nn_idx[idx]
            if nn_t - m * tau < 0:
                continue

            valid_points += 1
            dist_m = r_d[idx]
            if dist_m == 0.0:
                diff = abs(s[orig_t - m * tau] - s[nn_t - m * tau])
                if diff > 0.0:
                    false_neighbors += 1
                continue

            diff = abs(s[orig_t - m * tau] - s[nn_t - m * tau])
            cond1 = diff / dist_m > r_tol
            dist_m1 = np.sqrt(dist_m**2 + diff**2)
            cond2 = dist_m1 / std_s > a_tol

            if cond1 or cond2:
                false_neighbors += 1

        if valid_points == 0:
            break

        fnn_pct = false_neighbors / valid_points
        fnn_percentages.append(fnn_pct)

        if fnn_pct < 0.01:
            return m

    for m_idx, pct in enumerate(fnn_percentages):
        if pct < 0.10:
            return m_idx + 1

    if len(fnn_percentages) > 0:
        return int(np.argmin(fnn_percentages) + 1)
    return 2


class TSFreshFeatureExtractor:
    r"""
    Encapsulate the full tsfresh rolling + extraction pipeline for daily SKU forecasting.

    *   fit_transform rolls every series, extracts the EfficientFCParameters battery, builds
        the next-day target and keeps only the FRESH-relevant features (no leakage, the target
        is the t+1 consumption aligned to the window ending at t).
    *   transform_latest reproduces rolling + extraction on new data and returns the last window
        per SKU, restricted to the features that survived selection, ready to predict tomorrow.
    *   Chronological only: never shuffles, never random-splits.
    *
    """

    def __init__(
        self,
        *,
        column_id: str = "id",
        column_sort: str = "fecha",
        column_value: str = "consumo_real",
        max_timeshift: int = 30,
        min_timeshift: int = 7,
        n_jobs: int = 4,
    ) -> None:
        self.column_id = column_id
        self.column_sort = column_sort
        self.column_value = column_value
        self.max_timeshift = max_timeshift
        self.min_timeshift = min_timeshift
        self.n_jobs = n_jobs

        self._X_filtrado: pd.DataFrame | None = None
        self._y_series: pd.Series | None = None
        self._selected_columns: list[str] | None = None

    def _to_pandas_sorted(self, df_pl: pl.DataFrame) -> pd.DataFrame:
        r"""Step 1: Polars -> pandas, parse the date, collapse to one row per (id, date), sort."""
        df_pd = df_pl.to_pandas()
        df_pd[self.column_sort] = pd.to_datetime(df_pd[self.column_sort])
        # Guarantee daily granularity: tsfresh and the (id, date) index require a unique
        # (id, date) key, so duplicate same-day records are summed into a single observation.
        df_pd = df_pd.groupby(
            [self.column_id, self.column_sort], as_index=False
        )[self.column_value].sum()
        return df_pd.sort_values([self.column_id, self.column_sort]).reset_index(drop=True)

    def _roll_and_extract(self, df_pd: pd.DataFrame) -> pd.DataFrame:
        r"""Steps 2-3: roll_time_series then extract_features, indexed by (id, window_end)."""
        print("🔄 Rolling...")
        rolled = roll_time_series(
            df_pd,
            column_id=self.column_id,
            column_sort=self.column_sort,
            max_timeshift=self.max_timeshift,
            min_timeshift=self.min_timeshift,
            n_jobs=self.n_jobs,
        )

        print("⚙️ Extrayendo features...")
        x = extract_features(
            rolled,
            column_id=self.column_id,
            column_sort=self.column_sort,
            column_value=self.column_value,
            default_fc_parameters=EfficientFCParameters(),
            impute_function=impute,
            n_jobs=self.n_jobs,
        )
        # roll_time_series ids are tuples (id, window_end_date); expose them as a MultiIndex.
        x.index = pd.MultiIndex.from_tuples(
            x.index, names=[self.column_id, self.column_sort]
        )
        return x

    def _build_target(self, df_pd: pd.DataFrame) -> pd.Series:
        r"""Step 4: next-day consumption (shift -1 within each id), indexed by (id, date)."""
        target = df_pd.copy()
        target["__target__"] = target.groupby(self.column_id)[self.column_value].shift(-1)
        return target.set_index([self.column_id, self.column_sort])["__target__"]

    def fit_transform(self, df_pl: pl.DataFrame) -> pd.DataFrame:
        r"""Roll, extract, build the target and return the FRESH-selected aligned feature matrix."""
        df_pd = self._to_pandas_sorted(df_pl)
        x = self._roll_and_extract(df_pd)

        # Step 4-5: align X with the next-day target on the shared (id, window_end) index.
        y_full = self._build_target(df_pd).reindex(x.index)
        mask = y_full.notna()
        x_alineado = x[mask]
        y_series = y_full[mask]

        # Step 6: FRESH hypothesis-test selection against the next-day target.
        print("🎯 Seleccionando features relevantes...")
        x_filtrado = select_features(x_alineado, y_series)
        if x_filtrado.shape[1] == 0:
            # FRESH found nothing significant: fall back to the full extracted battery so
            # downstream models are not handed an empty feature matrix.
            print("warning: select_features kept 0 features; falling back to the full battery.")
            x_filtrado = x_alineado

        # Step 7: persist the fitted state.
        self._X_filtrado = x_filtrado
        self._y_series = y_series
        self._selected_columns = list(x_filtrado.columns)
        return self._X_filtrado

    def get_target(self) -> pd.Series:
        r"""Return the next-day target series built during fit_transform."""
        if self._y_series is None:
            raise RuntimeError("Call fit_transform before get_target.")
        return self._y_series

    def transform_latest(self, df_pl: pl.DataFrame) -> pd.DataFrame:
        r"""Return the last window per SKU (X_hoy) restricted to the selected feature set."""
        if self._selected_columns is None:
            raise RuntimeError("Call fit_transform before transform_latest.")

        df_pd = self._to_pandas_sorted(df_pl)
        x = self._roll_and_extract(df_pd)

        # Keep only the features that survived select_features; backfill any missing one.
        for col in self._selected_columns:
            if col not in x.columns:
                x[col] = 0.0
        x = x[self._selected_columns]
        return x.groupby(level=0).last()

    def save(self, path: str) -> None:
        r"""Serialize the fitted extractor (state + hyperparameters) with joblib."""
        joblib.dump(
            {
                "column_id": self.column_id,
                "column_sort": self.column_sort,
                "column_value": self.column_value,
                "max_timeshift": self.max_timeshift,
                "min_timeshift": self.min_timeshift,
                "n_jobs": self.n_jobs,
                "_X_filtrado": self._X_filtrado,
                "_y_series": self._y_series,
                "_selected_columns": self._selected_columns,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "TSFreshFeatureExtractor":
        r"""Deserialize a fitted extractor previously written with save."""
        state = joblib.load(path)
        obj = cls(
            column_id=state["column_id"],
            column_sort=state["column_sort"],
            column_value=state["column_value"],
            max_timeshift=state["max_timeshift"],
            min_timeshift=state["min_timeshift"],
            n_jobs=state["n_jobs"],
        )
        obj._X_filtrado = state["_X_filtrado"]
        obj._y_series = state["_y_series"]
        obj._selected_columns = state["_selected_columns"]
        return obj
