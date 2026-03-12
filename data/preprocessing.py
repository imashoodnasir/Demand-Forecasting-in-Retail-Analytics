from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class SeriesScalerBundle:
    target_scalers: dict[str, StandardScaler]


def add_time_covariates(df: pd.DataFrame, time_col: str = "timestamp") -> pd.DataFrame:
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df["dayofweek"] = df[time_col].dt.dayofweek
    df["dayofmonth"] = df[time_col].dt.day
    df["month"] = df[time_col].dt.month
    df["weekofyear"] = df[time_col].dt.isocalendar().week.astype(int)
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    return df


def standardize_target_per_series(df: pd.DataFrame, series_id_col: str, target_col: str) -> tuple[pd.DataFrame, SeriesScalerBundle]:
    df = df.copy()
    scalers = {}
    scaled = np.zeros(len(df), dtype=float)
    for sid, grp in df.groupby(series_id_col):
        scaler = StandardScaler()
        vals = grp[[target_col]].values.astype(float)
        scaled_vals = scaler.fit_transform(vals).reshape(-1)
        scaled[grp.index.values] = scaled_vals
        scalers[sid] = scaler
    df[target_col + "_scaled"] = scaled
    return df, SeriesScalerBundle(target_scalers=scalers)


def chronological_split_indices(n: int, val_ratio: float, test_ratio: float) -> tuple[slice, slice, slice]:
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    n_train = n - n_val - n_test
    return slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, n)


def build_windows(
    df: pd.DataFrame,
    series_id_col: str,
    time_col: str,
    target_col_scaled: str,
    observed_cov_cols: list[str],
    future_cov_cols: list[str],
    input_length: int,
    horizon: int,
    min_series_length: int,
) -> list[dict[str, Any]]:
    samples = []
    df = df.sort_values([series_id_col, time_col]).reset_index(drop=True)
    all_obs_cols = [target_col_scaled] + observed_cov_cols
    for sid, grp in df.groupby(series_id_col):
        grp = grp.sort_values(time_col).reset_index(drop=True)
        if len(grp) < max(min_series_length, input_length + horizon):
            continue
        obs_values = grp[all_obs_cols].values.astype(float)
        fut_values = grp[future_cov_cols].values.astype(float) if future_cov_cols else np.zeros((len(grp), 0), dtype=float)
        target_values = grp[target_col_scaled].values.astype(float)
        raw_target = grp[target_col_scaled.replace("_scaled", "")].values.astype(float) if target_col_scaled.endswith("_scaled") else target_values
        for end in range(input_length, len(grp) - horizon + 1):
            x_hist = obs_values[end - input_length:end]
            x_future_cov = fut_values[end:end + horizon]
            y = target_values[end:end + horizon]
            y_raw = raw_target[end:end + horizon]
            samples.append({
                "series_id": sid,
                "x_hist": x_hist,
                "x_future_cov": x_future_cov,
                "y": y,
                "y_raw": y_raw,
                "train_hist_scaled": target_values[:end],
            })
    return samples
