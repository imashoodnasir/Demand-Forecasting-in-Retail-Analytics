from __future__ import annotations
import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.clip(np.abs(y_true), eps, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.clip(denom, eps, None)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def wrmsse_proxy(y_true: np.ndarray, y_pred: np.ndarray, train_hist: np.ndarray | None = None) -> float:
    """
    Proxy implementation for reproducible experiments when full benchmark hierarchy is not available.
    If train history is provided, scale by in-sample naive-1 error; otherwise falls back to RMSE.
    """
    if train_hist is None or len(train_hist) < 2:
        return rmse(y_true, y_pred)
    denom = np.mean((train_hist[1:] - train_hist[:-1]) ** 2)
    denom = max(denom, 1e-8)
    rmsse = np.sqrt(np.mean((y_true - y_pred) ** 2) / denom)
    return float(rmsse)


def confidence_interval_95(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    ci = 1.96 * std / np.sqrt(max(len(arr), 1))
    return mean, mean - ci, mean + ci
