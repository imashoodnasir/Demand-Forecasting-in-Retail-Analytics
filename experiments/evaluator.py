from __future__ import annotations
import numpy as np
import torch
from utils.metrics import mae, rmse, mape, smape, wrmsse_proxy


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, ys, ys_raw, hists = [], [], [], []
    for batch in loader:
        x_hist = batch["x_hist"].to(device)
        x_future_cov = batch["x_future_cov"].to(device)
        y = batch["y"].cpu().numpy()
        y_raw = batch["y_raw"].cpu().numpy()
        pred = model(x_hist, x_future_cov).cpu().numpy()
        hist = batch["train_hist_scaled"].cpu().numpy()
        preds.append(pred)
        ys.append(y)
        ys_raw.append(y_raw)
        hists.extend(list(hist))
    return np.concatenate(preds, axis=0), np.concatenate(ys, axis=0), np.concatenate(ys_raw, axis=0), hists


def evaluate_predictions(pred_scaled: np.ndarray, y_scaled: np.ndarray, y_raw: np.ndarray, train_hists: list[np.ndarray]) -> dict:
    flat_pred = pred_scaled.reshape(-1)
    flat_y = y_scaled.reshape(-1)
    flat_y_raw = y_raw.reshape(-1)
    metrics = {
        "mae_scaled": mae(flat_y, flat_pred),
        "rmse_scaled": rmse(flat_y, flat_pred),
        "mape_raw_proxy": mape(flat_y_raw, flat_pred),
        "smape_raw_proxy": smape(flat_y_raw, flat_pred),
    }
    wrmsse_vals = []
    for i in range(len(pred_scaled)):
        wrmsse_vals.append(wrmsse_proxy(y_scaled[i], pred_scaled[i], train_hists[i]))
    metrics["wrmsse_proxy"] = float(np.mean(wrmsse_vals))
    return metrics
