from __future__ import annotations
import argparse
import copy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import Config
from utils.reproducibility import set_seed
from utils.io import save_json
from utils.plotting import plot_noise_robustness
from data.preprocessing import add_time_covariates, standardize_target_per_series, build_windows, chronological_split_indices
from data.dataset import DemandForecastDataset
from models.covariate_transformer import CovariateAwareTransformer
from models.baselines import LSTMBaseline
from experiments.trainer import fit
from experiments.evaluator import predict, evaluate_predictions


def split_by_time(samples, val_ratio, test_ratio):
    n = len(samples)
    tr, va, te = chronological_split_indices(n, val_ratio, test_ratio)
    return samples[tr], samples[va], samples[te]


class NoisyDataset(DemandForecastDataset):
    def __init__(self, samples, noise_std=0.0):
        super().__init__(samples)
        self.noise_std = noise_std

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        if self.noise_std > 0:
            noise = torch.randn_like(item["x_hist"]) * self.noise_std
            item["x_hist"] = item["x_hist"] + noise
        return item


def prepare_data(csv_path, config):
    df = pd.read_csv(csv_path)
    df = add_time_covariates(df)
    obs = ["promo", "holiday", "dayofweek", "month", "weekofyear", "is_weekend"]
    fut = ["promo", "holiday", "dayofweek", "month", "weekofyear", "is_weekend"]
    df, _ = standardize_target_per_series(df, config.series_id_col, config.target_col)
    samples = build_windows(df, config.series_id_col, config.time_col, config.target_col + "_scaled", obs, fut, config.input_length, config.forecast_horizon, config.min_series_length)
    return split_by_time(samples, config.val_ratio, config.test_ratio)


def train_models(train_samples, val_samples, config):
    train_loader = DataLoader(DemandForecastDataset(train_samples), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(DemandForecastDataset(val_samples), batch_size=config.batch_size, shuffle=False)
    input_dim = train_samples[0]["x_hist"].shape[1]
    future_cov_dim = train_samples[0]["x_future_cov"].shape[1]

    proposed = CovariateAwareTransformer(input_dim, future_cov_dim, config.input_length, config.forecast_horizon, config.d_model, config.n_heads, config.num_encoder_layers, config.ff_dim, config.dropout, config.use_learnable_positional_encoding, True)
    _, proposed = fit(proposed, train_loader, val_loader, config)

    rnn = LSTMBaseline(input_dim=input_dim, horizon=config.forecast_horizon)
    _, rnn = fit(rnn, train_loader, val_loader, config)

    transformer_baseline = CovariateAwareTransformer(input_dim, 0, config.input_length, config.forecast_horizon, config.d_model, config.n_heads, config.num_encoder_layers, config.ff_dim, config.dropout, config.use_learnable_positional_encoding, False)
    _, transformer_baseline = fit(transformer_baseline, train_loader, val_loader, config)
    return {"RNN Baseline": rnn, "Transformer Baseline": transformer_baseline, "Proposed Model": proposed}


def evaluate_with_noise(model, test_samples, config, noise_std):
    loader = DataLoader(NoisyDataset(copy.deepcopy(test_samples), noise_std), batch_size=config.batch_size, shuffle=False)
    preds, ys, ys_raw, train_hists = predict(model, loader, torch.device(config.device))
    return evaluate_predictions(preds, ys, ys_raw, train_hists)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/raw/retail_h.csv")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    config = Config(device=args.device)
    set_seed(config.seed)
    train_samples, val_samples, test_samples = prepare_data(args.csv, config)
    models = train_models(train_samples, val_samples, config)

    noise_levels = [0, 10, 20, 30, 40]
    results = {}
    curves = {}
    for model_name, model in models.items():
        clean_metric = evaluate_with_noise(model, test_samples, config, 0.0)["rmse_scaled"]
        degradations = []
        per_noise = {}
        for nl in noise_levels:
            metrics = evaluate_with_noise(model, test_samples, config, nl / 100.0)
            deg = ((metrics["rmse_scaled"] - clean_metric) / max(clean_metric, 1e-8)) * 100.0
            degradations.append(float(deg))
            per_noise[str(nl)] = metrics
        curves[model_name] = degradations
        results[model_name] = per_noise

    save_json(results, f"{config.log_dir}/robustness_results.json")
    plot_noise_robustness(noise_levels, curves, f"{config.figure_dir}/robustness_curve.png")
    print(curves)


if __name__ == "__main__":
    main()
