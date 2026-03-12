from __future__ import annotations
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import Config
from data.preprocessing import add_time_covariates, standardize_target_per_series, build_windows, chronological_split_indices
from data.dataset import DemandForecastDataset
from models.covariate_transformer import CovariateAwareTransformer
from experiments.evaluator import predict, evaluate_predictions


def split_by_time(samples, val_ratio, test_ratio):
    n = len(samples)
    tr, va, te = chronological_split_indices(n, val_ratio, test_ratio)
    return samples[tr], samples[va], samples[te]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/raw/retail_h.csv")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="retail_h")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    config = Config(dataset_name=args.dataset_name, device=args.device)
    df = pd.read_csv(args.csv)
    df = add_time_covariates(df, time_col=config.time_col)
    observed_covariates = ["promo", "holiday", "dayofweek", "month", "weekofyear", "is_weekend"]
    future_covariates = ["promo", "holiday", "dayofweek", "month", "weekofyear", "is_weekend"]
    df, _ = standardize_target_per_series(df, config.series_id_col, config.target_col)

    samples = build_windows(df, config.series_id_col, config.time_col, config.target_col + "_scaled", observed_covariates, future_covariates, config.input_length, config.forecast_horizon, config.min_series_length)
    _, _, test_samples = split_by_time(samples, config.val_ratio, config.test_ratio)
    test_loader = DataLoader(DemandForecastDataset(test_samples), batch_size=config.batch_size, shuffle=False)

    input_dim = test_samples[0]["x_hist"].shape[1]
    future_cov_dim = test_samples[0]["x_future_cov"].shape[1]
    model = CovariateAwareTransformer(input_dim, future_cov_dim, config.input_length, config.forecast_horizon, config.d_model, config.n_heads, config.num_encoder_layers, config.ff_dim, config.dropout, config.use_learnable_positional_encoding, True)
    model.load_state_dict(torch.load(args.ckpt, map_location=args.device))
    model = model.to(args.device)
    preds, ys, ys_raw, train_hists = predict(model, test_loader, torch.device(args.device))
    metrics = evaluate_predictions(preds, ys, ys_raw, train_hists)
    print(metrics)


if __name__ == "__main__":
    main()
