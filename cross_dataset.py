from __future__ import annotations
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import Config
from utils.reproducibility import set_seed
from utils.io import save_json
from data.preprocessing import add_time_covariates, standardize_target_per_series, build_windows, chronological_split_indices
from data.dataset import DemandForecastDataset
from models.covariate_transformer import CovariateAwareTransformer
from experiments.trainer import fit
from experiments.evaluator import predict, evaluate_predictions


def split_by_time(samples, val_ratio, test_ratio):
    n = len(samples)
    tr, va, te = chronological_split_indices(n, val_ratio, test_ratio)
    return samples[tr], samples[va], samples[te]


def make_samples(csv_path, config):
    df = pd.read_csv(csv_path)
    df = add_time_covariates(df)
    obs = ["promo", "holiday", "dayofweek", "month", "weekofyear", "is_weekend"]
    fut = ["promo", "holiday", "dayofweek", "month", "weekofyear", "is_weekend"]
    df, _ = standardize_target_per_series(df, config.series_id_col, config.target_col)
    return build_windows(df, config.series_id_col, config.time_col, config.target_col + "_scaled", obs, fut, config.input_length, config.forecast_horizon, config.min_series_length)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_csv", type=str, default="data/raw/retail_h.csv")
    parser.add_argument("--target_csv", type=str, default="data/raw/m4_like.csv")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    config = Config(device=args.device)
    set_seed(config.seed)

    source_samples = make_samples(args.source_csv, config)
    target_samples = make_samples(args.target_csv, config)
    train_samples, val_samples, _ = split_by_time(source_samples, config.val_ratio, config.test_ratio)
    _, _, target_test = split_by_time(target_samples, config.val_ratio, config.test_ratio)

    train_loader = DataLoader(DemandForecastDataset(train_samples), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(DemandForecastDataset(val_samples), batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(DemandForecastDataset(target_test), batch_size=config.batch_size, shuffle=False)

    input_dim = train_samples[0]["x_hist"].shape[1]
    future_cov_dim = train_samples[0]["x_future_cov"].shape[1]
    proposed = CovariateAwareTransformer(input_dim, future_cov_dim, config.input_length, config.forecast_horizon, config.d_model, config.n_heads, config.num_encoder_layers, config.ff_dim, config.dropout, config.use_learnable_positional_encoding, True)
    transformer_baseline = CovariateAwareTransformer(input_dim, 0, config.input_length, config.forecast_horizon, config.d_model, config.n_heads, config.num_encoder_layers, config.ff_dim, config.dropout, config.use_learnable_positional_encoding, False)

    _, proposed = fit(proposed, train_loader, val_loader, config)
    _, transformer_baseline = fit(transformer_baseline, train_loader, val_loader, config)

    results = {}
    for name, model in {"Transformer Baseline": transformer_baseline, "Proposed Model": proposed}.items():
        preds, ys, ys_raw, train_hists = predict(model, test_loader, torch.device(config.device))
        results[name] = evaluate_predictions(preds, ys, ys_raw, train_hists)

    save_json(results, f"{config.log_dir}/cross_dataset_results.json")
    print(results)


if __name__ == "__main__":
    main()
