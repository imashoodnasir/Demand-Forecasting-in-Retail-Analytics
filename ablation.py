from __future__ import annotations
import argparse
import pandas as pd
from torch.utils.data import DataLoader

from config import Config
from utils.reproducibility import set_seed
from utils.io import save_json
from data.preprocessing import add_time_covariates, standardize_target_per_series, build_windows, chronological_split_indices
from data.dataset import DemandForecastDataset
from models.covariate_transformer import CovariateAwareTransformer
from experiments.trainer import fit
from experiments.evaluator import predict, evaluate_predictions
import torch


def split_by_time(samples, val_ratio, test_ratio):
    n = len(samples)
    tr, va, te = chronological_split_indices(n, val_ratio, test_ratio)
    return samples[tr], samples[va], samples[te]


def run_variant(name, train_samples, val_samples, test_samples, config, use_covariates=True, n_heads=None, n_layers=None):
    train_loader = DataLoader(DemandForecastDataset(train_samples), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(DemandForecastDataset(val_samples), batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(DemandForecastDataset(test_samples), batch_size=config.batch_size, shuffle=False)
    input_dim = train_samples[0]["x_hist"].shape[1]
    future_cov_dim = train_samples[0]["x_future_cov"].shape[1]
    model = CovariateAwareTransformer(
        input_dim=input_dim,
        future_cov_dim=future_cov_dim,
        input_length=config.input_length,
        horizon=config.forecast_horizon,
        d_model=config.d_model,
        n_heads=n_heads or config.n_heads,
        num_encoder_layers=n_layers or config.num_encoder_layers,
        ff_dim=config.ff_dim,
        dropout=config.dropout,
        learnable_pe=config.use_learnable_positional_encoding,
        use_covariates=use_covariates,
    )
    _, best_model = fit(model, train_loader, val_loader, config)
    preds, ys, ys_raw, train_hists = predict(best_model, test_loader, torch.device(config.device))
    metrics = evaluate_predictions(preds, ys, ys_raw, train_hists)
    return {"variant": name, **metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/raw/retail_h.csv")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    config = Config(device=args.device)
    set_seed(config.seed)

    df = pd.read_csv(args.csv)
    df = add_time_covariates(df)
    obs = ["promo", "holiday", "dayofweek", "month", "weekofyear", "is_weekend"]
    fut = ["promo", "holiday", "dayofweek", "month", "weekofyear", "is_weekend"]
    df, _ = standardize_target_per_series(df, config.series_id_col, config.target_col)
    samples = build_windows(df, config.series_id_col, config.time_col, config.target_col + "_scaled", obs, fut, config.input_length, config.forecast_horizon, config.min_series_length)
    train_samples, val_samples, test_samples = split_by_time(samples, config.val_ratio, config.test_ratio)

    results = []
    results.append(run_variant("full_model", train_samples, val_samples, test_samples, config, True, config.n_heads, config.num_encoder_layers))
    results.append(run_variant("without_covariates", train_samples, val_samples, test_samples, config, False, config.n_heads, config.num_encoder_layers))
    results.append(run_variant("fewer_heads", train_samples, val_samples, test_samples, config, True, 2, config.num_encoder_layers))
    results.append(run_variant("fewer_layers", train_samples, val_samples, test_samples, config, True, config.n_heads, 1))
    save_json(results, f"{config.log_dir}/ablation_results.json")
    print(results)


if __name__ == "__main__":
    main()
