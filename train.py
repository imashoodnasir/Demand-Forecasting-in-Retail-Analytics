from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
import torch

from config import Config
from utils.reproducibility import set_seed
from utils.io import save_json, ensure_dir
from utils.plotting import plot_training_curves
from data.preprocessing import add_time_covariates, standardize_target_per_series, build_windows, chronological_split_indices
from data.dataset import DemandForecastDataset
from models.covariate_transformer import CovariateAwareTransformer
from experiments.trainer import fit
from experiments.evaluator import predict, evaluate_predictions


def split_by_time(samples, val_ratio, test_ratio):
    n = len(samples)
    tr, va, te = chronological_split_indices(n, val_ratio, test_ratio)
    return samples[tr], samples[va], samples[te]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/raw/retail_h.csv")
    parser.add_argument("--dataset_name", type=str, default="retail_h")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    config = Config(dataset_name=args.dataset_name, device=args.device)
    set_seed(config.seed)

    df = pd.read_csv(args.csv)
    df = add_time_covariates(df, time_col=config.time_col)
    observed_covariates = ["promo", "holiday", "dayofweek", "month", "weekofyear", "is_weekend"]
    future_covariates = ["promo", "holiday", "dayofweek", "month", "weekofyear", "is_weekend"]
    df, scalers = standardize_target_per_series(df, config.series_id_col, config.target_col)

    samples = build_windows(
        df=df,
        series_id_col=config.series_id_col,
        time_col=config.time_col,
        target_col_scaled=config.target_col + "_scaled",
        observed_cov_cols=observed_covariates,
        future_cov_cols=future_covariates,
        input_length=config.input_length,
        horizon=config.forecast_horizon,
        min_series_length=config.min_series_length,
    )
    train_samples, val_samples, test_samples = split_by_time(samples, config.val_ratio, config.test_ratio)

    train_loader = DataLoader(DemandForecastDataset(train_samples), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(DemandForecastDataset(val_samples), batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(DemandForecastDataset(test_samples), batch_size=config.batch_size, shuffle=False)

    input_dim = train_samples[0]["x_hist"].shape[1]
    future_cov_dim = train_samples[0]["x_future_cov"].shape[1] if train_samples[0]["x_future_cov"].ndim == 2 else 0
    model = CovariateAwareTransformer(
        input_dim=input_dim,
        future_cov_dim=future_cov_dim,
        input_length=config.input_length,
        horizon=config.forecast_horizon,
        d_model=config.d_model,
        n_heads=config.n_heads,
        num_encoder_layers=config.num_encoder_layers,
        ff_dim=config.ff_dim,
        dropout=config.dropout,
        learnable_pe=config.use_learnable_positional_encoding,
        use_covariates=True,
    )

    result, best_model = fit(model, train_loader, val_loader, config)
    ckpt_dir = ensure_dir(config.checkpoint_dir)
    ckpt_path = ckpt_dir / f"best_{config.dataset_name}.pt"
    torch.save(best_model.state_dict(), ckpt_path)

    preds, ys, ys_raw, train_hists = predict(best_model, test_loader, torch.device(config.device))
    metrics = evaluate_predictions(preds, ys, ys_raw, train_hists)
    metrics["best_val_loss"] = result.best_val_loss
    metrics["training_time_sec"] = result.training_time_sec
    plot_training_curves(result.train_losses, result.val_losses, f"{config.figure_dir}/training_curve_{config.dataset_name}.png")

    save_json(metrics, f"{config.log_dir}/metrics_{config.dataset_name}.json")
    config.save_json(f"{config.log_dir}/config_{config.dataset_name}.json")
    print("Training complete")
    print(metrics)


if __name__ == "__main__":
    main()
