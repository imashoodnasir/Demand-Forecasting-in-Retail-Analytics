from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass
class Config:
    # Paths
    project_dir: str = "."
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    output_dir: str = "outputs"
    checkpoint_dir: str = "outputs/checkpoints"
    log_dir: str = "outputs/logs"
    figure_dir: str = "outputs/figures"

    # Data columns
    series_id_col: str = "series_id"
    time_col: str = "timestamp"
    target_col: str = "target"
    static_categorical_cols: tuple = ()
    known_future_covariate_cols: tuple = ()
    observed_covariate_cols: tuple = ()

    # Windowing
    input_length: int = 56
    forecast_horizon: int = 14
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    min_series_length: int = 120

    # Training
    batch_size: int = 32
    num_workers: int = 0
    epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    early_stopping_patience: int = 5
    loss_name: str = "huber"  # mse, mae, huber

    # Model
    d_model: int = 128
    n_heads: int = 4
    num_encoder_layers: int = 3
    ff_dim: int = 256
    dropout: float = 0.1
    use_learnable_positional_encoding: bool = False

    # Reproducibility
    seed: int = 42
    device: str = "cpu"

    # Experiment
    dataset_name: str = "retail_h"
    save_every_epoch: bool = False

    def save_json(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)


DEFAULT_CONFIG = Config()
