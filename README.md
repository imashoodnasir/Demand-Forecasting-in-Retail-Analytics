# Covariate-Aware Transformer for Long-Horizon Retail Demand Forecasting

This repository provides a full Python implementation template for a paper-style workflow on long-horizon demand forecasting in retail analytics using a covariate-aware transformer. It includes data preparation, window generation, model training, evaluation, ablation analysis, robustness testing, and cross-dataset generalization.

## Project structure

```text
retail_transformer_project/
├── config.py
├── train.py
├── evaluate.py
├── ablation.py
├── robustness.py
├── cross_dataset.py
├── requirements.txt
├── README.md
├── data/
│   ├── raw/
│   ├── processed/
│   ├── dataset.py
│   ├── make_sample_data.py
│   └── preprocessing.py
├── experiments/
│   ├── trainer.py
│   └── evaluator.py
├── models/
│   ├── baselines.py
│   ├── covariate_transformer.py
│   └── positional_encoding.py
├── utils/
│   ├── io.py
│   ├── metrics.py
│   ├── plotting.py
│   └── reproducibility.py
└── outputs/
    ├── checkpoints/
    ├── figures/
    └── logs/
```

## What this code covers

The pipeline follows the practical implementation stages typically required by the paper:

1. environment setup and reproducibility
2. raw data loading and standardization
3. temporal covariate engineering
4. per-series target normalization
5. sliding-window sample generation
6. chronological train/validation/test splitting
7. PyTorch dataset and dataloader preparation
8. covariate-aware transformer model construction
9. baseline implementation
10. training and checkpointing
11. evaluation with multiple metrics
12. ablation analysis
13. robustness testing with Gaussian noise
14. cross-dataset transfer evaluation
15. logging and figure generation

## Input data format

Prepare a CSV with at least these columns:

- `series_id`
- `timestamp`
- `target`
- `promo`
- `holiday`

Optional columns can be added later. The current implementation automatically derives time covariates such as day of week, month, week of year, and weekend flags.

Example:

```csv
series_id,timestamp,target,promo,holiday,store_type
S001,2022-01-01,120.5,0,0,1
S001,2022-01-02,118.2,1,0,1
```

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Create sample datasets

```bash
python data/make_sample_data.py
```

This generates:

- `data/raw/retail_h.csv`
- `data/raw/m4_like.csv`

### 3. Train the proposed model

```bash
python train.py --csv data/raw/retail_h.csv --dataset_name retail_h --device cpu
```

Outputs are saved under:

- `outputs/checkpoints/`
- `outputs/logs/`
- `outputs/figures/`

### 4. Evaluate a saved checkpoint

```bash
python evaluate.py --csv data/raw/retail_h.csv --ckpt outputs/checkpoints/best_retail_h.pt --dataset_name retail_h --device cpu
```

### 5. Run ablation experiments

```bash
python ablation.py --csv data/raw/retail_h.csv --device cpu
```

### 6. Run robustness analysis

```bash
python robustness.py --csv data/raw/retail_h.csv --device cpu
```

### 7. Run cross-dataset generalization

```bash
python cross_dataset.py --source_csv data/raw/retail_h.csv --target_csv data/raw/m4_like.csv --device cpu
```

## Notes on the metrics

This implementation includes:

- MAE
- RMSE
- MAPE
- sMAPE
- a WRMSSE-style proxy

The provided WRMSSE implementation is a benchmark-compatible proxy rather than the exact M5-style hierarchical weighted version. To reproduce an exact benchmark score, replace the proxy with the official hierarchy-aware calculation used by the dataset protocol.

## Adapting to exact paper

To match your paper exactly, update the following parts:

- dataset-specific input files
- official split protocol
- exact covariates used in the manuscript
- official WRMSSE or dataset-specific metrics
- baseline list and hyperparameters
- training schedule and number of repeated runs

## Recommended next customizations

For a paper-exact version, you may want to add:

- static categorical embeddings for product or store IDs
- calendar embeddings for richer future covariates
- probabilistic forecasting heads
- multi-quantile loss
- hierarchical reconciliation
- dataset-specific benchmark loaders
- exact Figure 3 to Figure 7 plotting from experiment logs

## Reproducibility

The code sets random seeds for Python, NumPy, and PyTorch. For fully deterministic GPU results, keep in mind that some CUDA operations can still vary by hardware and software version.
