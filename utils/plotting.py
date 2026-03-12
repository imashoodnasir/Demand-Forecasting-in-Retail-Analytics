from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(train_losses, val_losses, save_path: str):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_predictions(y_true, predictions_dict, save_path: str, title: str = "Forecast vs Ground Truth"):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label="Ground Truth")
    for name, pred in predictions_dict.items():
        plt.plot(pred, label=name)
    plt.xlabel("Time Step")
    plt.ylabel("Demand")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_noise_robustness(noise_levels, curves_dict, save_path: str):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for name, vals in curves_dict.items():
        plt.plot(noise_levels, vals, marker="o", label=name)
    plt.xlabel("Noise Level (%)")
    plt.ylabel("Performance Degradation (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_bar(values_by_group, group_names, method_names, ylabel, save_path):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(group_names))
    width = 0.8 / len(method_names)
    plt.figure(figsize=(9, 5))
    for i, method in enumerate(method_names):
        vals = [values_by_group[g][i] for g in group_names]
        plt.bar(x + (i - (len(method_names)-1)/2)*width, vals, width=width, label=method)
    plt.xticks(x, group_names)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
