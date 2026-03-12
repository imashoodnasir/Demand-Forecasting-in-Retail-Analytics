from __future__ import annotations
import torch
from torch.utils.data import Dataset


class DemandForecastDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "series_id": s["series_id"],
            "x_hist": torch.tensor(s["x_hist"], dtype=torch.float32),
            "x_future_cov": torch.tensor(s["x_future_cov"], dtype=torch.float32),
            "y": torch.tensor(s["y"], dtype=torch.float32),
            "y_raw": torch.tensor(s["y_raw"], dtype=torch.float32),
            "train_hist_scaled": torch.tensor(s["train_hist_scaled"], dtype=torch.float32),
        }
