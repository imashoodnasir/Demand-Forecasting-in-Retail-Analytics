from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def generate_dataset(save_path: str, n_series: int = 30, n_steps: int = 220, seed: int = 42, domain_shift: float = 0.0):
    rng = np.random.default_rng(seed)
    rows = []
    for sid in range(n_series):
        base = rng.uniform(50, 200)
        trend = rng.uniform(-0.05, 0.15) + domain_shift * 0.02
        weekly_amp = rng.uniform(5, 25)
        promo_effect = rng.uniform(10, 30)
        timestamps = pd.date_range("2022-01-01", periods=n_steps, freq="D")
        promo = (rng.random(n_steps) > 0.85).astype(int)
        holiday = ((timestamps.month == 12) & (timestamps.day > 20)).astype(int)
        for t, ts in enumerate(timestamps):
            seasonal = weekly_amp * np.sin(2 * np.pi * t / 7.0)
            y = base + trend * t + seasonal + promo_effect * promo[t] + 12 * holiday[t] + rng.normal(0, 4 + domain_shift)
            rows.append({
                "series_id": f"S{sid:03d}",
                "timestamp": ts,
                "target": max(0.0, y),
                "promo": int(promo[t]),
                "holiday": int(holiday[t]),
                "store_type": int(sid % 3),
            })
    df = pd.DataFrame(rows)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)


if __name__ == "__main__":
    generate_dataset("data/raw/retail_h.csv", n_series=30, n_steps=220, seed=42, domain_shift=0.0)
    generate_dataset("data/raw/m4_like.csv", n_series=30, n_steps=220, seed=123, domain_shift=1.5)
    print("Sample datasets created in data/raw/")
