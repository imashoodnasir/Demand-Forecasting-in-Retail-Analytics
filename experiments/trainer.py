from __future__ import annotations
from dataclasses import dataclass
import copy
import time
import torch
import torch.nn as nn
from tqdm import tqdm


@dataclass
class TrainResult:
    train_losses: list[float]
    val_losses: list[float]
    best_state_dict: dict
    best_val_loss: float
    training_time_sec: float


def get_loss(loss_name: str) -> nn.Module:
    loss_name = loss_name.lower()
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "mae":
        return nn.L1Loss()
    return nn.HuberLoss()


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device):
    model.eval()
    losses = []
    for batch in loader:
        x_hist = batch["x_hist"].to(device)
        x_future_cov = batch["x_future_cov"].to(device)
        y = batch["y"].to(device)
        pred = model(x_hist, x_future_cov)
        loss = criterion(pred, y)
        losses.append(loss.item())
    return sum(losses) / max(len(losses), 1)


def fit(model, train_loader, val_loader, config):
    device = torch.device(config.device)
    model = model.to(device)
    criterion = get_loss(config.loss_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience = 0
    train_losses, val_losses = [], []
    start = time.time()

    for epoch in range(config.epochs):
        model.train()
        running = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}", leave=False):
            x_hist = batch["x_hist"].to(device)
            x_future_cov = batch["x_future_cov"].to(device)
            y = batch["y"].to(device)
            optimizer.zero_grad()
            pred = model(x_hist, x_future_cov)
            loss = criterion(pred, y)
            loss.backward()
            if config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            running.append(loss.item())

        train_loss = sum(running) / max(len(running), 1)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                break

    elapsed = time.time() - start
    model.load_state_dict(best_state)
    return TrainResult(train_losses, val_losses, best_state, best_val, elapsed), model
