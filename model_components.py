"""Custom estimators used in the serialized house-price pipelines."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin


def inverse_log_price(values):
    """Invert log prices while respecting the training target's quality cap."""
    return np.expm1(np.clip(values, 0.0, np.log1p(100_000_000.0)))


class TorchMLPRegressor(RegressorMixin, BaseEstimator):
    """Small tabular MLP trained in batches, using CUDA when available."""

    def __init__(self, device="cuda", epochs=12, batch_size=16384, learning_rate=0.002, random_state=42):
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        self.device_ = "cuda" if self.device == "cuda" and torch.cuda.is_available() else "cpu"
        device = torch.device(self.device_)
        features = np.asarray(X, dtype=np.float32)
        target = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        x_tensor = torch.from_numpy(features).to(device)
        y_tensor = torch.from_numpy(target).to(device)
        torch_model = torch.nn.Sequential(
            torch.nn.Linear(features.shape[1], 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        ).to(device)
        optimizer = torch.optim.AdamW(torch_model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        loss_function = torch.nn.MSELoss()
        row_count = len(features)
        torch_model.train()
        for epoch in range(self.epochs):
            order = torch.randperm(row_count, device=device)
            total_loss = 0.0
            seen = 0
            for start in range(0, row_count, self.batch_size):
                indices = order[start:start + self.batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(torch_model(x_tensor[indices]), y_tensor[indices])
                loss.backward()
                optimizer.step()
                batch_rows = len(indices)
                total_loss += float(loss.detach()) * batch_rows
                seen += batch_rows
            print(f"    Neural network epoch {epoch + 1}/{self.epochs}: log-price MSE {total_loss / seen:.4f}")
        self.model_ = torch_model.to("cpu").eval()
        self.n_features_in_ = features.shape[1]
        self.is_fitted_ = True
        del x_tensor, y_tensor
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return self

    def predict(self, X):
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("The neural network must be fitted before prediction.")
        features = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            return self.model_(torch.from_numpy(features)).numpy().reshape(-1)
