import math
import os
import random
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset


class WindowDataset(Dataset):
    """
    A unified PyTorch Dataset for windowed time-series data.
    The data are organised into a dictionary format.

    The dictionary is of form:
        X (torch.Tensor): Input sequences of shape (N, L, D).
        y_label (optional[torch.Tensor]): Target labels of shape (N, L).
        y_forecast (torch.Tensor): Forecast sequences of shape (N, H, D).
    """
    def __init__(self, X: Any, y_forecast: Optional[Any] = None, y_label: Optional[Any] = None):
        # Ensure input is a tensor
        self.X = torch.as_tensor(X, dtype=torch.float32)

        # Handle forecasting targets
        self.y_forecast = (torch.as_tensor(y_forecast, dtype=torch.float32) if y_forecast is not None else None)

        # Handle classification labels
        if y_label is None:
            self.y_label = None
        else:
            if np.issubdtype(np.asarray(y_label).dtype, np.integer):
                self.y_label = torch.as_tensor(y_label, dtype=torch.long)
            else:
                self.y_label = torch.as_tensor(y_label)

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Retrieve a single sample by index and return a dictionary."""
        # Include output window
        out = {"x": self.X[idx]}

        # Conditionally add forecasting and classification tasks
        if self.y_forecast is not None:
            out["y_forecast"] = self.y_forecast[idx]
        if self.y_label is not None:
            out["y_label"] = self.y_label[idx]

        return out


def add_source(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    Add a source metadata column to a Pandas DataFrame, when data is handled that is recorded by different subjects
    or processes.
    Return the augmented DataFrame.
    """
    # Create copy to not modify the original DataFrame
    df = df.copy()

    # Assign the source as a new column
    df["__source__"] = source
    return df


class MLPEncoder(nn.Module):
    """
    MLP that encodes inputs into the latent space.
    """
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: tuple, dropout: float=0.0):
        super().__init__()
        layers = []
        prev = input_dim

        # Iteratively build the hidden layers based on the hidden_dims variable
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True),]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h

        # Add final layer
        layers.append(nn.Linear(prev, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Pass input through the MLP layers.
        """
        z_e = self.net(x)

        return z_e


class MLPDecoder(nn.Module):
    """
    Decodes the latent vector [..., latent_dim] back to [..., L, D].
    """
    def __init__(self, output_dim, latent_dim, hidden_dims, dropout=0.0):
        super().__init__()
        layers = []
        prev = latent_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), ]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, z):
        x_hat = self.net(z)

        return x_hat


def make_loaders_from_out(out, config):
    X_train = out["train_ds"].X
    y_train = out["train_ds"].y_label
    X_val = out["val_ds"].X
    y_val = out["val_ds"].y_label
    X_test = out["test_ds"].X
    y_test = out["test_ds"].y_label

    batch_size = config["train"].get("batch_size", 32)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


def _dc_to_dict(obj):
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return obj


def sample_param(spec: Any) -> Any:
    """
    Sample one hyperparameter value from a specification.
    """
    # If input is a categorical list
    if isinstance(spec, list):
        return random.choice(spec)

    # Ranges / distributions
    if isinstance(spec, tuple):
        # loguniform
        if len(spec) == 3 and spec[0] == "loguniform":
            _, low, high = spec
            if low <= 0 or high <= 0:
                raise ValueError("loguniform bounds must be > 0")
            log_low, log_high = math.log(low), math.log(high)
            return math.exp(random.uniform(log_low, log_high))

        # uniform
        if len(spec) == 2:
            low, high = spec
            return random.uniform(low, high)

    # Fallback: treat as fixed value
    return spec


def to_numpy(a):
    if a is None:
        return None
    if isinstance(a, np.ndarray):
        return a
    if torch.is_tensor(a):
        return a.detach().cpu().numpy()
    raise TypeError(f"Unsupported type for saving: {type(a)}")


def save_split(split_name, data_dict, folder, dataset):
    ds = data_dict.get(f"{split_name}_ds", None)
    if ds is None:
        return

    # These attribute names must match your WindowDataset implementation
    X = getattr(ds, "X", None)  # torch.Tensor (N, L, D)
    y = getattr(ds, "y_label", None)  # torch.Tensor (N,) or None

    if X is None:
        raise AttributeError(
            f"{split_name}_ds has no attribute 'X'. Adjust save_processed_data() to your WindowDataset."
        )

    X_np = to_numpy(X)
    y_np = to_numpy(y) if y is not None else None

    save_path = os.path.join(folder, f"{dataset}_{split_name}_windows.npz")
    if y_np is None:
        np.savez_compressed(save_path, X=X_np)
    else:
        np.savez_compressed(save_path, X=X_np, y=y_np)










