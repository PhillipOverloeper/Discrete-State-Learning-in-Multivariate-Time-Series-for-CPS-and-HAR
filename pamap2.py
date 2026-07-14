from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utils import WindowDataset, save_split


class PAMAP2_Dataloader:
    """
    PAMAP2 dataset loader.



    Output format from `_prepare_tep_datasets()`:
      {
        "train_ds": WindowDataset,
        "val_ds": WindowDataset,
        "test_ds": WindowDataset,
        "scaler": fitted sklearn scaler,
        "meta": dict with shapes and column names,
      }
    """

    def __init__(
        self,
        root: str = "data/PAMAP2",
        window_length: int = 128,
        stride: int = 24,
        horizon: int = 0,
        scaler_type: str = "standard",
        downsample: int = 5,
        filenames: Optional[Dict[str, str]] = None,
        val_run_fraction: float = 0.15,
        test_run_fraction:float = 0.2,
    ):
        """
        Args:
            root: Directory containing the RData files and also cache/processed folders.
            window_length: Sliding window length L (number of samples per window).
            stride: Sliding window stride S.
            horizon: Forecast horizon H (currently unused; reserved for later).
            scaler_type: "standard" or "minmax".
            downsample: Keep every `downsample`-th row (e.g., 10 => iloc[::10]).
            filenames: Optional override for the default file names.
            val_run_fraction: Fraction of train runs held out for validation.
        """
        self.root = root
        self.window_length = window_length
        self.stride = stride
        self.horizon = horizon
        self.scaler_type = scaler_type
        self.downsample = downsample
        self.val_run_fraction = val_run_fraction
        self.test_run_fraction = test_run_fraction
        if filenames is None:
            self.filenames = [f"data{i}.csv" for i in range(1, 8)]
        else:
            self.filenames = filenames

        self.scaler = None  # fitted sklearn scaler


    def _load_csv_objects(self, data_paths):
        """
        Reads specified files (data_paths) and returns them as a list of Pandas dataframes.
        """
        # Load csv files
        paths = [Path(self.root, path) for path in data_paths]
        dataframes = [pd.read_csv(dataset, header=None) for dataset in paths]

        return dataframes


    def _add_subjects(self, data_list):

        for i, df in enumerate(data_list):
            df.insert(2, "subject", i + 1)
            df.columns = range(df.shape[1])

        return data_list


    def _get_feature_columns(self, df):
        """
        Return numeric feature columns, excluding metadata and internal helper columns.
        """
        # Extract the feature columns
        feature_cols = list(df.columns[2:])

        # If no feature columns are available, raise error
        if not feature_cols:
            raise ValueError("No numeric feature columns found.")
        return feature_cols

    def _create_windows(self, X, y):
        """
        Create sliding windows from a single run.

        Args:
            X: Array of shape (T, D).
            y: Array of shape (T,) (faultNumber label per timestep).

        Returns:
            X_windows: (N, L, D)
            y_windows: (N,) using the label of the last time step in each window.
        """
        # Extract shapes
        T, D = X.shape
        L = self.window_length
        S = self.stride

        # Compute number of windows
        n_windows = (T - L) // S + 1
        if n_windows <= 0:
            return np.empty((0, L, D), dtype=np.float32), np.empty((0,), dtype=y.dtype)

        #
        idx = (np.arange(n_windows)[:, None] * S) + np.arange(L)[None, :]
        X_win = X[idx].astype(np.float32)
        y_win = y[idx]
        return X_win, y_win


    def save_processed_data(self, data_dict, folder = None):
        """
        Saves processed windows (train/val/test), fitted scaler, and metadata.

        Files saved:
          - tep_train_windows.npz
          - tep_val_windows.npz
          - tep_test_windows.npz
          - scaler.joblib
          - metadata.joblib

        Args:
            data_dict: Output of `_prepare_tep_datasets()`.
            folder: Output directory. Default: <root>/processed
        """
        if folder is None:
            folder = os.path.join(self.root, "processed")
        os.makedirs(folder, exist_ok=True)

        save_split("train", data_dict, folder, "pamap2")
        save_split("val", data_dict, folder, "pamap2")
        save_split("test", data_dict, folder, "pamap2")

        if data_dict.get("scaler", None) is not None:
            joblib.dump(data_dict["scaler"], os.path.join(folder, "scaler.joblib"))
        joblib.dump(data_dict.get("meta", {}), os.path.join(folder, "metadata.joblib"))

        print(f"Processed dataset saved to {folder}")

    def load_processed_data(
        self,
        folder: Optional[str] = None,
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load processed windows from disk.

        Args:
            folder: Directory containing processed files. Default: <root>/processed
            as_datasets: If True, returns WindowDataset objects. If False, returns torch tensors.
            device: Optional torch device (e.g., "cpu", "cuda"). If provided, tensors are moved.

        Returns:
            Dict with keys:
              - train_ds / val_ds / test_ds (if as_datasets=True)
                OR train_tensors / val_tensors / test_tensors (if as_datasets=False)
              - scaler
              - meta
        """
        if folder is None:
            folder = os.path.join(self.root, "processed")

        def load_split(split_name: str):
            path = os.path.join(folder, f"PAMAP2_{split_name}_windows.npz")
            if not os.path.exists(path):
                return None, None

            d = np.load(path)
            X = torch.tensor(d["X"], dtype=torch.float32)
            y = torch.tensor(d["y"], dtype=torch.long)

            if device is not None:
                X = X.to(device)
                if y is not None:
                    y = y.to(device)

            return X, y

        X_train, y_train = load_split("train")
        X_val, y_val = load_split("val")
        X_test, y_test = load_split("test")

        scaler_path = os.path.join(folder, "scaler.joblib")
        meta_path = os.path.join(folder, "metadata.joblib")

        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        meta = joblib.load(meta_path) if os.path.exists(meta_path) else {}

        out = {
            "train_ds": WindowDataset(X_train.cpu().numpy(), y_label=y_train.cpu().numpy()) if X_train is not None else None,
            "val_ds": WindowDataset(X_val.cpu().numpy(), y_label=y_val.cpu().numpy()) if X_val is not None else None,
            "test_ds": WindowDataset(X_test.cpu().numpy(), y_label=y_test.cpu().numpy()) if X_test is not None else None,
            "scaler": scaler,
            "meta": meta,
        }
        return out


    def _prepare_pamap2_datasets(self):
        """
        Prepare train/val/test datasets for PAMAP2.
        """
        # Check if files have been specified
        if self.filenames is None:
            raise ValueError("Please specify filenames.")
        print("Loading data...")

        # Load the CSV files
        all_data = self._load_csv_objects(self.filenames)
        all_data = self._add_subjects(all_data)

        # Concatenate all dataframes
        all_data = pd.concat(all_data, axis=0, ignore_index=True)
        subjects = all_data[2].dropna().unique()
        all_data = all_data.loc[all_data[1] != 0]

        # Separate by test_run_fraction, but adhering to the "subjects" column
        n_subj = len(subjects)
        n_test = int(round(n_subj * self.test_run_fraction))
        n_test = max(1, min(n_test, n_subj - 1))

        train_subjects = set(subjects[:n_subj - n_test])
        test_subjects = set(subjects[n_subj - n_test:])

        train_raw = all_data[all_data[2].isin(train_subjects)].reset_index(drop=True)
        test_df = all_data[all_data[2].isin(test_subjects)].reset_index(drop=True)

        # Extract the feature columns
        feature_cols = self._get_feature_columns(train_raw)
        print(f"Using {len(feature_cols)} features")

        # Check, if enough train data is available
        if len(train_raw) < 2:
            raise ValueError("Need at least 2 runs to split into train/val by run.")

        # Split into train and validation sets, but adhering to the "subjects" column
        subjects = train_raw[2].dropna().unique()
        n_subj = len(subjects)
        n_val = int(round(n_subj * self.val_run_fraction))
        n_val = max(1, min(n_val, n_subj - 1))

        val_subjects = set(subjects[:n_val])
        train_subjects = set(subjects[n_val:])

        train_df = all_data[all_data[2].isin(train_subjects)].reset_index(drop=True)
        val_df = all_data[all_data[2].isin(val_subjects)].reset_index(drop=True)

        # Set up scaler and fit on the train dataset
        print("Preprocessing / scaling")
        st = str(self.scaler_type).lower()
        if st == "minmax":
            self.scaler = MinMaxScaler()
        elif st == "standard":
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaler_type={self.scaler_type}")
        self.scaler.fit(train_df[feature_cols].to_numpy(dtype=np.float32))

        def windowize_by_run(df):
            # Set up lists
            X_all, y_all = [], []
            run_lengths = []

            # Group by the 3rd column (index 2), which is your Subject ID
            for rk, g in df.groupby(2, sort=False):
                run_lengths.append(len(g))

                # Skip runs that are too short
                if len(g) < self.window_length + self.horizon:
                    continue

                # feature_cols should already exclude indices 0, 1, and 2
                # Scale the data (Column 0 is ignored here because it's not in feature_cols)
                X = self.scaler.transform(g[feature_cols].to_numpy(dtype=np.float32))

                # Target y is the 2nd column (index 1)
                y = g[1].to_numpy()

                X_win, y_win = self._create_windows(X, y)

                if len(X_win) == 0:
                    continue

                X_all.append(X_win)
                y_all.append(y_win)

            if not X_all:
                if run_lengths:
                    msg = (
                        f"No windows created. window_length={self.window_length}, stride={self.stride}, "
                        f"min={min(run_lengths)}, median={int(np.median(run_lengths))}, max={max(run_lengths)}"
                    )
                else:
                    msg = "No windows created and no runs found."
                raise ValueError(msg)

            return np.concatenate(X_all, axis=0), np.concatenate(y_all, axis=0)

        # --- Create train/val/test windows ---
        print(f"Creating windows (L={self.window_length}, S={self.stride})")
        X_train_win, y_train_win = windowize_by_run(train_df)
        X_val_win, y_val_win = windowize_by_run(val_df)
        X_test_win, y_test_win = windowize_by_run(test_df)

        # Wrap into datasets
        train_ds = WindowDataset(X_train_win, y_label=y_train_win)
        val_ds = WindowDataset(X_val_win, y_label=y_val_win)
        test_ds = WindowDataset(X_test_win, y_label=y_test_win)

        meta = {
            "dataset": "pamap2",
            "feature_cols": feature_cols,
            "D": int(X_train_win.shape[-1]),
            "L": int(self.window_length),
            "S": int(self.stride),
            "H": int(self.horizon),
            "n_train_windows": int(X_train_win.shape[0]),
            "n_val_windows": int(X_val_win.shape[0]),
            "n_test_windows": int(X_test_win.shape[0]),
            "n_train_runs": int(train_ds.__len__()),
            "n_val_runs": int(val_ds.__len__()),
            "n_test_runs": int(test_ds.__len__()),
            "scaler_type": st,
        }

        return {
            "train_ds": train_ds,
            "val_ds": val_ds,
            "test_ds": test_ds,
            "scaler": self.scaler,
            "meta": meta,
        }



#loader = PAMAP2_Dataloader()

#out = loader._prepare_pamap2_datasets()
#loader.save_processed_data(data_dict=out)
#out = loader.load_processed_data()
#print(out["train_ds"])
