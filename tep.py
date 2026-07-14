from __future__ import annotations

import os
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import pyreadr
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utils import WindowDataset, add_source


class TEPDataLoader:
    """
    Tennessee Eastman Process (TEP) dataset loader.

    This loader supports:
      - Reading the canonical TEP .RData files (fault-free + faulty, train + test)
      - Optional caching of downsampled raw DataFrames to speed up iteration
      - Run-aware windowing (no windows crossing run boundaries)
      - Train/val split by simulation runs (chronological)
      - Train-only scaling (fit on train runs, transform val/test)
      - Saving processed windows (train/val/test) to disk under /processed
      - Loading processed windows back as PyTorch tensors and/or WindowDataset objects

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
        root: str = "data/tep",
        window_length: int = 128,
        stride: int = 24,
        horizon: int = 0,
        scaler_type: str = "standard",
        downsample: int = 1,
        filenames: Optional[Dict[str, str]] = None,
        val_run_fraction: float = 0.15,
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

        if filenames is None:
            self.filenames = {
                "fault_free_train": "TEP_FaultFree_Training.RData",
                "fault_free_test": "TEP_FaultFree_Testing.RData",
                "faulty_train": "TEP_Faulty_Training.RData",
                "faulty_test": "TEP_Faulty_Testing.RData",
            }
        else:
            self.filenames = filenames

        self.scaler = None  # fitted sklearn scaler


    def _load_rdata_objects(self, path):
        """
        Read an .RData file and return the first object inside as a DataFrame.
        """
        # Load RData file
        res = pyreadr.read_r(path)
        if len(res.keys()) == 0:
            raise ValueError(f"No objects found inside RData: {path}")

        # Return first object
        return res[str(list(res.keys())[0])]

    def _cache_paths(self, cache_dir = None):
        """
        Compute cache paths for downsampled raw train/test DataFrames.

        The cache key includes the downsample factor to keep variants separate.
        """
        # Create cache base directory
        if cache_dir is None:
            cache_dir = os.path.join(self.root, "cache")
        os.makedirs(cache_dir, exist_ok=True)

        # Create file paths for train/test and meta
        tag = f"ds{self.downsample}"
        train_path = os.path.join(cache_dir, f"tep_train_raw_{tag}.parquet")
        test_path = os.path.join(cache_dir, f"tep_test_raw_{tag}.parquet")
        meta_path = os.path.join(cache_dir, f"tep_meta_raw_{tag}.joblib")
        return train_path, test_path, meta_path

    def _save_raw_cache(self, train_raw, test_raw, cache_dir = None):
        """
        Save downsampled raw DataFrames to cache.
        """
        # Get cache paths
        train_path, test_path, meta_path = self._cache_paths(cache_dir)

        # Save files as parquet
        try:
            train_raw.to_parquet(train_path, index=False)
            test_raw.to_parquet(test_path, index=False)
            joblib.dump({"downsample": self.downsample}, meta_path)
            print(f"Saved raw cache: {train_path}, {test_path}")
            return
        except Exception as e:
            print(f"Parquet save failed ({e}). Falling back to pickle.")


    def _load_raw_cache(self, cache_dir = None):
        """
        Load cached downsampled raw DataFrames if present, else return (None, None, None).
        """
        train_path, test_path, meta_path = self._cache_paths(cache_dir)

        # Try parquet files
        if os.path.exists(train_path) and os.path.exists(test_path):
            train_raw = pd.read_parquet(train_path)
            test_raw = pd.read_parquet(test_path)
            meta = joblib.load(meta_path) if os.path.exists(meta_path) else {}
            print(f"Loaded raw cache: {train_path}, {test_path}")
            return train_raw, test_raw, meta

        return None, None, None


    def _get_feature_columns(self, df):
        """
        Return numeric feature columns, excluding metadata and internal helper columns.
        """
        # Set up metadata columns
        metadata_cols = {
            "faultNumber", "simulationRun", "sample",
            "__source__", "__run_key__"
        }

        # Extract feature columns (actual data columns)
        feature_cols = [
            c for c in df.columns
            if c not in metadata_cols and pd.api.types.is_numeric_dtype(df[c])
        ]

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

    # ---------------------------------------------------------------------
    # Processed dataset save/load
    # ---------------------------------------------------------------------
    def save_processed_data(self, data_dict: Dict[str, Any], folder: Optional[str] = None) -> None:
        """
        Save processed windows (train/val/test), fitted scaler, and metadata.

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

        def to_numpy(a):
            if a is None:
                return None
            if isinstance(a, np.ndarray):
                return a
            if torch.is_tensor(a):
                return a.detach().cpu().numpy()
            raise TypeError(f"Unsupported type for saving: {type(a)}")

        def save_split(split_name: str):
            ds = data_dict.get(f"{split_name}_ds", None)
            if ds is None:
                return

            # These attribute names must match your WindowDataset implementation
            X = getattr(ds, "X", None)            # torch.Tensor (N, L, D)
            y = getattr(ds, "y_label", None)      # torch.Tensor (N,) or None

            if X is None:
                raise AttributeError(
                    f"{split_name}_ds has no attribute 'X'. Adjust save_processed_data() to your WindowDataset."
                )

            X_np = to_numpy(X)
            y_np = to_numpy(y) if y is not None else None

            save_path = os.path.join(folder, f"tep_{split_name}_windows.npz")
            if y_np is None:
                np.savez_compressed(save_path, X=X_np)
            else:
                np.savez_compressed(save_path, X=X_np, y=y_np)

        save_split("train")
        save_split("val")
        save_split("test")

        if data_dict.get("scaler", None) is not None:
            joblib.dump(data_dict["scaler"], os.path.join(folder, "scaler.joblib"))
        joblib.dump(data_dict.get("meta", {}), os.path.join(folder, "metadata.joblib"))

        print(f"Processed dataset saved to {folder}")

    def load_processed_data(
        self,
        folder: Optional[str] = None,
        device: Optional[str] = None,
        anomaly: Optional[bool] = False,
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
        if anomaly:
            folder = os.path.join(self.root, "anomaly")
        else:
            folder = os.path.join(self.root, "processed")

        def load_split(split_name: str):
            path = os.path.join(folder, f"tep_{split_name}_windows.npz")
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

    def prepare_anomaly_detection_dataset(self, folder: Optional[str] = None):
        # Set up paths to data and check if they exist
        paths = {k: os.path.join(self.root, v) for k, v in self.filenames.items()}
        for k, p in paths.items():
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing file for {k}: {p}")

        # If cached data exists, load it
        print("Loading data...")

        # Load the RData files
        obj_ff_tr = add_source(self._load_rdata_objects(paths["fault_free_train"]), "ff_tr")
        obj_ff_te = add_source(self._load_rdata_objects(paths["fault_free_test"]), "ff_te")

        # Concatenate different train and test data (faulty and fault-free)
        train_raw = pd.DataFrame(obj_ff_tr).reset_index(drop=True)
        test_raw = pd.DataFrame(obj_ff_te).reset_index(drop=True)

        if folder is None:
            folder = os.path.join(self.root, "anomaly")
        os.makedirs(folder, exist_ok=True)

        # Extract the feature columns
        feature_cols = self._get_feature_columns(train_raw)
        print(f"Using {len(feature_cols)} features")

        # Set up required columns
        required = {"simulationRun", "sample", "faultNumber", "__source__"}
        missing = required - set(train_raw.columns)
        if missing:
            raise ValueError(f"TEP data missing required columns: {sorted(missing)}")

        # Create __run_key__ to avoid duplicates, when splitting into train and validation
        train_raw["__run_key__"] = (train_raw["__source__"].astype(str)
                                    + "_" + train_raw["simulationRun"].astype(str))
        run_keys = train_raw["__run_key__"].drop_duplicates().tolist()
        if len(run_keys) < 2:
            raise ValueError("Need at least 2 runs to split into train/val by run.")

        # Split into train and validation sets
        n_val = max(1, int(round(len(run_keys) * self.val_run_fraction)))
        n_train = len(run_keys) - n_val
        train_run_keys = set(run_keys[:n_train])
        val_run_keys = set(run_keys[n_train:])

        train_df = train_raw[train_raw["__run_key__"].isin(train_run_keys)].copy()
        val_df = train_raw[train_raw["__run_key__"].isin(val_run_keys)].copy()

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

        # Ensure test has run keys
        test_raw["__run_key__"] = test_raw["__source__"].astype(str) + "_" + test_raw["simulationRun"].astype(str)

        # --- Windowize helper (no run-boundary windows) ---
        def windowize_by_run(df):
            # Set up lists
            X_all, y_all = [], []
            run_lengths = []

            for rk, g in df.groupby("__run_key__", sort=False):
                run_lengths.append(len(g))

                # Skip runs that are too short (common after downsampling)
                if len(g) < self.window_length + self.horizon:
                    continue

                # Scale the data
                X = self.scaler.transform(g[feature_cols].to_numpy(dtype=np.float32))
                y = g["faultNumber"].to_numpy()

                X_win, y_win = self._create_windows(X, y)

                if len(X_win) == 0:
                    continue

                X_all.append(X_win)
                y_all.append(y_win)

            if not X_all:
                if run_lengths:
                    msg = (
                        f"No windows created. window_length={self.window_length}, stride={self.stride}, "
                        f"downsample={self.downsample}. Run lengths after downsample: "
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
        X_test_win, y_test_win = windowize_by_run(test_raw)

        # Wrap into datasets
        train_ds = WindowDataset(X_train_win, y_label=y_train_win)
        val_ds = WindowDataset(X_val_win, y_label=y_val_win)
        test_ds = WindowDataset(X_test_win, y_label=y_test_win)

        meta = {
            "dataset": "tep",
            "feature_cols": feature_cols,
            "D": int(X_train_win.shape[-1]),
            "L": int(self.window_length),
            "S": int(self.stride),
            "H": int(self.horizon),
            "n_train_windows": int(X_train_win.shape[0]),
            "n_val_windows": int(X_val_win.shape[0]),
            "n_test_windows": int(X_test_win.shape[0]),
            "n_train_runs": int(len(train_run_keys)),
            "n_val_runs": int(len(val_run_keys)),
            "scaler_type": st,
            "downsample": int(self.downsample),
        }


        def to_numpy(a):
            if a is None:
                return None
            if isinstance(a, np.ndarray):
                return a
            if torch.is_tensor(a):
                return a.detach().cpu().numpy()
            raise TypeError(f"Unsupported type for saving: {type(a)}")
        def save_split(ds, name):

            if ds is None:
                return

            # These attribute names must match your WindowDataset implementation
            X = getattr(ds, "X", None)            # torch.Tensor (N, L, D)
            y = getattr(ds, "y_label", None)      # torch.Tensor (N,) or None

            if X is None:
                raise AttributeError(
                    f"{name}ds has no attribute 'X'. Adjust save_processed_data() to your WindowDataset."
                )

            X_np = to_numpy(X)
            y_np = to_numpy(y) if y is not None else None

            save_path = os.path.join(folder, f"tep_{name}_windows.npz")
            if y_np is None:
                np.savez_compressed(save_path, X=X_np)
            else:
                np.savez_compressed(save_path, X=X_np, y=y_np)

        save_split(train_ds, "train")
        save_split(val_ds, "val")
        save_split(test_ds, "test")

        if self.scaler is not None:
            joblib.dump(self.scaler, os.path.join(folder, "scaler.joblib"))
        joblib.dump(meta, os.path.join(folder, "metadata.joblib"))

        print(f"Processed dataset saved to {folder}")

        return {
            "train_ds": train_ds,
            "val_ds": val_ds,
            "test_ds": test_ds,
            "scaler": self.scaler,
            "meta": meta,
        }





    def _prepare_tep_datasets(self):
        """
        Prepare train/val/test datasets for TEP.

        Steps:
          1) Load cached downsampled raw DataFrames if available, else read RData and create cache.
          2) Select numeric features.
          3) Create per-run keys and split train runs into train/val chronologically.
          4) Fit scaler on train only and transform val/test.
          5) Create windows within each run, then concatenate across runs.
          6) Wrap into WindowDataset and return along with scaler + metadata.
        """
        # Set up paths to data and check if they exist
        paths = {k: os.path.join(self.root, v) for k, v in self.filenames.items()}
        for k, p in paths.items():
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing file for {k}: {p}")

        # If cached data exists, load it
        print("Loading data...")
        train_raw, test_raw, _cache_meta = self._load_raw_cache()
        if train_raw is None:
            print("No cache found. Loading RData and creating cache...")

            # Load the RData files
            obj_ff_tr = add_source(self._load_rdata_objects(paths["fault_free_train"]), "ff_tr")
            obj_f_tr = add_source(self._load_rdata_objects(paths["faulty_train"]), "f_tr")
            obj_ff_te = add_source(self._load_rdata_objects(paths["fault_free_test"]), "ff_te")
            obj_f_te = add_source(self._load_rdata_objects(paths["faulty_test"]), "f_te")

            # Concatenate different train and test data (faulty and fault-free)
            train_raw = pd.concat([obj_ff_tr, obj_f_tr], axis=0, ignore_index=True)
            test_raw = pd.concat([obj_ff_te, obj_f_te], axis=0, ignore_index=True)

            # If downsample parameter is specified, downsample the data
            if self.downsample is not None and self.downsample > 1:
                train_raw = train_raw.iloc[::self.downsample, :].reset_index(drop=True)
                test_raw = test_raw.iloc[::self.downsample, :].reset_index(drop=True)

            # Save downsampled data as .parquet files
            self._save_raw_cache(train_raw, test_raw)
        else:
            print(f"Using cached downsampled raw data (downsample={self.downsample}).")

        # Extract the feature columns
        feature_cols = self._get_feature_columns(train_raw)
        print(f"Using {len(feature_cols)} features")

        # Set up required columns
        required = {"simulationRun", "sample", "faultNumber", "__source__"}
        missing = required - set(train_raw.columns)
        if missing:
            raise ValueError(f"TEP data missing required columns: {sorted(missing)}")

        # Create __run_key__ to avoid duplicates, when splitting into train and validation
        train_raw["__run_key__"] = (train_raw["__source__"].astype(str)
                                    + "_" + train_raw["simulationRun"].astype(str))
        run_keys = train_raw["__run_key__"].drop_duplicates().tolist()
        if len(run_keys) < 2:
            raise ValueError("Need at least 2 runs to split into train/val by run.")

        # Split into train and validation sets
        n_val = max(1, int(round(len(run_keys) * self.val_run_fraction)))
        n_train = len(run_keys) - n_val
        train_run_keys = set(run_keys[:n_train])
        val_run_keys = set(run_keys[n_train:])

        train_df = train_raw[train_raw["__run_key__"].isin(train_run_keys)].copy()
        val_df = train_raw[train_raw["__run_key__"].isin(val_run_keys)].copy()

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

        # Ensure test has run keys
        test_raw["__run_key__"] = test_raw["__source__"].astype(str) + "_" + test_raw["simulationRun"].astype(str)

        # --- Windowize helper (no run-boundary windows) ---
        def windowize_by_run(df):
            # Set up lists
            X_all, y_all = [], []
            run_lengths = []

            for rk, g in df.groupby("__run_key__", sort=False):
                run_lengths.append(len(g))

                # Skip runs that are too short (common after downsampling)
                if len(g) < self.window_length + self.horizon:
                    continue

                # Scale the data
                X = self.scaler.transform(g[feature_cols].to_numpy(dtype=np.float32))
                y = g["faultNumber"].to_numpy()

                X_win, y_win = self._create_windows(X, y)

                if len(X_win) == 0:
                    continue

                X_all.append(X_win)
                y_all.append(y_win)

            if not X_all:
                if run_lengths:
                    msg = (
                        f"No windows created. window_length={self.window_length}, stride={self.stride}, "
                        f"downsample={self.downsample}. Run lengths after downsample: "
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
        X_test_win, y_test_win = windowize_by_run(test_raw)

        # Wrap into datasets
        train_ds = WindowDataset(X_train_win, y_label=y_train_win)
        val_ds = WindowDataset(X_val_win, y_label=y_val_win)
        test_ds = WindowDataset(X_test_win, y_label=y_test_win)

        meta = {
            "dataset": "tep",
            "feature_cols": feature_cols,
            "D": int(X_train_win.shape[-1]),
            "L": int(self.window_length),
            "S": int(self.stride),
            "H": int(self.horizon),
            "n_train_windows": int(X_train_win.shape[0]),
            "n_val_windows": int(X_val_win.shape[0]),
            "n_test_windows": int(X_test_win.shape[0]),
            "n_train_runs": int(len(train_run_keys)),
            "n_val_runs": int(len(val_run_keys)),
            "scaler_type": st,
            "downsample": int(self.downsample),
        }

        return {
            "train_ds": train_ds,
            "val_ds": val_ds,
            "test_ds": test_ds,
            "scaler": self.scaler,
            "meta": meta,
        }



#loader = TEPDataLoader()
#loader.prepare_anomaly_detection_dataset()
#out = loader.load_processed_data(anomaly=True)
#out = loader._prepare_tep_datasets()
#loader.save_processed_data(data_dict=out)
#out = loader.load_processed_data()
#print(out["train_ds"])