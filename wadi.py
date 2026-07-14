from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utils import WindowDataset, save_split


class WADI_Dataloader:
    """
    WADI (normal) dataset loader.

    - Reads the WADI normal CSV with a 3-line preamble and a real header line starting with:
        Row,Date,Time,...

    - Extracts:
        X = all sensor columns (numeric features)
        y = PLANT_START_STOP_LOG (plant state) by default

    - Splits chronologically into train/val/test.
    - Scales using train only.
    - Creates sliding windows of shape (N, L, D) and labels y_win (N,) using last timestep label.
    """

    def __init__(
        self,
        root: str = "data/WADI",
        filename: str = "WADI_normal.csv",   # set this to your actual file name
        window_length: int = 128,
        stride: int = 24,
        horizon: int = 0,                   # reserved
        scaler_type: str = "standard",
        downsample: int = 1,
        val_fraction: float = 0.15,
        test_fraction: float = 0.20,
        label_cols: Optional[list] = None,    # if None, auto-detect PLANT_START_STOP_LOG
        shorten_colnames: bool = True,      # turn \\WIN...\TAG into TAG for readability
        fillna_method: str = "ffill",       # "ffill", "interpolate", or "zero"
    ):
        self.root = root
        self.filename = filename
        self.window_length = window_length
        self.stride = stride
        self.horizon = horizon
        self.scaler_type = scaler_type
        self.downsample = downsample
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.label_cols = label_cols
        self.shorten_colnames = shorten_colnames
        self.fillna_method = fillna_method

        if self.label_cols is None:
            self.label_cols = [
                # Primary pumps
                "1_P_001_STATUS", "1_P_002_STATUS", "1_P_003_STATUS", "1_P_004_STATUS", "1_P_005_STATUS",
                "1_P_006_STATUS",
                "2_P_001_STATUS", "2_P_002_STATUS", "2_P_003_STATUS", "2_P_004_STATUS",


                #"1_MV_001_STATUS", "1_MV_002_STATUS", "1_MV_003_STATUS", "1_MV_004_STATUS",
                #"2_MV_001_STATUS", "2_MV_002_STATUS", "2_MV_003_STATUS", "2_MV_004_STATUS", "2_MV_005_STATUS",
                #"2_MV_006_STATUS",

                # Solenoid valves
                #"2_SV_101_STATUS", "2_SV_201_STATUS", "2_SV_301_STATUS", "2_SV_401_STATUS", "2_SV_501_STATUS",
                #"2_SV_601_STATUS",
            ]

        self.scaler = None

    # ----------------------------
    # CSV parsing helpers
    # ----------------------------
    def _find_header_row_index(self, path: Path) -> int:
        """
        Find the line index (0-based) where the real CSV header starts.
        We detect the line that begins with 'Row,Date,Time'.
        """
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if line.startswith("Row,Date,Time"):
                    return i
        raise ValueError("Could not find header line starting with 'Row,Date,Time' in the WADI file.")

    def _shorten_name(self, col: str) -> str:
        """
        Convert '\\\\WIN-...\\LOG_DATA\\...\\1_AIT_001_PV' -> '1_AIT_001_PV'
        Keep 'Row', 'Date', 'Time' unchanged.
        """
        if col in ("Row", "Date", "Time"):
            return col
        # Split on backslash, keep last token
        parts = str(col).split("\\")
        return parts[-1] if parts else str(col)

    def _read_wadi_csv(self) -> pd.DataFrame:
        path = Path(self.root) / self.filename
        if not path.exists():
            raise FileNotFoundError(f"WADI file not found: {path}")

        header_idx = self._find_header_row_index(path)

        # Read from the real header line onward
        df = pd.read_csv(
            path,
            skiprows=header_idx,
            low_memory=False,
        )

        # Optional: shorten column names for sanity
        if self.shorten_colnames:
            df.columns = [self._shorten_name(c) for c in df.columns]

        # Downsample early (keeps chronological ordering)
        if self.downsample and self.downsample > 1:
            df = df.iloc[:: self.downsample].reset_index(drop=True)

        return df

    def _infer_label_col(self, df: pd.DataFrame) -> list:
        for col in self.label_cols:
            if col not in df.columns:
                raise ValueError(f"label_col='{col}' not found in columns.")
        return self.label_cols

        # Auto-detect by exact tag name
        candidates = [c for c in df.columns if str(c).strip() == "PLANT_START_STOP_LOG"]
        if len(candidates) == 1:
            return candidates[0]

        # Fallback: contains substring
        candidates = [c for c in df.columns if "PLANT_START_STOP_LOG" in str(c)]
        if len(candidates) == 1:
            return candidates[0]

        # If your file differs, you must pass label_col explicitly
        raise ValueError(
            "Could not uniquely infer label column for plant state. "
            "Pass label_col=... explicitly (expected 'PLANT_START_STOP_LOG')."
        )

    def _clean_and_cast_numeric(
        self, df: pd.DataFrame, feature_cols: list[str], label_cols: list[str]
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Ensure feature columns are numeric; handle missing values.
        Return cleaned df (features only) and y as integer array.
        """
        Xdf = df[feature_cols].copy()

        # Convert all feature columns to numeric (coerce bad strings to NaN)
        for c in feature_cols:
            Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce")

        # Missing value strategy
        if self.fillna_method == "ffill":
            Xdf = Xdf.ffill().bfill()
        elif self.fillna_method == "interpolate":
            Xdf = Xdf.interpolate(limit_direction="both")
            Xdf = Xdf.ffill().bfill()
        elif self.fillna_method == "zero":
            Xdf = Xdf.fillna(0.0)
        else:
            raise ValueError(f"Unknown fillna_method='{self.fillna_method}'")
        Xdf = Xdf.dropna(axis=1, how='all')

        # Label
        y = df[label_cols].copy()
        for label_column in label_cols:
            y[label_column] = pd.to_numeric(y[label_column], errors="coerce").fillna(0).astype(np.int64).to_numpy()

        y = y.dropna(axis=1, how="all")
        row_tuples = pd.Series(list(y.itertuples(index=False, name=None)))

        # 2. Map unique combinations to integers
        # codes: the integer labels for your 1.2M rows
        # uniques: the index mapping back to the original 26D vectors
        codes, uniques = pd.factorize(row_tuples)

        # 3. Add to a new dataframe or replace
        y_integers = pd.Series(codes, name="label_id")


        return Xdf, y_integers

    def _get_feature_columns(self, df: pd.DataFrame, label_cols: list) -> list[str]:
        """
        Feature columns = everything except Row/Date/Time and label_col.
        """
        drop = {"Row", "Date", "Time", *label_cols}
        feature_cols = [c for c in df.columns if c not in drop]

        if not feature_cols:
            raise ValueError("No feature columns found after excluding Row/Date/Time and label column.")

        return feature_cols

    # ----------------------------
    # Windowing
    # ----------------------------
    def _create_windows(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        T, D = X.shape
        L = self.window_length
        S = self.stride

        n_windows = (T - L) // S + 1
        if n_windows <= 0:
            return np.empty((0, L, D), dtype=np.float32), np.empty((0,), dtype=y.dtype)

        idx = (np.arange(n_windows)[:, None] * S) + np.arange(L)[None, :]
        X_win = X[idx].astype(np.float32)
        y = y.to_numpy()
        y_win  = y[idx]  # label of last timestep in window
        return X_win, y_win

    # ----------------------------
    # Public prepare/save/load
    # ----------------------------
    def _prepare_wadi_datasets(self) -> Dict[str, Any]:
        print("Loading WADI normal data...")
        df = self._read_wadi_csv()

        label_cols = self._infer_label_col(df)
        feature_cols = self._get_feature_columns(df, label_cols)

        print(f"Label columns: {label_cols}")
        print(f"Using {len(feature_cols)} feature columns")

        # Clean numeric + extract y
        Xdf, y = self._clean_and_cast_numeric(df, feature_cols, label_cols)

        # Chronological split (important for time series)
        n = len(Xdf)
        if n < (self.window_length + 1):
            raise ValueError(f"Not enough rows ({n}) for window_length={self.window_length}")

        n_test = int(round(n * self.test_fraction))
        n_val = int(round(n * self.val_fraction))
        n_test = max(1, min(n_test, n - 2))
        n_val = max(1, min(n_val, n - n_test - 1))

        n_train = n - n_val - n_test
        if n_train <= 0:
            raise ValueError("Train split is empty; reduce val_fraction/test_fraction.")

        # Split
        X_train_df = Xdf.iloc[:n_train]
        y_train = y[:n_train]

        X_val_df = Xdf.iloc[n_train : n_train + n_val]
        y_val = y[n_train : n_train + n_val]

        X_test_df = Xdf.iloc[n_train + n_val :]
        y_test = y[n_train + n_val :]

        # Fit scaler on train only
        print("Preprocessing / scaling...")
        st = str(self.scaler_type).lower()
        if st == "minmax":
            self.scaler = MinMaxScaler()
        elif st == "standard":
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaler_type={self.scaler_type}")

        self.scaler.fit(X_train_df.to_numpy(dtype=np.float32))

        X_train = self.scaler.transform(X_train_df.to_numpy(dtype=np.float32))
        X_val = self.scaler.transform(X_val_df.to_numpy(dtype=np.float32))
        X_test = self.scaler.transform(X_test_df.to_numpy(dtype=np.float32))

        print(f"Creating windows (L={self.window_length}, S={self.stride})...")
        X_train_win, y_train_win = self._create_windows(X_train, y_train)
        X_val_win, y_val_win = self._create_windows(X_val, y_val)
        X_test_win, y_test_win = self._create_windows(X_test, y_test)


        if len(X_train_win) == 0 or len(X_val_win) == 0 or len(X_test_win) == 0:
            raise ValueError(
                "At least one split produced 0 windows. "
                "Decrease window_length or adjust splits/stride/downsample."
            )

        train_ds = WindowDataset(X_train_win, y_label=y_train_win)
        val_ds = WindowDataset(X_val_win, y_label=y_val_win)
        test_ds = WindowDataset(X_test_win, y_label=y_test_win)

        meta = {
            "dataset": "wadi_normal",
            "filename": self.filename,
            "label_col": label_cols,
            "feature_cols": feature_cols,
            "D": int(X_train_win.shape[-1]),
            "L": int(self.window_length),
            "S": int(self.stride),
            "H": int(self.horizon),
            "downsample": int(self.downsample),
            "n_rows_total": int(n),
            "n_rows_train": int(n_train),
            "n_rows_val": int(n_val),
            "n_rows_test": int(n_test),
            "n_train_windows": int(X_train_win.shape[0]),
            "n_val_windows": int(X_val_win.shape[0]),
            "n_test_windows": int(X_test_win.shape[0]),
            "scaler_type": st,
            "fillna_method": self.fillna_method,
        }

        return {
            "train_ds": train_ds,
            "val_ds": val_ds,
            "test_ds": test_ds,
            "scaler": self.scaler,
            "meta": meta,
        }

    def save_processed_data(self, data_dict: Dict[str, Any], folder: Optional[str] = None) -> None:
        if folder is None:
            folder = os.path.join(self.root, "processed")
        os.makedirs(folder, exist_ok=True)

        save_split("train", data_dict, folder, "wadi")
        save_split("val", data_dict, folder, "wadi")
        save_split("test", data_dict, folder, "wadi")

        if data_dict.get("scaler") is not None:
            joblib.dump(data_dict["scaler"], os.path.join(folder, "scaler.joblib"))
        joblib.dump(data_dict.get("meta", {}), os.path.join(folder, "metadata.joblib"))

        print(f"Processed dataset saved to {folder}")

    def load_processed_data(self, folder: Optional[str] = None, device: Optional[str] = None) -> Dict[str, Any]:
        if folder is None:
            folder = os.path.join(self.root, "processed")

        def load_split(split_name: str):
            path = os.path.join(folder, f"wadi_{split_name}_windows.npz")
            if not os.path.exists(path):
                return None, None
            d = np.load(path)
            X = torch.tensor(d["X"], dtype=torch.float32)
            y = torch.tensor(d["y"], dtype=torch.long)
            if device is not None:
                X = X.to(device)
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

