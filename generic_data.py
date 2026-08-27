"""Configuration-driven preprocessing and loading for custom time-series datasets."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utils import WindowDataset

REQUIRED_SPEC_KEYS = {"name", "file_pattern", "format", "label_column"}
VALID_FORMATS = {"csv", "parquet"}
VALID_SPLITS = {"group", "chronological"}
VALID_SCALERS = {"standard", "minmax", "none"}
VALID_FILL_METHODS = {"drop", "ffill", "interpolate", "zero"}


def load_dataset_spec(path: Path) -> dict[str, Any]:
    """Load and validate a custom-dataset JSON specification."""
    with path.open(encoding="utf-8") as handle:
        spec = json.load(handle)

    missing = REQUIRED_SPEC_KEYS - set(spec)
    if missing:
        raise ValueError(f"Dataset specification is missing: {', '.join(sorted(missing))}")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", str(spec["name"])):
        raise ValueError("name may contain only letters, numbers, underscores, and hyphens")
    if spec["format"] not in VALID_FORMATS:
        raise ValueError(f"format must be one of {sorted(VALID_FORMATS)}")
    if spec.get("split_strategy", "group") not in VALID_SPLITS:
        raise ValueError(f"split_strategy must be one of {sorted(VALID_SPLITS)}")
    if spec.get("scaler", "standard") not in VALID_SCALERS:
        raise ValueError(f"scaler must be one of {sorted(VALID_SCALERS)}")
    if spec.get("fillna", "ffill") not in VALID_FILL_METHODS:
        raise ValueError(f"fillna must be one of {sorted(VALID_FILL_METHODS)}")

    for key in ("val_fraction", "test_fraction"):
        value = float(spec.get(key, 0.15 if key == "val_fraction" else 0.2))
        if not 0 < value < 1:
            raise ValueError(f"{key} must be between 0 and 1")
    if float(spec.get("val_fraction", 0.15)) + float(spec.get("test_fraction", 0.2)) >= 1:
        raise ValueError("val_fraction + test_fraction must be less than 1")
    if int(spec.get("window_length", 128)) < 1 or int(spec.get("stride", 24)) < 1:
        raise ValueError("window_length and stride must be positive")
    return spec


def _read_frames(raw_dir: Path, spec: dict[str, Any]) -> pd.DataFrame:
    paths = sorted(raw_dir.glob(spec["file_pattern"]))
    if not paths:
        raise FileNotFoundError(
            f"No files matching {spec['file_pattern']!r} found below {raw_dir}"
        )

    read_options = spec.get("read_options", {})
    frames = []
    for path in paths:
        if spec["format"] == "csv":
            frame = pd.read_csv(path, **read_options)
        else:
            frame = pd.read_parquet(path, **read_options)
        frame = frame.copy()
        frame["__source_file__"] = path.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _fill_missing(frame: pd.DataFrame, columns: list[str], method: str, group_column: str) -> pd.DataFrame:
    frame = frame.copy()
    if method == "zero":
        frame[columns] = frame[columns].fillna(0)
    elif method == "drop":
        frame = frame.dropna(subset=columns)
    else:
        grouped = frame.groupby(group_column, sort=False)[columns]
        if method == "ffill":
            frame[columns] = grouped.transform(lambda values: values.ffill().bfill())
        else:
            frame[columns] = grouped.transform(
                lambda values: values.interpolate(limit_direction="both")
            )
    if frame[columns].isna().any().any():
        raise ValueError("Missing values remain after applying the configured fillna method")
    return frame


def _split_frame(
    frame: pd.DataFrame,
    group_column: str,
    strategy: str,
    val_fraction: float,
    test_fraction: float,
) -> dict[str, pd.DataFrame]:
    if strategy == "chronological":
        n_rows = len(frame)
        train_end = int(n_rows * (1 - val_fraction - test_fraction))
        val_end = int(n_rows * (1 - test_fraction))
        if train_end == 0 or val_end <= train_end or val_end >= n_rows:
            raise ValueError("Dataset is too small for the requested chronological split")
        splits = {
            "train": frame.iloc[:train_end].copy(),
            "val": frame.iloc[train_end:val_end].copy(),
            "test": frame.iloc[val_end:].copy(),
        }
        # Give every chronological segment its own boundary identifier.
        for name, split in splits.items():
            split["__window_group__"] = split[group_column].astype(str) + f"__{name}"
        return splits

    groups = frame[group_column].drop_duplicates().tolist()
    if len(groups) < 3:
        raise ValueError(
            "Group splitting needs at least three groups. Add a group_column, provide one file per run, "
            "or use split_strategy='chronological'."
        )
    n_test = max(1, int(round(len(groups) * test_fraction)))
    n_val = max(1, int(round(len(groups) * val_fraction)))
    if n_test + n_val >= len(groups):
        raise ValueError("Not enough groups for non-empty train, validation, and test splits")
    train_groups = set(groups[: len(groups) - n_val - n_test])
    val_groups = set(groups[len(groups) - n_val - n_test : len(groups) - n_test])
    test_groups = set(groups[len(groups) - n_test :])
    return {
        "train": frame[frame[group_column].isin(train_groups)].copy(),
        "val": frame[frame[group_column].isin(val_groups)].copy(),
        "test": frame[frame[group_column].isin(test_groups)].copy(),
    }


def _make_windows(
    frame: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    group_column: str,
    window_length: int,
    stride: int,
    scaler: Any,
) -> tuple[np.ndarray, np.ndarray]:
    X_windows, y_windows = [], []
    effective_group = "__window_group__" if "__window_group__" in frame else group_column
    for _, group in frame.groupby(effective_group, sort=False):
        if len(group) < window_length:
            continue
        X = group[feature_columns].to_numpy(dtype=np.float32)
        if scaler is not None:
            X = scaler.transform(X).astype(np.float32)
        y = group[label_column].to_numpy(dtype=np.int64)
        starts = range(0, len(group) - window_length + 1, stride)
        for start in starts:
            stop = start + window_length
            X_windows.append(X[start:stop])
            y_windows.append(y[start:stop])
    if not X_windows:
        raise ValueError(
            f"No windows created; every group must contain at least {window_length} rows"
        )
    return np.stack(X_windows), np.stack(y_windows)


def prepare_custom_dataset(raw_dir: Path, spec_path: Path, output_root: Path) -> Path:
    """Apply the shared split, scaling, and windowing pipeline to a custom dataset."""
    spec = load_dataset_spec(spec_path)
    frame = _read_frames(raw_dir, spec)
    label_column = spec["label_column"]
    if label_column not in frame:
        raise ValueError(f"Label column {label_column!r} does not exist")

    group_column = spec.get("group_column") or "__source_file__"
    if group_column not in frame:
        raise ValueError(f"Group column {group_column!r} does not exist")
    time_column = spec.get("time_column")
    if time_column:
        if time_column not in frame:
            raise ValueError(f"Time column {time_column!r} does not exist")
        frame = frame.sort_values([group_column, time_column], kind="stable").reset_index(drop=True)

    excluded = {
        label_column,
        group_column,
        "__source_file__",
        *spec.get("exclude_columns", []),
    }
    if time_column:
        excluded.add(time_column)
    requested_features = spec.get("feature_columns")
    if requested_features:
        missing_features = set(requested_features) - set(frame.columns)
        if missing_features:
            raise ValueError(f"Feature columns do not exist: {sorted(missing_features)}")
        feature_columns = list(requested_features)
    else:
        feature_columns = [
            column for column in frame.select_dtypes(include=[np.number]).columns if column not in excluded
        ]
    if not feature_columns:
        raise ValueError("No numeric feature columns were found")

    # Convert arbitrary categorical labels to stable integer IDs used by all models.
    label_values = sorted(frame[label_column].dropna().unique().tolist(), key=str)
    label_mapping = {str(value): index for index, value in enumerate(label_values)}
    encoded = frame[label_column].map({value: index for index, value in enumerate(label_values)})
    frame[label_column] = encoded
    columns_to_fill = feature_columns + [label_column]
    frame = _fill_missing(frame, columns_to_fill, spec.get("fillna", "ffill"), group_column)

    val_fraction = float(spec.get("val_fraction", 0.15))
    test_fraction = float(spec.get("test_fraction", 0.2))
    splits = _split_frame(
        frame,
        group_column,
        spec.get("split_strategy", "group"),
        val_fraction,
        test_fraction,
    )

    scaler_name = spec.get("scaler", "standard")
    scaler = None
    if scaler_name == "standard":
        scaler = StandardScaler()
    elif scaler_name == "minmax":
        scaler = MinMaxScaler()
    if scaler is not None:
        scaler.fit(splits["train"][feature_columns].to_numpy(dtype=np.float32))

    output_dir = output_root / spec["name"] / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    window_length = int(spec.get("window_length", 128))
    stride = int(spec.get("stride", 24))
    split_sizes = {}
    for name, split in splits.items():
        X, y = _make_windows(
            split,
            feature_columns,
            label_column,
            group_column,
            window_length,
            stride,
            scaler,
        )
        np.savez_compressed(output_dir / f"{name}_windows.npz", X=X, y=y)
        split_sizes[name] = int(len(X))

    metadata = {
        "dataset": spec["name"],
        "feature_columns": feature_columns,
        "label_column": label_column,
        "label_mapping": label_mapping,
        "group_column": group_column,
        "window_length": window_length,
        "stride": stride,
        "D": len(feature_columns),
        "L": window_length,
        "S": stride,
        "scaler": scaler_name,
        "split_strategy": spec.get("split_strategy", "group"),
        "split_windows": split_sizes,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if scaler is not None:
        joblib.dump(scaler, output_dir / "scaler.joblib")
    return output_dir


class GenericDataLoader:
    """Load custom datasets produced by :func:`prepare_custom_dataset`."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def load_processed_data(self, device: str | None = None) -> dict[str, Any]:
        processed = self.root / "processed"

        def load_split(name: str) -> WindowDataset | None:
            path = processed / f"{name}_windows.npz"
            if not path.is_file():
                return None
            data = np.load(path)
            X = torch.as_tensor(data["X"], dtype=torch.float32)
            y = torch.as_tensor(data["y"], dtype=torch.long)
            if device:
                X, y = X.to(device), y.to(device)
            return WindowDataset(X, y_label=y)

        metadata_path = processed / "metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
        scaler_path = processed / "scaler.joblib"
        return {
            "train_ds": load_split("train"),
            "val_ds": load_split("val"),
            "test_ds": load_split("test"),
            "scaler": joblib.load(scaler_path) if scaler_path.is_file() else None,
            "meta": metadata,
        }
