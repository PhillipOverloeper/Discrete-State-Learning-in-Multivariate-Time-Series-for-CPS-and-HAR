from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path("Daten - Backup/metrics")
WIDE_OUT = Path("metrics_compiled_wide.csv")
LONG_OUT = Path("metrics_compiled_long.csv")


def try_parse_number(x: Any) -> Any:
    """Convert numeric-looking values to int/float where possible."""
    if isinstance(x, (int, float, bool)) or x is None:
        return x
    if isinstance(x, str):
        s = x.strip()
        try:
            if re.fullmatch(r"[+-]?\d+", s):
                return int(s)
            if re.fullmatch(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?", s):
                return float(s)
        except Exception:
            pass
    return x


def flatten_json(obj: Any, prefix: str = "") -> dict[str, Any]:
    """
    Flatten nested dicts/lists into a flat dict.
    Lists of scalars are kept as JSON strings to avoid exploding rows.
    """
    items: dict[str, Any] = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{prefix}.{k}" if prefix else str(k)
            items.update(flatten_json(v, new_key))
    elif isinstance(obj, list):
        # Keep scalar lists as JSON strings; flatten list-of-dicts by index.
        if all(not isinstance(v, (dict, list)) for v in obj):
            items[prefix] = json.dumps(obj)
        else:
            for i, v in enumerate(obj):
                new_key = f"{prefix}[{i}]"
                items.update(flatten_json(v, new_key))
    else:
        items[prefix] = try_parse_number(obj)

    return items


def parse_regular_path(path: Path) -> dict[str, Any] | None:
    """
    Parse:
      metrics/<dataset>/<method>/<K>/<anything>_state_metrics_<seed>.json
    """
    try:
        rel = path.relative_to(ROOT)
        parts = rel.parts
        if len(parts) != 4:
            return None

        dataset, method, k_str, filename = parts

        m = re.fullmatch(r"(.+?)_state_metrics_(\d+)\.json", filename)
        if not m:
            return None

        seed = int(m.group(2))
        return {
            "dataset": dataset,
            "method": method,
            "K": k_str,
            "seed": seed,
            "task": "main",
            "source_type": "regular",
            "file_path": str(path),
        }
    except Exception as e:
        print(f"[DEBUG regular] Failed on {path}: {e}")
        return None


def parse_anomaly_path(path: Path) -> dict[str, Any] | None:
    """
    Parse:
      metrics/<dataset>/<method>/anomaly_detection/<K>/<anything>_state_metrics_<seed>.json
    """
    try:
        rel = path.relative_to(ROOT)
        parts = rel.parts
        if len(parts) != 5:
            return None

        dataset, method, anomaly_dir, k_str, filename = parts
        if anomaly_dir != "anomaly_detection":
            return None

        m = re.fullmatch(r"(.+?)_state_metrics_(\d+)\.json", filename)
        if not m:
            return None

        seed = int(m.group(2))
        return {
            "dataset": dataset,
            "method": method,
            "K": k_str,
            "seed": seed,
            "task": "anomaly_detection",
            "source_type": "anomaly",
            "file_path": str(path),
        }
    except Exception as e:
        print(f"[DEBUG anomaly] Failed on {path}: {e}")
        return None


def parse_path(path: Path) -> dict[str, Any] | None:
    """Try both supported path formats."""
    meta = parse_anomaly_path(path)
    if meta is not None:
        return meta
    meta = parse_regular_path(path)
    if meta is not None:
        return meta
    return None


def classify_task_and_metric(flat_key: str, source_task: str) -> tuple[str, str]:
    """
    Map raw JSON key names into a cleaner task/metric split for the long CSV.
    """
    key = flat_key.lower()

    if source_task == "anomaly_detection":
        return "anomaly_detection", flat_key

    if "ari" in key or key == "v" or "v_measure" in key or "v-measure" in key:
        return "intrinsic", flat_key

    if "classification" in key or key in "f1":
        return "classification", flat_key

    if "forecast" in key or "r2" in key:
        return "forecasting", flat_key

    if "anomaly" in key or "anomaly_auc_roc" in key:
        return "anomaly_detection", flat_key

    return "unknown", flat_key


def load_json_file(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Could not read {path}: {e}")
        return None


def build_dataframes(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []

    for path in root.rglob("*.json"):
        meta = parse_path(path)
        if meta is None:
            print(f"[SKIP] Unrecognized path format: {path}")
            continue

        payload = load_json_file(path)
        if payload is None:
            continue

        flat = flatten_json(payload)
        wide_row = {**meta, **flat}
        wide_rows.append(wide_row)

        for metric_name, value in flat.items():
            # Keep only scalar numeric values in the long CSV.
            if isinstance(value, bool):
                value = int(value)
            if not isinstance(value, (int, float)) or pd.isna(value):
                continue

            task_group, clean_metric = classify_task_and_metric(metric_name, meta["task"])
            long_rows.append(
                {
                    "dataset": meta["dataset"],
                    "task": task_group,
                    "metric": clean_metric,
                    "K": meta["K"],
                    "method": meta["method"],
                    "seed": meta["seed"],
                    "value": value,
                    "source_type": meta["source_type"],
                    "file_path": meta["file_path"],
                }
            )

    wide_df = pd.DataFrame(wide_rows)
    long_df = pd.DataFrame(long_rows)

    if not wide_df.empty:
        wide_df = wide_df.sort_values(["dataset", "method", "K", "seed"]).reset_index(drop=True)
    if not long_df.empty:
        long_df = long_df.sort_values(["dataset", "task", "metric", "method", "K", "seed"]).reset_index(drop=True)

    return wide_df, long_df


def main() -> None:
    wide_df, long_df = build_dataframes(ROOT)

    wide_df.to_csv(WIDE_OUT, index=False)
    long_df.to_csv(LONG_OUT, index=False)

    print(f"Saved wide CSV: {WIDE_OUT} ({len(wide_df)} rows)")
    print(f"Saved long CSV: {LONG_OUT} ({len(long_df)} rows)")

    if not long_df.empty:
        print("\nDetected metric names:")
        for name in sorted(long_df["metric"].unique()):
            print(f"  - {name}")


if True:
    main()