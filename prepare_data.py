"""Prepare raw benchmark datasets for the experiment runner."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"


def _find_one(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if not matches:
        raise FileNotFoundError(f"Could not find {name!r} below {root}")
    return matches[0]


def _prepare_har(raw_dir: Path) -> None:
    target = DATA_ROOT / "HAR"
    target.mkdir(parents=True, exist_ok=True)

    features_file = _find_one(raw_dir, "features.txt")
    features = pd.read_csv(features_file, sep=r"\s+", header=None, names=["index", "name"])
    # The UCI archive contains duplicate feature names; suffixes keep CSV columns unique.
    feature_names = [f"{name}__{index}" for index, name in zip(features["index"], features["name"])]

    frames = []
    for split in ("train", "test"):
        X = pd.read_csv(_find_one(raw_dir, f"X_{split}.txt"), sep=r"\s+", header=None)
        y = pd.read_csv(_find_one(raw_dir, f"y_{split}.txt"), header=None)[0]
        subjects = pd.read_csv(_find_one(raw_dir, f"subject_{split}.txt"), header=None)[0]
        X.columns = feature_names
        X["subject"] = subjects.to_numpy()
        X["Activity"] = y.to_numpy()
        frames.append(X)

    data = pd.concat(frames, ignore_index=True)
    for subject, frame in data.groupby("subject", sort=True):
        frame.to_csv(target / f"data{int(subject)}.csv", index=False)

    from uci_har import UCI_HAR_Dataloader

    loader = UCI_HAR_Dataloader(root=str(target))
    loader.save_processed_data(loader._prepare_har_datasets())


def _prepare_subject_files(dataset: str, raw_dir: Path) -> None:
    if dataset == "pamap2":
        patterns = ("subject*.dat",)
        target = DATA_ROOT / "PAMAP2"
        from pamap2 import PAMAP2_Dataloader as Loader

        method_name = "_prepare_pamap2_datasets"
    else:
        patterns = ("mHealth_subject*.log", "mhealth_subject*.log")
        target = DATA_ROOT / "MHEALTH"
        from mhealth import MHEALTH_Dataloader as Loader

        method_name = "_prepare_mhealth_datasets"

    files = []
    for pattern in patterns:
        files.extend(raw_dir.rglob(pattern))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No raw {dataset} subject files found below {raw_dir}")

    target.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(files, start=1):
        frame = pd.read_csv(source, sep=r"\s+", header=None)
        frame.to_csv(target / f"data{index}.csv", index=False, header=False)

    loader = Loader(root=str(target), filenames=[f"data{i}.csv" for i in range(1, len(files) + 1)])
    prepared = getattr(loader, method_name)()
    loader.save_processed_data(prepared)


def _prepare_tep(raw_dir: Path) -> None:
    target = DATA_ROOT / "tep"
    target.mkdir(parents=True, exist_ok=True)
    names = (
        "TEP_FaultFree_Training.RData",
        "TEP_FaultFree_Testing.RData",
        "TEP_Faulty_Training.RData",
        "TEP_Faulty_Testing.RData",
    )
    for name in names:
        source = _find_one(raw_dir, name)
        destination = target / name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)

    from tep import TEPDataLoader

    loader = TEPDataLoader(root=str(target))
    loader.save_processed_data(loader._prepare_tep_datasets())
    loader.prepare_anomaly_detection_dataset()


def _prepare_wadi(raw_dir: Path) -> None:
    target = DATA_ROOT / "WADI"
    target.mkdir(parents=True, exist_ok=True)
    source = _find_one(raw_dir, "WADI_normal.csv")
    destination = target / "WADI_normal.csv"
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)

    from wadi import WADI_Dataloader

    loader = WADI_Dataloader(root=str(target))
    loader.save_processed_data(loader._prepare_wadi_datasets())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", choices=("har", "mhealth", "pamap2", "tep", "wadi"))
    source.add_argument(
        "--spec",
        type=Path,
        help="JSON specification for a custom dataset (see configs/datasets/example.json).",
    )
    parser.add_argument(
        "--raw-dir",
        required=True,
        type=Path,
        help="Directory containing the downloaded and extracted raw dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir.expanduser().resolve()
    if not raw_dir.is_dir():
        raise NotADirectoryError(f"Raw-data directory does not exist: {raw_dir}")

    if args.spec:
        from generic_data import prepare_custom_dataset

        output = prepare_custom_dataset(
            raw_dir,
            args.spec.expanduser().resolve(),
            DATA_ROOT / "custom",
        )
        print(f"Processed custom dataset saved to {output}")
    elif args.dataset == "har":
        _prepare_har(raw_dir)
    elif args.dataset in {"mhealth", "pamap2"}:
        _prepare_subject_files(args.dataset, raw_dir)
    elif args.dataset == "tep":
        _prepare_tep(raw_dir)
    else:
        _prepare_wadi(raw_dir)


if __name__ == "__main__":
    main()
