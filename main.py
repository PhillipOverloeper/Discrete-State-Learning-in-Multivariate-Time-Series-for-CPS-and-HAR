import argparse
import copy
import json
import os
import re
import statistics
from pathlib import Path
from typing import Any, Dict, Type

import joblib
import optuna

from CatVAE import compute_catvae_anomaly_score, compute_catvae_metrics, train_catvae
from generic_data import GenericDataLoader
from HMM import compute_hmm_anomaly_score, compute_hmm_metrics, train_hmm
from kMeans import compute_kmeans_anomaly_score, compute_kmeans_metrics, train_kmeans
from metrics import load_metric_summary
from mhealth import MHEALTH_Dataloader
from pamap2 import PAMAP2_Dataloader
from RBM import compute_rbm_anomaly_score, compute_rbm_metrics, train_rbm
from SOMVAE import compute_somvae_anomaly_score, compute_somvae_metrics, train_somvae
from tep import TEPDataLoader
from uci_har import UCI_HAR_Dataloader
from VQ_VAE import compute_vqvae_anomaly_score, compute_vqvae_metrics, train_vqvae
from wadi import WADI_Dataloader

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIRECTORIES = {
    "tep": "tep",
    "har": "HAR",
    "pamap2": "PAMAP2",
    "mhealth": "MHEALTH",
    "wadi": "WADI",
}
CUSTOM_DATA_DIRECTORIES: dict[str, Path] = {}


def _dataset_name(value: str) -> str:
    """Validate names before using them as output-directory components."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise argparse.ArgumentTypeError(
            "Dataset names may contain only letters, numbers, underscores, and hyphens."
        )
    return value


# Dataset registry
DATASET_LOADERS: Dict[str, Type] = {
    "tep": TEPDataLoader,
    "har": UCI_HAR_Dataloader,
    "pamap2": PAMAP2_Dataloader,
    "mhealth": MHEALTH_Dataloader,
    "wadi": WADI_Dataloader,
}

# Configuration paths registry
CONFIG_PATHS: Dict[str, Dict[str, Path]] = {
    "vqvae": {
        "tep": PROJECT_ROOT / "configs/vqvae_tep.json",
        "har": PROJECT_ROOT / "configs/vqvae_har.json",
        "pamap2": PROJECT_ROOT / "configs/vqvae_pamap2.json",
        "mhealth": PROJECT_ROOT / "configs/vqvae_mhealth.json",
        "wadi": PROJECT_ROOT / "configs/vqvae_wadi.json",
    },
    "catvae": {
        "tep": PROJECT_ROOT / "configs/catvae_tep.json",
        "har": PROJECT_ROOT / "configs/catvae_har.json",
        "pamap2": PROJECT_ROOT / "configs/catvae_pamap2.json",
        "mhealth": PROJECT_ROOT / "configs/catvae_mhealth.json",
        "wadi": PROJECT_ROOT / "configs/catvae_wadi.json",
    },
    "somvae": {
        "tep": PROJECT_ROOT / "configs/somvae_tep.json",
        "har": PROJECT_ROOT / "configs/somvae_har.json",
        "pamap2": PROJECT_ROOT / "configs/somvae_pamap2.json",
        "mhealth": PROJECT_ROOT / "configs/somvae_mhealth.json",
        "wadi": PROJECT_ROOT / "configs/somvae_wadi.json",
    },
    "hmm": {
        "tep": PROJECT_ROOT / "configs/hmm_tep.json",
        "har": PROJECT_ROOT / "configs/hmm_har.json",
        "pamap2": PROJECT_ROOT / "configs/hmm_pamap2.json",
        "mhealth": PROJECT_ROOT / "configs/hmm_mhealth.json",
        "wadi": PROJECT_ROOT / "configs/hmm_wadi.json",
    },
    "kmeans": {
        "tep": PROJECT_ROOT / "configs/kmeans_tep.json",
        "har": PROJECT_ROOT / "configs/kmeans_har.json",
        "pamap2": PROJECT_ROOT / "configs/kmeans_pamap2.json",
        "mhealth": PROJECT_ROOT / "configs/kmeans_mhealth.json",
        "wadi": PROJECT_ROOT / "configs/kmeans_wadi.json",
    },
    # Post-paper extension: this model was not part of the ETFA experiments.
    "rbm": {},
}
DEFAULT_CONFIG_PATHS = {
    model: PROJECT_ROOT / "configs" / "defaults" / f"{model}.json"
    for model in CONFIG_PATHS
}


def parse_args():
    """
    Parse the configuration inputs.
    """
    parser = argparse.ArgumentParser(
        description="Run the ETFA discrete-state models and explicitly marked post-paper extensions.",
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(CONFIG_PATHS.keys()),
        help="Model to run.",
    )

    parser.add_argument(
        "--dataset",
        required=True,
        type=_dataset_name,
        help="Built-in or prepared custom dataset name.",
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=["train", "evaluate", "tune", "anomaly"],
        help="Experiment node.",
    )

    parser.add_argument(
        "--states",
        type=int,
        nargs="+",
        default=[4],
        help="State-space sizes to run (default: 4).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Random seeds to run (default: 42).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=30,
        help="Number of Optuna trials in tune mode (default: 30).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional JSON configuration overriding configs/<model>_<dataset>.json.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Custom dataset directory containing processed/ (default: data/custom/<dataset>).",
    )

    return parser.parse_args()


def _trial_to_dict(trial: optuna.trial.FrozenTrial) -> dict:
    """
    Convert an Optuna trial to a JSON-serialisable dict.
    """
    return {
        "number": trial.number,
        "state": trial.state.name,
        "value": trial.value,
        "params": trial.params,
        "user_attrs": trial.user_attrs,
        "system_attrs": trial.system_attrs,
        "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else None,
        "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else None,
        "duration_seconds": trial.duration.total_seconds() if trial.duration else None,
    }


def tune_optuna_and_save_all_runs(objective, base_config: dict, dataset: str, model: str, n_trials: int,
                                  seeds: list[int], save_dir: str, dim: Any):
    """
    Conducts a hyperparameter search for the specified model and dataset and saves all runs,
    including the best one.
    """
    # Create study object
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    # Extract best parameters
    best_params = study.best_params
    best_value = study.best_value

    best_config = copy.deepcopy(base_config)
    for k, v in best_params.items():
        if k == "lr" or k == "batch_size":
            best_config["train"][k] = v
        else:
            best_config["model"][k] = v

    # Set up the folders
    run_dir = Path(save_dir) / dataset / model /f"{str(dim)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save the best configuration
    best_path = run_dir / "optuna_best_config.json"
    with open(best_path, "w") as f:
        json.dump(best_config, f, indent=2)

    # Save the summary
    summary_path = run_dir / "optuna_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "model": model,
                "dataset": dataset,
                "n_trials": n_trials,
                "seeds": list(seeds),
                "direction": study.direction.name,
                "best_value": best_value,
                "best_params": best_params,
            },
            f,
            indent=2,
        )

    # Save all trials as CSV and JSON
    trials_csv_path = run_dir / "optuna_trials.csv"
    df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
    df.to_csv(trials_csv_path, index=False)

    trials_jsonl_path = run_dir / "optuna_trials.jsonl"
    with open(trials_jsonl_path, "w") as f:
        for t in study.trials:
            f.write(json.dumps(_trial_to_dict(t)) + "\n")

    # Save whole study object as pickle
    try:
        study_pkl_path = run_dir / "optuna_study.pkl"
        joblib.dump(study, study_pkl_path)
        saved_pkl = True
    except Exception:
        saved_pkl = False

    print("\n[OPTUNA] Finished.")
    print(f"[OPTUNA] Best value = {best_value:.6f}")
    print(f"[OPTUNA] Best params = {best_params}")
    print(f"[OPTUNA] Saved best config to: {best_path}")
    print(f"[OPTUNA] Saved summary to: {summary_path}")
    print(f"[OPTUNA] Saved ALL trials CSV to: {trials_csv_path}")
    print(f"[OPTUNA] Saved ALL trials JSONL to: {trials_jsonl_path}")
    if saved_pkl:
        print(f"[OPTUNA] Saved study pickle to: {study_pkl_path}")

    return best_config, study


def load_config(
    model: str,
    dataset: str,
    *,
    config_path: Path | None = None,
    tuned: bool = False,
    states: str | None = None,
) -> dict:
    """
    Return the configuration for the given model and dataset.
    """
    if config_path is not None:
        path = config_path.expanduser().resolve()
    elif tuned:
        path = PROJECT_ROOT / "tuning_results" / dataset / model / str(states) / "optuna_best_config.json"
    else:
        path = CONFIG_PATHS[model].get(dataset)
        if path is None:
            path = DEFAULT_CONFIG_PATHS[model]
    if not path.is_file():
        raise FileNotFoundError(
            f"Configuration not found: {path}. Run tuning first or pass --config."
        )
    with open(path) as f:
        return json.load(f)



def _dataset_root(dataset: str) -> Path:
    if dataset in DATA_DIRECTORIES:
        return PROJECT_ROOT / "data" / DATA_DIRECTORIES[dataset]
    return CUSTOM_DATA_DIRECTORIES.get(dataset, PROJECT_ROOT / "data" / "custom" / dataset)


def load_dataset(dataset: str, anomaly=False):
    """
    Load the given dataset.
    """
    if dataset not in DATASET_LOADERS:
        if anomaly:
            raise ValueError("Anomaly mode currently supports only the built-in TEP dataset.")
        loader = GenericDataLoader(_dataset_root(dataset))
        out = loader.load_processed_data()
        _validate_dataset(out, dataset)
        return out
    if not anomaly:
        loader_cls = DATASET_LOADERS[dataset]
        loader = loader_cls(root=str(_dataset_root(dataset)))
        out = loader.load_processed_data()
        _validate_dataset(out, dataset)
        return out
    else:
        loader_cls = DATASET_LOADERS[dataset]
        loader = loader_cls(root=str(_dataset_root(dataset)))
        anomaly_out = loader.load_processed_data()
        normal_out = loader.load_processed_data(anomaly=True)
        _validate_dataset(normal_out, dataset)
        _validate_dataset(anomaly_out, dataset)
        return normal_out, anomaly_out


def _validate_dataset(out: dict, dataset: str) -> None:
    """Fail early with an actionable message when processed data is absent."""
    missing = [name for name in ("train_ds", "val_ds", "test_ds") if out.get(name) is None]
    if missing:
        data_dir = _dataset_root(dataset)
        raise FileNotFoundError(
            f"Missing processed {dataset!r} splits: {', '.join(missing)}. "
            f"Place the prepared dataset below {data_dir} as described in README.md."
        )


def train_and_evaluate(model: str, dataset: str, config: dict) -> dict:
    """
    Train and evaluate the given model on the given dataset and return the entire metrics.
    """
    # Load the dataset
    out = load_dataset(dataset)

    # Train the given model on the given dataset
    if model == "vqvae":
        train_vqvae(out, config)
        metrics = compute_vqvae_metrics(config, out, dataset)
    elif model == "catvae":
        train_catvae(out, config)
        metrics = compute_catvae_metrics(config, out, dataset)
    elif model == "somvae":
        train_somvae(out, config)
        metrics = compute_somvae_metrics(config, out, dataset)
    elif model == "hmm":
        train_hmm(out, config)
        metrics = compute_hmm_metrics(config, out, dataset)
    elif model == "kmeans":
        train_kmeans(out, config)
        metrics = compute_kmeans_metrics(config, out, dataset)
    elif model == "rbm":
        train_rbm(out, config)
        metrics = compute_rbm_metrics(config, out, dataset)
    else:
        raise ValueError(f"Unknown model: {model}")

    return metrics

def train(model: str, dataset: str, config: dict, seeds: list, states: Any):
    """
    Train and evaluate the given model on the given dataset.
    """
    # Load the dataset
    out = load_dataset(dataset)
    for s in seeds:
        cfg_seeded = copy.deepcopy(config)
        cfg_seeded["seed"] = int(s)

        cfg_seeded["train"]["ckpt_name"] = f"{model}_best_{s}.pt"
        cfg_seeded["train"]["ckpt_dir"] = str(
            PROJECT_ROOT / "checkpoints" / model / dataset / str(states)
        )

        # Train the given model on the given dataset
        if model == "vqvae":
            train_vqvae(out, cfg_seeded)
        elif model == "catvae":
            train_catvae(out, cfg_seeded)
        elif model == "somvae":
            train_somvae(out, cfg_seeded)
        elif model == "hmm":
            train_hmm(out, cfg_seeded)
        elif model == "kmeans":
            train_kmeans(out, cfg_seeded)
        elif model == "rbm":
            train_rbm(out, cfg_seeded)
        else:
            raise ValueError(f"Unknown model: {model}")

def train_anomaly(model: str, dataset: str, config: dict, seeds: list, states: Any):
    """
    Train and evaluate the given model on the given dataset and return the entire metrics.
    """
    # Load the dataset
    normal_out, anomaly_out = load_dataset(dataset, True)

    for s in seeds:
        cfg_seeded = copy.deepcopy(config)
        cfg_seeded["seed"] = int(s)

        cfg_seeded["train"]["ckpt_name"] = f"{model}_best_{s}.pt"
        cfg_seeded["train"]["ckpt_dir"] = str(
            PROJECT_ROOT / "anomaly_checkpoints" / model / dataset / str(states)
        )

        # Train the given model on the given dataset
        if model == "vqvae":
            train_vqvae(normal_out, cfg_seeded)
            compute_vqvae_anomaly_score(cfg_seeded, normal_out, anomaly_out, dataset)
        elif model == "catvae":
            train_catvae(normal_out, cfg_seeded)
            compute_catvae_anomaly_score(cfg_seeded, normal_out, anomaly_out, dataset)
        elif model == "somvae":
            train_somvae(normal_out, cfg_seeded)
            compute_somvae_anomaly_score(cfg_seeded, normal_out, anomaly_out, dataset)
        elif model == "hmm":
            train_hmm(normal_out, cfg_seeded)
            compute_hmm_anomaly_score(cfg_seeded, normal_out, anomaly_out, dataset)
        elif model == "kmeans":
            train_kmeans(normal_out, cfg_seeded)
            compute_kmeans_anomaly_score(cfg_seeded, normal_out, anomaly_out, dataset)
        elif model == "rbm":
            train_rbm(normal_out, cfg_seeded)
            compute_rbm_anomaly_score(cfg_seeded, normal_out, anomaly_out, dataset)
        else:
            raise ValueError(f"Unknown model: {model}")

def evaluation(model: str, dataset: str, config: dict, seeds: list, states: str):
    """
    Train and evaluate the given model on the given dataset and return the entire metrics.
    """
    # Load the dataset
    out = load_dataset(dataset)
    for s in seeds:
        cfg_seeded = copy.deepcopy(config)
        cfg_seeded["seed"] = int(s)

        cfg_seeded["train"]["ckpt_name"] = f"{model}_best_{s}.pt"
        cfg_seeded["train"]["ckpt_dir"] = str(
            PROJECT_ROOT / "checkpoints" / model / dataset / str(states)
        )

        # Train the given model on the given dataset
        if model == "vqvae":
            compute_vqvae_metrics(cfg_seeded, out, dataset)
        elif model == "catvae":
            compute_catvae_metrics(cfg_seeded, out, dataset)
        elif model == "somvae":
            compute_somvae_metrics(cfg_seeded, out, dataset)
        elif model == "hmm":
            hmm_ckpt = os.path.join(
                cfg_seeded["train"]["ckpt_dir"],
                cfg_seeded["train"]["ckpt_name"],
            )
            if not os.path.exists(hmm_ckpt):
                print(f"Skipping seed {s}: {hmm_ckpt} not found")
                continue
            compute_hmm_metrics(cfg_seeded, out, dataset)
        elif model == "kmeans":
            compute_kmeans_metrics(cfg_seeded, out, dataset)
        elif model == "rbm":
            compute_rbm_metrics(cfg_seeded, out, dataset)
        else:
            raise ValueError(f"Unknown model: {model}")


def train_models(model: str, dataset: str, config: dict) -> float:
    """
    Train the given model on the given dataset and returns the best validations loss.
    Used for hyperparameter tuning.
    """
    # Load the dataset
    out = load_dataset(dataset)

    # Train the given model on the given dataset
    if model == "vqvae":
        metrics = train_vqvae(out, config)
    elif model == "catvae":
        metrics = train_catvae(out, config)
    elif model == "somvae":
        metrics = train_somvae(out, config)
    elif model == "hmm":
        metrics = train_hmm(out, config)
    elif model == "rbm":
        metrics = train_rbm(out, config)
    else:
        raise ValueError(f"Unknown model: {model}")

    return metrics["best_val_total"]


def tune_optuna(model: str, dataset: str, base_config: dict, n_trials: int = 3, seeds=(0, 1, 2),
                save_dir: str = "tuning_results", state_dims: list[int] | None = None):
    """
    Bayesian Optimization using Optuna (TPE). Minimises the validation loss.
    """
    # Create path for dataset and model, if it does not exist yet
    path = Path(os.path.join(save_dir, dataset, model))
    path.mkdir(parents=True, exist_ok=True)

    def run_tuning(trial: optuna.Trial) -> float:
        """
        Set up the Optuna objective for the given model and conduct hyperparameter tuning for the given seeds. .
        """
        # Copy the layout of the configuration
        cfg = copy.deepcopy(base_config)

        # Suggest hyperparameter ranges based on given model
        if model == "vqvae":
            cfg["train"]["lr"] = trial.suggest_float("lr", 1e-5, 5e-2, log=True)
            cfg["train"]["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
            cfg["model"]["beta"] = trial.suggest_float("beta", 0.0, 1.5, log=False)
        elif model == "catvae":
            cfg["train"]["lr"] = trial.suggest_float("lr", 1e-5, 5e-2, log=True)
            cfg["train"]["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
            cfg["model"]["beta"] = trial.suggest_float("beta", 0.0, 1.5, log=False)
        elif model == "somvae":
            cfg["train"]["lr"] = trial.suggest_float("lr", 1e-5, 5e-2, log=True)
            cfg["train"]["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
            cfg["model"]["alpha"] = trial.suggest_float("alpha", 0.0, 1.5, log=False)
            cfg["model"]["beta"] = trial.suggest_float("beta", 0.0, 1.5, log=False)
            cfg["model"]["gamma"] = trial.suggest_float("gamma", 0.0, 1.5, log=False)
            cfg["model"]["tau"] = trial.suggest_float("tau", 0.0, 1.5, log=False)
        elif model == "hmm":
            cfg["model"]["covariance_type"] = trial.suggest_categorical(
                "covariance_type", ["diag", "spherical", "full"]
            )
        elif model == "rbm":
            cfg["train"]["lr"] = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
            cfg["train"]["batch_size"] = trial.suggest_categorical(
                "batch_size", [32, 64, 128, 256]
            )
            cfg["model"]["cd_steps"] = trial.suggest_int("cd_steps", 1, 5)
        else:
            raise ValueError(f"Unknown model: {model}")

        # Evaluate across multiple seeds
        scores = []
        for s in seeds:
            cfg_seeded = copy.deepcopy(cfg)
            cfg_seeded["seed"] = int(s)
            # Train the model with the given set of hyperparameters and seed
            score = train_models(model, dataset, cfg_seeded)
            scores.append(score)

        # Compute mean and standard deviation of the scores
        mean_score = statistics.mean(scores)
        std_score = statistics.pstdev(scores) if len(scores) > 1 else 0.0

        trial.set_user_attr("scores_per_seed", scores)
        trial.set_user_attr("std_score", std_score)

        return mean_score



    state_dims = state_dims or [4, 6, 9, 12, 16, 20, 25]
    if model == "vqvae":
        state_dim = state_dims
        for dim in state_dim:
            print(f"Train for dimension {dim}")
            base_config["model"]["num_embeddings"] = dim
            best_config, study = tune_optuna_and_save_all_runs(run_tuning, base_config, dataset, model, n_trials, seeds,
                                                               save_dir, dim)
    elif model == "catvae":
        state_dim = state_dims
        for dim in state_dim:
            print(f"Train for dimension {dim}")
            base_config["model"]["categorical_dim"] = dim
            best_config, study = tune_optuna_and_save_all_runs(run_tuning, base_config, dataset, model, n_trials, seeds,
                                                               save_dir, dim)
    elif model == "somvae":
        state_dim = [_som_grid(dim) for dim in state_dims]
        for dim in state_dim:
            print(f"Train for dimension {dim}")
            base_config["model"]["som_dim"] = dim
            best_config, study = tune_optuna_and_save_all_runs(run_tuning, base_config, dataset, model, n_trials, seeds,
                                                               save_dir, dim)
    elif model == "hmm":
        state_dim = state_dims
        for dim in state_dim:
            print(f"Train for dimension {dim}")
            base_config["model"]["num_states"] = dim
            best_config, study = tune_optuna_and_save_all_runs(run_tuning, base_config, dataset, model, n_trials, seeds,
                                                               save_dir, dim)

    elif model == "rbm":
        for dim in state_dims:
            print(f"Train for dimension {dim}")
            base_config["model"]["num_hidden"] = dim
            best_config, study = tune_optuna_and_save_all_runs(
                run_tuning, base_config, dataset, model, n_trials, seeds, save_dir, dim
            )

    else:
        raise ValueError(f"Hyperparameter tuning is not implemented for model {model!r}.")


    return best_config


def _som_grid(num_states: int) -> tuple[int, int]:
    """Return the most balanced integer SOM grid for a state count."""
    rows = int(num_states ** 0.5)
    while rows > 1 and num_states % rows:
        rows -= 1
    return rows, num_states // rows


def _config_for_states(config: dict, model: str, states: int) -> tuple[dict, Any]:
    """Copy a base configuration and set its model-specific state-space size."""
    configured = copy.deepcopy(config)
    state_label: Any = states
    if model == "vqvae":
        configured["model"]["num_embeddings"] = states
    elif model == "catvae":
        configured["model"]["categorical_dim"] = states
    elif model == "somvae":
        state_label = _som_grid(states)
        configured["model"]["som_dim"] = list(state_label)
    elif model == "hmm":
        configured["model"]["num_states"] = states
    elif model == "kmeans":
        configured["model"]["num_embeddings"] = states
        configured["model"]["n_init"] = states
    elif model == "rbm":
        configured["model"]["num_hidden"] = states
    return configured, state_label


def run_experiment(
    model: str,
    dataset: str,
    mode: str,
    states: list[int],
    seeds: list[int],
    config_path: Path | None = None,
    trials: int = 30,
    data_dir: Path | None = None,
):
    """
    Main function to either conduct hyperparameter search of simple training of the given model
    on the given dataset.
    """
    _dataset_name(dataset)
    if data_dir is not None:
        CUSTOM_DATA_DIRECTORIES[dataset] = data_dir.expanduser().resolve()
    base_config = load_config(model, dataset, config_path=config_path)

    if mode == "tune":
        print("Begin hyperparameter tuning")
        tune_optuna(model, dataset, base_config, trials, seeds=seeds, state_dims=states)
    elif mode == "evaluate":
        print("Begin Evaluation")
        for num_states in states:
            config, state_label = _config_for_states(base_config, model, num_states)
            evaluation(model, dataset, config, seeds, str(state_label))
            load_metric_summary(model, dataset, str(state_label))
    elif mode == "train":
        print("Begin Training")
        for num_states in states:
            config, state_label = _config_for_states(base_config, model, num_states)
            train(model, dataset, config, seeds, str(state_label))
    elif mode == "anomaly":
        if dataset != "tep":
            raise ValueError("Anomaly mode currently supports only the TEP dataset.")
        train_for_anomaly(model, dataset, base_config, states, seeds)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def train_for_anomaly(model, dataset, base_config, states, seeds):
    """
    Train the modek for anomaly detection and then analyse it
    """
    for num_states in states:
        config, state_label = _config_for_states(base_config, model, num_states)
        train_anomaly(model, dataset, config, seeds, str(state_label))




if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    args = parse_args()
    run_experiment(
        model=args.model,
        dataset=args.dataset,
        mode=args.mode,
        states=args.states,
        seeds=args.seeds,
        config_path=args.config,
        trials=args.trials,
        data_dir=args.data_dir,
    )
