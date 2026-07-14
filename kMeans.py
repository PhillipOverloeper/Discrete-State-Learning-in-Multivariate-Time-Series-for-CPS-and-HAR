import os
from dataclasses import dataclass

import joblib
import numpy as np
import torch
from sklearn.cluster import KMeans

from metrics import compute_downstream_metrics, compute_state_label_metrics, save_state_metrics
from utils import _dc_to_dict


@dataclass(frozen=True)
class ModelConfig:
    num_embeddings: int = 12
    init: str = "k-means++"
    n_init: int = 10


@dataclass(frozen=True)
class TrainConfig:
    max_iter: int = 200
    ckpt_dir: str = "./checkpoints/kmeans/"
    ckpt_name: str = "kmeans_best.joblib"

    seed: int = 42


def parse_model_config(config):
    m = config["model"]
    return ModelConfig(
        num_embeddings=int(m.get("num_embeddings", 12)),
        init=str(m.get("init", "k-means++")),
        n_init=int(m.get("n_init", 12)),
    )


def parse_train_config(config):
    t = config["train"]

    seed = int(config.get("seed", 42))

    return TrainConfig(
        max_iter=int(t.get("max_iter",50)),
        ckpt_dir=str(t.get("ckpt_dir", "./checkpoints/kmeans/")),
        ckpt_name=str(t.get("ckpt_name", "kmeans_best.pkl")),
        seed=seed,
    )


def train_kmeans(out, config):
    m = parse_model_config(config)
    t = parse_train_config(config)

    np.random.seed(t.seed)

    X_train = out["train_ds"].X
    N, L, D = X_train.shape
    X_train_flat = X_train.reshape(-1, D)

    X_val = out["val_ds"].X
    X_val_flat = X_val.reshape(-1, D)


    model = KMeans(
        n_clusters=m.num_embeddings,
        init=m.init,
        n_init=m.n_init,
        max_iter=t.max_iter,
        random_state=t.seed,
        verbose=0
    )

    model.fit(X_train_flat)

    train_inertia = model.inertia_ / len(X_train_flat)

    # Calculate Val Loss (Squared distances to the centers found on Train)
    # This is the standard way to get a "Val Loss" for k-means
    val_distances = model.transform(X_val_flat)
    val_inertia = np.sum(np.min(val_distances, axis=1) ** 2) / len(X_val_flat)

    history = {
        "train_inertia": [train_inertia],
        "val_inertia": [val_inertia],
    }

    # Save to match your pickle/joblib pattern
    os.makedirs(t.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(t.ckpt_dir, t.ckpt_name)
    joblib.dump({"model": model, "history": history, "config": config}, ckpt_path)

    print(f"KMeans Final | Train Inertia: {train_inertia:.6f} | Val Inertia: {val_inertia:.6f}")

    return {"history": history, "ckpt_path": ckpt_path}


def compute_kmeans_metrics(config, out, dataset, save=True):
    m = parse_model_config(config)
    t = parse_train_config(config)
    ckpt_path = os.path.join(t.ckpt_dir, t.ckpt_name)

    # Load the dictionary and extract the model
    ckpt = joblib.load(ckpt_path)
    model = ckpt["model"]

    # Get test data
    X_test = out["test_ds"].X
    y_test = out["test_ds"].y_label
    N, L, D = X_test.shape

    # 2. Flatten for the model prediction
    X_test_flat = X_test.reshape(-1, D)
    predicted_states_flat = model.predict(X_test_flat)

    # 3. Reshape predicted labels back to the "Windowed" format (N, L)
    # This converts the flat array of predictions into N windows of length L
    predicted_states_windowed = predicted_states_flat.reshape(N, L)

    # 4. Final conversion to multidimensional tensors
    # states_list_test will now be a tensor of shape [N, L]
    states_list_test = [torch.tensor(predicted_states_windowed, dtype=torch.long)]
    labels_list_test = [torch.as_tensor(y_test, dtype=torch.long).detach().clone()]

    # Get train data
    X_train = out["train_ds"].X
    y_train = out["train_ds"].y_label
    N, L, D = X_train.shape
    X_train_flat = X_train.reshape(-1, D)
    # 1. Predict cluster indices (The "States")
    predicted_states = model.predict(X_train_flat)
    predicted_states = predicted_states.reshape(N, L)
    # 2. Prepare State-Label Metrics
    # We wrap them in lists to match your compute_state_label_metrics signature
    states_list_train = [torch.tensor(predicted_states, dtype=torch.long)]
    labels_list_train = [torch.as_tensor(y_train, dtype=torch.long).detach().clone()]

    class_results, forecast_results, _ = compute_downstream_metrics(train_latents=states_list_train, train_labels=labels_list_train, train_features=[X_train],
                               test_latents=states_list_test, test_labels=labels_list_test, test_features=[X_test],
                               classification=True, forecasting=True)

    # 3. Compute ARI, NMI, and Purity
    state_metrics, per_state_entropy, per_state_purity, state_majority_label = compute_state_label_metrics(
        latent_data_list=states_list_test,
        valid_states=labels_list_test,
        num_total_states=m.num_embeddings,
    )

    combined_metrics = state_metrics
    combined_metrics.update(class_results)
    combined_metrics.update(forecast_results)

    if save:
        out_path = save_state_metrics(
            combined_metrics,
            per_state_entropy,
            per_state_purity,
            state_majority_label,
            save_dir=f"metrics/{dataset}/kmeans/{m.num_embeddings}",
            filename=f"kmeans_state_metrics_{t.seed}.json",
            extra={
                "run_name": "kmeans_baseline",
                "model_config": _dc_to_dict(m),
                "train_config": _dc_to_dict(t),
                "data_meta": out.get("meta", {}),
                "train_history": ckpt.get("history", {})
            },
        )
        print(f"Saved KMeans clustering metrics for {dataset}")
        return combined_metrics, out_path

    return combined_metrics

def compute_kmeans_anomaly_score(config, normal_out, anomaly_out, dataset, save=True):
    m = parse_model_config(config)
    t = parse_train_config(config)
    ckpt_path = os.path.join(t.ckpt_dir, t.ckpt_name)

    # Load the dictionary and extract the model
    ckpt = joblib.load(ckpt_path)
    model = ckpt["model"]

    # Get test data
    X_test_normal = normal_out["test_ds"].X
    N, L, D = X_test_normal.shape
    # 2. Flatten for the model prediction
    X_test_normal_flat = X_test_normal.reshape(-1, D)
    distances_normal = model.transform(X_test_normal_flat)
    min_distances_normal = np.min(distances_normal, axis=1)

    # Get train data
    X_test_anomaly = anomaly_out["test_ds"].X
    N, L, D = X_test_anomaly.shape
    X_test_anomaly_flat = X_test_anomaly.reshape(-1, D)
    # 1. Predict cluster indices (The "States")
    distances_anomaly = model.transform(X_test_anomaly_flat)
    min_distances_anomaly = np.min(distances_anomaly, axis=1)

    _, _, anomaly_metrics = compute_downstream_metrics(test_ff_error=min_distances_normal,
                                                       test_faulty_error=min_distances_anomaly,
                                                       anomaly_detection=True)
    # Merge everything into one dict for saving
    combined_metrics = {}
    combined_metrics.update(anomaly_metrics)
    if save:
        out_path = save_state_metrics(
            combined_metrics,
            {},
            {},
            {},
            save_dir=f"metrics/{dataset}/kmeans/anomaly_detection/{str(m.num_embeddings)}/",
            filename=f"kmeans_state_metrics_{t.seed}.json",
            extra={
                "run_name": "kmeans_baseline",
                "split": "test",
                "checkpoint_path": ckpt_path,
                "model_config": _dc_to_dict(m),
                "train_config": _dc_to_dict(t),
                "normal data_meta": normal_out.get("meta", {}),
                "anomaly data_meta": anomaly_out.get("meta", {}),
            },
        )
        print("Saved metrics to:", out_path)
        return combined_metrics, out_path
    else:
        return combined_metrics














