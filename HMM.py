import copy
import os
import pickle
from dataclasses import dataclass

import numpy as np
import torch
from hmmlearn.hmm import GaussianHMM

from metrics import compute_downstream_metrics, compute_state_label_metrics, save_state_metrics
from utils import _dc_to_dict, make_loaders_from_out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class ModelConfig:
    num_states: int = 128
    emission: str = "gaussian"
    covariance_type: str = "diag"
    topology: str = "ergodic"
    left_right_self_prob: float = 0.9

    init_params: str = "stmc"
    params: str = "stmc"


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 100
    early_stop_patience: int = 10

    tol: float = 0.0

    ckpt_dir: str = "./checkpoints/hmm/"
    ckpt_name: str = "hmm_best.pkl"

    seed: int = 42


# ----------------------------
# Parse functions (HMM)
# ----------------------------
def parse_model_config(config):
    m = config["model"]
    return ModelConfig(
        num_states=int(m.get("num_states", 120)),
        emission=str(m.get("emission", "gaussian")),
        covariance_type=str(m.get("covariance_type", "diag")),
        topology=str(m.get("topology", "ergodic")),
        left_right_self_prob=float(m.get("left_right_self_prob", 0.9)),
        init_params=str(m.get("init_params", "stmc")),
        params=str(m.get("params", "stmc")),
    )


def parse_train_config(config):
    t = config["train"]

    seed = int(config.get("seed", t.get("seed", 42)))

    return TrainConfig(
        epochs=int(t.get("epochs", 50)),
        early_stop_patience=int(t.get("early_stop_patience", 15)),
        tol=float(t.get("tol", 0.0)),
        ckpt_dir=str(t.get("ckpt_dir", "./checkpoints/hmm/")),
        ckpt_name=str(t.get("ckpt_name", "hmm_best.pkl")),
        seed=seed,
    )


def loader_to_hmm_arrays(loader):
    X_list = []
    lengths = []
    labels = []
    data = []

    for batch in loader:
        x, y = batch
        data.append(x)
        x = x.to(device)
        B, L, D = x.shape

        # 1. Flatten the batch into a single continuous stream of observations
        # X_cat needs to be [(B * L), D]
        x_flattened = x.reshape(B * L, D).detach().cpu().numpy()
        X_list.append(x_flattened)

        # 2. Track the length of each individual sequence in this batch
        # Each of the B sequences has length L
        lengths.extend([L] * B)

        # 3. Handle labels (typically one label per sequence)
        labels.append(y.detach().cpu().numpy())

        # Concatenate all batches into final arrays
    X_cat = np.concatenate(X_list, axis=0)
    labels = np.concatenate(labels, axis=0)

    return X_cat, data, lengths, labels


def _is_valid_hmm(hmm, atol=1e-8):
    try:
        sp = hmm.startprob_
        tm = hmm.transmat_
    except AttributeError:
        return False

    if not np.all(np.isfinite(sp)) or not np.all(np.isfinite(tm)):
        return False
    if np.any(sp < 0) or np.any(tm < 0):
        return False
    if not np.isclose(sp.sum(), 1.0, atol=atol):
        return False

    row_sums = tm.sum(axis=1)
    if not np.all(np.isclose(row_sums, 1.0, atol=atol)):
        return False

    return True


def train_hmm(out, config):
    m = parse_model_config(config)
    t = parse_train_config(config)

    np.random.seed(t.seed)
    torch.manual_seed(t.seed)

    os.makedirs(t.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(t.ckpt_dir, t.ckpt_name)

    train_loader, val_loader, _ = make_loaders_from_out(out, config)

    X_train, _, lengths_train, _ = loader_to_hmm_arrays(train_loader)
    X_val, _, lengths_val, _ = loader_to_hmm_arrays(val_loader)

    n = m.num_states

    hmm = GaussianHMM(
        n_components=n,
        covariance_type=m.covariance_type,
        n_iter=1,                 # one EM step per outer epoch
        tol=t.tol,
        random_state=t.seed,
        verbose=False,
        min_covar=1e-3,
        params="stmc",
        init_params="stmc",
        startprob_prior=np.full(n, 2.0),
        transmat_prior=np.full((n, n), 2.0),
    )

    best_val = float("inf")
    best_epoch = -1
    patience_left = t.early_stop_patience

    history = {
        "train_nll": [],
        "val_nll": [],
        "best_ckpt_path": ckpt_path,
        "failed": False,
        "failure_reason": None,
    }

    for epoch in range(1, t.epochs + 1):
        try:
            # One EM iteration
            hmm.fit(X_train, lengths_train)

            # From the second epoch onward, do not reinitialize parameters
            hmm.init_params = ""

            # Skip degenerate models before calling score()
            if not _is_valid_hmm(hmm):
                history["failed"] = True
                history["failure_reason"] = (
                    f"Invalid HMM after fit at epoch {epoch}: "
                    f"row sums={hmm.transmat_.sum(axis=1)}"
                )
                break

            # total log-likelihood / number of samples
            train_nll = -hmm.score(X_train, lengths_train) / max(X_train.shape[0], 1)
            val_nll = -hmm.score(X_val, lengths_val) / max(X_val.shape[0], 1)

        except ValueError as e:
            history["failed"] = True
            history["failure_reason"] = f"ValueError at epoch {epoch}: {e}"
            break

        history["train_nll"].append(float(train_nll))
        history["val_nll"].append(float(val_nll))

        if val_nll < best_val:
            best_val = val_nll
            best_epoch = epoch
            patience_left = t.early_stop_patience

            with open(ckpt_path, "wb") as f:
                pickle.dump(
                    {
                        "history": history,
                        "epoch": epoch,
                        "hmm": copy.deepcopy(hmm),
                        "config": config,
                        "val_nll": best_val,
                    },
                    f,
                )
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

        print(
            f"Epoch {epoch:03d} | "
            f"train NLL {train_nll:.6f} | "
            f"val NLL {val_nll:.6f} | "
            f"best @ {best_epoch:03d}"
        )

    return {
        "history": history,
        "best_val_total": best_val,
        "best_epoch": best_epoch,
        "ckpt_path": ckpt_path if best_epoch != -1 else None,
        "status": "ok" if best_epoch != -1 else "failed",
    }


def evaluate_hmm(hmm, loader):
    X, data, lengths, labels = loader_to_hmm_arrays(loader)

    scores_per_window = []
    start = 0

    # Wir gehen jedes Fenster einzeln durch
    for L in lengths:
        x_i = X[start:start + L]
        # Berechne die Log-Likelihood für dieses eine Fenster
        # Je niedriger (negativer) dieser Wert, desto "unwahrscheinlicher" ist die Sequenz
        log_prob_i = hmm.score(x_i)

        # Für die Anomalie-Funktion: NLL (negativ machen, damit hohe Werte = Anomalie)
        # Optional: Durch L teilen, um Fenster unterschiedlicher Länge vergleichbar zu machen
        nll_i = -log_prob_i / L

        scores_per_window.append(nll_i)
        start += L

    if len(lengths) == 0:
        return {"nll": float("nan"), "avg_logprob": float("nan")}, [], None

    logprob = hmm.score(X, lengths)
    nll_per_window = -logprob / len(lengths)
    avg_logprob_per_timestep = logprob / np.sum(lengths)

    # Decode (Viterbi) per-window
    states_per_window = []
    start = 0
    for L in lengths:
        x_i = X[start:start + L]   # [L, D]
        start += L
        z_i = hmm.predict(x_i)     # [L]
        states_per_window.append(z_i)

    return {
        "nll": float(nll_per_window),
        "avg_logprob": float(avg_logprob_per_timestep),
    }, data, X, states_per_window, labels, scores_per_window


def compute_hmm_metrics(config, out, dataset, save=True):
    m = parse_model_config(config)
    t = parse_train_config(config)

    np.random.seed(t.seed)

    ckpt_path = os.path.join(t.ckpt_dir, t.ckpt_name)
    train_loader, _, test_loader = make_loaders_from_out(out, config)

    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    hmm = ckpt["hmm"]

    train_metrics, train_features, _, train_states_per_window, train_labels_per_window, _ = evaluate_hmm(hmm, train_loader)
    test_metrics, test_features, _, test_states_per_window, test_labels_per_window, _ = evaluate_hmm(hmm, test_loader)

    # Your existing function
    state_metrics, per_state_entropy, per_state_purity, state_majority_label = compute_state_label_metrics(
        latent_data_list=test_states_per_window,
        valid_states=test_labels_per_window,
        num_total_states=m.num_states,
    )



    test_features = [torch.tensor(ten) for ten in test_features]
    train_states_per_window = [torch.tensor(ten) for ten in train_states_per_window]
    test_states_per_window = [torch.tensor(ten) for ten in test_states_per_window]

    class_results, forecast_results, _ = compute_downstream_metrics(train_latents=train_states_per_window,
                                                                    train_labels=train_labels_per_window,
                                                                    train_features=train_features,
                                                                    test_latents=test_states_per_window, test_labels=test_labels_per_window,
                                                                    test_features=test_features,
                                                                    classification=True, forecasting=True)


    combined_metrics = {
        "test_nll": float(test_metrics["nll"]),
        "test_avg_logprob": float(test_metrics["avg_logprob"]),
        **state_metrics,
    }
    combined_metrics.update(class_results)
    combined_metrics.update(forecast_results)

    if save:
        out_path = save_state_metrics(
            combined_metrics,
            per_state_entropy,
            per_state_purity,
            state_majority_label,
            save_dir=f"metrics/{dataset}/hmm/{str(m.num_states)}",
            filename=f"hmm_state_metrics_{t.seed}.json",
            extra={
                "run_name": "hmm_baseline",
                "split": "test",
                "checkpoint_path": ckpt_path,
                "model_config": _dc_to_dict(m),
                "train_config": _dc_to_dict(t),
                "data_meta": out.get("meta", {}),
            },
        )
        print("Saved metrics to:", out_path)
        return combined_metrics, out_path

    return combined_metrics

def compute_hmm_anomaly_score(config, normal_out, anomaly_out, dataset, save=True):
    m = parse_model_config(config)
    t = parse_train_config(config)

    torch.manual_seed(t.seed)
    np.random.seed(t.seed)

    ckpt_path = os.path.join(t.ckpt_dir, t.ckpt_name)
    _, _, normal_test_loader = make_loaders_from_out(normal_out, config)
    _, _, anomaly_test_loader = make_loaders_from_out(anomaly_out, config)

    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    hmm = ckpt["hmm"]

    # IMPORTANT: evaluate_vqvae should return:
    # test_metrics: dict with keys total/recon/vq/perplexity
    # labels: list[tensor] or tensor
    # states: list[tensor] or tensor
    _, _, _, _, _, normal_recon = evaluate_hmm(hmm, normal_test_loader)
    _, _, _, _, _, anomaly_recon = evaluate_hmm(hmm, anomaly_test_loader)

    _, _, anomaly_metrics = compute_downstream_metrics(test_ff_error=normal_recon,
                                                       test_faulty_error=anomaly_recon,
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
            save_dir=f"metrics/{dataset}/hmm/anomaly_detection/{str(m.num_states)}/",
            filename=f"hmm_state_metrics_{t.seed}.json",
            extra={
                "run_name": "hmm_baseline",
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











