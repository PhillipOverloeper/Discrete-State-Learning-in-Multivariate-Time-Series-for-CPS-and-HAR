from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, mean_squared_error, roc_auc_score


class DownstreamEvaluator:
    def __init__(self, eps=1e-12):
        self.eps = eps

    def evaluate_forecasting(self, train_latents, train_features, test_latents, test_features):
        """
        Evaluates how well a state s_t predicts the next feature x_{t+1}.
        Fits the mapping on training data and tests on unseen test data.
        """
        # 1. LEARNING PHASE: Map State ID -> Average Feature Vector (using TRAIN)
        state_map = defaultdict(list)
        for lat, feat in zip(train_latents, train_features):
            # Flatten window if necessary (e.g., if shape is [67, 128, dim])
            l_flat = lat.detach().cpu().numpy().ravel() if isinstance(lat, torch.Tensor) else np.asarray(lat).ravel()
            f_np = feat.detach().cpu().numpy() if isinstance(feat, torch.Tensor) else np.asarray(feat)
            f_flat = f_np.reshape(-1, f_np.shape[-1])

            for s, x in zip(l_flat, f_flat):
                state_map[int(s)].append(x)

        # Compute the "Centroid" for each state
        state_means = {s: np.mean(vals, axis=0) for s, vals in state_map.items()}
        global_mean = np.mean(
            np.concatenate([
                f.detach().cpu().numpy() if isinstance(f, torch.Tensor) else np.asarray(f)
                for f in train_features
            ]), axis=(0,1) # This will now work on a 3D volume (Batch, Time, Feature)
        )

        # 2. EVALUATION PHASE: Predict x_{t+1} (using TEST)
        preds = []

        targets = []

        for lat, feat in zip(test_latents, test_features):

            l_flat = lat.detach().cpu().numpy().ravel() if isinstance(lat, torch.Tensor) else np.asarray(lat).ravel()
            f_np = feat.detach().cpu().numpy() if isinstance(feat, torch.Tensor) else np.asarray(feat)
            f_flat = f_np.reshape(-1, f_np.shape[-1])

            for i in range(len(l_flat) - 1):

                s_t = int(l_flat[i])
                x_next = f_flat[i + 1]
                # Look up the mean for state s_t; fallback to global mean if state is new
                pred = state_means.get(s_t, global_mean)
                preds.append(pred)
                targets.append(x_next)

        # 3. METRIC CALCULATION
        mse = mean_squared_error(targets, preds)

        # Calculate Variance Explained (R^2 equivalent for multi-dim)
        # 1 - (Unexplained Var / Total Var)
        total_var = np.var(targets)
        r2_simulated = 1 - (mse / (total_var + self.eps))

        return {
            "forecasting_mse": float(mse),
            "forecasting_r2_score": float(r2_simulated),
            "unseen_states_in_test": len(
                set(np.concatenate([
                    t.detach().cpu().numpy().ravel() if isinstance(t, torch.Tensor) else np.asarray(t).ravel()
                    for t in test_latents
                ])) - set(state_means.keys())
            )
        }

    def evaluate_classification(self, train_latents, train_labels, test_latents, test_labels):
        """
        Linear Probe: How much class information is linearly accessible from state IDs?
        We use a 1-hot encoding of states as features.
        """

        def to_one_hot(latents, num_states):
            flat = np.concatenate([
                x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
                for x in latents
            ]).ravel()
            one_hot = np.zeros((flat.size, num_states))
            one_hot[np.arange(flat.size), flat.astype(int)] = 1
            return one_hot

        all_states = np.concatenate([
            x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
            for x in (train_latents + test_latents)
        ])
        num_states = int(all_states.max() + 1)

        X_train = to_one_hot(train_latents, num_states)
        y_train = np.concatenate([
            y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
            for y in train_labels
        ]).ravel()
        X_test = to_one_hot(test_latents, num_states)
        y_test = np.concatenate([
            y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
            for y in test_labels
        ]).ravel()

        clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        # Normalize it to see percentages
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        return {
            "classification_f1_macro": float(f1_score(y_test, y_pred, average='macro')),
            "classification_accuracy": float(clf.score(X_test, y_test)),
            "confusion_matrix": cm_normalized.tolist()
        }

    def evaluate_anomaly_detection(self, test_ff_error, test_faulty_error):
        """
        Universelle Methode zur Evaluierung von Anomaliedetektion.

        Args:
            test_ff_scores (list or np.array): Liste von Scores (z.B. MSE oder NLL)
                                               für fehlerfreie Testdaten.
            test_faulty_scores (list or np.array): Liste von Scores für fehlerbehaftete Testdaten.
        """
        # Konvertierung in Arrays für schnellere Berechnung
        ff_scores = to_numpy(test_ff_error)
        faulty_scores = to_numpy(test_faulty_error)

        # 1. Labels erstellen: 0 für Fault-Free (FF), 1 for Faulty
        y_true = np.concatenate([
            np.zeros(len(ff_scores)),
            np.ones(len(faulty_scores))
        ])

        # 2. Scores zusammenführen
        y_scores = np.concatenate([ff_scores, faulty_scores])

        # 3. Metriken berechnen
        # AUC-ROC: Misst die Trennbarkeit der beiden Verteilungen
        try:
            auc = roc_auc_score(y_true, y_scores)
        except ValueError:
            # Falls nur eine Klasse vorhanden ist oder Scores ungültig sind
            auc = 0.5

        mean_ff = np.mean(ff_scores)
        mean_faulty = np.mean(faulty_scores)

        # Ratio: Wie viel "lauter" ist der Fehler im Vergleich zum Normalzustand?
        # Vermeidung von Division durch Null durch ein kleines Epsilon
        ratio = mean_faulty / (mean_ff + 1e-9)

        return {
            "anomaly_auc_roc": float(auc),
            "mean_ff_score": float(mean_ff),
            "mean_faulty_score": float(mean_faulty),
            "surprise_ratio": float(ratio)
        }

def to_numpy(data):
    # Handles both a list of tensors and a single long list/tensor
    if isinstance(data, list):
        # as_tensor is more flexible than stack for mixed list inputs
        return torch.as_tensor(data).cpu().numpy()
    elif isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.array(data)


def compute_downstream_metrics(train_latents=None, train_labels=None, train_features=None,
                               test_latents=None, test_labels=None, test_features=None,
                               test_ff_error=None, test_faulty_error=None,
                               classification=False, anomaly_detection=False,
                               forecasting=False):

    evaluator = DownstreamEvaluator()
    class_results, forecast_results, anomaly_results = None, None, None

    if classification:
        class_results = evaluator.evaluate_classification(train_latents, train_labels, test_latents, test_labels)

    if forecasting:
        forecast_results = evaluator.evaluate_forecasting(train_latents, train_features, test_latents, test_features)

    if anomaly_detection:
        anomaly_results = evaluator.evaluate_anomaly_detection(test_ff_error, test_faulty_error)

    return class_results, forecast_results, anomaly_results

def compute_state_label_metrics(latent_data_list, valid_states, num_total_states=None, eps=1e-12):
    """
    Compute metrics based on the joint distribution of discrete states S and labels Y.

    Args:
        latent_data_list: list of 1D tensors of state IDs
        valid_states:     list of 1D tensors of ground-truth labels (same shapes)
        num_total_states: (optional) total possible states (e.g., codebook size K).
                         If provided, utilization = states_used / num_total_states.
        eps: small constant for numerical stability in logs/divisions

    Returns:
        metrics: dict with keys including:
            - avg_label_entropy: H(Y | S)
            - avg_state_entropy: H(S | Y)
            - mutual_information: I(Y; S)
            - entropy_labels: H(Y)
            - entropy_states: H(S)
            - purity
            - nmi_sqrt, nmi_min
            - homogeneity, completeness, v_measure
            - ari
            - num_states_used, state_utilization, top1_mass/top5_mass/top10_mass
            - effective_num_states
        per_state_entropy: dict[state_id -> H(Y | S = state_id)]
        per_state_purity:  dict[state_id -> max_y p(y|s)]
        state_majority_label: dict[state_id -> y*]
    """
    # counts for S, Y, and (S,Y)
    state_counts = Counter()
    label_counts = Counter()
    joint_counts = defaultdict(Counter)  # joint_counts[s][y]

    for latent, labels in zip(latent_data_list, valid_states):
        # Check latent
        if torch.is_tensor(latent):
            latent_np = latent.detach().cpu().numpy().ravel()
        else:
            latent_np = np.asarray(latent).ravel()

        # Check labels
        if torch.is_tensor(labels):
            labels_np = labels.detach().cpu().numpy().ravel()
        else:
            labels_np = np.asarray(labels).ravel()

        if latent_np.shape[0] != labels_np.shape[0]:
            raise ValueError(f"latent and labels length mismatch: {latent_np.shape[0]} vs {labels_np.shape[0]}")
        for s, y in zip(latent_np, labels_np):
            s = int(s)
            y = int(y)
            state_counts[s] += 1
            label_counts[y] += 1
            joint_counts[s][y] += 1

    total = sum(state_counts.values())
    if total == 0:
        metrics = {
            "avg_label_entropy": np.nan,
            "avg_state_entropy": np.nan,
            "mutual_information": np.nan,
            "entropy_labels": np.nan,
            "entropy_states": np.nan,
            "purity": np.nan,
            "nmi_sqrt": np.nan,
            "nmi_min": np.nan,
            "homogeneity": np.nan,
            "completeness": np.nan,
            "v_measure": np.nan,
            "ari": np.nan,
            "num_states_used": 0,
            "state_utilization": np.nan,
            "top1_mass": np.nan,
            "top5_mass": np.nan,
            "top10_mass": np.nan,
            "effective_num_states": np.nan,
        }
        return metrics, {}, {}, {}

    # --- Entropy of labels H(Y) ---
    H_Y = 0.0
    for y, c_y in label_counts.items():
        p_y = c_y / total
        H_Y -= p_y * np.log(p_y + eps)

    # --- Entropy of states H(S) ---
    H_S = 0.0
    for s, c_s in state_counts.items():
        p_s = c_s / total
        H_S -= p_s * np.log(p_s + eps)

    # --- Mutual information I(Y; S) ---
    mutual_information = 0.0
    for s, c_s in state_counts.items():
        p_s = c_s / total
        for y, c_sy in joint_counts[s].items():
            p_sy = c_sy / total
            p_y = label_counts[y] / total
            mutual_information += p_sy * np.log((p_sy + eps) / ((p_s + eps) * (p_y + eps)))

    # --- Average conditional entropy H(Y | S) + per-state ---
    avg_H_Y_given_S = 0.0
    per_state_entropy = {}
    per_state_purity = {}
    state_majority_label = {}

    for s, c_s in state_counts.items():
        # H(Y|S=s)
        H = 0.0
        # purity stats
        max_c = 0
        argmax_y = None

        for y, c_sy in joint_counts[s].items():
            p_y_given_s = c_sy / c_s
            H -= p_y_given_s * np.log(p_y_given_s + eps)
            if c_sy > max_c:
                max_c = c_sy
                argmax_y = y

        per_state_entropy[s] = float(H)
        per_state_purity[s] = float(max_c / c_s) if c_s > 0 else 0.0
        state_majority_label[s] = int(argmax_y) if argmax_y is not None else -1

        p_s = c_s / total
        avg_H_Y_given_S += p_s * H

    # --- Average conditional entropy H(S | Y) ---
    # H(S|Y) = sum_y p(y) H(S|Y=y)
    avg_H_S_given_Y = 0.0
    for y, c_y in label_counts.items():
        # distribution over states given this y:
        H = 0.0
        # compute counts of states for this label
        # we can iterate over states that have this y in joint_counts
        for s, c_s in state_counts.items():
            c_sy = joint_counts[s].get(y, 0)
            if c_sy == 0:
                continue
            p_s_given_y = c_sy / c_y
            H -= p_s_given_y * np.log(p_s_given_y + eps)
        avg_H_S_given_Y += (c_y / total) * H

    # --- Purity (global) ---
    # Purity = sum_s max_y count(s,y) / N
    purity = sum(max(joint_counts[s].values()) for s in state_counts.keys()) / total

    # --- NMI variants ---
    nmi_sqrt = mutual_information / (np.sqrt(max(H_Y * H_S, eps)))
    nmi_min = mutual_information / max(min(H_Y, H_S), eps)

    # --- Homogeneity / Completeness / V-measure ---
    homogeneity = mutual_information / max(H_Y, eps)
    completeness = mutual_information / max(H_S, eps)
    if homogeneity + completeness == 0:
        v_measure = 0.0
    else:
        v_measure = 2.0 * homogeneity * completeness / (homogeneity + completeness)

    # --- ARI (Adjusted Rand Index) ---
    # Build contingency matrix from counters (without sklearn)
    # Map labels/states to indices
    y_vals = list(label_counts.keys())
    s_vals = list(state_counts.keys())
    y_index = {y: i for i, y in enumerate(y_vals)}
    s_index = {s: j for j, s in enumerate(s_vals)}

    C = np.zeros((len(y_vals), len(s_vals)), dtype=np.int64)
    for s, ys in joint_counts.items():
        j = s_index[s]
        for y, c_sy in ys.items():
            i = y_index[y]
            C[i, j] = c_sy

    def comb2(n: int) -> float:
        return n * (n - 1) / 2.0

    N = total
    sum_comb_c = float(sum(comb2(int(nij)) for nij in C.ravel()))
    sum_comb_rows = float(sum(comb2(int(ni)) for ni in C.sum(axis=1)))
    sum_comb_cols = float(sum(comb2(int(nj)) for nj in C.sum(axis=0)))
    comb_N = comb2(N)

    expected = (sum_comb_rows * sum_comb_cols) / max(comb_N, eps)
    max_index = 0.5 * (sum_comb_rows + sum_comb_cols)
    denom = (max_index - expected)
    ari = 0.0 if denom == 0 else (sum_comb_c - expected) / denom

    # --- State usage diagnostics ---
    counts_arr = np.array(list(state_counts.values()), dtype=np.int64)
    sorted_counts = np.sort(counts_arr)[::-1]
    top1_mass = float(sorted_counts[:1].sum() / N)
    top5_mass = float(sorted_counts[:5].sum() / N) if sorted_counts.size >= 5 else float(sorted_counts.sum() / N)
    top10_mass = float(sorted_counts[:10].sum() / N) if sorted_counts.size >= 10 else float(sorted_counts.sum() / N)

    num_states_used = int(len(state_counts))
    if num_total_states is None:
        state_utilization = np.nan
    else:
        state_utilization = float(num_states_used / max(int(num_total_states), 1))

    effective_num_states = float(np.exp(H_S))  # "effective" number of states (in nats)

    metrics = {
        "avg_label_entropy": float(avg_H_Y_given_S),
        "avg_state_entropy": float(avg_H_S_given_Y),
        "mutual_information": float(mutual_information),
        "entropy_labels": float(H_Y),
        "entropy_states": float(H_S),

        "purity": float(purity),
        "nmi_sqrt": float(nmi_sqrt),
        "nmi_min": float(nmi_min),
        "homogeneity": float(homogeneity),
        "completeness": float(completeness),
        "v_measure": float(v_measure),
        "ari": float(ari),

        "num_states_used": num_states_used,
        "state_utilization": state_utilization,
        "top1_mass": top1_mass,
        "top5_mass": top5_mass,
        "top10_mass": top10_mass,
        "effective_num_states": effective_num_states,
    }

    return metrics, per_state_entropy, per_state_purity, state_majority_label


def _to_jsonable(obj):
    """
    Convert common non-JSON-serializable objects (numpy types, etc.) into JSON-safe Python types.
    """
    # numpy scalars
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
    except Exception:
        pass

    # torch scalars
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            if obj.numel() == 1:
                return obj.item()
            return obj.detach().cpu().tolist()
    except Exception:
        pass

    # fallback
    return obj


def save_state_metrics(
    metrics: dict,
    per_state_entropy: dict,
    per_state_purity: dict,
    state_majority_label: dict,
    save_dir: str = "metrics",
    filename: str = None,
    extra: dict = None,
):
    """
    Saves metrics to a JSON file.

    Args:
        metrics: global metrics dict
        per_state_entropy: dict[state -> float]
        per_state_purity: dict[state -> float]
        state_majority_label: dict[state -> int]
        save_dir: directory to write JSON file into
        filename: optional fixed filename; if None, creates timestamped filename
        extra: optional dict of additional metadata to store (e.g., config, ckpt path)
    """
    os.makedirs(save_dir, exist_ok=True)

    if filename is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"state_metrics_{ts}.json"

    payload = {
        "metrics": {k: _to_jsonable(v) for k, v in metrics.items()},
        "per_state_entropy": {str(k): _to_jsonable(v) for k, v in per_state_entropy.items()},
        "per_state_purity": {str(k): _to_jsonable(v) for k, v in per_state_purity.items()},
        "state_majority_label": {str(k): _to_jsonable(v) for k, v in state_majority_label.items()},
    }

    if extra is not None:
        payload["extra"] = {k: _to_jsonable(v) for k, v in extra.items()}

    out_path = os.path.join(save_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    return out_path


def load_metric_summary(
    model_name: str,
    dataset: str,
    num_states: str,
    metrics_root: str | Path = "metrics",
) -> dict[str, dict[str, float | int]]:
    """
    Load all metric files for one (model_name, dataset, num_states) setting,
    then compute mean and std across seeds for each metric.

    Expected file pattern:
        metrics/<dataset>/<model_name>/<num_states>/vqvae_state_metrics_<seed>.json

    Returns:
        {
            "ari": {"mean": ..., "std": ..., "n": ...},
            "purity": {"mean": ..., "std": ..., "n": ...},
            ...
        }
    """
    metrics_dir = Path(metrics_root) / dataset / model_name / num_states
    files = sorted(metrics_dir.glob(f"{model_name}_state_metrics_*.json"))

    if not files:
        raise FileNotFoundError(f"No metric files found in {metrics_dir}")

    collected: dict[str, list[float]] = {}

    for file in files:
        with file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        metrics = data.get("metrics")
        if not isinstance(metrics, dict):
            continue

        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                collected.setdefault(metric_name, []).append(float(value))

    summary: dict[str, dict[str, float | int]] = {}
    for metric_name, values in collected.items():
        if not values:
            continue

        summary[metric_name] = {
            "mean": mean(values),
            "std": stdev(values) if len(values) > 1 else 0.0,
            "n": len(values),
        }

    out_path = metrics_dir / "metrics_summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
