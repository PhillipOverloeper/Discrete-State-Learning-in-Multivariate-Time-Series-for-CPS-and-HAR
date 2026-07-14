"""Gaussian–Bernoulli RBM baseline added after the ETFA paper experiments."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics import compute_downstream_metrics, compute_state_label_metrics, save_state_metrics
from utils import _dc_to_dict, make_loaders_from_out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class ModelConfig:
    num_hidden: int = 4
    cd_steps: int = 1


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    early_stop_patience: int = 15
    ckpt_dir: str = "./checkpoints/rbm/"
    ckpt_name: str = "rbm_best.pt"
    seed: int = 42


def parse_model_config(config):
    model = config["model"]
    return ModelConfig(
        num_hidden=int(model.get("num_hidden", 4)),
        cd_steps=int(model.get("cd_steps", 1)),
    )


def parse_train_config(config):
    train = config["train"]
    return TrainConfig(
        epochs=int(train.get("epochs", 100)),
        batch_size=int(train.get("batch_size", 64)),
        lr=float(train.get("lr", 1e-3)),
        weight_decay=float(train.get("weight_decay", 0.0)),
        early_stop_patience=int(train.get("early_stop_patience", 15)),
        ckpt_dir=str(train.get("ckpt_dir", "./checkpoints/rbm/")),
        ckpt_name=str(train.get("ckpt_name", "rbm_best.pt")),
        seed=int(config.get("seed", train.get("seed", 42))),
    )


class GaussianBernoulliRBM(nn.Module):
    """RBM with Gaussian visible units and Bernoulli hidden units."""

    def __init__(self, input_dim: int, num_hidden: int):
        super().__init__()
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.weight = nn.Parameter(torch.empty(input_dim, num_hidden))
        self.visible_bias = nn.Parameter(torch.zeros(input_dim))
        self.hidden_bias = nn.Parameter(torch.zeros(num_hidden))
        nn.init.normal_(self.weight, mean=0.0, std=0.01)

    def hidden_probabilities(self, visible: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(visible @ self.weight + self.hidden_bias)

    def sample_hidden(self, visible: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probabilities = self.hidden_probabilities(visible)
        return probabilities, torch.bernoulli(probabilities)

    def visible_mean(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden @ self.weight.T + self.visible_bias

    def free_energy(self, visible: torch.Tensor) -> torch.Tensor:
        visible_term = 0.5 * torch.sum((visible - self.visible_bias) ** 2, dim=-1)
        hidden_term = torch.sum(F.softplus(visible @ self.weight + self.hidden_bias), dim=-1)
        return visible_term - hidden_term

    def contrastive_divergence_loss(self, visible: torch.Tensor, steps: int) -> torch.Tensor:
        negative_visible = visible.detach()
        for _ in range(max(1, steps)):
            _, hidden_sample = self.sample_hidden(negative_visible)
            negative_visible = self.visible_mean(hidden_sample).detach()
        return self.free_energy(visible).mean() - self.free_energy(negative_visible).mean()

    def reconstruct(self, visible: torch.Tensor) -> torch.Tensor:
        return self.visible_mean(self.hidden_probabilities(visible))

    def states(self, visible: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.hidden_probabilities(visible), dim=-1)


@torch.no_grad()
def _validation_reconstruction_loss(model, loader) -> float:
    model.eval()
    losses = []
    for features, _ in loader:
        flat = features.to(device).reshape(-1, features.shape[-1])
        losses.append(F.mse_loss(model.reconstruct(flat), flat).item())
    return float(np.mean(losses)) if losses else float("inf")


def train_rbm(out, config):
    model_config = parse_model_config(config)
    train_config = parse_train_config(config)
    torch.manual_seed(train_config.seed)
    np.random.seed(train_config.seed)

    model = GaussianBernoulliRBM(out["meta"]["D"], model_config.num_hidden).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )
    train_loader, val_loader, _ = make_loaders_from_out(out, config)
    os.makedirs(train_config.ckpt_dir, exist_ok=True)
    checkpoint_path = os.path.join(train_config.ckpt_dir, train_config.ckpt_name)

    best_val = float("inf")
    best_epoch = -1
    patience_left = train_config.early_stop_patience
    history = {"train_cd": [], "val_reconstruction": [], "best_ckpt_path": checkpoint_path}

    for epoch in range(1, train_config.epochs + 1):
        model.train()
        batch_losses = []
        for features, _ in train_loader:
            flat = features.to(device).reshape(-1, features.shape[-1])
            optimizer.zero_grad(set_to_none=True)
            loss = model.contrastive_divergence_loss(flat, model_config.cd_steps)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        train_loss = float(np.mean(batch_losses))
        val_loss = _validation_reconstruction_loss(model, val_loader)
        history["train_cd"].append(train_loss)
        history["val_reconstruction"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            patience_left = train_config.early_stop_patience
            torch.save(
                {
                    "model_state_dict": copy.deepcopy(model.state_dict()),
                    "config": config,
                    "history": history,
                    "epoch": epoch,
                    "val_reconstruction": best_val,
                },
                checkpoint_path,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

        print(
            f"Epoch {epoch:03d} | train CD {train_loss:.6f} | "
            f"val reconstruction {val_loss:.6f} | best @ {best_epoch:03d}"
        )

    return {
        "history": history,
        "best_val_total": best_val,
        "best_epoch": best_epoch,
        "ckpt_path": checkpoint_path if best_epoch >= 0 else None,
        "device": str(device),
    }


def _load_model(config, out):
    model_config = parse_model_config(config)
    train_config = parse_train_config(config)
    checkpoint_path = os.path.join(train_config.ckpt_dir, train_config.ckpt_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = GaussianBernoulliRBM(out["meta"]["D"], model_config.num_hidden).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint, model_config, train_config


@torch.no_grad()
def _evaluate(model, loader):
    features, labels, states = [], [], []
    for batch_features, batch_labels in loader:
        batch_features = batch_features.to(device)
        flat = batch_features.reshape(-1, batch_features.shape[-1])
        batch_states = model.states(flat).reshape(batch_features.shape[:-1])
        features.append(batch_features.cpu())
        labels.append(batch_labels.cpu())
        states.append(batch_states.cpu())
    return features, labels, states


def compute_rbm_metrics(config, out, dataset, save=True):
    model, checkpoint, model_config, train_config = _load_model(config, out)
    train_loader, _, test_loader = make_loaders_from_out(out, config)
    train_features, train_labels, train_states = _evaluate(model, train_loader)
    test_features, test_labels, test_states = _evaluate(model, test_loader)

    class_results, forecast_results, _ = compute_downstream_metrics(
        train_latents=train_states,
        train_labels=train_labels,
        train_features=train_features,
        test_latents=test_states,
        test_labels=test_labels,
        test_features=test_features,
        classification=True,
        forecasting=True,
    )
    state_metrics, per_state_entropy, per_state_purity, state_majority_label = compute_state_label_metrics(
        latent_data_list=test_states,
        valid_states=test_labels,
        num_total_states=model_config.num_hidden,
    )
    combined_metrics = {**state_metrics, **class_results, **forecast_results}
    if not save:
        return combined_metrics

    output_path = save_state_metrics(
        combined_metrics,
        per_state_entropy,
        per_state_purity,
        state_majority_label,
        save_dir=f"metrics/{dataset}/rbm/{model_config.num_hidden}",
        filename=f"rbm_state_metrics_{train_config.seed}.json",
        extra={
            "run_name": "rbm_post_paper_extension",
            "paper_model": False,
            "model_config": _dc_to_dict(model_config),
            "train_config": _dc_to_dict(train_config),
            "data_meta": out.get("meta", {}),
            "train_history": checkpoint.get("history", {}),
        },
    )
    print(f"Saved RBM clustering metrics for {dataset}")
    return combined_metrics, output_path


@torch.no_grad()
def compute_rbm_anomaly_score(config, normal_out, anomaly_out, dataset, save=True):
    model, checkpoint, model_config, train_config = _load_model(config, normal_out)

    def reconstruction_errors(out):
        errors = []
        _, _, loader = make_loaders_from_out(out, config)
        for features, _ in loader:
            flat = features.to(device).reshape(-1, features.shape[-1])
            error = torch.mean((model.reconstruct(flat) - flat) ** 2, dim=-1)
            errors.append(error.cpu().numpy())
        return np.concatenate(errors)

    normal_errors = reconstruction_errors(normal_out)
    anomaly_errors = reconstruction_errors(anomaly_out)
    _, _, anomaly_metrics = compute_downstream_metrics(
        test_ff_error=normal_errors,
        test_faulty_error=anomaly_errors,
        anomaly_detection=True,
    )
    if not save:
        return anomaly_metrics

    output_path = save_state_metrics(
        anomaly_metrics,
        {},
        {},
        {},
        save_dir=f"metrics/{dataset}/rbm/anomaly_detection/{model_config.num_hidden}",
        filename=f"rbm_state_metrics_{train_config.seed}.json",
        extra={
            "run_name": "rbm_post_paper_extension",
            "paper_model": False,
            "model_config": _dc_to_dict(model_config),
            "train_config": _dc_to_dict(train_config),
            "train_history": checkpoint.get("history", {}),
        },
    )
    return anomaly_metrics, output_path
