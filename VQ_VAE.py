import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics import compute_downstream_metrics, compute_state_label_metrics, save_state_metrics
from utils import MLPDecoder, MLPEncoder, _dc_to_dict, make_loaders_from_out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class ModelConfig:
    latent_dim: int = 64
    num_embeddings: int = 128
    embedding_dim: int = 128
    beta: float = 0.25
    enc_hidden: tuple = (256, 256)
    dec_hidden: tuple = (256, 256)
    dropout: float = 0.0


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 1
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip_norm: int = 1

    recon_loss_type: str = "mse"
    recon_loss_weight: float = 1.0
    vq_loss_weight: float = 1.0

    early_stop_patience: int = 15

    ckpt_dir: str = "./checkpoints/vq-vae/"
    ckpt_name: str = "vq-vae_best.pt"

    seed: int = 42


def parse_model_config(config):
    m = config["model"]
    return ModelConfig(
        latent_dim = int(m.get("latent_dim", 64)),
        num_embeddings = int(m.get("num_embeddings", 512)),
        embedding_dim = int(m.get("embedding_dim", 64)),
        beta = float(m.get("beta", 0.25)),
        enc_hidden = tuple(m.get("enc_hidden", (256, 256))),
        dec_hidden = tuple(m.get("dec_hidden", (256, 256))),
        dropout = float(m.get("dropout", 0.0)),
    )


def parse_train_config(config):
    t = config["train"]
    seed = int(config.get("seed", 42))
    return TrainConfig(
        epochs = int(t.get("epochs", 50)),
        batch_size = int(t.get("batch_size", 256)),
        lr = float(t.get("lr", 2e-4)),
        weight_decay = float(t.get("weight_decay", 0.0)),
        grad_clip_norm = t.get("grad_clip_norm", 1.0),

        recon_loss_type = str(t.get("recon_loss_type", "mse")),
        recon_loss_weight = float(t.get("recon_weight", 1.0)),
        vq_loss_weight = float(t.get("vq_weight", 1.0)),

        early_stop_patience = int(t.get("early_stop_patience", 15)),

        ckpt_dir=str(t.get("ckpt_dir", "checkpoints")),
        ckpt_name=str(t.get("ckpt_name", "vqvae_best.pt")),

        seed=seed,
    )



class VectorQuantizer(nn.Module):
    """
    Standard VQ-VAE quantizer (Oord et al. 2016).
    """
    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z_e):
        # z_e shape: [B, T, D] (Batch, Time, Dim)
        B, T, D = z_e.shape

        # Flatten to [B*T, D] for distance calculation
        z_flattened = z_e.reshape(-1, D)

        # Compute distances (L2: a^2 + b^2 - 2ab)
        # Using self.embedding.weight which is [num_embeddings, D]
        z_sq = torch.sum(z_flattened ** 2, dim=1, keepdim=True)
        e_sq = torch.sum(self.embedding.weight ** 2, dim=1).unsqueeze(0)
        ze = z_flattened @ self.embedding.weight.t()
        distances = z_sq + e_sq - 2 * ze  # [B*T, num_embeddings]

        # Quantize
        indices = torch.argmin(distances, dim=1)  # [B*T]
        z_q = self.embedding(indices)  # [B*T, D]

        # Losses
        codebook_loss = F.mse_loss(z_q, z_flattened.detach())
        commitment_loss = F.mse_loss(z_q.detach(), z_flattened)
        vq_loss = codebook_loss + self.beta * commitment_loss

        # Straight-through estimator
        # This allows gradients to bypass the non-differentiable argmin
        z_q_st = z_flattened + (z_q - z_flattened).detach()

        # Reshape back to original dimensions [B, T, D]
        z_q_st = z_q_st.view(B, T, D)
        indices = indices.view(B, T)  # Useful to see the "state sequence"

        # Perplexity Fix: Calculate across the entire batch (B*T)
        encodings = F.one_hot(indices.view(-1), self.num_embeddings).type(z_e.dtype)
        avg_probs = torch.mean(encodings, dim=0)  # Average over all samples in batch
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return z_q_st, vq_loss, perplexity, encodings, indices


class VQVAE(nn.Module):
    def __init__(self, windows_length, input_dim, latent_dim, num_embeddings, beta=0.25,
                 enc_hidden=(256, 256), dec_hidden=(256, 256), dropout=0.0):
        super().__init__()
        # Initialise parameters
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.beta = beta
        self.enc_hidden = enc_hidden
        self.dec_hidden = dec_hidden
        self.dropout = dropout

        # Initialise MLPs
        self.encoder = MLPEncoder(self.input_dim, self.latent_dim, self.enc_hidden, self.dropout)
        self.decoder = MLPDecoder(self.input_dim, self.latent_dim, self.dec_hidden, self.dropout)
        self.vq = VectorQuantizer(self.num_embeddings, self.latent_dim, self.beta)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, perplexity, encodings, indices = self.vq.forward(z_e)
        x_hat = self.decoder(z_q)
        return x_hat, vq_loss, perplexity, indices


def train_vqvae(out, config):

    # Load config
    m = parse_model_config(config)
    t = parse_train_config(config)

    # Set seeds for reproducibility
    torch.manual_seed(t.seed)
    np.random.seed(t.seed)

    # Set up model
    model = VQVAE(
        windows_length=out["meta"]["L"],
        input_dim=out["meta"]["D"],
        latent_dim=m.latent_dim,
        num_embeddings=m.num_embeddings,
        beta=m.beta,
        enc_hidden=m.enc_hidden,
        dec_hidden=m.dec_hidden,
        dropout=m.dropout,
    ).to(device)

    os.makedirs(t.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(t.ckpt_dir, t.ckpt_name)

    train_loader, val_loader, _ = make_loaders_from_out(out, config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=t.lr, weight_decay=t.weight_decay)

    best_val = float("inf")
    best_epoch = -1
    patience_left = t.early_stop_patience

    history = {
        "train_total": [],
        "train_recon": [],
        "train_vq": [],
        "train_perplexity": [],
        "val_total": [],
        "val_recon": [],
        "val_vq": [],
        "val_perplexity": [],
        "best_ckpt_path": ckpt_path,
    }

    for epoch in range(1, t.epochs + 1):
        model.train()

        train_total_sum = 0.0
        train_recon_sum = 0.0
        train_vq_sum = 0.0
        train_perplex_sum = 0.0
        n_batches = 0



        for x in train_loader:
            x = x[0].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            x_hat, vq_loss, perplexity, _indices = model(x)
            recon_loss = F.mse_loss(x_hat, x)
            total = t.recon_loss_weight * recon_loss + t.vq_loss_weight * vq_loss
            total.backward()

            #if t.grad_clip_norm is not None:
            #    nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip_norm)

            optimizer.step()

            train_total_sum += float(total.item())
            train_recon_sum += float(recon_loss.item())
            train_vq_sum += float(vq_loss.item())
            train_perplex_sum += float(perplexity.item())
            n_batches += 1

        train_metrics = {
            "total": train_total_sum / max(n_batches, 1),
            "recon": train_recon_sum / max(n_batches, 1),
            "vq": train_vq_sum / max(n_batches, 1),
            "perplexity": train_perplex_sum / max(n_batches, 1),
        }

        history["train_total"].append(train_metrics["total"])
        history["train_recon"].append(train_metrics["recon"])
        history["train_vq"].append(train_metrics["vq"])
        history["train_perplexity"].append(train_metrics["perplexity"])

        val_metrics, _, _, _, _ = evaluate_vqvae(model, val_loader, t)

        history["val_total"].append(val_metrics["total"])
        history["val_recon"].append(val_metrics["recon"])
        history["val_vq"].append(val_metrics["vq"])
        history["val_perplexity"].append(val_metrics["perplexity"])

        # Checkpointing + early stopping
        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            best_epoch = epoch
            patience_left = t.early_stop_patience

            torch.save(
                {
                    "history": history,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "val_total": best_val,
                },
                ckpt_path,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

        # Optional: print progress (safe default for scripts)
        print(
            f"Epoch {epoch:03d} | "
            f"train total {train_metrics['total']:.6f} (recon {train_metrics['recon']:.6f}, vq {train_metrics['vq']:.6f}, perp {train_metrics['perplexity']:.2f}) | "
            f"val total {val_metrics['total']:.6f} (recon {val_metrics['recon']:.6f}, vq {val_metrics['vq']:.6f}, perp {val_metrics['perplexity']:.2f}) | "
            f"best @ {best_epoch:03d}"
        )


    return {
        "history": history,
        "best_val_total": best_val,
        "best_epoch": best_epoch,
        "ckpt_path": ckpt_path,
        "device": str(device),
    }



@torch.no_grad()
def evaluate_vqvae(model, loader, t):
    model.eval()
    total_loss_sum = 0.0
    recon_sum = 0.0
    vq_sum = 0.0
    perplex_sum = 0.0
    n_batches = 0
    _indices = 0
    states = []
    indices = []
    data = []
    recon_losses = []

    for i, batch in enumerate(loader):

        x, y = batch

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x_hat, vq_loss, perplexity, _indices_out = model(x)

        indices.append(_indices_out)
        states.append(y)
        data.append(x)

        recon_loss = F.mse_loss(x_hat, x)
        recon_losses.append(recon_loss)
        total = t.recon_loss_weight * recon_loss + t.vq_loss_weight * vq_loss

        total_loss_sum += float(total.item())
        recon_sum += float(recon_loss.item())
        vq_sum += float(vq_loss.item())
        perplex_sum += float(perplexity.item())
        n_batches += 1

    if n_batches == 0:
        return {"total": float("nan"), "recon": float("nan"), "vq": float("nan"), "perplexity": float("nan")}, [], []

    return {
        "total": total_loss_sum / n_batches,
        "recon": recon_sum / n_batches,
        "vq": vq_sum / n_batches,
        "perplexity": perplex_sum / n_batches,
    }, data, indices, states, recon_losses

def compute_vqvae_anomaly_score(config, normal_out, anomaly_out, dataset, save=True):
    m = parse_model_config(config)
    t = parse_train_config(config)

    torch.manual_seed(t.seed)
    np.random.seed(t.seed)

    # Set up model
    model = VQVAE(
        windows_length=normal_out["meta"]["L"],
        input_dim=anomaly_out["meta"]["D"],
        latent_dim=m.latent_dim,
        num_embeddings=m.num_embeddings,
        beta=m.beta,
        enc_hidden=m.enc_hidden,
        dec_hidden=m.dec_hidden,
        dropout=m.dropout,
    ).to(device)

    ckpt_path = os.path.join(t.ckpt_dir, t.ckpt_name)
    normal_train_loader, _, normal_test_loader = make_loaders_from_out(normal_out, config)
    _, _, anomaly_test_loader = make_loaders_from_out(anomaly_out, config)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # IMPORTANT: evaluate_vqvae should return:
    # test_metrics: dict with keys total/recon/vq/perplexity
    # labels: list[tensor] or tensor
    # states: list[tensor] or tensor
    _, _, _, _, normal_recon = evaluate_vqvae(model, normal_test_loader, t)
    _, _, _, _, anomaly_recon = evaluate_vqvae(model, anomaly_test_loader, t)

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
            save_dir=f"metrics/{dataset}/vqvae/anomaly_detection/{str(m.num_embeddings)}/",
            filename=f"vqvae_state_metrics_{t.seed}.json",
            extra={
                "run_name": "vqvae_baseline",
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

def compute_vqvae_metrics(config, out, dataset, save=True):
    m = parse_model_config(config)
    t = parse_train_config(config)

    torch.manual_seed(t.seed)
    np.random.seed(t.seed)

    # Set up model
    model = VQVAE(
        windows_length=out["meta"]["L"],
        input_dim=out["meta"]["D"],
        latent_dim=m.latent_dim,
        num_embeddings=m.num_embeddings,
        beta=m.beta,
        enc_hidden=m.enc_hidden,
        dec_hidden=m.dec_hidden,
        dropout=m.dropout,
    ).to(device)

    ckpt_path = os.path.join(t.ckpt_dir, t.ckpt_name)
    train_loader, _, test_loader = make_loaders_from_out(out, config)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # IMPORTANT: evaluate_vqvae should return:
    # test_metrics: dict with keys total/recon/vq/perplexity
    # labels: list[tensor] or tensor
    # states: list[tensor] or tensor
    train_metrics, train_features, train_labels, train_states, _ = evaluate_vqvae(model, train_loader, t)
    test_metrics, test_features, test_labels, test_states, _ = evaluate_vqvae(model, test_loader, t)

    class_results, forecast_results, _ = compute_downstream_metrics(train_latents=train_labels, train_labels=train_states, train_features=train_features,
                               test_latents=test_labels, test_labels=test_states, test_features=test_features,
                               classification=True, forecasting=True)

    # Keep reconstruction metrics separate
    recon_metrics = {
        "test_total": float(test_metrics["total"]),
        "test_recon": float(test_metrics["recon"]),
        "test_vq": float(test_metrics["vq"]),
        "test_perplexity": float(test_metrics["perplexity"]),
    }

    # Compute state-label metrics without overwriting recon_metrics
    state_metrics, per_state_entropy, per_state_purity, state_majority_label = compute_state_label_metrics(
        latent_data_list=test_labels,     # predicted state IDs
        valid_states=test_states,         # ground truth labels
        num_total_states=m.num_embeddings,
    )

    # Merge everything into one dict for saving
    combined_metrics = {}
    combined_metrics.update(recon_metrics)
    combined_metrics.update(state_metrics)
    combined_metrics.update(class_results)
    combined_metrics.update(forecast_results)

    if save:
        out_path = save_state_metrics(
            combined_metrics,
            per_state_entropy,
            per_state_purity,
            state_majority_label,
            save_dir=f"metrics/{dataset}/vqvae/{str(m.num_embeddings)}/",
            filename=f"vqvae_state_metrics_{t.seed}.json",
            extra={
                "run_name": "vqvae_baseline",
                "split": "test",
                "checkpoint_path": ckpt_path,
                "model_config": _dc_to_dict(m),
                "train_config": _dc_to_dict(t),
                "data_meta": out.get("meta", {}),
            },
        )

        print("Saved metrics to:", out_path)
        return combined_metrics, out_path
    else:
        return combined_metrics


def vq_vae_visualisation_setup(out, config):
    m = parse_model_config(config)
    t = parse_train_config(config)

    # Set up model
    model = VQVAE(
        windows_length=out["meta"]["L"],
        input_dim=out["meta"]["D"],
        latent_dim=m.latent_dim,
        num_embeddings=m.num_embeddings,
        beta=m.beta,
        enc_hidden=m.enc_hidden,
        dec_hidden=m.dec_hidden,
        dropout=m.dropout,
    )

    model = model.to(device)

    ckpt_path = os.path.join(t.ckpt_dir, t.ckpt_name)
    _, _, test_loader = make_loaders_from_out(out, config)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # IMPORTANT: evaluate_vqvae should return:
    # test_metrics: dict with keys total/recon/vq/perplexity
    # labels: list[tensor] or tensor
    # states: list[tensor] or tensor
    _, _, labels, states = evaluate_vqvae(model, test_loader, t)

    return labels, states