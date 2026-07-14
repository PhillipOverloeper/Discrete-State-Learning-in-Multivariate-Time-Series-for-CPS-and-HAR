import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics import compute_downstream_metrics, compute_state_label_metrics, save_state_metrics
from utils import MLPDecoder, MLPEncoder, _dc_to_dict, make_loaders_from_out

# Check whether GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass(frozen=True)
class ModelConfig:
    latent_dim: int = 64
    som_dim: tuple = (4, 4)
    alpha: float = 0.01
    beta: float = 0.25
    gamma: float = 0.01
    tau: float = 0.01
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

    early_stop_patience: int = 15,

    ckpt_dir: str = "./checkpoints/somvae/"
    ckpt_name: str = "somvae_best.pt"

    seed: int = 42


def parse_model_config(config):
    m = config["model"]
    return ModelConfig(
        latent_dim = int(m.get("latent_dim", 64)),
        som_dim = tuple(m.get("som_dim", (4, 4))),
        alpha=float(m.get("alpha", 1.0)),
        beta = float(m.get("beta", 1.0)),
        gamma=float(m.get("gamma", 1.0)),
        tau=float(m.get("tau", 1.0)),
        enc_hidden = tuple(m.get("enc_hidden", (256, 256))),
        dec_hidden = tuple(m.get("dec_hidden", (256, 256))),
        dropout = float(m.get("dropout", 0.0)),
    )


def parse_train_config(config):
    t = config["train"]
    seed = int(config.get("seed", t.get("seed", 42)))
    return TrainConfig(
        epochs = int(t.get("epochs", 50)),
        batch_size = int(t.get("batch_size", 256)),
        lr = float(t.get("lr", 2e-4)),
        weight_decay = float(t.get("weight_decay", 0.0)),
        grad_clip_norm = t.get("grad_clip_norm", 1.0),

        early_stop_patience = int(t.get("early_stop_patience", 15)),

        ckpt_dir=str(t.get("ckpt_dir", "checkpoints")),
        ckpt_name=str(t.get("ckpt_name", "somvae_best.pt")),

        seed=seed,
    )


class SOMVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=8, som_dim=None, input_length=16,
                 enc_hidden_dims=(16, 16), dec_hidden_dims=(16,16), alpha=1., beta=1., gamma=1., tau=1., dropout=1.0):
        """
        Initialize a Self-Organizing Map Variational Autoencoder (SOM-VAE) model.

        Args:
            latent_dim (int): Dimensionality of the latent space.
            som_dim (list): Dimensions of the self-organizing map grid [rows, columns].
            input_length (int): Length of the input data.
            alpha (float): Weight parameter for the SOM loss.
            beta (float): Weight parameter for the reconstruction loss.
            gamma (float): Weight parameter for the commitment loss.
            tau (float): Weight parameter for the topographic loss.
        """

        super(SOMVAE, self).__init__()

        # Store hyperparameters
        if som_dim is None:
            self.som_dim = (4, 3)
        else:
            self.som_dim = som_dim
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.enc_hidden_dims = enc_hidden_dims
        self.dec_hidden_dims = dec_hidden_dims
        self.input_length = input_length
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.dropout = dropout

        # Initialize SOM embeddings
        self.embeddings = nn.Parameter(
            nn.init.trunc_normal_(torch.empty((self.som_dim[0], self.som_dim[1], self.latent_dim)),
                                  std=0.05, a=-0.1, b=0.1))

        self.transition_logits = nn.Parameter(torch.zeros(*(self.som_dim + self.som_dim)))

        # Define the encoder networks
        self.encoder = MLPEncoder(self.input_dim, self.latent_dim, self.enc_hidden_dims)
        self.q_decoder = MLPDecoder(self.input_dim, self.latent_dim, self.dec_hidden_dims)
        self.e_decoder = MLPDecoder(self.input_dim, self.latent_dim, self.dec_hidden_dims)


    def forward(self, x):
        """
        Forward pass through the SOM-VAE model.

        Args:
            x (Tensor): Input data.

        Returns:
            x_hat_q (Tensor): Reconstructed input using the q-decoder.
            x_hat_e (Tensor): Reconstructed input using the e-decoder.
            z_e (Tensor): Latent representation obtained from the encoder.
            z_q (Tensor): Latent representation obtained from the q-decoder.
            k (Tensor): SOM cluster assignments.
            z_dist_flat (Tensor): Flattened distribution of latent vectors.
            z_q_neighbors (Tensor): Latent representations of SOM neighbors.
        """
        # Compute z_e, z_q, z_q_neighbors, and other necessary components
        x = x.to(device)
        z_e = self.encoder(x)
        k = self._k(z_e)  # Compute SOM cluster assignments

        z_q = self.z_q_calc(k)  # Compute latent representation from the q-decoder

        # Compute reconstructions
        x_hat_q = self.q_decoder(z_q)  # Reconstructed input using the q-decoder
        x_hat_e = self.e_decoder(z_e)  # Reconstructed input using the e-decoder

        return x_hat_q, x_hat_e, z_e, z_q, k


    def z_q_calc(self, k):
        """
        Find embeddings for each k.

        Parameters:
            k (torch.Tensor): Tensor of the positions of the embeddings.

        Returns:
            z_q (torch.Tensor): Embeddings for each k.
        """
        # Split k into row and column components
        k = k.to(device)
        k_1 = k // self.som_dim[1]
        k_2 = k % self.som_dim[1]
        z_q = self.embeddings[k_1, k_2]

        return z_q


    def _k(self, z_e):
        """
        Picks the index of the closest embedding for every encoding.

        Args:
            z_e (Tensor): Latent representations from the encoder.

        Returns:
            k (Tensor): Indices of the closest SOM embeddings for each encoding.
        """
        # Calculate the squared distances between z_e and SOM embeddings
        z_e = z_e.to(device)
        z_dist_flat = torch.cdist(z_e, self.embeddings.view(-1, self.latent_dim), p=2)**2

        # Find the index of the closest embedding for each encoding
        k = torch.argmin(z_dist_flat, dim=-1)

        return k


    def get_probs(self):
        # Ensures probabilities sum to 1 across the last two dims (the SOM grid)
        return F.softmax(self.transition_logits.view(self.som_dim[0], self.som_dim[1], -1), dim=-1).view(
            *(self.som_dim + self.som_dim))


    def loss(self, x, x_hat_q, x_hat_e, z_q, z_e, k):
        """
        Computes the overall loss for the SOM-VAE model.

        Args:
            x (Tensor): Input data.
            x_hat_q (Tensor): Reconstructed input using the q-decoder.
            x_hat_e (Tensor): Reconstructed input using the e-decoder.
            z_q (Tensor): Latent representation obtained from the q-decoder.
            z_e (Tensor): Latent representation obtained from the encoder.
            z_q_neighbors (Tensor): Latent representations of SOM neighbors.
            k (Tensor): SOM cluster assignments.
            z_dist_flat (Tensor): Flattened squared distances between z_e and SOM embeddings.

        Returns:
            total_loss (Tensor): Overall loss for the SOM-VAE model.
        """

        x = x.to(device)
        x_hat_q = x_hat_q.to(device)
        x_hat_e = x_hat_e.to(device)
        z_q = z_q.to(device)
        z_e = z_e.to(device)
        k = k.to(device)

        # Reconstruction loss for both q-decoder and e-decoder
        loss_rec = F.mse_loss(x_hat_q, x) + F.mse_loss(x_hat_e, x)

        # Commitment loss
        loss_commit = F.mse_loss(z_e, z_q.detach())

        # Topographic loss
        loss_som = F.mse_loss(z_q, z_e.detach())

        # Loss related to SOM probabilities
        probs = self.get_probs()
        k_curr_row = k // self.som_dim[1]
        k_curr_col = k % self.som_dim[1]
        # Shift for Markov property (t-1)
        k_prev_row = torch.cat([k_curr_row[:1], k_curr_row[:-1]])
        k_prev_col = torch.cat([k_curr_col[:1], k_curr_col[:-1]])

        trans_probs = probs[k_prev_row, k_prev_col, k_curr_row, k_curr_col]
        loss_prob = -self.gamma * torch.mean(torch.log(trans_probs + 1e-10))
        total_loss = loss_rec + self.alpha * loss_commit + self.beta * loss_som + loss_prob




        return total_loss, loss_rec, loss_commit, loss_som, loss_prob

    @torch.no_grad()
    def get_discrete_states(self, x: torch.Tensor) -> torch.Tensor:
        """
        Converts the input data into discrete SOM indices.

        Args:
            x: Input data [Batch, Input_Length, Dimension].
        Returns:
            k: The index of the winning SOM node [Batch, Input Length].
        """
        self.eval()

        # Encode to latent space
        z_e = self.encoder(x)

        # Compute distance to all SOM nodes
        emb_flat = self.embeddings.view(-1, self.latent_dim)
        distances = torch.cdist(z_e, emb_flat, p=2)**2

        k = torch.argmin(distances, dim=-1)

        return k


def train_somvae(out, config):
    """
    Set up training and trains the SOM-VAE
    """
    # Load config
    m = parse_model_config(config)
    t = parse_train_config(config)

    # Set seeds for reproducibility
    torch.manual_seed(t.seed)
    np.random.seed(t.seed)

    # Set up model
    model = SOMVAE(
        input_dim=out["meta"]["D"],
        som_dim=m.som_dim,
        alpha=m.alpha,
        beta=m.beta,
        gamma=m.gamma,
        tau=m.tau,
        enc_hidden_dims=m.enc_hidden,
        dec_hidden_dims=m.dec_hidden,
        dropout=m.dropout,
    )
    model = model.to(device)
    print("Model has been set up")

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
        "train_commit": [],
        "train_som": [],
        "train_probs": [],
        "val_total": [],
        "val_recon": [],
        "val_commit": [],
        "val_som": [],
        "val_probs": [],
        "best_ckpt_path": ckpt_path,
    }

    print("Begin training")
    for epoch in range(1, t.epochs + 1):
        model.train()

        train_total_sum = 0.0
        train_recon_sum = 0.0
        train_commit_sum = 0.0
        train_som_sum = 0.0
        train_probs_sum = 0.0
        n_batches = 0

        for x, _ in train_loader:
            x = x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            x_hat_q, x_hat_e, z_e, z_q, k = model(x)

            total, recon_loss, commit_loss, som_loss, probs_loss = model.loss(x, x_hat_q, x_hat_e, z_q, z_e, k)
            total.backward()

            #if t.grad_clip_norm is not None:
            #    nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip_norm)

            optimizer.step()

            train_total_sum += float(total.item())
            train_recon_sum += float(recon_loss.item())
            train_commit_sum += float(commit_loss.item())
            train_som_sum += float(som_loss.item())
            train_probs_sum += float(probs_loss.item())

            n_batches += 1

        train_metrics = {
            "total": train_total_sum / max(n_batches, 1),
            "recon": train_recon_sum / max(n_batches, 1),
            "commit": train_commit_sum / max(n_batches, 1),
            "som": train_som_sum / max(n_batches, 1),
            "probs": train_probs_sum / max(n_batches, 1),
        }

        history["train_total"].append(train_metrics["total"])
        history["train_recon"].append(train_metrics["recon"])
        history["train_commit"].append(train_metrics["commit"])
        history["train_som"].append(train_metrics["som"])
        history["train_probs"].append(train_metrics["probs"])

        val_metrics, _, _, _, _ = evaluate_somvae(model, val_loader, t)

        history["val_total"].append(val_metrics["total"])
        history["val_recon"].append(val_metrics["recon"])
        history["val_commit"].append(val_metrics["commit"])
        history["val_som"].append(val_metrics["som"])
        history["val_probs"].append(val_metrics["probs"])

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
            f"train total {train_metrics['total']:.6f} (recon {train_metrics['recon']:.6f}, commit {train_metrics['commit']:.6f}, som {train_metrics['som']:.6f}, probs {train_metrics['probs']:.6f}) | "
            f"val total {val_metrics['total']:.6f} (recon {val_metrics['recon']:.6f}, commit {val_metrics['commit']:.6f}, som {val_metrics['som']:.6f}, probs {val_metrics['probs']:.6f}) | "
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
def evaluate_somvae(model, loader,t):
    model.eval()
    total_loss_sum = 0.0
    recon_sum = 0.0
    commit_sum = 0.0
    som_sum = 0.0
    probs_sum = 0.0
    n_batches = 0
    _indices = 0
    states = []
    y_labels = []
    data = []
    recon_data = []

    for (x, y) in loader:
        x = x.to(device, non_blocking=True)
        x_hat_q, x_hat_e, z_e, z_q, k = model(x)
        data.append(x)

        total, recon_loss, commit_loss, som_loss, probs_loss = model.loss(x, x_hat_q, x_hat_e, z_q, z_e, k)
        y_label = model.get_discrete_states(x)
        states.append(y)
        y_labels.append(y_label)
        recon_data.append(recon_loss)

        total_loss_sum += float(total)
        recon_sum += float(recon_loss)
        commit_sum += float(commit_loss)
        som_sum += float(som_loss)
        probs_sum += float(probs_loss)
        n_batches += 1

    if n_batches == 0:
        return {"total": float("nan"), "recon": float("nan"), "commit": float("nan"), "som": float("nan"), "probs": float("nan")}

    return {
        "total": total_loss_sum / n_batches,
        "recon": recon_sum / n_batches,
        "commit": commit_sum / n_batches,
        "som": som_sum / n_batches,
        "probs": probs_sum / n_batches,
    }, data, y_labels, states, recon_data


def compute_somvae_metrics(config, out, dataset):
    m = parse_model_config(config)
    t = parse_train_config(config)

    torch.manual_seed(t.seed)
    np.random.seed(t.seed)

    # Set up model
    model = SOMVAE(
        input_dim=out["meta"]["D"],
        som_dim=m.som_dim,
        alpha=m.alpha,
        beta=m.beta,
        gamma=m.gamma,
        tau=m.tau,
        enc_hidden_dims=m.enc_hidden,
        dec_hidden_dims=m.dec_hidden,
        dropout=m.dropout,
    )
    model = model.to(device)

    ckpt_path = os.path.join(t.ckpt_dir, t.ckpt_name)
    train_loader, _, test_loader = make_loaders_from_out(out, config)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # IMPORTANT: evaluate_vqvae should return:
    train_metrics, train_features, train_labels, train_states, _ = evaluate_somvae(model, train_loader, t)
    test_metrics, test_features, test_labels, test_states, _ = evaluate_somvae(model, test_loader, t)

    # Keep reconstruction metrics separate
    recon_metrics = {
        "test_total": float(test_metrics["total"]),
        "test_recon": float(test_metrics["recon"]),
        "test_commit": float(test_metrics["commit"]),
        "test_som": float(test_metrics["som"]),
        "test_probs": float(test_metrics["probs"]),
    }

    class_results, forecast_results, _ = compute_downstream_metrics(train_latents=train_labels, train_labels=train_states, train_features=train_features,
                               test_latents=test_labels, test_labels=test_states, test_features=test_features,
                               classification=True, forecasting=True)

    # Compute state-label metrics without overwriting recon_metrics
    state_metrics, per_state_entropy, per_state_purity, state_majority_label = compute_state_label_metrics(
        latent_data_list=test_labels,     # predicted state IDs
        valid_states=test_states,         # ground truth labels
        num_total_states=m.som_dim[0] * m.som_dim[1],  # if your function supports this arg; otherwise remove
    )

    # Merge everything into one dict for saving
    combined_metrics = {}
    combined_metrics.update(recon_metrics)
    combined_metrics.update(state_metrics)
    combined_metrics.update(class_results)
    combined_metrics.update(forecast_results)

    out_path = save_state_metrics(
        combined_metrics,
        per_state_entropy,
        per_state_purity,
        state_majority_label,
        save_dir=f"metrics/{dataset}/somvae/{str(m.som_dim)}/",
        filename=f"somvae_state_metrics_{t.seed}.json",
        extra={
            "run_name": "somvae_baseline",
            "split": "test",
            "checkpoint_path": ckpt_path,
            "model_config": _dc_to_dict(m),
            "train_config": _dc_to_dict(t),
            "data_meta": out.get("meta", {}),
        },
    )

    print("Saved metrics to:", out_path)
    return combined_metrics, out_path

def compute_somvae_anomaly_score(config, normal_out, anomaly_out, dataset, save=True):
    m = parse_model_config(config)
    t = parse_train_config(config)

    torch.manual_seed(t.seed)
    np.random.seed(t.seed)

    model = SOMVAE(
        input_dim=normal_out["meta"]["D"],
        som_dim=m.som_dim,
        alpha=m.alpha,
        beta=m.beta,
        gamma=m.gamma,
        tau=m.tau,
        enc_hidden_dims=m.enc_hidden,
        dec_hidden_dims=m.dec_hidden,
        dropout=m.dropout,
    )
    model = model.to(device)

    ckpt_path = os.path.join(t.ckpt_dir, t.ckpt_name)
    normal_train_loader, _, normal_test_loader = make_loaders_from_out(normal_out, config)
    _, _, anomaly_test_loader = make_loaders_from_out(anomaly_out, config)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # IMPORTANT: evaluate_vqvae should return:
    # test_metrics: dict with keys total/recon/vq/perplexity
    # labels: list[tensor] or tensor
    # states: list[tensor] or tensor
    _, _, _, _, normal_recon = evaluate_somvae(model, normal_test_loader, t)
    _, _, _, _, anomaly_recon = evaluate_somvae(model, anomaly_test_loader, t)

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
            save_dir=f"metrics/{dataset}/somvae/anomaly_detection/{str(m.som_dim)}/",
            filename=f"somvae_state_metrics_{t.seed}.json",
            extra={
                "run_name": "somvae_baseline",
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