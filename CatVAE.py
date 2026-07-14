import os
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from metrics import compute_downstream_metrics, compute_state_label_metrics, save_state_metrics
from utils import MLPDecoder, MLPEncoder, _dc_to_dict, make_loaders_from_out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass(frozen=True)
class ModelConfig:
    latent_dim: int = 64
    categorical_dim: int = 24
    beta: float = 0.25
    temperature: float = 1.0
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

    ckpt_dir: str = "./checkpoints/catvae/"
    ckpt_name: str = "catvae_best.pt"

    seed: int = 42


def parse_model_config(config):
    m = config["model"]
    return ModelConfig(
        latent_dim = int(m.get("latent_dim", 64)),
        categorical_dim = int(m.get("categorical_dim", 24)),
        beta = float(m.get("beta", 0.25)),
        temperature = float(m.get("temperature", 1.0)),
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
        ckpt_name=str(t.get("ckpt_name", "catvae_best.pt")),

        seed=seed,
    )


class CategoricalVAE(nn.Module):
    def __init__(self, input_dim: int, categorical_dim: int,  enc_out_dim: tuple[int],
                 dec_out_dim: tuple[int], beta: float, temperature: float, dropout: float=0.0):
        super(CategoricalVAE, self).__init__()
        # Initialise parameters
        self.input_dim = input_dim
        self.enc_out_dim = enc_out_dim
        self.dec_out_dim = dec_out_dim
        self.categorical_dim = categorical_dim
        self.temp = temperature
        self.beta = beta
        self.dropout = dropout

        # Build Encoder
        self.encoder = MLPEncoder(self.input_dim, self.categorical_dim, self.enc_out_dim, self.dropout)
        self.decoder = MLPDecoder(self.input_dim*2, self.categorical_dim, self.dec_out_dim, self.dropout)

        # Categorical prior
        self.pz = torch.distributions.OneHotCategorical(
            1. / self.categorical_dim * torch.ones(1, self.categorical_dim, device=device))

    def forward(self, x):
        pzx_logits, mu, sigma = self.catvae_training(x)
        loss_dct = self.loss_function(x, pzx_logits, mu, sigma)
        return loss_dct['Loss'], loss_dct["recon_loss"], loss_dct["KLD_cat"]

    def encode(self, input_data):
        """
        Encodes the input by passing through the encoder network and returns the latent codes.

        Args:
            - input_data (Torch.Tensor): The data

        Return:
            - z_out (Torch.Tensor):
        """
        result = self.encoder(input_data)

        return result


    def decode(self, z):
        """
        Computes parameters for pxz from samples of pzx

        Args:
            - z (Torch.Tensor): Embeddings

        Return:
            - mu (Torch.Tensor): Mean of the decoder
            - sigma (Torch.Tensor): Standard deviation of the decoder
        """
        result = self.decoder(z)
        mu, logvar = torch.split(result, self.input_dim, dim=-1)
        #mu = self.fc_mu_x(result)
        #logvar = self.fc_logvar_x(result)
        #sigma = torch.cat(
        #    [torch.diag(torch.exp(logvar[i, :])) for i in range(z.shape[0])]
        #).view(-1, self.input_dim, self.input_dim)
        #sigma = torch.diag_embed(torch.exp(logvar))
        return mu, logvar

    def sample_gumble(self, logits, eps = 1e-7):
        """
        Gumbel-softmax trick to sample from Categorical Distribution

        Args:
            - logits (Torch.Tensor): One hot encodings
            - eps (float): Uncertainty

        Return:
            - s (Torch.Tensor): Softmax
        """
        # Sample from Gumbel
        u = torch.rand_like(logits)
        g = - torch.log(- torch.log(u + eps) + eps)
        s = F.softmax((logits + g) / self.temp, dim=-1)
        return s


    def catvae_training(self, x):
        """
        Conduct the training step for the CatVAE.
        """
        # First compute parameters of categorical distribution pzx
        pzx_logits = self.encode(x)
        # Create one hot categorical distribution object for use in loss func
        # pzx = torch.distributions.OneHotCategorical(logits=pzx_logits)
        # Sample from pzx
        z = F.gumbel_softmax(pzx_logits, tau=self.temp, hard=False)
        # Decode into mu and sigma
        mu, logvar = self.decode(z)
        # Construct multivariate distribution object for pxz
        #pxz = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=sigma)
        return pzx_logits, mu, logvar


    def get_states(self, x):
        """
        Computation of the discretised states
        """
        # First compute parameters of categorical dist. pzx
        pzx_logits = self.encode(x)
        # Compute states by using the argmax of logits
        z_states = torch.argmax(pzx_logits, dim=-1)

        return z_states

    def loss_function(self, x, pzx_logits, mu, logvar):
        """
        Loss function for the CatVAE.
        """
        # Compute the recon loss
        var = torch.exp(logvar).clamp_min(1e-8)
        recon_element = (x - mu) ** 2 / var + logvar
        recon_loss = 0.5 * torch.mean(torch.sum(recon_element, dim=-1))

        # Compute KL-divergence for categorical distribution
        probs = F.softmax(pzx_logits, dim=-1)
        log_probs = F.log_softmax(pzx_logits, dim=-1)
        # KL = sum(q * (log(q) - log(p)))
        prior_log_prob = torch.log(torch.tensor(1.0 / self.categorical_dim, device=device))
        kl_loss = torch.sum(probs * (log_probs - prior_log_prob), dim=-1).mean()
        loss = recon_loss + self.beta * kl_loss

        return {'Loss': loss, 'recon_loss': recon_loss, 'KLD_cat': kl_loss}


def train_catvae(out, config):
    """
    Set up training and trains the CatVAE.
    """
    # Load config
    m = parse_model_config(config)
    t = parse_train_config(config)

    # Set seeds for reproducibility
    torch.manual_seed(t.seed)
    np.random.seed(t.seed)

    # Set up model
    model = CategoricalVAE(
        input_dim=out["meta"]["D"],
        categorical_dim=m.categorical_dim,
        beta=m.beta,
        temperature=m.temperature,
        enc_out_dim=m.enc_hidden,
        dec_out_dim=m.dec_hidden,
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
        "train_kl": [],
        "val_total": [],
        "val_recon": [],
        "val_kl": [],
        "best_ckpt_path": ckpt_path,
    }

    print("Begin training")
    for epoch in range(1, t.epochs + 1):
        model.train()

        train_total_sum = 0.0
        train_recon_sum = 0.0
        train_kl_sum = 0.0
        n_batches = 0

        for x, _ in train_loader:
            x = x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            total, recon_loss, kl_loss = model(x)
            total.backward()

            #if t.grad_clip_norm is not None:
            #    nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip_norm)

            optimizer.step()

            train_total_sum += float(total.item())
            train_recon_sum += float(recon_loss.item())
            train_kl_sum += float(kl_loss.item())
            n_batches += 1

        train_metrics = {
            "total": train_total_sum / max(n_batches, 1),
            "recon": train_recon_sum / max(n_batches, 1),
            "kl": train_kl_sum / max(n_batches, 1),
        }

        history["train_total"].append(train_metrics["total"])
        history["train_recon"].append(train_metrics["recon"])
        history["train_kl"].append(train_metrics["kl"])

        val_metrics, _, _, _, _ = evaluate_catvae(model, val_loader, t)

        history["val_total"].append(val_metrics["total"])
        history["val_recon"].append(val_metrics["recon"])
        history["val_kl"].append(val_metrics["kl"])

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
            f"train total {train_metrics['total']:.6f} (recon {train_metrics['recon']:.6f}, kl {train_metrics['kl']:.6f}) | "
            f"val total {val_metrics['total']:.6f} (recon {val_metrics['recon']:.6f}, kl {val_metrics['kl']:.6f}) | "
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
def evaluate_catvae(model, loader,t):
    model.eval()
    total_loss_sum = 0.0
    recon_sum = 0.0
    kl_sum = 0.0
    n_batches = 0
    _indices = 0
    states = []
    y_labels = []
    data = []
    recon_data = []

    for (x, y) in loader:
        x = x.to(device, non_blocking=True)
        total, recon_loss, kl_loss = model(x)
        y_label = model.get_states(x)
        states.append(y)
        y_labels.append(y_label)
        data.append(x)

        total_loss_sum += float(total.item())
        recon_data.append(recon_loss)
        recon_sum += float(recon_loss.item())
        kl_sum += float(kl_loss.item())
        n_batches += 1

    if n_batches == 0:
        return {"total": float("nan"), "recon": float("nan"), "kl": float("nan")}

    return {
        "total": total_loss_sum / n_batches,
        "recon": recon_sum / n_batches,
        "kl": kl_sum / n_batches,
    }, data, y_labels, states, recon_data


def compute_catvae_metrics(config, out, dataset):
    m = parse_model_config(config)
    t = parse_train_config(config)

    torch.manual_seed(t.seed)
    np.random.seed(t.seed)

    # Set up model
    model = CategoricalVAE(
        input_dim=out["meta"]["D"],
        categorical_dim=m.categorical_dim,
        beta=m.beta,
        temperature=m.temperature,
        enc_out_dim=m.enc_hidden,
        dec_out_dim=m.dec_hidden,
        dropout=m.dropout,
    )
    model = model.to(device)

    print("Model", model.categorical_dim)

    ckpt_path = os.path.join(t.ckpt_dir, t.ckpt_name)
    train_loader, _, test_loader = make_loaders_from_out(out, config)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # IMPORTANT: evaluate_vqvae should return:
    train_metrics, train_features, train_labels, train_states, _ = evaluate_catvae(model, train_loader, t)
    test_metrics, test_features, test_labels, test_states, _ = evaluate_catvae(model, test_loader, t)

    class_results, forecast_results, _ = compute_downstream_metrics(train_latents=train_labels,
                                                                    train_labels=train_states,
                                                                    train_features=train_features,
                                                                    test_latents=test_labels, test_labels=test_states,
                                                                    test_features=test_features,
                                                                    classification=True, forecasting=True)

    # Keep reconstruction metrics separate
    recon_metrics = {
        "test_total": float(test_metrics["total"]),
        "test_recon": float(test_metrics["recon"]),
        "test_kl": float(test_metrics["kl"]),
    }

    # Compute state-label metrics without overwriting recon_metrics
    state_metrics, per_state_entropy, per_state_purity, state_majority_label = compute_state_label_metrics(
        latent_data_list=test_labels,     # predicted state IDs
        valid_states=test_states,         # ground truth labels
        num_total_states=m.categorical_dim,  # if your function supports this arg; otherwise remove
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
        save_dir=f"metrics/{dataset}/catvae/{str(m.categorical_dim)}/",
        filename=f"catvae_state_metrics_{t.seed}.json",
        extra={
            "run_name": "catvae_baseline",
            "split": "test",
            "checkpoint_path": ckpt_path,
            "model_config": _dc_to_dict(m),
            "train_config": _dc_to_dict(t),
            "data_meta": out.get("meta", {}),
        },
    )

    print("Saved metrics to:", out_path)
    return combined_metrics, out_path

def compute_catvae_anomaly_score(config, normal_out, anomaly_out, dataset, save=True):
    m = parse_model_config(config)
    t = parse_train_config(config)

    torch.manual_seed(t.seed)
    np.random.seed(t.seed)

    # Set up model
    model = CategoricalVAE(
        input_dim=normal_out["meta"]["D"],
        categorical_dim=m.categorical_dim,
        beta=m.beta,
        temperature=m.temperature,
        enc_out_dim=m.enc_hidden,
        dec_out_dim=m.dec_hidden,
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
    _, _, _, _, normal_recon = evaluate_catvae(model, normal_test_loader, t)
    _, _, _, _, anomaly_recon = evaluate_catvae(model, anomaly_test_loader, t)

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
            save_dir=f"metrics/{dataset}/catvae/anomaly_detection/{str(m.categorical_dim)}/",
            filename=f"catvae_state_metrics_{t.seed}.json",
            extra={
                "run_name": "catvae_baseline",
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
