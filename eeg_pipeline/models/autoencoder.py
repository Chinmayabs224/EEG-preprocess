"""Autoencoder model for anomaly detection."""
# ================================================================
# Task 4: Anomaly Detection (Deep Autoencoder)
#
# Input  : SNN spike-rate features (N, SNN_SPIKE_DIM)
# Learns : compressed representation of NORMAL EEG
# Detect : anomalies via reconstruction error threshold
# ================================================================

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from config import cfg


# ── Variational Autoencoder ──────────────────────────────────────
class EEGAutoencoder(nn.Module):
    """
    Variational Autoencoder on SNN spike features.

    Trained only on non-seizure (normal) windows.
    Anomaly score = reconstruction MSE.

    Input/Output dim: SNN_SPIKE_DIM (64)
    Latent dim      : AE_LATENT_DIM (32)
    """

    def __init__(self):
        super().__init__()
        d   = cfg.SNN_SPIKE_DIM
        lat = cfg.AE_LATENT_DIM

        # ── Encoder ──────────────────────────────────────
        self.encoder = nn.Sequential(
            nn.Linear(d,  128), nn.LayerNorm(128), nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64), nn.LayerNorm(64),  nn.GELU(),
        )
        self.fc_mu      = nn.Linear(64, lat)
        self.fc_log_var = nn.Linear(64, lat)

        # ── Decoder ──────────────────────────────────────
        self.decoder = nn.Sequential(
            nn.Linear(lat, 64),  nn.LayerNorm(64),  nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64,  128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, d)
        )

    # ── Reparameterisation trick ──────────────────────────
    def reparameterise(self,
                        mu     : torch.Tensor,
                        log_var: torch.Tensor
                        ) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu                          # deterministic at test

    def forward(self, x: torch.Tensor
                ) -> tuple[torch.Tensor,
                           torch.Tensor,
                           torch.Tensor]:
        """
        Returns
        -------
        x_hat   : reconstructed input
        mu      : latent mean
        log_var : latent log-variance
        """
        h       = self.encoder(x)
        mu      = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        z       = self.reparameterise(mu, log_var)
        x_hat   = self.decoder(z)
        return x_hat, mu, log_var

    def reconstruction_error(self, x: torch.Tensor
                               ) -> torch.Tensor:
        """Per-sample MSE (used as anomaly score)."""
        x_hat, _, _ = self.forward(x)
        return ((x - x_hat) ** 2).mean(dim=1)   # (batch,)


# ── VAE Loss ─────────────────────────────────────────────────────
def vae_loss(x      : torch.Tensor,
             x_hat  : torch.Tensor,
             mu     : torch.Tensor,
             log_var: torch.Tensor,
             beta   : float = 1.0) -> torch.Tensor:
    """ELBO loss = reconstruction + β × KL divergence."""
    recon = nn.functional.mse_loss(x_hat, x, reduction="mean")
    kl    = -0.5 * (1 + log_var
                    - mu.pow(2)
                    - log_var.exp()).mean()
    return recon + beta * kl


# ── Anomaly Detector wrapper ─────────────────────────────────────
class AnomalyDetector:
    """
    Trains VAE on normal EEG; flags anomalies at inference.
    """

    def __init__(self, device: torch.device):
        self.device    = device
        self.model     = EEGAutoencoder().to(device)
        self.threshold = None
        self.train_errors = None

    # ── Training ──────────────────────────────────────────
    def fit(self,
            X_normal   : np.ndarray,
            n_epochs   : int = cfg.EPOCHS,
            lr         : float = cfg.LEARNING_RATE,
            batch_size : int = cfg.BATCH_SIZE) -> list[float]:
        """
        Train on normal (non-seizure) windows only.
        """
        from torch.utils.data import DataLoader, TensorDataset

        ds     = TensorDataset(
            torch.tensor(X_normal, dtype=torch.float32))
        loader = DataLoader(ds, batch_size=batch_size,
                            shuffle=True)
        opt    = torch.optim.Adam(self.model.parameters(), lr=lr)
        sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=n_epochs)

        losses = []
        print("[AE] Training VAE Autoencoder ...")
        for epoch in range(1, n_epochs + 1):
            self.model.train()
            ep_loss = 0.0
            for (batch,) in loader:
                batch  = batch.to(self.device)
                x_hat, mu, lv = self.model(batch)
                loss   = vae_loss(batch, x_hat, mu, lv)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), 1.0)
                opt.step()
                ep_loss += loss.item() * len(batch)
            sched.step()
            ep_loss /= len(X_normal)
            losses.append(ep_loss)
            if epoch % 10 == 0:
                print(f"  Epoch [{epoch:03d}/{n_epochs}] "
                      f"Loss: {ep_loss:.4f}")

        # ── Set threshold from training reconstruction error ─
        errs = self._batch_errors(X_normal)
        self.train_errors = errs
        self.threshold    = np.percentile(
            errs, cfg.AE_THRESHOLD_PCT)
        print(f"[AE] Anomaly threshold "
              f"(p{cfg.AE_THRESHOLD_PCT}): "
              f"{self.threshold:.4f}")
        return losses

    # ── Inference ─────────────────────────────────────────
    def _batch_errors(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        errors = []
        with torch.no_grad():
            for i in range(0, len(X), 256):
                batch = torch.tensor(
                    X[i:i+256], dtype=torch.float32
                ).to(self.device)
                err = self.model.reconstruction_error(batch)
                errors.extend(err.cpu().numpy())
        return np.array(errors)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray,
                                               np.ndarray]:
        """
        Returns
        -------
        scores   : (N,)  reconstruction error (anomaly score)
        is_anomaly: (N,) bool  True = anomaly detected
        """
        assert self.threshold is not None, "Call fit() first"
        scores     = self._batch_errors(X)
        is_anomaly = scores > self.threshold
        return scores, is_anomaly

    # ── Visualisation ─────────────────────────────────────
    def plot_anomaly_scores(self,
                             scores     : np.ndarray,
                             true_labels: np.ndarray,
                             fname      : str = "ae_scores.png"):
        t   = np.arange(len(scores))
        fig, axes = plt.subplots(2, 1, figsize=(16, 8),
                                  sharex=True)

        # ── Score timeline ────────────────────────────────
        ax = axes[0]
        ax.plot(t, scores, lw=0.8, color="steelblue",
                label="Reconstruction Error")
        ax.axhline(self.threshold, color="red", lw=1.5,
                   linestyle="--", label=f"Threshold "
                   f"(p{cfg.AE_THRESHOLD_PCT})")
        # Shade true seizure regions
        in_sei = False
        for i, lb in enumerate(true_labels):
            if lb == 1 and not in_sei:
                xs = t[i]; in_sei = True
            elif lb == 0 and in_sei:
                ax.axvspan(xs, t[i], alpha=0.25,
                           color="red", label="True Seizure"
                           if xs == t[0] else "")
                in_sei = False
        ax.set_ylabel("Anomaly Score (MSE)", fontsize=11)
        ax.set_title("VAE Anomaly Detection Timeline", fontsize=13)
        ax.legend(fontsize=9)

        # ── Binary anomaly ────────────────────────────────
        ax = axes[1]
        detected = (scores > self.threshold).astype(int)
        ax.fill_between(t, detected, alpha=0.6,
                        color="orange", label="Detected Anomaly")
        ax.fill_between(t, true_labels, alpha=0.4,
                        color="red",    label="True Seizure")
        ax.set_ylim(0, 1.4)
        ax.set_xlabel("Epoch Index", fontsize=11)
        ax.set_ylabel("Label", fontsize=11)
        ax.legend(fontsize=9)

        plt.tight_layout()
        path = os.path.join(cfg.RESULTS_DIR, fname)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"[AE] Anomaly plot saved: {path}")

    def plot_latent_space(self,
                           X          : np.ndarray,
                           labels     : np.ndarray,
                           fname      : str = "ae_latent.png"):
        """2-D UMAP/t-SNE of latent representations."""
        from sklearn.manifold import TSNE

        self.model.eval()
        latents = []
        with torch.no_grad():
            for i in range(0, len(X), 256):
                batch = torch.tensor(
                    X[i:i+256], dtype=torch.float32
                ).to(self.device)
                h     = self.model.encoder(batch)
                mu    = self.model.fc_mu(h)
                latents.append(mu.cpu().numpy())

        Z    = np.concatenate(latents, axis=0)
        Z2d  = TSNE(n_components=2,
                    random_state=42).fit_transform(Z)

        fig, ax = plt.subplots(figsize=(9, 7))
        for cls, clr, lbl in [(0, "steelblue", "Normal"),
                               (1, "tomato",    "Seizure")]:
            m = labels == cls
            ax.scatter(Z2d[m, 0], Z2d[m, 1],
                       c=clr, alpha=0.4, s=8, label=lbl)
        ax.set_title("VAE Latent Space (t-SNE)", fontsize=13)
        ax.legend(fontsize=11)
        plt.tight_layout()
        path = os.path.join(cfg.RESULTS_DIR, fname)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.show()

    def save(self, path: str = None):
        path = path or os.path.join(cfg.MODEL_DIR, "vae.pth")
        torch.save({
            "state_dict": self.model.state_dict(),
            "threshold" : self.threshold
        }, path)
        print(f"[AE] Model saved: {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.threshold = ckpt["threshold"]