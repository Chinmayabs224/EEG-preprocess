"""
Model 1: Autoencoder for Unsupervised Anomaly Detection.

The Autoencoder is trained ONLY on non-seizure (normal) windows.
During inference, seizure windows produce high reconstruction error
because the model has never seen seizure patterns before.

Architecture::

    Encoder: (seq_len, 64) → FC(128) → FC(32)  → latent z
    Decoder: latent z → FC(128) → FC(seq_len * 64) → reconstruct

Output:
    - Latent vector z (32-dim) — used as input for LSTM/Transformer
    - Reconstruction error — used as anomaly score
"""

import os
import numpy as np
import torch
import torch.nn as nn
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class EEGAutoencoder(nn.Module):
    """
    Sequence Autoencoder for EEG spike features.
    
    Encodes a (seq_len, 64) input into a 32-dim latent space
    and reconstructs it back. Reconstruction error serves as
    anomaly score for seizure detection.
    """

    def __init__(self, seq_len: int = 30, feature_dim: int = 64, latent_dim: int = 32):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.input_dim = seq_len * feature_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, latent_dim),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, self.input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode (B, seq_len, 64) → (B, latent_dim)."""
        B = x.size(0)
        flat = x.reshape(B, -1)
        return self.encoder(flat)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode (B, latent_dim) → (B, seq_len, 64)."""
        B = z.size(0)
        out = self.decoder(z)
        return out.reshape(B, self.seq_len, self.feature_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (B, seq_len, 64)

        Returns
        -------
        reconstruction : torch.Tensor, shape (B, seq_len, 64)
        latent : torch.Tensor, shape (B, latent_dim)
        """
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample reconstruction error (MSE).
        High error = anomaly (likely seizure).
        
        Returns shape (B,)
        """
        recon, _ = self.forward(x)
        mse = ((x - recon) ** 2).mean(dim=(1, 2))
        return mse


def train_autoencoder(
    model: EEGAutoencoder,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-3,
    save_dir: str = "./models",
) -> Dict[str, list]:
    """
    Train the Autoencoder on normal-only data.

    Returns training history dict.
    """
    from .utils import EarlyStopping

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5, min_lr=1e-6,
    )
    criterion = nn.MSELoss()

    os.makedirs(save_dir, exist_ok=True)
    early_stop = EarlyStopping(
        patience=12, mode="min",
        save_path=os.path.join(save_dir, "autoencoder_best.pt"),
    )

    history = {"train_loss": [], "val_loss": []}

    print(f"\n{'█' * 50}")
    print(f"█  AUTOENCODER TRAINING (Normal-only)")
    print(f"█  Device: {device}")
    print(f"{'█' * 50}\n")

    for epoch in range(1, epochs + 1):
        # ── Train ────────────────────────────────────────
        model.train()
        train_loss = 0.0
        n_train = 0

        for x_batch, _ in train_loader:
            x_batch = x_batch.to(device)

            recon, _ = model(x_batch)
            loss = criterion(recon, x_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * x_batch.size(0)
            n_train += x_batch.size(0)

        train_loss /= max(n_train, 1)

        # ── Validate ─────────────────────────────────────
        model.eval()
        val_loss = 0.0
        n_val = 0

        with torch.no_grad():
            for x_batch, _ in val_loader:
                x_batch = x_batch.to(device)
                recon, _ = model(x_batch)
                loss = criterion(recon, x_batch)
                val_loss += loss.item() * x_batch.size(0)
                n_val += x_batch.size(0)

        val_loss /= max(n_val, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{epochs} │ "
            f"train_loss={train_loss:.6f} │ "
            f"val_loss={val_loss:.6f} │ "
            f"lr={current_lr:.2e}"
        )

        if early_stop(val_loss, model):
            print(f"  ⏹ Early stopping at epoch {epoch}")
            break

    # Reload best checkpoint
    best_path = os.path.join(save_dir, "autoencoder_best.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))

    return history


def evaluate_autoencoder(
    model: EEGAutoencoder,
    test_loader,
    device: torch.device,
    threshold_percentile: float = 95.0,
) -> Dict[str, float]:
    """
    Evaluate Autoencoder for anomaly detection on the test set.

    The threshold for seizure detection is set at the given percentile
    of reconstruction errors on the test set.

    Returns classification metrics.
    """
    from .utils import compute_metrics

    model = model.to(device)
    model.eval()

    all_errors = []
    all_labels = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            errors = model.anomaly_score(x_batch)
            all_errors.append(errors.cpu().numpy())
            # For autoencoder mode, y_batch is x_batch; we need actual labels
            # y_batch was set to x_batch in autoencoder mode, so we check
            if y_batch.dim() == 1:
                all_labels.append(y_batch.numpy())
            else:
                # In autoencoder mode, labels were passed as sequences
                # We need to retrieve actual labels from the dataset
                pass

    all_errors = np.concatenate(all_errors)
    if all_labels:
        all_labels = np.concatenate(all_labels)
    else:
        logger.warning("No labels available for autoencoder evaluation")
        return {"threshold": float(np.percentile(all_errors, threshold_percentile))}

    # Set threshold at the percentile of reconstruction errors
    threshold = np.percentile(all_errors, threshold_percentile)
    y_pred = (all_errors > threshold).astype(int)
    y_prob = all_errors / (all_errors.max() + 1e-8)  # normalize to [0, 1]

    metrics = compute_metrics(all_labels, y_pred, y_prob)
    metrics["threshold"] = float(threshold)
    metrics["mean_error_normal"] = float(all_errors[all_labels == 0].mean()) if (all_labels == 0).any() else 0.0
    metrics["mean_error_seizure"] = float(all_errors[all_labels == 1].mean()) if (all_labels == 1).any() else 0.0

    return metrics
