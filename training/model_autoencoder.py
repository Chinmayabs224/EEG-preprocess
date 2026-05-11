"""
Model 1: Anomaly Detection for Seizure Detection.

Three anomaly detection strategies:

    1. **Deep SVDD** — One-class deep learning that learns a hypersphere
       enclosing normal data. Distance from center = anomaly score.
    2. **Isolation Forest** (sklearn) — Tree-based anomaly detector that
       isolates outliers using random feature splits. Fast and robust.
    3. **Autoencoder** (legacy) — Reconstruction error as anomaly score.

The combined anomaly score averages Deep SVDD + Isolation Forest for
much sharper separation than reconstruction error alone.

Architecture (Deep SVDD)::

    Input(seq_len × feature_dim) → FC(512) → FC(256) → FC(128) → FC(32)
    Normal data maps close to learned center c; anomalies are far.

Architecture (Autoencoder, legacy)::

    Encoder: (seq_len, 64) → FC(128) → FC(32)  → latent z
    Decoder: latent z → FC(128) → FC(seq_len * 64) → reconstruct
"""

import os
import numpy as np
import torch
import torch.nn as nn
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════
# Deep SVDD (One-Class Deep Learning)
# ═════════════════════════════════════════════════════════════════

class DeepSVDDNetwork(nn.Module):
    """
    Feature extraction network for Deep SVDD.

    Maps input to a latent space where normal data clusters around
    a learned center. Anomalies are far from this center.
    """

    def __init__(self, input_dim: int, latent_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepSVDD(nn.Module):
    """
    Deep Support Vector Data Description.

    Trains a network to map normal data close to a hypersphere center.
    Anomaly score = distance from center in latent space.
    """

    def __init__(self, seq_len: int = 30, feature_dim: int = 64, latent_dim: int = 32):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.input_dim = seq_len * feature_dim
        self.latent_dim = latent_dim

        self.network = DeepSVDDNetwork(self.input_dim, latent_dim)
        # Center will be initialized after first forward pass on normal data
        self.register_buffer("center", torch.zeros(latent_dim))
        self.center_initialized = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map input to latent space. x: (B, seq_len, feature_dim)."""
        B = x.size(0)
        flat = x.reshape(B, -1)
        return self.network(flat)

    def init_center(self, loader, device: torch.device):
        """Initialize center as the mean of all normal training data in latent space."""
        self.eval()
        self.to(device)
        embeddings = []
        with torch.no_grad():
            for x_batch, _ in loader:
                x_batch = x_batch.to(device)
                z = self.forward(x_batch)
                embeddings.append(z)
        embeddings = torch.cat(embeddings, dim=0)
        center = embeddings.mean(dim=0)
        # Avoid center being too close to zero (prevents collapsed solutions)
        center[(torch.abs(center) < 0.01)] = 0.01
        self.center = center
        self.center_initialized = True
        logger.info(f"Deep SVDD center initialized (dim={self.latent_dim})")

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample anomaly score (distance from center).
        High score = anomaly (likely seizure). Returns shape (B,).
        """
        z = self.forward(x)
        return torch.sum((z - self.center) ** 2, dim=1)


# ═════════════════════════════════════════════════════════════════
# Legacy Autoencoder
# ═════════════════════════════════════════════════════════════════

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


# ═════════════════════════════════════════════════════════════════
# Combined Anomaly Detector
# ═════════════════════════════════════════════════════════════════

class CombinedAnomalyDetector:
    """
    Combines Deep SVDD + Isolation Forest for robust anomaly detection.

    The combined score is the average of both normalized scores,
    providing much better separation than either method alone.
    """

    def __init__(self, seq_len: int = 30, feature_dim: int = 64, latent_dim: int = 32):
        self.deep_svdd = DeepSVDD(seq_len, feature_dim, latent_dim)
        self.isolation_forest = None  # Fitted during training
        self.seq_len = seq_len
        self.feature_dim = feature_dim

    def fit_isolation_forest(self, X_normal: np.ndarray, contamination: float = 0.01):
        """
        Fit Isolation Forest on normal training data.

        Parameters
        ----------
        X_normal : np.ndarray, shape (N, seq_len * feature_dim) or (N, feature_dim)
        """
        from sklearn.ensemble import IsolationForest

        self.isolation_forest = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            max_samples="auto",
            random_state=42,
            n_jobs=-1,
        )
        self.isolation_forest.fit(X_normal)
        logger.info(f"Isolation Forest fitted on {len(X_normal)} normal samples")

    def get_isolation_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores from Isolation Forest.
        Returns scores in [0, 1] where higher = more anomalous.
        """
        if self.isolation_forest is None:
            return np.zeros(len(X))

        # score_samples returns negative for anomalies
        raw = -self.isolation_forest.score_samples(X)
        # Normalize to [0, 1]
        if raw.max() > raw.min():
            return (raw - raw.min()) / (raw.max() - raw.min())
        return np.zeros(len(X))


# ═════════════════════════════════════════════════════════════════
# Training functions
# ═════════════════════════════════════════════════════════════════

def train_deep_svdd(
    model: DeepSVDD,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-3,
    save_dir: str = "./models",
) -> Dict[str, list]:
    """
    Train Deep SVDD: minimize distance from center for normal data.

    Returns training history dict.
    """
    from .utils import EarlyStopping

    model = model.to(device)

    # Initialize center from first pass through training data
    if not model.center_initialized:
        model.init_center(train_loader, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5, min_lr=1e-6,
    )

    os.makedirs(save_dir, exist_ok=True)
    early_stop = EarlyStopping(
        patience=12, mode="min",
        save_path=os.path.join(save_dir, "autoencoder_best.pt"),
    )

    history = {"train_loss": [], "val_loss": []}

    print(f"\n{'█' * 50}")
    print(f"█  DEEP SVDD TRAINING (One-Class Learning)")
    print(f"█  Device: {device}")
    print(f"{'█' * 50}\n")

    for epoch in range(1, epochs + 1):
        # ── Train ────────────────────────────────────────
        model.train()
        train_loss = 0.0
        n_train = 0

        for x_batch, _ in train_loader:
            x_batch = x_batch.to(device)
            z = model(x_batch)
            # SVDD loss: minimize distance to center
            loss = torch.mean(torch.sum((z - model.center) ** 2, dim=1))

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
                z = model(x_batch)
                loss = torch.mean(torch.sum((z - model.center) ** 2, dim=1))
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
            f"lr={current_lr:.2e}",
            flush=True
        )

        if early_stop(val_loss, model):
            print(f"  ⏹ Early stopping at epoch {epoch}")
            break

    # Reload best checkpoint
    best_path = os.path.join(save_dir, "autoencoder_best.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))

    return history


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
            f"lr={current_lr:.2e}",
            flush=True
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
    model,
    test_loader,
    device: torch.device,
    threshold_percentile: float = 95.0,
    isolation_forest=None,
) -> Dict[str, float]:
    """
    Evaluate anomaly detector on the test set.

    Works with both DeepSVDD and EEGAutoencoder models.
    Optionally combines scores with Isolation Forest.

    The threshold for seizure detection is set at the given percentile
    of anomaly scores on the test set.

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

    # Combine with Isolation Forest if available
    if isolation_forest is not None:
        # Get IF scores on the flattened test data
        flat_data = []
        with torch.no_grad():
            for x_batch, _ in test_loader:
                B = x_batch.size(0)
                flat_data.append(x_batch.reshape(B, -1).numpy())
        flat_data = np.concatenate(flat_data)
        if_scores = isolation_forest.get_isolation_scores(flat_data)

        # Normalize SVDD/AE errors to [0, 1]
        if all_errors.max() > all_errors.min():
            norm_errors = (all_errors - all_errors.min()) / (all_errors.max() - all_errors.min())
        else:
            norm_errors = np.zeros_like(all_errors)

        # Average both scores
        all_errors = 0.6 * norm_errors + 0.4 * if_scores

    # Set threshold at the percentile of anomaly scores
    threshold = np.percentile(all_errors, threshold_percentile)
    y_pred = (all_errors > threshold).astype(int)
    y_prob = all_errors / (all_errors.max() + 1e-8)  # normalize to [0, 1]

    metrics = compute_metrics(all_labels, y_pred, y_prob)
    metrics["threshold"] = float(threshold)
    metrics["mean_error_normal"] = float(all_errors[all_labels == 0].mean()) if (all_labels == 0).any() else 0.0
    metrics["mean_error_seizure"] = float(all_errors[all_labels == 1].mean()) if (all_labels == 1).any() else 0.0

    return metrics
