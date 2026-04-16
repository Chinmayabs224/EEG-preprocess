"""
Model 3: 1D CNN for Seizure Classification.

Treats each spike-rate feature window as a 1D signal and applies
convolutional layers to detect local spike patterns. Works on
individual windows (no sequence dependency).

Architecture::

    Input (64,) → reshape → Conv1D(64,32,k=3) → Conv1D(32,64,k=3)
                → Conv1D(64,128,k=3) → AdaptivePool → FC(64) → FC(2)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class EEGCNNClassifier(nn.Module):
    """
    1D Convolutional Neural Network for EEG spike feature classification.

    Treats the 64-dimensional spike-rate vector as a 1D signal
    with 1 channel and applies 1D convolutions.

    Parameters
    ----------
    input_dim : int
        Feature dimension (default 64).
    num_classes : int
        Output classes (default 2).
    dropout : float
        Dropout rate (default 0.4).
    """

    def __init__(
        self,
        input_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.input_dim = input_dim

        # Conv blocks: treat 64-dim feature as a 1D signal of length 64
        self.conv_blocks = nn.Sequential(
            # Block 1
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),

            # Block 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout * 0.5),

            # Block 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            # Block 4
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(4),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),

            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, input_dim) — single window of spike features

        Returns
        -------
        logits : (B, num_classes)
        """
        # Reshape: (B, 64) → (B, 1, 64) for 1D conv
        x = x.unsqueeze(1)  # (B, 1, input_dim)
        features = self.conv_blocks(x)
        logits = self.classifier(features)
        return logits


def train_cnn(
    model: EEGCNNClassifier,
    train_loader,
    val_loader,
    device: torch.device,
    class_weights: torch.Tensor,
    epochs: int = 50,
    lr: float = 1e-3,
    save_dir: str = "./models",
) -> Dict[str, list]:
    """
    Train the CNN classifier with weighted loss.

    Returns training history dict.
    """
    from .utils import EarlyStopping, compute_metrics

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    os.makedirs(save_dir, exist_ok=True)
    early_stop = EarlyStopping(
        patience=15, mode="max",
        save_path=os.path.join(save_dir, "cnn_best.pt"),
    )

    history = {
        "train_loss": [], "val_loss": [],
        "train_f1": [], "val_f1": [],
        "val_auc": [],
    }

    print(f"\n{'█' * 50}")
    print(f"█  CNN CLASSIFIER TRAINING")
    print(f"█  Device: {device}")
    print(f"{'█' * 50}\n")

    for epoch in range(1, epochs + 1):
        # ── Train ────────────────────────────────────────
        model.train()
        train_loss = 0.0
        n_train = 0
        all_preds, all_true = [], []

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * x_batch.size(0)
            n_train += x_batch.size(0)

            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(y_batch.cpu().numpy())

        train_loss /= max(n_train, 1)
        train_metrics = compute_metrics(np.array(all_true), np.array(all_preds))

        # ── Validate ─────────────────────────────────────
        model.eval()
        val_loss = 0.0
        n_val = 0
        val_preds, val_true, val_probs = [], [], []

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * x_batch.size(0)
                n_val += x_batch.size(0)

                probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds = logits.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(y_batch.cpu().numpy())
                val_probs.extend(probs)

        val_loss /= max(n_val, 1)
        val_metrics = compute_metrics(
            np.array(val_true), np.array(val_preds), np.array(val_probs)
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_metrics["f1"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_auc"].append(val_metrics.get("auc_roc", 0))

        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{epochs} │ "
            f"loss={train_loss:.4f}/{val_loss:.4f} │ "
            f"F1={train_metrics['f1']:.3f}/{val_metrics['f1']:.3f} │ "
            f"AUC={val_metrics.get('auc_roc', 0):.3f}"
        )

        if early_stop(val_metrics["f1"], model):
            print(f"  ⏹ Early stopping at epoch {epoch}")
            break

    # Reload best
    best_path = os.path.join(save_dir, "cnn_best.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))

    return history


def evaluate_cnn(
    model: EEGCNNClassifier,
    test_loader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate CNN on test set. Returns metrics dict."""
    from .utils import compute_metrics

    model = model.to(device)
    model.eval()

    all_preds, all_true, all_probs = [], [], []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_true.extend(y_batch.numpy())
            all_probs.extend(probs)

    return compute_metrics(np.array(all_true), np.array(all_preds), np.array(all_probs))
