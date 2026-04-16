"""
Model 2: Bidirectional LSTM for Sequential Seizure Classification.

Processes temporal sequences of SNN spike-rate features and classifies
each sequence as seizure / non-seizure. Uses bidirectional LSTM with
attention mechanism to capture both forward and backward temporal context.

Architecture::

    Input (seq_len, 64) → BiLSTM(128) → Attention → FC(64) → FC(2)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class Attention(nn.Module):
    """Simple additive attention over LSTM hidden states."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        lstm_output : (B, T, hidden_dim)

        Returns
        -------
        context : (B, hidden_dim)
        """
        weights = torch.softmax(self.attn(lstm_output), dim=1)  # (B, T, 1)
        context = (weights * lstm_output).sum(dim=1)             # (B, hidden_dim)
        return context


class EEGLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM with Attention for EEG seizure classification.

    Parameters
    ----------
    input_dim : int
        Feature dimension per timestep (default 64).
    hidden_dim : int
        LSTM hidden size (default 128).
    num_layers : int
        Number of stacked LSTM layers (default 2).
    num_classes : int
        Output classes (default 2: normal/seizure).
    dropout : float
        Dropout rate (default 0.4).
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Attention
        self.attention = Attention(hidden_dim * 2)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),

            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, seq_len, input_dim)

        Returns
        -------
        logits : (B, num_classes)
        """
        # Project input
        x = self.input_proj(x)  # (B, T, hidden_dim)

        # BiLSTM
        lstm_out, _ = self.lstm(x)  # (B, T, hidden*2)

        # Attention pooling
        context = self.attention(lstm_out)  # (B, hidden*2)

        # Classify
        logits = self.classifier(context)
        return logits


def train_lstm(
    model: EEGLSTMClassifier,
    train_loader,
    val_loader,
    device: torch.device,
    class_weights: torch.Tensor,
    epochs: int = 50,
    lr: float = 1e-3,
    save_dir: str = "./models",
) -> Dict[str, list]:
    """
    Train the BiLSTM classifier with weighted loss.
    
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
        save_path=os.path.join(save_dir, "lstm_best.pt"),
    )

    history = {
        "train_loss": [], "val_loss": [],
        "train_f1": [], "val_f1": [],
        "val_auc": [],
    }

    print(f"\n{'█' * 50}")
    print(f"█  LSTM CLASSIFIER TRAINING")
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
    best_path = os.path.join(save_dir, "lstm_best.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))

    return history


def evaluate_lstm(
    model: EEGLSTMClassifier,
    test_loader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate LSTM on test set. Returns metrics dict."""
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
