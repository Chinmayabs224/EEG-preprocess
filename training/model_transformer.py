"""
Model 4: Transformer for Sequential Seizure Classification.

Uses multi-head self-attention to capture long-range temporal dependencies
in the SNN spike-rate feature sequences. Positional encoding ensures
the model understands temporal ordering.

Architecture::

    Input (seq_len, 64) → Positional Encoding → TransformerEncoder(4 layers)
                        → CLS token pooling → FC(64) → FC(2)
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence position awareness."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class EEGTransformerClassifier(nn.Module):
    """
    Transformer Encoder for EEG seizure classification.

    Uses a learnable [CLS] token prepended to the sequence.
    The output of the [CLS] token after transformer encoding is used
    for classification.

    Parameters
    ----------
    input_dim : int
        Feature dimension per timestep (default 64).
    d_model : int
        Transformer embedding dimension (default 128).
    nhead : int
        Number of attention heads (default 4).
    num_layers : int
        Number of transformer encoder layers (default 4).
    dim_feedforward : int
        FFN hidden dimension (default 256).
    num_classes : int
        Output classes (default 2).
    dropout : float
        Dropout rate (default 0.3).
    max_seq_len : int
        Maximum sequence length (default 200).
    """

    def __init__(
        self,
        input_dim: int = 64,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        num_classes: int = 2,
        dropout: float = 0.3,
        max_seq_len: int = 200,
    ):
        super().__init__()
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional encoding (max_len = seq_len + 1 for CLS)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len + 1, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for better training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
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
        B = x.size(0)

        # Project input
        x = self.input_proj(x)  # (B, T, d_model)

        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)          # (B, T+1, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer(x)  # (B, T+1, d_model)

        # Use [CLS] token output for classification
        cls_output = x[:, 0, :]  # (B, d_model)
        logits = self.classifier(cls_output)
        return logits


def train_transformer(
    model: EEGTransformerClassifier,
    train_loader,
    val_loader,
    device: torch.device,
    class_weights: torch.Tensor,
    epochs: int = 50,
    lr: float = 5e-4,
    save_dir: str = "./models",
) -> Dict[str, list]:
    """
    Train the Transformer classifier with weighted loss.

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
        save_path=os.path.join(save_dir, "transformer_best.pt"),
    )

    history = {
        "train_loss": [], "val_loss": [],
        "train_f1": [], "val_f1": [],
        "val_auc": [],
    }

    print(f"\n{'█' * 50}")
    print(f"█  TRANSFORMER CLASSIFIER TRAINING")
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
    best_path = os.path.join(save_dir, "transformer_best.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))

    return history


def evaluate_transformer(
    model: EEGTransformerClassifier,
    test_loader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate Transformer on test set. Returns metrics dict."""
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
