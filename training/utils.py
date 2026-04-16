"""
Shared training utilities: metrics, early stopping, logging, model saving.
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from typing import Dict, Optional
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════
# Metrics
# ═════════════════════════════════════════════════════════════════

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    For seizure detection, we prioritize Recall (sensitivity) and F1
    over raw accuracy due to extreme class imbalance.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "specificity": 0.0,
    }

    # Specificity = TN / (TN + FP)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["specificity"] = tn / max(tn + fp, 1)
        metrics["true_positives"] = int(tp)
        metrics["false_positives"] = int(fp)
        metrics["true_negatives"] = int(tn)
        metrics["false_negatives"] = int(fn)

    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["auc_roc"] = 0.0

    return metrics


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """Pretty-print metrics."""
    print(f"\n{'═' * 50}")
    print(f"  {prefix} Evaluation Results")
    print(f"{'═' * 50}")
    print(f"  Accuracy    : {metrics.get('accuracy', 0):.4f}")
    print(f"  Precision   : {metrics.get('precision', 0):.4f}")
    print(f"  Recall      : {metrics.get('recall', 0):.4f}  ← (Seizure Sensitivity)")
    print(f"  F1 Score    : {metrics.get('f1', 0):.4f}  ← (Primary Metric)")
    print(f"  Specificity : {metrics.get('specificity', 0):.4f}")
    if "auc_roc" in metrics:
        print(f"  AUC-ROC     : {metrics.get('auc_roc', 0):.4f}")
    print(f"{'═' * 50}\n")


# ═════════════════════════════════════════════════════════════════
# Early Stopping
# ═════════════════════════════════════════════════════════════════

class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Monitors a metric (e.g., val_loss or val_f1) and stops training
    if no improvement is seen for ``patience`` epochs.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min",
        save_path: Optional[str] = None,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.save_path = save_path

        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Returns True if should stop.
        """
        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(model)
            return False

        if self.mode == "min":
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
            self._save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False

    def _save_checkpoint(self, model: nn.Module):
        if self.save_path:
            torch.save(model.state_dict(), self.save_path)


# ═════════════════════════════════════════════════════════════════
# Weighted Loss for Class Imbalance
# ═════════════════════════════════════════════════════════════════

def get_class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for CrossEntropyLoss.
    
    For CHB-MIT: ~95% non-seizure, ~5% seizure → seizure weight ≈ 19x.
    """
    counts = np.bincount(labels.astype(int), minlength=2)
    counts = np.maximum(counts, 1)  # avoid division by zero
    total = counts.sum()
    weights = total / (len(counts) * counts)
    weights = weights / weights.sum() * len(counts)  # normalize
    logger.info(f"Class weights: normal={weights[0]:.2f}, seizure={weights[1]:.2f}")
    return torch.tensor(weights, dtype=torch.float32).to(device)


# ═════════════════════════════════════════════════════════════════
# Training History Plotting
# ═════════════════════════════════════════════════════════════════

def plot_training_history(
    history: Dict[str, list],
    save_path: str,
    model_name: str = "Model",
):
    """
    Plot training curves: loss, F1, accuracy over epochs.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{model_name} Training History", fontsize=16, fontweight="bold")

    # Loss
    axes[0].plot(history["train_loss"], label="Train Loss", linewidth=2)
    axes[0].plot(history["val_loss"], label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # F1 Score
    if "train_f1" in history:
        axes[1].plot(history["train_f1"], label="Train F1", linewidth=2)
        axes[1].plot(history["val_f1"], label="Val F1", linewidth=2)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("F1 Score")
        axes[1].set_title("F1 Score (Primary Metric)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    # AUC-ROC
    if "val_auc" in history:
        axes[2].plot(history["val_auc"], label="Val AUC-ROC", linewidth=2, color="green")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("AUC-ROC")
        axes[2].set_title("AUC-ROC")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved training plot: {save_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    model_name: str = "Model",
):
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.colorbar(im)

    labels = ["Normal", "Seizure"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"{model_name} — Confusion Matrix", fontsize=14, fontweight="bold")

    # Add text annotations
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=16, fontweight="bold", color=color)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved confusion matrix: {save_path}")


# ═════════════════════════════════════════════════════════════════
# Model Saving / Loading Utils
# ═════════════════════════════════════════════════════════════════

def save_model_and_metrics(
    model: nn.Module,
    metrics: Dict[str, float],
    model_name: str,
    save_dir: str,
):
    """Save model weights and metrics as JSON."""
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, f"{model_name}_best.pt")
    torch.save(model.state_dict(), model_path)

    metrics_path = os.path.join(save_dir, f"{model_name}_metrics.json")
    # Convert numpy types to Python types for JSON
    clean_metrics = {k: float(v) if isinstance(v, (np.floating, float)) else int(v)
                     for k, v in metrics.items()}
    with open(metrics_path, "w") as f:
        json.dump(clean_metrics, f, indent=2)

    logger.info(f"Saved {model_name}: {model_path}")
    logger.info(f"Saved metrics: {metrics_path}")
