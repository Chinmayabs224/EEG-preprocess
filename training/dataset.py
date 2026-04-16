"""
Dataset loader for downstream model training.

Loads .npz files produced by spike_pipeline, handles class imbalance,
and provides PyTorch DataLoaders for training/validation/test splits.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════
# Dataset
# ═════════════════════════════════════════════════════════════════

class EEGSpikeDataset(Dataset):
    """
    PyTorch dataset that loads SNN features and labels from .npz files.

    Each .npz file contains:
        - ``snn_features``: shape ``(n_windows, 64)`` — spike-rate features
        - ``snn_labels``: shape ``(n_windows,)`` — per-window binary labels

    For the Autoencoder, only non-seizure samples are used during training.
    For LSTM/CNN/Transformer, all samples are used with class weights.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        seq_len: int = 30,
        mode: str = "classification",
    ):
        """
        Parameters
        ----------
        features : np.ndarray, shape ``(N, 64)``
        labels : np.ndarray, shape ``(N,)``
        seq_len : int
            Number of consecutive windows to group into one sequence
            for LSTM/Transformer input.
        mode : str
            ``"classification"`` — return (sequence, label) pairs.
            ``"autoencoder"`` — return (sequence, sequence) for reconstruction.
        """
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.seq_len = seq_len
        self.mode = mode

        # Instead of allocating a massive dense array of shape (M, seq_len, 64),
        # we will slice `self.features` dynamically in `__getitem__`. This prevents OOM
        # errors on 2-core CPU machines.
        n = len(self.features)
        self.num_sequences = max(0, n - seq_len + 1)
        
        # We need the labels array for the WeightedRandomSampler
        if self.num_sequences > 0:
            self.seq_labels = self.labels[self.seq_len - 1 : n]
        else:
            self.seq_labels = np.array([], dtype=np.int64)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        # Lazy sequence slicing
        x = torch.tensor(self.features[idx : idx + self.seq_len], dtype=torch.float32)
        y = torch.tensor(self.seq_labels[idx], dtype=torch.long)

        if self.mode == "autoencoder":
            return x, x  # reconstruct input
        return x, y


class EEGSingleWindowDataset(Dataset):
    """
    Flat dataset: each item is a single window (1, 64) with its label.
    Used for CNN which treats each window independently.
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ═════════════════════════════════════════════════════════════════
# Loading from disk
# ═════════════════════════════════════════════════════════════════

def load_all_npz(
    spikes_dir: str,
    max_files: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Recursively load all .npz files and concatenate SNN features + labels.

    Parameters
    ----------
    spikes_dir : str
        Root directory containing subject subdirs with .npz files.
    max_files : int, optional
        Limit number of files (for debugging).

    Returns
    -------
    features : np.ndarray, shape ``(N_total, 64)``
    labels : np.ndarray, shape ``(N_total,)``
    file_boundaries : list of int
        Cumulative index boundaries per file (for subject-aware splitting).
    """
    spikes_path = Path(spikes_dir)
    all_features = []
    all_labels = []
    file_boundaries = [0]
    loaded = 0

    npz_files = sorted(spikes_path.rglob("*.npz"))
    logger.info(f"Found {len(npz_files)} .npz files in {spikes_dir}")

    for npz_file in npz_files:
        if max_files and loaded >= max_files:
            break

        try:
            data = np.load(npz_file, allow_pickle=True)

            # Prefer snn_features + snn_labels (window-level)
            if "snn_features" in data and "snn_labels" in data:
                feats = data["snn_features"]
                labs = data["snn_labels"]
            else:
                logger.warning(f"  Skipping {npz_file.name}: missing snn_features/snn_labels")
                continue

            # Validate shapes
            if feats.ndim != 2 or labs.ndim != 1:
                logger.warning(f"  Skipping {npz_file.name}: unexpected shapes {feats.shape}, {labs.shape}")
                continue

            if len(feats) != len(labs):
                logger.warning(f"  Skipping {npz_file.name}: shape mismatch feats={feats.shape} labs={labs.shape}")
                continue

            if len(feats) == 0:
                continue

            all_features.append(feats)
            all_labels.append(labs)
            file_boundaries.append(file_boundaries[-1] + len(feats))
            loaded += 1

        except Exception as e:
            logger.warning(f"  Error loading {npz_file.name}: {e}")
            continue

    if not all_features:
        raise ValueError(f"No valid .npz files found in {spikes_dir}")

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    n_seizure = int(labels.sum())
    n_normal = len(labels) - n_seizure
    logger.info(
        f"Loaded {loaded} files → {len(features)} windows "
        f"(seizure={n_seizure}, normal={n_normal}, "
        f"ratio={n_seizure / max(len(labels), 1):.2%})"
    )

    return features, labels, file_boundaries


# ═════════════════════════════════════════════════════════════════
# Train/Val/Test split
# ═════════════════════════════════════════════════════════════════

def split_data(
    features: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Stratified train/val/test split.

    Returns dict with keys "train", "val", "test", each mapping to
    ``(features, labels)`` tuples.
    """
    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        features, labels, test_size=test_size,
        random_state=random_state, stratify=labels,
    )

    # Second split: train vs val
    relative_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=relative_val,
        random_state=random_state, stratify=y_trainval,
    )

    logger.info(
        f"Split: train={len(X_train)} val={len(X_val)} test={len(X_test)} "
        f"(seizure: train={y_train.sum()}, val={y_val.sum()}, test={y_test.sum()})"
    )

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }


# ═════════════════════════════════════════════════════════════════
# DataLoader factory with class-imbalance handling
# ═════════════════════════════════════════════════════════════════

def get_dataloaders(
    splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
    seq_len: int = 30,
    batch_size: int = 64,
    mode: str = "classification",
    use_weighted_sampler: bool = True,
) -> Dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders with optional WeightedRandomSampler
    to combat class imbalance.
    """
    loaders = {}

    for split_name, (feats, labs) in splits.items():
        if mode == "cnn":
            dataset = EEGSingleWindowDataset(feats, labs)
        else:
            dataset = EEGSpikeDataset(feats, labs, seq_len=seq_len, mode=mode)

        sampler = None
        shuffle = (split_name == "train")

        if split_name == "train" and use_weighted_sampler and mode != "autoencoder":
            # Compute class weights for balanced sampling
            if mode == "cnn":
                sample_labels = labs
            else:
                sample_labels = dataset.seq_labels

            class_counts = np.bincount(sample_labels, minlength=2)
            # Avoid division by zero
            class_counts = np.maximum(class_counts, 1)
            class_weights = 1.0 / class_counts.astype(np.float64)
            sample_weights = class_weights[sample_labels]
            sampler = WeightedRandomSampler(
                weights=torch.tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=True,
            )
            shuffle = False  # sampler handles shuffling

        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=0,
            pin_memory=True,
            drop_last=(split_name == "train"),
        )

    return loaders


def get_autoencoder_dataloaders(
    splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
    seq_len: int = 30,
    batch_size: int = 64,
) -> Dict[str, DataLoader]:
    """
    DataLoaders for Autoencoder training.
    Training set uses ONLY normal (non-seizure) windows.
    Val/Test sets use ALL windows for anomaly scoring.
    """
    # Filter training data to only include non-seizure
    train_feats, train_labs = splits["train"]
    normal_mask = (train_labs == 0)
    train_normal_feats = train_feats[normal_mask]
    train_normal_labs = train_labs[normal_mask]

    logger.info(
        f"Autoencoder training: using {len(train_normal_feats)} normal windows "
        f"(discarded {(~normal_mask).sum()} seizure windows from training)"
    )

    modified_splits = {
        "train": (train_normal_feats, train_normal_labs),
        "val": splits["val"],
        "test": splits["test"],
    }

    return get_dataloaders(
        modified_splits, seq_len=seq_len, batch_size=batch_size,
        mode="autoencoder", use_weighted_sampler=False,
    )
