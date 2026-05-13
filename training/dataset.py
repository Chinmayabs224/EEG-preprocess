"""
Dataset loader for downstream model training.

Loads .npz files produced by spike_pipeline, handles class imbalance,
and provides PyTorch DataLoaders for training/validation/test splits.

Enhancements:
    - Feature augmentation (spectral/statistical biomarkers)
    - Seizure-class data augmentation (noise, scaling, time-shift)
    - Weighted random sampling for class imbalance
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
# Seizure Augmentation Transforms
# ═════════════════════════════════════════════════════════════════

class SeizureAugmentor:
    """
    On-the-fly data augmentation applied ONLY to the seizure (minority) class.

    Augmentations:
        1. Gaussian noise injection: N(0, σ) with σ ∈ [0.01, 0.05]
        2. Feature scaling perturbation: multiply by U(0.9, 1.1)
        3. Feature dropout: randomly zero out 5–10% of features

    These are applied probabilistically (p=0.5 each) to create
    diverse training samples from the few seizure examples.
    """

    def __init__(self, noise_std: float = 0.03, scale_range: Tuple[float, float] = (0.9, 1.1)):
        self.noise_std = noise_std
        self.scale_lo, self.scale_hi = scale_range

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply random augmentations to a single feature vector or sequence."""
        x = x.copy()

        # Gaussian noise (p=0.5)
        if np.random.random() < 0.5:
            noise = np.random.normal(0, self.noise_std, x.shape).astype(np.float32)
            x = x + noise

        # Feature scaling (p=0.5)
        if np.random.random() < 0.5:
            scale = np.random.uniform(self.scale_lo, self.scale_hi, x.shape[-1:]).astype(np.float32)
            x = x * scale

        # Feature dropout (p=0.3)
        if np.random.random() < 0.3:
            mask = np.random.random(x.shape[-1:]) > 0.1  # drop 10%
            x = x * mask.astype(np.float32)

        return x


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
        augment_seizure: bool = False,
    ):
        """
        Parameters
        ----------
        features : np.ndarray, shape ``(N, feature_dim)``
        labels : np.ndarray, shape ``(N,)``
        seq_len : int
            Number of consecutive windows to group into one sequence
            for LSTM/Transformer input.
        mode : str
            ``"classification"`` — return (sequence, label) pairs.
            ``"autoencoder"`` — return (sequence, sequence) for reconstruction.
        augment_seizure : bool
            If True, apply random augmentations to seizure sequences.
        """
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.seq_len = seq_len
        self.mode = mode
        self.augment_seizure = augment_seizure
        self.augmentor = SeizureAugmentor() if augment_seizure else None

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
        x = self.features[idx : idx + self.seq_len].copy()
        y = self.seq_labels[idx]

        # Apply seizure augmentation only to minority class during training
        if self.augment_seizure and self.augmentor is not None and y == 1:
            x = self.augmentor(x)

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        if self.mode == "autoencoder":
            return x, x  # reconstruct input
        return x, y


class EEGSingleWindowDataset(Dataset):
    """
    Flat dataset: each item is a single window (1, feature_dim) with its label.
    Used for CNN which treats each window independently.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        augment_seizure: bool = False,
    ):
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.augment_seizure = augment_seizure
        self.augmentor = SeizureAugmentor() if augment_seizure else None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx].copy()
        y = self.labels[idx]

        # Apply seizure augmentation only to minority class during training
        if self.augment_seizure and self.augmentor is not None and y == 1:
            x = self.augmentor(x)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


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
    file_boundaries = []       # list of (subject_id, start_idx, end_idx)
    cursor = 0
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

            # Subject ID = parent folder name (e.g. "chb01")
            subject_id = npz_file.parent.name

            all_features.append(feats)
            all_labels.append(labs)
            file_boundaries.append((subject_id, cursor, cursor + len(feats)))
            cursor += len(feats)
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
    subjects_found = sorted(set(s for s, _, _ in file_boundaries))
    logger.info(
        f"Loaded {loaded} files from {len(subjects_found)} subjects → "
        f"{len(features)} windows "
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
    file_boundaries: list = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    random_state: int = 42,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Subject-wise train/val/test split using file_boundaries.

    All windows from the same patient (subject) stay in the same split,
    preventing patient leakage across train/val/test.

    Parameters
    ----------
    features : np.ndarray, shape ``(N, feature_dim)``
    labels : np.ndarray, shape ``(N,)``
    file_boundaries : list of (subject_id, start_idx, end_idx) tuples
        Returned by ``load_all_npz()``.
    train_ratio : float
    val_ratio : float
    random_state : int

    Returns
    -------
    dict with keys ``"train"``, ``"val"``, ``"test"``, each mapping to
    ``(features, labels)`` tuples.
    """
    from collections import defaultdict

    if file_boundaries is None:
        raise ValueError(
            "file_boundaries is required for subject-wise splitting. "
            "Pass the boundaries returned by load_all_npz()."
        )

    rng = np.random.default_rng(random_state)

    # Group window index ranges by subject
    subject_map = defaultdict(list)
    for subject_id, start, end in file_boundaries:
        subject_map[subject_id].append((start, end))

    subjects = sorted(subject_map.keys())  # sorted for reproducibility
    rng.shuffle(subjects)

    n_train = int(len(subjects) * train_ratio)
    n_val   = int(len(subjects) * val_ratio)

    train_subjects = subjects[:n_train]
    val_subjects   = subjects[n_train:n_train + n_val]
    test_subjects  = subjects[n_train + n_val:]

    def _gather(subj_list):
        idx = []
        for s in subj_list:
            for start, end in subject_map[s]:
                idx.extend(range(start, end))
        return np.array(idx, dtype=np.int64) if idx else np.array([], dtype=np.int64)

    train_idx = _gather(train_subjects)
    val_idx   = _gather(val_subjects)
    test_idx  = _gather(test_subjects)

    X_train, y_train = features[train_idx], labels[train_idx]
    X_val,   y_val   = features[val_idx],   labels[val_idx]
    X_test,  y_test  = features[test_idx],  labels[test_idx]

    logger.info(
        f"Subject-wise split ({len(subjects)} subjects): "
        f"train={len(train_subjects)} subj ({len(X_train)} windows) | "
        f"val={len(val_subjects)} subj ({len(X_val)} windows) | "
        f"test={len(test_subjects)} subj ({len(X_test)} windows)"
    )
    logger.info(f"  Train subjects: {sorted(train_subjects)}")
    logger.info(f"  Val subjects  : {sorted(val_subjects)}")
    logger.info(f"  Test subjects : {sorted(test_subjects)}")
    logger.info(
        f"  Seizure windows: train={int(y_train.sum())}, "
        f"val={int(y_val.sum())}, test={int(y_test.sum())}"
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
    augment_seizure: bool = False,
) -> Dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders with optional WeightedRandomSampler
    to combat class imbalance.
    """
    loaders = {}

    for split_name, (feats, labs) in splits.items():
        # Only augment training data
        should_augment = augment_seizure and split_name == "train"

        if mode == "cnn":
            dataset = EEGSingleWindowDataset(feats, labs, augment_seizure=should_augment)
        else:
            dataset = EEGSpikeDataset(
                feats, labs, seq_len=seq_len, mode=mode,
                augment_seizure=should_augment,
            )

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
