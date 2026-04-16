"""
dataset.py
==========
Loads all processed .npz spike files produced by spike_pipeline/.

Each .npz file contains:
  • spike_features / snn_features : (T, 64)  float32
  • has_seizure                   : bool / int
  • binary_labels                 : (T,)  int  (per-timestep seizure flag)

This module provides:
  • EEGSpikeDataset  – PyTorch Dataset wrapping all .npz files
  • build_dataloaders – train/val/test split with weighted sampler
"""

from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _load_npz(path: Path) -> Tuple[np.ndarray, int]:
    """
    Load one .npz file and return (features, label).
    Features shape: (T, 64).  label: 0 or 1.
    """
    data = np.load(path, allow_pickle=True)

    # Try both key names the pipeline may have written
    if "snn_features" in data:
        feats = data["snn_features"].astype(np.float32)
    elif "spike_features" in data:
        feats = data["spike_features"].astype(np.float32)
    else:
        # Fall back: look for any 2-D float array
        feats = None
        for k in data.files:
            arr = data[k]
            if arr.ndim == 2 and arr.shape[1] == 64:
                feats = arr.astype(np.float32)
                break
        if feats is None:
            raise KeyError(f"Cannot find feature array in {path}. Keys: {data.files}")

    label = int(bool(data["has_seizure"])) if "has_seizure" in data else 0
    return feats, label


# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────

class EEGSpikeDataset(Dataset):
    """
    Wraps all .npz files in a directory tree.

    Each item is (features, label) where:
      features : torch.Tensor shape (T, 64)
      label    : torch.LongTensor  scalar  (0=normal, 1=seizure)

    Parameters
    ----------
    spike_dir   : Root directory that contains chbXX/ subdirectories
    min_t       : Minimum number of time-steps; shorter chunks are skipped
    max_t       : If set, truncate features to this many time-steps
    pad_to      : If set, pad / truncate every sample to exactly this T
                  (required when mixing files of different durations)
    transform   : Optional callable applied to the feature tensor
    """

    def __init__(
        self,
        spike_dir: str | Path,
        min_t: int = 10,
        max_t: Optional[int] = None,
        pad_to: Optional[int] = None,
        transform=None,
    ):
        self.spike_dir = Path(spike_dir)
        self.min_t = min_t
        self.max_t = max_t
        self.pad_to = pad_to
        self.transform = transform

        self.paths: List[Path] = []
        self.labels: List[int] = []

        self._discover()

    def _discover(self):
        all_npz = sorted(self.spike_dir.rglob("*.npz"))
        logger.info(f"Discovered {len(all_npz)} .npz files in {self.spike_dir}")

        skipped = 0
        for p in all_npz:
            try:
                feats, label = _load_npz(p)
                if feats.shape[0] < self.min_t:
                    skipped += 1
                    continue
                self.paths.append(p)
                self.labels.append(label)
            except Exception as e:
                logger.warning(f"Skipping {p.name}: {e}")
                skipped += 1

        n_seizure = sum(self.labels)
        n_normal  = len(self.labels) - n_seizure
        logger.info(
            f"Dataset ready: {len(self.paths)} samples "
            f"({n_seizure} seizure / {n_normal} normal, skipped {skipped})"
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        feats, label = _load_npz(self.paths[idx])

        # ------ truncate / pad to fixed length ------
        if self.max_t is not None:
            feats = feats[: self.max_t]

        if self.pad_to is not None:
            T_cur = feats.shape[0]
            if T_cur < self.pad_to:
                pad = np.zeros((self.pad_to - T_cur, feats.shape[1]), dtype=np.float32)
                feats = np.concatenate([feats, pad], axis=0)
            else:
                feats = feats[: self.pad_to]

        x = torch.from_numpy(feats)          # (T, 64)
        y = torch.tensor(label, dtype=torch.long)

        if self.transform:
            x = self.transform(x)

        return x, y

    # ---- convenience properties ----
    @property
    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights for weighted loss / sampler."""
        labels = np.array(self.labels)
        counts = np.bincount(labels, minlength=2).astype(float)
        counts = np.maximum(counts, 1)        # avoid division by zero
        weights = 1.0 / counts
        weights /= weights.sum()
        return torch.tensor(weights, dtype=torch.float32)

    @property
    def sample_weights(self) -> torch.Tensor:
        """Per-sample weight for WeightedRandomSampler."""
        cw = self.class_weights
        return torch.tensor([cw[l].item() for l in self.labels], dtype=torch.float32)


# ──────────────────────────────────────────────────────────────
# DataLoader builder
# ──────────────────────────────────────────────────────────────

def build_dataloaders(
    spike_dir: str | Path,
    batch_size: int = 32,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    num_workers: int = 0,
    pad_to: Optional[int] = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, EEGSpikeDataset]:
    """
    Build train / val / test DataLoaders from all .npz files.

    Train loader uses WeightedRandomSampler to compensate for the
    severe class imbalance (~95% non-seizure).

    Returns
    -------
    (train_loader, val_loader, test_loader, full_dataset)
    """
    rng = np.random.default_rng(seed)

    full_ds = EEGSpikeDataset(spike_dir=spike_dir, pad_to=pad_to)
    n = len(full_ds)

    # Stratified split (preserve seizure ratio in every split)
    indices = np.arange(n)
    labels  = np.array(full_ds.labels)

    normal_idx  = indices[labels == 0]
    seizure_idx = indices[labels == 1]

    def _split(arr):
        rng.shuffle(arr)
        n1 = int(len(arr) * train_ratio)
        n2 = int(len(arr) * val_ratio)
        return arr[:n1], arr[n1:n1+n2], arr[n1+n2:]

    tr_n, va_n, te_n = _split(normal_idx)
    tr_s, va_s, te_s = _split(seizure_idx)

    train_idx = np.concatenate([tr_n, tr_s])
    val_idx   = np.concatenate([va_n, va_s])
    test_idx  = np.concatenate([te_n, te_s])

    from torch.utils.data import Subset

    train_ds = Subset(full_ds, train_idx.tolist())
    val_ds   = Subset(full_ds, val_idx.tolist())
    test_ds  = Subset(full_ds, test_idx.tolist())

    # Weighted sampler for training (handles imbalance)
    train_sample_weights = full_ds.sample_weights[train_idx]
    sampler = WeightedRandomSampler(
        weights=train_sample_weights,
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=sampler, num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(
        f"Splits → train: {len(train_ds)} | val: {len(val_ds)} | test: {len(test_ds)}"
    )
    return train_loader, val_loader, test_loader, full_ds
