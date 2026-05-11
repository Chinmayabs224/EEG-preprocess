"""
Feature Augmentor: Inject spectral-like and statistical features
into the SNN spike-rate feature vectors.

Since raw EEG is not available at training time (only 64-dim SNN
spike-rate vectors from .npz files), we derive additional biomarkers
from the spike-rate vectors themselves:

    - Band-like groupings (5 groups of ~13 neurons each)
    - Statistical moments (mean, variance, skewness, kurtosis)
    - Line-length (sum of absolute first differences)
    - Hjorth parameters (activity, mobility, complexity)

This increases the feature dimension from 64 → 75 (64 + 11 derived).
"""

import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FeatureAugmentor:
    """
    Augments 64-dim SNN spike-rate features with 11 derived biomarkers.

    Must be fit on training data before transforming any data.
    The z-score normalization is fitted on training data only.

    Usage::

        aug = FeatureAugmentor()
        aug.fit(X_train)           # fit normalization on training set
        X_train_aug = aug.transform(X_train)  # (N, 75)
        X_val_aug   = aug.transform(X_val)    # uses training stats
    """

    # Approximate "band" groupings over the 64 SNN neurons
    # (analogous to δ, θ, α, β, γ frequency bands)
    BAND_SLICES = [
        slice(0, 13),    # "delta-like" (neurons 0–12)
        slice(13, 26),   # "theta-like" (neurons 13–25)
        slice(26, 39),   # "alpha-like" (neurons 26–38)
        slice(39, 52),   # "beta-like"  (neurons 39–51)
        slice(52, 64),   # "gamma-like" (neurons 52–63)
    ]

    def __init__(self):
        self.fitted = False
        self.mean_ = None
        self.std_ = None

    def _extract_derived(self, X: np.ndarray) -> np.ndarray:
        """
        Extract 11 derived features from each row of X (shape N×64).

        Features (per sample):
            [0-4]  Band-energy proxies (sum-of-squares in each of 5 groups)
            [5]    Line-length (sum of |x[i+1] - x[i]|)
            [6]    Hjorth Activity (variance)
            [7]    Hjorth Mobility (std of first diff / std of signal)
            [8]    Hjorth Complexity (mobility of first diff / mobility of signal)
            [9]    Skewness
            [10]   Kurtosis

        Returns shape (N, 11).
        """
        N = X.shape[0]
        derived = np.empty((N, 11), dtype=np.float32)

        for i, sl in enumerate(self.BAND_SLICES):
            band = X[:, sl]
            derived[:, i] = np.sum(band ** 2, axis=1)  # band energy

        # Line-length
        diffs = np.diff(X, axis=1)  # (N, 63)
        derived[:, 5] = np.sum(np.abs(diffs), axis=1)

        # Hjorth Activity = variance of signal
        activity = np.var(X, axis=1) + 1e-10
        derived[:, 6] = activity

        # Hjorth Mobility = std(first_diff) / std(signal)
        diff_std = np.std(diffs, axis=1) + 1e-10
        sig_std = np.sqrt(activity)
        mobility = diff_std / sig_std
        derived[:, 7] = mobility

        # Hjorth Complexity = mobility(first_diff) / mobility(signal)
        diff2 = np.diff(diffs, axis=1)  # (N, 62)
        diff2_std = np.std(diff2, axis=1) + 1e-10
        mobility_diff = diff2_std / diff_std
        complexity = mobility_diff / (mobility + 1e-10)
        derived[:, 8] = complexity

        # Skewness
        mean = np.mean(X, axis=1, keepdims=True)
        centered = X - mean
        m3 = np.mean(centered ** 3, axis=1)
        m2 = np.mean(centered ** 2, axis=1)
        derived[:, 9] = m3 / (m2 ** 1.5 + 1e-10)

        # Kurtosis
        m4 = np.mean(centered ** 4, axis=1)
        derived[:, 10] = m4 / (m2 ** 2 + 1e-10) - 3.0

        return derived

    def fit(self, X_train: np.ndarray) -> "FeatureAugmentor":
        """
        Fit the normalization statistics on training features.

        Parameters
        ----------
        X_train : np.ndarray, shape (N, 64)
        """
        derived = self._extract_derived(X_train)
        self.mean_ = derived.mean(axis=0)
        self.std_ = derived.std(axis=0) + 1e-8
        self.fitted = True
        logger.info(
            f"FeatureAugmentor fitted: {X_train.shape[0]} samples, "
            f"derived dim=11, total output dim=75"
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features: concatenate 64 original + 11 z-scored derived.

        Parameters
        ----------
        X : np.ndarray, shape (N, 64)

        Returns
        -------
        np.ndarray, shape (N, 75)
        """
        if not self.fitted:
            raise RuntimeError("FeatureAugmentor must be fit() before transform()")

        derived = self._extract_derived(X)
        # Z-score normalize using training statistics
        derived = (derived - self.mean_) / self.std_

        return np.concatenate([X, derived], axis=1).astype(np.float32)

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """Fit on X_train and return augmented features."""
        self.fit(X_train)
        return self.transform(X_train)

    @property
    def output_dim(self) -> int:
        """Total output feature dimension."""
        return 75  # 64 original + 11 derived
