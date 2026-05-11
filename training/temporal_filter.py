"""
Temporal Post-Processing Filters for Seizure Detection.

Implements persistence filters that dramatically reduce false positives
by requiring temporal consistency before emitting a seizure alarm.

Key insight: seizures persist for ≥10 seconds (often ≥30 sec), so
a single noisy window triggering a positive is almost always a false alarm.

Two filters are provided:
    1. MedianFilter — smooth probability curve via rolling median
    2. PersistenceFilter — require k-of-N consecutive windows above threshold

These can be applied after individual model predictions or after
ensemble combination, before thresholding.
"""

import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MedianFilter:
    """
    Rolling median filter to smooth jagged per-window probabilities.

    Parameters
    ----------
    window_size : int
        Number of consecutive windows to median-filter over.
        Default 5 (= 5 minutes if each window is 60 seconds).
    """

    def __init__(self, window_size: int = 5):
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self.window_size = window_size

    def apply(self, probs: np.ndarray) -> np.ndarray:
        """
        Apply rolling median to probability array.

        Parameters
        ----------
        probs : np.ndarray, shape (N,)

        Returns
        -------
        smoothed : np.ndarray, shape (N,)
        """
        if self.window_size <= 1 or len(probs) < self.window_size:
            return probs.copy()

        n = len(probs)
        smoothed = np.empty_like(probs)
        half = self.window_size // 2

        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            smoothed[i] = np.median(probs[lo:hi])

        return smoothed


class PersistenceFilter:
    """
    Persistence filter: require k out of the last N consecutive windows
    to exceed the threshold before emitting a seizure alarm.

    This is the single most impactful post-processing step for reducing
    false positives in seizure detection.

    Parameters
    ----------
    k : int
        Minimum number of windows that must exceed threshold.
    n : int
        Size of the look-back window.
    threshold : float
        Probability threshold for counting a window as positive.
    """

    def __init__(self, k: int = 3, n: int = 5, threshold: float = 0.5):
        if k > n:
            raise ValueError(f"k ({k}) cannot exceed n ({n})")
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.n = n
        self.threshold = threshold

    def apply(self, probs: np.ndarray) -> np.ndarray:
        """
        Apply persistence filter to probability array.

        A window is marked as seizure (1) only if at least k of the
        last n windows (including itself) have probability >= threshold.

        Parameters
        ----------
        probs : np.ndarray, shape (N,)
            Raw probabilities from model/ensemble.

        Returns
        -------
        filtered_preds : np.ndarray, shape (N,)
            Binary predictions after persistence filtering.
        """
        n_samples = len(probs)
        binary = (probs >= self.threshold).astype(int)
        filtered = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            lo = max(0, i - self.n + 1)
            window = binary[lo : i + 1]
            if window.sum() >= self.k:
                filtered[i] = 1

        return filtered


class TemporalPostProcessor:
    """
    Combined temporal post-processor that chains:
        1. MedianFilter (optional) — smooth probabilities
        2. PersistenceFilter — enforce temporal consistency

    Parameters
    ----------
    smooth_window : int
        Median filter window size. Set to 0 or 1 to disable.
    persistence_k : int
        Minimum positive windows in persistence check.
    persistence_n : int
        Look-back window for persistence check.
    threshold : float
        Seizure probability threshold.
    """

    def __init__(
        self,
        smooth_window: int = 5,
        persistence_k: int = 3,
        persistence_n: int = 5,
        threshold: float = 0.5,
    ):
        self.median_filter = MedianFilter(window_size=smooth_window) if smooth_window > 1 else None
        self.persistence_filter = PersistenceFilter(
            k=persistence_k, n=persistence_n, threshold=threshold,
        )
        self.threshold = threshold

    def process(
        self,
        probs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply full temporal post-processing pipeline.

        Parameters
        ----------
        probs : np.ndarray, shape (N,)
            Raw probabilities.

        Returns
        -------
        smoothed_probs : np.ndarray, shape (N,)
            Probabilities after median smoothing.
        filtered_preds : np.ndarray, shape (N,)
            Binary predictions after persistence filtering.
        """
        # Step 1: Median smoothing
        if self.median_filter is not None:
            smoothed = self.median_filter.apply(probs)
        else:
            smoothed = probs.copy()

        # Step 2: Persistence filtering
        filtered = self.persistence_filter.apply(smoothed)

        return smoothed, filtered

    def __repr__(self) -> str:
        mw = self.median_filter.window_size if self.median_filter else 0
        return (
            f"TemporalPostProcessor("
            f"smooth={mw}, "
            f"persist={self.persistence_filter.k}/{self.persistence_filter.n}, "
            f"thresh={self.threshold})"
        )
