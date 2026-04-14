"""
EEG signal preprocessing for the spike encoding pipeline.

Applies the standard cleaning chain:
    1. Artifact clipping  (±threshold µV)
    2. Notch filter        (60 Hz power-line noise)
    3. Bandpass filter      (0.5–40 Hz Butterworth)
    4. Common average reference (CAR)
    5. Channel-wise z-score normalization
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

from .config import PipelineConfig


# ── Internal filter functions ────────────────────────────────────

def _clip_artifacts(data: np.ndarray, threshold: float) -> np.ndarray:
    """Hard-clip extreme amplitudes (muscle / movement artifacts)."""
    return np.clip(data, -threshold, threshold)


def _notch_filter(data: np.ndarray, freq: float, fs: int) -> np.ndarray:
    """Remove power-line noise at *freq* Hz."""
    b, a = iirnotch(freq, Q=30, fs=fs)
    return filtfilt(b, a, data, axis=-1)


def _bandpass_filter(data: np.ndarray,
                     low: float,
                     high: float,
                     fs: int,
                     order: int = 4) -> np.ndarray:
    """Apply Butterworth bandpass filter."""
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, data, axis=-1)


def _common_average_reference(data: np.ndarray) -> np.ndarray:
    """Subtract the mean across channels at each time point (CAR)."""
    return data - data.mean(axis=0, keepdims=True)


def _zscore_normalize(data: np.ndarray) -> np.ndarray:
    """Channel-wise z-score normalization."""
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True) + 1e-8
    return (data - mean) / std


# ═════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════

def preprocess_chunk(chunk: np.ndarray,
                     fs: int,
                     config: PipelineConfig) -> np.ndarray:
    """
    Apply the full preprocessing pipeline to a single chunk.

    Parameters
    ----------
    chunk : np.ndarray, shape ``(n_channels, n_samples)``
        Raw EEG signal chunk.
    fs : int
        Sampling frequency.
    config : PipelineConfig
        Pipeline configuration with preprocessing parameters.

    Returns
    -------
    np.ndarray, shape ``(n_channels, n_samples)``, dtype float32
        Cleaned, normalized signal.
    """
    x = chunk.copy().astype(np.float64)

    # Step 1: Artifact clipping
    x = _clip_artifacts(x, config.artifact_thresh)

    # Step 2: Notch filter (60 Hz)
    x = _notch_filter(x, config.notch_freq, fs)

    # Step 3: Bandpass filter (0.5–40 Hz)
    x = _bandpass_filter(
        x, config.bandpass_low, config.bandpass_high,
        fs, config.bandpass_order
    )

    # Step 4: Common average reference
    x = _common_average_reference(x)

    # Step 5: Channel-wise z-score
    x = _zscore_normalize(x)

    return x.astype(np.float32)
