"""Preprocessing utilities for the EEG pipeline."""
# ================================================================
# EEG Preprocessing: Filtering, Artifact Removal, Normalisation
# ================================================================

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.preprocessing import StandardScaler
from config import cfg


# ── Filter design ────────────────────────────────────────────────
def _notch_filter(data: np.ndarray,
                  freq: float, fs: int) -> np.ndarray:
    """Remove power-line noise."""
    b, a = iirnotch(freq, Q=30, fs=fs)
    return filtfilt(b, a, data, axis=-1)


def _bandpass_filter(data  : np.ndarray,
                     low   : float,
                     high  : float,
                     fs    : int,
                     order : int = 4) -> np.ndarray:
    """Bandpass Butterworth filter."""
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, data, axis=-1)


def _clip_artifacts(data      : np.ndarray,
                    threshold : float) -> np.ndarray:
    """Hard clip extreme amplitudes (muscle artifacts)."""
    return np.clip(data, -threshold, threshold)


def _common_average_reference(data: np.ndarray) -> np.ndarray:
    """Subtract mean across channels (CAR)."""
    return data - data.mean(axis=0, keepdims=True)


def preprocess_signals(signals: np.ndarray,
                        fs    : int) -> np.ndarray:
    """
    Full preprocessing pipeline:
      1. Clip artifacts
      2. Notch filter (60 Hz)
      3. Bandpass filter (0.5–40 Hz)
      4. Common average reference
      5. Channel-wise z-score normalisation

    Parameters
    ----------
    signals : (N_CHANNELS, n_samples)

    Returns
    -------
    clean   : (N_CHANNELS, n_samples)  float32
    """
    x = signals.copy().astype(np.float64)

    # Step 1: Artifact clipping
    x = _clip_artifacts(x, cfg.ARTIFACT_THRESH)

    # Step 2: Notch filter
    x = _notch_filter(x, cfg.NOTCH_FREQ, fs)

    # Step 3: Bandpass
    x = _bandpass_filter(x,
                          cfg.BANDPASS_LOW,
                          cfg.BANDPASS_HIGH,
                          fs,
                          cfg.BANDPASS_ORDER)

    # Step 4: Common average reference
    x = _common_average_reference(x)

    # Step 5: Channel-wise z-score
    mean = x.mean(axis=1, keepdims=True)
    std  = x.std(axis=1,  keepdims=True) + 1e-8
    x    = (x - mean) / std

    return x.astype(np.float32)


def preprocess_all(records: list[dict]) -> list[dict]:
    """Apply preprocessing to all loaded records in-place."""
    print("[Preprocessor] Filtering all records ...")
    for rec in records:
        rec["signals"] = preprocess_signals(
            rec["signals"], rec["fs"])
        print(f"  ✓ Preprocessed {rec['fname']}")
    print("[Preprocessor] Done.\n")
    return records