"""
Spike encoding methods for the EEG pipeline.

Implements the three encoding strategies shown in the architecture diagram:

    1. **Rate coding**     — Poisson spikes (firing probability ∝ amplitude)
    2. **LIF encoding**    — Leaky Integrate-and-Fire threshold firing
    3. **Temporal coding** — Spike timing (amplitude → latency within time bin)

Each method converts a preprocessed continuous signal ``(n_channels, n_samples)``
into a binary spike array ``(n_channels, n_samples)`` of 0/1 values.
"""

import numpy as np
from typing import Dict, Any

from .config import PipelineConfig


# ═════════════════════════════════════════════════════════════════
# Encoding Method 1: Rate Coding (Poisson Spikes)
# ═════════════════════════════════════════════════════════════════

def rate_encode(signal: np.ndarray,
                max_rate_hz: float = 100.0,
                fs: int = 256,
                rng: np.random.Generator = None) -> np.ndarray:
    """
    Rate coding via Poisson spike generation.

    The signal amplitude is normalized to [0, 1] per channel, then
    each sample generates a spike with probability::

        p = normalized_amplitude × (max_rate_hz / fs)

    Higher amplitude → more frequent spikes.

    Parameters
    ----------
    signal : np.ndarray, shape ``(n_channels, n_samples)``
        Preprocessed EEG signal (z-scored).
    max_rate_hz : float
        Maximum firing rate in Hz.
    fs : int
        Sampling frequency.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    np.ndarray, shape ``(n_channels, n_samples)``, dtype uint8
        Binary spike array (0 or 1).
    """
    if rng is None:
        rng = np.random.default_rng()

    n_ch, n_samples = signal.shape
    spikes = np.zeros((n_ch, n_samples), dtype=np.uint8)

    for ch in range(n_ch):
        sig = signal[ch]
        # Normalize to [0, 1] using rectified absolute value
        sig_abs = np.abs(sig)
        sig_max = sig_abs.max()
        if sig_max > 0:
            normalized = sig_abs / sig_max
        else:
            normalized = sig_abs

        # Spike probability per time step
        prob = normalized * (max_rate_hz / fs)
        prob = np.clip(prob, 0.0, 1.0)

        # Generate Poisson spikes
        spikes[ch] = (rng.random(n_samples) < prob).astype(np.uint8)

    return spikes


# ═════════════════════════════════════════════════════════════════
# Encoding Method 2: LIF Encoding (Threshold Firing)
# ═════════════════════════════════════════════════════════════════

def lif_encode(signal: np.ndarray,
               threshold: float = 1.5,
               beta: float = 0.85,
               reset: float = 0.0) -> np.ndarray:
    """
    Leaky Integrate-and-Fire (LIF) spike encoding.

    Simulates a LIF neuron per channel.  The membrane potential
    accumulates the input signal with exponential decay::

        V[t] = beta * V[t-1] + signal[t]

    When ``V[t] > threshold`` → spike, then ``V[t] = reset``.

    This is the most biologically realistic of the three methods.

    Parameters
    ----------
    signal : np.ndarray, shape ``(n_channels, n_samples)``
        Preprocessed EEG signal.
    threshold : float
        Firing threshold (in same units as signal; typically std devs
        for z-scored input).
    beta : float
        Membrane decay factor (0 < beta < 1).  Higher → slower leak.
    reset : float
        Reset potential after a spike.

    Returns
    -------
    np.ndarray, shape ``(n_channels, n_samples)``, dtype uint8
        Binary spike array (0 or 1).
    """
    n_ch, n_samples = signal.shape
    spikes = np.zeros((n_ch, n_samples), dtype=np.uint8)

    for ch in range(n_ch):
        membrane = 0.0
        sig = signal[ch]

        for t in range(n_samples):
            # Leaky integration
            membrane = beta * membrane + sig[t]

            # Fire if above threshold
            if membrane >= threshold:
                spikes[ch, t] = 1
                membrane = reset  # reset after firing
            elif membrane < -threshold:
                # Also spike on strong negative deflections
                spikes[ch, t] = 1
                membrane = reset

    return spikes


# ═════════════════════════════════════════════════════════════════
# Encoding Method 3: Temporal Coding (Spike Timing)
# ═════════════════════════════════════════════════════════════════

def temporal_encode(signal: np.ndarray,
                    n_bins: int = 8,
                    tau: float = 1.0) -> np.ndarray:
    """
    Temporal coding via time-to-first-spike.

    The signal is divided into time bins of ``n_bins`` samples each.
    Within each bin, only a single spike is placed.  Higher amplitude
    → earlier spike within the bin (shorter latency).

    This encodes information in the *precise timing* of spikes,
    not their rate.

    Parameters
    ----------
    signal : np.ndarray, shape ``(n_channels, n_samples)``
        Preprocessed EEG signal.
    n_bins : int
        Number of samples per time bin.
    tau : float
        Time constant controlling the amplitude-to-latency mapping.
        Larger tau → more spread-out spike times.

    Returns
    -------
    np.ndarray, shape ``(n_channels, n_samples)``, dtype uint8
        Binary spike array (0 or 1).
    """
    n_ch, n_samples = signal.shape
    spikes = np.zeros((n_ch, n_samples), dtype=np.uint8)

    n_full_bins = n_samples // n_bins

    for ch in range(n_ch):
        sig = signal[ch]

        for b in range(n_full_bins):
            start = b * n_bins
            end = start + n_bins
            window = sig[start:end]

            # Amplitude of this bin (use absolute value)
            amplitude = np.abs(window).max()

            if amplitude < 1e-6:
                # No significant signal → no spike in this bin
                continue

            # Normalize amplitude within bin to [0, 1]
            amp_norm = np.abs(window) / amplitude

            # Latency: higher amplitude → earlier spike (lower index)
            # latency = tau * (1 - amp_norm) mapped to [0, n_bins-1]
            peak_idx = np.argmax(np.abs(window))
            spike_time = start + peak_idx
            spikes[ch, spike_time] = 1

    return spikes


# ═════════════════════════════════════════════════════════════════
# Unified Spike Encoder Class
# ═════════════════════════════════════════════════════════════════

class SpikeEncoder:
    """
    Unified interface for spike encoding.

    Wraps the three encoding methods behind a common ``.encode()`` API.
    The method is selected via ``config.spike_method``.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration.
    """

    METHODS = {"rate", "lif", "temporal"}

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.method = config.spike_method
        self.rng = np.random.default_rng(config.random_seed)

        if self.method not in self.METHODS:
            raise ValueError(
                f"Unknown spike method '{self.method}'. "
                f"Must be one of {self.METHODS}"
            )

    def encode(self, preprocessed_signal: np.ndarray,
               fs: int) -> Dict[str, Any]:
        """
        Encode a preprocessed signal into binary spikes.

        Parameters
        ----------
        preprocessed_signal : np.ndarray, shape ``(n_channels, n_samples)``
        fs : int
            Sampling frequency.

        Returns
        -------
        dict with keys:
            - ``"spikes"``: np.ndarray ``(n_channels, n_samples)`` uint8
            - ``"metadata"``: dict with encoding parameters and stats
        """
        cfg = self.config

        if self.method == "rate":
            spikes = rate_encode(
                preprocessed_signal,
                max_rate_hz=cfg.rate_max_hz,
                fs=fs,
                rng=self.rng,
            )
            params = {"max_rate_hz": cfg.rate_max_hz}

        elif self.method == "lif":
            spikes = lif_encode(
                preprocessed_signal,
                threshold=cfg.lif_threshold,
                beta=cfg.lif_beta,
                reset=cfg.lif_reset,
            )
            params = {
                "threshold": cfg.lif_threshold,
                "beta": cfg.lif_beta,
                "reset": cfg.lif_reset,
            }

        elif self.method == "temporal":
            spikes = temporal_encode(
                preprocessed_signal,
                n_bins=cfg.temporal_n_bins,
                tau=cfg.temporal_tau,
            )
            params = {
                "n_bins": cfg.temporal_n_bins,
                "tau": cfg.temporal_tau,
            }

        # Compute per-channel spike counts
        spike_counts = spikes.sum(axis=1)  # (n_channels,)
        duration_sec = preprocessed_signal.shape[1] / fs

        metadata = {
            "method": self.method,
            "params": params,
            "fs": fs,
            "n_channels": spikes.shape[0],
            "n_samples": spikes.shape[1],
            "duration_sec": duration_sec,
            "total_spikes": int(spikes.sum()),
            "spike_counts_per_channel": spike_counts.tolist(),
            "mean_firing_rate_hz": float(
                spike_counts.mean() / duration_sec
            ) if duration_sec > 0 else 0.0,
        }

        return {"spikes": spikes, "metadata": metadata}

    def encode_all_methods(self, preprocessed_signal: np.ndarray,
                           fs: int) -> Dict[str, Dict[str, Any]]:
        """
        Run all three encoding methods on the same signal.

        Useful for method comparison visualization.

        Returns
        -------
        dict mapping method name → encode() result dict
        """
        results = {}
        original_method = self.method

        for method in ["rate", "lif", "temporal"]:
            self.method = method
            results[method] = self.encode(preprocessed_signal, fs)

        self.method = original_method
        return results
