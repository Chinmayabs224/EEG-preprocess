"""
Data loading utilities for the spike encoding pipeline.

Provides chunk-based (generator) and full-file EDF loaders.
The chunk loader is the core memory-safety mechanism — it reads
only ``chunk_duration_sec`` seconds at a time via pyedflib, keeping
memory usage bounded regardless of file size.
"""

import numpy as np
import pyedflib
import logging
from typing import Generator, Tuple, Optional

from .config import PipelineConfig


# ═════════════════════════════════════════════════════════════════
# Chunk-Based EDF Loader (Generator)
# ═════════════════════════════════════════════════════════════════

def load_edf_chunks(
    filepath: str,
    config: PipelineConfig,
    logger: Optional[logging.Logger] = None,
) -> Generator[Tuple[np.ndarray, int, int], None, None]:
    """
    Generator that yields time-chunks of an EDF file without
    loading the entire recording into memory.

    Each chunk is ``config.chunk_duration_sec`` seconds long
    (last chunk may be shorter).

    Parameters
    ----------
    filepath : str
        Path to the ``.edf`` file.
    config : PipelineConfig
        Pipeline configuration.
    logger : logging.Logger, optional

    Yields
    ------
    tuple of (signals, chunk_start_sample, fs)
        signals : np.ndarray, shape ``(n_channels, chunk_samples)``
        chunk_start_sample : int, sample offset within the full recording
        fs : int, sampling frequency
    """
    try:
        reader = pyedflib.EdfReader(filepath)
    except Exception as exc:
        if logger:
            logger.error(f"Cannot open EDF file {filepath}: {exc}")
        return

    try:
        n_signals = reader.signals_in_file
        fs = int(reader.getSampleFrequency(0))
        total_samples = reader.getNSamples()[0]

        # Use actual channel count from file, capped at config
        n_ch = min(n_signals, config.n_channels)
        chunk_samples = int(config.chunk_duration_sec * fs)

        if logger:
            duration_min = total_samples / fs / 60
            logger.debug(
                f"  EDF: {n_signals} signals, {fs} Hz, "
                f"{total_samples} samples ({duration_min:.1f} min), "
                f"using {n_ch} channels, chunk={chunk_samples} samples"
            )

        # Yield chunks
        start = 0
        while start < total_samples:
            n_read = min(chunk_samples, total_samples - start)
            chunk = np.zeros((n_ch, n_read), dtype=np.float64)

            for ch in range(n_ch):
                chunk[ch, :] = reader.readSignal(ch, start, n_read)

            yield chunk, start, fs
            start += n_read

    finally:
        reader.close()


# ═════════════════════════════════════════════════════════════════
# Full-File EDF Loader (for visualization / small files)
# ═════════════════════════════════════════════════════════════════

def load_edf_full(
    filepath: str,
    config: PipelineConfig,
    logger: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, int]:
    """
    Load an entire EDF file into memory.

    Use this only for visualization or small files.  For large-scale
    processing, prefer :func:`load_edf_chunks`.

    Parameters
    ----------
    filepath : str
        Path to the ``.edf`` file.
    config : PipelineConfig

    Returns
    -------
    signals : np.ndarray, shape ``(n_channels, n_samples)``
    fs : int
    """
    try:
        with pyedflib.EdfReader(filepath) as reader:
            n_signals = reader.signals_in_file
            fs = int(reader.getSampleFrequency(0))
            n_samples = reader.getNSamples()[0]
            n_ch = min(n_signals, config.n_channels)

            signals = np.zeros((n_ch, n_samples), dtype=np.float64)
            for ch in range(n_ch):
                signals[ch, :] = reader.readSignal(ch)

        if logger:
            logger.debug(
                f"  Loaded full EDF: {n_ch} ch × {n_samples} samples "
                f"({n_samples / fs:.1f}s)"
            )
        return signals, fs

    except Exception as exc:
        if logger:
            logger.error(f"Failed to load {filepath}: {exc}")
        raise


# ═════════════════════════════════════════════════════════════════
# Label Builder
# ═════════════════════════════════════════════════════════════════

def build_labels(
    n_samples: int,
    seizure_intervals: list,
    fs: int,
    preictal_sec: int = 300,
) -> dict:
    """
    Build sample-resolution label arrays from seizure intervals.

    Parameters
    ----------
    n_samples : int
        Total number of samples in the recording.
    seizure_intervals : list of (start_sec, end_sec)
        Seizure intervals in seconds.
    fs : int
        Sampling frequency.
    preictal_sec : int
        Pre-ictal window duration in seconds (default 5 min).

    Returns
    -------
    dict with keys:
        - ``"binary"``: 0=interictal, 1=ictal
        - ``"preictal"``: 0=interictal, 1=pre-ictal, 2=ictal
    """
    binary = np.zeros(n_samples, dtype=np.int32)
    preictal = np.zeros(n_samples, dtype=np.int32)

    for start_s, end_s in seizure_intervals:
        s = int(start_s * fs)
        e = min(int(end_s * fs), n_samples)
        ps = max(0, s - preictal_sec * fs)

        binary[s:e] = 1           # ictal
        preictal[ps:s] = 1        # pre-ictal
        preictal[s:e] = 2         # ictal

    return {"binary": binary, "preictal": preictal}
