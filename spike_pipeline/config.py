"""
Configuration for the spike encoding pipeline.

All parameters are centralized here as a dataclass with sensible defaults
for the CHB-MIT Scalp EEG Database.  Override via CLI or direct assignment.
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class PipelineConfig:
    """Central configuration for the large-scale spike-encoding pipeline."""

    # ── Paths ────────────────────────────────────────────────────────
    data_dir: str = "/home/ubuntu/DESTINATION"
    output_dir: str = "./output"

    # Derived output sub-directories (created automatically)
    @property
    def spikes_dir(self) -> str:
        return os.path.join(self.output_dir, "spikes")

    @property
    def plots_dir(self) -> str:
        return os.path.join(self.output_dir, "plots")

    @property
    def logs_dir(self) -> str:
        return os.path.join(self.output_dir, "logs")

    @property
    def chunks_dir(self) -> str:
        return os.path.join(self.output_dir, "processed_chunks")

    # ── EEG Signal Parameters ────────────────────────────────────────
    sampling_rate: int = 256          # Hz
    n_channels: int = 23              # CHB-MIT has up to 23 channels
    channel_names: List[str] = field(default_factory=lambda: [
        "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
        "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
        "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
        "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
        "FZ-CZ", "CZ-PZ",
        "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8", "T8-P8",
    ])

    # ── Preprocessing ────────────────────────────────────────────────
    notch_freq: float = 60.0          # Power-line noise (Hz)
    bandpass_low: float = 0.5         # Hz
    bandpass_high: float = 40.0       # Hz
    bandpass_order: int = 4
    artifact_thresh: float = 500.0    # µV clip threshold

    # ── Windowing / Epoching ─────────────────────────────────────────
    epoch_len_sec: float = 2.0        # seconds per epoch
    window_size: int = 3              # stacked epochs → 6 s context

    @property
    def epoch_samples(self) -> int:
        return int(self.epoch_len_sec * self.sampling_rate)

    # ── Spike Encoding ───────────────────────────────────────────────
    spike_method: str = "lif"         # "rate" | "lif" | "temporal"

    # Rate coding parameters
    rate_max_hz: float = 100.0        # Max firing rate for rate coding

    # LIF encoding parameters
    lif_threshold: float = 1.5        # Threshold in std deviations
    lif_beta: float = 0.85            # Membrane decay factor
    lif_reset: float = 0.0            # Reset potential after spike

    # Temporal coding parameters
    temporal_n_bins: int = 8          # Number of time bins
    temporal_tau: float = 1.0         # Time constant

    # ── SNN Feature Extractor ────────────────────────────────────────
    snn_hidden: List[int] = field(default_factory=lambda: [256, 128])
    snn_spike_dim: int = 64           # Output feature dimension
    snn_time_steps: int = 20          # Simulation time steps
    snn_beta: float = 0.9             # LIF decay in SNN
    snn_threshold_val: float = 1.0    # LIF threshold in SNN
    n_filters: int = 8                # Filterbank bands
    freq_min: float = 0.5
    freq_max: float = 25.0

    @property
    def feature_dim(self) -> int:
        """Filterbank feature dimension = n_filters × n_channels × window_size."""
        return self.n_filters * self.n_channels * self.window_size

    # ── Scalability ──────────────────────────────────────────────────
    chunk_duration_sec: int = 60      # Read EDF in 60 s chunks (~2.2 MB each)
    n_workers: int = 4                # ProcessPoolExecutor workers
    batch_size: int = 256             # Batch size for SNN inference

    # ── Output ───────────────────────────────────────────────────────
    save_format: str = "npz"          # "npz" for compressed numpy
    plot_samples: int = 5             # Number of sample files to generate plots for
    plot_duration_sec: float = 10.0   # Duration of signal to show in plots

    # ── Misc ─────────────────────────────────────────────────────────
    random_seed: int = 42

    # ── Methods ──────────────────────────────────────────────────────
    def create_dirs(self) -> None:
        """Create all output directories."""
        for d in [self.spikes_dir, self.plots_dir,
                  self.logs_dir, self.chunks_dir]:
            os.makedirs(d, exist_ok=True)

    def __post_init__(self):
        """Validate configuration on creation."""
        valid_methods = {"rate", "lif", "temporal"}
        if self.spike_method not in valid_methods:
            raise ValueError(
                f"spike_method must be one of {valid_methods}, "
                f"got '{self.spike_method}'"
            )
        valid_formats = {"npz"}
        if self.save_format not in valid_formats:
            raise ValueError(
                f"save_format must be one of {valid_formats}, "
                f"got '{self.save_format}'"
            )
