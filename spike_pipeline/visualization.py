"""
Visualization utilities for spike encoding verification.

Generates 7 types of plots to verify that spikes are being generated
correctly and to compare encoding methods:

    1. Signal vs Spikes overlay
    2. Spike density plot
    3. Spike count histogram
    4. Spike activity heatmap
    5. Spike raster plot
    6. Method comparison (all 3 methods side-by-side)
    7. Seizure spike overlay
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/EC2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import logging
from typing import Dict, List, Optional, Tuple

from .config import PipelineConfig


# ── Style setup ──────────────────────────────────────────────────
try:
    plt.style.use("seaborn-v0_8-darkgrid")
except OSError:
    plt.style.use("ggplot")

COLORS = {
    "rate": "#4ECDC4",
    "lif": "#FF6B6B",
    "temporal": "#45B7D1",
    "signal": "#2C3E50",
    "seizure": "#E74C3C",
    "threshold": "#E67E22",
}


def _save_fig(fig, plots_dir: str, filename: str,
              logger: Optional[logging.Logger] = None):
    """Save figure and close."""
    path = os.path.join(plots_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    if logger:
        logger.info(f"  📊 Saved plot: {filename}")


# ═════════════════════════════════════════════════════════════════
# 1. Signal vs Spikes Overlay
# ═════════════════════════════════════════════════════════════════

def plot_signal_vs_spikes(
    signal: np.ndarray,
    spikes: np.ndarray,
    fs: int,
    config: PipelineConfig,
    file_id: str = "",
    n_channels_show: int = 3,
    logger: Optional[logging.Logger] = None,
):
    """
    Plot original EEG signal with spike markers overlaid.

    Shows threshold lines and spike locations for the first
    ``n_channels_show`` channels.
    """
    duration = min(config.plot_duration_sec, signal.shape[1] / fs)
    n_show = int(duration * fs)
    t = np.arange(n_show) / fs
    n_ch = min(n_channels_show, signal.shape[0], spikes.shape[0])

    fig, axes = plt.subplots(n_ch, 1, figsize=(16, 3 * n_ch), sharex=True)
    if n_ch == 1:
        axes = [axes]

    for i in range(n_ch):
        ax = axes[i]
        sig = signal[i, :n_show]
        spk = spikes[i, :n_show]

        # Plot signal
        ax.plot(t, sig, color=COLORS["signal"], lw=0.5, alpha=0.7,
                label="EEG Signal")

        # Plot spike markers
        spike_times = t[spk > 0]
        spike_vals = sig[spk > 0]
        ax.scatter(spike_times, spike_vals, color=COLORS["lif"],
                   s=8, zorder=5, alpha=0.8, label="Spikes")

        # Threshold lines (for z-scored data, std-based)
        if config.spike_method == "lif":
            thresh = config.lif_threshold
            ax.axhline(thresh, color=COLORS["threshold"], ls="--",
                       lw=1, alpha=0.5, label=f"Threshold ±{thresh:.1f}σ")
            ax.axhline(-thresh, color=COLORS["threshold"], ls="--",
                       lw=1, alpha=0.5)

        ch_name = (config.channel_names[i]
                   if i < len(config.channel_names) else f"Ch {i}")
        ax.set_ylabel(ch_name, fontsize=9)
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")
            ax.set_title(
                f"Signal vs Spikes — {file_id} "
                f"({config.spike_method} encoding)",
                fontsize=12, fontweight="bold",
            )

    axes[-1].set_xlabel("Time (s)", fontsize=10)
    fig.tight_layout()
    _save_fig(fig, config.plots_dir,
              f"signal_vs_spikes_{file_id}.png", logger)


# ═════════════════════════════════════════════════════════════════
# 2. Spike Density Plot
# ═════════════════════════════════════════════════════════════════

def plot_spike_density(
    spikes: np.ndarray,
    fs: int,
    config: PipelineConfig,
    file_id: str = "",
    window_sec: float = 1.0,
    logger: Optional[logging.Logger] = None,
):
    """Temporal spike density (spikes per second) as a smoothed line."""
    n_ch, n_samples = spikes.shape
    window = int(window_sec * fs)
    n_bins = n_samples // window

    fig, ax = plt.subplots(figsize=(14, 5))

    # Compute density for each channel, then average
    densities = np.zeros(n_bins)
    for ch in range(n_ch):
        for b in range(n_bins):
            s = b * window
            densities[b] += spikes[ch, s:s + window].sum()
    densities /= (n_ch * window_sec)  # spikes/sec averaged across channels

    t_bins = np.arange(n_bins) * window_sec
    ax.fill_between(t_bins, densities, alpha=0.3, color=COLORS["lif"])
    ax.plot(t_bins, densities, color=COLORS["lif"], lw=1.5)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Mean Spike Rate (spikes/s)", fontsize=11)
    ax.set_title(f"Spike Density — {file_id} ({config.spike_method})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, config.plots_dir,
              f"spike_density_{file_id}.png", logger)


# ═════════════════════════════════════════════════════════════════
# 3. Spike Count Histogram
# ═════════════════════════════════════════════════════════════════

def plot_spike_count_histogram(
    spikes: np.ndarray,
    config: PipelineConfig,
    file_id: str = "",
    logger: Optional[logging.Logger] = None,
):
    """Histogram of spike counts across channels."""
    spike_counts = spikes.sum(axis=1)  # per-channel counts
    n_ch = len(spike_counts)

    fig, ax = plt.subplots(figsize=(12, 5))

    ch_names = [config.channel_names[i] if i < len(config.channel_names)
                else f"Ch{i}" for i in range(n_ch)]

    bars = ax.bar(range(n_ch), spike_counts, color=COLORS["lif"],
                  edgecolor="white", alpha=0.85)
    ax.set_xticks(range(n_ch))
    ax.set_xticklabels(ch_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Spike Count", fontsize=11)
    ax.set_title(f"Spike Counts per Channel — {file_id} ({config.spike_method})",
                 fontsize=13, fontweight="bold")

    # Add value labels on top of bars
    for bar, count in zip(bars, spike_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{int(count)}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    _save_fig(fig, config.plots_dir,
              f"spike_histogram_{file_id}.png", logger)


# ═════════════════════════════════════════════════════════════════
# 4. Spike Activity Heatmap
# ═════════════════════════════════════════════════════════════════

def plot_spike_heatmap(
    spikes: np.ndarray,
    fs: int,
    config: PipelineConfig,
    file_id: str = "",
    bin_sec: float = 2.0,
    logger: Optional[logging.Logger] = None,
):
    """Heatmap of spike activity (channels × time bins)."""
    n_ch, n_samples = spikes.shape
    bin_samples = int(bin_sec * fs)
    n_bins = n_samples // bin_samples

    # Bin the spikes
    heatmap_data = np.zeros((n_ch, n_bins))
    for b in range(n_bins):
        s = b * bin_samples
        heatmap_data[:, b] = spikes[:, s:s + bin_samples].sum(axis=1)

    ch_names = [config.channel_names[i] if i < len(config.channel_names)
                else f"Ch{i}" for i in range(n_ch)]

    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(heatmap_data, aspect="auto", cmap="hot",
                   interpolation="nearest")

    ax.set_yticks(range(n_ch))
    ax.set_yticklabels(ch_names, fontsize=8)
    ax.set_xlabel(f"Time Bin ({bin_sec}s each)", fontsize=11)
    ax.set_ylabel("Channel", fontsize=11)
    ax.set_title(f"Spike Activity Heatmap — {file_id} ({config.spike_method})",
                 fontsize=13, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Spike Count per Bin")
    fig.tight_layout()
    _save_fig(fig, config.plots_dir,
              f"spike_heatmap_{file_id}.png", logger)


# ═════════════════════════════════════════════════════════════════
# 5. Spike Raster Plot
# ═════════════════════════════════════════════════════════════════

def plot_raster(
    spikes: np.ndarray,
    fs: int,
    config: PipelineConfig,
    file_id: str = "",
    duration_sec: float = 10.0,
    logger: Optional[logging.Logger] = None,
):
    """Classic neuroscience spike raster plot."""
    n_ch = spikes.shape[0]
    n_show = min(int(duration_sec * fs), spikes.shape[1])
    t = np.arange(n_show) / fs

    ch_names = [config.channel_names[i] if i < len(config.channel_names)
                else f"Ch{i}" for i in range(n_ch)]

    fig, ax = plt.subplots(figsize=(16, max(5, n_ch * 0.35)))

    for ch in range(n_ch):
        spike_times = t[spikes[ch, :n_show] > 0]
        ax.scatter(spike_times, np.full_like(spike_times, ch),
                   marker="|", s=10, color=COLORS["lif"], linewidths=0.8)

    ax.set_yticks(range(n_ch))
    ax.set_yticklabels(ch_names, fontsize=8)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Channel", fontsize=11)
    ax.set_title(f"Spike Raster — {file_id} ({config.spike_method})",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0, duration_sec)
    ax.invert_yaxis()
    fig.tight_layout()
    _save_fig(fig, config.plots_dir,
              f"spike_raster_{file_id}.png", logger)


# ═════════════════════════════════════════════════════════════════
# 6. Method Comparison (all 3 methods side-by-side)
# ═════════════════════════════════════════════════════════════════

def plot_method_comparison(
    signal: np.ndarray,
    all_spikes: Dict[str, np.ndarray],
    fs: int,
    config: PipelineConfig,
    file_id: str = "",
    channel: int = 0,
    logger: Optional[logging.Logger] = None,
):
    """
    Compare all 3 encoding methods on the same signal segment.

    Parameters
    ----------
    all_spikes : dict
        Mapping from method name → spike array.
    """
    duration = min(config.plot_duration_sec, signal.shape[1] / fs)
    n_show = int(duration * fs)
    t = np.arange(n_show) / fs
    sig = signal[channel, :n_show]

    methods = list(all_spikes.keys())
    n_methods = len(methods)

    fig, axes = plt.subplots(n_methods + 1, 1,
                             figsize=(16, 3 * (n_methods + 1)),
                             sharex=True)

    # Original signal
    axes[0].plot(t, sig, color=COLORS["signal"], lw=0.5)
    ch_name = (config.channel_names[channel]
               if channel < len(config.channel_names) else f"Ch{channel}")
    axes[0].set_ylabel(f"EEG ({ch_name})", fontsize=9)
    axes[0].set_title(
        f"Encoding Method Comparison — {file_id}, {ch_name}",
        fontsize=13, fontweight="bold",
    )

    # Each method
    for i, method in enumerate(methods, start=1):
        ax = axes[i]
        spk = all_spikes[method][channel, :n_show]
        color = COLORS.get(method, "#333333")

        spike_times = t[spk > 0]
        ax.eventplot([spike_times], lineoffsets=0.5, linelengths=0.8,
                     colors=[color], linewidths=0.8)
        ax.set_ylim(0, 1)
        ax.set_ylabel(method.upper(), fontsize=9, fontweight="bold")

        n_spikes = int(spk.sum())
        rate = n_spikes / duration if duration > 0 else 0
        ax.text(0.98, 0.85, f"{n_spikes} spikes ({rate:.0f}/s)",
                transform=ax.transAxes, ha="right", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    axes[-1].set_xlabel("Time (s)", fontsize=10)
    fig.tight_layout()
    _save_fig(fig, config.plots_dir,
              f"method_comparison_{file_id}.png", logger)


# ═════════════════════════════════════════════════════════════════
# 7. Seizure Spike Overlay
# ═════════════════════════════════════════════════════════════════

def plot_seizure_spike_overlay(
    signal: np.ndarray,
    spikes: np.ndarray,
    seizure_intervals: List[Tuple[int, int]],
    fs: int,
    config: PipelineConfig,
    file_id: str = "",
    channel: int = 0,
    logger: Optional[logging.Logger] = None,
):
    """
    Highlight seizure regions with spike density overlay.

    Shows whether spike activity increases during seizures.
    """
    n_samples = signal.shape[1]
    t = np.arange(n_samples) / fs

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})

    # Top: EEG signal with seizure shading
    sig = signal[channel]
    axes[0].plot(t, sig, color=COLORS["signal"], lw=0.3, alpha=0.7)

    for start_s, end_s in seizure_intervals:
        axes[0].axvspan(start_s, end_s, color=COLORS["seizure"],
                        alpha=0.2, label="Seizure")
        axes[1].axvspan(start_s, end_s, color=COLORS["seizure"],
                        alpha=0.2)

    ch_name = (config.channel_names[channel]
               if channel < len(config.channel_names) else f"Ch{channel}")
    axes[0].set_ylabel(f"EEG ({ch_name})", fontsize=10)
    axes[0].set_title(
        f"Seizure × Spike Activity — {file_id}",
        fontsize=13, fontweight="bold",
    )
    # Deduplicate legend
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend([handles[0]], [labels[0]], fontsize=9,
                       loc="upper right")

    # Bottom: spike density
    window_sec = 2.0
    window = int(window_sec * fs)
    n_bins = n_samples // window
    density = np.zeros(n_bins)
    n_ch = spikes.shape[0]
    for b in range(n_bins):
        s = b * window
        density[b] = spikes[:, s:s + window].sum() / (n_ch * window_sec)

    t_bins = np.arange(n_bins) * window_sec
    axes[1].fill_between(t_bins, density, alpha=0.4, color=COLORS["lif"])
    axes[1].plot(t_bins, density, color=COLORS["lif"], lw=1)
    axes[1].set_ylabel("Spike Rate\n(spikes/s)", fontsize=10)
    axes[1].set_xlabel("Time (s)", fontsize=10)

    fig.tight_layout()
    _save_fig(fig, config.plots_dir,
              f"seizure_spike_overlay_{file_id}.png", logger)


# ═════════════════════════════════════════════════════════════════
# Generate All Plots for a File
# ═════════════════════════════════════════════════════════════════

def generate_all_plots(
    signal: np.ndarray,
    spikes: np.ndarray,
    fs: int,
    config: PipelineConfig,
    file_id: str,
    seizure_intervals: Optional[list] = None,
    all_method_spikes: Optional[Dict[str, np.ndarray]] = None,
    logger: Optional[logging.Logger] = None,
):
    """
    Generate all 7 verification plot types for a single file.

    Parameters
    ----------
    signal : preprocessed signal ``(n_ch, n_samples)``
    spikes : spike array ``(n_ch, n_samples)``
    all_method_spikes : optional dict of method → spikes for comparison plot
    """
    plot_signal_vs_spikes(signal, spikes, fs, config, file_id, logger=logger)
    plot_spike_density(spikes, fs, config, file_id, logger=logger)
    plot_spike_count_histogram(spikes, config, file_id, logger=logger)
    plot_spike_heatmap(spikes, fs, config, file_id, logger=logger)
    plot_raster(spikes, fs, config, file_id, logger=logger)

    if all_method_spikes:
        plot_method_comparison(signal, all_method_spikes, fs, config,
                               file_id, logger=logger)

    if seizure_intervals:
        plot_seizure_spike_overlay(signal, spikes, seizure_intervals,
                                   fs, config, file_id, logger=logger)
