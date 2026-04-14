"""
SNN Feature Extractor for the spike encoding pipeline.

Implements Stage 5 of the architecture diagram:

    Spike Encoding → **SNN Feature Extractor** → spike trains (T×64)

Architecture::

    Input(feature_dim) → FC(256) → LIF → FC(128) → LIF → FC(64) → LIF

Uses ``snntorch`` Leaky Integrate-and-Fire neurons with learnable
beta and threshold parameters and surrogate gradient for training.

The filterbank extractor converts windowed epochs into feature vectors,
and the SNN encoder compresses them into 64-dimensional spike-rate features.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional
import logging

try:
    import snntorch as snn
    from snntorch import surrogate
    HAS_SNNTORCH = True
except ImportError:
    HAS_SNNTORCH = False

from scipy.signal import butter, filtfilt

from .config import PipelineConfig


# ═════════════════════════════════════════════════════════════════
# Filterbank Feature Extraction
# ═════════════════════════════════════════════════════════════════

class FilterbankExtractor:
    """
    M-band bandpass filterbank over a frequency range.

    Splits the signal into ``n_filters`` sub-bands and computes
    log-energy per band per channel.  This is the standard input
    representation for the SNN feature extractor.

    Parameters
    ----------
    config : PipelineConfig
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        bw = (config.freq_max - config.freq_min) / config.n_filters
        nyq = config.sampling_rate / 2.0

        self.filters = []
        for i in range(config.n_filters):
            lo = config.freq_min + i * bw
            hi = lo + bw
            # Clamp to avoid exceeding Nyquist
            hi = min(hi, nyq - 0.1)
            lo = max(lo, 0.1)
            b, a = butter(4, [lo / nyq, hi / nyq], btype="band")
            self.filters.append((b, a))

    def epoch_features(self, epoch: np.ndarray) -> np.ndarray:
        """
        Extract filterbank features from a single epoch.

        Parameters
        ----------
        epoch : np.ndarray, shape ``(n_channels, epoch_samples)``

        Returns
        -------
        np.ndarray, shape ``(n_filters × n_channels,)``
        """
        feats = []
        n_ch = epoch.shape[0]
        for ch in range(n_ch):
            sig = epoch[ch]
            for b, a in self.filters:
                filt = filtfilt(b, a, sig)
                energy = np.log1p(np.sum(filt ** 2))
                feats.append(energy)
        return np.array(feats, dtype=np.float32)

    def extract_windows(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Slide overlapping windows over the full recording and extract
        filterbank features.

        Parameters
        ----------
        signals : np.ndarray, shape ``(n_channels, n_samples)``
        labels : np.ndarray, shape ``(n_samples,)``
            Binary labels (0/1 per sample).

        Returns
        -------
        X : np.ndarray, shape ``(n_windows, feature_dim)``
        y : np.ndarray, shape ``(n_windows,)``
        """
        cfg = self.config
        ep = cfg.epoch_samples
        W = cfg.window_size
        n_epochs = signals.shape[1] // ep

        # Extract per-epoch features and labels
        ep_feats = []
        ep_labs = []
        for e in range(n_epochs):
            s = e * ep
            seg = signals[:, s:s + ep]
            lbl = labels[s:s + ep]
            ep_feats.append(self.epoch_features(seg))
            ep_labs.append(int(lbl.mean() > 0.5))

        ep_feats = np.array(ep_feats)   # (n_epochs, n_filters*n_ch)
        ep_labs = np.array(ep_labs)

        # Sliding window of W epochs
        X_list = []
        y_list = []
        for i in range(W - 1, len(ep_feats)):
            X_list.append(ep_feats[i - W + 1:i + 1].flatten())
            y_list.append(ep_labs[i])

        return (
            np.array(X_list, dtype=np.float32),
            np.array(y_list, dtype=np.int64),
        )


# ═════════════════════════════════════════════════════════════════
# SNN Encoder Network
# ═════════════════════════════════════════════════════════════════

class SNNEncoder(nn.Module):
    """
    Leaky Integrate-and-Fire SNN that encodes feature vectors
    into spike-rate representations.

    Architecture::

        Input(feature_dim) → FC(256) → LIF → FC(128) → LIF → FC(64) → LIF

    Parameters
    ----------
    config : PipelineConfig

    Notes
    -----
    Requires ``snntorch``.  If not installed, a fallback linear encoder
    is used instead.
    """

    def __init__(self, config: PipelineConfig):
        super().__init__()
        self.config = config

        if not HAS_SNNTORCH:
            # Fallback: simple linear encoder (no spiking)
            sizes = [config.feature_dim] + config.snn_hidden + [config.snn_spike_dim]
            layers = []
            for i in range(len(sizes) - 1):
                layers.append(nn.Linear(sizes[i], sizes[i + 1]))
                if i < len(sizes) - 2:
                    layers.append(nn.BatchNorm1d(sizes[i + 1]))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(0.3))
            self.fallback = nn.Sequential(*layers)
            self.use_snn = False
            return

        self.use_snn = True
        spike_grad = surrogate.fast_sigmoid(slope=25)
        sizes = [config.feature_dim] + config.snn_hidden + [config.snn_spike_dim]

        self.fc_layers = nn.ModuleList()
        self.lif_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        for i in range(len(sizes) - 1):
            self.fc_layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            self.lif_layers.append(
                snn.Leaky(
                    beta=config.snn_beta,
                    threshold=config.snn_threshold_val,
                    spike_grad=spike_grad,
                    learn_beta=True,
                    learn_threshold=True,
                )
            )
            self.bn_layers.append(nn.BatchNorm1d(sizes[i + 1]))

        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape ``(batch, feature_dim)``

        Returns
        -------
        spike_rate : torch.Tensor, shape ``(batch, snn_spike_dim)``
            Mean spike rate across time steps.
        spike_rec : torch.Tensor, shape ``(T, batch, snn_spike_dim)``
            Full spike train record.
        """
        if not self.use_snn:
            out = self.fallback(x)
            # Simulate spike rate as sigmoid output
            rate = torch.sigmoid(out)
            rec = rate.unsqueeze(0).repeat(self.config.snn_time_steps, 1, 1)
            return rate, rec

        T = self.config.snn_time_steps
        mem_states = [lif.init_leaky() for lif in self.lif_layers]
        spike_rec = []

        for _ in range(T):
            cur = x
            for i, (fc, lif, bn) in enumerate(
                zip(self.fc_layers, self.lif_layers, self.bn_layers)
            ):
                cur = fc(cur)
                cur = bn(cur)
                if i < len(self.fc_layers) - 1:
                    cur = self.dropout(cur)
                spk, mem_states[i] = lif(cur, mem_states[i])
                cur = spk
            spike_rec.append(spk)

        spike_rec = torch.stack(spike_rec, dim=0)   # (T, B, D)
        spike_rate = spike_rec.mean(dim=0)           # (B, D)
        return spike_rate, spike_rec

    def encode_numpy(
        self,
        X: np.ndarray,
        device: torch.device,
        batch_size: int = 256,
    ) -> np.ndarray:
        """
        Convenience wrapper: encode numpy arrays → numpy spike rates.

        Parameters
        ----------
        X : np.ndarray, shape ``(N, feature_dim)``
        device : torch.device
        batch_size : int

        Returns
        -------
        np.ndarray, shape ``(N, snn_spike_dim)``
        """
        self.eval()
        self.to(device)
        spike_rates = []

        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = torch.tensor(
                    X[i:i + batch_size], dtype=torch.float32
                ).to(device)
                rate, _ = self.forward(batch)
                spike_rates.append(rate.cpu().numpy())

        return np.concatenate(spike_rates, axis=0)


# ═════════════════════════════════════════════════════════════════
# Feature Extraction Convenience Function
# ═════════════════════════════════════════════════════════════════

def extract_snn_features(
    signals: np.ndarray,
    labels: np.ndarray,
    config: PipelineConfig,
    snn_model: Optional[SNNEncoder] = None,
    device: Optional[torch.device] = None,
    logger: Optional[logging.Logger] = None,
) -> dict:
    """
    End-to-end SNN feature extraction for a single recording.

    Steps:
        1. Extract filterbank features (windowed epochs)
        2. Pass through SNN encoder → 64-dim spike rates

    Parameters
    ----------
    signals : np.ndarray, shape ``(n_channels, n_samples)``
    labels : np.ndarray, shape ``(n_samples,)``
        Binary seizure labels.
    config : PipelineConfig
    snn_model : SNNEncoder, optional
        Pre-trained SNN encoder.  If None, creates one with random weights.
    device : torch.device, optional

    Returns
    -------
    dict with keys:
        - ``"filterbank_features"``: ``(n_windows, feature_dim)``
        - ``"snn_features"``: ``(n_windows, snn_spike_dim)``
        - ``"labels"``: ``(n_windows,)``
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Filterbank features
    extractor = FilterbankExtractor(config)
    X_feat, y = extractor.extract_windows(signals, labels)

    if logger:
        logger.debug(
            f"    Filterbank: {X_feat.shape[0]} windows, "
            f"dim={X_feat.shape[1]}"
        )

    # Step 2: SNN encoding
    if snn_model is None:
        snn_model = SNNEncoder(config)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_feat)

    snn_features = snn_model.encode_numpy(
        X_scaled, device, batch_size=config.batch_size
    )

    if logger:
        logger.debug(
            f"    SNN features: {snn_features.shape} "
            f"(T×{config.snn_spike_dim})"
        )

    return {
        "filterbank_features": X_feat,
        "snn_features": snn_features,
        "labels": y,
    }
