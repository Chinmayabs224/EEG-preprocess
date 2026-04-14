"""SNN encoder components for the EEG pipeline."""
# ================================================================
# SNN Encoder: Converts EEG feature vectors → Spike Trains
#
# Architecture:
#   Input(432) → FC(256) → LIF → FC(128) → LIF → FC(64) → LIF
#
# Output: spike-rate features (64-dim) per time window
# ================================================================

import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from config import cfg


# ── Filterbank feature extraction ────────────────────────────────
class FilterbankExtractor:
    """
    M=8 bandpass filterbank over 0.5–25 Hz.
    Returns log-energy per band per channel.
    """
    def __init__(self):
        bw     = (cfg.FREQ_MAX - cfg.FREQ_MIN) / cfg.N_FILTERS
        nyq    = cfg.SAMPLING_RATE / 2.0
        self.filters = []
        for i in range(cfg.N_FILTERS):
            lo = cfg.FREQ_MIN + i * bw
            hi = lo + bw
            b, a = butter(4, [lo/nyq, hi/nyq], btype='band')
            self.filters.append((b, a))

    def epoch_features(self, epoch: np.ndarray) -> np.ndarray:
        """
        epoch : (N_CH, EPOCH_SAMPLES)
        → (N_FILTERS * N_CH,)
        """
        feats = []
        for ch in range(epoch.shape[0]):
            sig = epoch[ch]
            for b, a in self.filters:
                filt   = filtfilt(b, a, sig)
                energy = np.log1p(np.sum(filt ** 2))
                feats.append(energy)
        return np.array(feats, dtype=np.float32)

    def extract_windows(self,
                        signals: np.ndarray,
                        labels : np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray]:
        """
        Slide W-epoch windows over the full recording.

        Returns
        -------
        X : (n_windows, FEATURE_DIM=432)
        y : (n_windows,)
        """
        ep   = cfg.EPOCH_SAMPLES
        W    = cfg.WINDOW_SIZE
        n_ep = signals.shape[1] // ep

        ep_feats, ep_labs = [], []
        for e in range(n_ep):
            s   = e * ep
            seg = signals[:, s:s + ep]
            lbl = labels[s:s + ep]
            ep_feats.append(self.epoch_features(seg))
            ep_labs.append(int(lbl.mean() > 0.5))

        ep_feats = np.array(ep_feats)   # (n_ep, 144)
        ep_labs  = np.array(ep_labs)    # (n_ep,)

        X_list, y_list = [], []
        for i in range(W - 1, len(ep_feats)):
            X_list.append(
                ep_feats[i - W + 1:i + 1].flatten())  # (432,)
            y_list.append(ep_labs[i])

        return (np.array(X_list, dtype=np.float32),
                np.array(y_list, dtype=np.int64))


# ── SNN Encoder Network ──────────────────────────────────────────
class SNNEncoder(nn.Module):
    """
    Leaky Integrate-and-Fire SNN that encodes feature vectors
    into spike-rate representations.

    Input  : (batch, FEATURE_DIM)
    Output : (batch, SNN_SPIKE_DIM)  — averaged spike rates
    """

    def __init__(self):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)
        sizes      = ([cfg.FEATURE_DIM]
                      + cfg.SNN_HIDDEN
                      + [cfg.SNN_SPIKE_DIM])

        self.fc_layers  = nn.ModuleList()
        self.lif_layers = nn.ModuleList()
        self.bn_layers  = nn.ModuleList()

        for i in range(len(sizes) - 1):
            self.fc_layers.append(
                nn.Linear(sizes[i], sizes[i + 1]))
            self.lif_layers.append(
                snn.Leaky(beta           = cfg.SNN_BETA,
                          threshold      = cfg.SNN_THRESHOLD,
                          spike_grad     = spike_grad,
                          learn_beta     = True,
                          learn_threshold= True))
            self.bn_layers.append(
                nn.BatchNorm1d(sizes[i + 1]))

        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        spike_rate : (batch, SNN_SPIKE_DIM)  mean spike rate
        spike_rec  : (T, batch, SNN_SPIKE_DIM) full spike train
        """
        T          = cfg.SNN_TIME_STEPS
        mem_states = [lif.init_leaky()
                      for lif in self.lif_layers]
        spike_rec  = []

        for _ in range(T):
            cur = x
            for i, (fc, lif, bn) in enumerate(
                    zip(self.fc_layers,
                        self.lif_layers,
                        self.bn_layers)):
                cur  = fc(cur)
                cur  = bn(cur)
                if i < len(self.fc_layers) - 1:
                    cur = self.dropout(cur)
                spk, mem_states[i] = lif(cur, mem_states[i])
                cur = spk
            spike_rec.append(spk)

        spike_rec  = torch.stack(spike_rec, dim=0)   # (T,B,D)
        spike_rate = spike_rec.mean(dim=0)            # (B,D)
        return spike_rate, spike_rec

    def encode_numpy(self,
                     X      : np.ndarray,
                     device : torch.device,
                     scaler : StandardScaler = None
                     ) -> np.ndarray:
        """
        Convenience wrapper for numpy arrays.

        Returns spike_rate : (N, SNN_SPIKE_DIM) numpy float32
        """
        if scaler is not None:
            X = scaler.transform(X)
        self.eval()
        self.to(device)
        spike_rates = []
        with torch.no_grad():
            for i in range(0, len(X), 256):
                batch = torch.tensor(
                    X[i:i+256], dtype=torch.float32
                ).to(device)
                rate, _ = self.forward(batch)
                spike_rates.append(rate.cpu().numpy())
        return np.concatenate(spike_rates, axis=0)