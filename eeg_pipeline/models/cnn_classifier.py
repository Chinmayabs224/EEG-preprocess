"""CNN classifier for disease classification."""
# ================================================================
# Task 2: Disease Classification (CNN)
#
# Input : raw EEG segments (batch, N_CH, EPOCH_SAMPLES)
# Output: (batch, 3)  → normal / focal / generalized
#
# Architecture:
#   Multi-scale temporal convolutions → channel attention
#   → spatial convolutions → FC → class
# ================================================================

import torch
import torch.nn as nn
import numpy as np
from config import cfg


# ── Channel Attention (SE-Block) ─────────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, n_channels: int, reduction: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Sequential(
            nn.Linear(n_channels, n_channels // reduction),
            nn.ReLU(),
            nn.Linear(n_channels // reduction, n_channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T)"""
        w = self.pool(x).squeeze(-1)        # (B, C)
        w = self.fc(w).unsqueeze(-1)        # (B, C, 1)
        return x * w


# ── Temporal Block ────────────────────────────────────────────────
class TemporalBlock(nn.Module):
    def __init__(self, in_ch   : int,
                       out_ch  : int,
                       kernel  : int,
                       dilation: int = 1):
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel,
                      dilation=dilation, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ELU(),
            nn.Dropout(cfg.CNN_DROPOUT)
        )
        self.res = (nn.Conv1d(in_ch, out_ch, 1)
                    if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.res(x)


# ── EEG CNN Classifier ───────────────────────────────────────────
class EEGCNNClassifier(nn.Module):
    """
    Multi-scale dilated CNN for EEG disease classification.

    Input  : (batch, N_CHANNELS, EPOCH_SAMPLES)
    Output : (batch, CNN_CLASSES)
    """

    def __init__(self):
        super().__init__()
        C = cfg.N_CHANNELS

        # ── Multi-scale temporal feature extraction ──────
        self.ms_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(C, 32, k, padding=k//2),
                nn.BatchNorm1d(32), nn.ELU()
            )
            for k in [3, 7, 15, 31]     # scales: ~12ms, 27ms, 59ms, 120ms
        ])
        # 4 scales × 32 = 128 feature maps
        self.attn1 = ChannelAttention(128)

        # ── Dilated temporal convolutions ────────────────
        self.dilated = nn.Sequential(
            TemporalBlock(128, 64, kernel=5, dilation=1),
            TemporalBlock(64,  64, kernel=5, dilation=2),
            TemporalBlock(64,  64, kernel=5, dilation=4),
            TemporalBlock(64,  64, kernel=5, dilation=8),
        )
        self.attn2 = ChannelAttention(64)

        # ── Spatial depthwise convolution across channels ─
        self.spatial = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ELU()
        )

        # ── Classifier head ──────────────────────────────
        self.pool       = nn.AdaptiveAvgPool1d(8)
        self.flatten    = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, cfg.CNN_CLASSES)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N_CH, EPOCH_SAMPLES)"""
        # Multi-scale feature maps
        ms = [conv(x) for conv in self.ms_convs]
        x  = torch.cat(ms, dim=1)              # (B, 128, T)
        x  = self.attn1(x)

        # Dilated convolutions
        x  = self.dilated(x)                   # (B, 64, T)
        x  = self.attn2(x)

        # Spatial
        x  = self.spatial(x)                   # (B, 128, T)

        # Classification
        x  = self.pool(x)                      # (B, 128, 8)
        x  = self.flatten(x)                   # (B, 1024)
        return self.classifier(x)              # (B, 3)

    def get_cam(self, x: torch.Tensor) -> np.ndarray:
        """Class Activation Map for interpretability."""
        self.eval()
        feats = []
        hooks = []

        def hook(module, inp, out):
            feats.append(out.detach())

        h = self.spatial.register_forward_hook(hook)
        with torch.no_grad():
            logits = self.forward(x)
        h.remove()

        weights = self.classifier[0].weight.data  # (256, 1024)
        pred    = logits.argmax(dim=1)[0].item()
        cam     = (weights[pred, :128].unsqueeze(1)
                   * feats[0][0]).mean(0).cpu().numpy()
        return cam


# ── Raw EEG dataset builder ──────────────────────────────────────
def build_cnn_dataset(records: list[dict]
                      ) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract non-overlapping EPOCH_SAMPLES segments + disease labels.

    Returns
    -------
    X : (N, N_CH, EPOCH_SAMPLES)
    y : (N,)
    """
    X_all, y_all = [], []
    ep = cfg.EPOCH_SAMPLES

    for rec in records:
        sig = rec["signals"]                   # (N_CH, n_samp)
        lbl = rec["labels"]["disease"]
        n_ep = sig.shape[1] // ep

        for i in range(n_ep):
            s   = i * ep
            seg = sig[:, s:s + ep]
            dl  = int(lbl[s:s + ep].mean() > 0.5)
            X_all.append(seg)
            y_all.append(dl)

    return (np.array(X_all, dtype=np.float32),
            np.array(y_all, dtype=np.int64))