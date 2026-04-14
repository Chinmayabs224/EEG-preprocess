"""LSTM-Transformer model for seizure prediction."""
# ================================================================
# Task 1: Seizure PREDICTION (future)
#
# Input : sequence of SNN spike-rate features
#         shape (batch, SEQ_LEN, SNN_SPIKE_DIM)
# Output: probability of seizure in next PRED_HORIZON epochs
#
# Dual head: LSTM branch + Transformer branch → fusion → logit
# ================================================================

import torch
import torch.nn as nn
import numpy as np
from config import cfg


# ── Positional Encoding ──────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


# ── LSTM Branch ──────────────────────────────────────────────────
class LSTMBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size   = cfg.SNN_SPIKE_DIM,
            hidden_size  = cfg.LSTM_HIDDEN,
            num_layers   = cfg.LSTM_LAYERS,
            batch_first  = True,
            dropout      = 0.3,
            bidirectional= True
        )
        self.out_dim = cfg.LSTM_HIDDEN * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (h, _) = self.lstm(x)         # out: (B,T,2H)
        # Concatenate last forward + backward hidden
        return torch.cat([h[-2], h[-1]], dim=-1)  # (B, 2H)


# ── Transformer Branch ───────────────────────────────────────────
class TransformerBranch(nn.Module):
    def __init__(self):
        super().__init__()
        d = cfg.TRANSFORMER_DIM
        self.proj   = nn.Linear(cfg.SNN_SPIKE_DIM, d)
        self.pe     = PositionalEncoding(d)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d,
            nhead           = cfg.TRANSFORMER_HEADS,
            dim_feedforward = d * 4,
            dropout         = 0.2,
            batch_first     = True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers = cfg.TRANSFORMER_LAYERS
        )
        self.out_dim = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                    # (B, T, d)
        x = self.pe(x)
        x = self.encoder(x)                 # (B, T, d)
        return x.mean(dim=1)                # (B, d)


# ── Fusion Classifier ────────────────────────────────────────────
class SeizurePredictionModel(nn.Module):
    """
    Dual-branch LSTM + Transformer seizure prediction model.

    Input  : (batch, SEQ_LEN, SNN_SPIKE_DIM)
    Output : (batch, 3)   logits for [interictal, pre-ictal, ictal]
    """

    def __init__(self):
        super().__init__()
        self.lstm_branch = LSTMBranch()
        self.trans_branch= TransformerBranch()

        fused = self.lstm_branch.out_dim + \
                self.trans_branch.out_dim      # 256 + 64 = 320

        self.fusion = nn.Sequential(
            nn.LayerNorm(fused),
            nn.Linear(fused, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)               # 3 classes
        )

        # Attention weights for interpretability
        self.attn_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_feat  = self.lstm_branch(x)   # (B, 256)
        trans_feat = self.trans_branch(x)  # (B, 64)
        fused      = torch.cat([lstm_feat, trans_feat], dim=-1)
        return self.fusion(fused)          # (B, 3)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)


# ── Sequence Dataset Builder ─────────────────────────────────────
def build_sequences(spike_features : np.ndarray,
                    preictal_labels: np.ndarray,
                    seq_len        : int = cfg.SEQ_LEN,
                    horizon        : int = cfg.PRED_HORIZON
                    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Create sliding sequences for temporal prediction.

    Parameters
    ----------
    spike_features  : (N, SNN_SPIKE_DIM)
    preictal_labels : (N,)  0=interictal, 1=preictal, 2=ictal
    seq_len         : look-back window
    horizon         : how many steps ahead to predict

    Returns
    -------
    X_seq : (n_seq, seq_len, SNN_SPIKE_DIM)
    y_seq : (n_seq,)  label at t+horizon
    """
    X_list, y_list = [], []
    for i in range(seq_len, len(spike_features) - horizon):
        X_list.append(spike_features[i - seq_len:i])
        y_list.append(preictal_labels[i + horizon])
    return (np.array(X_list, dtype=np.float32),
            np.array(y_list, dtype=np.int64))