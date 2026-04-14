"""Configuration for the EEG pipeline."""
# ================================================================
# Global Configuration for Multi-Task EEG Analysis Pipeline
# ================================================================

import os
import torch

class Config:
    # ── Paths ────────────────────────────────────────────────────
    DATA_DIR         = "./chb01"
    RESULTS_DIR      = "./results"
    MODEL_DIR        = "./saved_models"
    LOG_DIR          = "./logs"

    # ── EEG Signal Parameters ────────────────────────────────────
    SAMPLING_RATE    = 256          # Hz
    N_CHANNELS       = 18
    CHANNEL_NAMES    = [
        "FP1-F7","F7-T7","T7-P7","P7-O1",
        "FP1-F3","F3-C3","C3-P3","P3-O1",
        "FP2-F4","F4-C4","C4-P4","P4-O2",
        "FP2-F8","F8-T8","T8-P8","P8-O2",
        "FZ-CZ","CZ-PZ"
    ]

    # ── Seizure Annotations (chb01) ──────────────────────────────
    SEIZURE_FILES = {
        "chb01_03.edf": [(2996, 3036)],
        "chb01_04.edf": [(1467, 1494)],
        "chb01_15.edf": [(1732, 1772)],
        "chb01_16.edf": [(1015, 1066)],
        "chb01_18.edf": [(1720, 1810)],
        "chb01_21.edf": [(327,  420)],
        "chb01_26.edf": [(1862, 1963)],
    }

    # ── Preprocessing ────────────────────────────────────────────
    NOTCH_FREQ       = 60.0         # power line noise Hz
    BANDPASS_LOW     = 0.5          # Hz
    BANDPASS_HIGH    = 40.0         # Hz
    BANDPASS_ORDER   = 4
    ARTIFACT_THRESH  = 500.0        # µV clip threshold

    # ── Feature Extraction ───────────────────────────────────────
    EPOCH_LEN_SEC    = 2            # seconds
    EPOCH_SAMPLES    = EPOCH_LEN_SEC * SAMPLING_RATE   # 512
    N_FILTERS        = 8            # filterbank bands
    FREQ_MIN         = 0.5
    FREQ_MAX         = 25.0
    WINDOW_SIZE      = 3            # stacked epochs → 6 s context
    FEATURE_DIM      = N_FILTERS * N_CHANNELS * WINDOW_SIZE  # 432

    # ── SNN Encoder ──────────────────────────────────────────────
    SNN_TIME_STEPS   = 20
    SNN_BETA         = 0.9
    SNN_THRESHOLD    = 1.0
    SNN_HIDDEN       = [256, 128]
    SNN_SPIKE_DIM    = 64           # output spike feature size

    # ── LSTM/Transformer (Seizure Prediction) ────────────────────
    SEQ_LEN          = 30           # 30 epochs = 60 s look-back
    PRED_HORIZON     = 10           # predict 20 s ahead
    LSTM_HIDDEN      = 128
    LSTM_LAYERS      = 2
    TRANSFORMER_HEADS  = 4
    TRANSFORMER_LAYERS = 2
    TRANSFORMER_DIM    = 64
    PREICTAL_SEC     = 300          # 5 min pre-seizure label window

    # ── CNN (Disease Classification) ─────────────────────────────
    CNN_CLASSES      = 3            # normal / focal / generalized
    CNN_DROPOUT      = 0.3

    # ── Random Forest ────────────────────────────────────────────
    RF_N_ESTIMATORS  = 300
    RF_MAX_DEPTH     = 15
    RF_N_FEATURES    = "sqrt"

    # ── Autoencoder (Anomaly Detection) ──────────────────────────
    AE_LATENT_DIM    = 32
    AE_THRESHOLD_PCT = 95           # percentile for anomaly thresh

    # ── Training ─────────────────────────────────────────────────
    BATCH_SIZE       = 64
    EPOCHS           = 50
    LEARNING_RATE    = 1e-3
    WEIGHT_DECAY     = 1e-4
    PATIENCE         = 10
    CLASS_WEIGHT_SEI = 10.0         # seizure class up-weight

    # ── Device ───────────────────────────────────────────────────
    DEVICE = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")

    def __init__(self):
        for d in [self.RESULTS_DIR, self.MODEL_DIR, self.LOG_DIR]:
            os.makedirs(d, exist_ok=True)

cfg = Config()
