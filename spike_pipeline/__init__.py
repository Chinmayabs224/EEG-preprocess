"""
spike_pipeline — Large-scale EEG spike encoding pipeline for CHB-MIT dataset.

Stages (matching the EEG-SNN Hybrid Pipeline architecture):
    1. Raw EEG Input     — recursive EDF discovery + chunk-based loading
    2. Preprocessing     — bandpass, notch, CAR, z-score
    3. Windowing+Labeling— 2-sec epochs with seizure annotations
    4. Spike Encoding    — rate coding / LIF encoding / temporal coding
    5. SNN Feature Ext.  — LIF layers → membrane potential → spike trains (T×64)
    6. Save + Visualize  — per-file .npz output + 7 verification plot types
"""

__version__ = "1.0.0"
