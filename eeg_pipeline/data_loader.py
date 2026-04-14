"""Data loading utilities for the EEG pipeline."""
# ================================================================
# EDF Data Loader with complete annotation handling
# ================================================================

import os
import numpy as np
import pyedflib
from config import cfg


def load_edf(filepath: str) -> tuple[np.ndarray, int]:
    """
    Load raw EEG from an EDF file.

    Returns
    -------
    signals : (N_CHANNELS, n_samples)  float64
    fs      : sampling frequency
    """
    with pyedflib.EdfReader(filepath) as f:
        fs      = int(f.getSampleFrequency(0))
        n_sig   = min(f.signals_in_file, cfg.N_CHANNELS)
        n_samp  = f.getNSamples()[0]
        signals = np.zeros((cfg.N_CHANNELS, n_samp), dtype=np.float64)
        for i in range(n_sig):
            signals[i, :] = f.readSignal(i)
    return signals, fs


def make_sample_labels(n_samples   : int,
                       intervals   : list,
                       fs          : int,
                       preictal_sec: int = cfg.PREICTAL_SEC
                       ) -> dict[str, np.ndarray]:
    """
    Build three label arrays at sample resolution:

    binary_label   : 0=interictal, 1=ictal
    preictal_label : 0=interictal, 1=pre-ictal, 2=ictal
    disease_label  : 0=normal, 1=focal, 2=generalized (heuristic)
    """
    binary   = np.zeros(n_samples, dtype=np.int32)
    preictal = np.zeros(n_samples, dtype=np.int32)

    for start_s, end_s in intervals:
        s  = int(start_s * fs)
        e  = min(int(end_s * fs), n_samples)
        ps = max(0, s - preictal_sec * fs)

        binary[s:e]   = 1           # ictal
        preictal[ps:s] = 1          # pre-ictal
        preictal[s:e]  = 2          # ictal

    # Simple disease heuristic for chb01:
    # focal if any seizure present in file, else normal
    disease = np.zeros(n_samples, dtype=np.int32)
    if intervals:
        disease[:] = 1              # focal epilepsy

    return {
        "binary"  : binary,
        "preictal": preictal,
        "disease" : disease
    }


def load_dataset(data_dir: str) -> list[dict]:
    """
    Load all EDF files; return a list of record dicts.

    Each record:
      {
        "fname"   : str,
        "signals" : (N_CH, n_samples),
        "labels"  : { "binary", "preictal", "disease" },
        "fs"      : int,
        "seizures": list of (start_s, end_s)
      }
    """
    records = []
    edf_files = sorted(f for f in os.listdir(data_dir)
                       if f.endswith(".edf"))

    print(f"\n{'='*60}")
    print(f"  Loading {len(edf_files)} EDF files from {data_dir}")
    print(f"{'='*60}")

    for fname in edf_files:
        fpath = os.path.join(data_dir, fname)
        try:
            signals, fs = load_edf(fpath)
        except Exception as exc:
            print(f"  [SKIP] {fname}: {exc}")
            continue

        intervals = cfg.SEIZURE_FILES.get(fname, [])
        labels    = make_sample_labels(signals.shape[1],
                                       intervals, fs)
        n_seiz    = labels["binary"].sum()

        records.append({
            "fname"   : fname,
            "signals" : signals,
            "labels"  : labels,
            "fs"      : fs,
            "seizures": intervals
        })

        status = f"  ✓ {fname:<22} | "
        status += f"samples={signals.shape[1]:>7} | "
        status += f"seizure_samples={n_seiz:>6} | "
        status += f"seizures={len(intervals)}"
        print(status)

    print(f"\n  Total records loaded: {len(records)}\n")
    return records