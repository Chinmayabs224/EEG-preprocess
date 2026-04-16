#!/usr/bin/env python3
"""
Inference script: test trained models on new .npz files or an entire directory.

Usage::

    # Test all 4 models on an entire directory of .npz files
    python -m training.predict \
        --spikes_dir ~/EEG-preprocess/output/spikes \
        --models_dir ~/EEG-preprocess/models \
        --plots_dir  ~/EEG-preprocess/test_plots

    # Test on a single .npz file
    python -m training.predict \
        --npz_file ~/EEG-preprocess/output/spikes/chb01/chb01_03.npz \
        --models_dir ~/EEG-preprocess/models

    # Test only specific models
    python -m training.predict \
        --spikes_dir ~/EEG-preprocess/output/spikes \
        --models_dir ~/EEG-preprocess/models \
        --models lstm transformer
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test/Infer trained EEG seizure detection models on spike data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--spikes_dir", type=str,
        help="Directory containing .npz files (all will be tested)",
    )
    group.add_argument(
        "--npz_file", type=str,
        help="Path to a single .npz file to run inference on",
    )

    parser.add_argument(
        "--models_dir", type=str, default="./models",
        help="Directory containing trained model .pt files (default: ./models)",
    )
    parser.add_argument(
        "--plots_dir", type=str, default="./test_plots",
        help="Directory to save output plots (default: ./test_plots)",
    )
    parser.add_argument(
        "--models", nargs="+",
        default=["autoencoder", "lstm", "cnn", "transformer"],
        choices=["autoencoder", "lstm", "cnn", "transformer"],
        help="Which models to use for inference",
    )
    parser.add_argument(
        "--seq_len", type=int, default=30,
        help="Sequence length used during training (default: 30)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size for inference (default: 64)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Decision threshold for seizure prediction (default: 0.5)",
    )
    parser.add_argument(
        "--max_files", type=int, default=None,
        help="Limit number of .npz files to test (for quick checks)",
    )

    return parser.parse_args()


# ═════════════════════════════════════════════════════════════════
# Model Loading
# ═════════════════════════════════════════════════════════════════

def load_trained_models(models_dir: str, model_names: list, device: torch.device) -> dict:
    """Load saved model weights from models_dir."""
    from .model_autoencoder import EEGAutoencoder
    from .model_lstm import EEGLSTMClassifier
    from .model_cnn import EEGCNNClassifier
    from .model_transformer import EEGTransformerClassifier

    arch_map = {
        "autoencoder": lambda: EEGAutoencoder(seq_len=30, feature_dim=64, latent_dim=32),
        "lstm":        lambda: EEGLSTMClassifier(input_dim=64, hidden_dim=128, num_layers=2),
        "cnn":         lambda: EEGCNNClassifier(input_dim=64),
        "transformer": lambda: EEGTransformerClassifier(input_dim=64, d_model=128, nhead=4, num_layers=4),
    }

    loaded = {}
    for name in model_names:
        pt_path = os.path.join(models_dir, f"{name}_best.pt")
        if not os.path.exists(pt_path):
            logger.warning(f"  ✗ Model weights not found: {pt_path} — skipping {name}")
            continue

        model = arch_map[name]()
        model.load_state_dict(torch.load(pt_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        loaded[name] = model
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  ✓ Loaded {name} ({n_params:,} params) from {pt_path}")

    return loaded


# ═════════════════════════════════════════════════════════════════
# Single file inference
# ═════════════════════════════════════════════════════════════════

def predict_single_file(
    npz_path: str,
    models: dict,
    device: torch.device,
    seq_len: int = 30,
    threshold: float = 0.5,
    batch_size: int = 64,
) -> dict:
    """
    Run inference on a single .npz file.

    Returns predictions dict with:
        - per-model probabilities
        - ensemble prediction
        - window-level timeline
    """
    data = np.load(npz_path, allow_pickle=True)

    if "snn_features" not in data:
        raise ValueError(f"No snn_features in {npz_path}. Keys: {list(data.files)}")

    features = data["snn_features"].astype(np.float32)  # (N, 64)
    true_labels = data.get("snn_labels", None)
    if true_labels is not None:
        true_labels = true_labels.astype(int)

    file_has_seizure = bool(data.get("has_seizure", False))
    n_windows = len(features)

    if n_windows < seq_len:
        raise ValueError(f"File too short: {n_windows} windows < seq_len={seq_len}")

    # Build sequences
    sequences = np.array([
        features[i:i + seq_len]
        for i in range(n_windows - seq_len + 1)
    ])  # (M, seq_len, 64)

    seq_labels = None
    if true_labels is not None:
        seq_labels = np.array([
            true_labels[i + seq_len - 1]
            for i in range(n_windows - seq_len + 1)
        ])

    predictions = {}
    all_probs = {}

    for name, model in models.items():
        model_probs = []

        for i in range(0, len(sequences), batch_size):
            batch = torch.tensor(sequences[i:i + batch_size], dtype=torch.float32).to(device)

            with torch.no_grad():
                if name == "autoencoder":
                    errors = model.anomaly_score(batch).cpu().numpy()
                    model_probs.extend(errors)
                elif name == "cnn":
                    # CNN uses last window of each sequence
                    last_windows = batch[:, -1, :]  # (B, 64)
                    logits = model(last_windows)
                    probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    model_probs.extend(probs)
                else:
                    # LSTM / Transformer
                    logits = model(batch)
                    probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    model_probs.extend(probs)

        model_probs = np.array(model_probs)

        # Normalize autoencoder to [0, 1]
        if name == "autoencoder" and model_probs.max() > model_probs.min():
            model_probs = (model_probs - model_probs.min()) / (model_probs.max() - model_probs.min())

        all_probs[name] = model_probs
        predictions[f"{name}_prob"] = model_probs
        predictions[f"{name}_pred"] = (model_probs >= threshold).astype(int)

    # Ensemble (equal weights if no tuning info available)
    if len(all_probs) > 0:
        ensemble_prob = np.mean(list(all_probs.values()), axis=0)
        predictions["ensemble_prob"] = ensemble_prob
        predictions["ensemble_pred"] = (ensemble_prob >= threshold).astype(int)

    predictions["true_labels"] = seq_labels
    predictions["file_has_seizure"] = file_has_seizure
    predictions["n_sequences"] = len(sequences)
    predictions["seq_len"] = seq_len
    predictions["npz_path"] = npz_path

    return predictions


# ═════════════════════════════════════════════════════════════════
# Visualization
# ═════════════════════════════════════════════════════════════════

def plot_prediction_timeline(
    predictions: dict,
    save_path: str,
    file_name: str = "",
):
    """
    Plot per-window seizure probability over time for each model.
    Shows predicted probability as a line, with shaded seizure regions.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    model_keys = [k.replace("_prob", "") for k in predictions if k.endswith("_prob")]
    n_plots = len(model_keys)

    fig, axes = plt.subplots(n_plots, 1, figsize=(16, 4 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    true_labels = predictions.get("true_labels")
    colors = {
        "lstm": "#4C72B0",
        "cnn": "#DD8452",
        "transformer": "#55A868",
        "autoencoder": "#C44E52",
        "ensemble": "#8172B2",
    }

    for ax, name in zip(axes, model_keys):
        prob = predictions[f"{name}_prob"]
        x = np.arange(len(prob))

        # Plot probability line
        color = colors.get(name, "#333333")
        ax.plot(x, prob, lw=1.5, color=color, label=f"{name.upper()} Probability")
        ax.fill_between(x, prob, alpha=0.2, color=color)

        # Horizontal threshold line
        ax.axhline(0.5, color="k", lw=1, ls="--", alpha=0.5, label="Threshold (0.5)")

        # Shade true seizure regions
        if true_labels is not None:
            for i, lbl in enumerate(true_labels):
                if lbl == 1:
                    ax.axvspan(i - 0.5, i + 0.5, alpha=0.15, color="red")

        ax.set_ylabel("Seizure Probability", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{name.upper()} — Window-Level Predictions", fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

        # Mark predicted seizures
        pred = predictions[f"{name}_pred"]
        seizure_windows = np.where(pred == 1)[0]
        if len(seizure_windows) > 0:
            ax.scatter(seizure_windows, prob[seizure_windows],
                      color="red", s=20, zorder=5, alpha=0.7, label="Predicted Seizure")

    axes[-1].set_xlabel("Window Index (each = 60s chunk)", fontsize=11)
    red_patch = mpatches.Patch(color="red", alpha=0.15, label="True Seizure Region")
    fig.legend(handles=[red_patch], loc="upper left", fontsize=10)
    fig.suptitle(
        f"Seizure Prediction Timeline — {file_name}",
        fontsize=14, fontweight="bold", y=1.01,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved timeline plot: {save_path}")


def plot_ensemble_summary(
    all_file_results: list,
    save_path: str,
):
    """
    Across all test files: plot seizure detection rate per subject.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subjects = {}
    for r in all_file_results:
        subj = Path(r["npz_path"]).parent.name
        if subj not in subjects:
            subjects[subj] = {"total": 0, "detected": 0, "true_seizure": 0}
        subjects[subj]["total"] += 1
        if r["file_has_seizure"]:
            subjects[subj]["true_seizure"] += 1
        if "ensemble_pred" in r and r["ensemble_pred"].any():
            subjects[subj]["detected"] += 1

    subj_names = sorted(subjects.keys())
    detection_rates = [
        subjects[s]["detected"] / max(subjects[s]["true_seizure"], 1)
        for s in subj_names
    ]

    fig, ax = plt.subplots(figsize=(16, 5))
    bars = ax.bar(subj_names, detection_rates, color="#4C72B0", alpha=0.85, edgecolor="white")
    ax.axhline(1.0, color="green", ls="--", lw=1.5, label="Perfect detection")
    ax.set_xlabel("Subject", fontsize=12)
    ax.set_ylabel("Seizure Detection Rate", fontsize=12)
    ax.set_title("Ensemble — Seizure Detection Rate per Subject", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.2)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved subject summary: {save_path}")


# ═════════════════════════════════════════════════════════════════
# Evaluation metrics across all files
# ═════════════════════════════════════════════════════════════════

def aggregate_metrics(all_file_results: list, model_names: list) -> dict:
    """Aggregate predictions across all files and compute global metrics."""
    from .utils import compute_metrics

    agg = {}
    for name in model_names + ["ensemble"]:
        key_prob = f"{name}_prob"
        key_pred = f"{name}_pred"
        all_probs, all_preds, all_true = [], [], []

        for r in all_file_results:
            if key_prob not in r or r.get("true_labels") is None:
                continue
            all_probs.extend(r[key_prob])
            all_preds.extend(r[key_pred])
            all_true.extend(r["true_labels"])

        if not all_true:
            continue

        all_true = np.array(all_true)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)

        if len(np.unique(all_true)) > 1:
            metrics = compute_metrics(all_true, all_preds, all_probs)
        else:
            metrics = compute_metrics(all_true, all_preds)

        agg[name] = metrics

    return agg


# ═════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.plots_dir, exist_ok=True)

    print()
    print("█" * 60)
    print("█  EEG SEIZURE DETECTION — INFERENCE / TESTING")
    print("█" + "═" * 58 + "█")
    print(f"█  Models dir  : {args.models_dir}")
    print(f"█  Device      : {device}")
    print(f"█  Threshold   : {args.threshold}")
    print(f"█  Models      : {', '.join(args.models)}")
    print("█" * 60)
    print()

    # ── Load models ──────────────────────────────────────────────
    logger.info("Loading trained models...")
    models = load_trained_models(args.models_dir, args.models, device)

    if not models:
        print("\n✗ No trained models found. Run train_all.py first.")
        sys.exit(1)

    # ── Collect files ─────────────────────────────────────────────
    if args.npz_file:
        npz_files = [Path(args.npz_file)]
    else:
        npz_files = sorted(Path(args.spikes_dir).rglob("*.npz"))
        if args.max_files:
            npz_files = npz_files[:args.max_files]

    logger.info(f"Running inference on {len(npz_files)} file(s)...")

    # ── Run inference ─────────────────────────────────────────────
    all_results = []
    failed = 0

    for i, npz_path in enumerate(npz_files):
        try:
            result = predict_single_file(
                str(npz_path), models, device,
                seq_len=args.seq_len,
                threshold=args.threshold,
                batch_size=args.batch_size,
            )
            all_results.append(result)

            # Print per-file summary
            flag = "🔴 SEIZURE" if result["file_has_seizure"] else "🟢 Normal"
            ensemble_any = bool(result.get("ensemble_pred", np.array([0])).any())
            detected = "⚡ DETECTED" if ensemble_any else "  clear"
            print(f"  [{i+1:03d}/{len(npz_files)}] {npz_path.name:<35} {flag}  {detected}")

            # Save timeline plot (first 10 files OR seizure files)
            if i < 10 or result["file_has_seizure"]:
                timeline_path = os.path.join(
                    args.plots_dir, f"timeline_{npz_path.stem}.png"
                )
                plot_prediction_timeline(result, timeline_path, file_name=npz_path.name)

        except Exception as e:
            logger.warning(f"  ✗ Failed {npz_path.name}: {e}")
            failed += 1

    if not all_results:
        print("\n✗ All files failed. Check your .npz files.")
        sys.exit(1)

    # ── Aggregate metrics ────────────────────────────────────────
    logger.info("\nAggregating metrics across all files...")
    agg_metrics = aggregate_metrics(all_results, list(models.keys()))

    # ── Comparison chart ─────────────────────────────────────────
    if len(agg_metrics) > 1:
        from .utils import plot_model_comparison
        plot_model_comparison(
            agg_metrics,
            save_path=os.path.join(args.plots_dir, "model_comparison.png"),
        )

    # ── ROC curves ───────────────────────────────────────────────
    from .utils import plot_roc_curve
    for name, metrics in agg_metrics.items():
        # Collect probs and true labels again for ROC
        key_prob = f"{name}_prob"
        all_probs, all_true = [], []
        for r in all_results:
            if key_prob in r and r.get("true_labels") is not None:
                all_probs.extend(r[key_prob])
                all_true.extend(r["true_labels"])
        if all_probs and len(np.unique(all_true)) > 1:
            plot_roc_curve(
                np.array(all_true), np.array(all_probs),
                save_path=os.path.join(args.plots_dir, f"roc_{name}.png"),
                model_name=name.upper(),
            )

    # ── Subject-level summary ────────────────────────────────────
    if len(all_results) > 5:
        plot_ensemble_summary(
            all_results,
            save_path=os.path.join(args.plots_dir, "subject_detection_rate.png"),
        )

    # ── Final table ───────────────────────────────────────────────
    print("\n")
    print("█" * 65)
    print("█  TEST RESULTS — AGGREGATED ACROSS ALL FILES")
    print("█" * 65)
    print(f"\n{'Model':<15} {'Accuracy':>10} {'F1':>10} {'Recall':>10} {'AUC-ROC':>10} {'Precision':>10}")
    print("─" * 65)

    for name in list(models.keys()) + ["ensemble"]:
        if name in agg_metrics:
            m = agg_metrics[name]
            print(
                f"{name:<15} "
                f"{m.get('accuracy', 0):>10.4f} "
                f"{m.get('f1', 0):>10.4f} "
                f"{m.get('recall', 0):>10.4f} "
                f"{m.get('auc_roc', 0):>10.4f} "
                f"{m.get('precision', 0):>10.4f}"
            )

    print("─" * 65)
    print(f"\n  Files tested: {len(all_results)} | Failed: {failed}")
    print(f"  Plots saved : {args.plots_dir}")
    print()

    # Save test results JSON
    results_path = os.path.join(args.plots_dir, "test_results.json")
    with open(results_path, "w") as f:
        clean = {
            k: {
                kk: float(vv) if isinstance(vv, (float, np.floating)) else int(vv)
                for kk, vv in v.items()
            }
            for k, v in agg_metrics.items()
        }
        json.dump(clean, f, indent=2)
    logger.info(f"Test results saved: {results_path}")


if __name__ == "__main__":
    main()
