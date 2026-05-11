#!/usr/bin/env python3
"""
Inference script: test trained models on new .npz files or an entire directory.

Enhanced with:
    - Temporal post-processing (median smoothing + persistence filter)
    - Feature augmentation support
    - AUC-PR metrics
    - Precision-Recall curve plotting

Usage::
    python -m training.predict --spikes_dir ./output/spikes --models_dir ./models
    python -m training.predict --npz_file ./output/spikes/chb01/chb01_03.npz --models_dir ./models
    python -m training.predict --spikes_dir ./output/spikes --models_dir ./models --smooth_window 5 --persistence_k 3
"""

import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Test/Infer trained EEG seizure detection models.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--spikes_dir", type=str)
    group.add_argument("--npz_file", type=str)
    p.add_argument("--models_dir", type=str, default="./models")
    p.add_argument("--plots_dir", type=str, default="./test_plots")
    p.add_argument("--models", nargs="+", default=["autoencoder", "lstm", "cnn", "transformer"],
                   choices=["autoencoder", "lstm", "cnn", "transformer"])
    p.add_argument("--seq_len", type=int, default=10, help="Sequence length (default: 10)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--max_files", type=int, default=None)
    # New flags
    p.add_argument("--augment_features", action="store_true", help="Use augmented features (75 dim)")
    p.add_argument("--anomaly_method", default="deep_svdd", choices=["deep_svdd", "autoencoder"])
    p.add_argument("--cnn_mode", default="spatial", choices=["spatial", "flat"])
    p.add_argument("--smooth_window", type=int, default=5, help="Temporal median filter window")
    p.add_argument("--persistence_k", type=int, default=3, help="Persistence filter: min positive windows")
    p.add_argument("--persistence_n", type=int, default=5, help="Persistence filter: look-back window")
    return p.parse_args()


def load_trained_models(models_dir, model_names, device, seq_len=10, feature_dim=64,
                        anomaly_method="deep_svdd", cnn_mode="spatial"):
    """Load saved model weights from models_dir."""
    from .model_autoencoder import EEGAutoencoder, DeepSVDD
    from .model_lstm import EEGLSTMClassifier
    from .model_cnn import EEGCNNClassifier, EEGSpatialCNN
    from .model_transformer import EEGTransformerClassifier

    arch_map = {
        "autoencoder": lambda: (DeepSVDD(seq_len=seq_len, feature_dim=feature_dim, latent_dim=32)
                                if anomaly_method == "deep_svdd"
                                else EEGAutoencoder(seq_len=seq_len, feature_dim=feature_dim, latent_dim=32)),
        "lstm":        lambda: EEGLSTMClassifier(input_dim=feature_dim, hidden_dim=128, num_layers=2),
        "cnn":         lambda: (EEGSpatialCNN(input_dim=feature_dim) if cnn_mode == "spatial"
                                else EEGCNNClassifier(input_dim=feature_dim)),
        "transformer": lambda: EEGTransformerClassifier(input_dim=feature_dim, d_model=128, nhead=4,
                                                        num_layers=4, max_seq_len=seq_len + 10),
    }

    loaded = {}
    for name in model_names:
        pt_path = os.path.join(models_dir, f"{name}_best.pt")
        if not os.path.exists(pt_path):
            logger.warning(f"  ✗ Not found: {pt_path} — skipping {name}")
            continue
        model = arch_map[name]()
        model.load_state_dict(torch.load(pt_path, map_location=device, weights_only=True))
        model.to(device); model.eval()
        loaded[name] = model
        logger.info(f"  ✓ Loaded {name} ({sum(p.numel() for p in model.parameters()):,} params)")
    return loaded


def predict_single_file(npz_path, models, device, seq_len=10, threshold=0.5, batch_size=64,
                        augmentor=None, temporal_processor=None):
    """Run inference on a single .npz file with optional temporal smoothing."""
    data = np.load(npz_path, allow_pickle=True)
    if "snn_features" not in data:
        raise ValueError(f"No snn_features in {npz_path}")

    features = data["snn_features"].astype(np.float32)
    true_labels = data.get("snn_labels", None)
    if true_labels is not None:
        true_labels = true_labels.astype(int)

    # Apply feature augmentation
    if augmentor is not None:
        features = augmentor.transform(features)

    file_has_seizure = bool(data.get("has_seizure", False))
    n_windows = len(features)
    if n_windows < seq_len:
        raise ValueError(f"File too short: {n_windows} < seq_len={seq_len}")

    sequences = np.array([features[i:i + seq_len] for i in range(n_windows - seq_len + 1)])
    seq_labels = None
    if true_labels is not None:
        seq_labels = np.array([true_labels[i + seq_len - 1] for i in range(n_windows - seq_len + 1)])

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
                    last_windows = batch[:, -1, :]
                    logits = model(last_windows)
                    probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    model_probs.extend(probs)
                else:
                    logits = model(batch)
                    probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    model_probs.extend(probs)

        model_probs = np.array(model_probs)
        if name == "autoencoder" and model_probs.max() > model_probs.min():
            model_probs = (model_probs - model_probs.min()) / (model_probs.max() - model_probs.min())

        all_probs[name] = model_probs
        predictions[f"{name}_prob"] = model_probs
        predictions[f"{name}_pred"] = (model_probs >= threshold).astype(int)

    # Ensemble
    if len(all_probs) > 0:
        ensemble_prob = np.mean(list(all_probs.values()), axis=0)

        # Apply temporal post-processing
        if temporal_processor is not None:
            smoothed, filtered = temporal_processor.process(ensemble_prob)
            predictions["ensemble_prob"] = smoothed
            predictions["ensemble_pred"] = filtered
            predictions["ensemble_prob_raw"] = ensemble_prob
            predictions["ensemble_pred_raw"] = (ensemble_prob >= threshold).astype(int)
        else:
            predictions["ensemble_prob"] = ensemble_prob
            predictions["ensemble_pred"] = (ensemble_prob >= threshold).astype(int)

    predictions["true_labels"] = seq_labels
    predictions["file_has_seizure"] = file_has_seizure
    predictions["n_sequences"] = len(sequences)
    predictions["seq_len"] = seq_len
    predictions["npz_path"] = npz_path
    return predictions


def plot_prediction_timeline(predictions, save_path, file_name=""):
    """Plot per-window seizure probability over time for each model."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    model_keys = [k.replace("_prob", "") for k in predictions if k.endswith("_prob") and "_raw" not in k]
    n_plots = len(model_keys)
    fig, axes = plt.subplots(n_plots, 1, figsize=(16, 4 * n_plots), sharex=True)
    if n_plots == 1: axes = [axes]

    true_labels = predictions.get("true_labels")
    colors = {"lstm": "#4C72B0", "cnn": "#DD8452", "transformer": "#55A868",
              "autoencoder": "#C44E52", "ensemble": "#8172B2"}

    for ax, name in zip(axes, model_keys):
        prob = predictions[f"{name}_prob"]
        x = np.arange(len(prob))
        color = colors.get(name, "#333333")
        ax.plot(x, prob, lw=1.5, color=color, label=f"{name.upper()} Prob")
        ax.fill_between(x, prob, alpha=0.2, color=color)
        ax.axhline(0.5, color="k", lw=1, ls="--", alpha=0.5, label="Threshold")
        if true_labels is not None:
            for i, lbl in enumerate(true_labels):
                if lbl == 1: ax.axvspan(i - 0.5, i + 0.5, alpha=0.15, color="red")
        ax.set_ylabel("Seizure Prob", fontsize=10); ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{name.upper()} — Predictions", fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9); ax.grid(True, alpha=0.3)
        pred = predictions.get(f"{name}_pred")
        if pred is not None:
            sz = np.where(pred == 1)[0]
            if len(sz) > 0: ax.scatter(sz, prob[sz], color="red", s=20, zorder=5, alpha=0.7)

    axes[-1].set_xlabel("Window Index (each ≈ 60s)", fontsize=11)
    fig.suptitle(f"Seizure Prediction — {file_name}", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()


def aggregate_metrics(all_file_results, model_names):
    """Aggregate predictions across all files and compute global metrics."""
    from .utils import compute_metrics
    agg = {}
    for name in model_names + ["ensemble"]:
        key_prob, key_pred = f"{name}_prob", f"{name}_pred"
        all_probs, all_preds, all_true = [], [], []
        for r in all_file_results:
            if key_prob not in r or r.get("true_labels") is None: continue
            all_probs.extend(r[key_prob]); all_preds.extend(r[key_pred]); all_true.extend(r["true_labels"])
        if not all_true: continue
        all_true, all_preds, all_probs = np.array(all_true), np.array(all_preds), np.array(all_probs)
        agg[name] = compute_metrics(all_true, all_preds, all_probs) if len(np.unique(all_true)) > 1 else compute_metrics(all_true, all_preds)
    return agg


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.plots_dir, exist_ok=True)

    feature_dim = 75 if args.augment_features else 64

    print(f"\n{'█' * 60}")
    print("█  EEG SEIZURE DETECTION — ENHANCED INFERENCE")
    print(f"█  Device: {device} | Temporal: smooth={args.smooth_window}, persist={args.persistence_k}/{args.persistence_n}")
    print(f"{'█' * 60}\n")

    # Load models
    models = load_trained_models(args.models_dir, args.models, device, seq_len=args.seq_len,
                                 feature_dim=feature_dim, anomaly_method=args.anomaly_method,
                                 cnn_mode=args.cnn_mode)
    if not models:
        print("\n✗ No trained models found. Run train_all.py first."); sys.exit(1)

    # Feature augmentor (needs fitting - load from saved or fit on test data stats)
    augmentor = None
    if args.augment_features:
        from .feature_augmentor import FeatureAugmentor
        augmentor = FeatureAugmentor()
        # Will be fitted on first file's data as approximation
        augmentor._needs_fit = True

    # Temporal post-processor
    from .temporal_filter import TemporalPostProcessor
    temporal = TemporalPostProcessor(
        smooth_window=args.smooth_window, persistence_k=args.persistence_k,
        persistence_n=args.persistence_n, threshold=args.threshold,
    )

    # Collect files
    npz_files = [Path(args.npz_file)] if args.npz_file else sorted(Path(args.spikes_dir).rglob("*.npz"))
    if args.max_files: npz_files = npz_files[:args.max_files]
    logger.info(f"Running inference on {len(npz_files)} file(s)...")

    all_results = []; failed = 0
    for i, npz_path in enumerate(npz_files):
        try:
            # Fit augmentor on first file if needed
            if augmentor is not None and hasattr(augmentor, '_needs_fit') and augmentor._needs_fit:
                d = np.load(str(npz_path), allow_pickle=True)
                if "snn_features" in d:
                    augmentor.fit(d["snn_features"].astype(np.float32))
                    augmentor._needs_fit = False

            result = predict_single_file(
                str(npz_path), models, device, seq_len=args.seq_len,
                threshold=args.threshold, batch_size=args.batch_size,
                augmentor=augmentor, temporal_processor=temporal,
            )
            all_results.append(result)

            flag = "🔴 SEIZURE" if result["file_has_seizure"] else "🟢 Normal"
            ensemble_any = bool(result.get("ensemble_pred", np.array([0])).any())
            detected = "⚡ DETECTED" if ensemble_any else "  clear"
            print(f"  [{i+1:03d}/{len(npz_files)}] {npz_path.name:<35} {flag}  {detected}")

            if i < 10 or result["file_has_seizure"]:
                plot_prediction_timeline(result, os.path.join(args.plots_dir, f"timeline_{npz_path.stem}.png"),
                                         file_name=npz_path.name)
        except Exception as e:
            logger.warning(f"  ✗ Failed {npz_path.name}: {e}"); failed += 1

    if not all_results:
        print("\n✗ All files failed."); sys.exit(1)

    # Aggregate metrics
    agg_metrics = aggregate_metrics(all_results, list(models.keys()))

    # PR curves
    from .utils import plot_precision_recall_curve, plot_roc_curve
    for name in agg_metrics:
        key_prob = f"{name}_prob"
        all_probs, all_true = [], []
        for r in all_results:
            if key_prob in r and r.get("true_labels") is not None:
                all_probs.extend(r[key_prob]); all_true.extend(r["true_labels"])
        if all_probs and len(np.unique(all_true)) > 1:
            plot_precision_recall_curve(np.array(all_true), np.array(all_probs),
                                        save_path=os.path.join(args.plots_dir, f"pr_{name}.png"), model_name=name.upper())
            plot_roc_curve(np.array(all_true), np.array(all_probs),
                           save_path=os.path.join(args.plots_dir, f"roc_{name}.png"), model_name=name.upper())

    # Final table
    print(f"\n\n{'█' * 75}")
    print("█  TEST RESULTS — AGGREGATED")
    print(f"{'█' * 75}")
    print(f"\n{'Model':<15} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC-PR':>10} {'AUC-ROC':>10}")
    print("─" * 75)
    for name in list(models.keys()) + ["ensemble"]:
        if name in agg_metrics:
            m = agg_metrics[name]
            print(f"{name:<15} {m.get('accuracy',0):>10.4f} {m.get('precision',0):>10.4f} "
                  f"{m.get('recall',0):>10.4f} {m.get('f1',0):>10.4f} "
                  f"{m.get('auc_pr',0):>10.4f} {m.get('auc_roc',0):>10.4f}")
    print(f"{'─' * 75}\n  Files: {len(all_results)} | Failed: {failed} | Plots: {args.plots_dir}\n")

    with open(os.path.join(args.plots_dir, "test_results.json"), "w") as f:
        clean = {k: {kk: float(vv) if isinstance(vv, (float, np.floating)) else int(vv)
                      for kk, vv in v.items()} for k, v in agg_metrics.items()}
        json.dump(clean, f, indent=2)


if __name__ == "__main__":
    main()
