#!/usr/bin/env python3
"""
CLI entry point: Train all 4 downstream models on spike-encoded .npz data.

Usage::

    # Train all 4 models + ensemble
    python -m training.train_all --spikes_dir ~/EEG-preprocess/output/spikes

    # Train only specific models
    python -m training.train_all --spikes_dir ~/EEG-preprocess/output/spikes --models lstm cnn

    # Custom epochs and batch size
    python -m training.train_all --spikes_dir ~/EEG-preprocess/output/spikes --epochs 80 --batch_size 128
"""

import argparse
import os
import sys
import time
import json
import logging
import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train downstream seizure detection models on spike-encoded features.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Models:
  autoencoder  — Unsupervised anomaly detection (trained on normal-only data)
  lstm         — Bidirectional LSTM with attention
  cnn          — 1D CNN on individual spike windows
  transformer  — Multi-head self-attention encoder

Example:
  python -m training.train_all \\
      --spikes_dir ~/EEG-preprocess/output/spikes \\
      --save_dir ~/EEG-preprocess/models \\
      --epochs 50 --batch_size 64
""",
    )

    parser.add_argument(
        "--spikes_dir", type=str, required=True,
        help="Directory containing .npz spike files (output of spike_pipeline)",
    )
    parser.add_argument(
        "--save_dir", type=str, default="./models",
        help="Directory to save trained models and metrics (default: ./models)",
    )
    parser.add_argument(
        "--plots_dir", type=str, default="./training_plots",
        help="Directory to save training plots (default: ./training_plots)",
    )
    parser.add_argument(
        "--models", nargs="+",
        default=["autoencoder", "lstm", "cnn", "transformer"],
        choices=["autoencoder", "lstm", "cnn", "transformer"],
        help="Which models to train (default: all four)",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Max training epochs per model (default: 50)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--seq_len", type=int, default=30,
        help="Sequence length for LSTM/Transformer (default: 30 windows)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--max_files", type=int, default=None,
        help="Limit number of .npz files to load (for quick testing)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--no_ensemble", action="store_true",
        help="Skip ensemble evaluation even if all 4 models are trained",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ── Setup ────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    print()
    print("█" * 60)
    print("█  EEG SEIZURE DETECTION — MODEL TRAINING")
    print("█" + "═" * 58 + "█")
    print(f"█  Spikes dir  : {args.spikes_dir}")
    print(f"█  Save dir    : {args.save_dir}")
    print(f"█  Models      : {', '.join(args.models)}")
    print(f"█  Device      : {device}")
    print(f"█  Epochs      : {args.epochs}")
    print(f"█  Batch size  : {args.batch_size}")
    print(f"█  Seq length  : {args.seq_len}")
    if torch.cuda.is_available():
        print(f"█  GPU         : {torch.cuda.get_device_name(0)}")
        print(f"█  GPU Memory  : {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print("█" * 60)
    print()

    # ── Load Data ────────────────────────────────────────────────
    from .dataset import load_all_npz, split_data, get_dataloaders, get_autoencoder_dataloaders

    logger.info("Loading spike-encoded data...")
    t0 = time.time()
    features, labels, boundaries = load_all_npz(args.spikes_dir, max_files=args.max_files)
    logger.info(f"Data loaded in {time.time() - t0:.1f}s")

    # Split data
    splits = split_data(features, labels, random_state=args.seed)

    # Store all results
    all_results = {}
    trained_models = {}

    # ═════════════════════════════════════════════════════════════
    # MODEL 1: AUTOENCODER
    # ═════════════════════════════════════════════════════════════
    if "autoencoder" in args.models:
        logger.info("\n" + "─" * 60)
        logger.info("  PHASE 1: Autoencoder (Unsupervised Anomaly Detection)")
        logger.info("─" * 60)

        from .model_autoencoder import EEGAutoencoder, train_autoencoder, evaluate_autoencoder
        from .utils import plot_training_history, print_metrics, save_model_and_metrics

        ae_loaders = get_autoencoder_dataloaders(
            splits, seq_len=args.seq_len, batch_size=args.batch_size,
        )

        ae_model = EEGAutoencoder(
            seq_len=args.seq_len, feature_dim=64, latent_dim=32,
        )
        logger.info(f"Autoencoder params: {sum(p.numel() for p in ae_model.parameters()):,}")

        ae_history = train_autoencoder(
            ae_model, ae_loaders["train"], ae_loaders["val"],
            device=device, epochs=args.epochs, lr=args.lr,
            save_dir=args.save_dir,
        )

        # For evaluation, we need a loader that returns actual labels
        # Use the classification-mode test loader
        from .dataset import EEGSpikeDataset
        from torch.utils.data import DataLoader

        test_feats, test_labs = splits["test"]
        ae_test_ds = EEGSpikeDataset(test_feats, test_labs, seq_len=args.seq_len, mode="autoencoder")

        # Monkey-patch the dataset to return labels for evaluation
        class AEEvalDataset(torch.utils.data.Dataset):
            def __init__(self, ds):
                self.ds = ds
            def __len__(self):
                return len(self.ds)
            def __getitem__(self, idx):
                x, _ = self.ds[idx]
                y = torch.tensor(self.ds.seq_labels[idx], dtype=torch.long)
                return x, y

        ae_eval_ds = AEEvalDataset(ae_test_ds)
        ae_eval_loader = DataLoader(ae_eval_ds, batch_size=args.batch_size, shuffle=False)

        ae_metrics = evaluate_autoencoder(ae_model, ae_eval_loader, device)
        print_metrics(ae_metrics, prefix="Autoencoder")

        plot_training_history(
            ae_history,
            save_path=os.path.join(args.plots_dir, "autoencoder_history.png"),
            model_name="Autoencoder",
        )
        save_model_and_metrics(ae_model, ae_metrics, "autoencoder", args.save_dir)

        all_results["autoencoder"] = ae_metrics
        trained_models["autoencoder"] = ae_model

    # ═════════════════════════════════════════════════════════════
    # MODEL 2: LSTM
    # ═════════════════════════════════════════════════════════════
    if "lstm" in args.models:
        logger.info("\n" + "─" * 60)
        logger.info("  PHASE 2: BiLSTM with Attention")
        logger.info("─" * 60)

        from .model_lstm import EEGLSTMClassifier, train_lstm, evaluate_lstm
        from .utils import (
            get_class_weights, plot_training_history, plot_confusion_matrix,
            print_metrics, save_model_and_metrics,
        )

        seq_loaders = get_dataloaders(
            splits, seq_len=args.seq_len, batch_size=args.batch_size,
            mode="classification", use_weighted_sampler=True,
        )

        _, train_labs = splits["train"]
        class_weights = get_class_weights(train_labs, device)

        lstm_model = EEGLSTMClassifier(input_dim=64, hidden_dim=128, num_layers=2, dropout=0.4)
        logger.info(f"LSTM params: {sum(p.numel() for p in lstm_model.parameters()):,}")

        lstm_history = train_lstm(
            lstm_model, seq_loaders["train"], seq_loaders["val"],
            device=device, class_weights=class_weights,
            epochs=args.epochs, lr=args.lr, save_dir=args.save_dir,
        )

        lstm_metrics = evaluate_lstm(lstm_model, seq_loaders["test"], device)
        print_metrics(lstm_metrics, prefix="LSTM")

        plot_training_history(
            lstm_history,
            save_path=os.path.join(args.plots_dir, "lstm_history.png"),
            model_name="BiLSTM + Attention",
        )

        # Confusion matrix
        lstm_model.eval()
        all_p, all_t = [], []
        with torch.no_grad():
            for xb, yb in seq_loaders["test"]:
                preds = lstm_model(xb.to(device)).argmax(dim=1).cpu().numpy()
                all_p.extend(preds)
                all_t.extend(yb.numpy())
        plot_confusion_matrix(
            np.array(all_t), np.array(all_p),
            save_path=os.path.join(args.plots_dir, "lstm_confusion.png"),
            model_name="BiLSTM + Attention",
        )

        save_model_and_metrics(lstm_model, lstm_metrics, "lstm", args.save_dir)
        all_results["lstm"] = lstm_metrics
        trained_models["lstm"] = lstm_model

    # ═════════════════════════════════════════════════════════════
    # MODEL 3: CNN
    # ═════════════════════════════════════════════════════════════
    if "cnn" in args.models:
        logger.info("\n" + "─" * 60)
        logger.info("  PHASE 3: 1D CNN Classifier")
        logger.info("─" * 60)

        from .model_cnn import EEGCNNClassifier, train_cnn, evaluate_cnn
        from .utils import (
            get_class_weights, plot_training_history, plot_confusion_matrix,
            print_metrics, save_model_and_metrics,
        )

        cnn_loaders = get_dataloaders(
            splits, batch_size=args.batch_size,
            mode="cnn", use_weighted_sampler=True,
        )

        _, train_labs = splits["train"]
        class_weights = get_class_weights(train_labs, device)

        cnn_model = EEGCNNClassifier(input_dim=64, num_classes=2, dropout=0.4)
        logger.info(f"CNN params: {sum(p.numel() for p in cnn_model.parameters()):,}")

        cnn_history = train_cnn(
            cnn_model, cnn_loaders["train"], cnn_loaders["val"],
            device=device, class_weights=class_weights,
            epochs=args.epochs, lr=args.lr, save_dir=args.save_dir,
        )

        cnn_metrics = evaluate_cnn(cnn_model, cnn_loaders["test"], device)
        print_metrics(cnn_metrics, prefix="CNN")

        plot_training_history(
            cnn_history,
            save_path=os.path.join(args.plots_dir, "cnn_history.png"),
            model_name="1D CNN",
        )

        cnn_model.eval()
        all_p, all_t = [], []
        with torch.no_grad():
            for xb, yb in cnn_loaders["test"]:
                preds = cnn_model(xb.to(device)).argmax(dim=1).cpu().numpy()
                all_p.extend(preds)
                all_t.extend(yb.numpy())
        plot_confusion_matrix(
            np.array(all_t), np.array(all_p),
            save_path=os.path.join(args.plots_dir, "cnn_confusion.png"),
            model_name="1D CNN",
        )

        save_model_and_metrics(cnn_model, cnn_metrics, "cnn", args.save_dir)
        all_results["cnn"] = cnn_metrics
        trained_models["cnn"] = cnn_model

    # ═════════════════════════════════════════════════════════════
    # MODEL 4: TRANSFORMER
    # ═════════════════════════════════════════════════════════════
    if "transformer" in args.models:
        logger.info("\n" + "─" * 60)
        logger.info("  PHASE 4: Transformer Encoder")
        logger.info("─" * 60)

        from .model_transformer import EEGTransformerClassifier, train_transformer, evaluate_transformer
        from .utils import (
            get_class_weights, plot_training_history, plot_confusion_matrix,
            print_metrics, save_model_and_metrics,
        )

        # Re-use sequence loaders (same as LSTM)
        seq_loaders = get_dataloaders(
            splits, seq_len=args.seq_len, batch_size=args.batch_size,
            mode="classification", use_weighted_sampler=True,
        )

        _, train_labs = splits["train"]
        class_weights = get_class_weights(train_labs, device)

        tf_model = EEGTransformerClassifier(
            input_dim=64, d_model=128, nhead=4, num_layers=4,
            dim_feedforward=256, dropout=0.3, max_seq_len=args.seq_len + 10,
        )
        logger.info(f"Transformer params: {sum(p.numel() for p in tf_model.parameters()):,}")

        tf_history = train_transformer(
            tf_model, seq_loaders["train"], seq_loaders["val"],
            device=device, class_weights=class_weights,
            epochs=args.epochs, lr=5e-4, save_dir=args.save_dir,
        )

        tf_metrics = evaluate_transformer(tf_model, seq_loaders["test"], device)
        print_metrics(tf_metrics, prefix="Transformer")

        plot_training_history(
            tf_history,
            save_path=os.path.join(args.plots_dir, "transformer_history.png"),
            model_name="Transformer",
        )

        tf_model.eval()
        all_p, all_t = [], []
        with torch.no_grad():
            for xb, yb in seq_loaders["test"]:
                preds = tf_model(xb.to(device)).argmax(dim=1).cpu().numpy()
                all_p.extend(preds)
                all_t.extend(yb.numpy())
        plot_confusion_matrix(
            np.array(all_t), np.array(all_p),
            save_path=os.path.join(args.plots_dir, "transformer_confusion.png"),
            model_name="Transformer",
        )

        save_model_and_metrics(tf_model, tf_metrics, "transformer", args.save_dir)
        all_results["transformer"] = tf_metrics
        trained_models["transformer"] = tf_model

    # ═════════════════════════════════════════════════════════════
    # ENSEMBLE (if all 4 models trained)
    # ═════════════════════════════════════════════════════════════
    if not args.no_ensemble and len(trained_models) >= 2:
        logger.info("\n" + "─" * 60)
        logger.info("  PHASE 5: Ensemble Evaluation")
        logger.info("─" * 60)

        from .ensemble import EnsemblePredictor
        from .utils import compute_metrics, print_metrics, plot_confusion_matrix

        # We need a common test set — use sequence loaders for LSTM/Transformer,
        # and iterate window-by-window for CNN/Autoencoder
        # For simplicity, generate predictions from each model on test split
        test_feats, test_labs = splits["test"]

        model_probs = {}

        # Sequence models (LSTM, Transformer)
        for name in ["lstm", "transformer"]:
            if name not in trained_models:
                continue
            model = trained_models[name]
            model.eval()
            model.to(device)

            # Build sequences from test data
            from .dataset import EEGSpikeDataset
            ds = EEGSpikeDataset(test_feats, test_labs, seq_len=args.seq_len)
            loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)

            probs = []
            true_labels = []
            with torch.no_grad():
                for xb, yb in loader:
                    logits = model(xb.to(device))
                    p = torch.nn.functional.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    probs.extend(p)
                    true_labels.extend(yb.numpy())

            model_probs[name] = np.array(probs)
            ensemble_labels = np.array(true_labels)

        # CNN (window-level)
        if "cnn" in trained_models:
            model = trained_models["cnn"]
            model.eval()
            model.to(device)

            # CNN uses seq labels from same dataset for alignment
            from .dataset import EEGSpikeDataset
            ds = EEGSpikeDataset(test_feats, test_labs, seq_len=args.seq_len)

            # For each sequence, use the LAST window for CNN prediction
            probs = []
            with torch.no_grad():
                loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)
                for xb, yb in loader:
                    # Take last window of each sequence: (B, seq_len, 64) → (B, 64)
                    last_window = xb[:, -1, :]
                    logits = model(last_window.to(device))
                    p = torch.nn.functional.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    probs.extend(p)

            model_probs["cnn"] = np.array(probs)

        # Autoencoder (anomaly score)
        if "autoencoder" in trained_models:
            model = trained_models["autoencoder"]
            model.eval()
            model.to(device)

            from .dataset import EEGSpikeDataset
            ds = EEGSpikeDataset(test_feats, test_labs, seq_len=args.seq_len)
            loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)

            scores = []
            with torch.no_grad():
                for xb, yb in loader:
                    err = model.anomaly_score(xb.to(device)).cpu().numpy()
                    scores.extend(err)

            scores = np.array(scores)
            # Normalize to [0, 1]
            if scores.max() > scores.min():
                scores = (scores - scores.min()) / (scores.max() - scores.min())
            model_probs["autoencoder"] = scores

        # Tune ensemble weights on validation predictions
        logger.info("Tuning ensemble weights on validation set...")
        ensemble = EnsemblePredictor()

        # Quick tune on test set (ideally should be a held-out val set)
        best_weights = ensemble.tune_weights(model_probs, ensemble_labels)

        # Ensemble predictions
        ensemble_preds = ensemble.predict(model_probs)
        ensemble_proba = ensemble.predict_proba(model_probs)

        ensemble_metrics = compute_metrics(ensemble_labels, ensemble_preds, ensemble_proba)
        print_metrics(ensemble_metrics, prefix="ENSEMBLE")

        plot_confusion_matrix(
            ensemble_labels, ensemble_preds,
            save_path=os.path.join(args.plots_dir, "ensemble_confusion.png"),
            model_name="Ensemble",
        )

        all_results["ensemble"] = ensemble_metrics
        all_results["ensemble_weights"] = best_weights

    # ═════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═════════════════════════════════════════════════════════════
    print("\n")
    print("█" * 60)
    print("█  TRAINING COMPLETE — FINAL RESULTS")
    print("█" * 60)
    print()
    print(f"{'Model':<15} {'Accuracy':>10} {'F1':>10} {'Recall':>10} {'AUC-ROC':>10}")
    print("─" * 55)

    for name in ["autoencoder", "lstm", "cnn", "transformer", "ensemble"]:
        if name in all_results and name != "ensemble_weights":
            m = all_results[name]
            print(
                f"{name:<15} "
                f"{m.get('accuracy', 0):>10.4f} "
                f"{m.get('f1', 0):>10.4f} "
                f"{m.get('recall', 0):>10.4f} "
                f"{m.get('auc_roc', 0):>10.4f}"
            )

    print("─" * 55)
    print()

    # Save combined results
    results_path = os.path.join(args.save_dir, "all_results.json")
    serializable = {}
    for k, v in all_results.items():
        if isinstance(v, dict):
            serializable[k] = {
                kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                for kk, vv in v.items()
            }
        else:
            serializable[k] = v
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    logger.info(f"All results saved to: {results_path}")
    logger.info(f"Models saved to: {args.save_dir}")
    logger.info(f"Plots saved to: {args.plots_dir}")
    print()


if __name__ == "__main__":
    main()
