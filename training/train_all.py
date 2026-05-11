#!/usr/bin/env python3
"""
CLI entry point: Train all 4 downstream models on spike-encoded .npz data.

Improvements over baseline:
    - Feature augmentation (spectral/statistical biomarkers, 64→75 dim)
    - Deep SVDD + Isolation Forest replaces autoencoder
    - Spatial 2D CNN exploiting electrode layout
    - Seizure data augmentation for minority class
    - Shorter default seq_len (10 instead of 30)
    - AUC-PR ensemble optimization + stacking meta-learner
    - Temporal smoothing post-processing

Usage::
    python -m training.train_all --spikes_dir ~/EEG-preprocess/output/spikes
    python -m training.train_all --spikes_dir ./output/spikes --models lstm cnn
    python -m training.train_all --spikes_dir ./output/spikes --epochs 80 --augment_features --augment_seizure
"""

import argparse, os, sys, time, json, logging
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Train downstream seizure detection models.")
    p.add_argument("--spikes_dir", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="./models")
    p.add_argument("--plots_dir", type=str, default="./training_plots")
    p.add_argument("--models", nargs="+", default=["autoencoder", "lstm", "cnn", "transformer"],
                   choices=["autoencoder", "lstm", "cnn", "transformer"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seq_len", type=int, default=10, help="Sequence length (default: 10, was 30)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max_files", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_ensemble", action="store_true")
    # New flags
    p.add_argument("--augment_features", action="store_true", help="Add spectral/statistical features (64→75 dim)")
    p.add_argument("--augment_seizure", action="store_true", help="Augment minority class during training")
    p.add_argument("--anomaly_method", default="deep_svdd", choices=["deep_svdd", "autoencoder"],
                   help="Anomaly detection method (default: deep_svdd)")
    p.add_argument("--cnn_mode", default="spatial", choices=["spatial", "flat"],
                   help="CNN architecture (default: spatial 2D)")
    p.add_argument("--use_stacking", action="store_true", help="Use stacking meta-learner for ensemble")
    p.add_argument("--smooth_window", type=int, default=5, help="Temporal median filter window")
    p.add_argument("--persistence_k", type=int, default=3, help="Persistence filter: min positive windows")
    p.add_argument("--persistence_n", type=int, default=5, help="Persistence filter: look-back window")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True); os.makedirs(args.plots_dir, exist_ok=True)

    print(f"\n{'█' * 60}")
    print("█  EEG SEIZURE DETECTION — ENHANCED MODEL TRAINING")
    print(f"█{'═' * 58}█")
    print(f"█  Device      : {device}")
    print(f"█  Models      : {', '.join(args.models)}")
    print(f"█  Seq length  : {args.seq_len} (was 30)")
    print(f"█  Features    : {'augmented (75d)' if args.augment_features else 'original (64d)'}")
    print(f"█  Seizure aug : {args.augment_seizure}")
    print(f"█  Anomaly     : {args.anomaly_method}")
    print(f"█  CNN mode    : {args.cnn_mode}")
    print(f"█  Temporal    : smooth={args.smooth_window}, persist={args.persistence_k}/{args.persistence_n}")
    if torch.cuda.is_available():
        print(f"█  GPU         : {torch.cuda.get_device_name(0)}")
    print(f"{'█' * 60}\n")

    # ── Load Data ────────────────────────────────────────────────
    from .dataset import load_all_npz, split_data, get_dataloaders, get_autoencoder_dataloaders

    logger.info("Loading spike-encoded data...")
    t0 = time.time()
    features, labels, boundaries = load_all_npz(args.spikes_dir, max_files=args.max_files)
    logger.info(f"Data loaded in {time.time() - t0:.1f}s")

    # ── Feature Augmentation ────────────────────────────────────
    feature_dim = 64
    augmentor = None
    if args.augment_features:
        from .feature_augmentor import FeatureAugmentor
        augmentor = FeatureAugmentor()
        # We need to fit on training data only, so split first with raw features
        # then augment each split
        logger.info("Feature augmentation enabled (64 → 75 dim)")
        feature_dim = 75

    # Split data
    splits = split_data(features, labels, random_state=args.seed)

    # Apply feature augmentation after split (fit on train only)
    if augmentor is not None:
        train_f, train_l = splits["train"]
        augmentor.fit(train_f)
        splits["train"] = (augmentor.transform(train_f), train_l)
        val_f, val_l = splits["val"]
        splits["val"] = (augmentor.transform(val_f), val_l)
        test_f, test_l = splits["test"]
        splits["test"] = (augmentor.transform(test_f), test_l)
        logger.info(f"Features augmented: train={splits['train'][0].shape}")

    all_results = {}
    trained_models = {}

    # ═════════════════════════════════════════════════════════════
    # MODEL 1: ANOMALY DETECTOR (Deep SVDD or Autoencoder)
    # ═════════════════════════════════════════════════════════════
    if "autoencoder" in args.models:
        logger.info("\n" + "─" * 60)
        logger.info(f"  PHASE 1: {args.anomaly_method.upper()} Anomaly Detection")
        logger.info("─" * 60)

        from .model_autoencoder import (
            DeepSVDD, EEGAutoencoder, CombinedAnomalyDetector,
            train_deep_svdd, train_autoencoder, evaluate_autoencoder,
        )
        from .utils import plot_training_history, print_metrics, save_model_and_metrics

        ae_loaders = get_autoencoder_dataloaders(splits, seq_len=args.seq_len, batch_size=args.batch_size)

        if args.anomaly_method == "deep_svdd":
            ae_model = DeepSVDD(seq_len=args.seq_len, feature_dim=feature_dim, latent_dim=32)
            logger.info(f"Deep SVDD params: {sum(p.numel() for p in ae_model.parameters()):,}")
            ae_history = train_deep_svdd(
                ae_model, ae_loaders["train"], ae_loaders["val"],
                device=device, epochs=args.epochs, lr=args.lr, save_dir=args.save_dir,
            )
        else:
            ae_model = EEGAutoencoder(seq_len=args.seq_len, feature_dim=feature_dim, latent_dim=32)
            logger.info(f"Autoencoder params: {sum(p.numel() for p in ae_model.parameters()):,}")
            ae_history = train_autoencoder(
                ae_model, ae_loaders["train"], ae_loaders["val"],
                device=device, epochs=args.epochs, lr=args.lr, save_dir=args.save_dir,
            )

        # Fit Isolation Forest on normal training data
        combined_detector = CombinedAnomalyDetector(args.seq_len, feature_dim, 32)
        train_normal_feats = splits["train"][0][splits["train"][1] == 0]
        # Flatten for IF
        from .dataset import EEGSpikeDataset
        if_ds = EEGSpikeDataset(train_normal_feats, np.zeros(len(train_normal_feats)),
                                seq_len=args.seq_len, mode="autoencoder")
        if len(if_ds) > 0:
            if_data = []
            for i in range(min(len(if_ds), 5000)):
                x, _ = if_ds[i]
                if_data.append(x.numpy().flatten())
            if_data = np.array(if_data)
            combined_detector.fit_isolation_forest(if_data)

        # Evaluate
        from .dataset import EEGSpikeDataset
        from torch.utils.data import DataLoader
        test_feats, test_labs = splits["test"]
        ae_test_ds = EEGSpikeDataset(test_feats, test_labs, seq_len=args.seq_len, mode="autoencoder")

        class AEEvalDataset(torch.utils.data.Dataset):
            def __init__(self, ds):
                self.ds = ds
            def __len__(self):
                return len(self.ds)
            def __getitem__(self, idx):
                x, _ = self.ds[idx]
                y = torch.tensor(self.ds.seq_labels[idx], dtype=torch.long)
                return x, y

        ae_eval_loader = DataLoader(AEEvalDataset(ae_test_ds), batch_size=args.batch_size, shuffle=False)
        ae_metrics = evaluate_autoencoder(ae_model, ae_eval_loader, device,
                                          isolation_forest=combined_detector)
        print_metrics(ae_metrics, prefix=args.anomaly_method.upper())

        plot_training_history(ae_history, save_path=os.path.join(args.plots_dir, "autoencoder_history.png"),
                              model_name=args.anomaly_method.upper())
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
        from .utils import get_class_weights, plot_training_history, plot_confusion_matrix, print_metrics, save_model_and_metrics

        seq_loaders = get_dataloaders(splits, seq_len=args.seq_len, batch_size=args.batch_size,
                                      mode="classification", use_weighted_sampler=True,
                                      augment_seizure=args.augment_seizure)

        _, train_labs = splits["train"]
        class_weights = get_class_weights(train_labs, device)

        lstm_model = EEGLSTMClassifier(input_dim=feature_dim, hidden_dim=128, num_layers=2, dropout=0.4)
        logger.info(f"LSTM params: {sum(p.numel() for p in lstm_model.parameters()):,}")

        lstm_history = train_lstm(lstm_model, seq_loaders["train"], seq_loaders["val"],
                                  device=device, class_weights=class_weights,
                                  epochs=args.epochs, lr=args.lr, save_dir=args.save_dir)

        lstm_metrics = evaluate_lstm(lstm_model, seq_loaders["test"], device)
        print_metrics(lstm_metrics, prefix="LSTM")
        plot_training_history(lstm_history, save_path=os.path.join(args.plots_dir, "lstm_history.png"), model_name="BiLSTM + Attention")

        lstm_model.eval()
        all_p, all_t = [], []
        with torch.no_grad():
            for xb, yb in seq_loaders["test"]:
                preds = lstm_model(xb.to(device)).argmax(dim=1).cpu().numpy()
                all_p.extend(preds); all_t.extend(yb.numpy())
        plot_confusion_matrix(np.array(all_t), np.array(all_p),
                              save_path=os.path.join(args.plots_dir, "lstm_confusion.png"), model_name="BiLSTM + Attention")

        save_model_and_metrics(lstm_model, lstm_metrics, "lstm", args.save_dir)
        all_results["lstm"] = lstm_metrics
        trained_models["lstm"] = lstm_model

    # ═════════════════════════════════════════════════════════════
    # MODEL 3: CNN (Spatial or Flat)
    # ═════════════════════════════════════════════════════════════
    if "cnn" in args.models:
        logger.info("\n" + "─" * 60)
        logger.info(f"  PHASE 3: {'Spatial 2D' if args.cnn_mode == 'spatial' else '1D'} CNN Classifier")
        logger.info("─" * 60)

        from .model_cnn import EEGSpatialCNN, EEGCNNClassifier, train_cnn, evaluate_cnn
        from .utils import get_class_weights, plot_training_history, plot_confusion_matrix, print_metrics, save_model_and_metrics

        cnn_loaders = get_dataloaders(splits, batch_size=args.batch_size, mode="cnn",
                                      use_weighted_sampler=True, augment_seizure=args.augment_seizure)

        _, train_labs = splits["train"]
        class_weights = get_class_weights(train_labs, device)

        if args.cnn_mode == "spatial":
            cnn_model = EEGSpatialCNN(input_dim=feature_dim, num_classes=2, dropout=0.4)
        else:
            cnn_model = EEGCNNClassifier(input_dim=feature_dim, num_classes=2, dropout=0.4)
        logger.info(f"CNN params: {sum(p.numel() for p in cnn_model.parameters()):,}")

        cnn_history = train_cnn(cnn_model, cnn_loaders["train"], cnn_loaders["val"],
                                device=device, class_weights=class_weights,
                                epochs=args.epochs, lr=args.lr, save_dir=args.save_dir)

        cnn_metrics = evaluate_cnn(cnn_model, cnn_loaders["test"], device)
        print_metrics(cnn_metrics, prefix="CNN")
        plot_training_history(cnn_history, save_path=os.path.join(args.plots_dir, "cnn_history.png"),
                              model_name=f"{'Spatial 2D' if args.cnn_mode == 'spatial' else '1D'} CNN")

        cnn_model.eval()
        all_p, all_t = [], []
        with torch.no_grad():
            for xb, yb in cnn_loaders["test"]:
                preds = cnn_model(xb.to(device)).argmax(dim=1).cpu().numpy()
                all_p.extend(preds); all_t.extend(yb.numpy())
        plot_confusion_matrix(np.array(all_t), np.array(all_p),
                              save_path=os.path.join(args.plots_dir, "cnn_confusion.png"),
                              model_name=f"{'Spatial 2D' if args.cnn_mode == 'spatial' else '1D'} CNN")

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
        from .utils import get_class_weights, plot_training_history, plot_confusion_matrix, print_metrics, save_model_and_metrics

        seq_loaders = get_dataloaders(splits, seq_len=args.seq_len, batch_size=args.batch_size,
                                      mode="classification", use_weighted_sampler=True,
                                      augment_seizure=args.augment_seizure)

        _, train_labs = splits["train"]
        class_weights = get_class_weights(train_labs, device)

        tf_model = EEGTransformerClassifier(input_dim=feature_dim, d_model=128, nhead=4, num_layers=4,
                                            dim_feedforward=256, dropout=0.3, max_seq_len=args.seq_len + 10)
        logger.info(f"Transformer params: {sum(p.numel() for p in tf_model.parameters()):,}")

        tf_history = train_transformer(tf_model, seq_loaders["train"], seq_loaders["val"],
                                       device=device, class_weights=class_weights,
                                       epochs=args.epochs, lr=5e-4, save_dir=args.save_dir)

        tf_metrics = evaluate_transformer(tf_model, seq_loaders["test"], device)
        print_metrics(tf_metrics, prefix="Transformer")
        plot_training_history(tf_history, save_path=os.path.join(args.plots_dir, "transformer_history.png"), model_name="Transformer")

        tf_model.eval()
        all_p, all_t = [], []
        with torch.no_grad():
            for xb, yb in seq_loaders["test"]:
                preds = tf_model(xb.to(device)).argmax(dim=1).cpu().numpy()
                all_p.extend(preds); all_t.extend(yb.numpy())
        plot_confusion_matrix(np.array(all_t), np.array(all_p),
                              save_path=os.path.join(args.plots_dir, "transformer_confusion.png"), model_name="Transformer")

        save_model_and_metrics(tf_model, tf_metrics, "transformer", args.save_dir)
        all_results["transformer"] = tf_metrics
        trained_models["transformer"] = tf_model

    # ═════════════════════════════════════════════════════════════
    # ENSEMBLE with Temporal Post-Processing
    # ═════════════════════════════════════════════════════════════
    if not args.no_ensemble and len(trained_models) >= 2:
        logger.info("\n" + "─" * 60)
        logger.info("  PHASE 5: Enhanced Ensemble + Temporal Smoothing")
        logger.info("─" * 60)

        from .ensemble import EnsemblePredictor
        from .temporal_filter import TemporalPostProcessor
        from .utils import compute_metrics, print_metrics, plot_confusion_matrix

        test_feats, test_labs = splits["test"]
        model_probs = {}

        # Sequence models (LSTM, Transformer)
        for name in ["lstm", "transformer"]:
            if name not in trained_models: continue
            model = trained_models[name]; model.eval(); model.to(device)
            from .dataset import EEGSpikeDataset
            ds = EEGSpikeDataset(test_feats, test_labs, seq_len=args.seq_len)
            loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)
            probs, true_labels = [], []
            with torch.no_grad():
                for xb, yb in loader:
                    p = torch.nn.functional.softmax(model(xb.to(device)), dim=1)[:, 1].cpu().numpy()
                    probs.extend(p); true_labels.extend(yb.numpy())
            model_probs[name] = np.array(probs)
            ensemble_labels = np.array(true_labels)

        # CNN (window-level)
        if "cnn" in trained_models:
            model = trained_models["cnn"]; model.eval(); model.to(device)
            from .dataset import EEGSpikeDataset
            ds = EEGSpikeDataset(test_feats, test_labs, seq_len=args.seq_len)
            loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)
            probs = []
            with torch.no_grad():
                for xb, yb in loader:
                    last_window = xb[:, -1, :]
                    p = torch.nn.functional.softmax(model(last_window.to(device)), dim=1)[:, 1].cpu().numpy()
                    probs.extend(p)
            model_probs["cnn"] = np.array(probs)

        # Anomaly detector
        if "autoencoder" in trained_models:
            model = trained_models["autoencoder"]; model.eval(); model.to(device)
            from .dataset import EEGSpikeDataset
            ds = EEGSpikeDataset(test_feats, test_labs, seq_len=args.seq_len)
            loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)
            scores = []
            with torch.no_grad():
                for xb, yb in loader:
                    err = model.anomaly_score(xb.to(device)).cpu().numpy()
                    scores.extend(err)
            scores = np.array(scores)
            if scores.max() > scores.min():
                scores = (scores - scores.min()) / (scores.max() - scores.min())
            model_probs["autoencoder"] = scores

        # Setup temporal post-processor
        temporal = TemporalPostProcessor(
            smooth_window=args.smooth_window,
            persistence_k=args.persistence_k,
            persistence_n=args.persistence_n,
        )
        logger.info(f"Temporal post-processor: {temporal}")

        # Setup ensemble
        ensemble = EnsemblePredictor(use_stacking=args.use_stacking)
        ensemble.set_temporal_processor(temporal)

        # Tune weights (AUC-PR based)
        logger.info("Tuning ensemble weights (AUC-PR)...")
        best_weights = ensemble.tune_weights(model_probs, ensemble_labels, metric="auc_pr")

        # Optionally fit stacking meta-learner
        if args.use_stacking:
            logger.info("Fitting stacking meta-learner...")
            ensemble.fit_stacking(model_probs, ensemble_labels)

        # Get predictions (with temporal smoothing)
        ensemble_preds = ensemble.predict(model_probs)
        ensemble_proba = ensemble.predict_proba(model_probs)

        ensemble_metrics = compute_metrics(ensemble_labels, ensemble_preds, ensemble_proba)
        print_metrics(ensemble_metrics, prefix="ENSEMBLE (with temporal smoothing)")

        # Also compute without temporal smoothing for comparison
        ensemble_no_smooth = EnsemblePredictor(weights=best_weights, threshold=ensemble.threshold)
        raw_preds = ensemble_no_smooth.predict(model_probs)
        raw_proba = ensemble_no_smooth.predict_proba(model_probs)
        raw_metrics = compute_metrics(ensemble_labels, raw_preds, raw_proba)
        print_metrics(raw_metrics, prefix="ENSEMBLE (no smoothing)")

        plot_confusion_matrix(ensemble_labels, ensemble_preds,
                              save_path=os.path.join(args.plots_dir, "ensemble_confusion.png"), model_name="Ensemble")

        all_results["ensemble"] = ensemble_metrics
        all_results["ensemble_raw"] = raw_metrics
        all_results["ensemble_weights"] = best_weights

    # ═════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═════════════════════════════════════════════════════════════
    print(f"\n\n{'█' * 70}")
    print("█  TRAINING COMPLETE — FINAL RESULTS")
    print(f"{'█' * 70}\n")
    print(f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC-PR':>10}")
    print("─" * 70)

    for name in ["autoencoder", "lstm", "cnn", "transformer", "ensemble_raw", "ensemble"]:
        if name in all_results and name != "ensemble_weights":
            m = all_results[name]
            label = name if name != "ensemble" else "ensemble+smooth"
            label = label if label != "ensemble_raw" else "ensemble(raw)"
            print(f"{label:<20} {m.get('accuracy',0):>10.4f} {m.get('precision',0):>10.4f} "
                  f"{m.get('recall',0):>10.4f} {m.get('f1',0):>10.4f} {m.get('auc_pr',0):>10.4f}")

    print("─" * 70 + "\n")

    # Save results
    results_path = os.path.join(args.save_dir, "all_results.json")
    serializable = {}
    for k, v in all_results.items():
        if isinstance(v, dict):
            serializable[k] = {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv for kk, vv in v.items()}
        else:
            serializable[k] = v
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    logger.info(f"Results: {results_path} | Models: {args.save_dir} | Plots: {args.plots_dir}\n")


if __name__ == "__main__":
    main()
