# ================================================================
# main.py — End-to-End Multi-Task EEG Analysis Pipeline
#
# EEG Data (CHB01)
#   ↓ Preprocessing
#   ↓ SNN Encoder (spike features)
#   ├──→ LSTM/Transformer  → Seizure PREDICTION
#   ├──→ CNN               → Disease Classification
#   ├──→ Random Forest     → Interpretable Diagnosis
#   └──→ Autoencoder       → Anomaly Detection
# ================================================================

import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from config        import cfg
from data_loader   import load_dataset
from preprocessor  import preprocess_all
from snn_encoder   import FilterbankExtractor, SNNEncoder
from trainer       import (split_and_scale,
                            make_weighted_loader,
                            make_loader,
                            train_torch_model)
from evaluator     import (evaluate_torch_classifier,
                            evaluate_anomaly_detector)
from visualizer    import (plot_pipeline_overview,
                            plot_all_training_curves,
                            plot_all_confusion_matrices,
                            plot_multi_roc,
                            plot_multitask_overlay,
                            plot_summary_dashboard)

from models.lstm_transformer import (SeizurePredictionModel,
                                      build_sequences)
from models.cnn_classifier   import (EEGCNNClassifier,
                                      build_cnn_dataset)
from models.random_forest    import (InterpretableRF,
                                      build_rf_features)
from models.autoencoder      import AnomalyDetector

DEVICE = cfg.DEVICE


# ════════════════════════════════════════════════════════════════
# STEP 0: Pipeline Overview
# ════════════════════════════════════════════════════════════════
def step0_overview():
    print("\n" + "="*60)
    print("  STEP 0: Pipeline Overview")
    print("="*60)
    plot_pipeline_overview()


# ════════════════════════════════════════════════════════════════
# STEP 1: Data Loading
# ════════════════════════════════════════════════════════════════
def step1_load(data_dir: str) -> list[dict]:
    print("\n" + "="*60)
    print("  STEP 1: Loading EDF Data")
    print("="*60)
    return load_dataset(data_dir)


# ════════════════════════════════════════════════════════════════
# STEP 2: Preprocessing
# ════════════════════════════════════════════════════════════════
def step2_preprocess(records: list[dict]) -> list[dict]:
    print("\n" + "="*60)
    print("  STEP 2: Preprocessing (Notch · Bandpass · CAR · Z-score)")
    print("="*60)
    return preprocess_all(records)


# ════════════════════════════════════════════════════════════════
# STEP 3: SNN Encoding
# ════════════════════════════════════════════════════════════════
def step3_snn_encode(records: list[dict]
                     ) -> tuple[SNNEncoder,
                                np.ndarray,
                                np.ndarray,
                                np.ndarray,
                                dict]:
    """
    Returns
    -------
    snn_model    : trained SNNEncoder
    X_feat       : (N, FEATURE_DIM)  filterbank features
    y_binary     : (N,)              binary seizure labels
    y_preictal   : (N,)              3-class preictal labels
    spike_by_file: dict fname → spike_rates np.ndarray
    """
    print("\n" + "="*60)
    print("  STEP 3: SNN Encoder — Filterbank → Spike Features")
    print("="*60)

    extractor = FilterbankExtractor()
    snn_model = SNNEncoder().to(DEVICE)

    X_all, y_bin_all, y_pre_all = [], [], []
    spike_by_file = {}

    for rec in records:
        X_f, y_f = extractor.extract_windows(
            rec["signals"], rec["labels"]["binary"])

        # Align preictal labels
        ep  = cfg.EPOCH_SAMPLES
        W   = cfg.WINDOW_SIZE
        n_w = y_f.shape[0]
        pre_lbl = np.array([
            int(rec["labels"]["preictal"]
                [(W-1+i)*ep:(W-1+i)*ep+ep].mean() + 0.5)
            for i in range(n_w)
        ])

        X_all.append(X_f)
        y_bin_all.append(y_f)
        y_pre_all.append(pre_lbl)

    X_feat   = np.concatenate(X_all,     axis=0)
    y_binary = np.concatenate(y_bin_all, axis=0)
    y_pre    = np.concatenate(y_pre_all, axis=0)

    # ── Train SNN encoder with BCE loss ──────────────────
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import DataLoader, TensorDataset

    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X_feat)
    Xt      = torch.tensor(X_sc,    dtype=torch.float32)
    yt      = torch.tensor(y_binary, dtype=torch.long)
    ds      = TensorDataset(Xt, yt)
    loader  = DataLoader(ds, batch_size=cfg.BATCH_SIZE,
                         shuffle=True)

    w       = torch.tensor([1.0, cfg.CLASS_WEIGHT_SEI]).to(DEVICE)
    crit    = nn.CrossEntropyLoss(weight=w)
    opt     = torch.optim.Adam(snn_model.parameters(),
                                lr=cfg.LEARNING_RATE)

    # Add output classification head for pre-training
    clf_head = nn.Linear(cfg.SNN_SPIKE_DIM, 2).to(DEVICE)
    opt_head = torch.optim.Adam(clf_head.parameters(),
                                 lr=cfg.LEARNING_RATE)

    print("[SNN] Pre-training encoder ...")
    for epoch in range(15):
        snn_model.train(); clf_head.train()
        total_loss = 0.0
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); opt_head.zero_grad()
            rate, _ = snn_model(Xb)
            logits  = clf_head(rate)
            loss    = crit(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(
                snn_model.parameters(), 1.0)
            opt.step(); opt_head.step()
            total_loss += loss.item()
        print(f"  [SNN] Epoch {epoch+1:02d}/15 "
              f"Loss={total_loss/len(loader):.4f}")

    # ── Extract spike features for all files ─────────────
    print("\n[SNN] Encoding spike features ...")
    snn_model.eval()
    for rec, X_f in zip(records, X_all):
        X_sc_f = scaler.transform(X_f)
        rates  = snn_model.encode_numpy(
            X_sc_f, DEVICE, scaler=None)
        spike_by_file[rec["fname"]] = rates
        print(f"  ✓ {rec['fname']}: "
              f"spike shape={rates.shape}")

    # Global spike features
    spike_global = np.concatenate(
        list(spike_by_file.values()), axis=0)
    n_min = min(len(spike_global), len(y_binary))
    spike_global = spike_global[:n_min]
    y_binary     = y_binary[:n_min]
    y_pre        = y_pre[:n_min]

    print(f"\n[SNN] Spike features: {spike_global.shape}")
    print(f"[SNN] Seizure ratio: "
          f"{y_binary.mean()*100:.2f}%")

    # Save encoder
    torch.save(snn_model.state_dict(),
               os.path.join(cfg.MODEL_DIR, "snn_encoder.pth"))

    return snn_model, spike_global, y_binary, y_pre, spike_by_file


# ════════════════════════════════════════════════════════════════
# STEP 4: Task A — Seizure Prediction (LSTM + Transformer)
# ════════════════════════════════════════════════════════════════
def step4_seizure_prediction(spike_feats: np.ndarray,
                              y_preictal : np.ndarray
                              ) -> tuple[dict, dict]:
    print("\n" + "="*60)
    print("  STEP 4: Task A — Seizure Prediction "
          "(LSTM + Transformer)")
    print("="*60)

    X_seq, y_seq = build_sequences(spike_feats, y_preictal)
    print(f"[Pred] Sequence dataset: {X_seq.shape}  "
          f"y={y_seq.shape}")

    # Flatten for split, then reshape
    N, T, D = X_seq.shape
    X_flat   = X_seq.reshape(N, T*D)

    Xtr, Xval, Xte, ytr, yval, yte, scaler = \
        split_and_scale(X_flat, y_seq)

    # Reshape back to 3-D
    def r3(X): return X.reshape(-1, T, D)
    Xtr3, Xval3, Xte3 = r3(Xtr), r3(Xval), r3(Xte)

    def make_seq_loader(X3, y, weighted=False):
        Xt = torch.tensor(X3, dtype=torch.float32)
        yt = torch.tensor(y,  dtype=torch.long)
        from torch.utils.data import (TensorDataset,
                                       DataLoader,
                                       WeightedRandomSampler)
        ds = TensorDataset(Xt, yt)
        if weighted:
            counts = np.bincount(y)
            wts    = 1.0/counts
            sw     = torch.tensor([wts[c] for c in y],
                                   dtype=torch.float)
            sampler = WeightedRandomSampler(sw, len(sw))
            return DataLoader(ds,
                              batch_size=cfg.BATCH_SIZE,
                              sampler=sampler)
        return DataLoader(ds,
                          batch_size=cfg.BATCH_SIZE,
                          shuffle=False)

    train_dl = make_seq_loader(Xtr3, ytr, weighted=True)
    val_dl   = make_seq_loader(Xval3, yval)
    test_dl  = make_seq_loader(Xte3, yte)

    model   = SeizurePredictionModel().to(DEVICE)
    w       = torch.tensor(
        [1.0, 5.0, cfg.CLASS_WEIGHT_SEI]).to(DEVICE)
    crit    = nn.CrossEntropyLoss(weight=w)
    opt     = torch.optim.AdamW(model.parameters(),
                                 lr=cfg.LEARNING_RATE,
                                 weight_decay=cfg.WEIGHT_DECAY)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.EPOCHS)

    hist = train_torch_model(
        model, train_dl, val_dl, crit, opt, sched,
        DEVICE, "LSTM_Transformer")

    result = evaluate_torch_classifier(
        model, test_dl, DEVICE,
        ["Interictal","Pre-ictal","Ictal"])

    # Binary y for ROC (seizure vs rest)
    result["y_prob"]  = result["y_prob"]
    result["y_true"]  = (result["y_true"] > 0).astype(int)
    result["y_prob"]  = result["y_prob"][:, -1]
    result["auc"]     = float(
        __import__("sklearn.metrics", fromlist=["roc_auc_score"])
        .roc_auc_score(result["y_true"], result["y_prob"]))

    return hist, result


# ════════════════════════════════════════════════════════════════
# STEP 5: Task B — Disease Classification (CNN)
# ════════════════════════════════════════════════════════════════
def step5_cnn_classification(records: list[dict]
                              ) -> tuple[dict, dict]:
    print("\n" + "="*60)
    print("  STEP 5: Task B — Disease Classification (CNN)")
    print("="*60)

    X_cnn, y_cnn = build_cnn_dataset(records)
    print(f"[CNN] Dataset: {X_cnn.shape}  "
          f"classes={np.unique(y_cnn)}")

    # Split (no scaling — CNN handles raw EEG)
    Xtr, Xte, ytr, yte = train_test_split(
        X_cnn, y_cnn, test_size=0.2,
        stratify=y_cnn, random_state=42)
    Xtr, Xval, ytr, yval = train_test_split(
        Xtr, ytr, test_size=0.1,
        stratify=ytr, random_state=42)

    from torch.utils.data import (TensorDataset,
                                   DataLoader,
                                   WeightedRandomSampler)
    def make_raw_loader(X, y, weighted=False):
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.long)
        ds = TensorDataset(Xt, yt)
        if weighted:
            counts  = np.bincount(y)
            wts     = 1.0 / counts
            sw      = torch.tensor(
                [wts[c] for c in y], dtype=torch.float)
            sampler = WeightedRandomSampler(sw, len(sw))
            return DataLoader(ds,
                              batch_size=cfg.BATCH_SIZE,
                              sampler=sampler)
        return DataLoader(ds,
                          batch_size=cfg.BATCH_SIZE,
                          shuffle=False)

    train_dl = make_raw_loader(Xtr,  ytr,  weighted=True)
    val_dl   = make_raw_loader(Xval, yval)
    test_dl  = make_raw_loader(Xte,  yte)

    model   = EEGCNNClassifier().to(DEVICE)
    w       = torch.tensor(
        [1.0]*cfg.CNN_CLASSES, dtype=torch.float).to(DEVICE)
    crit    = nn.CrossEntropyLoss(weight=w)
    opt     = torch.optim.AdamW(model.parameters(),
                                 lr=cfg.LEARNING_RATE,
                                 weight_decay=cfg.WEIGHT_DECAY)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.EPOCHS)

    hist = train_torch_model(
        model, train_dl, val_dl, crit, opt, sched,
        DEVICE, "CNN_Classifier")

    result = evaluate_torch_classifier(
        model, test_dl, DEVICE,
        ["Normal","Focal","Generalized"])

    result["y_true"] = (result["y_true"] > 0).astype(int)
    result["y_prob"] = result["y_prob"][:, -1]
    result["auc"]    = float(
        __import__("sklearn.metrics", fromlist=["roc_auc_score"])
        .roc_auc_score(result["y_true"], result["y_prob"]))

    return hist, result


# ════════════════════════════════════════════════════════════════
# STEP 6: Task C — Interpretable Diagnosis (Random Forest)
# ════════════════════════════════════════════════════════════════
def step6_random_forest(records      : list[dict],
                         spike_by_file: dict
                         ) -> tuple[dict, dict]:
    print("\n" + "="*60)
    print("  STEP 6: Task C — Interpretable Diagnosis "
          "(Random Forest)")
    print("="*60)

    X_rf, y_rf = build_rf_features(records, spike_by_file)
    print(f"[RF] Feature matrix: {X_rf.shape}")

    Xtr, Xte, ytr, yte = train_test_split(
        X_rf, y_rf, test_size=0.2,
        stratify=y_rf, random_state=42)
    Xtr, Xval, ytr, yval = train_test_split(
        Xtr, ytr, test_size=0.1,
        stratify=ytr, random_state=42)

    rf = InterpretableRF()
    rf.build_feature_names(
        n_hc_feats  = cfg.N_CHANNELS * 10,
        n_spk_feats = cfg.SNN_SPIKE_DIM
    )
    rf.fit(Xtr, ytr)
    res = rf.evaluate(Xte, yte)

    imp_df = rf.compute_feature_importance(Xval, yval)
    rf.plot_importance(top_k=20)
    rf.save()

    print(f"\n[RF] Top-5 features:\n{imp_df.head()}")

    # Binary for ROC
    y_true_bin = (res["y_true"] > 0).astype(int)
    y_prob_bin = res["y_prob"][:, -1]
    from sklearn.metrics import roc_auc_score
    
    auc = 0.0
    try:
        if len(np.unique(y_true_bin)) > 1:
            auc = roc_auc_score(y_true_bin, y_prob_bin)
    except:
        pass  # Unable to compute AUC
    
    # Handle confusion matrix safely
    cm = res["cm"]
    if cm.shape == (2, 2):
        sensitivity = cm[1,1] / (cm[1,:].sum() + 1e-8)
        specificity = cm[0,0] / (cm[0,:].sum() + 1e-8)
    else:
        sensitivity = 0.0
        specificity = 0.0

    result = {
        "cm"         : cm,
        "y_true"     : y_true_bin,
        "y_pred"     : (res["y_pred"] > 0).astype(int),
        "y_prob"     : y_prob_bin,
        "auc"        : auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }
    hist = {"train_loss":[], "val_loss":[],
            "train_acc":[], "val_acc":[]}
    return hist, result


# ════════════════════════════════════════════════════════════════
# STEP 7: Task D — Anomaly Detection (VAE Autoencoder)
# ════════════════════════════════════════════════════════════════
def step7_anomaly_detection(spike_feats: np.ndarray,
                             y_binary   : np.ndarray
                             ) -> tuple[dict, dict]:
    print("\n" + "="*60)
    print("  STEP 7: Task D — Anomaly Detection (VAE)")
    print("="*60)

    # Train only on normal windows
    X_normal = spike_feats[y_binary == 0]
    print(f"[AE] Normal windows: {len(X_normal)}")
    print(f"[AE] Seizure windows (test): {(y_binary==1).sum()}")

    detector = AnomalyDetector(DEVICE)
    ae_hist  = detector.fit(X_normal,
                             n_epochs  = cfg.EPOCHS,
                             batch_size= cfg.BATCH_SIZE)

    # Evaluate on full set
    scores, is_anomaly = detector.predict(spike_feats)
    result = evaluate_anomaly_detector(
        detector, spike_feats, y_binary)

    # Build confusion matrix for anomaly detection
    y_pred_bin = is_anomaly.astype(int)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_binary, y_pred_bin)
    result["cm"] = cm

    # Visualise
    detector.plot_anomaly_scores(
        scores, y_binary, fname="ae_anomaly_scores.png")
    detector.plot_latent_space(
        spike_feats, y_binary, fname="ae_latent_space.png")
    detector.save()

    # Package for dashboard
    result["y_true"] = y_binary
    result["y_prob"] = scores     # continuous score
    hist = {"train_loss" : ae_hist,
            "val_loss"   : [],
            "train_acc"  : [],
            "val_acc"    : []}
    return hist, result


# ════════════════════════════════════════════════════════════════
# STEP 8: Final Visualisation & Summary
# ════════════════════════════════════════════════════════════════
def step8_final_summary(all_histories: dict,
                          all_results  : dict,
                          records      : list[dict],
                          spike_by_file: dict):
    print("\n" + "="*60)
    print("  STEP 8: Final Visualisation & Summary")
    print("="*60)

    plot_all_training_curves(
        {k: v for k, v in all_histories.items()
         if v["train_loss"]})
    plot_all_confusion_matrices(all_results)
    plot_multi_roc(all_results)
    plot_summary_dashboard(all_results)

    # Multi-task overlay on first seizure file
    target_file = "chb01_03.edf"
    for rec in records:
        if rec["fname"] == target_file:
            spk = spike_by_file.get(target_file,
                                     np.array([]))
            if len(spk):
                preds_dict = {
                    "Prediction"  : (spk.mean(1) > 0.5).astype(int),
                    "Classification": np.zeros(len(spk), dtype=int),
                    "RF Diagnosis": np.zeros(len(spk), dtype=int),
                    "AE Anomaly"  : np.zeros(len(spk), dtype=int),
                }
                plot_multitask_overlay(
                    rec["signals"],
                    rec["labels"]["binary"],
                    preds_dict,
                    fname=target_file
                )
            break

    # ── Print final report ─────────────────────────────
    print("\n" + "="*60)
    print("  FINAL PERFORMANCE SUMMARY")
    print("="*60)
    header = f"{'Model':<22} {'AUC':>6} {'Sens':>6} {'Spec':>6}"
    print(header)
    print("-" * len(header))
    for name, res in all_results.items():
        auc  = res.get("auc",  0)
        sens = res.get("sensitivity", 0)
        spec = res.get("specificity", 0)
        print(f"{name:<22} {auc:>6.3f} {sens:>6.3f} {spec:>6.3f}")
    print("="*60)


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════
def main():
    import argparse
    p = argparse.ArgumentParser(
        description="Multi-Task EEG Pipeline (SNN + Hybrid)")
    p.add_argument("--data_dir", default=cfg.DATA_DIR)
    p.add_argument("--skip_tasks", nargs="*", default=[],
                   choices=["pred","cnn","rf","ae"],
                   help="Tasks to skip")
    args = p.parse_args()

    print("\n" + "█"*60)
    print("█  MULTI-TASK EEG ANALYSIS PIPELINE  (SNN + Hybrid)")
    print("█" + "═"*58 + "█")
    print(f"█  Device : {DEVICE}")
    print(f"█  Data   : {args.data_dir}")
    print("█"*60 + "\n")

    # ── Steps 0–3 always run ──────────────────────────
    step0_overview()
    records     = step1_load(args.data_dir)
    records     = step2_preprocess(records)
    (snn_model,
     spike_feats,
     y_binary,
     y_preictal,
     spike_by_file) = step3_snn_encode(records)

    all_histories = {}
    all_results   = {}

    # ── Step 4: Seizure Prediction ────────────────────
    if "pred" not in args.skip_tasks:
        h, r = step4_seizure_prediction(spike_feats, y_preictal)
        all_histories["LSTM/Transformer"] = h
        all_results  ["LSTM/Transformer"] = r

    # ── Step 5: CNN Disease Classification ────────────
    if "cnn" not in args.skip_tasks:
        h, r = step5_cnn_classification(records)
        all_histories["CNN"] = h
        all_results  ["CNN"] = r

    # ── Step 6: Random Forest Diagnosis ───────────────
    if "rf" not in args.skip_tasks:
        h, r = step6_random_forest(records, spike_by_file)
        all_histories["Random Forest"] = h
        all_results  ["Random Forest"] = r

    # ── Step 7: Anomaly Detection ─────────────────────
    if "ae" not in args.skip_tasks:
        h, r = step7_anomaly_detection(spike_feats, y_binary)
        all_histories["Autoencoder"] = h
        all_results  ["Autoencoder"] = r

    # ── Step 8: Summary ───────────────────────────────
    step8_final_summary(all_histories, all_results,
                         records, spike_by_file)


if __name__ == "__main__":
    main()