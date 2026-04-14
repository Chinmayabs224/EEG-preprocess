"""Evaluation routines for the EEG pipeline."""
# ================================================================
# Unified Evaluator for all tasks
# ================================================================

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)
from config import cfg


def evaluate_torch_classifier(model     : nn.Module,
                                test_dl,
                                device   : torch.device,
                                class_names: list,
                                forward_fn = None) -> dict:
    """Evaluate a PyTorch classifier on a DataLoader."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for Xb, yb in test_dl:
            Xb = Xb.to(device)
            if forward_fn is not None:
                logits = forward_fn(model, Xb)
            else:
                logits = model(Xb)
            probs  = torch.softmax(logits, dim=1)
            preds  = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    cm          = confusion_matrix(y_true, y_pred)
    report      = classification_report(
        y_true, y_pred,
        target_names=class_names[:cm.shape[0]])

    n_cls = y_prob.shape[1]
    auc = 0.0
    ap = 0.0
    
    # Only compute AUC if there are multiple classes in y_true
    if len(np.unique(y_true)) > 1:
        if n_cls == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
            ap  = average_precision_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro")
            ap  = 0.0   # multi-class AP not computed here
    else:
        print(f"⚠ Warning: Only one class in y_true, skipping AUC computation")

    print(f"\n{'='*55}")
    print(report)
    print(f"Confusion Matrix:\n{cm}")
    print(f"ROC-AUC : {auc:.4f}")
    print(f"Avg Prec: {ap:.4f}")
    print(f"{'='*55}")

    return {
        "y_true": y_true, "y_pred": y_pred, "y_prob": y_prob,
        "cm": cm, "auc": auc, "ap": ap, "report": report
    }


def evaluate_anomaly_detector(detector,
                               X_test    : np.ndarray,
                               y_test    : np.ndarray) -> dict:
    """Evaluate VAE anomaly detector with threshold sweep."""
    scores, is_anomaly = detector.predict(X_test)

    # Threshold-based binary classification
    tp = ((is_anomaly == 1) & (y_test == 1)).sum()
    fp = ((is_anomaly == 1) & (y_test == 0)).sum()
    fn = ((is_anomaly == 0) & (y_test == 1)).sum()
    tn = ((is_anomaly == 0) & (y_test == 0)).sum()

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    auc         = roc_auc_score(y_test, scores)

    print(f"\n[AE] Anomaly Detection Results")
    print(f"  Sensitivity: {sensitivity:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  ROC-AUC    : {auc:.4f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "auc"        : auc,
        "scores"     : scores,
        "is_anomaly" : is_anomaly
    }