"""Visualization utilities for the EEG pipeline."""
# ================================================================
# Comprehensive Visualization for all tasks
# ================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from sklearn.metrics import roc_curve
from config import cfg

plt.style.use("seaborn-v0_8-darkgrid")
SAVE = lambda f: plt.savefig(
    os.path.join(cfg.RESULTS_DIR, f),
    dpi=150, bbox_inches="tight")


# ── 1. Pipeline overview diagram ─────────────────────────────────
def plot_pipeline_overview():
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.axis("off")

    boxes = [
        (5, 9.0, "EEG Data (CHB-MIT chb01)\n"
                  "18 ch · 256 Hz · 7 seizures",
         "#4472C4", "white"),
        (5, 7.5, "Preprocessing\nNotch·Bandpass·CAR·Z-score",
         "#ED7D31", "white"),
        (5, 6.0, "SNN Encoder\nFilterbank → LIF → Spike Rates (64-dim)",
         "#A9D18E", "black"),
    ]
    tasks = [
        (1.5, 3.5, "LSTM/Transformer\nSeizure PREDICTION",   "#FF6B6B"),
        (4.0, 3.5, "CNN\nDisease Classification",             "#4ECDC4"),
        (6.5, 3.5, "Random Forest\nInterpretable Diagnosis",  "#45B7D1"),
        (9.0, 3.5, "VAE Autoencoder\nAnomaly Detection",      "#96CEB4"),
    ]

    for x, y, text, color, tc in boxes:
        ax.add_patch(FancyBboxPatch(
            (x - 2, y - 0.45), 4, 0.9,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="k", lw=1.5))
        ax.text(x, y, text, ha="center", va="center",
                fontsize=9.5, color=tc, fontweight="bold")

    for x, y, text, color in tasks:
        ax.add_patch(FancyBboxPatch(
            (x - 1.3, y - 0.55), 2.6, 1.1,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="k", lw=1.5,
            alpha=0.85))
        ax.text(x, y, text, ha="center", va="center",
                fontsize=8.5, color="black", fontweight="bold")
        ax.annotate("", xy=(x, 4.2), xytext=(5, 5.55),
                    arrowprops=dict(
                        arrowstyle="->", color="gray", lw=1.5))

    for (xa, ya), (xb, yb) in [((5,8.55),(5,7.95)),
                                 ((5,7.05),(5,6.45))]:
        ax.annotate("", xy=(xb,yb), xytext=(xa,ya),
                    arrowprops=dict(
                        arrowstyle="->", color="k", lw=2))

    ax.set_title("Multi-Task EEG Analysis Pipeline  (SNN + Hybrid)",
                 fontsize=14, fontweight="bold", pad=10)
    SAVE("pipeline_overview.png")
    plt.show()


# ── 2. Training curves ───────────────────────────────────────────
def plot_all_training_curves(histories: dict):
    n = len(histories)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4*n))

    for row, (name, hist) in enumerate(histories.items()):
        for col, (m1, m2, title) in enumerate([
            ("train_loss","val_loss","Loss"),
            ("train_acc", "val_acc","Accuracy")
        ]):
            ax = axes[row, col] if n > 1 else axes[col]
            ax.plot(hist[m1], label="Train", lw=2)
            ax.plot(hist[m2], label="Val",   lw=2, ls="--")
            ax.set_title(f"{name} – {title}", fontsize=11)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(title)
            ax.legend()

    plt.suptitle("Training Histories – All Models",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    SAVE("all_training_curves.png")
    plt.show()


# ── 3. Confusion matrix grid ─────────────────────────────────────
def plot_all_confusion_matrices(results: dict):
    n   = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5))
    axes = [axes] if n == 1 else axes

    cls_names = {
        "LSTM/Transformer": ["Interictal","Pre-ictal","Ictal"],
        "CNN"              : ["Normal","Focal","General"],
        "Random Forest"    : ["Normal","Focal","General"],
        "Autoencoder"      : ["Normal","Anomaly"]
    }

    for ax, (name, res) in zip(axes, results.items()):
        cm    = res["cm"]
        names = cls_names.get(name, [str(i) for i in range(cm.shape[0])])
        cm_pct = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        sns.heatmap(cm_pct, annot=cm, fmt="d",
                    xticklabels=names[:cm.shape[1]],
                    yticklabels=names[:cm.shape[0]],
                    cmap="Blues", ax=ax,
                    annot_kws={"size":11})
        ax.set_title(f"{name}\nAUC={res.get('auc',0):.3f}",
                     fontsize=12)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    plt.suptitle("Confusion Matrices – All Tasks",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    SAVE("all_confusion_matrices.png")
    plt.show()


# ── 4. Multi-model ROC curves ────────────────────────────────────
def plot_multi_roc(results: dict):
    fig, ax = plt.subplots(figsize=(8, 7))
    colors  = ["#FF6B6B","#4ECDC4","#45B7D1","#96CEB4"]

    for (name, res), clr in zip(results.items(), colors):
        y_true = res["y_true"]
        y_prob = res["y_prob"]
        if y_prob.ndim == 2:
            y_prob_bin = y_prob[:, -1]
            y_true_bin = (y_true > 0).astype(int)
        else:
            y_prob_bin = y_prob
            y_true_bin = y_true

        fpr, tpr, _ = roc_curve(y_true_bin, y_prob_bin)
        auc = res.get("auc", 0.0)
        ax.plot(fpr, tpr, lw=2.5, color=clr,
                label=f"{name}  (AUC={auc:.3f})")

    ax.plot([0,1],[0,1],"k--", lw=1)
    ax.fill_between([0,1],[0,1], alpha=0.05, color="gray")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curves – All Models", fontsize=14)
    ax.legend(fontsize=11)
    SAVE("multi_roc_curves.png")
    plt.show()


# ── 5. EEG + multi-task prediction overlay ───────────────────────
def plot_multitask_overlay(signals      : np.ndarray,
                            binary_labels: np.ndarray,
                            pred_dict    : dict,
                            fname        : str,
                            duration_s   : int = 300):
    fs       = cfg.SAMPLING_RATE
    n_show   = duration_s * fs
    t        = np.arange(n_show) / fs

    n_tasks  = len(pred_dict)
    fig = plt.figure(figsize=(18, 3 + 2*n_tasks))
    gs  = gridspec.GridSpec(1 + n_tasks, 1,
                             height_ratios=[3]+[1]*n_tasks)

    # ── EEG trace (channel 0) ──────────────────────────
    ax0 = fig.add_subplot(gs[0])
    eeg = signals[0, :n_show]
    ax0.plot(t, eeg, lw=0.5, color="steelblue")
    ax0.set_ylabel("Amplitude (µV)", fontsize=10)
    ax0.set_title(f"EEG + Multi-Task Predictions — {fname}",
                  fontsize=13)

    _shade(ax0, t, binary_labels[:n_show],
           "red", 0.25, "True Seizure")
    ax0.legend(fontsize=9, loc="upper right")

    # ── Per-task prediction bars ───────────────────────
    colors = ["#FF6B6B","#4ECDC4","#45B7D1","#96CEB4"]
    for row, ((task_name, preds), clr) in enumerate(
            zip(pred_dict.items(), colors), start=1):
        ax = fig.add_subplot(gs[row], sharex=ax0)
        ep_len = cfg.EPOCH_SAMPLES * cfg.WINDOW_SIZE
        n_pred = min(n_show // ep_len, len(preds))
        pred_ex = np.repeat(preds[:n_pred], ep_len)[:n_show]

        ax.fill_between(t[:len(pred_ex)],
                        pred_ex > 0, alpha=0.65,
                        color=clr)
        ax.set_ylabel(task_name, fontsize=8, rotation=0,
                      labelpad=55)
        ax.set_ylim(0, 1.3)
        if row < 1 + n_tasks - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel("Time (s)", fontsize=11)

    plt.tight_layout()
    SAVE(f"multitask_overlay_{fname.replace('.edf','')}.png")
    plt.show()


def _shade(ax, t, labels, color, alpha, label):
    in_r = False; added = False
    for i, lb in enumerate(labels):
        if lb and not in_r:
            xs = t[i]; in_r = True
        elif not lb and in_r:
            kw = dict(color=color, alpha=alpha)
            if not added:
                kw["label"] = label; added = True
            ax.axvspan(xs, t[i], **kw); in_r = False


# ── 6. Summary dashboard ─────────────────────────────────────────
def plot_summary_dashboard(all_results: dict):
    """Bar chart comparing AUC across all models."""
    names  = list(all_results.keys())
    aucs   = [r.get("auc", 0) for r in all_results.values()]
    sens   = [r.get("sensitivity", 0) for r in all_results.values()]
    spec   = [r.get("specificity", 0) for r in all_results.values()]

    x   = np.arange(len(names))
    w   = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))

    b1 = ax.bar(x - w,   aucs, w, label="AUC",
                color="#4472C4", edgecolor="k", alpha=0.85)
    b2 = ax.bar(x,       sens, w, label="Sensitivity",
                color="#ED7D31", edgecolor="k", alpha=0.85)
    b3 = ax.bar(x + w,   spec, w, label="Specificity",
                color="#A9D18E", edgecolor="k", alpha=0.85)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2,
                    h + 0.01, f"{h:.2f}",
                    ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Multi-Task Performance Summary", fontsize=14)
    ax.legend(fontsize=11)
    ax.axhline(0.5, color="red", lw=1, ls="--", alpha=0.4)

    SAVE("summary_dashboard.png")
    plt.show()