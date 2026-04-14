"""Training routines for the EEG pipeline."""
# ================================================================
# Unified Trainer for all four downstream models
# ================================================================

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, TensorDataset,
                               WeightedRandomSampler)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import cfg


# ── Generic helper ───────────────────────────────────────────────
def split_and_scale(X       : np.ndarray,
                    y       : np.ndarray,
                    test_sz : float = 0.2,
                    val_sz  : float = 0.1
                    ) -> tuple:
    """
    Returns X_tr, X_val, X_te, y_tr, y_val, y_te, scaler
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_sz, stratify=y, random_state=42)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr, test_size=val_sz,
        stratify=y_tr, random_state=42)

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_val  = scaler.transform(X_val)
    X_te   = scaler.transform(X_te)

    return X_tr, X_val, X_te, y_tr, y_val, y_te, scaler


def make_weighted_loader(X: np.ndarray,
                          y: np.ndarray,
                          batch_size: int = cfg.BATCH_SIZE,
                          is_3d    : bool = False
                          ) -> DataLoader:
    """Create a DataLoader with class-balanced sampling."""
    counts  = np.bincount(y)
    weights = 1.0 / counts
    sw      = torch.tensor([weights[c] for c in y], dtype=torch.float)
    sampler = WeightedRandomSampler(sw, len(sw))

    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    ds = TensorDataset(Xt, yt)
    return DataLoader(ds, batch_size=batch_size, sampler=sampler)


def make_loader(X: np.ndarray,
                y: np.ndarray,
                batch_size: int = cfg.BATCH_SIZE,
                shuffle   : bool = False) -> DataLoader:
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(Xt, yt),
                      batch_size=batch_size, shuffle=shuffle)


# ── Generic PyTorch training loop ────────────────────────────────
def train_torch_model(model      : nn.Module,
                       train_dl  : DataLoader,
                       val_dl    : DataLoader,
                       criterion : nn.Module,
                       optimizer : torch.optim.Optimizer,
                       scheduler ,
                       device    : torch.device,
                       model_name: str,
                       epochs    : int = cfg.EPOCHS,
                       patience  : int = cfg.PATIENCE,
                       forward_fn= None) -> dict:
    """
    Generic training loop with early stopping.

    forward_fn: optional callable(model, batch_X) → logits
                for custom forward (e.g. SNN rate coding)
    """
    save_path = os.path.join(cfg.MODEL_DIR,
                             f"{model_name}_best.pth")
    history   = {"train_loss":[], "val_loss":[],
                 "train_acc" :[], "val_acc" :[]}
    best_val  = float("inf")
    patience_cnt = 0

    print(f"\n[{model_name}] Starting training ...")
    print(f"  Epochs={epochs}  Patience={patience}")

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0

        for Xb, yb in train_dl:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()

            if forward_fn is not None:
                logits = forward_fn(model, Xb)
            else:
                logits = model(Xb)

            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tr_loss    += loss.item() * len(yb)
            tr_correct += (logits.argmax(1) == yb).sum().item()
            tr_total   += len(yb)

        # ── Validation ─────────────────────────────────
        model.eval()
        vl_loss, vl_correct, vl_total = 0.0, 0, 0

        with torch.no_grad():
            for Xb, yb in val_dl:
                Xb, yb = Xb.to(device), yb.to(device)
                if forward_fn is not None:
                    logits = forward_fn(model, Xb)
                else:
                    logits = model(Xb)
                loss        = criterion(logits, yb)
                vl_loss    += loss.item() * len(yb)
                vl_correct += (logits.argmax(1) == yb).sum().item()
                vl_total   += len(yb)

        tr_l = tr_loss / tr_total
        vl_l = vl_loss / vl_total
        tr_a = tr_correct / tr_total
        vl_a = vl_correct / vl_total

        history["train_loss"].append(tr_l)
        history["val_loss"  ].append(vl_l)
        history["train_acc" ].append(tr_a)
        history["val_acc"   ].append(vl_a)
        scheduler.step()

        print(f"  [{epoch:03d}/{epochs}] "
              f"TrLoss={tr_l:.4f} TrAcc={tr_a:.4f} | "
              f"VlLoss={vl_l:.4f} VlAcc={vl_a:.4f}")

        if vl_l < best_val:
            best_val = vl_l
            patience_cnt = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(
        torch.load(save_path, map_location=device))
    print(f"[{model_name}] Best model loaded from {save_path}")
    return history