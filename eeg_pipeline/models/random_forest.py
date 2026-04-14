"""Random forest model for interpretable diagnosis."""
# ================================================================
# Task 3: Interpretable Diagnosis (Random Forest)
#
# Input : handcrafted + SNN spike features
# Output: diagnostic class + SHAP feature importance
# ================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (classification_report,
                              confusion_matrix,
                              roc_auc_score)
from sklearn.preprocessing import label_binarize
from config import cfg


# ── Handcrafted EEG Features ─────────────────────────────────────
def extract_handcrafted_features(signals: np.ndarray,
                                  epoch_len: int
                                  ) -> np.ndarray:
    """
    Statistical + spectral features per epoch per channel.

    Returns (n_epochs, n_features)  where
    n_features = N_CH × 10 statistical features
    """
    n_ch    = signals.shape[0]
    n_samp  = signals.shape[1]
    n_ep    = n_samp // epoch_len
    fs      = cfg.SAMPLING_RATE

    feat_list = []
    for e in range(n_ep):
        s   = e * epoch_len
        seg = signals[:, s:s + epoch_len]   # (N_CH, epoch_len)
        ep_feats = []

        for ch in range(n_ch):
            x  = seg[ch]
            fft_vals = np.abs(np.fft.rfft(x))
            freqs    = np.fft.rfftfreq(epoch_len, 1/fs)

            def band_power(lo, hi):
                mask = (freqs >= lo) & (freqs < hi)
                return fft_vals[mask].sum()

            ep_feats += [
                np.mean(x),                         # mean
                np.std(x),                          # std
                np.var(x),                          # variance
                np.abs(x).mean(),                   # mean absolute
                np.percentile(x, 75) -
                np.percentile(x, 25),               # IQR
                band_power(0.5, 4),                 # delta
                band_power(4,   8),                 # theta
                band_power(8,  12),                 # alpha
                band_power(12, 30),                 # beta
                band_power(30, 40),                 # gamma
            ]
        feat_list.append(ep_feats)

    return np.array(feat_list, dtype=np.float32)   # (n_ep, N_CH*10)


def build_rf_features(records     : list[dict],
                       spike_feats : dict[str, np.ndarray]
                       ) -> tuple[np.ndarray, np.ndarray]:
    """
    Concatenate handcrafted features + SNN spike rates.

    Returns X (N, feat_dim), y (N,)
    """
    X_all, y_all = [], []
    ep = cfg.EPOCH_SAMPLES
    W  = cfg.WINDOW_SIZE

    for rec in records:
        fname  = rec["fname"]
        sig    = rec["signals"]
        lbl    = rec["labels"]["disease"]

        hc  = extract_handcrafted_features(sig, ep)
        n_ep = hc.shape[0]

        if fname in spike_feats:
            spk = spike_feats[fname]
            # Align lengths
            n   = min(n_ep - W + 1, len(spk))
            hc_aligned = hc[W - 1: W - 1 + n]
            spk_aligned = spk[:n]
            X = np.concatenate([hc_aligned, spk_aligned], axis=1)
        else:
            n   = n_ep - W + 1
            X   = hc[W - 1: W - 1 + n]

        y = np.array([
            int(lbl[(W - 1 + i) * ep:
                    (W - 1 + i) * ep + ep].mean() > 0.5)
            for i in range(n)
        ])

        X_all.append(X)
        y_all.append(y)

    return (np.concatenate(X_all, axis=0),
            np.concatenate(y_all, axis=0))


# ── Random Forest Model ──────────────────────────────────────────
class InterpretableRF:
    """
    Random Forest with permutation importance and
    visualised feature ranking.
    """

    CLASS_NAMES = ["Normal", "Focal", "Generalized"]

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators = cfg.RF_N_ESTIMATORS,
            max_depth    = cfg.RF_MAX_DEPTH,
            max_features = cfg.RF_N_FEATURES,
            class_weight = "balanced",
            random_state = 42,
            n_jobs       = -1
        )
        self.feature_names   = None
        self.importance_df   = None

    def build_feature_names(self, n_hc_feats: int,
                             n_spk_feats: int) -> list[str]:
        stat_names = ["mean","std","var","abs_mean","iqr",
                      "delta","theta","alpha","beta","gamma"]
        hc_names   = [f"ch{c}_{s}"
                      for c in range(cfg.N_CHANNELS)
                      for s in stat_names]
        spk_names  = [f"spk_{i}" for i in range(n_spk_feats)]
        self.feature_names = hc_names[:n_hc_feats] + spk_names
        return self.feature_names

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        print("[RF] Training Random Forest ...")
        self.model.fit(X_train, y_train)
        print(f"[RF] OOB score: "
              f"{self.model.oob_score_:.4f}"
              if hasattr(self.model, 'oob_score_') else "")

    def evaluate(self, X_test: np.ndarray,
                       y_test: np.ndarray) -> dict:
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)

        print("\n[RF] Classification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=self.CLASS_NAMES[:len(np.unique(y_test))]))

        result = {
            "y_true": y_test,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "cm"    : confusion_matrix(y_test, y_pred)
        }
        return result

    def compute_feature_importance(self,
                                    X_val: np.ndarray,
                                    y_val: np.ndarray,
                                    top_k: int = 30) -> pd.DataFrame:
        """Permutation importance on validation set."""
        print("[RF] Computing permutation importance ...")
        perm = permutation_importance(
            self.model, X_val, y_val,
            n_repeats=10, random_state=42, n_jobs=-1)

        names = (self.feature_names if self.feature_names
                 else [f"f{i}" for i in range(X_val.shape[1])])

        df = pd.DataFrame({
            "feature"   : names[:X_val.shape[1]],
            "importance": perm.importances_mean,
            "std"       : perm.importances_std
        }).sort_values("importance", ascending=False).head(top_k)

        self.importance_df = df
        return df

    def plot_importance(self, top_k: int = 20):
        if self.importance_df is None:
            raise RuntimeError("Run compute_feature_importance first")
        df = self.importance_df.head(top_k)

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.RdYlGn(
            np.linspace(0.3, 0.9, len(df)))[::-1]
        bars = ax.barh(range(len(df)),
                       df["importance"].values,
                       xerr=df["std"].values,
                       color=colors, edgecolor="k",
                       linewidth=0.5, alpha=0.85)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df["feature"].values, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Permutation Importance", fontsize=12)
        ax.set_title(
            f"Top-{top_k} Diagnostic Features (Random Forest)",
            fontsize=13)
        ax.axvline(0, color="k", lw=0.8)
        plt.tight_layout()
        path = os.path.join(cfg.RESULTS_DIR,
                            "rf_feature_importance.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"[RF] Importance plot saved: {path}")

    def save(self, path: str = None):
        path = path or os.path.join(cfg.MODEL_DIR, "rf_model.pkl")
        joblib.dump(self.model, path)
        print(f"[RF] Model saved: {path}")

    def load(self, path: str):
        self.model = joblib.load(path)