"""
Ensemble layer: combines predictions from all 4 models.

Two ensemble strategies:

    1. **Weighted voting** (default):
       Final = w1 * p_lstm + w2 * p_cnn + w3 * p_transformer + w4 * anomaly_score
       Weights tuned on validation set to maximize AUC-PR (not F1 or ROC-AUC).

    2. **Stacking meta-learner**:
       Train a LogisticRegression on the 4 model probabilities.
       This learns correlations (e.g., CNN is reliable only when LSTM agrees).

Temporal post-processing (MedianFilter + PersistenceFilter) can be
applied after ensemble combination to further reduce false positives.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from typing import Dict, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    Ensemble of seizure detection models with two strategies:

    1. Weighted average + threshold
    2. Stacking meta-learner (LogisticRegression)

    Temporal post-processing is optionally applied after ensemble.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        threshold: float = 0.5,
        use_stacking: bool = False,
    ):
        if weights is None:
            # Default weights (will be tuned)
            weights = {
                "lstm": 0.35,
                "cnn": 0.20,
                "transformer": 0.30,
                "autoencoder": 0.15,
            }
        self.weights = weights
        self.threshold = threshold
        self.use_stacking = use_stacking
        self.meta_learner = None
        self.temporal_processor = None

    def set_temporal_processor(self, processor):
        """Attach a TemporalPostProcessor for post-ensemble filtering."""
        self.temporal_processor = processor

    def _weighted_combine(self, probs: Dict[str, np.ndarray]) -> np.ndarray:
        """Weighted average of model probabilities."""
        combined = np.zeros_like(list(probs.values())[0], dtype=np.float64)
        total_weight = 0.0

        for model_name, prob in probs.items():
            w = self.weights.get(model_name, 0.0)
            combined += w * prob
            total_weight += w

        if total_weight > 0:
            combined /= total_weight

        return combined

    def predict(
        self,
        probs: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Generate ensemble predictions.

        Parameters
        ----------
        probs : dict
            Keys are model names, values are arrays of shape (N,)
            with probabilities/scores in [0, 1].

        Returns
        -------
        predictions : np.ndarray, shape (N,)
            Binary predictions (0/1).
        """
        if self.use_stacking and self.meta_learner is not None:
            combined = self._stacking_predict_proba(probs)
        else:
            combined = self._weighted_combine(probs)

        # Apply temporal post-processing if configured
        if self.temporal_processor is not None:
            _, filtered = self.temporal_processor.process(combined)
            return filtered

        return (combined >= self.threshold).astype(int)

    def predict_proba(
        self,
        probs: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Return combined probability without thresholding."""
        if self.use_stacking and self.meta_learner is not None:
            combined = self._stacking_predict_proba(probs)
        else:
            combined = self._weighted_combine(probs)

        # Apply median smoothing only (not persistence) for probabilities
        if self.temporal_processor is not None and self.temporal_processor.median_filter is not None:
            combined = self.temporal_processor.median_filter.apply(combined)

        return combined

    def _stacking_predict_proba(self, probs: Dict[str, np.ndarray]) -> np.ndarray:
        """Get probability from stacking meta-learner."""
        model_names = sorted(probs.keys())
        X = np.column_stack([probs[name] for name in model_names])
        return self.meta_learner.predict_proba(X)[:, 1]

    # ─────────────────────────────────────────────────────────
    # Stacking Meta-Learner
    # ─────────────────────────────────────────────────────────

    def fit_stacking(
        self,
        probs: Dict[str, np.ndarray],
        y_true: np.ndarray,
    ) -> "EnsemblePredictor":
        """
        Train a LogisticRegression meta-learner on model probabilities.

        Parameters
        ----------
        probs : dict of model_name → probability array
        y_true : true labels (validation set)

        Returns
        -------
        self
        """
        model_names = sorted(probs.keys())
        X = np.column_stack([probs[name] for name in model_names])

        self.meta_learner = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )
        self.meta_learner.fit(X, y_true)
        self.use_stacking = True

        # Log feature importances (coefficients)
        coefs = dict(zip(model_names, self.meta_learner.coef_[0]))
        logger.info(f"Stacking meta-learner coefficients: {coefs}")

        # Evaluate on training data
        y_prob = self.meta_learner.predict_proba(X)[:, 1]
        y_pred = self.meta_learner.predict(X)
        ap = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
        f1 = f1_score(y_true, y_pred, zero_division=0)
        logger.info(f"Stacking meta-learner val AUC-PR={ap:.4f}, F1={f1:.4f}")

        return self

    # ─────────────────────────────────────────────────────────
    # Weight Tuning (AUC-PR based)
    # ─────────────────────────────────────────────────────────

    def tune_weights(
        self,
        probs: Dict[str, np.ndarray],
        y_true: np.ndarray,
        model_names: Optional[List[str]] = None,
        metric: str = "auc_pr",
    ) -> Dict[str, float]:
        """
        Grid search over weight combinations to maximize AUC-PR
        (or F1/Fβ) on the validation set.

        Parameters
        ----------
        probs : dict of model_name → probability array
        y_true : true labels
        model_names : list of model names to tune
        metric : str
            Target metric: "auc_pr" (default), "f1", or "f_beta"

        Returns
        -------
        best_weights : dict of model_name → weight
        """
        if model_names is None:
            model_names = list(probs.keys())

        best_score = -1.0
        best_weights = self.weights.copy()
        best_threshold = 0.5

        n_models = len(model_names)
        if n_models == 0:
            return best_weights

        def _score(combined, threshold):
            """Score a combined probability array."""
            if metric == "auc_pr":
                # AUC-PR doesn't need a threshold
                try:
                    return average_precision_score(y_true, combined)
                except ValueError:
                    return 0.0
            elif metric == "f_beta":
                preds = (combined >= threshold).astype(int)
                # Fβ with β=0.5 (penalize false positives more)
                from sklearn.metrics import fbeta_score
                return fbeta_score(y_true, preds, beta=0.5, zero_division=0)
            else:  # f1
                preds = (combined >= threshold).astype(int)
                return f1_score(y_true, preds, zero_division=0)

        # Grid search over weights and thresholds
        thresholds = np.arange(0.2, 0.8, 0.05) if metric != "auc_pr" else [0.5]

        for t in thresholds:
            if n_models >= 4:
                for w0 in np.arange(0.1, 0.6, 0.1):
                    for w1 in np.arange(0.1, 0.5, 0.1):
                        for w2 in np.arange(0.1, 0.5, 0.1):
                            w3 = 1.0 - w0 - w1 - w2
                            if w3 < 0.05 or w3 > 0.5:
                                continue
                            ws = {
                                model_names[0]: w0,
                                model_names[1]: w1,
                                model_names[2]: w2,
                                model_names[3]: w3,
                            }
                            combined = sum(
                                ws[m] * probs[m] for m in model_names
                            )
                            score = _score(combined, t)
                            if score > best_score:
                                best_score = score
                                best_weights = ws.copy()
                                best_threshold = t
            elif n_models == 3:
                for w0 in np.arange(0.1, 0.8, 0.1):
                    for w1 in np.arange(0.1, 0.8 - w0, 0.1):
                        w2 = 1.0 - w0 - w1
                        if w2 < 0.05:
                            continue
                        ws = {
                            model_names[0]: w0,
                            model_names[1]: w1,
                            model_names[2]: w2,
                        }
                        combined = sum(ws[m] * probs[m] for m in model_names)
                        score = _score(combined, t)
                        if score > best_score:
                            best_score = score
                            best_weights = ws.copy()
                            best_threshold = t
            elif n_models == 2:
                for w0 in np.arange(0.1, 0.9, 0.1):
                    ws = {model_names[0]: w0, model_names[1]: 1.0 - w0}
                    combined = sum(ws[m] * probs[m] for m in model_names)
                    score = _score(combined, t)
                    if score > best_score:
                        best_score = score
                        best_weights = ws.copy()
                        best_threshold = t
            else:
                ws = {model_names[0]: 1.0}
                combined = probs[model_names[0]]
                score = _score(combined, t)
                if score > best_score:
                    best_score = score
                    best_weights = ws.copy()
                    best_threshold = t

        self.weights = best_weights
        self.threshold = best_threshold

        logger.info(f"Tuned ensemble weights ({metric}): {best_weights}")
        logger.info(f"Tuned threshold: {best_threshold:.2f}")
        logger.info(f"Best val {metric}: {best_score:.4f}")

        return best_weights


def collect_model_predictions(
    models: Dict[str, torch.nn.Module],
    test_loader,
    device: torch.device,
    autoencoder_model=None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Collect probabilities from all models on the test set.

    Parameters
    ----------
    models : dict of model_name → model
        Classification models (LSTM, CNN, Transformer).
    test_loader : DataLoader
    device : torch.device
    autoencoder_model : EEGAutoencoder or DeepSVDD, optional

    Returns
    -------
    probs : dict of model_name → np.ndarray of probabilities
    labels : np.ndarray of true labels
    """
    all_probs = {name: [] for name in models}
    if autoencoder_model is not None:
        all_probs["autoencoder"] = []
    all_labels = []

    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        all_labels.extend(y_batch.numpy())

        with torch.no_grad():
            for name, model in models.items():
                model.eval()
                logits = model(x_batch)
                probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                all_probs[name].extend(probs)

            if autoencoder_model is not None:
                autoencoder_model.eval()
                errors = autoencoder_model.anomaly_score(x_batch)
                # Normalize to [0, 1] range
                errors = errors.cpu().numpy()
                all_probs["autoencoder"].extend(errors)

    # Convert to arrays
    result = {name: np.array(probs) for name, probs in all_probs.items()}

    # Normalize autoencoder errors to [0, 1]
    if "autoencoder" in result:
        ae_scores = result["autoencoder"]
        if ae_scores.max() > ae_scores.min():
            result["autoencoder"] = (ae_scores - ae_scores.min()) / (ae_scores.max() - ae_scores.min())
        else:
            result["autoencoder"] = np.zeros_like(ae_scores)

    return result, np.array(all_labels)
