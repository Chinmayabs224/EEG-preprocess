"""
Ensemble layer: combines predictions from all 4 models.

Weighted voting:
    Final = w1 * p_lstm + w2 * p_cnn + w3 * p_transformer + w4 * recon_error

Weights are tuned on the validation set via grid search.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    Weighted ensemble of 4 seizure detection models.
    
    Combines:
        - LSTM probability
        - CNN probability
        - Transformer probability
        - Autoencoder anomaly score (normalized to [0, 1])
    
    Weights are tuned on the validation set to maximize F1 score.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        threshold: float = 0.5,
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
        combined = np.zeros_like(list(probs.values())[0], dtype=np.float64)
        total_weight = 0.0

        for model_name, prob in probs.items():
            w = self.weights.get(model_name, 0.0)
            combined += w * prob
            total_weight += w

        if total_weight > 0:
            combined /= total_weight

        return (combined >= self.threshold).astype(int)

    def predict_proba(
        self,
        probs: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Return combined probability without thresholding."""
        combined = np.zeros_like(list(probs.values())[0], dtype=np.float64)
        total_weight = 0.0

        for model_name, prob in probs.items():
            w = self.weights.get(model_name, 0.0)
            combined += w * prob
            total_weight += w

        if total_weight > 0:
            combined /= total_weight
        return combined

    def tune_weights(
        self,
        probs: Dict[str, np.ndarray],
        y_true: np.ndarray,
        model_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Grid search over weight combinations to maximize F1 score
        on the validation set.
        
        Parameters
        ----------
        probs : dict of model_name → probability array
        y_true : true labels
        model_names : list of model names to tune (uses all available by default)
        
        Returns
        -------
        best_weights : dict of model_name → weight
        """
        if model_names is None:
            model_names = list(probs.keys())

        best_f1 = 0.0
        best_weights = self.weights.copy()
        best_threshold = 0.5

        # Grid search over weights (step=0.05) and thresholds
        steps = np.arange(0.0, 1.05, 0.05)

        n_models = len(model_names)
        if n_models == 0:
            return best_weights

        # For efficiency, do a coarse search first
        for t in np.arange(0.3, 0.7, 0.05):
            if n_models >= 4:
                # Simplified: try some weight combos
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
                            preds = (combined >= t).astype(int)
                            f1 = f1_score(y_true, preds, zero_division=0)
                            if f1 > best_f1:
                                best_f1 = f1
                                best_weights = ws.copy()
                                best_threshold = t
            else:
                # For fewer models, exhaustive search
                for w0 in np.arange(0.1, 0.9, 0.1):
                    remaining = 1.0 - w0
                    if n_models == 1:
                        ws = {model_names[0]: 1.0}
                    else:
                        for w1 in np.arange(0.1, remaining, 0.1):
                            if n_models == 2:
                                ws = {model_names[0]: w0, model_names[1]: 1.0 - w0}
                            else:
                                w2 = remaining - w1
                                if w2 < 0.05:
                                    continue
                                ws = {
                                    model_names[0]: w0,
                                    model_names[1]: w1,
                                    model_names[2]: w2,
                                }

                            combined = sum(ws[m] * probs[m] for m in model_names)
                            preds = (combined >= t).astype(int)
                            f1 = f1_score(y_true, preds, zero_division=0)
                            if f1 > best_f1:
                                best_f1 = f1
                                best_weights = ws.copy()
                                best_threshold = t

        self.weights = best_weights
        self.threshold = best_threshold

        logger.info(f"Tuned ensemble weights: {best_weights}")
        logger.info(f"Tuned threshold: {best_threshold:.2f}")
        logger.info(f"Best val F1: {best_f1:.4f}")

        return best_weights


def collect_model_predictions(
    models: Dict[str, torch.nn.Module],
    test_loader,
    device: torch.device,
    autoencoder_model=None,
) -> Dict[str, np.ndarray]:
    """
    Collect probabilities from all models on the test set.
    
    Parameters
    ----------
    models : dict of model_name → model
        Classification models (LSTM, CNN, Transformer).
    test_loader : DataLoader
    device : torch.device
    autoencoder_model : EEGAutoencoder, optional
    
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
