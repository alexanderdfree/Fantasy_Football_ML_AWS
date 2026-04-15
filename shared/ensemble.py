"""Learned ensemble: per-target convex combination of multiple models."""

import numpy as np
from scipy.optimize import minimize


class LearnedEnsemble:
    """Learn per-target blending weights on validation predictions.

    For each target (and total), finds non-negative weights summing to 1
    that minimize MAE on the validation set.
    """

    def __init__(self, target_names, model_names):
        self.target_names = target_names
        self.model_names = model_names
        self.weights = {}  # {target: np.array of weights}

    def fit(self, val_preds_by_model, y_val_dict):
        """Fit per-target weights on validation predictions.

        Total is derived as sum of per-target blends (not learned separately)
        to avoid overfitting on small val sets.

        Args:
            val_preds_by_model: {model_name: {target: np.array, "total": np.array}}
            y_val_dict: {target: np.array, "total": np.array}
        """
        for target in self.target_names:
            y_true = y_val_dict[target]
            pred_matrix = np.column_stack(
                [val_preds_by_model[m][target] for m in self.model_names]
            )
            n_models = len(self.model_names)
            w0 = np.ones(n_models) / n_models

            def mae_obj(w):
                blend = pred_matrix @ w
                return np.mean(np.abs(y_true - blend))

            result = minimize(
                mae_obj, w0, method="SLSQP",
                bounds=[(0, 1)] * n_models,
                constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
            )
            self.weights[target] = result.x if result.success else w0

    def predict(self, test_preds_by_model):
        """Apply learned weights to test predictions.

        Total is the sum of per-target blended predictions.

        Args:
            test_preds_by_model: {model_name: {target: np.array, "total": np.array}}

        Returns:
            {target: np.array, "total": np.array}
        """
        preds = {}
        for target in self.target_names:
            pred_matrix = np.column_stack(
                [test_preds_by_model[m][target] for m in self.model_names]
            )
            preds[target] = pred_matrix @ self.weights[target]
        preds["total"] = sum(preds[t] for t in self.target_names)
        return preds

    def print_weights(self):
        """Print learned weights per target."""
        print("\n  Ensemble weights (total = sum of per-target blends):")
        for target in self.target_names:
            parts = [f"{m}={w:.3f}" for m, w in zip(self.model_names, self.weights[target])]
            print(f"    {target}: {', '.join(parts)}")
