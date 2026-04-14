"""Generic Ridge regression multi-target model for any position."""

import numpy as np
from src.models.linear import RidgeModel


class RidgeMultiTarget:
    """Separate Ridge models for each target in a multi-target decomposition.

    Works for any position — target names are passed at construction time.
    """

    def __init__(self, target_names: list[str], alpha: float = 1.0):
        self.target_names = target_names
        self._models = {name: RidgeModel(alpha=alpha) for name in target_names}

    def fit(self, X_train: np.ndarray, y_train_dict: dict) -> None:
        for name, model in self._models.items():
            model.fit(X_train, y_train_dict[name])

    def predict(self, X: np.ndarray) -> dict:
        """Returns dict of per-target predictions (clamped >= 0) plus total."""
        preds = {
            name: np.maximum(model.predict(X), 0)
            for name, model in self._models.items()
        }
        preds["total"] = sum(preds[t] for t in self.target_names)
        return preds

    def predict_total(self, X: np.ndarray) -> np.ndarray:
        return self.predict(X)["total"]

    def get_feature_importance(self, feature_names: list) -> dict:
        return {
            name: model.get_feature_importance(feature_names)
            for name, model in self._models.items()
        }

    def save(self, model_dir: str) -> None:
        for name, model in self._models.items():
            model.save(f"{model_dir}/{name}")

    def load(self, model_dir: str) -> None:
        for name, model in self._models.items():
            model.load(f"{model_dir}/{name}")
