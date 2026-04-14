import numpy as np
from src.models.linear import RidgeModel


class MultiTargetRidge:
    """N separate Ridge models, one per target. Generic across positions."""

    def __init__(self, target_names: list[str], alpha: float = 1.0):
        self.target_names = target_names
        self.models = {name: RidgeModel(alpha=alpha) for name in target_names}

    def fit(self, X_train: np.ndarray, y_train_dict: dict) -> None:
        for name in self.target_names:
            self.models[name].fit(X_train, y_train_dict[name])

    def predict(self, X: np.ndarray) -> dict:
        preds = {}
        for name in self.target_names:
            preds[name] = np.maximum(self.models[name].predict(X), 0)
        preds["total"] = sum(preds[name] for name in self.target_names)
        return preds

    def get_feature_importance(self, feature_names: list) -> dict:
        return {
            name: self.models[name].get_feature_importance(feature_names)
            for name in self.target_names
        }

    def save(self, model_dir: str) -> None:
        for name in self.target_names:
            self.models[name].save(f"{model_dir}/{name}")

    def load(self, model_dir: str) -> None:
        for name in self.target_names:
            self.models[name].load(f"{model_dir}/{name}")
