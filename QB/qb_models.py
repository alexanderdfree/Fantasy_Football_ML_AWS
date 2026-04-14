"""QB-specific Ridge regression multi-target model."""

import numpy as np
from src.models.linear import RidgeModel


class QBRidgeMultiTarget:
    """Three separate Ridge models predicting passing_floor, rushing_floor, td_points."""

    def __init__(self, alpha: float = 1.0):
        self.passing_model = RidgeModel(alpha=alpha)
        self.rushing_model = RidgeModel(alpha=alpha)
        self.td_model = RidgeModel(alpha=alpha)
        self.target_names = ["passing_floor", "rushing_floor", "td_points"]

    def fit(self, X_train: np.ndarray, y_train_dict: dict) -> None:
        """
        Args:
            X_train: Feature array, shape (n_samples, n_features)
            y_train_dict: {
                "passing_floor": np.ndarray shape (n_samples,),
                "rushing_floor": np.ndarray shape (n_samples,),
                "td_points": np.ndarray shape (n_samples,),
            }
        """
        self.passing_model.fit(X_train, y_train_dict["passing_floor"])
        self.rushing_model.fit(X_train, y_train_dict["rushing_floor"])
        self.td_model.fit(X_train, y_train_dict["td_points"])

    def predict(self, X: np.ndarray) -> dict:
        """Returns dict of per-target predictions plus total.

        Sub-target predictions are clamped to non-negative since these
        represent physical quantities (passing_yards*0.04, rushing_yards*0.1, TDs*points).
        """
        preds = {
            "passing_floor": np.maximum(self.passing_model.predict(X), 0),
            "rushing_floor": np.maximum(self.rushing_model.predict(X), 0),
            "td_points": np.maximum(self.td_model.predict(X), 0),
        }
        preds["total"] = preds["passing_floor"] + preds["rushing_floor"] + preds["td_points"]
        return preds

    def predict_total(self, X: np.ndarray) -> np.ndarray:
        """Convenience: returns just the total fantasy points prediction."""
        preds = self.predict(X)
        return preds["total"]

    def get_feature_importance(self, feature_names: list) -> dict:
        """Per-target feature importance from Ridge coefficients."""
        return {
            "passing_floor": self.passing_model.get_feature_importance(feature_names),
            "rushing_floor": self.rushing_model.get_feature_importance(feature_names),
            "td_points": self.td_model.get_feature_importance(feature_names),
        }

    def save(self, model_dir: str = "QB/outputs/models") -> None:
        self.passing_model.save(f"{model_dir}/passing")
        self.rushing_model.save(f"{model_dir}/rushing")
        self.td_model.save(f"{model_dir}/td")

    def load(self, model_dir: str = "QB/outputs/models") -> None:
        self.passing_model.load(f"{model_dir}/passing")
        self.rushing_model.load(f"{model_dir}/rushing")
        self.td_model.load(f"{model_dir}/td")
