import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class RidgeModel:
    def __init__(self, alpha: float = 1.0):
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def get_feature_importance(self, feature_names: list[str]) -> pd.Series:
        importance = pd.Series(
            np.abs(self.model.coef_), index=feature_names
        ).sort_values(ascending=False)
        return importance

    def save(self, model_dir: str = "outputs/models") -> None:
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.scaler, f"{model_dir}/scaler.pkl")
        joblib.dump(self.model, f"{model_dir}/ridge_model.pkl")

    def load(self, model_dir: str = "outputs/models") -> None:
        self.scaler = joblib.load(f"{model_dir}/scaler.pkl")
        self.model = joblib.load(f"{model_dir}/ridge_model.pkl")
