import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class RidgeModel:
    def __init__(self, alpha: float = 1.0, pca_n_components: int | None = None):
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha)
        self.pca_n_components = pca_n_components
        self.pca = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X_scaled = self.scaler.fit_transform(X_train)
        if self.pca_n_components:
            from sklearn.decomposition import PCA

            self.pca = PCA(n_components=self.pca_n_components)
            X_scaled = self.pca.fit_transform(X_scaled)
        else:
            # Re-fit with no PCA: drop any stale PCA from a prior load/fit so
            # predict() doesn't apply a transform trained on different data.
            self.pca = None
        self.model.fit(X_scaled, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        if self.pca is not None:
            X_scaled = self.pca.transform(X_scaled)
        return self.model.predict(X_scaled)

    def get_feature_importance(self, feature_names: list[str]) -> pd.Series:
        if self.pca is not None:
            # Map PCA coefficients back to original features via loadings
            original_coefs = self.pca.components_.T @ self.model.coef_
            importance = pd.Series(np.abs(original_coefs), index=feature_names).sort_values(
                ascending=False
            )
        else:
            importance = pd.Series(np.abs(self.model.coef_), index=feature_names).sort_values(
                ascending=False
            )
        return importance

    def save(self, model_dir: str = "outputs/models") -> None:
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.scaler, f"{model_dir}/scaler.pkl")
        joblib.dump(self.model, f"{model_dir}/ridge_model.pkl")
        pca_path = f"{model_dir}/pca.pkl"
        if self.pca is not None:
            joblib.dump(self.pca, pca_path)
        elif os.path.exists(pca_path):
            # A prior run saved a PCA here; this run doesn't use one. Remove
            # it so load() won't resurrect a stale PCA with mismatched shape.
            os.remove(pca_path)

    def load(self, model_dir: str = "outputs/models") -> None:
        self.scaler = joblib.load(f"{model_dir}/scaler.pkl")
        self.model = joblib.load(f"{model_dir}/ridge_model.pkl")
        pca_path = f"{model_dir}/pca.pkl"
        if os.path.exists(pca_path):
            self.pca = joblib.load(pca_path)
        else:
            # No PCA on disk for this run; clear any stale PCA left on self
            # from a previous load, otherwise predict() would apply the old
            # transform to freshly-loaded scaler output.
            self.pca = None
