import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler


class ElasticNetModel:
    """ElasticNet linear model (L1 + L2).

    Mirrors RidgeModel's fit/predict/save/load interface but intentionally
    omits the PCA branch: L1's coordinate-wise sparsity is incompatible with
    PCA's rotated basis (zeroing components != zeroing features). Persists a
    sidecar meta.json with ``{alpha, l1_ratio, converged, n_iter}`` so a
    reviewer can tell whether the CV-selected hyperparameters converged.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        max_iter: int = 5000,
        tol: float = 1e-4,
    ):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.scaler = StandardScaler()
        self.model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            tol=tol,
            random_state=0,
        )
        self.converged = True
        self.n_iter = 0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X_scaled = self.scaler.fit_transform(X_train)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            self.model.fit(X_scaled, y_train)
        self.converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        n_iter_attr = getattr(self.model, "n_iter_", None)
        # sklearn reports n_iter_ as a scalar for single-output ElasticNet.
        if isinstance(n_iter_attr, np.ndarray):
            self.n_iter = int(n_iter_attr.max())
        elif n_iter_attr is not None:
            self.n_iter = int(n_iter_attr)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self.scaler.transform(X))

    def get_feature_importance(self, feature_names: list[str]) -> pd.Series:
        return pd.Series(np.abs(self.model.coef_), index=feature_names).sort_values(ascending=False)

    def save(self, model_dir: str = "outputs/models") -> None:
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.scaler, f"{model_dir}/scaler.pkl")
        joblib.dump(self.model, f"{model_dir}/elasticnet_model.pkl")
        meta = {
            "alpha": float(self.alpha),
            "l1_ratio": float(self.l1_ratio),
            "converged": bool(self.converged),
            "n_iter": int(self.n_iter),
        }
        with open(f"{model_dir}/meta.json", "w") as f:
            json.dump(meta, f)

    def load(self, model_dir: str = "outputs/models") -> None:
        self.scaler = joblib.load(f"{model_dir}/scaler.pkl")
        self.model = joblib.load(f"{model_dir}/elasticnet_model.pkl")
        meta_path = f"{model_dir}/meta.json"
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            self.alpha = meta.get("alpha", self.model.alpha)
            self.l1_ratio = meta.get("l1_ratio", self.model.l1_ratio)
            self.converged = meta.get("converged", True)
            self.n_iter = meta.get("n_iter", 0)
        else:
            self.alpha = self.model.alpha
            self.l1_ratio = self.model.l1_ratio
