"""Generic multi-target models for any position (Ridge, Ordinal, LightGBM)."""

import json
import os
import shutil
import sys

import joblib
import lightgbm as lgb
import mord
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from src.models.elastic_net import ElasticNetModel
from src.models.linear import RidgeModel

# LightGBM thread count for tree learning. ``-1`` (all cores) crashes on macOS
# under nested OpenMP runtimes (libomp from numpy/sklearn vs. LightGBM's
# bundled libgomp) — the segfault reproduces inside ``Dataset.__init_from_np2d``
# the moment the construct call hits the OMP region. Linux (CI + EC2
# production) is safe and gets full parallelism; other platforms stay at 1
# unless ``LGBM_N_JOBS`` is set explicitly to override.
_LGBM_N_JOBS = -1 if sys.platform.startswith("linux") else int(os.environ.get("LGBM_N_JOBS", "1"))


class TwoStageRidge:
    """Two-stage model for zero-inflated targets (e.g., rushing_tds).

    Stage 1: Logistic regression classifies P(target > 0).
    Stage 2: Ridge regresses E[target | target > 0] on positive-only subset.
    Prediction: 0 when P < threshold, else E[target | target > 0].
    """

    def __init__(self, clf_C=0.001, ridge_alpha=0.01, threshold=0.5):
        self.clf_C = clf_C
        self.ridge_alpha = ridge_alpha
        self.threshold = threshold

    def fit(self, X_train, y_train):
        self.scaler_clf = StandardScaler()
        X_s = self.scaler_clf.fit_transform(X_train)
        self.clf = LogisticRegression(C=self.clf_C, max_iter=1000, solver="lbfgs")
        self.clf.fit(X_s, (y_train > 0).astype(int))

        pos_mask = y_train > 0
        self.scaler_reg = StandardScaler()
        X_pos = self.scaler_reg.fit_transform(X_train[pos_mask])
        self.reg = Ridge(alpha=self.ridge_alpha)
        self.reg.fit(X_pos, y_train[pos_mask])

    def predict(self, X):
        p = self.clf.predict_proba(self.scaler_clf.transform(X))[:, 1]
        e = np.maximum(self.reg.predict(self.scaler_reg.transform(X)), 0)
        return np.where(p >= self.threshold, e, 0)

    def save(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.scaler_clf, f"{model_dir}/scaler_clf.pkl")
        joblib.dump(self.clf, f"{model_dir}/classifier.pkl")
        joblib.dump(self.scaler_reg, f"{model_dir}/scaler_reg.pkl")
        joblib.dump(self.reg, f"{model_dir}/ridge_model.pkl")

    def load(self, model_dir):
        self.scaler_clf = joblib.load(f"{model_dir}/scaler_clf.pkl")
        self.clf = joblib.load(f"{model_dir}/classifier.pkl")
        self.scaler_reg = joblib.load(f"{model_dir}/scaler_reg.pkl")
        self.reg = joblib.load(f"{model_dir}/ridge_model.pkl")


class OrdinalTDClassifier:
    """Ordinal logistic regression for discrete TD count predictions.

    Converts raw TD counts to integer class labels, fits mord.LogisticAT
    (cumulative logit model with all-thresholds variant), and predicts
    E[TDs] via class probabilities. Enforces P(Y >= k) monotonically
    decreasing.
    """

    def __init__(
        self, class_values: list[float] | str = "auto", n_classes: int = 4, alpha: float = 1.0
    ):
        self.alpha = alpha
        self._class_values_cfg = class_values  # e.g. [0, 1, 2, 3] raw counts
        self._n_classes = n_classes

    # -- internal helpers --------------------------------------------------
    def _points_to_labels(self, y: np.ndarray) -> np.ndarray:
        """Map raw target values to integer class labels.

        For ``class_values=[a, a+s, a+2s, ...]`` with a uniform step ``s``,
        label = round((y - a) / s). Current configs use
        ``class_values=[0, 1, 2, 3]`` (step=1, raw TD counts).
        """
        if isinstance(self._class_values_cfg, list):
            cv = self._class_values_cfg
            base = cv[0] if len(cv) > 0 else 0
            step = (cv[1] - cv[0]) if len(cv) > 1 else 1
            if step == 0:
                step = 1
            labels = np.round((y - base) / step).astype(int)
            labels = np.clip(labels, 0, self._n_classes - 1)
        else:
            # "auto" — assume raw TD counts
            labels = np.round(y).astype(int)
            labels = np.clip(labels, 0, self._n_classes - 1)
        return labels

    def _compute_class_point_values(self, y: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Compute empirical mean target value for each class."""
        values = np.zeros(self._n_classes)
        for k in range(self._n_classes):
            mask = labels == k
            values[k] = (
                y[mask].mean()
                if mask.any()
                else (
                    self._class_values_cfg[k] if isinstance(self._class_values_cfg, list) else k * 5
                )
            )
        return values

    # -- sklearn-compatible interface --------------------------------------
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        labels = self._points_to_labels(y_train)
        self._n_classes = max(self._n_classes, labels.max() + 1)
        self.class_point_values_ = self._compute_class_point_values(y_train, labels)

        self.scaler_ = StandardScaler()
        X_s = self.scaler_.fit_transform(X_train)

        self.clf_ = mord.LogisticAT(alpha=self.alpha, max_iter=1000)
        self.clf_.fit(X_s, labels)

    def _predict_proba(self, X_scaled: np.ndarray) -> np.ndarray:
        """Compute class probabilities from the cumulative model.

        mord.LogisticAT stores:
          - coef_  : (n_features,) shared coefficient vector
          - theta_  : (n_trained_classes-1,) ordered thresholds

        Cumulative probs: P(Y <= k) = sigmoid(theta_k - X @ coef_)
        Class probs:      P(Y = 0) = P(Y <= 0)
                          P(Y = k) = P(Y <= k) - P(Y <= k-1)
                          P(Y = K-1) = 1 - P(Y <= K-2)

        ``self.class_point_values_`` may advertise more classes than mord
        actually trained on (happens when training labels never hit the
        upper classes — e.g. rare 3-TD games). Any unseen upper class gets
        zero probability mass.
        """
        n = X_scaled.shape[0]
        n_advertised = len(self.class_point_values_)
        linear = X_scaled @ self.clf_.coef_

        # P(Y <= k) for k = 0, ..., n_trained-2
        thetas = self.clf_.theta_
        n_trained = len(thetas) + 1

        proba = np.zeros((n, n_advertised))
        if n_trained == 1:
            # Degenerate: mord saw only one class in training. Assign all
            # mass to that class (index 0 == class_point_values_[0]).
            proba[:, 0] = 1.0
            return proba

        cum_le = np.column_stack(
            [1.0 / (1.0 + np.exp(-(theta - linear))) for theta in thetas]
        )  # (n, n_trained-1)
        proba[:, 0] = cum_le[:, 0]
        for k in range(1, n_trained - 1):
            proba[:, k] = cum_le[:, k] - cum_le[:, k - 1]
        # Terminal trained class gets the residual from the last cum_le.
        proba[:, n_trained - 1] = 1.0 - cum_le[:, -1]

        proba = np.clip(proba, 0, 1)
        row_sums = proba.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # defensive — all-zero rows stay zero
        proba /= row_sums
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return E[target] = sum(P(class_k) * class_values_k)."""
        proba = self._predict_proba(self.scaler_.transform(X))
        return proba @ self.class_point_values_

    def save(self, model_dir: str) -> None:
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.scaler_, f"{model_dir}/td_scaler.pkl")
        joblib.dump(self.clf_, f"{model_dir}/td_classifier.pkl")
        meta = {
            "class_point_values": self.class_point_values_.tolist(),
            "n_classes": int(self._n_classes),
            "alpha": self.alpha,
        }
        with open(f"{model_dir}/td_classifier_meta.json", "w") as f:
            json.dump(meta, f)

    def load(self, model_dir: str) -> None:
        self.scaler_ = joblib.load(f"{model_dir}/td_scaler.pkl")
        self.clf_ = joblib.load(f"{model_dir}/td_classifier.pkl")
        with open(f"{model_dir}/td_classifier_meta.json") as f:
            meta = json.load(f)
        self.class_point_values_ = np.array(meta["class_point_values"])
        self._n_classes = meta["n_classes"]
        self.alpha = meta["alpha"]


class GatedOrdinalTDClassifier:
    """Binary gate (logistic) + ordinal classification on positives.

    Stage 1: LogisticRegression classifies P(target > 0) with hard threshold.
    Stage 2: OrdinalTDClassifier over {1, 2, 3+} TDs on the positive subset.
    Prediction: 0 when P < threshold, else E[target | target > 0].
    """

    def __init__(
        self,
        class_values: list[float] | str = "auto",
        n_classes: int = 4,
        alpha: float = 1.0,
        clf_C: float = 0.001,
        threshold: float = 0.5,
    ):
        self.clf_C = clf_C
        self.threshold = threshold
        # Ordinal stage operates on classes {1, 2, 3+} (no zero class)
        self._class_values_cfg = class_values
        self._ordinal_alpha = alpha
        self._n_classes = n_classes

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        # Stage 1: binary gate
        self.scaler_gate_ = StandardScaler()
        X_s = self.scaler_gate_.fit_transform(X_train)
        self.gate_ = LogisticRegression(C=self.clf_C, max_iter=1000, solver="lbfgs")
        self.gate_.fit(X_s, (y_train > 0).astype(int))

        # Stage 2: ordinal on positives only, over {1, 2, 3+} TDs
        pos_mask = y_train > 0
        if isinstance(self._class_values_cfg, list):
            pos_values = self._class_values_cfg[1:]  # drop the 0 class
        else:
            pos_values = self._class_values_cfg
        self.ordinal_ = OrdinalTDClassifier(
            class_values=pos_values,
            n_classes=self._n_classes - 1,  # one fewer class (no zero)
            alpha=self._ordinal_alpha,
        )
        self.ordinal_.fit(X_train[pos_mask], y_train[pos_mask])

    def predict(self, X: np.ndarray) -> np.ndarray:
        p = self.gate_.predict_proba(self.scaler_gate_.transform(X))[:, 1]
        ev = np.maximum(self.ordinal_.predict(X), 0)
        return np.where(p >= self.threshold, ev, 0)

    def save(self, model_dir: str) -> None:
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.scaler_gate_, f"{model_dir}/scaler_clf.pkl")
        joblib.dump(self.gate_, f"{model_dir}/classifier.pkl")
        self.ordinal_.save(model_dir)
        meta_path = f"{model_dir}/td_classifier_meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        meta["gated"] = True
        meta["clf_C"] = self.clf_C
        meta["threshold"] = self.threshold
        with open(meta_path, "w") as f:
            json.dump(meta, f)

    def load(self, model_dir: str) -> None:
        self.scaler_gate_ = joblib.load(f"{model_dir}/scaler_clf.pkl")
        self.gate_ = joblib.load(f"{model_dir}/classifier.pkl")
        self.ordinal_ = OrdinalTDClassifier()
        self.ordinal_.load(model_dir)
        with open(f"{model_dir}/td_classifier_meta.json") as f:
            meta = json.load(f)
        self.clf_C = meta.get("clf_C", 0.001)
        self.threshold = meta.get("threshold", 0.5)


class RidgeMultiTarget:
    """Separate Ridge models for each target in a multi-target decomposition.

    Works for any position — target names are passed at construction time.
    Accepts a single alpha (shared) or a dict mapping target names to alphas.
    """

    def __init__(
        self,
        target_names: list[str],
        alpha: float | dict[str, float] = 1.0,
        two_stage_targets: dict | None = None,
        classification_targets: dict | None = None,
        pca_n_components: int | None = None,
        non_negative_targets: set | None = None,
    ):
        self.target_names = target_names
        # Which targets are clamped to >= 0. Default: all targets.
        # Override for targets that can be negative.
        self.non_negative_targets = (
            set(target_names) if non_negative_targets is None else non_negative_targets
        )
        self._two_stage_targets = two_stage_targets or {}
        self._classification_targets = classification_targets or {}
        special = set(self._two_stage_targets) | set(self._classification_targets)
        if isinstance(alpha, dict):
            missing = set(target_names) - set(alpha) - special
            if missing:
                raise ValueError(f"alpha dict missing keys for targets: {missing}")
            self._alphas = {name: alpha.get(name, 1.0) for name in target_names}
        else:
            self._alphas = {name: alpha for name in target_names}
        self._models = {}
        for name in target_names:
            if name in self._classification_targets:
                cfg = dict(self._classification_targets[name])
                model_type = cfg.pop("type", "ordinal")
                if model_type == "gated_ordinal":
                    self._models[name] = GatedOrdinalTDClassifier(**cfg)
                else:
                    self._models[name] = OrdinalTDClassifier(**cfg)
            elif name in self._two_stage_targets:
                self._models[name] = TwoStageRidge(**self._two_stage_targets[name])
            else:
                self._models[name] = RidgeModel(
                    alpha=self._alphas[name], pca_n_components=pca_n_components
                )

    def fit(self, X_train: np.ndarray, y_train_dict: dict) -> None:
        for name, model in self._models.items():
            model.fit(X_train, y_train_dict[name])

    def predict(self, X: np.ndarray) -> dict:
        """Returns dict of per-target predictions."""
        preds = {}
        for name, model in self._models.items():
            pred = model.predict(X)
            if name in self.non_negative_targets:
                pred = np.maximum(pred, 0)
            preds[name] = pred
        return preds

    def predict_total(self, X: np.ndarray) -> np.ndarray:
        preds = self.predict(X)
        return sum(preds[t] for t in self.target_names)

    def get_feature_importance(self, feature_names: list) -> dict:
        return {
            name: model.get_feature_importance(feature_names)
            for name, model in self._models.items()
            if hasattr(model, "get_feature_importance")
        }

    def save(self, model_dir: str) -> None:
        for name, model in self._models.items():
            target_dir = f"{model_dir}/{name}"
            # Wipe any prior run's artifacts before saving. load() infers the
            # model type from files on disk (td_classifier_meta.json → gated,
            # pca.pkl → PCA-enabled), so a leftover sidecar from a previous
            # run with a different model type or feature count survives the
            # save and crashes at inference.
            if os.path.isdir(target_dir):
                shutil.rmtree(target_dir)
            model.save(target_dir)

    def load(self, model_dir: str) -> None:
        for name in self.target_names:
            target_dir = f"{model_dir}/{name}"
            meta_path = f"{target_dir}/td_classifier_meta.json"
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                if meta.get("gated"):
                    self._models[name] = GatedOrdinalTDClassifier()
                else:
                    self._models[name] = OrdinalTDClassifier()
            elif os.path.exists(f"{target_dir}/classifier.pkl"):
                self._models[name] = TwoStageRidge()
            else:
                self._models[name] = RidgeModel()
            self._models[name].load(target_dir)


class ElasticNetMultiTarget:
    """ElasticNet parallel to RidgeMultiTarget (L1+L2 linear baseline).

    Replaces only the vanilla RidgeModel branch with ElasticNet; two-stage and
    ordinal classification targets keep their existing domain-specific classes
    (those aren't plain linear regressions). Never uses PCA — L1 on a rotated
    basis zeros components, not original features, which defeats the purpose.
    """

    def __init__(
        self,
        target_names: list[str],
        alpha: float | dict[str, float] = 1.0,
        l1_ratio: float | dict[str, float] = 0.5,
        two_stage_targets: dict | None = None,
        classification_targets: dict | None = None,
        non_negative_targets: set | None = None,
        max_iter: int = 5000,
        tol: float = 1e-4,
    ):
        self.target_names = target_names
        self.non_negative_targets = (
            set(target_names) if non_negative_targets is None else non_negative_targets
        )
        self._two_stage_targets = two_stage_targets or {}
        self._classification_targets = classification_targets or {}
        special = set(self._two_stage_targets) | set(self._classification_targets)
        if isinstance(alpha, dict):
            missing = set(target_names) - set(alpha) - special
            if missing:
                raise ValueError(f"alpha dict missing keys for targets: {missing}")
            self._alphas = {name: alpha.get(name, 1.0) for name in target_names}
        else:
            self._alphas = {name: alpha for name in target_names}
        if isinstance(l1_ratio, dict):
            missing = set(target_names) - set(l1_ratio) - special
            if missing:
                raise ValueError(f"l1_ratio dict missing keys for targets: {missing}")
            self._l1_ratios = {name: l1_ratio.get(name, 0.5) for name in target_names}
        else:
            self._l1_ratios = {name: l1_ratio for name in target_names}
        self._models = {}
        for name in target_names:
            if name in self._classification_targets:
                cfg = dict(self._classification_targets[name])
                model_type = cfg.pop("type", "ordinal")
                if model_type == "gated_ordinal":
                    self._models[name] = GatedOrdinalTDClassifier(**cfg)
                else:
                    self._models[name] = OrdinalTDClassifier(**cfg)
            elif name in self._two_stage_targets:
                self._models[name] = TwoStageRidge(**self._two_stage_targets[name])
            else:
                self._models[name] = ElasticNetModel(
                    alpha=self._alphas[name],
                    l1_ratio=self._l1_ratios[name],
                    max_iter=max_iter,
                    tol=tol,
                )

    def fit(self, X_train: np.ndarray, y_train_dict: dict) -> None:
        for name, model in self._models.items():
            model.fit(X_train, y_train_dict[name])

    def predict(self, X: np.ndarray) -> dict:
        preds = {}
        for name, model in self._models.items():
            pred = model.predict(X)
            if name in self.non_negative_targets:
                pred = np.maximum(pred, 0)
            preds[name] = pred
        return preds

    def predict_total(self, X: np.ndarray) -> np.ndarray:
        preds = self.predict(X)
        return sum(preds[t] for t in self.target_names)

    def get_feature_importance(self, feature_names: list) -> dict:
        return {
            name: model.get_feature_importance(feature_names)
            for name, model in self._models.items()
            if hasattr(model, "get_feature_importance")
        }

    def convergence_report(self) -> dict:
        """Per-target convergence status for ElasticNet heads.

        Used by the pipeline to log whether CV-selected (alpha, l1_ratio) pairs
        actually converged. Two-stage and ordinal heads are omitted — they
        don't expose a ConvergenceWarning path in a way that's meaningful here.
        """
        return {
            name: {"converged": model.converged, "n_iter": model.n_iter}
            for name, model in self._models.items()
            if isinstance(model, ElasticNetModel)
        }

    def save(self, model_dir: str) -> None:
        for name, model in self._models.items():
            target_dir = f"{model_dir}/{name}"
            # Match RidgeMultiTarget's guard: load() infers the model type from
            # on-disk sidecars, so a prior run's stale files corrupt the inferred
            # type. Wiping guarantees a clean state per save.
            if os.path.isdir(target_dir):
                shutil.rmtree(target_dir)
            model.save(target_dir)

    def load(self, model_dir: str) -> None:
        for name in self.target_names:
            target_dir = f"{model_dir}/{name}"
            td_meta_path = f"{target_dir}/td_classifier_meta.json"
            if os.path.exists(td_meta_path):
                with open(td_meta_path) as f:
                    meta = json.load(f)
                if meta.get("gated"):
                    self._models[name] = GatedOrdinalTDClassifier()
                else:
                    self._models[name] = OrdinalTDClassifier()
            elif os.path.exists(f"{target_dir}/classifier.pkl"):
                self._models[name] = TwoStageRidge()
            else:
                self._models[name] = ElasticNetModel()
            self._models[name].load(target_dir)


class LightGBMMultiTarget:
    """Separate LightGBM regressors per target (mirrors RidgeMultiTarget interface)."""

    def __init__(
        self,
        target_names,
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        min_child_samples=20,
        min_split_gain=0.0,
        objective="huber",
        seed=42,
    ):
        self.target_names = target_names
        self._params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            min_child_samples=min_child_samples,
            min_split_gain=min_split_gain,
            objective=objective,
            random_state=seed,
            n_jobs=_LGBM_N_JOBS,
            verbosity=-1,
        )
        self._models = {name: lgb.LGBMRegressor(**self._params) for name in target_names}
        self._feature_names = None

    def fit(self, X_train, y_train_dict, X_val=None, y_val_dict=None, feature_names=None):
        self._feature_names = feature_names
        # Wrap inputs in a named DataFrame so fit sees the same feature names
        # predict() will pass later — otherwise sklearn warns about feature-name
        # mismatch between training and inference.
        if feature_names is not None:
            X_train = pd.DataFrame(X_train, columns=feature_names)
            if X_val is not None:
                X_val = pd.DataFrame(X_val, columns=feature_names)
        for name, model in self._models.items():
            callbacks = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)]
            if X_val is not None and y_val_dict is not None:
                model.fit(
                    X_train,
                    y_train_dict[name],
                    eval_set=[(X_val, y_val_dict[name])],
                    callbacks=callbacks,
                )
                print(f"  {name}: best_iteration={model.best_iteration_}")
            else:
                model.fit(X_train, y_train_dict[name])

    def predict(self, X):
        # Always wrap X in a DataFrame with whatever names fit saw so sklearn
        # doesn't warn. lightgbm auto-assigns "Column_i" names during a numpy
        # fit, so pull those when the user didn't supply feature_names.
        if isinstance(X, pd.DataFrame):
            X_in = X
        elif self._feature_names is not None:
            X_in = pd.DataFrame(X, columns=self._feature_names)
        else:
            first = next(iter(self._models.values()))
            X_in = pd.DataFrame(X, columns=getattr(first, "feature_names_in_", None))
        preds = {name: np.maximum(model.predict(X_in), 0) for name, model in self._models.items()}
        return preds

    def get_feature_importance(self, feature_names):
        result = {}
        for name, model in self._models.items():
            importance = model.feature_importances_
            s = pd.Series(importance, index=feature_names)
            result[name] = s.sort_values(ascending=False)
        return result

    def save(self, model_dir):
        lgb_dir = f"{model_dir}/lightgbm"
        os.makedirs(lgb_dir, exist_ok=True)
        for name, model in self._models.items():
            joblib.dump(model, f"{lgb_dir}/{name}.pkl")
        meta = {"target_names": self.target_names, "params": self._params}
        if self._feature_names is not None:
            meta["feature_names"] = list(self._feature_names)
        with open(f"{lgb_dir}/meta.json", "w") as f:
            json.dump(meta, f)

    def load(self, model_dir):
        lgb_dir = f"{model_dir}/lightgbm"
        with open(f"{lgb_dir}/meta.json") as f:
            meta = json.load(f)
        self.target_names = meta["target_names"]
        self._feature_names = meta.get("feature_names")
        self._models = {}
        for name in self.target_names:
            self._models[name] = joblib.load(f"{lgb_dir}/{name}.pkl")
