# src/models.py
"""Utility wrappers that pick GPU (cuML) implementations when available
otherwise fall back to scikit-learn.
"""
from __future__ import annotations

from typing import Dict, Any

import numpy as np

GPU_AVAILABLE = False
try:
    from cuml.linear_model import LogisticRegression as cuLR  # type: ignore
    from cuml.decomposition import PCA as cuPCA  # type: ignore
    from cuml.discriminant_analysis import LinearDiscriminantAnalysis as cuLDA  # type: ignore

    GPU_AVAILABLE = True
except ImportError:
    pass

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402


__all__ = [
    "fit_predict_lr",
    "fit_predict_lda",
    "fit_predict_pca_lr",
    "fit_predict_rf",
]


DEF_PARAMS: Dict[str, Any] = dict(max_iter=2000, random_state=42)


def fit_predict_lr(X_train, X_test, y_train):
    if GPU_AVAILABLE:
        model = cuLR(**DEF_PARAMS)
    else:
        model = LogisticRegression(**DEF_PARAMS)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def fit_predict_lda(X_train, X_test, y_train):
    if GPU_AVAILABLE:
        model = cuLDA()
    else:
        model = LinearDiscriminantAnalysis()
    model.fit(X_train, y_train)
    return model.predict(X_test)


def fit_predict_pca_lr(X_train, X_test, y_train, n_components=10):
    if GPU_AVAILABLE:
        pca = cuPCA(n_components=n_components, random_state=42)
        Xp_train = pca.fit_transform(X_train)
        Xp_test = pca.transform(X_test)
        lr = cuLR(**DEF_PARAMS)
    else:
        pca = PCA(n_components=n_components, random_state=42)
        Xp_train = pca.fit_transform(X_train)
        Xp_test = pca.transform(X_test)
        lr = LogisticRegression(**DEF_PARAMS)
    lr.fit(Xp_train, y_train)
    return lr.predict(Xp_test)


# ---------------------------------------------------------------------------
# Random Forest (paper's main classifier)
# ---------------------------------------------------------------------------

def fit_predict_rf(X_train, X_test, y_train):
    """Fit a random forest and return predictions.

    If RAPIDS cuML is available, use its GPU implementation; otherwise fall back
    to scikit-learn's RandomForestClassifier with 100 trees.
    """
    if GPU_AVAILABLE:
        try:
            from cuml.ensemble import RandomForestClassifier as cuRF  # type: ignore

            model = cuRF(n_estimators=100, random_state=42)
        except ImportError:  # pragma: no cover
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    model.fit(X_train, y_train)
    return model.predict(X_test)


# ---------------------------------------------------------------------------
# Deep Learning & Ensemble (Improvements)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Deep Learning & Ensemble (Improvements)
# ---------------------------------------------------------------------------

from sklearn.ensemble import VotingClassifier
try:
    from mlp_torch import fit_predict_torch, TorchSklearnWrapper
except ImportError:
    from src.mlp_torch import fit_predict_torch, TorchSklearnWrapper

def fit_predict_mlp(X_train, X_test, y_train):
    """Fit a Multi-Layer Perceptron (Deep Learning) classifier using PyTorch (GPU)."""
    return fit_predict_torch(X_train, X_test, y_train, epochs=10)


def fit_predict_voting(X_train, X_test, y_train):
    """Fit a Voting Classifier (Ensemble) of RF + MLP (GPU)."""
    # We combine the two best performers: Random Forest and MLP
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    # Use the sklearn-compatible wrapper for the PyTorch model
    mlp = TorchSklearnWrapper(epochs=10)
    
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('mlp', mlp)],
        voting='soft'
    )
    ensemble.fit(X_train, y_train)
    return ensemble.predict(X_test)
