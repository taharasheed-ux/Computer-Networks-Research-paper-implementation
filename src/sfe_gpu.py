# src/sfe_gpu.py
"""GPU-accelerated stochastic feature elimination (S_FE).

We reuse the logic from the notebook but plug in cuML LogisticRegression if
available, otherwise fallback to scikit-learn so the script remains runnable
on CPU.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

try:
    from cuml.linear_model import LogisticRegression as cuLR  # type: ignore
    GPU_AVAILABLE = True
except ImportError:
    from sklearn.linear_model import LogisticRegression  # type: ignore
    GPU_AVAILABLE = False

RND = np.random.default_rng(42)


def _evaluate_subset_f1(X_subset: np.ndarray, y: np.ndarray, val_split=0.2) -> float:
    if X_subset.shape[1] == 0:
        return -1.0
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_subset, y, test_size=val_split, stratify=y, random_state=42
    )
    if GPU_AVAILABLE:
        model = cuLR(max_iter=1000)
    else:
        model = LogisticRegression(max_iter=1000, solver="liblinear")
    try:
        model.fit(X_tr, y_tr)
    except Exception:
        return -1.0
    y_pred = model.predict(X_val)
    return float(f1_score(y_val, y_pred, zero_division=0))


def s_fe_stochastic_search(
    X: np.ndarray,
    y: np.ndarray,
    feat_names: List[str],
    *,
    top_k: int = 20,
    n_restarts: int = 1,
    n_iter: int = 30,
    val_split: float = 0.2,
) -> Tuple[List[str], float]:
    """Return (selected_features, best_score)."""
    # Remove constant features then pick top-k by correlation
    non_const = [i for i in range(X.shape[1]) if np.std(X[:, i]) > 0]
    corrs = [abs(np.corrcoef(X[:, i], y)[0, 1]) for i in non_const]
    top_rel = np.argsort(corrs)[-top_k:]
    top_idx = np.array([non_const[i] for i in top_rel])
    X_top = X[:, top_idx]

    best_global = {"score": -1.0, "mask": None}

    for r in range(n_restarts):
        mask = RND.choice([False, True], size=top_k, p=[0.6, 0.4])
        if not mask.any():
            mask[RND.integers(0, top_k)] = True
        cur_score = _evaluate_subset_f1(X_top[:, mask], y, val_split)
        best_local = {"score": cur_score, "mask": mask.copy()}

        for it in range(n_iter):
            i = RND.integers(0, top_k)
            new_mask = mask.copy()
            new_mask[i] = ~new_mask[i]
            if not new_mask.any():
                continue
            new_score = _evaluate_subset_f1(X_top[:, new_mask], y, val_split)
            if new_score >= cur_score:
                mask = new_mask
                cur_score = new_score
                if cur_score > best_local["score"]:
                    best_local = {"score": cur_score, "mask": mask.copy()}
        if best_local["score"] > best_global["score"]:
            best_global = best_local

    sel_idx = top_idx[np.where(best_global["mask"])[0]]
    sel_features = [feat_names[i] for i in sel_idx]
    return sel_features, float(best_global["score"])
