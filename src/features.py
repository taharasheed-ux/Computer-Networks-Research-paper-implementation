# src/features.py
"""GPU-friendly helpers for loading the CIC-IDS flow CSV and preparing
X (features) and y (binary labels).

If a CUDA GPU and RAPIDS/cuDF are available we keep everything on the GPU;
otherwise we silently fall back to pandas / NumPy so the script can still run
on CPU-only machines (helpful for unit tests).
"""
from __future__ import annotations

import os
from typing import Tuple, List

import numpy as np

# Try to import RAPIDS stack; if not found we will fall back.
try:
    import cudf  # type: ignore
    RAPIDS_AVAILABLE = True
except ImportError:
    import pandas as pd  # type: ignore  # noqa: F401
    RAPIDS_AVAILABLE = False

__all__ = [
    "load_flow_csv",
    "select_numeric_features",
]


def load_flow_csv(path: str):
    """Load the CIC flow CSV into a DataFrame (cuDF if possible)."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if RAPIDS_AVAILABLE:
        return cudf.read_csv(path, low_memory=False)
    else:
        import pandas as pd
        return pd.read_csv(path, low_memory=False)


def _strip_cols(df):
    """Trim whitespace around column names – CIC headers have leading spaces."""
    df.columns = [c.strip() for c in df.columns]
    return df


_DROP_NON_FEATURE = {
    "Flow ID",
    "Src IP",
    "Dst IP",
    "Src Port",
    "Dst Port",
    "Protocol",
    "Timestamp",
    "Label",  # keep original but drop from X
    "binary_label",  # created later
}


def select_numeric_features(df) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return (X, y, feature_names) as NumPy arrays ready for scikit/cuML.

    Steps:
    1. Ensure a binary `binary_label` column (BENIGN→0, else→1).
    2. Drop non-numeric or identifier columns.
    3. Fill NaNs with 0 (fast & usually safe for flow stats).
    """
    df = _strip_cols(df)

    # detect label column
    lbl_candidates = [c for c in df.columns if c.lower() == "label"]
    if not lbl_candidates:
        raise ValueError("CSV does not contain a 'Label' column.")
    label_col = lbl_candidates[0]

    df[label_col] = df[label_col].astype(str)
    df["binary_label"] = df[label_col].str.upper().apply(lambda x: 0 if x == "BENIGN" else 1)

    # numeric feature columns only (and not in drop list)
    if RAPIDS_AVAILABLE:
        num_mask = df.dtypes.replace({"float64": True, "int64": True}).fillna(False)
        numeric_cols = [c for c in df.columns if num_mask[c] and c not in _DROP_NON_FEATURE]
        X_df = df[numeric_cols].fillna(0)
        # Replace ±inf (can appear in Bytes/s divisions) with 0
        X_df = X_df.replace([np.inf, -np.inf], 0)
        X = X_df.to_numpy(copy=False)
        y = df["binary_label"].to_numpy()
    else:
        import pandas as pd
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in _DROP_NON_FEATURE]
        X_df = (
            df[numeric_cols]
            .replace([np.inf, -np.inf], 0)
            .fillna(0)
            .astype(float)
        )
        X = X_df.values
        y = df["binary_label"].values

    return X, y, numeric_cols
