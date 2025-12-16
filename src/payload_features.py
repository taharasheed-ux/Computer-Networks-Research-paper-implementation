# src/payload_features.py
"""Compute the five-element S_FE payload feature vector used in
`Efficient Feature Engineering-Based Anomaly Detection for Network Security`.

The paper defines (offline mode):
    1. Trust Value – running benign ratio for source-IP (computed outside this
       module; supplied when calling :func:`make_feature_row`).
    2. Byte Frequency Analysis (BFA) entropy – Shannon entropy of the 256-bin
       byte frequency vector.
    3. Byte Entropy – Shannon entropy of the raw payload bytes.
    4. Payload Length – size in bytes.
    5. Stream Index – ordinal position of the packet within a flow / stream.

Only pure-Python + NumPy are required, so the code works on any machine.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import List, Tuple

import numpy as np

__all__ = [
    "byte_entropy",
    "byte_freq_entropy",
    "payload_len",
    "hash_value",
    "make_feature_row",
]


# ---------------------------------------------------------------------------
# Entropy helpers
# ---------------------------------------------------------------------------

def _shannon_entropy(probs: np.ndarray) -> float:
    """Return Shannon entropy for an array of probabilities."""
    if probs.size == 0:
        return 0.0
    nz = probs > 0
    if not np.any(nz):
        return 0.0
    return float(-np.sum(probs[nz] * np.log2(probs[nz])))


def byte_entropy(payload: bytes) -> float:
    """Compute Shannon entropy of *payload* at byte-level (0-255 distribution)."""
    if not payload:
        return 0.0
    counts = np.bincount(np.frombuffer(payload, dtype=np.uint8), minlength=256)
    probs = counts / counts.sum()
    return _shannon_entropy(probs)


def byte_freq_entropy(payload: bytes) -> float:
    """BFA: Standard deviation of the byte frequency counts.

    The paper lists 'Byte Frequency Analysis' (BFA) and 'Byte Entropy' (BE)
    as distinct features. Since BE is the Shannon entropy, BFA is likely
    a statistical measure of the distribution shape, such as standard deviation.
    """
    if not payload:
        return 0.0
    counts = np.bincount(np.frombuffer(payload, dtype=np.uint8), minlength=256)
    # Normalize to probabilities to be independent of payload length
    probs = counts / counts.sum()
    return float(np.std(probs))


def payload_len(payload: bytes) -> int:
    """Return length of *payload* in bytes."""
    return len(payload)


# ---------------------------------------------------------------------------
# Optional hash feature (real-time 3-feature variant)
# ---------------------------------------------------------------------------

try:
    import mmh3  # type: ignore

    def hash_value(payload: bytes, n: int = 32) -> int:
        """Return unsigned 32-bit MurmurHash3 of the first *n* bytes."""
        if not payload:
            return 0
        return mmh3.hash_bytes(payload[:n], signed=False)

except ImportError:  # graceful fallback – not critical for offline 5-feature set

    def hash_value(payload: bytes, n: int = 32) -> int:  # type: ignore[override]
        """Fallback 32-bit FNV-1a hash if *mmh3* is not installed."""
        if not payload:
            return 0
        h = 2166136261
        for b in payload[:n]:
            h ^= b
            h *= 16777619
            h &= 0xFFFFFFFF
        return h


# ---------------------------------------------------------------------------
# Public helper: build one feature row
# ---------------------------------------------------------------------------

def make_feature_row(
    payload: bytes,
    *,
    trust_value: float,
    stream_idx: int,
    direction: int = 0,
    use_hash: bool = True,
) -> List[float]:
    """Return feature list in paper order.

    Parameters
    ----------
    payload : bytes
        Application payload of the packet / flow.
    trust_value : float
        Trust value computed for *src-IP* before including this flow.
    stream_idx : int
        Ordinal index of the packet inside its stream (starting at 0).
    direction : int
        0 for Outgoing, 1 for Incoming (or vice versa).
    use_hash : bool
        Whether to include the Hash Value feature.
    """
    row = [
        float(trust_value),
        float(byte_freq_entropy(payload)),
        float(byte_entropy(payload)),
        float(payload_len(payload)),
        float(stream_idx),
        float(direction),
    ]
    if use_hash:
        row.append(float(hash_value(payload)))
    return row

